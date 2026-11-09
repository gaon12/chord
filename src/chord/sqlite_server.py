"""Optional read-only MCP access to a dedicated, non-sensitive SQLite file.

Run with ``python -m chord.sqlite_server --db-path notes.db``. The file must
already exist. Do not point this server at the bot's reminder database: MCP
tools have no Discord user or channel identity with which to enforce access.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:  # MCP 1.x used the FastMCP name.
    from mcp.server.fastmcp import FastMCP as MCPServer

MAX_ROWS = 200
MAX_SQL_LENGTH = 10_000
MAX_RESULT_BYTES = 32_768


def _authorize(action, _first, second, _database, _source) -> int:
    """Deny writes, ATTACH, PRAGMA, and extension/file-access functions."""
    if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_RECURSIVE):
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION and (second or "").lower() not in {
        "load_extension",
        "readfile",
        "writefile",
    }:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def query_database(path: Path, sql: str) -> str:
    """Execute one bounded read, with SQLite enforcing the access rules."""
    if not sql.strip() or len(sql) > MAX_SQL_LENGTH:
        return json.dumps({"error": "Provide a query of at most 10000 characters."})
    try:
        # URI mode=ro prevents accidental database creation and all writes.
        # The authorizer additionally prevents opening another file via ATTACH.
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            # SQLite applies this limit while building values, before Python
            # receives them. A result-size check alone would be too late for
            # a query such as zeroblob(1000000000).
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, MAX_RESULT_BYTES)
            connection.set_authorizer(_authorize)
            steps = 0

            def stop_expensive_query() -> int:
                nonlocal steps
                steps += 1
                return int(steps > 1000)

            connection.set_progress_handler(stop_expensive_query, 1000)
            cursor = connection.execute(sql)
            rows = cursor.fetchmany(MAX_ROWS + 1)
            payload = json.dumps(
                {"rows": [dict(row) for row in rows[:MAX_ROWS]], "truncated": len(rows) > MAX_ROWS},
                ensure_ascii=False,
                default=lambda value: "[binary data]",
            )
            if len(payload.encode("utf-8")) > MAX_RESULT_BYTES:
                return json.dumps({"error": "Result too large. Select fewer or smaller values."})
            return payload
    except sqlite3.Error:
        # Database paths and SQL error details must not enter a public chat.
        return json.dumps({"error": "Query failed. Only bounded, read-only queries are allowed."})


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only SQLite MCP server")
    parser.add_argument("--db-path", type=Path, required=True)
    args = parser.parse_args()
    server = MCPServer("chord-sqlite-readonly")

    @server.tool()
    def db_tables() -> str:
        """List user tables in the dedicated database."""
        return query_database(
            args.db_path,
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",
        )

    @server.tool()
    def db_query(sql: str) -> str:
        """Run a read-only query; at most 200 rows and 32 KiB are returned."""
        return query_database(args.db_path, sql)

    server.run()


if __name__ == "__main__":
    main()
