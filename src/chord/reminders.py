"""Reminder persistence and scheduling primitives (SQLite-backed).

Reminders are rows in ``chord.db`` (same database the sqlite MCP server
exposes, so power users can query them there too):

    CREATE TABLE reminders (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        channel_id INTEGER NOT NULL,
        due        TEXT    NOT NULL,   -- ISO-8601 with offset (UTC stored)
        text       TEXT    NOT NULL,
        done       INTEGER NOT NULL DEFAULT 0
    )

Times are normalized to timezone-aware UTC datetimes at the boundary;
``due()`` returns everything scheduled before ``now`` that has not been
delivered yet.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class Reminder:
    """One scheduled message."""

    id: int
    channel_id: int
    due: datetime
    text: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Reminder:
        return cls(
            id=int(row["id"]),
            channel_id=int(row["channel_id"]),
            due=datetime.fromisoformat(row["due"]),
            text=row["text"],
        )


def utc_now() -> datetime:
    return datetime.now(UTC)


def to_utc(moment: datetime) -> datetime:
    """Normalize any aware datetime to UTC (naive input is rejected)."""
    if moment.tzinfo is None:
        raise ValueError("naive datetimes are not allowed; attach a timezone")
    return moment.astimezone(UTC)


class ReminderStore:
    """CRUD for reminders in one SQLite file."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reminders (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel_id INTEGER NOT NULL,
                    due        TEXT    NOT NULL,
                    text       TEXT    NOT NULL,
                    done       INTEGER NOT NULL DEFAULT 0
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    def add(self, channel_id: int, due: datetime, text: str) -> int:
        """Insert one reminder and return its id."""
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO reminders (channel_id, due, text) VALUES (?, ?, ?)",
                (channel_id, to_utc(due).isoformat(), text),
            )
            return int(cursor.lastrowid)

    def due(self, now: datetime | None = None) -> list[Reminder]:
        """Everything not yet delivered whose time has come."""
        now = to_utc(now) if now else utc_now()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM reminders WHERE done = 0 AND due <= ? ORDER BY due",
                (now.isoformat(),),
            ).fetchall()
        return [Reminder.from_row(row) for row in rows]

    def pending_for_channel(self, channel_id: int) -> list[Reminder]:
        """Not-yet-delivered reminders of one channel, soonest first."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM reminders WHERE done = 0 AND channel_id = ? ORDER BY due",
                (channel_id,),
            ).fetchall()
        return [Reminder.from_row(row) for row in rows]

    def mark_done(self, reminder_id: int) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE reminders SET done = 1 WHERE id = ?", (reminder_id,))
