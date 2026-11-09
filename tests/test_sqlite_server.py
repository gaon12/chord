"""Read-only database access must not escape to other files or perform writes."""

import json
import sqlite3

import pytest

from chord.sqlite_server import query_database


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "notes.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE notes (text TEXT)")
        connection.execute("INSERT INTO notes VALUES ('hello')")
    return path


def test_select_and_cte_can_read_notes(database):
    result = json.loads(query_database(database, "WITH n AS (SELECT * FROM notes) SELECT * FROM n"))
    assert result == {"rows": [{"text": "hello"}], "truncated": False}


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM notes",
        "DROP TABLE notes",
        "PRAGMA database_list",
        "ATTACH DATABASE ':memory:' AS other",
        "SELECT load_extension('bad')",
        "SELECT 1; DELETE FROM notes",
    ],
)
def test_dangerous_statements_are_refused(database, sql):
    assert "error" in json.loads(query_database(database, sql))
    assert json.loads(query_database(database, "SELECT COUNT(*) AS n FROM notes"))["rows"] == [
        {"n": 1}
    ]


def test_opening_a_missing_database_does_not_create_it(tmp_path):
    path = tmp_path / "missing.db"
    assert "error" in json.loads(query_database(path, "SELECT 1"))
    assert not path.exists()


def test_large_results_and_expensive_queries_are_bounded(database):
    assert "error" in json.loads(query_database(database, "SELECT hex(zeroblob(20000))"))
    sql = "WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n) SELECT sum(x) FROM n"
    assert "error" in json.loads(query_database(database, sql))
