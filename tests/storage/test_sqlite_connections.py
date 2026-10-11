"""Connections must release their database files before a store operation returns."""

import ast
import os
from pathlib import Path
import sqlite3

import pytest

from career_agent.storage.context import CareerContextStore
from career_agent.storage.sqlite import open_connection


def test_open_connection_commits_and_closes(tmp_path: Path) -> None:
    path = tmp_path / "records.sqlite3"
    with open_connection(path) as connection:
        connection.execute("CREATE TABLE records (value TEXT)")
        connection.execute("INSERT INTO records VALUES ('saved')")
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")
    with open_connection(path) as reader:
        assert reader.execute("SELECT value FROM records").fetchall() == [("saved",)]


def test_open_connection_rolls_back_and_closes_on_error(tmp_path: Path) -> None:
    path = tmp_path / "records.sqlite3"
    with open_connection(path) as connection:
        connection.execute("CREATE TABLE records (value TEXT)")
    with pytest.raises(RuntimeError):
        with open_connection(path) as connection:
            connection.execute("INSERT INTO records VALUES ('discarded')")
            raise RuntimeError("abort")
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")
    with open_connection(path) as reader:
        assert reader.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 0


def test_store_releases_file_after_success_and_error(tmp_path: Path) -> None:
    # Windows rejects removal while SQLite still holds a file handle.
    path = tmp_path / "context.sqlite3"
    store = CareerContextStore(path)
    with store._connect() as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS close_regression (value TEXT)")
        connection.execute("INSERT INTO close_regression VALUES ('saved')")
    with pytest.raises(RuntimeError):
        with store._connect() as connection:
            connection.execute("INSERT INTO close_regression VALUES ('discarded')")
            raise RuntimeError("abort")
    with store._connect() as connection:
        assert connection.execute("SELECT value FROM close_regression").fetchall() == [("saved",)]
    os.remove(path)


def test_only_connection_owners_call_sqlite_connect() -> None:
    root = Path(__file__).resolve().parents[2] / "src"
    allowed = {"storage/sqlite.py", "storage/checkpoints.py"}
    offenders = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "sqlite3"
                and node.func.attr == "connect"
                and path.relative_to(root / "career_agent").as_posix() not in allowed
            ):
                offenders.append(f"{path}:{node.lineno}")
    assert offenders == []
