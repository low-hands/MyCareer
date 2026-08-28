from __future__ import annotations

import os
from pathlib import Path
import sqlite3

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver


class SQLiteCheckpointOwner:
    """Owns one synchronous SQLite checkpointer and its connection lifecycle."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        self._connection = sqlite3.connect(
            self.path,
            timeout=30.0,
            check_same_thread=False,
        )
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA busy_timeout=30000")
        self.saver = SqliteSaver(
            self._connection,
            serde=JsonPlusSerializer(
                pickle_fallback=False,
                allowed_json_modules=None,
                allowed_msgpack_modules=None,
            ),
        )
        self.saver.setup()
        os.chmod(self.path, 0o600)
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._connection.close()
        self._closed = True

    def __enter__(self) -> SQLiteCheckpointOwner:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
