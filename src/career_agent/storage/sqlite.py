"""Short-lived SQLite connections with explicit transaction and close semantics."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Iterator


@contextmanager
def open_connection(
    path: str | Path,
    *,
    timeout: float = 30.0,
    pragmas: tuple[str, ...] = (),
    uri: bool = False,
) -> Iterator[sqlite3.Connection]:
    """Run pragmas before the transaction; journal_mode=WAL requires this."""
    connection = sqlite3.connect(path, timeout=timeout, uri=uri)
    try:
        for pragma in pragmas:
            connection.execute(pragma)
        with connection:
            yield connection
    finally:
        connection.close()
