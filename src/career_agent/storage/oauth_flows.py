from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3

from career_agent.storage.schema import apply_schema


@dataclass(frozen=True)
class OAuthFlow:
    state: str
    user_id: str
    code_verifier: str
    expires_at: datetime
    connection_kind: str = "gmail"


class SQLiteOAuthFlowStore:
    """Short-lived PKCE state stored beside local conversation context.

    The verifier is single-use and expires after the caller-selected OAuth
    window (currently ten minutes). A backup of ``context.sqlite3`` taken in
    that window therefore contains it; this is an accepted local-only storage
    tradeoff, not a claim that conversation backups exclude OAuth material.
    """

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        with self._connect() as connection:
            apply_schema(
                connection,
                "oauth_flows",
                2,
                self._migrate,
                {2: self._upgrade_to_v2},
            )
        os.chmod(self.path, 0o600)

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS oauth_flows (
                state TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                code_verifier TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                used_at TEXT,
                connection_kind TEXT NOT NULL DEFAULT 'gmail'
            )
            """
        )
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(oauth_flows)")
        }
        if "connection_kind" not in columns:
            connection.execute(
                "ALTER TABLE oauth_flows ADD COLUMN connection_kind TEXT NOT NULL DEFAULT 'gmail'"
            )

    @staticmethod
    def _upgrade_to_v2(connection: sqlite3.Connection) -> None:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(oauth_flows)")
        }
        if "connection_kind" not in columns:
            connection.execute(
                "ALTER TABLE oauth_flows ADD COLUMN "
                "connection_kind TEXT NOT NULL DEFAULT 'gmail'"
            )

    def create(self, flow: OAuthFlow) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO oauth_flows(state, user_id, code_verifier, expires_at, used_at, connection_kind) "
                "VALUES (?, ?, ?, ?, NULL, ?)",
                (
                    flow.state,
                    flow.user_id,
                    flow.code_verifier,
                    flow.expires_at.isoformat(),
                    flow.connection_kind,
                ),
            )

    def peek(self, state: str, *, now: datetime | None = None) -> OAuthFlow | None:
        """Read a live flow's routing metadata without consuming its verifier."""

        checked_at = now or datetime.now(timezone.utc)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state, user_id, code_verifier, expires_at, connection_kind "
                "FROM oauth_flows WHERE state = ? AND used_at IS NULL",
                (state,),
            ).fetchone()
        if row is None or datetime.fromisoformat(row[3]) <= checked_at:
            return None
        return self._flow(row)

    def consume(self, state: str, *, now: datetime | None = None) -> OAuthFlow | None:
        consumed_at = now or datetime.now(timezone.utc)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT state, user_id, code_verifier, expires_at, connection_kind FROM oauth_flows "
                "WHERE state = ? AND used_at IS NULL",
                (state,),
            ).fetchone()
            if row is None or datetime.fromisoformat(row[3]) <= consumed_at:
                return None
            connection.execute(
                "UPDATE oauth_flows SET used_at = ? WHERE state = ? AND used_at IS NULL",
                (consumed_at.isoformat(), state),
            )
        return self._flow(row)

    @staticmethod
    def _flow(row: tuple) -> OAuthFlow:
        return OAuthFlow(
            state=row[0],
            user_id=row[1],
            code_verifier=row[2],
            expires_at=datetime.fromisoformat(row[3]),
            connection_kind=row[4],
        )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30.0)
