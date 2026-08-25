from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sqlite3
from uuid import uuid4

from career_agent.agent.resume_analysis_contracts import ResumeAnalysisResult
from career_agent.services.resume_analysis import ResumeAnalysisDraft


class SQLiteResumeAnalysisDraftStore:
    """Short-lived storage for unconfirmed structured resume analyses."""

    def __init__(self, path: Path, *, ttl: timedelta = timedelta(days=30)) -> None:
        if ttl <= timedelta(0):
            raise ValueError("Resume analysis draft TTL must be positive")
        self.path = path.expanduser()
        self.ttl = ttl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        with self._connect() as connection:
            self._migrate(connection)
        os.chmod(self.path, 0o600)

    def create(
        self,
        *,
        user_id: str,
        resume_version_id: str,
        result: ResumeAnalysisResult,
        now: datetime | None = None,
    ) -> ResumeAnalysisDraft:
        if not user_id.strip() or not resume_version_id.strip():
            raise ValueError("user_id and resume_version_id are required")
        timestamp = self._utc(now or datetime.now(timezone.utc))
        draft = ResumeAnalysisDraft(
            id=f"resume_analysis_{uuid4().hex}",
            user_id=user_id,
            resume_version_id=resume_version_id,
            result=result,
            created_at=timestamp,
            updated_at=timestamp,
            expires_at=timestamp + self.ttl,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO resume_analysis_drafts(
                    id, user_id, resume_version_id, status, result_json,
                    created_at, updated_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    draft.id,
                    draft.user_id,
                    draft.resume_version_id,
                    draft.status,
                    draft.result.model_dump_json(),
                    draft.created_at.isoformat(),
                    draft.updated_at.isoformat(),
                    draft.expires_at.isoformat(),
                ),
            )
        os.chmod(self.path, 0o600)
        return draft

    def get(
        self,
        *,
        user_id: str,
        analysis_id: str,
        now: datetime | None = None,
    ) -> ResumeAnalysisDraft | None:
        timestamp = self._utc(now or datetime.now(timezone.utc))
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, user_id, resume_version_id, status, result_json,
                       created_at, updated_at, expires_at
                FROM resume_analysis_drafts
                WHERE id = ? AND user_id = ? AND expires_at > ?
                """,
                (analysis_id, user_id, timestamp.isoformat()),
            ).fetchone()
        return self._draft(row) if row is not None else None

    def delete_expired(self, *, now: datetime | None = None) -> int:
        timestamp = self._utc(now or datetime.now(timezone.utc))
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM resume_analysis_drafts WHERE expires_at <= ?",
                (timestamp.isoformat(),),
            )
        return cursor.rowcount

    def mark_confirmed(
        self,
        *,
        user_id: str,
        analysis_id: str,
        now: datetime | None = None,
    ) -> ResumeAnalysisDraft:
        timestamp = self._utc(now or datetime.now(timezone.utc))
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT id, user_id, resume_version_id, status, result_json,
                       created_at, updated_at, expires_at
                FROM resume_analysis_drafts
                WHERE id = ? AND user_id = ? AND expires_at > ?
                """,
                (analysis_id, user_id, timestamp.isoformat()),
            ).fetchone()
            if row is None:
                raise ValueError("Resume analysis draft not found")
            current = self._draft(row)
            if current.status == "confirmed":
                return current
            if current.status != "pending":
                raise ValueError(f"Cannot confirm {current.status} resume analysis")
            connection.execute(
                """
                UPDATE resume_analysis_drafts
                SET status = 'confirmed', updated_at = ?
                WHERE id = ? AND user_id = ? AND status = 'pending'
                """,
                (timestamp.isoformat(), analysis_id, user_id),
            )
        return current.model_copy(
            update={"status": "confirmed", "updated_at": timestamp}
        )

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS resume_analysis_drafts (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                resume_version_id TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('pending', 'confirmed', 'rejected')),
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS resume_analysis_drafts_user_version_created_idx
            ON resume_analysis_drafts(user_id, resume_version_id, created_at DESC)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS resume_analysis_drafts_expiry_idx
            ON resume_analysis_drafts(expires_at)
            """
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @staticmethod
    def _draft(row: tuple[object, ...]) -> ResumeAnalysisDraft:
        return ResumeAnalysisDraft(
            id=row[0],
            user_id=row[1],
            resume_version_id=row[2],
            status=row[3],
            result=ResumeAnalysisResult.model_validate(json.loads(str(row[4]))),
            created_at=row[5],
            updated_at=row[6],
            expires_at=row[7],
        )

    @staticmethod
    def _utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("datetime must be timezone-aware")
        return value.astimezone(timezone.utc)
