from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from career_agent.domain.interview_preparation import InterviewPreparationResult
from career_agent.storage.schema import apply_schema


class StoredInterviewPreparation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    user_id: str
    interview_round_id: str
    application_id: str
    job_posting_id: str
    jd_snapshot_id: str
    resume_version_id: str
    input_fingerprint: str
    worker_version: str
    result: InterviewPreparationResult
    created_at: datetime


class SQLiteInterviewPreparationStore:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            apply_schema(connection, "interview_preparations", 1, self._migrate)
        os.chmod(self.path, 0o600)

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
                CREATE TABLE IF NOT EXISTS interview_preparations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    interview_round_id TEXT NOT NULL,
                    application_id TEXT NOT NULL,
                    job_posting_id TEXT NOT NULL,
                    jd_snapshot_id TEXT NOT NULL,
                    resume_version_id TEXT NOT NULL,
                    input_fingerprint TEXT NOT NULL,
                    worker_version TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(user_id, interview_round_id, input_fingerprint, worker_version)
                )
                """
        )
        connection.execute(
            """
                CREATE INDEX IF NOT EXISTS interview_preparations_user_round_idx
                ON interview_preparations(user_id, interview_round_id, created_at DESC)
                """
        )

    def find(
        self,
        *,
        user_id: str,
        interview_round_id: str,
        input_fingerprint: str,
        worker_version: str,
    ) -> StoredInterviewPreparation | None:
        with self._connect() as connection:
            row = connection.execute(
                self._SELECT
                + " WHERE user_id = ? AND interview_round_id = ? AND input_fingerprint = ? AND worker_version = ?",
                (user_id, interview_round_id, input_fingerprint, worker_version),
            ).fetchone()
        return self._record(row) if row else None

    def find_latest_for_round(
        self, *, user_id: str, interview_round_id: str
    ) -> StoredInterviewPreparation | None:
        """The most recent preparation written for one interview.

        The entity-keyed way back to a preparation, matching what job research
        and mock interview reports already offer. Without it a preparation was
        reachable only while it was the active one or still named by a message
        in the recent window, so it became permanently unreachable to the agent
        the moment both lapsed — alone among the three report kinds.

        Uses the (user_id, interview_round_id, created_at DESC) index the table
        already declares for exactly this shape of read.
        """
        with self._connect() as connection:
            row = connection.execute(
                self._SELECT
                + " WHERE user_id = ? AND interview_round_id = ?"
                + " ORDER BY created_at DESC LIMIT 1",
                (user_id, interview_round_id),
            ).fetchone()
        return self._record(row) if row else None

    def get(
        self, *, user_id: str, preparation_id: str
    ) -> StoredInterviewPreparation | None:
        with self._connect() as connection:
            row = connection.execute(
                self._SELECT + " WHERE id = ? AND user_id = ?",
                (preparation_id, user_id),
            ).fetchone()
        return self._record(row) if row else None

    def save(
        self,
        *,
        user_id: str,
        interview_round_id: str,
        application_id: str,
        job_posting_id: str,
        jd_snapshot_id: str,
        resume_version_id: str,
        input_fingerprint: str,
        worker_version: str,
        result: InterviewPreparationResult,
    ) -> StoredInterviewPreparation:
        created_at = datetime.now(timezone.utc)
        preparation_id = f"interview_preparation_{uuid4().hex}"
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO interview_preparations(
                    id, user_id, interview_round_id, application_id,
                    job_posting_id, jd_snapshot_id, resume_version_id,
                    input_fingerprint, worker_version, result_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    preparation_id, user_id, interview_round_id, application_id,
                    job_posting_id, jd_snapshot_id, resume_version_id,
                    input_fingerprint, worker_version, result.model_dump_json(),
                    created_at.isoformat(),
                ),
            )
        stored = self.find(
            user_id=user_id, interview_round_id=interview_round_id,
            input_fingerprint=input_fingerprint, worker_version=worker_version,
        )
        if stored is None:
            raise RuntimeError("Failed to persist interview preparation")
        return stored

    _SELECT = (
        "SELECT id, user_id, interview_round_id, application_id, job_posting_id, "
        "jd_snapshot_id, resume_version_id, input_fingerprint, worker_version, "
        "result_json, created_at FROM interview_preparations"
    )

    def clear_user(self, *, user_id: str) -> int:
        """Delete every preparation of ``user_id``; each belongs to an interview round."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            return connection.execute(
                "DELETE FROM interview_preparations WHERE user_id = ?", (user_id,)
            ).rowcount

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30.0)

    @staticmethod
    def _record(row) -> StoredInterviewPreparation:
        return StoredInterviewPreparation(
            id=row[0], user_id=row[1], interview_round_id=row[2],
            application_id=row[3], job_posting_id=row[4], jd_snapshot_id=row[5],
            resume_version_id=row[6], input_fingerprint=row[7],
            worker_version=row[8],
            result=InterviewPreparationResult.model_validate_json(row[9]),
            created_at=row[10],
        )
