from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from career_agent.agent.resume_job_match_contracts import ResumeJobMatchResult
from career_agent.storage.schema import apply_schema


class StoredResumeJobMatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    user_id: str
    resume_version_id: str
    job_posting_id: str
    jd_snapshot_id: str
    matcher_version: str
    evidence_fingerprint: str
    result: ResumeJobMatchResult
    created_at: datetime


class SQLiteResumeJobMatchStore:
    """Persists immutable resume/JD comparison snapshots in the local resume DB."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            apply_schema(connection, "resume_job_matches", 1, self._migrate)
        os.chmod(self.path, 0o600)

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
                CREATE TABLE IF NOT EXISTS resume_job_matches (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    resume_version_id TEXT NOT NULL,
                    job_posting_id TEXT NOT NULL,
                    jd_snapshot_id TEXT NOT NULL,
                    matcher_version TEXT NOT NULL,
                    evidence_fingerprint TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(
                        user_id,
                        resume_version_id,
                        jd_snapshot_id,
                        matcher_version,
                        evidence_fingerprint
                    )
                )
                """
        )
        connection.execute(
            """
                CREATE INDEX IF NOT EXISTS resume_job_matches_user_created_idx
                ON resume_job_matches(user_id, created_at DESC)
                """
        )
        connection.execute(
            """
                CREATE INDEX IF NOT EXISTS resume_job_matches_user_job_idx
                ON resume_job_matches(user_id, job_posting_id, created_at DESC)
                """
        )

    def find(
        self,
        *,
        user_id: str,
        resume_version_id: str,
        jd_snapshot_id: str,
        matcher_version: str,
        evidence_fingerprint: str,
    ) -> StoredResumeJobMatch | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, user_id, resume_version_id, job_posting_id,
                       jd_snapshot_id, matcher_version, evidence_fingerprint,
                       result_json, created_at
                FROM resume_job_matches
                WHERE user_id = ? AND resume_version_id = ?
                  AND jd_snapshot_id = ? AND matcher_version = ?
                  AND evidence_fingerprint = ?
                """,
                (
                    user_id,
                    resume_version_id,
                    jd_snapshot_id,
                    matcher_version,
                    evidence_fingerprint,
                ),
            ).fetchone()
        return self._record(row) if row else None

    def find_latest_for_job(
        self,
        *,
        user_id: str,
        job_posting_id: str,
    ) -> StoredResumeJobMatch | None:
        """The most recent match recorded for a job, whatever resume produced it.

        Comparison reads what already exists rather than matching again, so a
        job the user never matched simply has no row here and is reported as
        unknown instead of silently triggering a worker call.
        """
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, user_id, resume_version_id, job_posting_id,
                       jd_snapshot_id, matcher_version, evidence_fingerprint,
                       result_json, created_at
                FROM resume_job_matches
                WHERE user_id = ? AND job_posting_id = ?
                ORDER BY created_at DESC, rowid DESC
                LIMIT 1
                """,
                (user_id, job_posting_id),
            ).fetchone()
        return self._record(row) if row else None

    def get(self, *, user_id: str, match_id: str) -> StoredResumeJobMatch | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, user_id, resume_version_id, job_posting_id,
                       jd_snapshot_id, matcher_version, evidence_fingerprint,
                       result_json, created_at
                FROM resume_job_matches
                WHERE id = ? AND user_id = ?
                """,
                (match_id, user_id),
            ).fetchone()
        return self._record(row) if row else None

    def save(
        self,
        *,
        user_id: str,
        resume_version_id: str,
        job_posting_id: str,
        jd_snapshot_id: str,
        matcher_version: str,
        evidence_fingerprint: str,
        result: ResumeJobMatchResult,
    ) -> StoredResumeJobMatch:
        created_at = datetime.now(timezone.utc)
        match_id = f"resume_job_match_{uuid4().hex}"
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO resume_job_matches(
                    id, user_id, resume_version_id, job_posting_id,
                    jd_snapshot_id, matcher_version, evidence_fingerprint,
                    result_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    match_id,
                    user_id,
                    resume_version_id,
                    job_posting_id,
                    jd_snapshot_id,
                    matcher_version,
                    evidence_fingerprint,
                    result.model_dump_json(),
                    created_at.isoformat(),
                ),
            )
        stored = self.find(
            user_id=user_id,
            resume_version_id=resume_version_id,
            jd_snapshot_id=jd_snapshot_id,
            matcher_version=matcher_version,
            evidence_fingerprint=evidence_fingerprint,
        )
        if stored is None:
            raise RuntimeError("Failed to persist resume-job match")
        return stored

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @staticmethod
    def _record(row: tuple[object, ...]) -> StoredResumeJobMatch:
        return StoredResumeJobMatch(
            id=row[0],
            user_id=row[1],
            resume_version_id=row[2],
            job_posting_id=row[3],
            jd_snapshot_id=row[4],
            matcher_version=row[5],
            evidence_fingerprint=row[6],
            result=ResumeJobMatchResult.model_validate_json(row[7]),
            created_at=row[8],
        )
