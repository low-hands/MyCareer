from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from career_agent.agent.resume_job_match_contracts import ResumeJobMatchResult
from career_agent.storage.schema import apply_schema


class ResumeJobMatchInputs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    job_title: str
    company_name: str
    resume_id: str
    resume_name: str
    resume_version_number: int
    resume_created_at: datetime
    jd_version: int
    jd_captured_at: datetime


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
    inputs: ResumeJobMatchInputs | None = None


class SQLiteResumeJobMatchStore:
    """Persists immutable resume/JD comparison snapshots in the local resume DB."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            apply_schema(
                connection, "resume_job_matches", 2, self._migrate,
                {2: self._add_inputs},
            )
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
        SQLiteResumeJobMatchStore._add_inputs(connection)

    @staticmethod
    def _add_inputs(connection: sqlite3.Connection) -> None:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(resume_job_matches)")}
        if "inputs_json" not in columns:
            connection.execute("ALTER TABLE resume_job_matches ADD COLUMN inputs_json TEXT")

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
                       result_json, created_at, inputs_json
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
        jd_snapshot_id: str | None = None,
    ) -> StoredResumeJobMatch | None:
        """The newest match for a job, optionally restricted to one JD snapshot."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, user_id, resume_version_id, job_posting_id,
                       jd_snapshot_id, matcher_version, evidence_fingerprint,
                       result_json, created_at, inputs_json
                FROM resume_job_matches
                WHERE user_id = ? AND job_posting_id = ?
                  AND (? IS NULL OR jd_snapshot_id = ?)
                ORDER BY created_at DESC, rowid DESC
                LIMIT 1
                """,
                (user_id, job_posting_id, jd_snapshot_id, jd_snapshot_id),
            ).fetchone()
        return self._record(row) if row else None

    def list_for_job(
        self, *, user_id: str, job_posting_id: str, limit: int = 20, offset: int = 0
    ) -> tuple[StoredResumeJobMatch, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, user_id, resume_version_id, job_posting_id,
                       jd_snapshot_id, matcher_version, evidence_fingerprint,
                       result_json, created_at, inputs_json
                FROM resume_job_matches
                WHERE user_id = ? AND job_posting_id = ?
                ORDER BY created_at DESC, rowid DESC
                LIMIT ? OFFSET ?
                """,
                (user_id, job_posting_id, limit, offset),
            ).fetchall()
        return tuple(self._record(row) for row in rows)

    def count_for_job(self, *, user_id: str, job_posting_id: str) -> int:
        with self._connect() as connection:
            return connection.execute(
                "SELECT COUNT(*) FROM resume_job_matches WHERE user_id = ? AND job_posting_id = ?",
                (user_id, job_posting_id),
            ).fetchone()[0]

    def get(self, *, user_id: str, match_id: str) -> StoredResumeJobMatch | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, user_id, resume_version_id, job_posting_id,
                       jd_snapshot_id, matcher_version, evidence_fingerprint,
                       result_json, created_at, inputs_json
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
        inputs: ResumeJobMatchInputs | None = None,
    ) -> StoredResumeJobMatch:
        created_at = datetime.now(timezone.utc)
        match_id = f"resume_job_match_{uuid4().hex}"
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO resume_job_matches(
                    id, user_id, resume_version_id, job_posting_id,
                    jd_snapshot_id, matcher_version, evidence_fingerprint,
                    result_json, created_at, inputs_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    inputs.model_dump_json() if inputs else None,
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
            inputs=ResumeJobMatchInputs.model_validate_json(row[9]) if row[9] else None,
        )
