from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sqlite3
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from career_agent.agent.resume_tailoring_contracts import ResumeTailoringResult


class StoredResumeTailoringDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    user_id: str
    match_id: str
    tailoring_goal: str | None = None
    status: Literal["pending"] = "pending"
    worker_version: str
    result: ResumeTailoringResult
    created_at: datetime
    expires_at: datetime


class SQLiteResumeTailoringDraftStore:
    DEFAULT_TTL = timedelta(days=30)

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS resume_tailoring_drafts (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    match_id TEXT NOT NULL,
                    tailoring_goal TEXT,
                    status TEXT NOT NULL CHECK(status IN ('pending')),
                    worker_version TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS resume_tailoring_user_created_idx
                ON resume_tailoring_drafts(user_id, created_at DESC)
                """
            )
        os.chmod(self.path, 0o600)

    def create(
        self,
        *,
        user_id: str,
        match_id: str,
        tailoring_goal: str | None,
        worker_version: str,
        result: ResumeTailoringResult,
    ) -> StoredResumeTailoringDraft:
        created_at = datetime.now(timezone.utc)
        draft = StoredResumeTailoringDraft(
            id=f"resume_tailoring_{uuid4().hex}",
            user_id=user_id,
            match_id=match_id,
            tailoring_goal=tailoring_goal,
            worker_version=worker_version,
            result=result,
            created_at=created_at,
            expires_at=created_at + self.DEFAULT_TTL,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO resume_tailoring_drafts(
                    id, user_id, match_id, tailoring_goal, status,
                    worker_version, result_json, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    draft.id,
                    draft.user_id,
                    draft.match_id,
                    draft.tailoring_goal,
                    draft.status,
                    draft.worker_version,
                    draft.result.model_dump_json(),
                    draft.created_at.isoformat(),
                    draft.expires_at.isoformat(),
                ),
            )
        return draft

    def get(
        self,
        *,
        user_id: str,
        draft_id: str,
        now: datetime | None = None,
    ) -> StoredResumeTailoringDraft | None:
        effective_now = now or datetime.now(timezone.utc)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, user_id, match_id, tailoring_goal, status,
                       worker_version, result_json, created_at, expires_at
                FROM resume_tailoring_drafts
                WHERE id = ? AND user_id = ? AND expires_at > ?
                """,
                (draft_id, user_id, effective_now.isoformat()),
            ).fetchone()
        return self._record(row) if row else None

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30.0)

    @staticmethod
    def _record(row: tuple[object, ...]) -> StoredResumeTailoringDraft:
        return StoredResumeTailoringDraft(
            id=row[0],
            user_id=row[1],
            match_id=row[2],
            tailoring_goal=row[3],
            status=row[4],
            worker_version=row[5],
            result=ResumeTailoringResult.model_validate_json(row[6]),
            created_at=row[7],
            expires_at=row[8],
        )
