from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sqlite3
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from career_agent.storage.schema import apply_schema
from career_agent.agent.resume_tailoring_contracts import (
    ResumeReviewTrace,
    ResumeTailoringResult,
)


class ParentTailoringDraftNotRevisableError(ValueError):
    """Raised when lineage creation loses a finalized/superseded parent race."""


class StoredResumeTailoringDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    user_id: str
    match_id: str
    parent_draft_id: str | None = None
    revision_number: int = Field(default=1, ge=1)
    revision_feedback: str | None = None
    tailoring_goal: str | None = None
    status: Literal[
        "pending", "in_review", "reviewed", "finalized", "superseded"
    ] = "pending"
    worker_version: str
    result: ResumeTailoringResult
    automated_review: ResumeReviewTrace | None = None
    change_reviews: tuple[TailoringChangeReview, ...] = ()
    created_at: datetime
    expires_at: datetime

    @property
    def pending_change_indices(self) -> tuple[int, ...]:
        decided = {review.change_index for review in self.change_reviews}
        return tuple(
            index
            for index in range(1, len(self.result.changes) + 1)
            if index not in decided
        )


class TailoringChangeReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    change_index: int
    decision: Literal["accepted", "rejected"]
    feedback: str | None = None
    reviewed_at: datetime


class SQLiteResumeTailoringDraftStore:
    DEFAULT_TTL = timedelta(days=30)

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            apply_schema(connection, "resume_tailoring", 1, self._migrate)
        os.chmod(self.path, 0o600)

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
                CREATE TABLE IF NOT EXISTS resume_tailoring_drafts (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                match_id TEXT NOT NULL,
                parent_draft_id TEXT,
                revision_number INTEGER NOT NULL DEFAULT 1,
                revision_feedback TEXT,
                tailoring_goal TEXT,
                status TEXT NOT NULL CHECK(status IN ('pending')),
                review_status TEXT NOT NULL DEFAULT 'pending',
                worker_version TEXT NOT NULL,
                result_json TEXT NOT NULL,
                automated_review_json TEXT,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
            """
        )
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(resume_tailoring_drafts)"
            ).fetchall()
        }
        if "review_status" not in columns:
            connection.execute(
                """
                ALTER TABLE resume_tailoring_drafts
                ADD COLUMN review_status TEXT NOT NULL DEFAULT 'pending'
                """
            )
        if "automated_review_json" not in columns:
            connection.execute(
                """
                ALTER TABLE resume_tailoring_drafts
                ADD COLUMN automated_review_json TEXT
                """
            )
        if "parent_draft_id" not in columns:
            connection.execute(
                "ALTER TABLE resume_tailoring_drafts ADD COLUMN parent_draft_id TEXT"
            )
        if "revision_number" not in columns:
            connection.execute(
                """
                ALTER TABLE resume_tailoring_drafts
                ADD COLUMN revision_number INTEGER NOT NULL DEFAULT 1
                """
            )
        if "revision_feedback" not in columns:
            connection.execute(
                "ALTER TABLE resume_tailoring_drafts ADD COLUMN revision_feedback TEXT"
            )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS resume_tailoring_change_reviews (
                draft_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                change_index INTEGER NOT NULL CHECK(change_index > 0),
                decision TEXT NOT NULL CHECK(decision IN ('accepted', 'rejected')),
                feedback TEXT,
                reviewed_at TEXT NOT NULL,
                PRIMARY KEY(draft_id, change_index),
                FOREIGN KEY(draft_id) REFERENCES resume_tailoring_drafts(id)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS resume_tailoring_user_created_idx
            ON resume_tailoring_drafts(user_id, created_at DESC)
            """
        )

    def create(
        self,
        *,
        user_id: str,
        match_id: str,
        parent_draft_id: str | None = None,
        revision_number: int = 1,
        revision_feedback: str | None = None,
        tailoring_goal: str | None,
        worker_version: str,
        result: ResumeTailoringResult,
        automated_review: ResumeReviewTrace | None = None,
    ) -> StoredResumeTailoringDraft:
        created_at = datetime.now(timezone.utc)
        draft = StoredResumeTailoringDraft(
            id=f"resume_tailoring_{uuid4().hex}",
            user_id=user_id,
            match_id=match_id,
            parent_draft_id=parent_draft_id,
            revision_number=revision_number,
            revision_feedback=revision_feedback,
            tailoring_goal=tailoring_goal,
            worker_version=worker_version,
            result=result,
            automated_review=automated_review,
            created_at=created_at,
            expires_at=created_at + self.DEFAULT_TTL,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO resume_tailoring_drafts(
                    id, user_id, match_id, parent_draft_id, revision_number,
                    revision_feedback, tailoring_goal, status,
                    review_status, worker_version, result_json, created_at,
                    expires_at, automated_review_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    draft.id,
                    draft.user_id,
                    draft.match_id,
                    draft.parent_draft_id,
                    draft.revision_number,
                    draft.revision_feedback,
                    draft.tailoring_goal,
                    draft.status,
                    draft.status,
                    draft.worker_version,
                    draft.result.model_dump_json(),
                    draft.created_at.isoformat(),
                    draft.expires_at.isoformat(),
                    (
                        draft.automated_review.model_dump_json()
                        if draft.automated_review is not None
                        else None
                    ),
                ),
            )
            if draft.parent_draft_id is not None:
                updated = connection.execute(
                    """
                    UPDATE resume_tailoring_drafts
                    SET review_status = 'superseded'
                    WHERE id = ? AND user_id = ?
                      AND review_status NOT IN ('finalized', 'superseded')
                    """,
                    (draft.parent_draft_id, draft.user_id),
                ).rowcount
                if updated != 1:
                    raise ParentTailoringDraftNotRevisableError(
                        "Parent tailoring draft is no longer eligible for revision"
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
                SELECT id, user_id, match_id, parent_draft_id, revision_number,
                       revision_feedback, tailoring_goal, review_status,
                       worker_version, result_json, automated_review_json,
                       created_at, expires_at
                FROM resume_tailoring_drafts
                WHERE id = ? AND user_id = ? AND expires_at > ?
                """,
                (draft_id, user_id, effective_now.isoformat()),
            ).fetchone()
            review_rows = (
                connection.execute(
                    """
                    SELECT change_index, decision, feedback, reviewed_at
                    FROM resume_tailoring_change_reviews
                    WHERE draft_id = ? AND user_id = ?
                    ORDER BY change_index
                    """,
                    (draft_id, user_id),
                ).fetchall()
                if row
                else ()
            )
        return self._record(row, review_rows) if row else None

    def get_for_display(
        self,
        *,
        user_id: str,
        draft_id: str,
    ) -> StoredResumeTailoringDraft | None:
        """Read a historical draft without reviving its expired capabilities.

        Mutation paths continue to call :meth:`get`, which enforces the TTL.
        This method exists only for the person's read-only report card, where
        an expired or superseded draft must remain honestly inspectable.
        """
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, user_id, match_id, parent_draft_id, revision_number,
                       revision_feedback, tailoring_goal, review_status,
                       worker_version, result_json, automated_review_json,
                       created_at, expires_at
                FROM resume_tailoring_drafts
                WHERE id = ? AND user_id = ?
                """,
                (draft_id, user_id),
            ).fetchone()
            review_rows = (
                connection.execute(
                    """
                    SELECT change_index, decision, feedback, reviewed_at
                    FROM resume_tailoring_change_reviews
                    WHERE draft_id = ? AND user_id = ?
                    ORDER BY change_index
                    """,
                    (draft_id, user_id),
                ).fetchall()
                if row
                else ()
            )
        return self._record(row, review_rows) if row else None

    def list_with_unresolved_gaps(
        self,
        *,
        user_id: str,
        now: datetime | None = None,
        limit: int = 20,
    ) -> tuple[StoredResumeTailoringDraft, ...]:
        """Return live, non-superseded drafts whose result still has gaps."""
        if limit < 1:
            return ()
        effective_now = now or datetime.now(timezone.utc)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, user_id, match_id, parent_draft_id, revision_number,
                       revision_feedback, tailoring_goal, review_status,
                       worker_version, result_json, automated_review_json,
                       created_at, expires_at
                FROM resume_tailoring_drafts
                WHERE user_id = ? AND expires_at > ?
                  AND review_status != 'superseded'
                ORDER BY created_at DESC
                """,
                (user_id, effective_now.isoformat()),
            ).fetchall()
            drafts = []
            for row in rows:
                if not ResumeTailoringResult.model_validate_json(
                    row[9]
                ).unresolved_gaps:
                    continue
                review_rows = connection.execute(
                    """
                    SELECT change_index, decision, feedback, reviewed_at
                    FROM resume_tailoring_change_reviews
                    WHERE draft_id = ? AND user_id = ?
                    ORDER BY change_index
                    """,
                    (row[0], user_id),
                ).fetchall()
                draft = self._record(row, review_rows)
                drafts.append(draft)
                if len(drafts) >= limit:
                    break
        return tuple(drafts)

    def review_changes(
        self,
        *,
        user_id: str,
        draft_id: str,
        accepted_change_indices: tuple[int, ...],
        rejected_change_indices: tuple[int, ...],
        feedback: str | None = None,
    ) -> StoredResumeTailoringDraft | None:
        now = datetime.now(timezone.utc)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT result_json, review_status
                FROM resume_tailoring_drafts
                WHERE id = ? AND user_id = ? AND expires_at > ?
                """,
                (draft_id, user_id, now.isoformat()),
            ).fetchone()
            if row is None:
                return None
            if row[1] in {"finalized", "superseded"}:
                raise ValueError(
                    "Finalized or superseded tailoring decisions cannot be changed"
                )
            result = ResumeTailoringResult.model_validate_json(row[0])
            all_indices = (*accepted_change_indices, *rejected_change_indices)
            if not all_indices:
                raise ValueError("At least one change decision is required")
            if len(set(all_indices)) != len(all_indices):
                raise ValueError("Change decisions must be unique and cannot overlap")
            invalid = [
                index
                for index in all_indices
                if index < 1 or index > len(result.changes)
            ]
            if invalid:
                raise ValueError(f"Unknown tailoring change index: {invalid[0]}")
            for decision, indices in (
                ("accepted", accepted_change_indices),
                ("rejected", rejected_change_indices),
            ):
                for index in indices:
                    connection.execute(
                        """
                        INSERT INTO resume_tailoring_change_reviews(
                            draft_id, user_id, change_index, decision,
                            feedback, reviewed_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(draft_id, change_index) DO UPDATE SET
                            decision=excluded.decision,
                            feedback=excluded.feedback,
                            reviewed_at=excluded.reviewed_at
                        """,
                        (
                            draft_id,
                            user_id,
                            index,
                            decision,
                            feedback,
                            now.isoformat(),
                        ),
                    )
            reviewed_count = connection.execute(
                """
                SELECT COUNT(*) FROM resume_tailoring_change_reviews
                WHERE draft_id = ? AND user_id = ?
                """,
                (draft_id, user_id),
            ).fetchone()[0]
            review_status = (
                "reviewed" if reviewed_count == len(result.changes) else "in_review"
            )
            connection.execute(
                """
                UPDATE resume_tailoring_drafts SET review_status = ?
                WHERE id = ? AND user_id = ?
                """,
                (review_status, draft_id, user_id),
            )
        return self.get(user_id=user_id, draft_id=draft_id)

    def set_automated_review(
        self,
        *,
        user_id: str,
        draft_id: str,
        automated_review: ResumeReviewTrace,
        result: ResumeTailoringResult | None = None,
    ) -> StoredResumeTailoringDraft | None:
        """Persist background QA without changing the user's decisions."""
        with self._connect() as connection:
            if result is None:
                updated = connection.execute(
                    """
                    UPDATE resume_tailoring_drafts
                    SET automated_review_json = ?
                    WHERE id = ? AND user_id = ? AND review_status NOT IN ('finalized', 'superseded')
                    """,
                    (automated_review.model_dump_json(), draft_id, user_id),
                ).rowcount
            else:
                updated = connection.execute(
                    """
                    UPDATE resume_tailoring_drafts
                    SET result_json = ?, automated_review_json = ?
                    WHERE id = ? AND user_id = ? AND review_status NOT IN ('finalized', 'superseded')
                    """,
                    (
                        result.model_dump_json(),
                        automated_review.model_dump_json(),
                        draft_id,
                        user_id,
                    ),
                ).rowcount
        return self.get(user_id=user_id, draft_id=draft_id) if updated else None

    def mark_finalized(
        self, *, user_id: str, draft_id: str
    ) -> StoredResumeTailoringDraft | None:
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE resume_tailoring_drafts SET review_status = 'finalized'
                WHERE id = ? AND user_id = ? AND review_status IN ('reviewed', 'finalized')
                """,
                (draft_id, user_id),
            ).rowcount
        return self.get(user_id=user_id, draft_id=draft_id) if updated else None

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @staticmethod
    def _record(
        row: tuple[object, ...],
        review_rows: tuple[tuple[object, ...], ...] | list[tuple[object, ...]] = (),
    ) -> StoredResumeTailoringDraft:
        return StoredResumeTailoringDraft(
            id=row[0],
            user_id=row[1],
            match_id=row[2],
            parent_draft_id=row[3],
            revision_number=row[4],
            revision_feedback=row[5],
            tailoring_goal=row[6],
            status=row[7],
            worker_version=row[8],
            result=ResumeTailoringResult.model_validate_json(row[9]),
            automated_review=(
                ResumeReviewTrace.model_validate_json(row[10]) if row[10] else None
            ),
            change_reviews=tuple(
                TailoringChangeReview(
                    change_index=review[0],
                    decision=review[1],
                    feedback=review[2],
                    reviewed_at=review[3],
                )
                for review in review_rows
            ),
            created_at=row[11],
            expires_at=row[12],
        )
