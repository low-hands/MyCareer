"""Single-use capture intents and durable continuation deliveries."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sqlite3
from typing import Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from career_agent.storage.schema import apply_schema


DEFAULT_INTENT_TTL = timedelta(hours=2)
"""How long a search the agent opened keeps pointing at its conversation.

Browsing a search page and reading a few JDs takes minutes, not seconds, so
this is generous; but a save made hours later belongs to whatever the user is
doing then, not to a conversation they have long since left.
"""

MAX_PENDING_EVENTS = 50

CONTINUATION_TTL = timedelta(hours=24)
"""How long a saved job may wait to be continued into its conversation.

A conversation that stays in a mock interview or an open confirmation keeps
the event pending. After a day the save is no longer "just now", so the event
expires and the page says so instead of continuing it later out of context.
"""

ContinuationStatus = Literal["pending", "completed", "discarded", "failed", "expired"]


class JobCaptureIntent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    user_id: str
    conversation_id: str
    platform: str
    keyword: str
    city: str | None
    created_at: datetime
    expires_at: datetime
    source_turn_id: str | None = None
    consumed_at: datetime | None = None
    consumed_event_id: str | None = None


class JobCapturedEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    user_id: str
    conversation_id: str
    intent_id: str
    job_posting_id: str
    jd_snapshot_id: str
    title: str
    company_name: str
    created_at: datetime
    acknowledged_at: datetime | None = None
    continuation_status: ContinuationStatus = "pending"
    continuation_turn_id: str | None = None


class JobCaptureRecording(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event: JobCapturedEvent
    created: bool
    """False when the exact snapshot was already recorded against this intent."""


class JobCaptureStore(Protocol):
    def create_intent(
        self,
        *,
        user_id: str,
        conversation_id: str,
        platform: str,
        keyword: str,
        city: str | None,
        source_turn_id: str | None = None,
        ttl: timedelta = DEFAULT_INTENT_TTL,
    ) -> JobCaptureIntent: ...

    def get_live_intent(
        self, *, user_id: str, intent_id: str, now: datetime | None = None
    ) -> JobCaptureIntent | None: ...

    def get_intent(
        self, *, user_id: str, intent_id: str
    ) -> JobCaptureIntent | None: ...

    def record_capture(
        self,
        *,
        intent: JobCaptureIntent,
        job_posting_id: str,
        jd_snapshot_id: str,
        title: str,
        company_name: str,
    ) -> JobCaptureRecording | None: ...

    def list_pending_continuations(
        self, *, limit: int = MAX_PENDING_EVENTS
    ) -> tuple[JobCapturedEvent, ...]: ...

    def expire_continuations(
        self, *, now: datetime | None = None
    ) -> tuple[JobCapturedEvent, ...]: ...

    def settle_continuation(
        self, *, user_id: str, event_id: str,
        status: Literal["completed", "discarded", "failed", "expired"],
        turn_id: str | None = None,
    ) -> None: ...

    def retry_continuation(
        self, *, user_id: str, event_id: str, now: datetime | None = None
    ) -> bool: ...

    def list_pending_events(
        self,
        *,
        user_id: str,
        conversation_id: str | None = None,
        limit: int = MAX_PENDING_EVENTS,
    ) -> tuple[JobCapturedEvent, ...]: ...

    def acknowledge_event(self, *, user_id: str, event_id: str) -> bool: ...

    def get_event(
        self, *, user_id: str, event_id: str
    ) -> JobCapturedEvent | None: ...


class SQLiteJobCaptureStore:
    """Shares ``jobs.sqlite3`` with the posting repository, under its own version."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("BEGIN IMMEDIATE")
            apply_schema(
                connection, "job_captures", 2, self._migrate,
                {2: self._upgrade_continuations},
                finalize=self._finalize,
            )
        os.chmod(self.path, 0o600)

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS job_capture_intents (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                keyword TEXT NOT NULL,
                city TEXT,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                source_turn_id TEXT,
                consumed_at TEXT,
                consumed_event_id TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS job_captured_events (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                intent_id TEXT NOT NULL,
                job_posting_id TEXT NOT NULL,
                jd_snapshot_id TEXT NOT NULL,
                title TEXT NOT NULL,
                company_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                acknowledged_at TEXT,
                continuation_status TEXT NOT NULL DEFAULT 'pending',
                continuation_turn_id TEXT,
                UNIQUE(intent_id, jd_snapshot_id),
                FOREIGN KEY(intent_id) REFERENCES job_capture_intents(id)
            )
            """
        )

    @staticmethod
    def _finalize(connection: sqlite3.Connection) -> None:
        connection.execute(
            "CREATE INDEX IF NOT EXISTS job_capture_intents_user_idx "
            "ON job_capture_intents(user_id, expires_at)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS job_captured_events_pending_idx "
            "ON job_captured_events(user_id, acknowledged_at, created_at)"
        )

    @staticmethod
    def _upgrade_continuations(connection: sqlite3.Connection) -> None:
        """v1 → v2: backend-owned continuation state, once, from a recorded v1.

        Intents gain columns in place. Events are rebuilt because v2 also moves
        the uniqueness key from ``(intent_id, job_posting_id)`` to
        ``(intent_id, jd_snapshot_id)``, and SQLite cannot alter a constraint.
        The copy names every column: nothing depends on the v1 column order.
        An acknowledged v1 event was continued by the page and counts as
        completed; an unacknowledged one stays pending for the dispatcher,
        unless an earlier event already consumed the same intent.
        """
        for name in ("source_turn_id", "consumed_at", "consumed_event_id"):
            connection.execute(f"ALTER TABLE job_capture_intents ADD COLUMN {name} TEXT")
        connection.execute(
            """
            CREATE TABLE job_captured_events_v2 (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL, intent_id TEXT NOT NULL,
                job_posting_id TEXT NOT NULL, jd_snapshot_id TEXT NOT NULL,
                title TEXT NOT NULL, company_name TEXT NOT NULL,
                created_at TEXT NOT NULL, acknowledged_at TEXT,
                continuation_status TEXT NOT NULL DEFAULT 'pending',
                continuation_turn_id TEXT,
                UNIQUE(intent_id, jd_snapshot_id),
                FOREIGN KEY(intent_id) REFERENCES job_capture_intents(id)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO job_captured_events_v2 (
                id, user_id, conversation_id, intent_id, job_posting_id,
                jd_snapshot_id, title, company_name, created_at, acknowledged_at,
                continuation_status, continuation_turn_id
            )
            SELECT
                id, user_id, conversation_id, intent_id, job_posting_id,
                jd_snapshot_id, title, company_name, created_at, acknowledged_at,
                CASE WHEN acknowledged_at IS NULL THEN 'pending' ELSE 'completed' END,
                NULL
            FROM job_captured_events
            """
        )
        connection.execute("DROP TABLE job_captured_events")
        connection.execute("ALTER TABLE job_captured_events_v2 RENAME TO job_captured_events")
        connection.execute(
            """
            UPDATE job_capture_intents SET
                consumed_event_id = (
                    SELECT id FROM job_captured_events
                    WHERE intent_id = job_capture_intents.id ORDER BY created_at, id LIMIT 1
                ),
                consumed_at = (
                    SELECT created_at FROM job_captured_events
                    WHERE intent_id = job_capture_intents.id ORDER BY created_at, id LIMIT 1
                )
            """
        )
        connection.execute(
            """
            UPDATE job_captured_events SET continuation_status = 'discarded'
            WHERE continuation_status = 'pending' AND id NOT IN (
                SELECT consumed_event_id FROM job_capture_intents
                WHERE consumed_event_id IS NOT NULL
            )
            """
        )

    def create_intent(
        self,
        *,
        user_id: str,
        conversation_id: str,
        platform: str,
        keyword: str,
        city: str | None,
        source_turn_id: str | None = None,
        ttl: timedelta = DEFAULT_INTENT_TTL,
    ) -> JobCaptureIntent:
        if not user_id or not conversation_id or not keyword:
            raise ValueError("user_id, conversation_id, and keyword are required")
        if ttl <= timedelta(0):
            raise ValueError("ttl must be positive")
        now = datetime.now(timezone.utc)
        intent = JobCaptureIntent(
            id=f"capint_{uuid4().hex}",
            user_id=user_id,
            conversation_id=conversation_id,
            platform=platform,
            keyword=keyword,
            city=city,
            created_at=now,
            expires_at=now + ttl,
            source_turn_id=source_turn_id,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO job_capture_intents(
                    id, user_id, conversation_id, platform, keyword, city,
                    created_at, expires_at, source_turn_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    intent.id,
                    intent.user_id,
                    intent.conversation_id,
                    intent.platform,
                    intent.keyword,
                    intent.city,
                    intent.created_at.isoformat(),
                    intent.expires_at.isoformat(),
                    intent.source_turn_id,
                ),
            )
        return intent

    def get_live_intent(
        self, *, user_id: str, intent_id: str, now: datetime | None = None
    ) -> JobCaptureIntent | None:
        intent = self.get_intent(user_id=user_id, intent_id=intent_id)
        moment = now or datetime.now(timezone.utc)
        if intent is None or intent.expires_at <= moment or intent.consumed_at is not None:
            return None
        return intent

    def get_intent(
        self, *, user_id: str, intent_id: str
    ) -> JobCaptureIntent | None:
        if not user_id or not intent_id:
            return None
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, user_id, conversation_id, platform, keyword, city,
                       created_at, expires_at, source_turn_id, consumed_at, consumed_event_id
                FROM job_capture_intents
                WHERE id = ? AND user_id = ?
                """,
                (intent_id, user_id),
            ).fetchone()
        return self._intent_from_row(row) if row is not None else None

    def record_capture(
        self,
        *,
        intent: JobCaptureIntent,
        job_posting_id: str,
        jd_snapshot_id: str,
        title: str,
        company_name: str,
    ) -> JobCaptureRecording | None:
        """Atomically consume a live intent, or replay its exact saved snapshot."""
        if not job_posting_id or not jd_snapshot_id:
            raise ValueError("job_posting_id and jd_snapshot_id are required")
        now = datetime.now(timezone.utc)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT id, user_id, conversation_id, platform, keyword, city,
                       created_at, expires_at, source_turn_id, consumed_at, consumed_event_id
                FROM job_capture_intents WHERE id = ? AND user_id = ? AND conversation_id = ?
                """,
                (intent.id, intent.user_id, intent.conversation_id),
            ).fetchone()
            if row is None:
                return None
            intent = self._intent_from_row(row)
            row = connection.execute(
                f"SELECT {self._EVENT_COLUMNS} FROM job_captured_events "
                "WHERE intent_id = ? AND jd_snapshot_id = ? AND job_posting_id = ?",
                (intent.id, jd_snapshot_id, job_posting_id),
            ).fetchone()
            if row is not None:
                return JobCaptureRecording(event=self._event_from_row(row), created=False)
            if intent.consumed_at is not None or intent.expires_at <= now:
                return None
            event = JobCapturedEvent(
                id=f"jobcap_{uuid4().hex}",
                user_id=intent.user_id,
                conversation_id=intent.conversation_id,
                intent_id=intent.id,
                job_posting_id=job_posting_id,
                jd_snapshot_id=jd_snapshot_id,
                title=title,
                company_name=company_name,
                created_at=now,
            )
            connection.execute(
                """
                INSERT INTO job_captured_events(
                    id, user_id, conversation_id, intent_id, job_posting_id,
                    jd_snapshot_id, title, company_name, created_at,
                    acknowledged_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    event.id,
                    event.user_id,
                    event.conversation_id,
                    event.intent_id,
                    event.job_posting_id,
                    event.jd_snapshot_id,
                    event.title,
                    event.company_name,
                    event.created_at.isoformat(),
                ),
            )
            connection.execute(
                """
                UPDATE job_capture_intents SET consumed_at = ?, consumed_event_id = ?
                WHERE id = ?
                """,
                (now.isoformat(), event.id, intent.id),
            )
        return JobCaptureRecording(event=event, created=True)

    def list_pending_continuations(
        self, *, limit: int = MAX_PENDING_EVENTS
    ) -> tuple[JobCapturedEvent, ...]:
        """Oldest pending continuations first, bounded per dispatcher pass."""
        if not 1 <= limit <= MAX_PENDING_EVENTS:
            raise ValueError(f"limit must be between 1 and {MAX_PENDING_EVENTS}")
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT {self._EVENT_COLUMNS} FROM job_captured_events "
                "WHERE continuation_status = 'pending' ORDER BY created_at, id LIMIT ?",
                (limit,),
            ).fetchall()
        return tuple(self._event_from_row(row) for row in rows)

    def expire_continuations(
        self, *, now: datetime | None = None
    ) -> tuple[JobCapturedEvent, ...]:
        """Settle pending continuations older than ``CONTINUATION_TTL`` as expired."""
        cutoff = ((now or datetime.now(timezone.utc)) - CONTINUATION_TTL).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                f"SELECT {self._EVENT_COLUMNS} FROM job_captured_events "
                "WHERE continuation_status = 'pending' AND created_at < ? "
                "ORDER BY created_at, id",
                (cutoff,),
            ).fetchall()
            connection.execute(
                "UPDATE job_captured_events SET continuation_status = 'expired' "
                "WHERE continuation_status = 'pending' AND created_at < ?",
                (cutoff,),
            )
        return tuple(
            self._event_from_row(row).model_copy(update={"continuation_status": "expired"})
            for row in rows
        )

    def settle_continuation(
        self, *, user_id: str, event_id: str,
        status: Literal["completed", "discarded", "failed", "expired"],
        turn_id: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE job_captured_events SET continuation_status = ?, continuation_turn_id = ?
                WHERE user_id = ? AND id = ? AND continuation_status = 'pending'
                """,
                (status, turn_id, user_id, event_id),
            )

    def retry_continuation(
        self, *, user_id: str, event_id: str, now: datetime | None = None
    ) -> bool:
        """Re-queue this user's failed continuation, unless it is past its TTL.

        Retrying an event that would expire on the next dispatcher pass would
        only turn "failed" into "expired" behind the user's click.
        """
        cutoff = ((now or datetime.now(timezone.utc)) - CONTINUATION_TTL).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE job_captured_events SET continuation_status = 'pending', acknowledged_at = NULL
                WHERE user_id = ? AND id = ? AND continuation_status = 'failed'
                  AND created_at >= ?
                """,
                (user_id, event_id, cutoff),
            )
        return cursor.rowcount == 1

    def get_event(
        self, *, user_id: str, event_id: str
    ) -> JobCapturedEvent | None:
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT {self._EVENT_COLUMNS} FROM job_captured_events "
                "WHERE user_id = ? AND id = ?",
                (user_id, event_id),
            ).fetchone()
        return self._event_from_row(row) if row is not None else None

    def list_pending_events(
        self,
        *,
        user_id: str,
        conversation_id: str | None = None,
        limit: int = MAX_PENDING_EVENTS,
    ) -> tuple[JobCapturedEvent, ...]:
        if not user_id:
            return ()
        if not 1 <= limit <= MAX_PENDING_EVENTS:
            raise ValueError(f"limit must be between 1 and {MAX_PENDING_EVENTS}")
        clauses = ["user_id = ?", "acknowledged_at IS NULL"]
        params: list[object] = [user_id]
        if conversation_id is not None:
            clauses.append("conversation_id = ?")
            params.append(conversation_id)
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT {self._EVENT_COLUMNS} FROM job_captured_events
                WHERE {" AND ".join(clauses)}
                ORDER BY created_at ASC, id ASC
                LIMIT ?
                """,
                params,
            ).fetchall()
        return tuple(self._event_from_row(row) for row in rows)

    def acknowledge_event(self, *, user_id: str, event_id: str) -> bool:
        """Mark the event delivered. False when it is unknown, foreign, or already acknowledged."""
        if not user_id or not event_id:
            return False
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE job_captured_events SET acknowledged_at = ?
                WHERE id = ? AND user_id = ? AND acknowledged_at IS NULL
                """,
                (datetime.now(timezone.utc).isoformat(), event_id, user_id),
            )
            return cursor.rowcount == 1

    _EVENT_COLUMNS = (
        "id, user_id, conversation_id, intent_id, job_posting_id, jd_snapshot_id, "
        "title, company_name, created_at, acknowledged_at, "
        "continuation_status, continuation_turn_id"
    )

    @staticmethod
    def _intent_from_row(row: tuple) -> JobCaptureIntent:
        return JobCaptureIntent(
            id=row[0],
            user_id=row[1],
            conversation_id=row[2],
            platform=row[3],
            keyword=row[4],
            city=row[5],
            created_at=datetime.fromisoformat(row[6]),
            expires_at=datetime.fromisoformat(row[7]),
            source_turn_id=row[8],
            consumed_at=datetime.fromisoformat(row[9]) if row[9] else None,
            consumed_event_id=row[10],
        )

    @staticmethod
    def _event_from_row(row: tuple) -> JobCapturedEvent:
        return JobCapturedEvent(
            id=row[0],
            user_id=row[1],
            conversation_id=row[2],
            intent_id=row[3],
            job_posting_id=row[4],
            jd_snapshot_id=row[5],
            title=row[6],
            company_name=row[7],
            created_at=datetime.fromisoformat(row[8]),
            acknowledged_at=(
                datetime.fromisoformat(row[9]) if row[9] is not None else None
            ),
            continuation_status=row[10],
            continuation_turn_id=row[11],
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection
