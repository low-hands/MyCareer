"""Correlation between an agent-opened BOSS search and the job saved from it.

``open_job_search`` hands the browser a URL and the turn ends. What the user
saves afterwards arrives through the extension as a plain capture with no
notion of which conversation asked for it, so the agent never learns that the
search it started has produced a job. This store closes that gap with two
records:

- a ``capture intent``: created when the agent opens the search, bound to the
  user, the originating conversation and the search target, and expiring on
  its own so an old tab cannot wake a conversation weeks later;
- a ``job_captured`` event: written when a capture arrives carrying a live
  intent, bound to the exact posting and snapshot that were saved, and kept
  until the conversation page acknowledges it.

The intent id never travels inside the BOSS URL: the page passes it to the
extension over the local bridge, and the extension keeps it against the tab
it opened. A capture without an intent is an ordinary library save and leaves
no event behind.

Events are the durable half of the protocol. The page that started the search
may be closed when the save happens; the event waits in this table and the
next page that opens picks it up. Recording is keyed on ``(intent, posting)``
so a second click on the same save button, or the endpoint retrying, cannot
produce a second event — and therefore not a second turn.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sqlite3
from typing import Protocol
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


class JobCaptureRecording(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event: JobCapturedEvent
    created: bool
    """False when the same posting was already recorded against this intent."""


class JobCaptureStore(Protocol):
    def create_intent(
        self,
        *,
        user_id: str,
        conversation_id: str,
        platform: str,
        keyword: str,
        city: str | None,
        ttl: timedelta = DEFAULT_INTENT_TTL,
    ) -> JobCaptureIntent: ...

    def get_live_intent(
        self, *, user_id: str, intent_id: str, now: datetime | None = None
    ) -> JobCaptureIntent | None: ...

    def record_capture(
        self,
        *,
        intent: JobCaptureIntent,
        job_posting_id: str,
        jd_snapshot_id: str,
        title: str,
        company_name: str,
    ) -> JobCaptureRecording: ...

    def list_pending_events(
        self,
        *,
        user_id: str,
        conversation_id: str | None = None,
        limit: int = MAX_PENDING_EVENTS,
    ) -> tuple[JobCapturedEvent, ...]: ...

    def acknowledge_event(self, *, user_id: str, event_id: str) -> bool: ...


class SQLiteJobCaptureStore:
    """Shares ``jobs.sqlite3`` with the posting repository, under its own version."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            apply_schema(connection, "job_captures", 1, self._migrate)
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
                expires_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS job_capture_intents_user_idx "
            "ON job_capture_intents(user_id, expires_at)"
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
                UNIQUE(intent_id, job_posting_id),
                FOREIGN KEY(intent_id) REFERENCES job_capture_intents(id)
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS job_captured_events_pending_idx "
            "ON job_captured_events(user_id, acknowledged_at, created_at)"
        )

    def create_intent(
        self,
        *,
        user_id: str,
        conversation_id: str,
        platform: str,
        keyword: str,
        city: str | None,
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
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO job_capture_intents(
                    id, user_id, conversation_id, platform, keyword, city,
                    created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
                ),
            )
        return intent

    def get_live_intent(
        self, *, user_id: str, intent_id: str, now: datetime | None = None
    ) -> JobCaptureIntent | None:
        """The intent, only if it belongs to ``user_id`` and has not expired.

        Ownership is checked here rather than by the caller so a foreign id
        and an unknown id are indistinguishable: both are just "no intent".
        """
        if not user_id or not intent_id:
            return None
        moment = now or datetime.now(timezone.utc)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, user_id, conversation_id, platform, keyword, city,
                       created_at, expires_at
                FROM job_capture_intents
                WHERE id = ? AND user_id = ?
                """,
                (intent_id, user_id),
            ).fetchone()
        if row is None:
            return None
        intent = self._intent_from_row(row)
        if intent.expires_at <= moment:
            return None
        return intent

    def record_capture(
        self,
        *,
        intent: JobCaptureIntent,
        job_posting_id: str,
        jd_snapshot_id: str,
        title: str,
        company_name: str,
    ) -> JobCaptureRecording:
        """Bind the saved posting to the intent's conversation, once.

        The second save of the same posting under the same intent returns the
        first event unchanged, even if it has already been acknowledged: the
        conversation was told, and telling it again is what a duplicate click
        must not do. Saving a *different* posting from the same search is a
        new event — the user picked two jobs, and both deserve analysis.
        """
        if not job_posting_id or not jd_snapshot_id:
            raise ValueError("job_posting_id and jd_snapshot_id are required")
        now = datetime.now(timezone.utc)
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
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO job_captured_events(
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
            if cursor.rowcount == 1:
                return JobCaptureRecording(event=event, created=True)
            row = connection.execute(
                f"""
                SELECT {self._EVENT_COLUMNS} FROM job_captured_events
                WHERE intent_id = ? AND job_posting_id = ?
                """,
                (intent.id, job_posting_id),
            ).fetchone()
        return JobCaptureRecording(event=self._event_from_row(row), created=False)

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
        "title, company_name, created_at, acknowledged_at"
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
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection
