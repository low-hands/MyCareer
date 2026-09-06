from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from uuid import uuid4

from career_agent.domain.memory_scope import (
    CanonicalScope,
    ScopeResolution,
    ScopeResolutionQueueEvent,
    ScopeResolutionQueueItem,
)
from career_agent.harness.memory_telemetry import content_digest
from career_agent.security.redaction import redact_text
from career_agent.storage.schema import apply_schema


class SQLiteScopeResolutionStore:
    """Durable pre-key queue in the same file as the agent context."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            apply_schema(
                connection,
                "memory_scope",
                2,
                self._baseline,
                {2: self._upgrade_to_v2},
            )
        os.chmod(self.path, 0o600)

    def enqueue(self, resolution: ScopeResolution) -> ScopeResolutionQueueItem:
        if resolution.resolved:
            raise ValueError("resolved scope proposals do not belong in the queue")
        proposal = resolution.proposal
        now = datetime.now(timezone.utc)
        proposed_value = redact_text(proposal.proposed_value)
        content_digest = self.content_digest(proposal.proposed_value)
        idempotency_key = self._idempotency_key(
            user_id=proposal.user_id,
            source_kind=proposal.source_kind,
            source_id=proposal.source_id,
            family=proposal.family,
            subject_id=proposal.subject_id,
            relation=proposal.relation,
            content_digest=content_digest,
        )
        item = ScopeResolutionQueueItem(
            id=f"scope_resolution_{uuid4().hex}",
            user_id=proposal.user_id,
            conversation_id=proposal.conversation_id,
            family=proposal.family,
            subject_id=proposal.subject_id,
            relation=proposal.relation,
            proposed_value=proposed_value,
            source_kind=proposal.source_kind,
            source_id=proposal.source_id,
            status="unresolved",
            reason=resolution.reason or "Canonical scope is unresolved.",
            candidate_scope_keys=resolution.candidate_scope_keys,
            content_digest=content_digest,
            idempotency_key=idempotency_key,
            created_at=now,
            updated_at=now,
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            inserted = connection.execute(
                """
                INSERT OR IGNORE INTO scope_resolution_queue(
                    id, user_id, conversation_id, family, subject_id,
                    relation, proposed_value,
                    source_kind, source_id, status, reason,
                    candidate_scope_keys_json, resolved_scope_key,
                    content_digest, idempotency_key, clarification_attempts,
                    created_at, updated_at, clarification_requested_at, resolved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, 0, ?, ?, NULL, NULL)
                """,
                (
                    item.id,
                    item.user_id,
                    item.conversation_id,
                    item.family,
                    item.subject_id,
                    item.relation,
                    item.proposed_value,
                    item.source_kind,
                    item.source_id,
                    item.status,
                    item.reason,
                    json.dumps(item.candidate_scope_keys, ensure_ascii=False),
                    item.content_digest,
                    item.idempotency_key,
                    item.created_at.isoformat(),
                    item.updated_at.isoformat(),
                ),
            ).rowcount
            if inserted:
                self._insert_event(
                    connection,
                    ScopeResolutionQueueEvent(
                        id=f"scope_resolution_event_{uuid4().hex}",
                        queue_item_id=item.id,
                        user_id=item.user_id,
                        event_type="enqueued",
                        previous_status=None,
                        new_status="unresolved",
                        reason=item.reason,
                        occurred_at=now,
                    ),
                )
                return item
            if proposal.conversation_id is not None:
                connection.execute(
                    """
                    UPDATE scope_resolution_queue
                    SET conversation_id = ?, updated_at = ?
                    WHERE user_id = ? AND idempotency_key = ?
                      AND status IN ('unresolved', 'clarification_requested')
                    """,
                    (
                        proposal.conversation_id,
                        now.isoformat(),
                        item.user_id,
                        item.idempotency_key,
                    ),
                )
            row = connection.execute(
                self._select_sql("WHERE user_id = ? AND idempotency_key = ?"),
                (item.user_id, item.idempotency_key),
            ).fetchone()
        if row is None:
            raise RuntimeError("scope queue idempotency lookup failed")
        return self._item(row)

    def get(
        self, *, user_id: str, queue_item_id: str
    ) -> ScopeResolutionQueueItem | None:
        with self._connect() as connection:
            row = connection.execute(
                self._select_sql("WHERE id = ? AND user_id = ?"),
                (queue_item_id, user_id),
            ).fetchone()
        return self._item(row) if row else None

    def list_open(
        self, *, user_id: str, limit: int = 50
    ) -> tuple[ScopeResolutionQueueItem, ...]:
        if not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200")
        with self._connect() as connection:
            rows = connection.execute(
                self._select_sql(
                    """
                    WHERE user_id = ?
                      AND status IN ('unresolved', 'clarification_requested')
                    ORDER BY created_at, id
                    LIMIT ?
                    """
                ),
                (user_id, limit),
            ).fetchall()
        return tuple(self._item(row) for row in rows)

    def list_all(
        self, *, user_id: str
    ) -> tuple[ScopeResolutionQueueItem, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                self._select_sql("WHERE user_id = ? ORDER BY created_at, id"),
                (user_id,),
            ).fetchall()
        return tuple(self._item(row) for row in rows)

    def request_clarification(
        self, *, user_id: str, queue_item_id: str
    ) -> ScopeResolutionQueueItem:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._get_for_update(connection, user_id, queue_item_id)
            if current.status in {"resolved", "dismissed"}:
                raise ValueError("terminal scope queue items cannot be clarified")
            now = datetime.now(timezone.utc)
            connection.execute(
                """
                UPDATE scope_resolution_queue
                SET status = 'clarification_requested',
                    clarification_attempts = clarification_attempts + 1,
                    clarification_requested_at = ?,
                    updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (now.isoformat(), now.isoformat(), current.id, user_id),
            )
            self._insert_event(
                connection,
                ScopeResolutionQueueEvent(
                    id=f"scope_resolution_event_{uuid4().hex}",
                    queue_item_id=current.id,
                    user_id=user_id,
                    event_type="clarification_requested",
                    previous_status=current.status,
                    new_status="clarification_requested",
                    occurred_at=now,
                ),
            )
            row = connection.execute(
                self._select_sql("WHERE id = ? AND user_id = ?"),
                (current.id, user_id),
            ).fetchone()
        return self._item(row)

    def resolve(
        self,
        *,
        user_id: str,
        queue_item_id: str,
        canonical_scope: CanonicalScope,
        reason: str | None = None,
    ) -> ScopeResolutionQueueItem:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._get_for_update(connection, user_id, queue_item_id)
            if current.status == "resolved":
                if current.resolved_scope_key == canonical_scope.scope_key:
                    return current
                raise ValueError("scope queue item was resolved to a different key")
            if current.status == "dismissed":
                raise ValueError("dismissed scope queue items cannot be resolved")
            if (
                current.family != canonical_scope.family
                or current.subject_id != canonical_scope.subject_id
            ):
                raise ValueError("resolved scope must preserve family and subject")
            now = datetime.now(timezone.utc)
            connection.execute(
                """
                UPDATE scope_resolution_queue
                SET status = 'resolved', resolved_scope_key = ?,
                    resolved_at = ?, updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (
                    canonical_scope.scope_key,
                    now.isoformat(),
                    now.isoformat(),
                    current.id,
                    user_id,
                ),
            )
            self._insert_event(
                connection,
                ScopeResolutionQueueEvent(
                    id=f"scope_resolution_event_{uuid4().hex}",
                    queue_item_id=current.id,
                    user_id=user_id,
                    event_type="resolved",
                    previous_status=current.status,
                    new_status="resolved",
                    reason=reason,
                    occurred_at=now,
                ),
            )
            row = connection.execute(
                self._select_sql("WHERE id = ? AND user_id = ?"),
                (current.id, user_id),
            ).fetchone()
        return self._item(row)

    def dismiss(
        self,
        *,
        user_id: str,
        queue_item_id: str,
        reason: str,
    ) -> ScopeResolutionQueueItem:
        if not reason.strip():
            raise ValueError("dismissal reason is required")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._get_for_update(connection, user_id, queue_item_id)
            if current.status == "dismissed":
                return current
            if current.status == "resolved":
                raise ValueError("resolved scope queue items cannot be dismissed")
            now = datetime.now(timezone.utc)
            connection.execute(
                """
                UPDATE scope_resolution_queue
                SET status = 'dismissed', reason = ?, resolved_at = ?, updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (
                    reason.strip(),
                    now.isoformat(),
                    now.isoformat(),
                    current.id,
                    user_id,
                ),
            )
            self._insert_event(
                connection,
                ScopeResolutionQueueEvent(
                    id=f"scope_resolution_event_{uuid4().hex}",
                    queue_item_id=current.id,
                    user_id=user_id,
                    event_type="dismissed",
                    previous_status=current.status,
                    new_status="dismissed",
                    reason=reason.strip(),
                    occurred_at=now,
                ),
            )
            row = connection.execute(
                self._select_sql("WHERE id = ? AND user_id = ?"),
                (current.id, user_id),
            ).fetchone()
        return self._item(row)

    def list_events(
        self, *, user_id: str, queue_item_id: str
    ) -> tuple[ScopeResolutionQueueEvent, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, queue_item_id, user_id, event_type, previous_status,
                       new_status, reason, occurred_at
                FROM scope_resolution_events
                WHERE queue_item_id = ? AND user_id = ?
                ORDER BY rowid
                """,
                (queue_item_id, user_id),
            ).fetchall()
        return tuple(self._event(row) for row in rows)

    @staticmethod
    def content_digest(value: str) -> str:
        return content_digest(value)

    @staticmethod
    def _idempotency_key(**parts: str) -> str:
        payload = json.dumps(
            parts,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _select_sql(suffix: str) -> str:
        return f"""
            SELECT id, user_id, conversation_id, family, subject_id,
                   relation, proposed_value,
                   source_kind, source_id, status, reason,
                   candidate_scope_keys_json, resolved_scope_key,
                   content_digest, idempotency_key, clarification_attempts,
                   created_at, updated_at, clarification_requested_at, resolved_at
            FROM scope_resolution_queue
            {suffix}
        """

    def _get_for_update(
        self, connection: sqlite3.Connection, user_id: str, queue_item_id: str
    ) -> ScopeResolutionQueueItem:
        row = connection.execute(
            self._select_sql("WHERE id = ? AND user_id = ?"),
            (queue_item_id, user_id),
        ).fetchone()
        if row is None:
            raise ValueError("scope resolution queue item not found")
        return self._item(row)

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection, event: ScopeResolutionQueueEvent
    ) -> None:
        connection.execute(
            """
            INSERT INTO scope_resolution_events(
                id, queue_item_id, user_id, event_type, previous_status,
                new_status, reason, occurred_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.id,
                event.queue_item_id,
                event.user_id,
                event.event_type,
                event.previous_status,
                event.new_status,
                event.reason,
                event.occurred_at.isoformat(),
            ),
        )

    @staticmethod
    def _item(row: tuple[object, ...]) -> ScopeResolutionQueueItem:
        return ScopeResolutionQueueItem(
            id=row[0],
            user_id=row[1],
            conversation_id=row[2],
            family=row[3],
            subject_id=row[4],
            relation=row[5],
            proposed_value=row[6],
            source_kind=row[7],
            source_id=row[8],
            status=row[9],
            reason=row[10],
            candidate_scope_keys=tuple(json.loads(str(row[11]))),
            resolved_scope_key=row[12],
            content_digest=row[13],
            idempotency_key=row[14],
            clarification_attempts=row[15],
            created_at=row[16],
            updated_at=row[17],
            clarification_requested_at=row[18],
            resolved_at=row[19],
        )

    @staticmethod
    def _event(row: tuple[object, ...]) -> ScopeResolutionQueueEvent:
        return ScopeResolutionQueueEvent(
            id=row[0],
            queue_item_id=row[1],
            user_id=row[2],
            event_type=row[3],
            previous_status=row[4],
            new_status=row[5],
            reason=row[6],
            occurred_at=row[7],
        )

    @staticmethod
    def _baseline(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS scope_resolution_queue (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                conversation_id TEXT,
                family TEXT NOT NULL CHECK (
                    family IN ('person_intent', 'target_role_intent', 'career_evidence')
                ),
                subject_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                proposed_value TEXT NOT NULL,
                source_kind TEXT NOT NULL CHECK (
                    source_kind IN ('job_intent', 'career_evidence', 'episode_consolidation')
                ),
                source_id TEXT NOT NULL,
                status TEXT NOT NULL CHECK (
                    status IN (
                        'unresolved', 'clarification_requested',
                        'resolved', 'dismissed'
                    )
                ),
                reason TEXT NOT NULL,
                candidate_scope_keys_json TEXT NOT NULL,
                resolved_scope_key TEXT,
                content_digest TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                clarification_attempts INTEGER NOT NULL DEFAULT 0
                    CHECK (clarification_attempts >= 0),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                clarification_requested_at TEXT,
                resolved_at TEXT,
                UNIQUE(user_id, idempotency_key),
                CHECK (
                    (status = 'resolved' AND resolved_scope_key IS NOT NULL AND resolved_at IS NOT NULL)
                    OR (status != 'resolved' AND resolved_scope_key IS NULL)
                ),
                CHECK (
                    status != 'dismissed' OR resolved_at IS NOT NULL
                )
            );

            CREATE TABLE IF NOT EXISTS scope_resolution_events (
                id TEXT PRIMARY KEY,
                queue_item_id TEXT NOT NULL REFERENCES scope_resolution_queue(id),
                user_id TEXT NOT NULL,
                event_type TEXT NOT NULL CHECK (
                    event_type IN (
                        'enqueued', 'clarification_requested', 'resolved', 'dismissed'
                    )
                ),
                previous_status TEXT CHECK (
                    previous_status IS NULL OR previous_status IN (
                        'unresolved', 'clarification_requested', 'resolved', 'dismissed'
                    )
                ),
                new_status TEXT NOT NULL CHECK (
                    new_status IN (
                        'unresolved', 'clarification_requested', 'resolved', 'dismissed'
                    )
                ),
                reason TEXT,
                occurred_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS scope_resolution_queue_user_status_idx
                ON scope_resolution_queue(user_id, status, created_at);
            CREATE INDEX IF NOT EXISTS scope_resolution_queue_source_idx
                ON scope_resolution_queue(user_id, source_kind, source_id);
            CREATE INDEX IF NOT EXISTS scope_resolution_events_item_idx
                ON scope_resolution_events(queue_item_id, occurred_at);
            """
        )

    @staticmethod
    def _upgrade_to_v2(connection: sqlite3.Connection) -> None:
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(scope_resolution_queue)"
            )
        }
        if "conversation_id" not in columns:
            connection.execute(
                "ALTER TABLE scope_resolution_queue "
                "ADD COLUMN conversation_id TEXT"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection
