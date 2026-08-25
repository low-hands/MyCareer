from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
from uuid import uuid4

from career_agent.domain.action_center import (
    ActionCandidate,
    ActionItem,
    ActionItemEvent,
    ActionStatus,
    ActionType,
)


class SQLiteActionItemStore:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            self._migrate(connection)
        os.chmod(self.path, 0o600)

    def upsert_candidate(
        self,
        *,
        user_id: str,
        candidate: ActionCandidate,
        now: datetime,
    ) -> ActionItem:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                self._ACTION_SELECT + " WHERE user_id = ? AND stable_key = ?",
                (user_id, candidate.stable_key),
            ).fetchone()
            if row is None:
                item = ActionItem(
                    id=f"action_item_{uuid4().hex}",
                    user_id=user_id,
                    stable_key=candidate.stable_key,
                    action_type=candidate.action_type,
                    source_type=candidate.source_type,
                    source_id=candidate.source_id,
                    application_id=candidate.application_id,
                    title=candidate.title,
                    summary=candidate.summary,
                    due_at=candidate.due_at,
                    status="open",
                    created_at=now,
                    updated_at=now,
                )
                connection.execute(
                    """
                    INSERT INTO action_items(
                        id, user_id, stable_key, action_type, source_type,
                        source_id, application_id, title, summary, due_at,
                        status, snoozed_until, created_at, updated_at, resolved_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    self._values(item),
                )
                self._insert_event(connection, item, "created", None, "open", now)
                return item
            item = self._item(row)
            if item.status in {"completed", "dismissed"}:
                return item
            status: ActionStatus = item.status
            snoozed_until = item.snoozed_until
            event_type = "refreshed"
            if status == "snoozed" and snoozed_until is not None and snoozed_until <= now:
                status = "open"
                snoozed_until = None
                event_type = "reopened"
            updated = item.model_copy(
                update={
                    "application_id": candidate.application_id,
                    "title": candidate.title,
                    "summary": candidate.summary,
                    "due_at": candidate.due_at,
                    "status": status,
                    "snoozed_until": snoozed_until,
                    "updated_at": now,
                }
            )
            connection.execute(
                """
                UPDATE action_items SET
                    application_id = ?, title = ?, summary = ?, due_at = ?,
                    status = ?, snoozed_until = ?, updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (
                    updated.application_id, updated.title, updated.summary,
                    self._iso(updated.due_at), updated.status,
                    self._iso(updated.snoozed_until), updated.updated_at.isoformat(),
                    updated.id, updated.user_id,
                ),
            )
            if event_type == "reopened":
                self._insert_event(
                    connection, updated, event_type, item.status, updated.status, now
                )
            return updated

    def resolve_missing(
        self,
        *,
        user_id: str,
        managed_types: tuple[ActionType, ...],
        active_keys: frozenset[str],
        now: datetime,
    ) -> None:
        if not managed_types:
            return
        placeholders = ",".join("?" for _ in managed_types)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                self._ACTION_SELECT
                + f" WHERE user_id = ? AND action_type IN ({placeholders}) "
                "AND status IN ('open', 'snoozed')",
                (user_id, *managed_types),
            ).fetchall()
            for row in rows:
                item = self._item(row)
                if item.stable_key in active_keys:
                    continue
                connection.execute(
                    """
                    UPDATE action_items
                    SET status = 'completed', snoozed_until = NULL,
                        updated_at = ?, resolved_at = ?
                    WHERE id = ? AND user_id = ?
                    """,
                    (now.isoformat(), now.isoformat(), item.id, user_id),
                )
                self._insert_event(
                    connection, item, "completed", item.status, "completed", now
                )

    def get(self, *, user_id: str, action_item_id: str) -> ActionItem | None:
        with self._connect() as connection:
            row = connection.execute(
                self._ACTION_SELECT + " WHERE id = ? AND user_id = ?",
                (action_item_id, user_id),
            ).fetchone()
        return self._item(row) if row else None

    def list(
        self,
        *,
        user_id: str,
        statuses: tuple[ActionStatus, ...] = ("open", "snoozed"),
        limit: int = 100,
    ) -> tuple[ActionItem, ...]:
        query = self._ACTION_SELECT + " WHERE user_id = ?"
        params: list[object] = [user_id]
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            query += f" AND status IN ({placeholders})"
            params.extend(statuses)
        query += " ORDER BY due_at IS NULL, due_at, created_at LIMIT ?"
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return tuple(self._item(row) for row in rows)

    def transition(
        self,
        *,
        user_id: str,
        action_item_id: str,
        status: ActionStatus,
        snoozed_until: datetime | None = None,
        now: datetime | None = None,
    ) -> ActionItem | None:
        changed_at = now or datetime.now(timezone.utc)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                self._ACTION_SELECT + " WHERE id = ? AND user_id = ?",
                (action_item_id, user_id),
            ).fetchone()
            if row is None:
                return None
            item = self._item(row)
            if item.status in {"completed", "dismissed"}:
                return item
            resolved_at = changed_at if status in {"completed", "dismissed"} else None
            updated = item.model_copy(
                update={
                    "status": status,
                    "snoozed_until": snoozed_until if status == "snoozed" else None,
                    "updated_at": changed_at,
                    "resolved_at": resolved_at,
                }
            )
            # Validate model_copy updates before persistence.
            updated = ActionItem.model_validate(updated.model_dump())
            connection.execute(
                """
                UPDATE action_items SET status = ?, snoozed_until = ?,
                    updated_at = ?, resolved_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (
                    updated.status, self._iso(updated.snoozed_until),
                    updated.updated_at.isoformat(), self._iso(updated.resolved_at),
                    updated.id, updated.user_id,
                ),
            )
            event_type = {
                "completed": "completed",
                "dismissed": "dismissed",
                "snoozed": "snoozed",
                "open": "reopened",
            }[status]
            self._insert_event(
                connection, updated, event_type, item.status, updated.status, changed_at
            )
        return updated

    def list_events(
        self, *, user_id: str, action_item_id: str
    ) -> tuple[ActionItemEvent, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, user_id, action_item_id, event_type,
                       previous_status, new_status, occurred_at
                FROM action_item_events
                WHERE user_id = ? AND action_item_id = ?
                ORDER BY occurred_at, rowid
                """,
                (user_id, action_item_id),
            ).fetchall()
        return tuple(self._event(row) for row in rows)

    _ACTION_SELECT = (
        "SELECT id, user_id, stable_key, action_type, source_type, source_id, "
        "application_id, title, summary, due_at, status, snoozed_until, "
        "created_at, updated_at, resolved_at FROM action_items"
    )

    def _migrate(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS action_items (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                stable_key TEXT NOT NULL,
                action_type TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_id TEXT NOT NULL,
                application_id TEXT,
                title TEXT NOT NULL,
                summary TEXT NOT NULL,
                due_at TEXT,
                status TEXT NOT NULL,
                snoozed_until TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                resolved_at TEXT,
                UNIQUE(user_id, stable_key)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS action_item_events (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                action_item_id TEXT NOT NULL REFERENCES action_items(id),
                event_type TEXT NOT NULL,
                previous_status TEXT,
                new_status TEXT NOT NULL,
                occurred_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS action_items_user_due_idx
            ON action_items(user_id, status, due_at)
            """
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @classmethod
    def _values(cls, item: ActionItem) -> tuple[object, ...]:
        return (
            item.id, item.user_id, item.stable_key, item.action_type,
            item.source_type, item.source_id, item.application_id, item.title,
            item.summary, cls._iso(item.due_at), item.status,
            cls._iso(item.snoozed_until), item.created_at.isoformat(),
            item.updated_at.isoformat(), cls._iso(item.resolved_at),
        )

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection,
        item: ActionItem,
        event_type: str,
        previous_status: ActionStatus | None,
        new_status: ActionStatus,
        occurred_at: datetime,
    ) -> None:
        connection.execute(
            """
            INSERT INTO action_item_events(
                id, user_id, action_item_id, event_type,
                previous_status, new_status, occurred_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"action_item_event_{uuid4().hex}", item.user_id, item.id,
                event_type, previous_status, new_status, occurred_at.isoformat(),
            ),
        )

    @staticmethod
    def _item(row: tuple[object, ...]) -> ActionItem:
        return ActionItem(
            id=row[0], user_id=row[1], stable_key=row[2], action_type=row[3],
            source_type=row[4], source_id=row[5], application_id=row[6],
            title=row[7], summary=row[8], due_at=row[9], status=row[10],
            snoozed_until=row[11], created_at=row[12], updated_at=row[13],
            resolved_at=row[14],
        )

    @staticmethod
    def _event(row: tuple[object, ...]) -> ActionItemEvent:
        return ActionItemEvent(
            id=row[0], user_id=row[1], action_item_id=row[2], event_type=row[3],
            previous_status=row[4], new_status=row[5], occurred_at=row[6],
        )

    @staticmethod
    def _iso(value: datetime | None) -> str | None:
        return value.isoformat() if value is not None else None
