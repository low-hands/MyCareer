from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
from typing import Literal

from pydantic import TypeAdapter

from career_agent.harness.streaming import (
    ArtifactReadyEvent,
    ClientActionEvent,
    ContentDeltaEvent,
    InteractionRequiredEvent,
    PublicStreamEvent,
    ReportReadyEvent,
    TurnCompletedEvent,
    TurnSuspendedEvent,
)
from career_agent.storage.schema import apply_schema


TurnReceiptStatus = Literal["RUNNING", "COMMITTED", "FAILED"]

REPLAYED_EVENT_TYPES = (
    ContentDeltaEvent,
    InteractionRequiredEvent,
    ArtifactReadyEvent,
    ReportReadyEvent,
    ClientActionEvent,
    TurnSuspendedEvent,
    TurnCompletedEvent,
)
"""The events that *are* a turn's answer, as opposed to commentary on its progress.

Progress, capability and heartbeat events describe work that a replay does not
do again, so they are not stored; ``turn_started`` and ``turn_failed`` carry the
identity of the attempt that emits them and are produced by the replayer.
"""

_EVENTS = TypeAdapter(list[PublicStreamEvent])


@dataclass(frozen=True)
class TurnReceipt:
    user_id: str
    conversation_id: str
    request_id: str
    turn_id: str
    status: TurnReceiptStatus
    events: tuple[PublicStreamEvent, ...]
    started_at: datetime
    settled_at: datetime | None


class SQLiteTurnReceiptStore:
    """One durable row per ``(user, conversation, Idempotency-Key)``.

    The action execution ledger anchors each *external write* of a turn to the
    request that asked for it, so a repeated request cannot repeat a write. It
    says nothing about the turn as a whole: the same request replayed would still
    run the model again, append a second exchange to the transcript and possibly
    answer differently. This store closes that gap at the turn level.

    ``begin`` is the only decision point. A request that finds no row, or a
    ``FAILED`` one, owns a fresh attempt; a ``COMMITTED`` row hands back what the
    first attempt answered so the caller replays it verbatim instead of
    executing; a ``RUNNING`` row means the first attempt has not finished — or
    died without settling, which is the same thing from here: the transcript is
    the place to find out which, and a new request identity is how to move on.

    A failed attempt does not keep its key. Whoever retries under the same key
    wants the turn to happen, and the ledger still protects every write the
    failed attempt already reached, so re-executing is the safe reading.
    """

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            apply_schema(connection, "turn_receipts", 1, self._migrate)
        os.chmod(self.path, 0o600)

    def begin(
        self,
        *,
        user_id: str,
        conversation_id: str,
        request_id: str,
        turn_id: str,
        now: datetime | None = None,
    ) -> TurnReceipt | None:
        """Claim the key for ``turn_id``, or return the receipt that holds it.

        ``None`` means the caller owns this attempt and must ``commit`` or
        ``fail`` it. Anything else is another attempt's receipt, ``COMMITTED``
        or ``RUNNING``, and the caller must not execute.
        """

        started_at = now or datetime.now(timezone.utc)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                self._SELECT
                + " WHERE user_id = ? AND conversation_id = ? AND request_id = ?",
                (user_id, conversation_id, request_id),
            ).fetchone()
            if row is not None:
                existing = self._receipt(row)
                if existing.status != "FAILED":
                    return existing
            connection.execute(
                """
                INSERT OR REPLACE INTO turn_receipts(
                    user_id, conversation_id, request_id, turn_id, status,
                    events_json, started_at, settled_at
                ) VALUES (?, ?, ?, ?, 'RUNNING', '[]', ?, NULL)
                """,
                (user_id, conversation_id, request_id, turn_id, started_at.isoformat()),
            )
        return None

    def commit(
        self,
        *,
        user_id: str,
        conversation_id: str,
        request_id: str,
        turn_id: str,
        events: tuple[PublicStreamEvent, ...],
        now: datetime | None = None,
    ) -> None:
        self._settle(
            user_id=user_id,
            conversation_id=conversation_id,
            request_id=request_id,
            turn_id=turn_id,
            status="COMMITTED",
            events=coalesce_content(events),
            now=now,
        )

    def fail(
        self,
        *,
        user_id: str,
        conversation_id: str,
        request_id: str,
        turn_id: str,
        now: datetime | None = None,
    ) -> None:
        self._settle(
            user_id=user_id,
            conversation_id=conversation_id,
            request_id=request_id,
            turn_id=turn_id,
            status="FAILED",
            events=(),
            now=now,
        )

    def get(
        self,
        *,
        user_id: str,
        conversation_id: str,
        request_id: str,
    ) -> TurnReceipt | None:
        with self._connect() as connection:
            row = connection.execute(
                self._SELECT
                + " WHERE user_id = ? AND conversation_id = ? AND request_id = ?",
                (user_id, conversation_id, request_id),
            ).fetchone()
        return self._receipt(row) if row is not None else None

    def _settle(
        self,
        *,
        user_id: str,
        conversation_id: str,
        request_id: str,
        turn_id: str,
        status: TurnReceiptStatus,
        events: tuple[PublicStreamEvent, ...],
        now: datetime | None,
    ) -> None:
        settled_at = now or datetime.now(timezone.utc)
        with self._connect() as connection:
            # Only the attempt that owns the row may settle it. A stale settle
            # from an attempt that lost the key must not overwrite the winner.
            connection.execute(
                """
                UPDATE turn_receipts
                SET status = ?, events_json = ?, settled_at = ?
                WHERE user_id = ? AND conversation_id = ? AND request_id = ?
                  AND turn_id = ? AND status = 'RUNNING'
                """,
                (
                    status,
                    json.dumps(
                        [event.model_dump(mode="json") for event in events],
                        ensure_ascii=False,
                    ),
                    settled_at.isoformat(),
                    user_id,
                    conversation_id,
                    request_id,
                    turn_id,
                ),
            )

    _SELECT = (
        "SELECT user_id, conversation_id, request_id, turn_id, status, "
        "events_json, started_at, settled_at FROM turn_receipts"
    )

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS turn_receipts (
                user_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                status TEXT NOT NULL,
                events_json TEXT NOT NULL,
                started_at TEXT NOT NULL,
                settled_at TEXT,
                PRIMARY KEY(user_id, conversation_id, request_id)
            );
            """
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @staticmethod
    def _receipt(row) -> TurnReceipt:
        return TurnReceipt(
            user_id=row[0],
            conversation_id=row[1],
            request_id=row[2],
            turn_id=row[3],
            status=row[4],
            events=tuple(_EVENTS.validate_python(json.loads(row[5]))),
            started_at=datetime.fromisoformat(row[6]),
            settled_at=datetime.fromisoformat(row[7]) if row[7] else None,
        )


def coalesce_content(
    events: tuple[PublicStreamEvent, ...],
) -> tuple[PublicStreamEvent, ...]:
    """Merge runs of ``content_delta`` into one event each.

    Chunking was transport pacing for the live reader; a replay hands over text
    that is already complete, and storing it as one piece keeps the receipt
    small and its text trivially equal to what was streamed.
    """

    merged: list[PublicStreamEvent] = []
    text: list[str] = []

    def flush() -> None:
        if text:
            merged.append(ContentDeltaEvent(delta="".join(text), delivery="synthetic"))
            text.clear()

    for event in events:
        if isinstance(event, ContentDeltaEvent):
            text.append(event.delta)
            continue
        flush()
        merged.append(event)
    flush()
    return tuple(merged)
