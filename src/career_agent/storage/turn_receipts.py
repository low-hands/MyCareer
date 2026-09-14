from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sqlite3
from typing import Iterable, Literal

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
ReceiptContentStatus = Literal["available", "deleted", "expired"]
RECEIPT_BODY_TTL = timedelta(hours=24)

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
    content_status: ReceiptContentStatus = "available"


def _create_redactions_table(connection: sqlite3.Connection) -> None:
    columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(turn_receipt_redactions)")
    }
    if columns and "turn_id" not in columns:
        # The first shape marked a conversation forever. Carry its rows over as
        # "every turn it has right now", which is all they could correctly mean.
        connection.execute(
            "ALTER TABLE turn_receipt_redactions RENAME TO turn_receipt_redactions_v1"
        )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS turn_receipt_redactions (
            user_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            redacted_at TEXT NOT NULL,
            PRIMARY KEY(user_id, conversation_id, turn_id)
        )
        """
    )
    if columns and "turn_id" not in columns:
        now = datetime.now(timezone.utc)
        for user_id, conversation_id in connection.execute(
            "SELECT user_id, conversation_id FROM turn_receipt_redactions_v1"
        ).fetchall():
            redact_conversation_receipts_on(connection, user_id, conversation_id, now=now)
        connection.execute("DROP TABLE turn_receipt_redactions_v1")


def redact_turn_receipts_on(
    connection: sqlite3.Connection,
    user_id: str,
    conversation_id: str,
    turn_ids: Iterable[str],
    *,
    now: datetime | None = None,
) -> None:
    """Clear the receipts of exactly these turns, settled or still running.

    A receipt holds what one turn answered, so content removed from the
    transcript is only ever inside the receipts of the turns that showed it.
    Other turns of the same conversation keep replaying normally.
    """

    turn_ids = tuple(dict.fromkeys(turn_id for turn_id in turn_ids if turn_id))
    if turn_ids:
        _redact_on(connection, user_id, conversation_id, turn_ids, now=now)


def redact_conversation_receipts_on(
    connection: sqlite3.Connection,
    user_id: str,
    conversation_id: str,
    *,
    now: datetime | None = None,
) -> None:
    """Clear every receipt this conversation has at this moment.

    For deleting a conversation whole, or for content whose turn is unknown.
    The turns are read inside the caller's transaction, so a turn that begins
    afterwards is not one of them and keeps its receipt: no clock comparison
    decides which side of the deletion it fell on.
    """

    if not _has_receipts_table(connection):
        return
    turn_ids = tuple(
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT turn_id FROM turn_receipts "
            "WHERE user_id = ? AND conversation_id = ?",
            (user_id, conversation_id),
        ).fetchall()
    )
    redact_turn_receipts_on(connection, user_id, conversation_id, turn_ids, now=now)


def _has_receipts_table(connection: sqlite3.Connection) -> bool:
    return bool(connection.execute("PRAGMA table_info(turn_receipts)").fetchall())


def _redact_on(
    connection: sqlite3.Connection,
    user_id: str,
    conversation_id: str,
    turn_ids: tuple[str, ...],
    *,
    now: datetime | None,
) -> None:
    redacted_at = (now or datetime.now(timezone.utc)).isoformat()
    _create_redactions_table(connection)
    connection.executemany(
        "INSERT OR REPLACE INTO turn_receipt_redactions VALUES (?, ?, ?, ?)",
        ((user_id, conversation_id, turn_id, redacted_at) for turn_id in turn_ids),
    )
    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(turn_receipts)")
    }
    if not columns:
        return
    content_status = ", content_status = 'deleted'" if "content_status" in columns else ""
    placeholders = ",".join("?" for _ in turn_ids)
    connection.execute(
        f"UPDATE turn_receipts SET events_json = '[]'{content_status} "
        f"WHERE user_id = ? AND conversation_id = ? AND turn_id IN ({placeholders})",
        (user_id, conversation_id, *turn_ids),
    )


def _is_redacted_on(
    connection: sqlite3.Connection,
    user_id: str,
    conversation_id: str,
    turn_id: str,
) -> bool:
    return connection.execute(
        "SELECT 1 FROM turn_receipt_redactions "
        "WHERE user_id = ? AND conversation_id = ? AND turn_id = ?",
        (user_id, conversation_id, turn_id),
    ).fetchone() is not None


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
            apply_schema(
                connection, "turn_receipts", 2, self._migrate,
                {2: self._upgrade_to_v2},
            )
            self._purge_expired_on(connection, datetime.now(timezone.utc))
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
            self._purge_expired_on(connection, started_at)
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
        body_expires_at: datetime | None = None,
    ) -> None:
        self._settle(
            user_id=user_id,
            conversation_id=conversation_id,
            request_id=request_id,
            turn_id=turn_id,
            status="COMMITTED",
            events=coalesce_content(events),
            now=now,
            body_expires_at=body_expires_at,
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
        now: datetime | None = None,
    ) -> TurnReceipt | None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._purge_expired_on(connection, now or datetime.now(timezone.utc))
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
        body_expires_at: datetime | None = None,
    ) -> None:
        settled_at = now or datetime.now(timezone.utc)
        deadline = min(
            settled_at + RECEIPT_BODY_TTL,
            body_expires_at or settled_at + RECEIPT_BODY_TTL,
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            redacted = _is_redacted_on(connection, user_id, conversation_id, turn_id)
            # Only the attempt that owns the row may settle it. A stale settle
            # from an attempt that lost the key must not overwrite the winner.
            connection.execute(
                """
                UPDATE turn_receipts
                SET status = ?, events_json = ?, settled_at = ?, content_status = ?,
                    body_expires_at = ?
                WHERE user_id = ? AND conversation_id = ? AND request_id = ?
                  AND turn_id = ? AND status = 'RUNNING'
                """,
                (
                    status,
                    json.dumps(
                        [] if redacted else [event.model_dump(mode="json") for event in events],
                        ensure_ascii=False,
                    ),
                    settled_at.isoformat(),
                    "deleted" if redacted else "available",
                    deadline.isoformat(),
                    user_id,
                    conversation_id,
                    request_id,
                    turn_id,
                ),
            )
            self._purge_expired_on(connection, settled_at)

    _SELECT = (
        "SELECT user_id, conversation_id, request_id, turn_id, status, "
        "events_json, started_at, settled_at, content_status FROM turn_receipts"
    )

    def purge_expired(self, *, now: datetime | None = None) -> int:
        with self._connect() as connection:
            return self._purge_expired_on(connection, now or datetime.now(timezone.utc))

    @staticmethod
    def _purge_expired_on(connection: sqlite3.Connection, now: datetime) -> int:
        # A redaction only has to outlive the receipts it could still catch,
        # and nothing settles later than its own body would have lasted.
        connection.execute(
            "DELETE FROM turn_receipt_redactions WHERE julianday(redacted_at) <= julianday(?)",
            ((now - RECEIPT_BODY_TTL).isoformat(),),
        )
        return connection.execute(
            """
            UPDATE turn_receipts SET events_json = '[]', content_status = 'expired'
            WHERE status = 'COMMITTED' AND content_status = 'available'
              AND (julianday(settled_at) <= julianday(?)
                   OR julianday(body_expires_at) <= julianday(?))
            """,
            ((now - RECEIPT_BODY_TTL).isoformat(), now.isoformat()),
        ).rowcount

    @staticmethod
    def _upgrade_to_v2(connection: sqlite3.Connection) -> None:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(turn_receipts)")
        }
        if "content_status" not in columns:
            connection.execute(
                "ALTER TABLE turn_receipts ADD COLUMN content_status "
                "TEXT NOT NULL DEFAULT 'available'"
            )
        if "body_expires_at" not in columns:
            connection.execute("ALTER TABLE turn_receipts ADD COLUMN body_expires_at TEXT")
        _create_redactions_table(connection)
        connection.execute(
            """
            UPDATE turn_receipts SET events_json = '[]', content_status = 'deleted'
            WHERE EXISTS (
                SELECT 1 FROM turn_receipt_redactions AS redactions
                WHERE redactions.user_id = turn_receipts.user_id
                  AND redactions.conversation_id = turn_receipts.conversation_id
                  AND redactions.turn_id = turn_receipts.turn_id
            )
            """
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
                content_status TEXT NOT NULL DEFAULT 'available',
                body_expires_at TEXT,
                PRIMARY KEY(user_id, conversation_id, request_id)
            );
            """
        )
        _create_redactions_table(connection)

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
            content_status=row[8],
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
