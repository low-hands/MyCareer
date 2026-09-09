from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
from typing import Literal

from career_agent.harness.observability import (
    EventType,
    ModelCallCategory,
    RunEvent,
    RunTrace,
    conversation_trace_key,
    safe_trace_fields,
    validate_model_call_category,
)
from career_agent.storage.schema import apply_schema


class SQLiteTraceRecorder:
    """Best-effort runtime telemetry in a file independent of business stores."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        with self._connect() as connection:
            apply_schema(
                connection,
                "run_events",
                2,
                self._baseline,
                {2: self._upgrade_to_v2},
            )
        os.chmod(self.path, 0o600)

    def record(
        self,
        run_id: str,
        event_type: EventType,
        stage: str,
        *,
        attempt: int | None = None,
        duration_ms: int | None = None,
        outcome: Literal["started", "succeeded", "failed", "interrupted"] = "started",
        details: dict | None = None,
        error_code: str | None = None,
        error_detail: str | None = None,
        recoverable: bool | None = None,
        model_call_category: ModelCallCategory | None = None,
    ) -> RunEvent:
        validate_model_call_category(event_type, model_call_category)
        safe_details, safe_error = safe_trace_fields(
            details=details,
            error_detail=error_detail,
        )
        occurred_at = datetime.now(timezone.utc)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) FROM run_events WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            sequence = int(row[0]) + 1
            event = RunEvent(
                run_id=run_id,
                sequence=sequence,
                event_type=event_type,
                stage=stage,
                attempt=attempt,
                occurred_at=occurred_at,
                duration_ms=duration_ms,
                outcome=outcome,
                details=safe_details,
                error_code=error_code,
                error_detail=safe_error,
                recoverable=recoverable,
                model_call_category=model_call_category,
            )
            connection.execute(
                """
                INSERT INTO run_events(
                    run_id, sequence, event_type, stage, attempt, occurred_at,
                    duration_ms, outcome, details_json, error_code,
                    error_detail, recoverable, model_call_category
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.run_id,
                    event.sequence,
                    event.event_type,
                    event.stage,
                    event.attempt,
                    event.occurred_at.isoformat(),
                    event.duration_ms,
                    event.outcome,
                    json.dumps(event.details, ensure_ascii=False, separators=(",", ":")),
                    event.error_code,
                    event.error_detail,
                    None if event.recoverable is None else int(event.recoverable),
                    event.model_call_category,
                ),
            )
        os.chmod(self.path, 0o600)
        return event

    def snapshot(self, run_id: str) -> RunTrace:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT run_id, sequence, event_type, stage, attempt, occurred_at,
                       duration_ms, outcome, details_json, error_code,
                       error_detail, recoverable, model_call_category
                FROM run_events
                WHERE run_id = ?
                ORDER BY sequence
                """,
                (run_id,),
            ).fetchall()
        return RunTrace(
            run_id=run_id,
            events=tuple(self._event(row) for row in rows),
        )

    def list_conversation_events(
        self, *, user_id: str, conversation_id: str
    ) -> tuple[RunEvent, ...]:
        """Read compact/call events across turns in execution order."""

        key = conversation_trace_key(user_id, conversation_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT run_id, sequence, event_type, stage, attempt, occurred_at,
                       duration_ms, outcome, details_json, error_code,
                       error_detail, recoverable, model_call_category
                FROM run_events
                WHERE event_type IN ('context_compacted', 'model_succeeded')
                  AND json_extract(details_json, '$.conversation_key') = ?
                ORDER BY occurred_at, rowid
                """,
                (key,),
            ).fetchall()
        return tuple(self._event(row) for row in rows)

    def list_memory_events(
        self,
        *,
        user_id: str,
        conversation_id: str,
        max_events: int = 10_000,
    ) -> tuple[RunEvent, ...]:
        """Read P2 memory observations without claiming version comparability.

        Every conversation-scoped producer must carry the same pseudonymous join
        key. Filtering each event directly makes a missing producer key visible
        in tests instead of letting an unrelated keyed event admit the whole run.
        """

        if max_events < 1:
            raise ValueError("max_events must be positive")
        key = conversation_trace_key(user_id, conversation_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT run_id, sequence, event_type, stage, attempt, occurred_at,
                       duration_ms, outcome, details_json, error_code,
                       error_detail, recoverable, model_call_category
                FROM run_events
                WHERE event_type IN (
                    'memory_context_observed',
                    'memory_tombstone_observed'
                )
                  AND json_extract(details_json, '$.conversation_key') = ?
                ORDER BY occurred_at DESC, rowid DESC
                LIMIT ?
                """,
                (key, max_events),
            ).fetchall()
        return tuple(self._event(row) for row in reversed(rows))

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @staticmethod
    def _baseline(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS run_events (
                run_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                stage TEXT NOT NULL,
                attempt INTEGER,
                occurred_at TEXT NOT NULL,
                duration_ms INTEGER,
                outcome TEXT NOT NULL,
                details_json TEXT NOT NULL,
                error_code TEXT,
                error_detail TEXT,
                recoverable INTEGER,
                model_call_category TEXT,
                PRIMARY KEY(run_id, sequence)
            )
            """
        )
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(run_events)"
            ).fetchall()
        }
        if "model_call_category" not in columns:
            connection.execute(
                "ALTER TABLE run_events ADD COLUMN model_call_category TEXT"
            )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS run_events_occurred_at_idx
            ON run_events(occurred_at)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS run_events_type_outcome_idx
            ON run_events(event_type, outcome)
            """
        )

    @staticmethod
    def _event(row: tuple[object, ...]) -> RunEvent:
        return RunEvent(
            run_id=row[0],
            sequence=row[1],
            event_type=row[2],
            stage=row[3],
            attempt=row[4],
            occurred_at=row[5],
            duration_ms=row[6],
            outcome=row[7],
            details=json.loads(str(row[8])),
            error_code=row[9],
            error_detail=row[10],
            recoverable=(None if row[11] is None else bool(row[11])),
            model_call_category=row[12],
        )

    @staticmethod
    def _upgrade_to_v2(connection: sqlite3.Connection) -> None:
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(run_events)"
            ).fetchall()
        }
        if "model_call_category" not in columns:
            connection.execute(
                "ALTER TABLE run_events ADD COLUMN model_call_category TEXT"
            )
