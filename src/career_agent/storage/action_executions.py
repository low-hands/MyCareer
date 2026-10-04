from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Literal

from career_agent.storage.schema import apply_schema


ActionExecutionStatus = Literal["PENDING", "SUCCEEDED", "FAILED"]
OperationPhase = Literal[
    "PREPARED",
    "RUNNING",
    "RECONCILIATION_REQUIRED",
    "EFFECT_COMMITTED",
    "COMPLETED",
    "FAILED",
]
OperationRecoveryPolicy = Literal["retry", "reconcile"]
RESULT_STATE_RECEIPT_KEY = "__result_state__"
"""Reserved receipt field carrying the original reducer result state."""


class ActionExecutionConflictError(RuntimeError):
    """One durable request slot was reused for a different write."""


class ActionExecutionReconciliationRequiredError(ActionExecutionConflictError):
    """The conflicting slot is still ``PENDING``, so its outcome is unknown.

    A subclass, not a sibling: every caller that already refuses to execute on a
    conflict keeps refusing, unchanged. What the narrower type adds is that the
    slot is not a dead end — it names an action that was prepared and may have
    reached the outside world, and that state has to be resolved before anything
    else can happen in this slot.

    Raising the general conflict here would leave the row unreachable. The
    caller is blocked, the row stays ``PENDING`` forever, and nothing in the
    system can say whether the external effect happened. A policy change would
    then permanently strand exactly the executions it was meant to govern.

    The distinction is between two different questions, which the previous
    single check conflated:

    * *May a new action start here?* — governed by policy, arguments and tool.
      A mismatch is a refusal.
    * *What happened to the action already started here?* — a **read** of
      external state. Nothing about "may I act" can gate finding out what was
      already done.

    This mirrors what Calendar already does one layer up: an epoch change
    supersedes a proposal only while it is still ``pending``; once it is
    ``executing`` or ``reconciliation_required`` the service falls through to
    recovery regardless of the policy change (``services/calendar.py``). It also
    matches CapLease, where revocation prevents *subsequent preparation* while a
    Prepared slot is always resolvable by recovery
    (https://arxiv.org/html/2608.01710v1).

    After reconciliation the outcome decides what may follow:

    * confirmed executed — settle with ``succeed``; policy is not consulted,
      because the effect already happened;
    * confirmed not executed — settle with ``fail`` and say so in the detail,
      then a fresh attempt needs a **new request identity**. Re-sending under
      the identity that is now terminal would look like an unrelated failure to
      whoever retries it.
    """

    def __init__(self, execution: "ActionExecution") -> None:
        super().__init__(
            f"request slot is bound to pending action {execution.action_id}; "
            "reconcile its outcome before starting a different write here"
        )
        self.execution = execution


class ActionExecutionAlreadyFailedError(RuntimeError):
    """A terminally failed action was replayed under the same request identity."""


@dataclass(frozen=True)
class ActionExecution:
    action_id: str
    turn_id: str
    user_id: str
    conversation_id: str
    anchor: str
    request_id: str | None
    write_slot: int
    tool_name: str
    fingerprint: str
    policy_epoch: int
    retry_safe: bool
    recovery_policy: OperationRecoveryPolicy
    input_references: dict[str, str | int | float | bool | None]
    output_references: dict[str, str | int | float | bool | None]
    external_reference: str | None
    status: ActionExecutionStatus
    phase: OperationPhase
    attempt_count: int
    output: dict[str, str | int | float | bool | None]
    error_code: str | None
    error_detail: str | None
    started_at: datetime
    last_attempt_at: datetime | None
    effect_committed_at: datetime | None
    completed_at: datetime | None
    settled_at: datetime | None

    @property
    def operation_id(self) -> str:
        """Stable operation identity shared by retries and recovery."""

        return self.action_id

    @property
    def capability(self) -> str:
        return self.tool_name


class SQLiteActionExecutionStore:
    """Durable admission slots for writes without their own authorization record.

    This is the request-anchored form of the protocol already used by Calendar.
    Identity and arguments deliberately stay separate: the slot allocates the
    action id, while ``fingerprint`` only detects a conflicting reuse.
    """

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            apply_schema(
                connection,
                "action_executions",
                3,
                self._migrate,
                {2: self._upgrade_to_v2, 3: self._upgrade_to_v3},
            )
        os.chmod(self.path, 0o600)

    def prepare(
        self,
        *,
        user_id: str,
        conversation_id: str,
        turn_id: str | None = None,
        anchor: str,
        request_id: str | None,
        write_slot: int,
        tool_name: str,
        fingerprint: str,
        policy_epoch: int,
        replay_allowed: bool = False,
        recovery_policy: OperationRecoveryPolicy = "reconcile",
        input_references: dict[str, str | int | float | bool | None] | None = None,
        now: datetime | None = None,
    ) -> tuple[ActionExecution, bool]:
        started_at = now or datetime.now(timezone.utc)
        if recovery_policy not in {"retry", "reconcile"}:
            raise ValueError("operation recovery policy must be retry or reconcile")
        references = dict(input_references or {})
        self._validate_output(references)
        action_id = self.action_id(
            user_id=user_id,
            conversation_id=conversation_id,
            anchor=anchor,
            write_slot=write_slot,
            tool_name=tool_name,
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                self._SELECT
                + " WHERE user_id = ? AND conversation_id = ?"
                " AND anchor = ? AND write_slot = ?",
                (user_id, conversation_id, anchor, write_slot),
            ).fetchone()
            if row is not None:
                existing = self._execution(row)
                if (
                    existing.tool_name != tool_name
                    or existing.fingerprint != fingerprint
                    or existing.policy_epoch != policy_epoch
                ):
                    # Both branches refuse to execute. They differ in what the
                    # caller is told to do next, and only a pending row has a
                    # next step that is not "give up": its outcome is unknown
                    # and stays unknown until someone reconciles it.
                    if existing.status == "PENDING":
                        raise ActionExecutionReconciliationRequiredError(existing)
                    raise ActionExecutionConflictError(
                        "the request write slot is already bound to a different action"
                    )
                return existing, False
            execution = ActionExecution(
                action_id=action_id,
                turn_id=turn_id or anchor,
                user_id=user_id,
                conversation_id=conversation_id,
                anchor=anchor,
                request_id=request_id,
                write_slot=write_slot,
                tool_name=tool_name,
                fingerprint=fingerprint,
                policy_epoch=policy_epoch,
                retry_safe=request_id is not None and replay_allowed,
                recovery_policy=recovery_policy,
                input_references=references,
                output_references={},
                external_reference=None,
                status="PENDING",
                phase="PREPARED",
                attempt_count=0,
                output={},
                error_code=None,
                error_detail=None,
                started_at=started_at,
                last_attempt_at=None,
                effect_committed_at=None,
                completed_at=None,
                settled_at=None,
            )
            connection.execute(
                """
                INSERT INTO action_executions(
                    action_id, turn_id, user_id, conversation_id, anchor, request_id,
                    write_slot, tool_name, fingerprint, policy_epoch, retry_safe, status,
                    recovery_policy, input_references_json, output_references_json,
                    external_reference, phase, attempt_count, output_json, error_code, error_detail,
                    started_at, last_attempt_at, effect_committed_at, completed_at,
                    settled_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._values(execution),
            )
        return execution, True

    def mark_running(
        self,
        *,
        action_id: str,
        now: datetime | None = None,
    ) -> ActionExecution:
        """Record one physical attempt before any side effect is invoked."""

        attempted_at = now or datetime.now(timezone.utc)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            changed = connection.execute(
                """
                UPDATE action_executions
                SET phase = 'RUNNING', attempt_count = attempt_count + 1,
                    last_attempt_at = ?
                WHERE action_id = ? AND status = 'PENDING'
                  AND phase IN ('PREPARED', 'RUNNING')
                """,
                (attempted_at.isoformat(), action_id),
            )
            if changed.rowcount != 1:
                raise ValueError("operation is not available for execution")
            row = connection.execute(
                self._SELECT + " WHERE action_id = ?", (action_id,)
            ).fetchone()
        return self._execution(row)

    def require_reconciliation(
        self,
        *,
        action_id: str,
        error_code: str,
        error_detail: str,
    ) -> ActionExecution:
        """Make an uncertain effect explicit without pretending it failed."""

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            changed = connection.execute(
                """
                UPDATE action_executions
                SET phase = 'RECONCILIATION_REQUIRED', error_code = ?,
                    error_detail = ?
                WHERE action_id = ? AND status = 'PENDING'
                """,
                (error_code[:100], error_detail[:2000], action_id),
            )
            row = connection.execute(
                self._SELECT + " WHERE action_id = ?", (action_id,)
            ).fetchone()
        if row is None:
            raise ValueError("operation not found")
        if changed.rowcount != 1:
            raise ValueError("operation is no longer pending")
        return self._execution(row)

    def succeed(
        self,
        *,
        action_id: str,
        output: dict[str, str | int | float | bool | None],
        output_references: dict[str, str | int | float | bool | None] | None = None,
        external_reference: str | None = None,
        now: datetime | None = None,
    ) -> ActionExecution:
        self._validate_output(output)
        references = dict(output_references or {})
        self._validate_output(references)
        settled_at = now or datetime.now(timezone.utc)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            changed = connection.execute(
                """
                UPDATE action_executions
                SET status = 'SUCCEEDED', phase = 'EFFECT_COMMITTED',
                    output_json = ?, output_references_json = ?, external_reference = ?,
                    effect_committed_at = ?, settled_at = ?,
                    error_code = NULL, error_detail = NULL
                WHERE action_id = ? AND status = 'PENDING'
                """,
                (
                    json.dumps(output, ensure_ascii=False, sort_keys=True),
                    json.dumps(references, ensure_ascii=False, sort_keys=True),
                    external_reference,
                    settled_at.isoformat(),
                    settled_at.isoformat(),
                    action_id,
                ),
            )
            if changed.rowcount != 1:
                raise ValueError("action execution is no longer pending")
            row = connection.execute(
                self._SELECT + " WHERE action_id = ?", (action_id,)
            ).fetchone()
        return self._execution(row)

    def fail(
        self,
        *,
        action_id: str,
        error_code: str,
        error_detail: str,
        now: datetime | None = None,
    ) -> ActionExecution:
        settled_at = now or datetime.now(timezone.utc)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            changed = connection.execute(
                """
                UPDATE action_executions
                SET status = 'FAILED', phase = 'FAILED', error_code = ?,
                    error_detail = ?, settled_at = ?
                WHERE action_id = ? AND status = 'PENDING'
                """,
                (error_code, error_detail[:2000], settled_at.isoformat(), action_id),
            )
            row = connection.execute(
                self._SELECT + " WHERE action_id = ?", (action_id,)
            ).fetchone()
        if row is None:
            raise ValueError("action execution not found")
        if changed.rowcount != 1:
            # Symmetric with ``succeed``. Returning the untouched row would tell
            # the caller "done" while its finding was discarded: an operator
            # settling a second time would read the earlier outcome back as if
            # it were their own result.
            raise ValueError("action execution is no longer pending")
        return self._execution(row)

    def complete_for_anchor(
        self,
        *,
        user_id: str,
        conversation_id: str,
        anchor: str,
        now: datetime | None = None,
    ) -> tuple[ActionExecution, ...]:
        """Close effects after their reducer projection and turn commit succeed."""

        completed_at = now or datetime.now(timezone.utc)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE action_executions
                SET phase = 'COMPLETED', completed_at = ?
                WHERE user_id = ? AND conversation_id = ? AND anchor = ?
                  AND status = 'SUCCEEDED' AND phase = 'EFFECT_COMMITTED'
                """,
                (
                    completed_at.isoformat(),
                    user_id,
                    conversation_id,
                    anchor,
                ),
            )
            rows = connection.execute(
                self._SELECT
                + " WHERE user_id = ? AND conversation_id = ? AND anchor = ?"
                " ORDER BY write_slot, started_at",
                (user_id, conversation_id, anchor),
            ).fetchall()
        return tuple(self._execution(row) for row in rows)

    def list_pending(
        self, *, user_id: str | None = None
    ) -> tuple[ActionExecution, ...]:
        query = self._SELECT + " WHERE status = 'PENDING'"
        params: tuple[object, ...] = ()
        if user_id is not None:
            query += " AND user_id = ?"
            params = (user_id,)
        query += " ORDER BY started_at, action_id"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return tuple(self._execution(row) for row in rows)

    def list_incomplete(
        self, *, user_id: str | None = None
    ) -> tuple[ActionExecution, ...]:
        """Return operations that still need execution, recovery, or projection."""

        query = self._SELECT + " WHERE phase NOT IN ('COMPLETED', 'FAILED')"
        params: tuple[object, ...] = ()
        if user_id is not None:
            query += " AND user_id = ?"
            params = (user_id,)
        query += " ORDER BY started_at, action_id"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return tuple(self._execution(row) for row in rows)

    def get(self, *, action_id: str) -> ActionExecution | None:
        """Return one execution without changing its reconciliation state."""

        with self._connect() as connection:
            row = connection.execute(
                self._SELECT + " WHERE action_id = ?", (action_id,)
            ).fetchone()
        return self._execution(row) if row is not None else None

    def list_for_anchor(
        self, *, user_id: str, conversation_id: str, anchor: str
    ) -> tuple[ActionExecution, ...]:
        """Every slot one request identity opened, settled or not.

        The interrupted-turn record needs both: a settled row says the write
        happened, and a pending one says it started and nobody knows. The
        in-process ledger this replaced could only ever report the first —
        it appended after the call returned, so the case it existed for, a
        process dying mid-call, left it empty.
        """
        with self._connect() as connection:
            rows = connection.execute(
                self._SELECT
                + " WHERE user_id = ? AND conversation_id = ? AND anchor = ?"
                " ORDER BY write_slot, started_at",
                (user_id, conversation_id, anchor),
            ).fetchall()
        return tuple(self._execution(row) for row in rows)

    @staticmethod
    def action_id(
        *,
        user_id: str,
        conversation_id: str,
        anchor: str,
        write_slot: int,
        tool_name: str,
    ) -> str:
        material = "\0".join(
            (user_id, conversation_id, anchor, str(write_slot), tool_name)
        )
        return "action_" + hashlib.sha256(material.encode()).hexdigest()

    _SELECT = (
        "SELECT action_id, turn_id, user_id, conversation_id, anchor, request_id, "
        "write_slot, tool_name, fingerprint, policy_epoch, retry_safe, status, "
        "recovery_policy, input_references_json, output_references_json, external_reference, "
        "phase, attempt_count, output_json, error_code, error_detail, started_at, "
        "last_attempt_at, effect_committed_at, completed_at, settled_at "
        "FROM action_executions"
    )

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS action_executions (
                action_id TEXT PRIMARY KEY,
                turn_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                anchor TEXT NOT NULL,
                request_id TEXT,
                write_slot INTEGER NOT NULL,
                tool_name TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                policy_epoch INTEGER NOT NULL,
                retry_safe INTEGER NOT NULL,
                recovery_policy TEXT NOT NULL,
                input_references_json TEXT NOT NULL,
                output_references_json TEXT NOT NULL,
                external_reference TEXT,
                status TEXT NOT NULL,
                phase TEXT NOT NULL,
                attempt_count INTEGER NOT NULL,
                output_json TEXT NOT NULL,
                error_code TEXT,
                error_detail TEXT,
                started_at TEXT NOT NULL,
                last_attempt_at TEXT,
                effect_committed_at TEXT,
                completed_at TEXT,
                settled_at TEXT,
                UNIQUE(user_id, conversation_id, anchor, write_slot)
            );
            CREATE INDEX IF NOT EXISTS action_executions_pending_idx
                ON action_executions(status, started_at);
            """
        )

    @staticmethod
    def _upgrade_to_v2(connection: sqlite3.Connection) -> None:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(action_executions)")
        }
        additions = (
            ("turn_id", "TEXT"),
            ("phase", "TEXT"),
            ("attempt_count", "INTEGER NOT NULL DEFAULT 0"),
            ("last_attempt_at", "TEXT"),
            ("effect_committed_at", "TEXT"),
            ("completed_at", "TEXT"),
        )
        for name, declaration in additions:
            if name not in columns:
                connection.execute(
                    f"ALTER TABLE action_executions ADD COLUMN {name} {declaration}"
                )
        connection.execute(
            """
            UPDATE action_executions
            SET turn_id = COALESCE(turn_id, anchor),
                phase = COALESCE(
                    phase,
                    CASE status
                        WHEN 'SUCCEEDED' THEN 'COMPLETED'
                        WHEN 'FAILED' THEN 'FAILED'
                        ELSE 'RECONCILIATION_REQUIRED'
                    END
                ),
                attempt_count = CASE
                    WHEN attempt_count = 0 THEN 1 ELSE attempt_count
                END,
                effect_committed_at = CASE
                    WHEN status = 'SUCCEEDED'
                    THEN COALESCE(effect_committed_at, settled_at)
                    ELSE effect_committed_at
                END,
                completed_at = CASE
                    WHEN status = 'SUCCEEDED'
                    THEN COALESCE(completed_at, settled_at)
                    ELSE completed_at
                END
            """
        )

    @staticmethod
    def _upgrade_to_v3(connection: sqlite3.Connection) -> None:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(action_executions)")
        }
        additions = (
            ("recovery_policy", "TEXT NOT NULL DEFAULT 'reconcile'"),
            ("input_references_json", "TEXT NOT NULL DEFAULT '{}'"),
            ("output_references_json", "TEXT NOT NULL DEFAULT '{}'"),
            ("external_reference", "TEXT"),
        )
        for name, declaration in additions:
            if name not in columns:
                connection.execute(
                    f"ALTER TABLE action_executions ADD COLUMN {name} {declaration}"
                )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @staticmethod
    def _validate_output(output: dict[str, object]) -> None:
        for key, value in output.items():
            if not isinstance(key, str) or not key or len(key) > 100:
                raise ValueError("action output keys must be short non-empty strings")
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise ValueError("action output may contain scalar receipts only")
            if isinstance(value, str) and len(value) > 2000:
                raise ValueError("action output strings may not exceed 2000 characters")

    @staticmethod
    def _execution(row) -> ActionExecution:
        return ActionExecution(
            action_id=row[0], turn_id=row[1], user_id=row[2], conversation_id=row[3],
            anchor=row[4], request_id=row[5], write_slot=row[6],
            tool_name=row[7], fingerprint=row[8], policy_epoch=row[9],
            retry_safe=bool(row[10]), status=row[11],
            recovery_policy=row[12], input_references=json.loads(row[13]),
            output_references=json.loads(row[14]), external_reference=row[15],
            phase=row[16], attempt_count=row[17], output=json.loads(row[18]),
            error_code=row[19], error_detail=row[20], started_at=datetime.fromisoformat(row[21]),
            last_attempt_at=datetime.fromisoformat(row[22]) if row[22] else None,
            effect_committed_at=datetime.fromisoformat(row[23]) if row[23] else None,
            completed_at=datetime.fromisoformat(row[24]) if row[24] else None,
            settled_at=datetime.fromisoformat(row[25]) if row[25] else None,
        )

    @staticmethod
    def _values(execution: ActionExecution) -> tuple[object, ...]:
        return (
            execution.action_id, execution.turn_id, execution.user_id,
            execution.conversation_id, execution.anchor, execution.request_id,
            execution.write_slot,
            execution.tool_name, execution.fingerprint, execution.policy_epoch,
            int(execution.retry_safe), execution.status, execution.recovery_policy,
            json.dumps(execution.input_references, ensure_ascii=False, sort_keys=True),
            json.dumps(execution.output_references, ensure_ascii=False, sort_keys=True),
            execution.external_reference, execution.phase,
            execution.attempt_count,
            json.dumps(execution.output, ensure_ascii=False, sort_keys=True),
            execution.error_code, execution.error_detail,
            execution.started_at.isoformat(),
            execution.last_attempt_at.isoformat() if execution.last_attempt_at else None,
            execution.effect_committed_at.isoformat() if execution.effect_committed_at else None,
            execution.completed_at.isoformat() if execution.completed_at else None,
            execution.settled_at.isoformat() if execution.settled_at else None,
        )


# New code uses the operation terminology. The old names remain stable for CLI,
# persisted data, and integrations compiled against the previous API.
OperationRecord = ActionExecution
SQLiteOperationJournal = SQLiteActionExecutionStore
