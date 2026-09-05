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
    user_id: str
    conversation_id: str
    anchor: str
    request_id: str | None
    write_slot: int
    tool_name: str
    fingerprint: str
    policy_epoch: int
    retry_safe: bool
    status: ActionExecutionStatus
    output: dict[str, str | int | float | bool | None]
    error_code: str | None
    error_detail: str | None
    started_at: datetime
    settled_at: datetime | None


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
                1,
                self._migrate,
            )
        os.chmod(self.path, 0o600)

    def prepare(
        self,
        *,
        user_id: str,
        conversation_id: str,
        anchor: str,
        request_id: str | None,
        write_slot: int,
        tool_name: str,
        fingerprint: str,
        policy_epoch: int,
        replay_allowed: bool = False,
        now: datetime | None = None,
    ) -> tuple[ActionExecution, bool]:
        started_at = now or datetime.now(timezone.utc)
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
                user_id=user_id,
                conversation_id=conversation_id,
                anchor=anchor,
                request_id=request_id,
                write_slot=write_slot,
                tool_name=tool_name,
                fingerprint=fingerprint,
                policy_epoch=policy_epoch,
                retry_safe=request_id is not None and replay_allowed,
                status="PENDING",
                output={},
                error_code=None,
                error_detail=None,
                started_at=started_at,
                settled_at=None,
            )
            connection.execute(
                """
                INSERT INTO action_executions(
                    action_id, user_id, conversation_id, anchor, request_id,
                    write_slot, tool_name, fingerprint, policy_epoch, retry_safe,
                    status, output_json, error_code, error_detail,
                    started_at, settled_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._values(execution),
            )
        return execution, True

    def succeed(
        self,
        *,
        action_id: str,
        output: dict[str, str | int | float | bool | None],
        now: datetime | None = None,
    ) -> ActionExecution:
        self._validate_output(output)
        settled_at = now or datetime.now(timezone.utc)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            changed = connection.execute(
                """
                UPDATE action_executions
                SET status = 'SUCCEEDED', output_json = ?, settled_at = ?,
                    error_code = NULL, error_detail = NULL
                WHERE action_id = ? AND status = 'PENDING'
                """,
                (
                    json.dumps(output, ensure_ascii=False, sort_keys=True),
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
                SET status = 'FAILED', error_code = ?, error_detail = ?, settled_at = ?
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
        "SELECT action_id, user_id, conversation_id, anchor, request_id, "
        "write_slot, tool_name, fingerprint, policy_epoch, retry_safe, status, "
        "output_json, error_code, error_detail, started_at, settled_at "
        "FROM action_executions"
    )

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS action_executions (
                action_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                anchor TEXT NOT NULL,
                request_id TEXT,
                write_slot INTEGER NOT NULL,
                tool_name TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                policy_epoch INTEGER NOT NULL,
                retry_safe INTEGER NOT NULL,
                status TEXT NOT NULL,
                output_json TEXT NOT NULL,
                error_code TEXT,
                error_detail TEXT,
                started_at TEXT NOT NULL,
                settled_at TEXT,
                UNIQUE(user_id, conversation_id, anchor, write_slot)
            );
            CREATE INDEX IF NOT EXISTS action_executions_pending_idx
                ON action_executions(status, started_at);
            """
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
            action_id=row[0], user_id=row[1], conversation_id=row[2],
            anchor=row[3], request_id=row[4], write_slot=row[5],
            tool_name=row[6], fingerprint=row[7], policy_epoch=row[8],
            retry_safe=bool(row[9]), status=row[10], output=json.loads(row[11]),
            error_code=row[12], error_detail=row[13], started_at=datetime.fromisoformat(row[14]),
            settled_at=datetime.fromisoformat(row[15]) if row[15] else None,
        )

    @staticmethod
    def _values(execution: ActionExecution) -> tuple[object, ...]:
        return (
            execution.action_id, execution.user_id, execution.conversation_id,
            execution.anchor, execution.request_id, execution.write_slot,
            execution.tool_name, execution.fingerprint, execution.policy_epoch,
            int(execution.retry_safe), execution.status,
            json.dumps(execution.output, ensure_ascii=False, sort_keys=True),
            execution.error_code, execution.error_detail,
            execution.started_at.isoformat(),
            execution.settled_at.isoformat() if execution.settled_at else None,
        )
