"""Actions an owner rule stopped, held until the owner answers.

An owner rule with the verdict ``review`` means "you may do this, but not
without asking me". That is only true if the asking survives the turn it
happened in: the model is not consulted again when the answer arrives, so the
action the owner approves has to be the same action, with the same arguments,
that was stopped — recorded somewhere that outlives the process.

Without this table ``review`` degrades into ``deny``. The runtime refuses,
tells the model to ask, the user says yes, and the next turn re-evaluates the
same unchanged rule and refuses again. A rule the owner cannot satisfy is not a
review gate; it is a permanent block wearing one's name.

The shape is taken from the calendar proposal path, which already solves this
for one capability: seal the intent with a hash of its payload, give it an
expiry, and let execution consume it exactly once. This is that protocol with
the capability left open.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from career_agent.storage.schema import apply_schema


DEFAULT_EXPIRY_MINUTES = 60
DEFAULT_EXECUTION_LEASE_SECONDS = 30
"""How long a stopped action waits for its answer.

Long enough for a person to read the question and reply, short enough that a
"yes" cannot land on an action whose context the user no longer remembers
proposing. Enforced at read time rather than by a sweep, so an expired
confirmation stops working even if nothing has run to clean it up.
"""

ConfirmationStatus = Literal[
    "PENDING",
    "APPLYING",
    "EXECUTED",
    "FAILED",
    "RECONCILIATION_REQUIRED",
    "CANCELLED",
]


class CapabilityConfirmationExpiredError(RuntimeError):
    """The owner answered, but too late for this sealed action."""


class CapabilityConfirmationSettledError(RuntimeError):
    """Already consumed or cancelled; a second answer changes nothing."""


class CapabilityConfirmationInProgressError(RuntimeError):
    """Another process currently holds the execution lease."""


def arguments_hash(arguments: dict[str, Any]) -> str:
    """Bind the sealed contract to its exact parameters.

    The owner is approving one concrete action, not a capability in general.
    Hashing the projected arguments is what makes "approve" mean "approve
    this", and what lets execution refuse if anything about the action drifted
    between the question and the answer.
    """

    canonical = json.dumps(
        arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class CapabilityConfirmation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    confirmation_id: str
    user_id: str
    conversation_id: str
    capability: str
    display_summary: str
    arguments: dict[str, Any]
    arguments_hash: str
    policy_revision: int
    status: ConfirmationStatus
    attempt_count: int = 0
    created_at: datetime
    expires_at: datetime
    lease_expires_at: datetime | None = None
    settled_at: datetime | None = None


class SQLiteCapabilityConfirmationStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        with self._connect() as connection:
            apply_schema(
                connection,
                "capability_confirmations",
                2,
                self._migrate,
                {2: self._upgrade_to_v2},
            )
        os.chmod(self.path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30.0)

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS capability_confirmations (
                confirmation_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                capability TEXT NOT NULL,
                display_summary TEXT NOT NULL DEFAULT '',
                arguments_json TEXT NOT NULL,
                arguments_hash TEXT NOT NULL,
                policy_revision INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                lease_expires_at TEXT,
                settled_at TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS capability_confirmations_open_action_idx
                ON capability_confirmations(
                    user_id, conversation_id, capability, arguments_hash
                )
                WHERE status IN ('PENDING', 'APPLYING')
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS capability_confirmations_open_idx "
            "ON capability_confirmations(user_id, conversation_id, status)"
        )

    @staticmethod
    def _upgrade_to_v2(connection: sqlite3.Connection) -> None:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(capability_confirmations)")
        }
        if "policy_revision" not in columns:
            connection.execute(
                "ALTER TABLE capability_confirmations ADD COLUMN "
                "policy_revision INTEGER NOT NULL DEFAULT 0"
            )
        if "display_summary" not in columns:
            connection.execute(
                "ALTER TABLE capability_confirmations ADD COLUMN "
                "display_summary TEXT NOT NULL DEFAULT ''"
            )
        if "attempt_count" not in columns:
            connection.execute(
                "ALTER TABLE capability_confirmations ADD COLUMN "
                "attempt_count INTEGER NOT NULL DEFAULT 0"
            )
        if "lease_expires_at" not in columns:
            connection.execute(
                "ALTER TABLE capability_confirmations ADD COLUMN lease_expires_at TEXT"
            )
        # v1 consumed approval before it ran the capability. Such a row cannot
        # truthfully be called executed after an upgrade.
        connection.execute(
            "UPDATE capability_confirmations SET status='RECONCILIATION_REQUIRED' "
            "WHERE status='CONFIRMED'"
        )
        connection.execute("DROP INDEX IF EXISTS capability_confirmations_pending_idx")
        # One pending seal per identical action. A model that proposes the same
        # thing twice must not stack two questions the owner has to answer
        # separately, and re-proposing has to return the seal already waiting or
        # the first question's id would go stale while still on screen.
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS capability_confirmations_open_action_idx
                ON capability_confirmations(
                    user_id, conversation_id, capability, arguments_hash
                )
                WHERE status IN ('PENDING', 'APPLYING')
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS capability_confirmations_open_idx
                ON capability_confirmations(user_id, conversation_id, status)
            """
        )

    _SELECT = (
        "SELECT confirmation_id, user_id, conversation_id, capability, display_summary, "
        "arguments_json, arguments_hash, policy_revision, status, attempt_count, "
        "created_at, expires_at, lease_expires_at, settled_at "
        "FROM capability_confirmations"
    )

    @staticmethod
    def _row(row: tuple) -> CapabilityConfirmation:
        return CapabilityConfirmation(
            confirmation_id=row[0],
            user_id=row[1],
            conversation_id=row[2],
            capability=row[3],
            display_summary=row[4],
            arguments=json.loads(row[5]),
            arguments_hash=row[6],
            policy_revision=row[7],
            status=row[8],
            attempt_count=row[9],
            created_at=datetime.fromisoformat(row[10]),
            expires_at=datetime.fromisoformat(row[11]),
            lease_expires_at=datetime.fromisoformat(row[12]) if row[12] else None,
            settled_at=datetime.fromisoformat(row[13]) if row[13] else None,
        )

    def seal(
        self,
        *,
        user_id: str,
        conversation_id: str,
        capability: str,
        display_summary: str,
        arguments: dict[str, Any],
        policy_revision: int,
        expires_in_minutes: int = DEFAULT_EXPIRY_MINUTES,
    ) -> CapabilityConfirmation:
        """Record the stopped action, or return the seal already waiting for it."""

        digest = arguments_hash(arguments)
        now = datetime.now(timezone.utc)
        with self._connect() as connection:
            existing = connection.execute(
                self._SELECT
                + " WHERE user_id = ? AND conversation_id = ? AND capability = ?"
                + "   AND arguments_hash = ? AND status IN ('PENDING', 'APPLYING')",
                (user_id, conversation_id, capability, digest),
            ).fetchone()
            if existing is not None:
                confirmation = self._row(existing)
                if (
                    confirmation.expires_at > now
                    and confirmation.policy_revision == policy_revision
                ):
                    return confirmation
                if confirmation.status == "APPLYING":
                    return confirmation
                # Expired while still pending: settle it as cancelled so the
                # unique index frees up, then seal a fresh one below. The user
                # is being asked again, which is correct — the old question's
                # answer window closed.
                connection.execute(
                    "UPDATE capability_confirmations SET status = 'CANCELLED', "
                    "settled_at = ? WHERE confirmation_id = ?",
                    (now.isoformat(), confirmation.confirmation_id),
                )
            confirmation = CapabilityConfirmation(
                confirmation_id=uuid4().hex,
                user_id=user_id,
                conversation_id=conversation_id,
                capability=capability,
                display_summary=display_summary[:500],
                arguments=arguments,
                arguments_hash=digest,
                policy_revision=policy_revision,
                status="PENDING",
                created_at=now,
                expires_at=now + timedelta(minutes=expires_in_minutes),
            )
            connection.execute(
                "INSERT INTO capability_confirmations(confirmation_id, user_id, "
                "conversation_id, capability, display_summary, arguments_json, arguments_hash, "
                "policy_revision, status, attempt_count, created_at, expires_at, "
                "lease_expires_at, settled_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, NULL, NULL)",
                (
                    confirmation.confirmation_id,
                    user_id,
                    conversation_id,
                    capability,
                    confirmation.display_summary,
                    json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str),
                    digest,
                    policy_revision,
                    "PENDING",
                    confirmation.created_at.isoformat(),
                    confirmation.expires_at.isoformat(),
                ),
            )
        return confirmation

    def get(self, confirmation_id: str) -> CapabilityConfirmation | None:
        with self._connect() as connection:
            row = connection.execute(
                self._SELECT + " WHERE confirmation_id = ?", (confirmation_id,)
            ).fetchone()
        return self._row(row) if row is not None else None

    def pending_for_conversation(
        self,
        *,
        user_id: str,
        conversation_id: str,
        policy_revision: int | None,
    ) -> tuple[CapabilityConfirmation, ...]:
        """Pending approvals still valid under the policy the caller is showing."""

        now = datetime.now(timezone.utc)
        with self._connect() as connection:
            rows = connection.execute(
                self._SELECT
                + " WHERE user_id = ? AND conversation_id = ? AND status = 'PENDING'"
                + " ORDER BY created_at",
                (user_id, conversation_id),
            ).fetchall()
        return tuple(
            confirmation
            for confirmation in (self._row(row) for row in rows)
            if confirmation.expires_at > now
            and (
                policy_revision is None
                or confirmation.policy_revision == policy_revision
            )
        )

    def active_for_conversation(
        self, *, user_id: str, conversation_id: str
    ) -> tuple[CapabilityConfirmation, ...]:
        """Rows whose interaction identity may still arrive from a client."""

        with self._connect() as connection:
            rows = connection.execute(
                self._SELECT
                + " WHERE user_id=? AND conversation_id=? "
                "AND status IN ('PENDING', 'APPLYING') ORDER BY created_at",
                (user_id, conversation_id),
            ).fetchall()
        return tuple(self._row(row) for row in rows)

    def claim(
        self,
        *,
        confirmation_id: str,
        user_id: str,
        lease_seconds: int = DEFAULT_EXECUTION_LEASE_SECONDS,
    ) -> CapabilityConfirmation:
        """Acquire the execution lease without claiming that execution finished."""

        now = datetime.now(timezone.utc)
        lease_expires_at = now + timedelta(seconds=lease_seconds)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "UPDATE capability_confirmations SET status = 'APPLYING', "
                "attempt_count = attempt_count + 1, lease_expires_at = ? "
                "WHERE confirmation_id = ? AND user_id = ? AND expires_at > ? "
                "AND (status = 'PENDING' OR "
                "     (status = 'APPLYING' AND lease_expires_at <= ?))",
                (
                    lease_expires_at.isoformat(), confirmation_id, user_id,
                    now.isoformat(), now.isoformat(),
                ),
            )
            if cursor.rowcount == 1:
                row = connection.execute(
                    self._SELECT + " WHERE confirmation_id = ?", (confirmation_id,)
                ).fetchone()
                return self._row(row)
            row = connection.execute(
                self._SELECT + " WHERE confirmation_id = ? AND user_id = ?",
                (confirmation_id, user_id),
            ).fetchone()
        if row is None:
            raise CapabilityConfirmationSettledError("no such confirmation")
        confirmation = self._row(row)
        if confirmation.status == "APPLYING":
            raise CapabilityConfirmationInProgressError(confirmation.confirmation_id)
        if confirmation.status != "PENDING":
            raise CapabilityConfirmationSettledError(confirmation.status)
        raise CapabilityConfirmationExpiredError(confirmation.confirmation_id)

    def settle(
        self,
        *,
        confirmation_id: str,
        user_id: str,
        status: Literal["EXECUTED", "FAILED", "RECONCILIATION_REQUIRED"],
    ) -> CapabilityConfirmation:
        now = datetime.now(timezone.utc)
        with self._connect() as connection:
            changed = connection.execute(
                "UPDATE capability_confirmations SET status=?, settled_at=?, "
                "lease_expires_at=NULL WHERE confirmation_id=? AND user_id=? "
                "AND status='APPLYING'",
                (status, now.isoformat(), confirmation_id, user_id),
            )
            row = connection.execute(
                self._SELECT + " WHERE confirmation_id=? AND user_id=?",
                (confirmation_id, user_id),
            ).fetchone()
        if row is None or changed.rowcount != 1:
            raise CapabilityConfirmationSettledError("confirmation is not applying")
        return self._row(row)

    def reconcile(
        self,
        *,
        confirmation_id: str,
        user_id: str,
        status: Literal["EXECUTED", "FAILED"],
    ) -> CapabilityConfirmation:
        """Record an operator/provider finding for an interrupted execution."""

        now = datetime.now(timezone.utc)
        with self._connect() as connection:
            changed = connection.execute(
                "UPDATE capability_confirmations SET status=?, settled_at=?, "
                "lease_expires_at=NULL WHERE confirmation_id=? AND user_id=? "
                "AND status IN ('APPLYING', 'RECONCILIATION_REQUIRED')",
                (status, now.isoformat(), confirmation_id, user_id),
            )
            row = connection.execute(
                self._SELECT + " WHERE confirmation_id=? AND user_id=?",
                (confirmation_id, user_id),
            ).fetchone()
        if row is None or changed.rowcount != 1:
            raise CapabilityConfirmationSettledError(
                "confirmation no longer requires reconciliation"
            )
        return self._row(row)

    def cancel(self, *, confirmation_id: str, user_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE capability_confirmations SET status = 'CANCELLED', "
                "settled_at = ? WHERE confirmation_id = ? AND user_id = ? "
                "  AND status = 'PENDING'",
                (datetime.now(timezone.utc).isoformat(), confirmation_id, user_id),
            )
            return cursor.rowcount == 1
