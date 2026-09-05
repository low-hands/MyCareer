from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sqlite3
from uuid import uuid4

from career_agent.domain.calendar import (
    CalendarAccount,
    CalendarChangeEvent,
    CalendarChangeProposal,
    CalendarEventLink,
    CalendarEventPayload,
    CalendarOperationExecution,
    CalendarProposalStatus,
)
from career_agent.storage.schema import apply_schema


class CalendarExecutionLeaseActiveError(RuntimeError):
    pass


class CalendarExecutionUnresolvedError(RuntimeError):
    pass


class SQLiteCalendarStore:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            apply_schema(
                connection,
                "calendar",
                3,
                self._migrate,
                {2: self._upgrade_to_v2, 3: self._upgrade_to_v3},
            )
        os.chmod(self.path, 0o600)

    def add_account(
        self,
        *,
        user_id: str,
        email_address: str,
        calendar_id: str,
        credential_ref: str,
        now: datetime | None = None,
    ) -> CalendarAccount:
        changed_at = now or datetime.now(timezone.utc)
        with self._connect() as connection:
            row = connection.execute(
                self._ACCOUNT_SELECT
                + " WHERE user_id = ? AND provider = 'google' AND email_address = ? AND calendar_id = ?",
                (user_id, email_address, calendar_id),
            ).fetchone()
            if row is not None:
                account = self._account(row).model_copy(
                    update={
                        "credential_ref": credential_ref,
                        "status": "active",
                        "updated_at": changed_at,
                    }
                )
                connection.execute(
                    "UPDATE calendar_accounts SET credential_ref = ?, status = 'active', updated_at = ? WHERE id = ?",
                    (credential_ref, changed_at.isoformat(), account.id),
                )
                return account
            account = CalendarAccount(
                id=f"calendar_account_{uuid4().hex}",
                user_id=user_id,
                provider="google",
                email_address=email_address,
                calendar_id=calendar_id,
                credential_ref=credential_ref,
                created_at=changed_at,
                updated_at=changed_at,
            )
            connection.execute(
                """
                INSERT INTO calendar_accounts(
                    id, user_id, provider, email_address, calendar_id,
                    credential_ref, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    account.id, account.user_id, account.provider,
                    account.email_address, account.calendar_id,
                    account.credential_ref, account.status,
                    account.created_at.isoformat(), account.updated_at.isoformat(),
                ),
            )
        return account

    def get_account(
        self, *, user_id: str, calendar_account_id: str
    ) -> CalendarAccount | None:
        with self._connect() as connection:
            row = connection.execute(
                self._ACCOUNT_SELECT + " WHERE id = ? AND user_id = ?",
                (calendar_account_id, user_id),
            ).fetchone()
        return self._account(row) if row else None

    def list_accounts(self, *, user_id: str) -> tuple[CalendarAccount, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                self._ACCOUNT_SELECT
                + " WHERE user_id = ? AND status = 'active' ORDER BY created_at",
                (user_id,),
            ).fetchall()
        return tuple(self._account(row) for row in rows)

    def get_link(
        self, *, user_id: str, calendar_account_id: str, interview_round_id: str
    ) -> CalendarEventLink | None:
        with self._connect() as connection:
            row = connection.execute(
                self._LINK_SELECT
                + " WHERE user_id = ? AND calendar_account_id = ? AND interview_round_id = ?",
                (user_id, calendar_account_id, interview_round_id),
            ).fetchone()
        return self._link(row) if row else None

    def list_links(self, *, user_id: str) -> tuple[CalendarEventLink, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                self._LINK_SELECT + " WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        return tuple(self._link(row) for row in rows)

    def create_proposal(self, proposal: CalendarChangeProposal) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            unresolved = connection.execute(
                self._PROPOSAL_SELECT
                + " WHERE user_id = ? AND calendar_account_id = ?"
                " AND interview_round_id = ?"
                " AND status IN ('executing', 'reconciliation_required')",
                (
                    proposal.user_id,
                    proposal.calendar_account_id,
                    proposal.interview_round_id,
                ),
            ).fetchone()
            if unresolved is not None:
                raise CalendarExecutionUnresolvedError(
                    "reconcile the existing calendar execution before preparing another proposal"
                )
            rows = connection.execute(
                self._PROPOSAL_SELECT
                + " WHERE user_id = ? AND calendar_account_id = ? AND interview_round_id = ? AND status = 'pending'",
                (
                    proposal.user_id, proposal.calendar_account_id,
                    proposal.interview_round_id,
                ),
            ).fetchall()
            for row in rows:
                old = self._proposal(row)
                connection.execute(
                    "UPDATE calendar_change_proposals SET status = 'superseded' WHERE id = ?",
                    (old.id,),
                )
                self._insert_event(
                    connection, old, "superseded", proposal.created_at,
                    "Replaced by a newer fixed-payload proposal.",
                )
            connection.execute(
                """
                INSERT INTO calendar_change_proposals(
                    id, user_id, calendar_account_id, interview_round_id,
                    operation, external_event_id, payload_json, payload_hash,
                    policy_epoch, status, created_at, expires_at, executed_at,
                    error_code, error_detail
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._proposal_values(proposal),
            )
            self._insert_event(
                connection, proposal, "proposed", proposal.created_at, None
            )

    def get_proposal(
        self, *, user_id: str, proposal_id: str
    ) -> CalendarChangeProposal | None:
        with self._connect() as connection:
            row = connection.execute(
                self._PROPOSAL_SELECT + " WHERE id = ? AND user_id = ?",
                (proposal_id, user_id),
            ).fetchone()
        return self._proposal(row) if row else None

    def set_proposal_status(
        self,
        *,
        user_id: str,
        proposal_id: str,
        status: CalendarProposalStatus,
        now: datetime,
        error_code: str | None = None,
        error_detail: str | None = None,
    ) -> CalendarChangeProposal | None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                self._PROPOSAL_SELECT + " WHERE id = ? AND user_id = ?",
                (proposal_id, user_id),
            ).fetchone()
            if row is None:
                return None
            proposal = self._proposal(row)
            executed_at = now if status == "executed" else None
            updated = CalendarChangeProposal.model_validate(
                proposal.model_copy(
                    update={
                        "status": status,
                        "executed_at": executed_at,
                        "error_code": error_code,
                        "error_detail": error_detail,
                    }
                ).model_dump()
            )
            connection.execute(
                """
                UPDATE calendar_change_proposals
                SET status = ?, executed_at = ?, error_code = ?, error_detail = ?
                WHERE id = ? AND user_id = ?
                """,
                (
                    updated.status, self._iso(updated.executed_at),
                    updated.error_code, updated.error_detail,
                    updated.id, updated.user_id,
                ),
            )
            self._insert_event(
                connection, updated, status, now, error_detail
            )
        return updated

    def get_execution(
        self, *, user_id: str, proposal_id: str
    ) -> CalendarOperationExecution | None:
        with self._connect() as connection:
            row = connection.execute(
                self._EXECUTION_SELECT + " WHERE user_id = ? AND proposal_id = ?",
                (user_id, proposal_id),
            ).fetchone()
        return self._execution(row) if row else None

    def get_unresolved_execution(
        self,
        *,
        user_id: str,
        calendar_account_id: str,
        interview_round_id: str,
    ) -> tuple[CalendarChangeProposal, CalendarOperationExecution] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT p.id
                FROM calendar_change_proposals AS p
                JOIN calendar_operation_executions AS e
                  ON e.proposal_id = p.id AND e.user_id = p.user_id
                WHERE p.user_id = ? AND p.calendar_account_id = ?
                  AND p.interview_round_id = ?
                  AND p.status IN ('executing', 'reconciliation_required')
                  AND e.status IN ('applying', 'reconciliation_required')
                ORDER BY e.updated_at DESC
                LIMIT 1
                """,
                (user_id, calendar_account_id, interview_round_id),
            ).fetchone()
            if row is None:
                return None
            proposal_row = connection.execute(
                self._PROPOSAL_SELECT + " WHERE id = ? AND user_id = ?",
                (row[0], user_id),
            ).fetchone()
            execution_row = connection.execute(
                self._EXECUTION_SELECT + " WHERE proposal_id = ? AND user_id = ?",
                (row[0], user_id),
            ).fetchone()
        if proposal_row is None or execution_row is None:
            raise ValueError("unresolved calendar execution is incomplete")
        return self._proposal(proposal_row), self._execution(execution_row)

    def claim_execution(
        self,
        *,
        proposal: CalendarChangeProposal,
        now: datetime,
        lease_duration: timedelta,
    ) -> tuple[CalendarChangeProposal, CalendarOperationExecution, bool]:
        """Persist intent before I/O and exclusively claim one external attempt.

        The boolean is true when a prior attempt exists, which means the caller
        must reconcile external state before it considers applying again.
        """

        lease_expires_at = now + lease_duration
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                self._PROPOSAL_SELECT + " WHERE id = ? AND user_id = ?",
                (proposal.id, proposal.user_id),
            ).fetchone()
            current = self._proposal(row) if row else None
            if current is None:
                raise ValueError("calendar proposal disappeared before execution")
            execution_row = connection.execute(
                self._EXECUTION_SELECT + " WHERE user_id = ? AND proposal_id = ?",
                (proposal.user_id, proposal.id),
            ).fetchone()
            execution = self._execution(execution_row) if execution_row else None

            if current.status == "pending":
                if execution is not None:
                    raise ValueError("pending calendar proposal already has an execution")
                existing_link_row = connection.execute(
                    self._LINK_SELECT
                    + " WHERE user_id = ? AND calendar_account_id = ?"
                    " AND interview_round_id = ?",
                    (
                        proposal.user_id,
                        proposal.calendar_account_id,
                        proposal.interview_round_id,
                    ),
                ).fetchone()
                prior_payload_hash = (
                    self._link(existing_link_row).last_payload_hash
                    if existing_link_row is not None
                    else None
                )
                execution = CalendarOperationExecution(
                    id=f"calendar_execution_{uuid4().hex}",
                    user_id=proposal.user_id,
                    proposal_id=proposal.id,
                    operation=proposal.operation,
                    external_event_id=proposal.external_event_id,
                    payload_hash=proposal.payload_hash,
                    prior_payload_hash=prior_payload_hash,
                    idempotency_key=proposal.id,
                    status="applying",
                    attempt_count=1,
                    started_at=now,
                    updated_at=now,
                    lease_expires_at=lease_expires_at,
                )
                connection.execute(
                    """
                    INSERT INTO calendar_operation_executions(
                        id, user_id, proposal_id, operation, external_event_id,
                        payload_hash, prior_payload_hash, idempotency_key, status, attempt_count,
                        started_at, updated_at, lease_expires_at, completed_at,
                        error_code, error_detail
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    self._execution_values(execution),
                )
                claimed = current.model_copy(update={"status": "executing"})
                connection.execute(
                    "UPDATE calendar_change_proposals SET status = 'executing'"
                    " WHERE id = ? AND user_id = ?",
                    (proposal.id, proposal.user_id),
                )
                self._insert_event(
                    connection, claimed, "execution_started", now,
                    f"execution_id={execution.id}; attempt=1",
                )
                return claimed, execution, False

            if current.status not in {"executing", "reconciliation_required"}:
                raise ValueError(f"calendar proposal is {current.status}")
            if execution is None:
                raise ValueError("calendar proposal has no execution intent")
            if execution.status == "applying" and now < execution.lease_expires_at:
                raise CalendarExecutionLeaseActiveError(
                    "calendar execution is still in progress"
                )
            if execution.status not in {"applying", "reconciliation_required"}:
                raise ValueError(f"calendar execution is {execution.status}")
            execution = execution.model_copy(
                update={
                    "status": "applying",
                    "attempt_count": execution.attempt_count + 1,
                    "updated_at": now,
                    "lease_expires_at": lease_expires_at,
                    "error_code": None,
                    "error_detail": None,
                }
            )
            connection.execute(
                """
                UPDATE calendar_operation_executions
                SET status = 'applying', attempt_count = ?, updated_at = ?,
                    lease_expires_at = ?, error_code = NULL, error_detail = NULL
                WHERE id = ? AND user_id = ?
                """,
                (
                    execution.attempt_count,
                    now.isoformat(),
                    lease_expires_at.isoformat(),
                    execution.id,
                    execution.user_id,
                ),
            )
            claimed = current.model_copy(
                update={
                    "status": "executing",
                    "error_code": None,
                    "error_detail": None,
                }
            )
            connection.execute(
                """
                UPDATE calendar_change_proposals
                SET status = 'executing', error_code = NULL, error_detail = NULL
                WHERE id = ? AND user_id = ?
                """,
                (proposal.id, proposal.user_id),
            )
            self._insert_event(
                connection, claimed, "execution_started", now,
                f"execution_id={execution.id}; attempt={execution.attempt_count}; reconcile_first",
            )
            return claimed, execution, True

    def mark_reconciliation_required(
        self,
        *,
        proposal: CalendarChangeProposal,
        execution: CalendarOperationExecution,
        now: datetime,
        error_code: str,
        error_detail: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            changed = connection.execute(
                """
                UPDATE calendar_operation_executions
                SET status = 'reconciliation_required', updated_at = ?,
                    error_code = ?, error_detail = ?
                WHERE id = ? AND user_id = ? AND status = 'applying'
                    AND attempt_count = ?
                """,
                (
                    now.isoformat(), error_code, error_detail, execution.id,
                    execution.user_id, execution.attempt_count,
                ),
            )
            if changed.rowcount != 1:
                return
            connection.execute(
                """
                UPDATE calendar_change_proposals
                SET status = 'reconciliation_required', error_code = ?, error_detail = ?
                WHERE id = ? AND user_id = ?
                """,
                (error_code, error_detail, proposal.id, proposal.user_id),
            )
            updated = proposal.model_copy(
                update={
                    "status": "reconciliation_required",
                    "error_code": error_code,
                    "error_detail": error_detail,
                }
            )
            self._insert_event(
                connection, updated, "reconciliation_required", now, error_detail
            )

    def fail_execution(
        self,
        *,
        proposal: CalendarChangeProposal,
        execution: CalendarOperationExecution,
        now: datetime,
        error_code: str,
        error_detail: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            changed = connection.execute(
                """
                UPDATE calendar_operation_executions
                SET status = 'failed', updated_at = ?, error_code = ?, error_detail = ?
                WHERE id = ? AND user_id = ? AND status = 'applying'
                    AND attempt_count = ?
                """,
                (
                    now.isoformat(), error_code, error_detail, execution.id,
                    execution.user_id, execution.attempt_count,
                ),
            )
            if changed.rowcount != 1:
                return
            connection.execute(
                """
                UPDATE calendar_change_proposals
                SET status = 'failed', error_code = ?, error_detail = ?
                WHERE id = ? AND user_id = ?
                """,
                (error_code, error_detail, proposal.id, proposal.user_id),
            )
            failed = proposal.model_copy(
                update={
                    "status": "failed",
                    "error_code": error_code,
                    "error_detail": error_detail,
                }
            )
            self._insert_event(connection, failed, "failed", now, error_detail)

    def complete_execution(
        self,
        *,
        proposal: CalendarChangeProposal,
        execution: CalendarOperationExecution,
        external_etag: str | None,
        external_html_link: str | None,
        now: datetime,
    ) -> tuple[CalendarChangeProposal, CalendarEventLink]:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                self._PROPOSAL_SELECT + " WHERE id = ? AND user_id = ?",
                (proposal.id, proposal.user_id),
            ).fetchone()
            current = self._proposal(row) if row else None
            if current is None or current.status not in {
                "executing", "reconciliation_required"
            }:
                raise ValueError("calendar proposal is no longer executing")
            execution_row = connection.execute(
                self._EXECUTION_SELECT + " WHERE id = ? AND user_id = ?",
                (execution.id, execution.user_id),
            ).fetchone()
            current_execution = self._execution(execution_row) if execution_row else None
            if (
                current_execution is None
                or current_execution.status != "applying"
                or current_execution.attempt_count != execution.attempt_count
            ):
                raise ValueError("calendar execution lease is no longer owned")
            existing_row = connection.execute(
                self._LINK_SELECT
                + " WHERE user_id = ? AND calendar_account_id = ? AND interview_round_id = ?",
                (
                    proposal.user_id, proposal.calendar_account_id,
                    proposal.interview_round_id,
                ),
            ).fetchone()
            existing = self._link(existing_row) if existing_row else None
            link = CalendarEventLink(
                id=existing.id if existing else f"calendar_link_{uuid4().hex}",
                user_id=proposal.user_id,
                calendar_account_id=proposal.calendar_account_id,
                interview_round_id=proposal.interview_round_id,
                external_event_id=proposal.external_event_id,
                external_etag=external_etag,
                external_html_link=external_html_link,
                status="cancelled" if proposal.operation == "cancel" else "active",
                last_payload_hash=proposal.payload_hash,
                created_at=existing.created_at if existing else now,
                updated_at=now,
            )
            connection.execute(
                """
                INSERT INTO calendar_event_links(
                    id, user_id, calendar_account_id, interview_round_id,
                    external_event_id, external_etag, external_html_link,
                    status, last_payload_hash, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, calendar_account_id, interview_round_id)
                DO UPDATE SET external_event_id = excluded.external_event_id,
                    external_etag = excluded.external_etag,
                    external_html_link = excluded.external_html_link,
                    status = excluded.status,
                    last_payload_hash = excluded.last_payload_hash,
                    updated_at = excluded.updated_at
                """,
                self._link_values(link),
            )
            executed = proposal.model_copy(
                update={
                    "status": "executed",
                    "executed_at": now,
                    "error_code": None,
                    "error_detail": None,
                }
            )
            connection.execute(
                """
                UPDATE calendar_change_proposals
                SET status = 'executed', executed_at = ?,
                    error_code = NULL, error_detail = NULL
                WHERE id = ?
                """,
                (now.isoformat(), proposal.id),
            )
            connection.execute(
                """
                UPDATE calendar_operation_executions
                SET status = 'succeeded', updated_at = ?, completed_at = ?,
                    error_code = NULL, error_detail = NULL
                WHERE id = ? AND user_id = ?
                """,
                (now.isoformat(), now.isoformat(), execution.id, execution.user_id),
            )
            self._insert_event(connection, executed, "executed", now, None)
        return executed, link

    def list_events(
        self, *, user_id: str, proposal_id: str
    ) -> tuple[CalendarChangeEvent, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, user_id, proposal_id, interview_round_id,
                       event_type, occurred_at, detail
                FROM calendar_change_events
                WHERE user_id = ? AND proposal_id = ?
                ORDER BY occurred_at, rowid
                """,
                (user_id, proposal_id),
            ).fetchall()
        return tuple(self._event(row) for row in rows)

    _ACCOUNT_SELECT = (
        "SELECT id, user_id, provider, email_address, calendar_id, "
        "credential_ref, status, created_at, updated_at FROM calendar_accounts"
    )
    _PROPOSAL_SELECT = (
        "SELECT id, user_id, calendar_account_id, interview_round_id, "
        "operation, external_event_id, payload_json, payload_hash, policy_epoch, "
        "status, created_at, expires_at, executed_at, error_code, error_detail "
        "FROM calendar_change_proposals"
    )
    _LINK_SELECT = (
        "SELECT id, user_id, calendar_account_id, interview_round_id, "
        "external_event_id, external_etag, external_html_link, status, "
        "last_payload_hash, created_at, updated_at FROM calendar_event_links"
    )
    _EXECUTION_SELECT = (
        "SELECT id, user_id, proposal_id, operation, external_event_id, "
        "payload_hash, prior_payload_hash, idempotency_key, status, attempt_count, started_at, "
        "updated_at, lease_expires_at, completed_at, error_code, error_detail "
        "FROM calendar_operation_executions"
    )

    def _migrate(self, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS calendar_accounts (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL, provider TEXT NOT NULL,
                email_address TEXT NOT NULL, calendar_id TEXT NOT NULL,
                credential_ref TEXT NOT NULL, status TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                UNIQUE(user_id, provider, email_address, calendar_id)
            );
            CREATE TABLE IF NOT EXISTS calendar_change_proposals (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                calendar_account_id TEXT NOT NULL REFERENCES calendar_accounts(id),
                interview_round_id TEXT NOT NULL, operation TEXT NOT NULL,
                external_event_id TEXT NOT NULL, payload_json TEXT,
                payload_hash TEXT NOT NULL, policy_epoch INTEGER NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL, expires_at TEXT NOT NULL,
                executed_at TEXT, error_code TEXT, error_detail TEXT
            );
            CREATE TABLE IF NOT EXISTS calendar_event_links (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                calendar_account_id TEXT NOT NULL REFERENCES calendar_accounts(id),
                interview_round_id TEXT NOT NULL, external_event_id TEXT NOT NULL,
                external_etag TEXT, external_html_link TEXT, status TEXT NOT NULL,
                last_payload_hash TEXT NOT NULL, created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, calendar_account_id, interview_round_id)
            );
            CREATE TABLE IF NOT EXISTS calendar_change_events (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                proposal_id TEXT NOT NULL REFERENCES calendar_change_proposals(id),
                interview_round_id TEXT NOT NULL, event_type TEXT NOT NULL,
                occurred_at TEXT NOT NULL, detail TEXT
            );
            CREATE TABLE IF NOT EXISTS calendar_operation_executions (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                proposal_id TEXT NOT NULL UNIQUE REFERENCES calendar_change_proposals(id),
                operation TEXT NOT NULL, external_event_id TEXT NOT NULL,
                payload_hash TEXT NOT NULL, prior_payload_hash TEXT,
                idempotency_key TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL, attempt_count INTEGER NOT NULL,
                started_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                lease_expires_at TEXT NOT NULL, completed_at TEXT,
                error_code TEXT, error_detail TEXT
            );
            CREATE INDEX IF NOT EXISTS calendar_proposals_user_status_idx
                ON calendar_change_proposals(user_id, status, created_at);
            CREATE INDEX IF NOT EXISTS calendar_links_user_interview_idx
                ON calendar_event_links(user_id, interview_round_id);
            CREATE INDEX IF NOT EXISTS calendar_executions_user_status_idx
                ON calendar_operation_executions(user_id, status, updated_at);
            """
        )

    @staticmethod
    def _upgrade_to_v2(connection: sqlite3.Connection) -> None:
        # The cumulative baseline creates the ledger table. This upgrade is a
        # deliberate compatibility boundary for files already registered as v1.
        connection.execute(
            "CREATE INDEX IF NOT EXISTS calendar_executions_user_status_idx "
            "ON calendar_operation_executions(user_id, status, updated_at)"
        )

    @staticmethod
    def _upgrade_to_v3(connection: sqlite3.Connection) -> None:
        # Proposals created before policy versioning belong to the original
        # execution policy. A future policy bump can then reject them without
        # guessing from timestamps or payload shape.
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(calendar_change_proposals)"
            )
        }
        if "policy_epoch" not in columns:
            connection.execute(
                "ALTER TABLE calendar_change_proposals "
                "ADD COLUMN policy_epoch INTEGER NOT NULL DEFAULT 1"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @staticmethod
    def _account(row) -> CalendarAccount:
        return CalendarAccount(
            id=row[0], user_id=row[1], provider=row[2], email_address=row[3],
            calendar_id=row[4], credential_ref=row[5], status=row[6],
            created_at=row[7], updated_at=row[8],
        )

    @staticmethod
    def _proposal(row) -> CalendarChangeProposal:
        return CalendarChangeProposal(
            id=row[0], user_id=row[1], calendar_account_id=row[2],
            interview_round_id=row[3], operation=row[4], external_event_id=row[5],
            payload=(
                CalendarEventPayload.model_validate_json(row[6])
                if row[6] is not None
                else None
            ),
            payload_hash=row[7], policy_epoch=row[8], status=row[9],
            created_at=row[10], expires_at=row[11], executed_at=row[12],
            error_code=row[13], error_detail=row[14],
        )

    @staticmethod
    def _link(row) -> CalendarEventLink:
        return CalendarEventLink(
            id=row[0], user_id=row[1], calendar_account_id=row[2],
            interview_round_id=row[3], external_event_id=row[4],
            external_etag=row[5], external_html_link=row[6], status=row[7],
            last_payload_hash=row[8], created_at=row[9], updated_at=row[10],
        )

    @staticmethod
    def _event(row) -> CalendarChangeEvent:
        return CalendarChangeEvent(
            id=row[0], user_id=row[1], proposal_id=row[2],
            interview_round_id=row[3], event_type=row[4], occurred_at=row[5],
            detail=row[6],
        )

    @staticmethod
    def _execution(row) -> CalendarOperationExecution:
        return CalendarOperationExecution(
            id=row[0], user_id=row[1], proposal_id=row[2], operation=row[3],
            external_event_id=row[4], payload_hash=row[5], prior_payload_hash=row[6],
            idempotency_key=row[7], status=row[8], attempt_count=row[9],
            started_at=row[10], updated_at=row[11], lease_expires_at=row[12],
            completed_at=row[13], error_code=row[14], error_detail=row[15],
        )

    @staticmethod
    def _proposal_values(proposal: CalendarChangeProposal) -> tuple[object, ...]:
        return (
            proposal.id, proposal.user_id, proposal.calendar_account_id,
            proposal.interview_round_id, proposal.operation,
            proposal.external_event_id,
            (
                proposal.payload.model_dump_json()
                if proposal.payload is not None
                else None
            ), proposal.payload_hash, proposal.policy_epoch,
            proposal.status, proposal.created_at.isoformat(),
            proposal.expires_at.isoformat(),
            SQLiteCalendarStore._iso(proposal.executed_at), proposal.error_code,
            proposal.error_detail,
        )

    @staticmethod
    def _link_values(link: CalendarEventLink) -> tuple[object, ...]:
        return (
            link.id, link.user_id, link.calendar_account_id,
            link.interview_round_id, link.external_event_id,
            link.external_etag, link.external_html_link, link.status,
            link.last_payload_hash, link.created_at.isoformat(),
            link.updated_at.isoformat(),
        )

    @staticmethod
    def _execution_values(execution: CalendarOperationExecution) -> tuple[object, ...]:
        return (
            execution.id, execution.user_id, execution.proposal_id,
            execution.operation, execution.external_event_id,
            execution.payload_hash, execution.prior_payload_hash,
            execution.idempotency_key, execution.status,
            execution.attempt_count, execution.started_at.isoformat(),
            execution.updated_at.isoformat(), execution.lease_expires_at.isoformat(),
            SQLiteCalendarStore._iso(execution.completed_at), execution.error_code,
            execution.error_detail,
        )

    @staticmethod
    def _insert_event(connection, proposal, event_type, occurred_at, detail) -> None:
        connection.execute(
            """
            INSERT INTO calendar_change_events(
                id, user_id, proposal_id, interview_round_id,
                event_type, occurred_at, detail
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"calendar_change_event_{uuid4().hex}", proposal.user_id,
                proposal.id, proposal.interview_round_id, event_type,
                occurred_at.isoformat(), detail,
            ),
        )

    @staticmethod
    def _iso(value: datetime | None) -> str | None:
        return value.isoformat() if value is not None else None
