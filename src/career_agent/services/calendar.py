from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from career_agent.connectors.calendar import (
    CalendarConnectorError,
    CalendarReconciliationResult,
    CalendarWriteResult,
)
from career_agent.domain.calendar import (
    CalendarAccount,
    CalendarChangeProposal,
    CalendarEventLink,
    CalendarEventPayload,
    CalendarOperation,
)
from career_agent.services.applications import ApplicationService
from career_agent.services.interviews import InterviewNotFoundError, InterviewService
from career_agent.storage.calendar import (
    CalendarExecutionLeaseActiveError,
    CalendarExecutionUnresolvedError,
    SQLiteCalendarStore,
)


class CalendarAccountNotFoundError(ValueError):
    pass


class CalendarProposalNotFoundError(ValueError):
    pass


class CalendarProposalConflictError(ValueError):
    pass


class CalendarSyncNotAvailableError(ValueError):
    pass


@dataclass(frozen=True)
class CalendarExecution:
    proposal: CalendarChangeProposal
    link: CalendarEventLink


class CalendarService:
    def __init__(
        self,
        store: SQLiteCalendarStore,
        interview_service: InterviewService,
        application_service: ApplicationService,
        connector_resolver,
        *,
        proposal_ttl: timedelta = timedelta(minutes=15),
        execution_lease: timedelta = timedelta(minutes=1),
    ) -> None:
        self._store = store
        self._interview_service = interview_service
        self._application_service = application_service
        self._connector_resolver = connector_resolver
        self._proposal_ttl = proposal_ttl
        if execution_lease <= timedelta(0):
            raise ValueError("calendar execution lease must be positive")
        self._execution_lease = execution_lease

    def list_accounts(self, *, user_id: str) -> tuple[CalendarAccount, ...]:
        return self._store.list_accounts(user_id=user_id)

    def list_links(self, *, user_id: str) -> tuple[CalendarEventLink, ...]:
        return self._store.list_links(user_id=user_id)

    def get_proposal(
        self, *, user_id: str, proposal_id: str
    ) -> CalendarChangeProposal:
        proposal = self._store.get_proposal(
            user_id=user_id, proposal_id=proposal_id
        )
        if proposal is None:
            raise CalendarProposalNotFoundError(proposal_id)
        return proposal

    def prepare_interview_sync(
        self,
        *,
        user_id: str,
        interview_round_id: str,
        calendar_account_id: str | None = None,
        now: datetime | None = None,
    ) -> CalendarChangeProposal:
        current = now or datetime.now(timezone.utc)
        account = self._select_account(
            user_id=user_id, calendar_account_id=calendar_account_id
        )
        self._reconcile_before_new_proposal(
            user_id=user_id,
            interview_round_id=interview_round_id,
            account=account,
            now=current,
        )
        operation, event_id, payload = self._expected_change(
            user_id=user_id,
            interview_round_id=interview_round_id,
            account=account,
        )
        payload_hash = self._hash(
            operation=operation,
            external_event_id=event_id,
            payload=payload,
        )
        link = self._store.get_link(
            user_id=user_id,
            calendar_account_id=account.id,
            interview_round_id=interview_round_id,
        )
        if (
            operation == "update"
            and link is not None
            and link.status == "active"
            and link.last_payload_hash == payload_hash
        ):
            raise CalendarSyncNotAvailableError("interview is already synchronized")
        proposal = CalendarChangeProposal(
            id=f"calendar_proposal_{uuid4().hex}",
            user_id=user_id,
            calendar_account_id=account.id,
            interview_round_id=interview_round_id,
            operation=operation,
            external_event_id=event_id,
            payload=payload,
            payload_hash=payload_hash,
            status="pending",
            created_at=current,
            expires_at=current + self._proposal_ttl,
        )
        try:
            self._store.create_proposal(proposal)
        except CalendarExecutionUnresolvedError as error:
            raise CalendarSyncNotAvailableError(str(error)) from error
        return proposal

    def execute_proposal(
        self,
        *,
        user_id: str,
        proposal_id: str,
        now: datetime | None = None,
    ) -> CalendarExecution:
        current = now or datetime.now(timezone.utc)
        proposal = self.get_proposal(user_id=user_id, proposal_id=proposal_id)
        if proposal.status == "executed":
            link = self._store.get_link(
                user_id=user_id,
                calendar_account_id=proposal.calendar_account_id,
                interview_round_id=proposal.interview_round_id,
            )
            if link is None:
                raise CalendarProposalConflictError("executed proposal has no link")
            return CalendarExecution(proposal=proposal, link=link)
        recovering = proposal.status in {"executing", "reconciliation_required"}
        if proposal.status not in {"pending", "executing", "reconciliation_required"}:
            raise CalendarProposalConflictError(
                f"calendar proposal is {proposal.status}; prepare a new proposal"
            )
        if proposal.status == "pending" and current >= proposal.expires_at:
            self._store.set_proposal_status(
                user_id=user_id, proposal_id=proposal.id,
                status="expired", now=current,
            )
            raise CalendarProposalConflictError(
                "calendar proposal expired; prepare a new proposal"
            )
        account = self._select_account(
            user_id=user_id, calendar_account_id=proposal.calendar_account_id
        )
        if proposal.status == "pending":
            operation, event_id, payload = self._expected_change(
                user_id=user_id,
                interview_round_id=proposal.interview_round_id,
                account=account,
                proposed_event_id=proposal.external_event_id,
            )
            expected_hash = self._hash(
                operation=operation, external_event_id=event_id, payload=payload
            )
            if (
                operation != proposal.operation
                or event_id != proposal.external_event_id
                or expected_hash != proposal.payload_hash
            ):
                self._store.set_proposal_status(
                    user_id=user_id, proposal_id=proposal.id,
                    status="superseded", now=current,
                )
                raise CalendarProposalConflictError(
                    "interview changed after approval preview; prepare a new proposal"
                )
        try:
            proposal, execution, must_reconcile = self._store.claim_execution(
                proposal=proposal,
                now=current,
                lease_duration=self._execution_lease,
            )
        except CalendarExecutionLeaseActiveError as error:
            raise CalendarConnectorError(
                "CALENDAR_EXECUTION_IN_PROGRESS",
                str(error),
                outcome_unknown=True,
            ) from error

        if recovering != must_reconcile:
            raise ValueError("calendar proposal and execution ledger disagree")

        try:
            connector = self._connector_resolver.resolve(account)
        except CalendarConnectorError as error:
            if must_reconcile:
                self._store.mark_reconciliation_required(
                    proposal=proposal,
                    execution=execution,
                    now=current,
                    error_code=error.code,
                    error_detail=str(error),
                )
                raise CalendarConnectorError(
                    error.code,
                    str(error),
                    outcome_unknown=True,
                ) from error
            self._store.fail_execution(
                proposal=proposal,
                execution=execution,
                now=current,
                error_code=error.code,
                error_detail=str(error),
            )
            raise

        if must_reconcile:
            try:
                reconciled: CalendarReconciliationResult = connector.reconcile(
                    operation=proposal.operation,
                    calendar_id=account.calendar_id,
                    external_event_id=proposal.external_event_id,
                    payload_hash=proposal.payload_hash,
                    prior_payload_hash=execution.prior_payload_hash,
                )
            except CalendarConnectorError as error:
                self._store.mark_reconciliation_required(
                    proposal=proposal,
                    execution=execution,
                    now=current,
                    error_code=error.code,
                    error_detail=str(error),
                )
                raise CalendarConnectorError(
                    error.code,
                    str(error),
                    outcome_unknown=True,
                ) from error
            if reconciled.outcome == "applied":
                result = reconciled.write_result
                if result is None:
                    raise ValueError("applied reconciliation has no write result")
                executed, link = self._store.complete_execution(
                    proposal=proposal,
                    execution=execution,
                    external_etag=result.etag,
                    external_html_link=result.html_link,
                    now=current,
                )
                return CalendarExecution(proposal=executed, link=link)
            if reconciled.outcome == "conflict":
                detail = (
                    "the external event exists but does not carry the approved "
                    "calendar payload hash"
                )
                self._store.mark_reconciliation_required(
                    proposal=proposal,
                    execution=execution,
                    now=current,
                    error_code="CALENDAR_RECONCILIATION_CONFLICT",
                    error_detail=detail,
                )
                raise CalendarConnectorError(
                    "CALENDAR_RECONCILIATION_CONFLICT",
                    detail,
                    outcome_unknown=True,
                )

        try:
            result: CalendarWriteResult = connector.apply(
                operation=proposal.operation,
                calendar_id=account.calendar_id,
                external_event_id=proposal.external_event_id,
                payload=proposal.payload,
                idempotency_key=execution.idempotency_key,
                payload_hash=execution.payload_hash,
            )
        except CalendarConnectorError as error:
            if error.outcome_unknown:
                self._store.mark_reconciliation_required(
                    proposal=proposal,
                    execution=execution,
                    now=current,
                    error_code=error.code,
                    error_detail=str(error),
                )
            else:
                self._store.fail_execution(
                    proposal=proposal,
                    execution=execution,
                    now=current,
                    error_code=error.code,
                    error_detail=str(error),
                )
            raise
        except Exception as error:
            detail = f"{type(error).__name__}: {error}"
            self._store.mark_reconciliation_required(
                proposal=proposal,
                execution=execution,
                now=current,
                error_code="CALENDAR_OUTCOME_UNKNOWN",
                error_detail=detail,
            )
            raise CalendarConnectorError(
                "CALENDAR_OUTCOME_UNKNOWN", detail, outcome_unknown=True
            ) from error
        try:
            executed, link = self._store.complete_execution(
                proposal=proposal,
                execution=execution,
                external_etag=result.etag,
                external_html_link=result.html_link,
                now=current,
            )
        except Exception as error:
            # The remote write succeeded but the local transaction did not.
            # Leave the durable intent recoverable; the next claim must GET and
            # verify the external marker before completing locally.
            detail = f"{type(error).__name__}: {error}"
            try:
                self._store.mark_reconciliation_required(
                    proposal=proposal,
                    execution=execution,
                    now=current,
                    error_code="CALENDAR_LOCAL_COMMIT_FAILED",
                    error_detail=detail,
                )
            except Exception:
                pass
            raise CalendarConnectorError(
                "CALENDAR_LOCAL_COMMIT_FAILED", detail, outcome_unknown=True
            ) from error
        return CalendarExecution(proposal=executed, link=link)

    def _reconcile_before_new_proposal(
        self,
        *,
        user_id: str,
        interview_round_id: str,
        account: CalendarAccount,
        now: datetime,
    ) -> None:
        """Resolve an unknown prior write before issuing another approval.

        The existing product policy requires a fresh preview after an uncertain
        write. That is safe only after the previous durable intent has been
        reconciled: otherwise a new approval could duplicate or overwrite an
        operation whose response was merely lost.
        """

        unresolved = self._store.get_unresolved_execution(
            user_id=user_id,
            calendar_account_id=account.id,
            interview_round_id=interview_round_id,
        )
        if unresolved is None:
            return
        prior_proposal, _ = unresolved
        try:
            claimed, execution, _ = self._store.claim_execution(
                proposal=prior_proposal,
                now=now,
                lease_duration=self._execution_lease,
            )
        except CalendarExecutionLeaseActiveError as error:
            raise CalendarSyncNotAvailableError(
                "the previous calendar execution is still in progress"
            ) from error
        try:
            connector = self._connector_resolver.resolve(account)
            reconciled = connector.reconcile(
                operation=claimed.operation,
                calendar_id=account.calendar_id,
                external_event_id=claimed.external_event_id,
                payload_hash=claimed.payload_hash,
                prior_payload_hash=execution.prior_payload_hash,
            )
        except CalendarConnectorError as error:
            self._store.mark_reconciliation_required(
                proposal=claimed,
                execution=execution,
                now=now,
                error_code=error.code,
                error_detail=str(error),
            )
            raise CalendarSyncNotAvailableError(
                "the previous calendar execution could not be reconciled"
            ) from error

        if reconciled.outcome == "applied":
            result = reconciled.write_result
            if result is None:
                raise ValueError("applied reconciliation has no write result")
            self._store.complete_execution(
                proposal=claimed,
                execution=execution,
                external_etag=result.etag,
                external_html_link=result.html_link,
                now=now,
            )
            return
        if reconciled.outcome == "not_applied":
            self._store.fail_execution(
                proposal=claimed,
                execution=execution,
                now=now,
                error_code="CALENDAR_RECONCILED_NOT_APPLIED",
                error_detail=(
                    "external state confirms that the previous operation was not applied"
                ),
            )
            return

        detail = (
            "the external event matches neither the prior local payload nor the "
            "approved payload"
        )
        self._store.mark_reconciliation_required(
            proposal=claimed,
            execution=execution,
            now=now,
            error_code="CALENDAR_RECONCILIATION_CONFLICT",
            error_detail=detail,
        )
        raise CalendarSyncNotAvailableError(detail)

    def _select_account(
        self, *, user_id: str, calendar_account_id: str | None
    ) -> CalendarAccount:
        if calendar_account_id is not None:
            account = self._store.get_account(
                user_id=user_id, calendar_account_id=calendar_account_id
            )
            if account is None or account.status != "active":
                raise CalendarAccountNotFoundError(calendar_account_id)
            return account
        accounts = self._store.list_accounts(user_id=user_id)
        if len(accounts) != 1:
            raise CalendarAccountNotFoundError(
                "select one calendar account" if accounts else "no calendar account"
            )
        return accounts[0]

    def _expected_change(
        self,
        *,
        user_id: str,
        interview_round_id: str,
        account: CalendarAccount,
        proposed_event_id: str | None = None,
    ) -> tuple[CalendarOperation, str, CalendarEventPayload | None]:
        try:
            interview = self._interview_service.get_interview(
                user_id=user_id, interview_round_id=interview_round_id
            ).interview
        except InterviewNotFoundError as error:
            raise CalendarSyncNotAvailableError("interview not found") from error
        link = self._store.get_link(
            user_id=user_id,
            calendar_account_id=account.id,
            interview_round_id=interview.id,
        )
        if interview.status == "cancelled":
            if link is None or link.status != "active":
                raise CalendarSyncNotAvailableError(
                    "cancelled interview has no active calendar event"
                )
            payload = (
                self._calendar_payload(user_id=user_id, interview=interview)
                if interview.scheduled_start is not None
                else None
            )
            return "cancel", link.external_event_id, payload
        if interview.status != "scheduled" or interview.scheduled_start is None:
            raise CalendarSyncNotAvailableError(
                "only scheduled or cancelled interviews can synchronize"
            )
        payload = self._calendar_payload(user_id=user_id, interview=interview)
        if link is not None and link.status == "active":
            return "update", link.external_event_id, payload
        event_id = proposed_event_id or (
            "ca"
            + hashlib.sha256(
                f"{user_id}:{account.id}:{interview.id}".encode()
            ).hexdigest()
        )
        return "create", event_id, payload

    def _calendar_payload(self, *, user_id: str, interview) -> CalendarEventPayload:
        application = self._application_service.get_application(
            user_id=user_id, application_id=interview.application_id
        )
        posting = application.job.posting
        timezone_name = interview.timezone or "Asia/Shanghai"
        try:
            zone = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as error:
            raise CalendarSyncNotAvailableError("interview timezone is invalid") from error
        start_at = interview.scheduled_start
        if start_at.tzinfo is None:
            start_at = start_at.replace(tzinfo=zone)
        end_at = interview.scheduled_end or (start_at + timedelta(hours=1))
        if end_at.tzinfo is None:
            end_at = end_at.replace(tzinfo=zone)
        label = interview.employer_label or "面试"
        details = [
            f"岗位：{posting.company_name} · {posting.title}",
            f"形式：{interview.interview_format}",
        ]
        if interview.meeting_url:
            details.append(f"会议链接：{interview.meeting_url}")
        if interview.contact_summary:
            details.append(f"联系人：{interview.contact_summary}")
        payload = CalendarEventPayload(
            title=f"{label} · {posting.company_name} · {posting.title}",
            description="\n".join(details),
            start_at=start_at,
            end_at=end_at,
            timezone=timezone_name,
            location=interview.location,
        )
        return payload

    @staticmethod
    def _hash(
        *,
        operation: CalendarOperation,
        external_event_id: str,
        payload: CalendarEventPayload | None,
    ) -> str:
        # Operation and external ID are compared separately during execution;
        # this digest tracks the event contents across create -> update cycles.
        canonical = json.dumps(
            payload.model_dump(mode="json") if payload else None,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode()).hexdigest()
