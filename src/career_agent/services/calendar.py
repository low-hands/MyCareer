from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from career_agent.connectors.calendar import (
    CalendarConnectorError,
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
from career_agent.storage.calendar import SQLiteCalendarStore


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
    ) -> None:
        self._store = store
        self._interview_service = interview_service
        self._application_service = application_service
        self._connector_resolver = connector_resolver
        self._proposal_ttl = proposal_ttl

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
        self._store.create_proposal(proposal)
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
        if proposal.status != "pending":
            raise CalendarProposalConflictError(
                f"calendar proposal is {proposal.status}; prepare a new proposal"
            )
        if current >= proposal.expires_at:
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
            connector = self._connector_resolver.resolve(account)
            result: CalendarWriteResult = connector.apply(
                operation=proposal.operation,
                calendar_id=account.calendar_id,
                external_event_id=proposal.external_event_id,
                payload=proposal.payload,
            )
        except CalendarConnectorError as error:
            self._store.set_proposal_status(
                user_id=user_id, proposal_id=proposal.id, status="failed",
                now=current, error_code=error.code, error_detail=str(error),
            )
            raise
        executed, link = self._store.complete_execution(
            proposal=proposal,
            external_etag=result.etag,
            external_html_link=result.html_link,
            now=current,
        )
        return CalendarExecution(proposal=executed, link=link)

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
