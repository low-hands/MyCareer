from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
from typing import Protocol

from career_agent.connectors.email_readonly import EmailConnectorResolver
from career_agent.connectors.gmail_readonly import GmailAPIError
from career_agent.domain.email_tracking import (
    ApplicationEmailCandidate,
    EmailAssessment,
    EmailEvent,
    EmailEventStatus,
    EmailSyncCursor,
    RemoteEmailContent,
    RemoteEmailMetadata,
)
from career_agent.domain.interviews import InterviewDetails
from career_agent.services.applications import (
    ApplicationInputNotFoundError,
    ApplicationService,
    ConcurrentApplicationUpdateError,
    InvalidApplicationTransitionError,
)
from career_agent.security.redaction import redact_text
from career_agent.storage.email_tracking import SQLiteEmailTrackingStore
from career_agent.services.interviews import (
    AmbiguousInterviewMatchError,
    InterviewApplicationConflictError,
    InterviewNotFoundError,
    InterviewService,
)


class EmailTrackingWorker(Protocol):
    classifier: str
    # Whether this worker's confidence is calibrated well enough to authorize a
    # durable application write without user approval. Heuristic workers must
    # leave this False: their scores are not comparable across workers.
    authorizes_auto_apply: bool

    def assess(
        self,
        *,
        metadata: RemoteEmailMetadata,
        content: RemoteEmailContent,
        applications: tuple[ApplicationEmailCandidate, ...],
    ) -> EmailAssessment: ...


class EmailAccountNotFoundError(ValueError):
    pass


class EmailEventNotFoundError(ValueError):
    pass


class EmailEventResolutionError(ValueError):
    pass


@dataclass(frozen=True)
class EmailSyncResult:
    accounts_synced: int
    messages_seen: int
    candidate_messages: int
    events_created: tuple[EmailEvent, ...]


class DeterministicEmailTrackingWorker:
    """Conservative baseline; an LLM worker can replace it behind the same contract."""

    classifier = "deterministic_email_tracking_v1"
    # Substring hits cannot justify silently rewriting an application status or
    # creating an interview round; every event it produces awaits confirmation.
    authorizes_auto_apply = False

    _EVENT_PATTERNS = (
        ("rejection", ("unfortunately", "regret to inform", "not move forward", "不合适", "未通过", "很遗憾")),
        ("offer", ("offer letter", "正式录用", "录用通知", "薪资方案")),
        ("interview_invitation", ("interview invitation", "schedule an interview", "面试邀请", "面试安排", "面试时间")),
        ("material_request", ("additional information", "provide documents", "补充材料", "提交材料", "笔试")),
        ("acknowledgement", ("application received", "application submitted", "收到您的简历", "投递成功", "申请已收到")),
    )

    def assess(
        self,
        *,
        metadata: RemoteEmailMetadata,
        content: RemoteEmailContent,
        applications: tuple[ApplicationEmailCandidate, ...],
    ) -> EmailAssessment:
        haystack = f"{metadata.sender}\n{metadata.subject}\n{content.text}".casefold()
        event_type = "unclear"
        for candidate_type, patterns in self._EVENT_PATTERNS:
            if any(pattern in haystack for pattern in patterns):
                event_type = candidate_type
                break
        scored = []
        for application in applications:
            company = application.company_name.casefold()
            title = application.job_title.casefold()
            score = int(bool(company and company in haystack)) * 2
            score += int(bool(title and title in haystack))
            if score:
                scored.append((score, application.application_id))
        scored.sort(reverse=True)
        application_id = None
        unique_match = False
        if scored:
            unique_match = len(scored) == 1 or scored[0][0] > scored[1][0]
            if unique_match:
                application_id = scored[0][1]
        confidence = 0.25
        if event_type != "unclear":
            confidence = 0.72
        if event_type != "unclear" and application_id is not None:
            confidence = 0.95 if scored[0][0] >= 2 else 0.88
        summary = (
            f"邮件被识别为 {event_type}；"
            + ("已唯一匹配投递记录。" if unique_match else "尚未唯一匹配投递记录。")
        )
        return EmailAssessment(
            event_type=event_type,
            application_id=application_id,
            confidence=confidence,
            summary=summary,
            interview_details=(
                InterviewDetails(change_type="invited")
                if event_type == "interview_invitation"
                else None
            ),
        )


class EmailTrackingService:
    _CANDIDATE_TERMS = (
        "application", "interview", "offer", "recruit", "candidate",
        "投递", "申请", "招聘", "面试", "笔试", "录用", "简历", "候选人",
    )

    def __init__(
        self,
        store: SQLiteEmailTrackingStore,
        application_service: ApplicationService,
        connector_resolver: EmailConnectorResolver,
        worker: EmailTrackingWorker | None = None,
        interview_service: InterviewService | None = None,
        *,
        auto_apply_confidence: float = 0.9,
        initial_sync_days: int = 90,
    ) -> None:
        self._store = store
        self._application_service = application_service
        self._connector_resolver = connector_resolver
        self._worker = worker or DeterministicEmailTrackingWorker()
        self._interview_service = interview_service
        self._auto_apply_confidence = auto_apply_confidence
        self._initial_sync_days = initial_sync_days

    def sync(
        self,
        *,
        user_id: str,
        account_id: str | None = None,
    ) -> EmailSyncResult:
        accounts = (
            (self._require_account(user_id=user_id, account_id=account_id),)
            if account_id
            else tuple(
                account
                for account in self._store.list_accounts(user_id=user_id)
                if account.status == "active"
            )
        )
        if not accounts:
            raise EmailAccountNotFoundError("no active email accounts")
        candidates = self._application_candidates(user_id=user_id)
        created_events: list[EmailEvent] = []
        seen_count = 0
        candidate_count = 0
        for account in accounts:
            connector = self._connector_resolver.resolve(
                provider=account.provider,
                email_address=account.email_address,
                credential_ref=account.credential_ref,
            )
            cursor = self._store.get_cursor(account_id=account.id)
            since = datetime.now(timezone.utc) - timedelta(days=self._initial_sync_days)
            try:
                batch = connector.sync_metadata(cursor=cursor, since=since)
            except GmailAPIError as error:
                if cursor is None or error.status_code != 404:
                    raise
                batch = connector.sync_metadata(cursor=None, since=since)
            seen_count += len(batch.messages)
            for metadata in batch.messages:
                # Match on the original, persist the scrubbed copy. A subject line
                # carries one-time codes ("您的验证码是 …") and magic links just as
                # often as the body does, and it reaches both the store and the
                # model. Scrubbing before the match would risk changing which
                # mails are recognised as candidates.
                is_candidate = self._is_candidate(metadata, candidates)
                metadata = self._scrub_metadata(metadata)
                message, inserted = self._store.save_message(
                    user_id=user_id,
                    account=account,
                    metadata=metadata,
                    candidate=is_candidate,
                )
                if not is_candidate:
                    continue
                candidate_count += 1
                if not inserted and self._store.get_event_for_message(
                    user_id=user_id, email_message_id=message.id
                ) is not None:
                    continue
                content = self._scrub(
                    connector.get_content(
                        external_message_id=metadata.external_message_id
                    )
                )
                content_sha256 = hashlib.sha256(content.text.encode("utf-8")).hexdigest()
                # Persist only the hash. Raw body remains inside this worker call.
                assessment = self._worker.assess(
                    metadata=metadata,
                    content=content,
                    applications=candidates,
                )
                message = self._store.set_message_assessment(
                    user_id=user_id,
                    message_id=message.id,
                    content_sha256=content_sha256,
                    application_id=assessment.application_id,
                    classification=assessment.event_type,
                )
                event, event_inserted = self._store.create_event(
                    user_id=user_id,
                    email_message_id=message.id,
                    assessment=assessment,
                    status="pending_confirmation",
                    occurred_at=metadata.received_at,
                    classifier=self._worker.classifier,
                )
                if event_inserted and self._can_auto_apply(assessment):
                    resolved = self._try_apply_event(
                        user_id=user_id,
                        event=event,
                        source_thread_id=metadata.external_thread_id,
                    )
                    if resolved is not None:
                        event = resolved
                if event_inserted:
                    created_events.append(event)
            self._store.save_cursor(
                EmailSyncCursor(
                    account_id=account.id,
                    cursor_type=("gmail_history_id" if account.provider == "gmail" else "imap_uid"),
                    value=batch.next_cursor_value,
                    uid_validity=batch.uid_validity,
                    updated_at=datetime.now(timezone.utc),
                )
            )
        return EmailSyncResult(
            accounts_synced=len(accounts),
            messages_seen=seen_count,
            candidate_messages=candidate_count,
            events_created=tuple(created_events),
        )

    def list_events(
        self,
        *,
        user_id: str,
        status: EmailEventStatus | None = None,
        limit: int = 20,
    ) -> tuple[EmailEvent, ...]:
        return self._store.list_events(user_id=user_id, status=status, limit=limit)

    def resolve_event(
        self,
        *,
        user_id: str,
        event_id: str,
        approve: bool,
        application_id: str | None = None,
        interview_round_id: str | None = None,
    ) -> EmailEvent:
        event = self._store.get_event(user_id=user_id, event_id=event_id)
        if event is None:
            raise EmailEventNotFoundError(event_id)
        if event.status != "pending_confirmation":
            return event
        if not approve:
            resolved = self._store.resolve_event(
                user_id=user_id,
                event_id=event.id,
                status="dismissed",
                application_id=application_id or event.application_id,
            )
            if resolved is None:
                raise EmailEventResolutionError("event changed before dismissal")
            return resolved
        target_application_id = application_id or event.application_id
        if target_application_id is None:
            raise EmailEventResolutionError("approval requires an application_id")
        event = event.model_copy(update={"application_id": target_application_id})
        message = self._store.get_message(
            user_id=user_id,
            email_message_id=event.email_message_id,
        )
        resolved = self._try_apply_event(
            user_id=user_id,
            event=event,
            source_thread_id=(message.external_thread_id if message else None),
            interview_round_id=interview_round_id,
        )
        if resolved is None:
            raise EmailEventResolutionError("email event conflicts with application state")
        return resolved

    def _try_apply_event(
        self,
        *,
        user_id: str,
        event: EmailEvent,
        source_thread_id: str | None = None,
        interview_round_id: str | None = None,
    ) -> EmailEvent | None:
        if event.application_id is None or event.event_type == "unclear":
            return None
        try:
            if (
                event.event_type == "interview_invitation"
                and self._interview_service is not None
            ):
                self._interview_service.record_email_event(
                    user_id=user_id,
                    application_id=event.application_id,
                    email_event_id=event.id,
                    source_thread_id=source_thread_id,
                    details=event.interview_details or InterviewDetails(),
                    occurred_at=event.occurred_at,
                    interview_round_id=interview_round_id,
                )
            self._application_service.apply_email_event(
                user_id=user_id,
                application_id=event.application_id,
                event_type=event.event_type,
                note=event.summary,
            )
        except (
            ApplicationInputNotFoundError,
            InvalidApplicationTransitionError,
            ConcurrentApplicationUpdateError,
            AmbiguousInterviewMatchError,
            InterviewApplicationConflictError,
            InterviewNotFoundError,
        ):
            return None
        return self._store.resolve_event(
            user_id=user_id,
            event_id=event.id,
            status="applied",
            application_id=event.application_id,
        )

    def _application_candidates(self, *, user_id: str) -> tuple[ApplicationEmailCandidate, ...]:
        summaries = self._application_service.list_applications(user_id=user_id, limit=100)
        return tuple(
            ApplicationEmailCandidate(
                application_id=item.application.id,
                company_name=item.job.posting.company_name,
                job_title=item.job.posting.title,
                status=item.application.status,
                submitted_at=item.application.submitted_at,
            )
            for item in summaries
            if item.application.status != "withdrawn"
        )

    @classmethod
    def _is_candidate(
        cls,
        metadata: RemoteEmailMetadata,
        applications: tuple[ApplicationEmailCandidate, ...],
    ) -> bool:
        haystack = f"{metadata.sender} {metadata.subject}".casefold()
        if any(term in haystack for term in cls._CANDIDATE_TERMS):
            return True
        return any(
            candidate.company_name.casefold() in haystack
            or candidate.job_title.casefold() in haystack
            for candidate in applications
        )

    @staticmethod
    def _scrub(content: RemoteEmailContent) -> RemoteEmailContent:
        """Strip credentials before the body reaches a model or a hash.

        Recruiting mail carries magic-link tokens, password resets, and one-time
        codes that the classifier never needs. Scrubbing here rather than inside
        the worker means every downstream consumer inherits it, and the content
        hash is taken over the scrubbed text so a resent mail with a rotated
        token still dedupes to the same digest.
        """
        scrubbed = redact_text(content.text)
        if scrubbed == content.text:
            return content
        return content.model_copy(update={"text": scrubbed})

    @staticmethod
    def _scrub_metadata(metadata: RemoteEmailMetadata) -> RemoteEmailMetadata:
        """Strip credentials from the envelope, not just the body.

        ``subject`` is persisted and sent to the model, so a code announced in the
        subject would leak on both paths even with a scrubbed body. ``sender`` is
        scrubbed for symmetry; a display name is free text the platform controls.
        """
        updates = {
            field: scrubbed
            for field, scrubbed in (
                ("sender", redact_text(metadata.sender)),
                ("subject", redact_text(metadata.subject)),
            )
            if scrubbed != getattr(metadata, field)
        }
        return metadata.model_copy(update=updates) if updates else metadata

    def _require_account(self, *, user_id: str, account_id: str):
        account = self._store.get_account(user_id=user_id, account_id=account_id)
        if account is None or account.status != "active":
            raise EmailAccountNotFoundError(account_id)
        return account

    def _can_auto_apply(self, assessment: EmailAssessment) -> bool:
        return (
            getattr(self._worker, "authorizes_auto_apply", False)
            and assessment.application_id is not None
            and assessment.event_type != "unclear"
            and assessment.confidence >= self._auto_apply_confidence
        )
