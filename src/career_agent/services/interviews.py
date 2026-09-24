from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from career_agent.domain.interviews import (
    InterviewDetails,
    InterviewRetroQuestion,
    InterviewRetroReport,
    InterviewRound,
    InterviewRoundEvent,
    InterviewSelfAssessment,
    InterviewStatus,
)
from career_agent.services.applications import (
    ApplicationInputNotFoundError,
    ApplicationService,
)
from career_agent.storage.interviews import SQLiteInterviewStore


class InterviewNotFoundError(ValueError):
    pass


class InterviewApplicationConflictError(ValueError):
    pass


class AmbiguousInterviewMatchError(ValueError):
    pass


@dataclass(frozen=True)
class InterviewDetail:
    interview: InterviewRound
    events: tuple[InterviewRoundEvent, ...]
    retros: tuple[InterviewRetroReport, ...] = ()


@dataclass(frozen=True)
class InterviewRecording:
    interview: InterviewRound
    created: bool


class InterviewService:
    def __init__(
        self,
        store: SQLiteInterviewStore,
        application_service: ApplicationService,
    ) -> None:
        self._store = store
        self._application_service = application_service

    def record_email_event(
        self,
        *,
        user_id: str,
        application_id: str,
        email_event_id: str,
        source_thread_id: str | None,
        details: InterviewDetails,
        occurred_at: datetime,
        interview_round_id: str | None = None,
    ) -> InterviewRecording:
        existing_for_event = self._store.find_for_email_event(
            user_id=user_id,
            email_event_id=email_event_id,
        )
        if existing_for_event is not None:
            return InterviewRecording(interview=existing_for_event, created=False)
        self._require_trackable_application(
            user_id=user_id,
            application_id=application_id,
        )
        matched = self._match_round(
            user_id=user_id,
            application_id=application_id,
            details=details,
            source_thread_id=source_thread_id,
            interview_round_id=interview_round_id,
        )
        if matched is None:
            if details.change_type != "invited":
                raise AmbiguousInterviewMatchError(
                    "an update or cancellation must identify an existing interview"
                )
            created = self._store.create(
                user_id=user_id,
                application_id=application_id,
                details=details,
                source="email_sync",
                email_event_id=email_event_id,
                source_thread_id=source_thread_id,
                occurred_at=occurred_at,
            )
            return InterviewRecording(interview=created, created=True)
        updated = self._store.update(
            round_=matched,
            details=details,
            source="email_sync",
            email_event_id=email_event_id,
            source_thread_id=source_thread_id,
            occurred_at=occurred_at,
        )
        return InterviewRecording(interview=updated, created=False)

    def create_manual(
        self,
        *,
        user_id: str,
        application_id: str,
        details: InterviewDetails,
    ) -> InterviewRound:
        self._require_trackable_application(
            user_id=user_id,
            application_id=application_id,
        )
        normalized = details.model_copy(update={"change_type": "invited"})
        return self._store.create(
            user_id=user_id,
            application_id=application_id,
            details=normalized,
            source="user_reported",
            email_event_id=None,
            source_thread_id=None,
            occurred_at=datetime.now(timezone.utc),
        )

    def list_interviews(
        self,
        *,
        user_id: str,
        application_id: str | None = None,
        statuses: tuple[InterviewStatus, ...] = (),
        limit: int = 50,
    ) -> tuple[InterviewRound, ...]:
        return self._store.list(
            user_id=user_id,
            application_id=application_id,
            statuses=statuses,
            limit=limit,
        )

    def update_manual(
        self,
        *,
        user_id: str,
        interview_round_id: str,
        details: InterviewDetails,
    ) -> InterviewRound:
        interview = self._store.get(
            user_id=user_id,
            interview_round_id=interview_round_id,
        )
        if interview is None:
            raise InterviewNotFoundError(interview_round_id)
        if interview.status == "completed":
            raise InterviewApplicationConflictError(
                "completed interview requires a correction event"
            )
        return self._store.update(
            round_=interview,
            details=details,
            source="user_reported",
            email_event_id=None,
            source_thread_id=None,
            occurred_at=datetime.now(timezone.utc),
        )

    def complete_interview(
        self,
        *,
        user_id: str,
        interview_round_id: str,
        completed_at: datetime | None = None,
    ) -> InterviewRound:
        interview = self._store.get(
            user_id=user_id,
            interview_round_id=interview_round_id,
        )
        if interview is None:
            raise InterviewNotFoundError(interview_round_id)
        try:
            completed = self._store.complete(
                round_=interview,
                occurred_at=completed_at or datetime.now(timezone.utc),
            )
        except ValueError as error:
            raise InterviewApplicationConflictError(str(error)) from error
        # Completing a real interview is also a pipeline milestone. Keep the
        # application card in sync without guessing an employer outcome.
        application = self._application_service.get_application(
            user_id=user_id,
            application_id=completed.application_id,
        ).application
        update_application = getattr(self._application_service, "update_application", None)
        if application.status == "interviewing" and callable(update_application):
            update_application(
                user_id=user_id,
                application_id=completed.application_id,
                status="interview_completed",
                note="面试已完成，等待招聘方结果。",
                source="user_reported",
            )
        return completed

    def get_interview(
        self, *, user_id: str, interview_round_id: str
    ) -> InterviewDetail:
        interview = self._store.get(
            user_id=user_id,
            interview_round_id=interview_round_id,
        )
        if interview is None:
            raise InterviewNotFoundError(interview_round_id)
        return InterviewDetail(
            interview=interview,
            events=self._store.list_events(
                user_id=user_id,
                interview_round_id=interview.id,
            ),
            retros=self._store.list_retros(
                user_id=user_id,
                interview_round_id=interview.id,
            ),
        )

    def record_retro(
        self,
        *,
        user_id: str,
        interview_round_id: str,
        source_notes: str,
        summary: str,
        questions: tuple[InterviewRetroQuestion, ...] = (),
        strengths: tuple[str, ...] = (),
        difficulties: tuple[str, ...] = (),
        interviewer_signals: tuple[str, ...] = (),
        next_focus: tuple[str, ...] = (),
        action_items: tuple[str, ...] = (),
        limitations: tuple[str, ...] = (),
        self_assessment: InterviewSelfAssessment = "uncertain",
    ) -> InterviewRetroReport:
        interview = self._store.get(
            user_id=user_id,
            interview_round_id=interview_round_id,
        )
        if interview is None:
            raise InterviewNotFoundError(interview_round_id)
        if interview.status != "completed":
            raise InterviewApplicationConflictError(
                "a real interview retro requires a user-confirmed completed interview"
            )
        return self._store.record_retro(
            round_=interview,
            source_notes=source_notes,
            summary=summary,
            questions=questions,
            strengths=strengths,
            difficulties=difficulties,
            interviewer_signals=interviewer_signals,
            next_focus=next_focus,
            action_items=action_items,
            limitations=limitations,
            self_assessment=self_assessment,
        )

    def list_retros(
        self,
        *,
        user_id: str,
        interview_round_id: str,
        limit: int = 50,
    ) -> tuple[InterviewRetroReport, ...]:
        interview = self._store.get(
            user_id=user_id,
            interview_round_id=interview_round_id,
        )
        if interview is None:
            raise InterviewNotFoundError(interview_round_id)
        return self._store.list_retros(
            user_id=user_id,
            interview_round_id=interview_round_id,
            limit=limit,
        )

    def _match_round(
        self,
        *,
        user_id: str,
        application_id: str,
        details: InterviewDetails,
        source_thread_id: str | None,
        interview_round_id: str | None,
    ) -> InterviewRound | None:
        if interview_round_id is not None:
            selected = self._store.get(
                user_id=user_id,
                interview_round_id=interview_round_id,
            )
            if selected is None:
                raise InterviewNotFoundError(interview_round_id)
            if selected.application_id != application_id:
                raise InterviewApplicationConflictError(
                    "interview and email event belong to different applications"
                )
            return selected
        if source_thread_id is not None:
            threaded = self._store.find_by_source_thread(
                user_id=user_id,
                application_id=application_id,
                source_thread_id=source_thread_id,
            )
            if threaded is not None:
                if details.change_type == "invited":
                    same_identity = (
                        details.scheduled_start is not None
                        and details.scheduled_start == threaded.scheduled_start
                    ) or (
                        details.meeting_url is not None
                        and details.meeting_url == threaded.meeting_url
                    )
                    has_new_identity = (
                        details.scheduled_start is not None
                        or details.meeting_url is not None
                    )
                    if same_identity or not has_new_identity:
                        return threaded
                    if (
                        threaded.status == "completed"
                        or details.employer_label is not None
                        and details.employer_label != threaded.employer_label
                    ):
                        return None
                    raise AmbiguousInterviewMatchError(
                        "same-thread invitation has a different appointment identity"
                    )
                if threaded.status == "completed":
                    if details.change_type != "invited":
                        raise AmbiguousInterviewMatchError(
                            "completed interview cannot receive schedule updates"
                        )
                return threaded
        identity_matches = self._store.find_by_identity(
            user_id=user_id,
            application_id=application_id,
            scheduled_start=details.scheduled_start,
            meeting_url=details.meeting_url,
        )
        if len(identity_matches) == 1:
            return identity_matches[0]
        if len(identity_matches) > 1:
            raise AmbiguousInterviewMatchError("multiple interviews match this schedule")
        if details.change_type != "invited":
            active = self._store.list(
                user_id=user_id,
                application_id=application_id,
                statuses=("identified", "scheduled"),
                limit=10,
            )
            if len(active) == 1:
                return active[0]
            if len(active) > 1:
                raise AmbiguousInterviewMatchError(
                    "multiple active interviews require explicit selection"
                )
        return None

    def _require_trackable_application(
        self, *, user_id: str, application_id: str
    ) -> None:
        try:
            detail = self._application_service.get_application(
                user_id=user_id,
                application_id=application_id,
            )
        except ApplicationInputNotFoundError as error:
            raise InterviewApplicationConflictError(
                "application is unavailable"
            ) from error
        if detail.application.status in {"offer", "rejected", "withdrawn"}:
            raise InterviewApplicationConflictError(
                "terminal application cannot receive a new interview"
            )
