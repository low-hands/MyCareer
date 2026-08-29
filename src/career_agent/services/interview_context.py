from __future__ import annotations

from dataclasses import dataclass

from career_agent.agent.interview_preparation_contracts import (
    InterviewLogisticsContext,
    InterviewPreparationContext,
    PreparationConfirmedFact,
    PriorInterviewQuestionContext,
    PriorInterviewRetroContext,
)
from career_agent.services.applications import ApplicationService
from career_agent.services.interviews import InterviewNotFoundError, InterviewService
from career_agent.storage.career_history import CareerHistoryStore
from career_agent.storage.resumes import ResumeStore, StoredResumeDocument


class InterviewContextInputNotFoundError(ValueError):
    pass


@dataclass(frozen=True)
class InterviewPreparationSources:
    application_id: str
    job_posting_id: str
    jd_snapshot_id: str
    resume_version_id: str
    document: StoredResumeDocument
    context: InterviewPreparationContext


class InterviewPreparationContextFactory:
    """Builds one source-grounded context shared by prep and mock workers."""

    def __init__(
        self,
        *,
        interviews: InterviewService,
        applications: ApplicationService,
        resumes: ResumeStore,
        career_history: CareerHistoryStore,
    ) -> None:
        self._interviews = interviews
        self._applications = applications
        self._resumes = resumes
        self._career_history = career_history

    def build(
        self,
        *,
        user_id: str,
        application_id: str,
        interview_round_id: str | None = None,
    ) -> InterviewPreparationSources:
        application = self._applications.get_application(
            user_id=user_id,
            application_id=application_id,
        )
        document = self._resumes.read_version_document(
            user_id=user_id,
            resume_version_id=application.application.resume_version_id,
        )
        if document is None:
            raise InterviewContextInputNotFoundError("resume_version")

        logistics = None
        target_sequence: int | None = None
        if interview_round_id is not None:
            try:
                detail = self._interviews.get_interview(
                    user_id=user_id,
                    interview_round_id=interview_round_id,
                )
            except InterviewNotFoundError as error:
                raise InterviewContextInputNotFoundError("interview_round") from error
            interview = detail.interview
            if interview.application_id != application_id:
                raise InterviewContextInputNotFoundError("interview_application")
            target_sequence = interview.sequence_number
            logistics = InterviewLogisticsContext(
                employer_label=interview.employer_label,
                scheduled_start=interview.scheduled_start,
                scheduled_end=interview.scheduled_end,
                timezone=interview.timezone,
                interview_format=interview.interview_format,
                location=interview.location,
                meeting_url=interview.meeting_url,
            )

        facts = tuple(
            PreparationConfirmedFact(
                claim=evidence.claim,
                source_locator=evidence.source_locator,
                source_quote=evidence.source_quote,
            )
            for evidence in self._career_history.list_evidence(
                user_id=user_id,
                verification_status="confirmed",
                source_resume_version_id=application.application.resume_version_id,
            )
            if evidence.source_locator is not None
            and evidence.source_quote is not None
        )

        prior_retros: list[PriorInterviewRetroContext] = []
        completed = self._interviews.list_interviews(
            user_id=user_id,
            application_id=application_id,
            statuses=("completed",),
            limit=50,
        )
        for interview in sorted(completed, key=lambda item: item.sequence_number):
            if target_sequence is not None and interview.sequence_number >= target_sequence:
                continue
            detail = self._interviews.get_interview(
                user_id=user_id,
                interview_round_id=interview.id,
            )
            if not detail.retros:
                continue
            # InterviewService returns retro versions in persisted creation order.
            report = detail.retros[-1]
            prior_retros.append(
                PriorInterviewRetroContext(
                    sequence_number=interview.sequence_number,
                    employer_label=interview.employer_label,
                    completed_at=interview.completed_at,
                    summary=report.summary,
                    questions=tuple(
                        PriorInterviewQuestionContext(
                            question=item.question,
                            answer_summary=item.answer_summary,
                            self_assessment=item.self_assessment,
                            notes=item.notes,
                        )
                        for item in report.questions
                    ),
                    strengths=report.strengths,
                    difficulties=report.difficulties,
                    interviewer_signals=report.interviewer_signals,
                    next_focus=report.next_focus,
                    action_items=report.action_items,
                    limitations=report.limitations,
                    self_assessment=report.self_assessment,
                )
            )

        return InterviewPreparationSources(
            application_id=application.application.id,
            job_posting_id=application.application.job_posting_id,
            jd_snapshot_id=application.application.jd_snapshot_id,
            resume_version_id=application.application.resume_version_id,
            document=document,
            context=InterviewPreparationContext(
                company_name=application.job.posting.company_name,
                role_title=application.job.posting.title,
                jd_text=application.job.snapshot.content,
                logistics=logistics,
                confirmed_facts=facts,
                prior_retros=tuple(prior_retros),
            ),
        )
