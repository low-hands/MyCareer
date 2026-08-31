"""Read-only HTTP surface for the user's own interface.

This is a different boundary from the agent's tools, and the difference is
deliberate. A tool returns an opaque observation because the model must not be
handed a complete JD or resume it could then paraphrase as its own conclusion.
The person looking at their own dashboard is under no such restriction: it is
their data, and withholding it from them would be a bug, not a safeguard.

What the two boundaries share is that neither may become a way around the
other. Nothing here is reachable by the model, and nothing here writes domain
state on the user's behalf: these endpoints answer "what do I have", and every
change still goes through the agent, where it gets confirmation and an audit
trail.

Identity is the caller's asserted ``user_id``, which is exactly as strong as the
rest of this deployment: a local single-user process with no CORS and no auth.
It is a scoping key, not an authorization boundary, and must not be treated as
one if this ever leaves localhost.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from career_agent.domain.action_center import ActionItem, DailyBrief
from career_agent.services.action_center import ActionCenterService
from career_agent.services.applications import ApplicationService
from career_agent.services.email_tracking import EmailTrackingService
from career_agent.services.interviews import InterviewService
from career_agent.storage.action_center import SQLiteActionItemStore
from career_agent.storage.applications import SQLiteApplicationStore
from career_agent.storage.calendar import SQLiteCalendarStore
from career_agent.storage.context import CareerContextStore
from career_agent.storage.email_tracking import SQLiteEmailTrackingStore
from career_agent.storage.interviews import SQLiteInterviewStore
from career_agent.storage.interview_preparations import SQLiteInterviewPreparationStore
from career_agent.storage.job_research import SQLiteJobResearchStore
from career_agent.storage.jobs import SQLiteJobPostingRepository
from career_agent.storage.mock_interviews import SQLiteMockInterviewStore
from career_agent.storage.resume_job_matches import SQLiteResumeJobMatchStore
from career_agent.storage.resume_tailoring import SQLiteResumeTailoringDraftStore
from career_agent.storage.resumes import ResumeStore
from career_agent.agent.interview_preparation_presenter import (
    render_interview_preparation,
)
from career_agent.agent.interview_retro_presenter import render_interview_retro
from career_agent.agent.job_research_presenter import render_job_research
from career_agent.agent.mock_interview_presenter import (
    INTERVIEW_TYPE_LABELS,
    render_mock_interview_report,
)
from career_agent.agent.resume_job_match_presenter import render_resume_job_match
from career_agent.agent.resume_tailoring_presenter import (
    TailoringChangeReviewView,
    render_resume_tailoring,
)
from career_agent.domain.job_research import (
    JobResearchDraft,
    JobResearchFindingDraft,
    JobResearchSourceDraft,
)
from career_agent.connectors.email_accounts import EnvironmentEmailConnectorResolver


class ActionItemView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    action_type: str
    source_type: str
    application_id: str | None = None
    title: str
    summary: str
    due_at: datetime | None = None
    status: str
    snoozed_until: datetime | None = None

    @classmethod
    def of(cls, item: ActionItem) -> "ActionItemView":
        # stable_key and source_id stay behind: they are derivation plumbing,
        # and echoing them would invite a client to key its own state on them.
        return cls(
            id=item.id,
            action_type=item.action_type,
            source_type=item.source_type,
            application_id=item.application_id,
            title=item.title,
            summary=item.summary,
            due_at=item.due_at,
            status=item.status,
            snoozed_until=item.snoozed_until,
        )


class DailyBriefResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    timezone: str
    generated_at: datetime
    overdue: tuple[ActionItemView, ...] = ()
    due_today: tuple[ActionItemView, ...] = ()
    upcoming: tuple[ActionItemView, ...] = ()
    no_due_date: tuple[ActionItemView, ...] = ()

    @classmethod
    def of(cls, brief: DailyBrief) -> "DailyBriefResponse":
        return cls(
            timezone=brief.timezone,
            generated_at=brief.generated_at,
            overdue=tuple(ActionItemView.of(item) for item in brief.overdue),
            due_today=tuple(ActionItemView.of(item) for item in brief.due_today),
            upcoming=tuple(ActionItemView.of(item) for item in brief.upcoming),
            no_due_date=tuple(ActionItemView.of(item) for item in brief.no_due_date),
        )


class ApplicationView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    status: str
    title: str
    company_name: str
    city: str | None = None
    salary: str | None = None
    submitted_at: datetime
    updated_at: datetime


class SavedJobView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    title: str
    company_name: str
    city: str | None = None
    salary: str | None = None
    source_name: str
    source_url: str | None = None
    availability_status: str
    captured_at: datetime
    last_checked_at: datetime
    application_status: str | None = None
    analysis_summary: str | None = None
    responsibilities: tuple[str, ...] = ()
    required_skills: tuple[str, ...] = ()
    preferred_qualifications: tuple[str, ...] = ()
    clarification_questions: tuple[str, ...] = ()
    analyzed_at: datetime | None = None


class ConversationView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    status: str
    title: str
    last_message_preview: str
    message_count: int
    active_workflow: str | None = None
    phase: str | None = None
    created_at: datetime
    last_active_at: datetime


class ConversationResourceView(BaseModel):
    """The report a message's prose was drawn from, for the UI to fetch.

    Carries the internal id because this response goes to the person who owns
    the report, not to the decision model. The model-facing projection replaces
    it with a turn-local index; these are two different audiences and only one
    of them can be trusted with an id it could pass to a tool.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str
    resource_id: str
    status_at_delivery: str | None = None
    anchored_by_other_job: bool | None = None


class ConversationMessageView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str
    content: str
    created_at: datetime
    resource: ConversationResourceView | None = None


class ConversationTranscriptResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    messages: tuple[ConversationMessageView, ...] = ()
    active_workflow: str | None = None
    phase: str | None = None


class ResumeView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    name: str
    target_role: str
    status: str
    latest_version_number: int
    version_count: int
    document_format: str
    byte_size: int
    updated_at: datetime


class CalendarAccountView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    provider: str
    email_address: str
    calendar_id: str
    status: str
    updated_at: datetime


class CalendarEventView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    interview_round_id: str
    status: str
    external_html_link: str | None = None
    updated_at: datetime


class CalendarWorkspaceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    accounts: tuple[CalendarAccountView, ...] = ()
    events: tuple[CalendarEventView, ...] = ()


class CompanyResearchView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    company_name: str
    anchor_job_title: str
    status: str
    focus: str | None = None
    summary: str
    finding_count: int
    source_count: int
    created_at: datetime


class ReportView(BaseModel):
    """One stored report, rendered for the person who owns it.

    The conversation row for a report-producing turn keeps only the short prose
    the user read; the report itself lives in its own entity. This is how the UI
    gets from that message's ``resource`` back to the full text, so ``body`` is
    the same rendered Markdown the turn originally put on screen rather than a
    second, thinner summary of it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str
    resource_id: str
    title: str
    subtitle: str
    body: str
    created_at: datetime


class DashboardStats(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    saved_jobs: int
    applications: int
    interviewing: int
    offers: int
    resumes: int
    research_reports: int
    calendar_accounts: int


class DashboardResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stats: DashboardStats
    application_statuses: dict[str, int]
    recent_jobs: tuple[SavedJobView, ...] = ()
    recent_applications: tuple[ApplicationView, ...] = ()
    next_actions: tuple[ActionItemView, ...] = ()


class WorkspaceReader:
    """Read models for the person's UI, separate from model-facing tools."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._jobs = SQLiteJobPostingRepository(Path(args.job_store).expanduser())
        self._resumes = ResumeStore(Path(args.resume_store).expanduser())
        self._applications = ApplicationService(
            SQLiteApplicationStore(Path(args.application_store).expanduser()),
            self._jobs,
            self._resumes,
        )
        self._calendar = SQLiteCalendarStore(Path(args.calendar_store).expanduser())
        self._research = SQLiteJobResearchStore(
            Path(args.job_research_store).expanduser()
        )
        self._context = CareerContextStore(Path(args.context_store).expanduser())
        self._mock_interviews = SQLiteMockInterviewStore(
            Path(args.mock_interview_store).expanduser()
        )
        self._interviews = SQLiteInterviewStore(
            Path(args.application_store).expanduser()
        )
        # Interview preparations share the resume store file, as the CLI wires
        # them: the preparation is keyed to one exact resume version.
        self._preparations = SQLiteInterviewPreparationStore(
            Path(args.resume_store).expanduser()
        )
        self._resume_matches = SQLiteResumeJobMatchStore(
            Path(args.resume_store).expanduser()
        )
        self._tailoring_drafts = SQLiteResumeTailoringDraftStore(
            Path(args.resume_store).expanduser()
        )

    def applications(self, *, user_id: str, limit: int = 100) -> tuple[ApplicationView, ...]:
        return tuple(
            ApplicationView(
                id=item.application.id,
                status=item.application.status,
                title=item.job.posting.title,
                company_name=item.job.posting.company_name,
                city=item.job.city,
                salary=item.job.salary,
                submitted_at=item.application.submitted_at,
                updated_at=item.application.updated_at,
            )
            for item in self._applications.list_applications(
                user_id=user_id,
                limit=limit,
            )
        )

    def jobs(self, *, user_id: str, limit: int = 100) -> tuple[SavedJobView, ...]:
        applications_by_job = {
            item.application.job_posting_id: item.application.status
            for item in self._applications.list_applications(
                user_id=user_id,
                limit=500,
            )
        }
        views = []
        for item in self._jobs.list_jobs(user_id=user_id, limit=limit):
            analysis = self._jobs.get_latest_analysis(
                user_id=user_id,
                job_posting_id=item.job_posting_id,
            )
            views.append(SavedJobView(
                id=item.job_posting_id,
                title=item.title,
                company_name=item.company_name,
                city=item.city,
                salary=item.salary,
                source_name=item.source_name,
                source_url=item.source_url,
                availability_status=item.availability_status,
                captured_at=item.captured_at,
                last_checked_at=item.last_checked_at,
                application_status=applications_by_job.get(item.job_posting_id),
                analysis_summary=analysis.analysis.job_summary if analysis else None,
                responsibilities=analysis.analysis.responsibilities if analysis else (),
                required_skills=analysis.analysis.required_skills if analysis else (),
                preferred_qualifications=(
                    analysis.analysis.preferred_qualifications if analysis else ()
                ),
                clarification_questions=(
                    analysis.analysis.clarification_questions if analysis else ()
                ),
                analyzed_at=analysis.created_at if analysis else None,
            ))
        return tuple(views)

    def job_count(self, *, user_id: str) -> int:
        return self._jobs.count_jobs(user_id=user_id)

    def conversations(
        self, *, user_id: str, limit: int = 50
    ) -> tuple[ConversationView, ...]:
        views = []
        for item in self._context.list_conversations(user_id=user_id, limit=limit):
            task = self._context.get_task(user_id, item.conversation_id)
            views.append(
                ConversationView(
                    id=item.conversation_id,
                    status=item.status,
                    title=item.title,
                    last_message_preview=item.last_message_preview,
                    message_count=item.message_count,
                    active_workflow=task.active_workflow if task else None,
                    phase=task.phase if task else None,
                    created_at=item.created_at,
                    last_active_at=item.last_active_at,
                )
            )
        return tuple(views)

    def conversation_messages(
        self,
        *,
        user_id: str,
        conversation_id: str,
        limit: int = 200,
    ) -> ConversationTranscriptResponse:
        if self._context.get_session(user_id, conversation_id) is None:
            return ConversationTranscriptResponse()
        task = self._context.get_task(user_id, conversation_id)
        return ConversationTranscriptResponse(
            messages=tuple(
                ConversationMessageView(
                    role=item.role,
                    content=item.content,
                    created_at=item.created_at,
                    resource=(
                        ConversationResourceView(
                            kind=item.resource_ref.kind,
                            resource_id=item.resource_ref.resource_id,
                            status_at_delivery=item.resource_ref.status_at_delivery,
                            anchored_by_other_job=item.resource_ref.anchored_by_other_job,
                        )
                        if item.resource_ref is not None
                        else None
                    ),
                )
                for item in self._context.list_messages(
                    user_id,
                    conversation_id,
                    limit=limit,
                )
            ),
            active_workflow=task.active_workflow if task else None,
            phase=task.phase if task else None,
        )

    def resumes(self, *, user_id: str) -> tuple[ResumeView, ...]:
        roles = {
            role.id: role
            for role in self._resumes.list_target_roles(user_id=user_id)
        }
        views = []
        for resume in self._resumes.list_resumes(user_id=user_id):
            versions = self._resumes.list_versions(
                user_id=user_id,
                resume_id=resume.id,
            )
            if not versions:
                continue
            latest = versions[0]
            views.append(
                ResumeView(
                    id=resume.id,
                    name=resume.name,
                    target_role=(
                        roles[resume.target_role_id].title
                        if resume.target_role_id in roles
                        else "未分类岗位"
                    ),
                    status=resume.status,
                    latest_version_number=latest.version_number,
                    version_count=len(versions),
                    document_format=latest.document_format,
                    byte_size=latest.byte_size,
                    updated_at=resume.updated_at,
                )
            )
        return tuple(views)

    def calendar(self, *, user_id: str) -> CalendarWorkspaceResponse:
        return CalendarWorkspaceResponse(
            accounts=tuple(
                CalendarAccountView(
                    id=item.id,
                    provider=item.provider,
                    email_address=item.email_address,
                    calendar_id=item.calendar_id,
                    status=item.status,
                    updated_at=item.updated_at,
                )
                for item in self._calendar.list_accounts(user_id=user_id)
            ),
            events=tuple(
                CalendarEventView(
                    id=item.id,
                    interview_round_id=item.interview_round_id,
                    status=item.status,
                    external_html_link=item.external_html_link,
                    updated_at=item.updated_at,
                )
                for item in self._calendar.list_links(user_id=user_id)
            ),
        )

    def research(
        self, *, user_id: str, limit: int = 100
    ) -> tuple[CompanyResearchView, ...]:
        views = []
        for report in self._research.list_reports(user_id=user_id, limit=limit):
            job = self._jobs.get_job(
                user_id=user_id,
                job_posting_id=report.job_posting_id,
            )
            if job is None:
                continue
            views.append(
                CompanyResearchView(
                    id=report.id,
                    company_name=job.posting.company_name,
                    anchor_job_title=job.posting.title,
                    status=report.status,
                    focus=report.scope.focus,
                    summary=report.summary,
                    finding_count=len(report.findings),
                    source_count=len(
                        self._research.list_sources(
                            user_id=user_id,
                            report_id=report.id,
                        )
                    ),
                    created_at=report.created_at,
                )
            )
        return tuple(views)

    def report(
        self,
        *,
        user_id: str,
        kind: str,
        resource_id: str,
        status_at_delivery: Literal["current", "outdated", "superseded"] | None = None,
        anchored_by_other_job: bool | None = None,
    ) -> ReportView | None:
        """Read back the report one conversation turn produced.

        Reached by the ``resource`` on a transcript message, so the kind comes
        from the same row as the id and is dispatched on rather than guessed. An
        unknown kind returns nothing instead of falling through to a lookup in
        the wrong store, where a colliding id would serve the wrong report.
        """
        readers = {
            "job_research_report": self._job_research_report,
            "mock_interview_report": self._mock_interview_report,
            "interview_preparation": self._interview_preparation,
            "interview_retro_report": self._interview_retro_report,
            "resume_job_match": self._resume_job_match,
            "resume_tailoring_draft": self._resume_tailoring_draft,
        }
        reader = readers.get(kind)
        if reader is None:
            return None
        if kind == "job_research_report":
            return self._job_research_report(
                user_id,
                resource_id,
                status_at_delivery=status_at_delivery,
                anchored_by_other_job=anchored_by_other_job,
            )
        return reader(user_id, resource_id)

    def _job_research_report(
        self,
        user_id: str,
        report_id: str,
        *,
        status_at_delivery: Literal["current", "outdated", "superseded"] | None = None,
        anchored_by_other_job: bool | None = None,
    ) -> ReportView | None:
        report = self._research.get_report(user_id=user_id, report_id=report_id)
        if report is None:
            return None
        sources = self._research.list_sources(user_id=user_id, report_id=report_id)
        job = self._jobs.get_job(
            user_id=user_id, job_posting_id=report.job_posting_id
        )
        company = job.posting.company_name if job is not None else "公司"
        # The presenter takes the draft shape the worker produced, which is the
        # persisted report minus its identifiers, so it is rebuilt here rather
        # than storing the rendered text a second time.
        draft = JobResearchDraft(
            summary=report.summary,
            sources=tuple(
                JobResearchSourceDraft(
                    source_key=source.source_key,
                    url=source.url,
                    title=source.title,
                    publisher=source.publisher,
                    published_at=source.published_at,
                    relevant_excerpt=source.relevant_excerpt,
                )
                for source in sources
            ),
            findings=tuple(
                JobResearchFindingDraft(
                    topic=finding.topic,
                    statement=finding.statement,
                    evidence_type=finding.evidence_type,
                    source_keys=finding.source_keys,
                    confidence=finding.confidence,
                )
                for finding in report.findings
            ),
            open_questions=report.open_questions,
            limitations=report.limitations,
        )
        return ReportView(
            kind="job_research_report",
            resource_id=report.id,
            title=f"{company} 公司调研",
            subtitle=f"{len(report.findings)} 条结论 · {len(sources)} 个来源",
            body=render_job_research(
                draft,
                status=status_at_delivery or report.status,
                user_provided_context=report.scope.user_provided_context,
                anchored_by_other_job=bool(anchored_by_other_job),
            ),
            created_at=report.created_at,
        )

    def _mock_interview_report(
        self, user_id: str, report_id: str
    ) -> ReportView | None:
        session_id = self._mock_interviews.find_report_session_id(
            user_id=user_id, report_id=report_id
        )
        if session_id is None:
            return None
        report = self._mock_interviews.get_report(
            user_id=user_id, session_id=session_id
        )
        session = self._mock_interviews.get_session(
            user_id=user_id, session_id=session_id
        )
        if report is None or session is None:
            return None
        interview_type = INTERVIEW_TYPE_LABELS.get(
            session.interview_type, session.interview_type
        )
        return ReportView(
            kind="mock_interview_report",
            resource_id=report.id,
            title="模拟面试报告",
            subtitle=f"{interview_type} · {len(report.question_results)} 题",
            body=render_mock_interview_report(report),
            created_at=report.created_at,
        )

    def _interview_preparation(
        self, user_id: str, preparation_id: str
    ) -> ReportView | None:
        stored = self._preparations.get(
            user_id=user_id, preparation_id=preparation_id
        )
        if stored is None:
            return None
        job = self._jobs.get_job(
            user_id=user_id, job_posting_id=stored.job_posting_id
        )
        subject = (
            f"{job.posting.company_name} {job.posting.title}"
            if job is not None
            else "面试"
        )
        return ReportView(
            kind="interview_preparation",
            resource_id=stored.id,
            title="面试准备材料",
            subtitle=(
                f"{subject} · {len(stored.result.likely_questions)} 个可能问题"
            ),
            body=render_interview_preparation(stored.result),
            created_at=stored.created_at,
        )

    def _interview_retro_report(
        self, user_id: str, retro_report_id: str
    ) -> ReportView | None:
        report = self._interviews.get_retro(
            user_id=user_id, retro_report_id=retro_report_id
        )
        if report is None:
            return None
        interview = self._interviews.get(
            user_id=user_id, interview_round_id=report.interview_round_id
        )
        subject = interview.employer_label if interview is not None else None
        return ReportView(
            kind="interview_retro_report",
            resource_id=report.id,
            title="真实面试复盘",
            subtitle=(
                f"{subject} · {len(report.questions)} 个问题"
                if subject
                else f"{len(report.questions)} 个问题"
            ),
            body=render_interview_retro(report),
            created_at=report.created_at,
        )

    def _resume_job_match(
        self, user_id: str, match_id: str
    ) -> ReportView | None:
        stored = self._resume_matches.get(user_id=user_id, match_id=match_id)
        if stored is None:
            return None
        job = self._jobs.get_job(
            user_id=user_id, job_posting_id=stored.job_posting_id
        )
        subject = (
            f"{job.posting.company_name} {job.posting.title}"
            if job is not None
            else "已保存岗位"
        )
        return ReportView(
            kind="resume_job_match",
            resource_id=stored.id,
            title="简历与岗位匹配",
            subtitle=f"{subject} · {stored.result.overall_fit}",
            body=render_resume_job_match(stored.result),
            created_at=stored.created_at,
        )

    def _resume_tailoring_draft(
        self, user_id: str, draft_id: str
    ) -> ReportView | None:
        stored = self._tailoring_drafts.get_for_display(
            user_id=user_id, draft_id=draft_id
        )
        if stored is None:
            return None
        expired = datetime.now(timezone.utc) >= stored.expires_at
        # A terminal lineage fact is more informative than the clock: an old
        # superseded draft must still say which newer revision replaced it.
        # TTL expiry is shown for drafts whose own review state is otherwise
        # still actionable-looking.
        display_status = (
            stored.status
            if stored.status in {"finalized", "superseded"}
            else "expired"
            if expired
            else stored.status
        )
        return ReportView(
            kind="resume_tailoring_draft",
            resource_id=stored.id,
            title="简历定制草稿",
            subtitle=(
                f"修订 {stored.revision_number} · {display_status} · "
                f"{len(stored.result.changes)} 条建议"
            ),
            body=render_resume_tailoring(
                stored.result,
                status=display_status,
                revision_number=stored.revision_number,
                change_reviews=tuple(
                    TailoringChangeReviewView.model_validate(
                        review.model_dump(mode="python")
                    )
                    for review in stored.change_reviews
                ),
            ),
            created_at=stored.created_at,
        )


def build_workspace_reader(args: argparse.Namespace) -> WorkspaceReader:
    return WorkspaceReader(args)


def build_action_center_service(args: argparse.Namespace) -> ActionCenterService:
    """Assemble the action centre without requiring any model configuration.

    Generated actions are derived from applications, interviews, and recorded
    email events, none of which needs a worker to read. EmailTrackingService
    falls back to its deterministic worker, so a dashboard keeps working on a
    machine where no API key is set.
    """
    job_repository = SQLiteJobPostingRepository(Path(args.job_store).expanduser())
    resume_store = ResumeStore(Path(args.resume_store).expanduser())
    application_service = ApplicationService(
        SQLiteApplicationStore(Path(args.application_store).expanduser()),
        job_repository,
        resume_store,
    )
    interview_service = InterviewService(
        SQLiteInterviewStore(Path(args.application_store).expanduser()),
        application_service,
    )
    email_tracking_service = EmailTrackingService(
        SQLiteEmailTrackingStore(Path(args.email_store).expanduser()),
        application_service,
        EnvironmentEmailConnectorResolver(),
        interview_service=interview_service,
    )
    return ActionCenterService(
        SQLiteActionItemStore(Path(args.action_store).expanduser()),
        application_service,
        email_tracking_service,
        interview_service,
    )


def build_read_router(
    action_center_factory: Callable[[], ActionCenterService],
    workspace_reader_factory: Callable[[], WorkspaceReader],
) -> APIRouter:
    """Wire the read endpoints against a lazily built service.

    The factory runs on the first request rather than at wiring time, so
    creating an app never opens the local databases as a side effect.
    """
    router = APIRouter(prefix="/v1")
    cached: dict[str, Any] = {}

    def action_center() -> ActionCenterService:
        if "service" not in cached:
            cached["service"] = action_center_factory()
        return cached["service"]

    def workspace() -> WorkspaceReader:
        if "workspace" not in cached:
            cached["workspace"] = workspace_reader_factory()
        return cached["workspace"]

    @router.get("/daily-brief", response_model=DailyBriefResponse)
    async def daily_brief(
        user_id: str = Query(min_length=1, max_length=200),
        timezone: str = Query(default="Asia/Shanghai", min_length=1, max_length=100),
    ) -> DailyBriefResponse:
        # This regenerates derived action items before answering. That is a
        # write, which a GET would normally not do, but the items are a pure
        # function of the pipeline keyed by stable_key: reading a stale brief
        # would be the actual surprise. Nothing the user authored is touched.
        return DailyBriefResponse.of(
            action_center().daily_brief(user_id=user_id, timezone_name=timezone)
        )

    @router.get("/applications", response_model=tuple[ApplicationView, ...])
    async def applications(
        user_id: str = Query(min_length=1, max_length=200),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> tuple[ApplicationView, ...]:
        return workspace().applications(user_id=user_id, limit=limit)

    @router.get("/jobs", response_model=tuple[SavedJobView, ...])
    async def jobs(
        user_id: str = Query(min_length=1, max_length=200),
        limit: int = Query(default=100, ge=1, le=100),
    ) -> tuple[SavedJobView, ...]:
        return workspace().jobs(user_id=user_id, limit=limit)

    @router.get("/conversations", response_model=tuple[ConversationView, ...])
    async def conversations(
        user_id: str = Query(min_length=1, max_length=200),
        limit: int = Query(default=50, ge=1, le=100),
    ) -> tuple[ConversationView, ...]:
        return workspace().conversations(user_id=user_id, limit=limit)

    @router.get(
        "/conversations/{conversation_id}/messages",
        response_model=ConversationTranscriptResponse,
    )
    async def conversation_messages(
        conversation_id: str,
        user_id: str = Query(min_length=1, max_length=200),
        limit: int = Query(default=200, ge=1, le=500),
    ) -> ConversationTranscriptResponse:
        return workspace().conversation_messages(
            user_id=user_id,
            conversation_id=conversation_id,
            limit=limit,
        )

    @router.get("/resumes", response_model=tuple[ResumeView, ...])
    async def resumes(
        user_id: str = Query(min_length=1, max_length=200),
    ) -> tuple[ResumeView, ...]:
        return workspace().resumes(user_id=user_id)

    @router.get("/calendar", response_model=CalendarWorkspaceResponse)
    async def calendar(
        user_id: str = Query(min_length=1, max_length=200),
    ) -> CalendarWorkspaceResponse:
        return workspace().calendar(user_id=user_id)

    @router.get("/company-research", response_model=tuple[CompanyResearchView, ...])
    async def company_research(
        user_id: str = Query(min_length=1, max_length=200),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> tuple[CompanyResearchView, ...]:
        return workspace().research(user_id=user_id, limit=limit)

    @router.get("/reports/{kind}/{resource_id}", response_model=ReportView)
    async def report(
        kind: str,
        resource_id: str,
        user_id: str = Query(min_length=1, max_length=200),
        status_at_delivery: Literal["current", "outdated", "superseded"] | None = Query(default=None),
        anchored_by_other_job: bool | None = Query(default=None),
    ) -> ReportView:
        view = workspace().report(
            user_id=user_id,
            kind=kind,
            resource_id=resource_id,
            status_at_delivery=status_at_delivery,
            anchored_by_other_job=anchored_by_other_job,
        )
        if view is None:
            # 404 for an unknown kind as well as a missing row: from the UI's
            # side both mean the message's resource does not resolve, and the
            # two are not worth distinguishing to a caller that only ever sends
            # a kind the transcript gave it.
            raise HTTPException(status_code=404, detail="报告不存在。")
        return view

    @router.get("/dashboard", response_model=DashboardResponse)
    async def dashboard(
        user_id: str = Query(min_length=1, max_length=200),
        timezone: str = Query(default="Asia/Shanghai", min_length=1, max_length=100),
    ) -> DashboardResponse:
        application_items = workspace().applications(user_id=user_id, limit=500)
        job_items = workspace().jobs(user_id=user_id, limit=6)
        saved_job_count = workspace().job_count(user_id=user_id)
        resume_items = workspace().resumes(user_id=user_id)
        research_items = workspace().research(user_id=user_id, limit=500)
        calendar_items = workspace().calendar(user_id=user_id)
        brief = DailyBriefResponse.of(
            action_center().daily_brief(user_id=user_id, timezone_name=timezone)
        )
        statuses: dict[str, int] = {}
        for item in application_items:
            statuses[item.status] = statuses.get(item.status, 0) + 1
        next_actions = (
            *brief.overdue,
            *brief.due_today,
            *brief.upcoming,
            *brief.no_due_date,
        )[:6]
        return DashboardResponse(
            stats=DashboardStats(
                saved_jobs=saved_job_count,
                applications=len(application_items),
                interviewing=statuses.get("interviewing", 0),
                offers=statuses.get("offer", 0),
                resumes=len(resume_items),
                research_reports=len(research_items),
                calendar_accounts=len(calendar_items.accounts),
            ),
            application_statuses=statuses,
            recent_jobs=job_items,
            recent_applications=application_items[:5],
            next_actions=next_actions,
        )

    return router
