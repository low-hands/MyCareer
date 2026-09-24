"""Read-only HTTP surface for the user's own interface.

This is a different boundary from the agent's tools, and the difference is
deliberate. A tool returns an opaque observation because the model must not be
handed a complete JD or resume it could then paraphrase as its own conclusion.
The person looking at their own dashboard is under no such restriction: it is
their data, and withholding it from them would be a bug, not a safeguard.

What the two boundaries share is that neither may become a way around the
other. Nothing here is reachable by the model. Most routes answer "what do I
have"; the small workspace-write surface records direct, reversible UI facts
such as "ignore this job" or "I observed this posting closed". Those are not
agent decisions and therefore do not detour through the model.

Identity is the API key the caller presents, never a ``user_id`` they send. It
used to be the latter, with this docstring warning that it was a scoping key and
not an authorization boundary — which is exactly what it was being used as by
anyone who could reach the port. Every route now derives the user from the
credential and requires the ``workspace:read`` scope, so a key issued to a
browser extension for capture cannot read any of this.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import re
from urllib.parse import quote
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from pathlib import Path
from typing import Any, Literal

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Response,
    UploadFile,
)
from fastapi import Path as FastAPIPath
from pydantic import BaseModel, ConfigDict, Field

from career_agent.agent.delivered_body_contracts import (
    BodyDependency,
    MockInterviewBodySource,
    ResumeAnalysisBodySource,
    SavedJobBodySource,
)
from career_agent.security.authentication import require_scope
from career_agent.storage.api_keys import (
    ApiKeyPrincipal,
    WORKSPACE_READ,
    WORKSPACE_WRITE,
)
from career_agent.domain.action_center import ActionItem, DailyBrief
from career_agent.services.action_center import ActionCenterService
from career_agent.services.applications import (
    ApplicationInputNotFoundError,
    ApplicationService,
)
from career_agent.services.email_tracking import EmailTrackingService
from career_agent.services.interviews import InterviewService
from career_agent.services.resume_import import (
    MAX_RESUME_IMPORT_BYTES,
    extract_resume_text,
    validate_resume_document,
)
from career_agent.storage.action_center import SQLiteActionItemStore
from career_agent.storage.applications import SQLiteApplicationStore
from career_agent.storage.calendar import SQLiteCalendarStore
from career_agent.storage.context import CareerContextStore
from career_agent.storage.capability_confirmations import SQLiteCapabilityConfirmationStore
from career_agent.storage.email_tracking import SQLiteEmailTrackingStore
from career_agent.storage.interviews import SQLiteInterviewStore
from career_agent.storage.interview_preparations import SQLiteInterviewPreparationStore
from career_agent.storage.job_research import SQLiteJobResearchStore
from career_agent.storage.jobs import SQLiteJobPostingRepository
from career_agent.storage.mock_interviews import SQLiteMockInterviewStore
from career_agent.storage.resume_job_matches import (
    SQLiteResumeJobMatchStore,
    StoredResumeJobMatch,
)
from career_agent.storage.resume_analysis import SQLiteResumeAnalysisDraftStore
from career_agent.storage.resume_tailoring import SQLiteResumeTailoringDraftStore
from career_agent.storage.resumes import (
    ResumeImportConflictError,
    ResumeStore,
    StoredResumeDocument,
)
from career_agent.domain.resume import ResumeVersion
from career_agent.agent.interview_preparation_presenter import (
    render_interview_preparation,
)
from career_agent.agent.interview_retro_presenter import render_interview_retro
from career_agent.agent.job_analysis_contracts import (
    SENIORITY_LABELS,
    JobAnalysisResult,
)
from career_agent.agent.resume_job_match_contracts import IntentAlignment
from career_agent.agent.job_analysis_presenter import render_job_analysis
from career_agent.agent.job_research_presenter import render_job_research
from career_agent.agent.mock_interview_presenter import (
    INTERVIEW_TYPE_LABELS,
    mock_interview_question_view,
    render_mock_interview_question,
    render_mock_interview_report,
)
from career_agent.agent.resume_job_match_presenter import render_resume_job_match
from career_agent.agent.resume_analysis_presenter import render_resume_analysis
from career_agent.harness.streaming import (
    InteractionRequiredEvent,
    capability_confirmation_event,
    resume_analysis_confirmation_event,
    questionnaire_event,
)
from career_agent.agent.resume_tailoring_presenter import (
    TailoringChangeReviewView,
    render_resume_tailoring,
)
from career_agent.agent.main_agent_contracts import (
    CONFIRMATION_SPECS,
    ConversationResourceReference,
    OwnerSettingsContext,
)
from career_agent.domain.job_research import (
    JobResearchDraft,
    JobResearchFindingDraft,
    JobResearchSourceDraft,
)
from career_agent.connectors.email_accounts import EnvironmentEmailConnectorResolver
from career_agent.storage.connector_secrets import KeyringConnectorSecretStore


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
    interview_round_number: int | None = None
    interview_round_label: str | None = None
    interview_status: str | None = None


class ApplicationCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    job_posting_id: str = Field(min_length=1, max_length=200)
    resume_version_id: str | None = Field(default=None, min_length=1, max_length=200)
    submitted_at: datetime | None = None
    note: str | None = Field(default=None, max_length=2_000)


RESUMABLE_MOCK_INTERVIEW_STATUSES = frozenset({"created", "active", "paused"})


class MockInterviewSessionView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str
    status: str
    interview_type: str
    interview_type_label: str
    question_count: int
    max_primary_questions: int
    report_id: str | None = None
    summary: str | None = None
    conversation_id: str | None = None
    created_at: datetime
    completed_at: datetime | None = None
    updated_at: datetime


class ApplicationMockInterviewsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    application_id: str
    title: str
    company_name: str
    sessions: tuple[MockInterviewSessionView, ...] = ()


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
    pursuit_status: Literal["open", "dismissed"] = "open"
    captured_at: datetime
    last_checked_at: datetime
    application_status: str | None = None
    analysis_summary: str | None = None
    responsibilities: tuple[str, ...] = ()
    required_skills: tuple[str, ...] = ()
    preferred_qualifications: tuple[str, ...] = ()
    clarification_questions: tuple[str, ...] = ()
    analyzed_at: datetime | None = None
    jd_snapshot_id: str | None = None
    jd_version: int | None = None
    # ``stale``: the newest analysis belongs to an earlier JD version, so the
    # current version is still waiting. ``ready`` always refers to the current
    # version; the historical analysis stays readable either way.
    jd_analysis_status: Literal["none", "ready", "stale"] = "none"
    jd_analysis_version: int | None = None
    jd_analysis: JobAnalysisResult | None = None
    resume_match_status: Literal["none", "ready", "stale"] = "none"
    resume_match_fit: str | None = None
    resume_match_at: datetime | None = None
    resume_match_count: int = 0


class ResumeJobMatchView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    report_id: str
    job_posting_id: str
    job_title: str | None = None
    company_name: str | None = None
    jd_snapshot_id: str
    jd_version: int | None = None
    jd_captured_at: datetime | None = None
    jd_available: bool
    current_jd: bool | None = None
    resume_version_id: str
    resume_id: str | None = None
    resume_name: str | None = None
    resume_version_number: int | None = None
    resume_created_at: datetime | None = None
    resume_available: bool
    matcher_version: str
    created_at: datetime
    overall_fit: str
    summary: str
    intent_alignment: IntentAlignment | None = None


class JobMatchHistoryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    items: tuple[ResumeJobMatchView, ...] = ()
    total: int
    limit: int
    offset: int


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


DELIVERED_BODY_KIND = "delivered_body"
"""The transcript resource kind for a body kept beside its row.

A row that condenses a body no card holds (a daily brief, a resume analysis,
a job comparison) carries one of these, so the UI can open what the stream
showed. Only the reader ever sees the kind: it is not a
``ConversationResourceReference`` and never reaches the decision model.
"""


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
    title: str | None = None
    """What to call the card before its body is fetched, when the kind alone
    does not say: a ``delivered_body`` may be a brief or a comparison."""
    description: str | None = None
    available: bool | None = None
    """Whether the referenced resource can still be opened.

    Set for user-supplied inputs such as a ``resume_version``: the snapshot
    (title, description) stays on the row after the resume is deleted, and
    this flag is what tells the UI to render it as gone rather than as a link.
    ``None`` means the kind does not track availability here."""
    resume_id: str | None = None
    """The owning resume of an available ``resume_version``, so the UI can
    address the document route without a second lookup."""


class ConversationMessageView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["user", "assistant"]
    content: str
    created_at: datetime
    resources: tuple[ConversationResourceView, ...] = ()
    """Plural because one turn can store two reports; see 070 (F/复核修正)."""


class ConversationTranscriptResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    messages: tuple[ConversationMessageView, ...] = ()
    active_workflow: str | None = None
    phase: str | None = None
    pending_interaction: InteractionRequiredEvent | None = None
    pending_interaction_body: str | None = None


def dedupe_adjacent_message_resources(
    messages: tuple[ConversationMessageView, ...],
) -> tuple[ConversationMessageView, ...]:
    """Enforce one visual owner for a resource within a user/reply pair.

    Older rows may contain the same attachment on both sides of a turn. The
    attachment belongs to the user message; assistant-produced resources keep
    their own card. This compatibility projection repairs historical rows
    without mutating the audit log.
    """
    projected: list[ConversationMessageView] = []
    for message in messages:
        resources = message.resources
        if message.role == "assistant" and projected:
            previous = projected[-1]
            if previous.role == "user" and previous.resources:
                owned = {
                    (resource.kind, resource.resource_id)
                    for resource in previous.resources
                }
                resources = tuple(
                    resource
                    for resource in resources
                    if (resource.kind, resource.resource_id) not in owned
                )
        projected.append(message.model_copy(update={"resources": resources}))
    return tuple(projected)


RESUME_DOCUMENT_MEDIA_TYPES: dict[str, tuple[str, str]] = {
    "pdf": ("application/pdf", "pdf"),
    "text": ("text/plain; charset=utf-8", "txt"),
    "markdown": ("text/markdown; charset=utf-8", "md"),
}

_FILENAME_UNSAFE = re.compile(r"[\x00-\x1f\x7f\\/:*?\"<>|;%]")


def resume_document_filename(
    resume_name: str, *, version_number: int, extension: str
) -> str:
    """A download name built from the resume, safe to place in a header.

    Control characters, path separators and header-significant punctuation
    are dropped; the stem is bounded so a long resume name cannot inflate the
    header. An empty stem falls back to a neutral name rather than to an
    extension-only filename.
    """
    stem = _FILENAME_UNSAFE.sub("", resume_name).strip(" .")
    stem = " ".join(stem.split())[:80] or "resume"
    return f"{stem}-v{version_number}.{extension}"


def content_disposition(disposition: str, filename: str) -> str:
    """RFC 6266 value with an ASCII fallback and a UTF-8 ``filename*``.

    Control characters are dropped here as well as upstream, so the header
    stays a single line whatever name it is handed; the ASCII form keeps the
    characters a plain quoted-string can carry, and the percent-encoded form
    carries the rest.
    """
    filename = _FILENAME_UNSAFE.sub("", filename)
    ascii_name = filename.encode("ascii", "ignore").decode("ascii")
    ascii_name = ascii_name.strip() or "resume"
    encoded = quote(filename, safe="")
    return f"{disposition}; filename=\"{ascii_name}\"; filename*=UTF-8''{encoded}"


class ResumeVersionView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    resume_id: str
    version_number: int
    document_format: str
    byte_size: int
    created_at: datetime
    change_summary: str


class ResumeView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    name: str
    target_role: str
    status: str
    latest_version_number: int
    latest_version_id: str
    version_count: int
    document_format: str
    byte_size: int
    updated_at: datetime
    versions: tuple[ResumeVersionView, ...] = ()
    """Every immutable version, newest first."""


class TargetRoleView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    title: str
    priority: int
    status: str


class ResumeImportResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    resume_id: str
    resume_version_id: str
    name: str
    version_number: int
    document_format: str
    byte_size: int


class EmailAccountView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    provider: str
    email_address: str
    status: str
    connection_status: str = "connected"
    needs_reauthorization: bool = False
    last_synced_at: datetime | None = None


class EmailEventView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    event_type: str
    status: str
    summary: str
    confidence: float
    application_id: str | None = None
    application_title: str | None = None
    company_name: str | None = None
    occurred_at: datetime


class EmailWorkspaceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    accounts: tuple[EmailAccountView, ...] = ()
    events: tuple[EmailEventView, ...] = ()


class CalendarAccountView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    provider: str
    email_address: str
    calendar_id: str
    status: str
    connection_status: str = "connected"
    needs_reauthorization: bool = False
    updated_at: datetime


class CalendarEventView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    interview_round_id: str
    application_id: str | None = None
    company_name: str | None = None
    job_title: str | None = None
    employer_label: str | None = None
    sequence_number: int | None = None
    interview_status: str | None = None
    scheduled_start: datetime | None = None
    scheduled_end: datetime | None = None
    timezone: str | None = None
    interview_format: str | None = None
    location: str | None = None
    meeting_url: str | None = None
    contact_summary: str | None = None
    sync_status: str = "synced"
    status: str
    external_html_link: str | None = None
    updated_at: datetime


class CalendarWorkspaceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    accounts: tuple[CalendarAccountView, ...] = ()
    events: tuple[CalendarEventView, ...] = ()
    month: str | None = None
    timezone: str = "Asia/Shanghai"


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
    availability: Literal["available", "expired"] = "available"
    resume_job_match: ResumeJobMatchView | None = None


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


def resume_version_change_summary(
    current: str | None, previous: str | None, *, has_previous: bool
) -> str:
    if not has_previous:
        return "初始版本"
    if current is None or previous is None:
        return "无法从文件中提取文本，暂无改动摘要"
    import difflib
    before = previous.splitlines()
    after = current.splitlines()
    matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    added: list[str] = []
    removed: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            removed.extend(before[i1:i2])
        if tag in ("replace", "insert"):
            added.extend(after[j1:j2])
    if not added and not removed:
        return "与上一版本内容一致"
    parts: list[str] = []
    if added:
        parts.append("新增：" + "；".join(added[:2]))
    if removed:
        parts.append("删改：" + "；".join(removed[:2]))
    result = " / ".join(parts)
    return result[:320] + ("…" if len(result) > 320 else "")


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
        email_store = getattr(
            args,
            "email_store",
            Path(args.context_store).expanduser().with_name("email.sqlite3"),
        )
        self._email = SQLiteEmailTrackingStore(Path(email_store).expanduser())
        self._research = SQLiteJobResearchStore(
            Path(args.job_research_store).expanduser()
        )
        self._context = CareerContextStore(Path(args.context_store).expanduser())
        self._capability_confirmations = SQLiteCapabilityConfirmationStore(
            Path(args.context_store).expanduser()
        )
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
        self._resume_analyses = SQLiteResumeAnalysisDraftStore(
            Path(args.resume_store).expanduser()
        )

    def applications(self, *, user_id: str, limit: int = 100) -> tuple[ApplicationView, ...]:
        views = []
        for item in self._applications.list_applications(
            user_id=user_id,
            limit=limit,
        ):
            latest = self._latest_interview(user_id=user_id, application_id=item.application.id)
            views.append(ApplicationView(
                id=item.application.id,
                status=item.application.status,
                title=item.job.posting.title,
                company_name=item.job.posting.company_name,
                city=item.job.city,
                salary=item.job.salary,
                submitted_at=item.application.submitted_at,
                updated_at=item.application.updated_at,
                interview_round_number=latest.sequence_number if latest else None,
                interview_round_label=latest.employer_label if latest else None,
                interview_status=latest.status if latest else None,
            )
        )
        return tuple(views)

    def _latest_interview(self, *, user_id: str, application_id: str):
        rounds = self._interviews.list(
            user_id=user_id,
            application_id=application_id,
            limit=100,
        )
        return max(rounds, key=lambda item: item.sequence_number, default=None)

    def application_mock_interviews(
        self,
        *,
        user_id: str,
        application_id: str,
        limit: int = 50,
    ) -> ApplicationMockInterviewsResponse:
        application = self._applications.get_application(
            user_id=user_id,
            application_id=application_id,
        )
        sessions = []
        for session in self._mock_interviews.list_sessions(
            user_id=user_id,
            application_id=application_id,
            limit=limit,
        ):
            report = self._mock_interviews.get_report(
                user_id=user_id,
                session_id=session.id,
            )
            completed_questions = (
                len(report.question_results)
                if report is not None
                else sum(
                    1
                    for turn in self._mock_interviews.list_turns(
                        user_id=user_id,
                        session_id=session.id,
                    )
                    if turn.turn_type == "primary" and turn.status == "evaluated"
                )
            )
            conversation_id = (
                self._context.conversation_owning_run(
                    user_id=user_id,
                    workflow="mock_interview",
                    run_id=session.id,
                )
                if session.status in RESUMABLE_MOCK_INTERVIEW_STATUSES
                else None
            )
            sessions.append(
                MockInterviewSessionView(
                    session_id=session.id,
                    status=session.status,
                    interview_type=session.interview_type,
                    interview_type_label=INTERVIEW_TYPE_LABELS.get(
                        session.interview_type,
                        session.interview_type,
                    ),
                    question_count=completed_questions,
                    max_primary_questions=session.max_primary_questions,
                    report_id=report.id if report is not None else None,
                    summary=report.summary if report is not None else None,
                    conversation_id=conversation_id,
                    created_at=session.created_at,
                    completed_at=session.completed_at,
                    updated_at=session.updated_at,
                )
            )
        return ApplicationMockInterviewsResponse(
            application_id=application.application.id,
            title=application.job.posting.title,
            company_name=application.job.posting.company_name,
            sessions=tuple(sessions),
        )

    def create_application(
        self,
        *,
        user_id: str,
        job_posting_id: str,
        resume_version_id: str | None,
        submitted_at: datetime | None,
        note: str | None,
    ) -> ApplicationView:
        created = self._applications.create_application(
            user_id=user_id,
            job_posting_id=job_posting_id,
            resume_version_id=resume_version_id,
            submitted_at=submitted_at,
            note=note,
        )
        detail = self._applications.get_application(
            user_id=user_id,
            application_id=created.application.id,
        )
        return ApplicationView(
            id=detail.application.id,
            status=detail.application.status,
            title=detail.job.posting.title,
            company_name=detail.job.posting.company_name,
            city=detail.job.city,
            salary=detail.job.salary,
            submitted_at=detail.application.submitted_at,
            updated_at=detail.application.updated_at,
            interview_round_number=(latest := self._latest_interview(
                user_id=user_id, application_id=detail.application.id
            )).sequence_number if latest else None,
            interview_round_label=latest.employer_label if latest else None,
            interview_status=latest.status if latest else None,
        )

    def clear_applications(self, *, user_id: str) -> int:
        return self._applications._application_store.clear_user(user_id=user_id)

    def delete_resume(self, *, user_id: str, resume_id: str) -> bool:
        return self._resumes.delete_resume(user_id=user_id, resume_id=resume_id)

    def delete_research(self, *, user_id: str, report_id: str) -> bool:
        return self._research.delete_report(user_id=user_id, report_id=report_id)

    def set_job_availability(
        self, *, user_id: str, job_posting_id: str, status: str
    ) -> bool:
        return self._jobs.mark_availability(
            user_id=user_id, job_posting_id=job_posting_id, status=status
        )

    def set_job_pursuit(
        self, *, user_id: str, job_posting_id: str, status: str
    ) -> bool:
        return self._jobs.set_pursuit_status(
            user_id=user_id, job_posting_id=job_posting_id, status=status
        )

    def delete_job(self, *, user_id: str, job_posting_id: str) -> Literal[
        "deleted", "not_found", "has_application"
    ]:
        # An application keeps the exact posting and JD snapshot as part of
        # its audit trail.  Deleting that input would make the application
        # unreadable, so permanent deletion is limited to library-only jobs.
        if job_posting_id in self._applications.list_job_posting_ids(
            user_id=user_id
        ):
            return "has_application"
        if not self._jobs.delete_job(
            user_id=user_id, job_posting_id=job_posting_id
        ):
            return "not_found"
        self._context.purge_delivered_body_dependency(
            user_id=user_id,
            dependency=BodyDependency(kind="job", resource_id=job_posting_id),
        )
        return "deleted"

    def job_detail(
        self, *, user_id: str, job_posting_id: str
    ) -> SavedJobDetailView | None:
        record = self._jobs.get_job(
            user_id=user_id, job_posting_id=job_posting_id
        )
        if record is None:
            return None
        return SavedJobDetailView(
            id=record.posting.id,
            title=record.posting.title,
            company_name=record.posting.company_name,
            city=record.city,
            salary=record.salary,
            source_name=record.posting.source_name,
            source_url=record.posting.source_url,
            availability_status=record.availability_status,
            pursuit_status=record.pursuit_status,
            jd_text=record.snapshot.content,
            jd_version=record.snapshot.version,
            captured_at=record.snapshot.captured_at,
        )

    def jd_snapshot(
        self, *, user_id: str, jd_snapshot_id: str
    ) -> SavedJobSnapshotView | None:
        snapshot = self._jobs.get_snapshot(
            user_id=user_id, jd_snapshot_id=jd_snapshot_id
        )
        if snapshot is None:
            return None
        record = self._jobs.get_job(
            user_id=user_id, job_posting_id=snapshot.job_posting_id
        )
        if record is None:
            return None
        return SavedJobSnapshotView(
            jd_snapshot_id=snapshot.id,
            job_posting_id=record.posting.id,
            title=record.posting.title,
            company_name=record.posting.company_name,
            source_name=record.posting.source_name,
            source_url=record.posting.source_url,
            jd_version=snapshot.version,
            latest_jd_version=record.snapshot.version,
            jd_text=snapshot.content,
            captured_at=snapshot.captured_at,
        )

    def jobs(
        self, *, user_id: str, limit: int = 100, include_dismissed: bool = False
    ) -> tuple[SavedJobView, ...]:
        applications_by_job = {
            item.application.job_posting_id: item.application.status
            for item in self._applications.list_applications(
                user_id=user_id,
                limit=500,
            )
        }
        views = []
        for item in self._jobs.list_jobs(
            user_id=user_id, limit=limit, include_dismissed=include_dismissed
        ):
            current = self._jobs.get_latest_analysis(
                user_id=user_id,
                job_posting_id=item.job_posting_id,
            )
            analysis = current or self._jobs.get_latest_analysis_any_snapshot(
                user_id=user_id,
                job_posting_id=item.job_posting_id,
            )
            analysis_version = (
                self._jobs.get_snapshot(
                    user_id=user_id, jd_snapshot_id=analysis.jd_snapshot_id
                )
                if analysis is not None and current is None
                else None
            )
            match = self._resume_matches.find_latest_for_job(
                user_id=user_id, job_posting_id=item.job_posting_id,
                jd_snapshot_id=item.jd_snapshot_id,
            ) or self._resume_matches.find_latest_for_job(
                user_id=user_id, job_posting_id=item.job_posting_id
            )
            views.append(SavedJobView(
                id=item.job_posting_id,
                title=item.title,
                company_name=item.company_name,
                city=item.city,
                salary=item.salary,
                source_name=item.source_name,
                pursuit_status=item.pursuit_status,
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
                jd_snapshot_id=item.jd_snapshot_id,
                jd_version=item.jd_version,
                jd_analysis_status=(
                    "none" if analysis is None else "ready" if current else "stale"
                ),
                jd_analysis_version=(
                    item.jd_version
                    if current is not None
                    else analysis_version.version
                    if analysis_version is not None
                    else None
                ),
                jd_analysis=analysis.analysis.to_result() if analysis else None,
                resume_match_status=(
                    "none"
                    if match is None
                    else "ready"
                    if match.jd_snapshot_id == item.jd_snapshot_id
                    else "stale"
                ),
                resume_match_fit=match.result.overall_fit if match else None,
                resume_match_at=match.created_at if match else None,
                resume_match_count=self._resume_matches.count_for_job(
                    user_id=user_id, job_posting_id=item.job_posting_id
                ),
            ))
        return tuple(views)

    def job_matches(
        self, *, user_id: str, job_posting_id: str, limit: int = 20, offset: int = 0
    ) -> JobMatchHistoryResponse | None:
        total = self._resume_matches.count_for_job(
            user_id=user_id, job_posting_id=job_posting_id
        )
        if not total and self._jobs.get_job(
            user_id=user_id, job_posting_id=job_posting_id
        ) is None:
            return None
        return JobMatchHistoryResponse(
            items=tuple(
                self._match_view(user_id, stored)
                for stored in self._resume_matches.list_for_job(
                    user_id=user_id, job_posting_id=job_posting_id,
                    limit=limit, offset=offset,
                )
            ),
            total=total, limit=limit, offset=offset,
        )

    def _match_view(
        self, user_id: str, stored: StoredResumeJobMatch
    ) -> ResumeJobMatchView:
        job = self._jobs.get_job(
            user_id=user_id, job_posting_id=stored.job_posting_id
        )
        snapshot = self._jobs.get_snapshot(
            user_id=user_id, jd_snapshot_id=stored.jd_snapshot_id
        )
        if snapshot is not None and snapshot.job_posting_id != stored.job_posting_id:
            snapshot = None
        source = self._resumes.get_version(
            user_id=user_id, resume_version_id=stored.resume_version_id
        )
        resume, version = source if source else (None, None)
        inputs = stored.inputs
        return ResumeJobMatchView(
            report_id=stored.id,
            job_posting_id=stored.job_posting_id,
            job_title=inputs.job_title if inputs else job.posting.title if job else None,
            company_name=inputs.company_name if inputs else job.posting.company_name if job else None,
            jd_snapshot_id=stored.jd_snapshot_id,
            jd_version=inputs.jd_version if inputs else snapshot.version if snapshot else None,
            jd_captured_at=inputs.jd_captured_at if inputs else snapshot.captured_at if snapshot else None,
            jd_available=snapshot is not None,
            current_jd=job.snapshot.id == stored.jd_snapshot_id if job else None,
            resume_version_id=stored.resume_version_id,
            resume_id=inputs.resume_id if inputs else resume.id if resume else None,
            resume_name=inputs.resume_name if inputs else resume.name if resume else None,
            resume_version_number=inputs.resume_version_number if inputs else version.version_number if version else None,
            resume_created_at=inputs.resume_created_at if inputs else version.created_at if version else None,
            resume_available=source is not None,
            matcher_version=stored.matcher_version,
            created_at=stored.created_at,
            overall_fit=stored.result.overall_fit,
            intent_alignment=stored.result.intent_alignment,
            summary=stored.result.summary,
        )

    def job_count(self, *, user_id: str) -> int:
        return self._jobs.count_jobs(user_id=user_id, include_dismissed=False)

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

    def delete_conversation(self, *, user_id: str, conversation_id: str) -> bool:
        return self._context.delete_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
        )

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
        pending_analysis = (
            self._resume_analyses.get(
                user_id=user_id,
                analysis_id=task.active_resume_analysis_id,
            )
            if task is not None
            and task.resume_analysis_status == "pending"
            and task.active_resume_analysis_id is not None
            else None
        )
        owner_settings = self._context.get_owner_settings(user_id) or OwnerSettingsContext()
        pending_confirmations = self._capability_confirmations.pending_for_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
            policy_revision=owner_settings.behavior_policy.revision,
        )
        pending_confirmation = (
            pending_confirmations[0] if pending_confirmations else None
        )
        records = self._context.list_message_records(
            user_id,
            conversation_id,
            limit=limit,
        )
        # Kept bodies follow their rows: a body on a row outside this page is
        # not listed, and one whose row is suppressed is already filtered out.
        kept_bodies: dict[int, list[ConversationResourceView]] = {}
        if records:
            for kept in self._context.list_delivered_body_references(
                user_id,
                conversation_id,
                from_sequence=records[0].sequence,
            ):
                kept_bodies.setdefault(kept.sequence, []).append(
                    ConversationResourceView(
                        kind=DELIVERED_BODY_KIND,
                        resource_id=kept.body_id,
                        title=kept.title,
                    )
                )
        messages = tuple(
                ConversationMessageView(
                    role=record.message.role,
                    content=record.message.content,
                    created_at=record.message.created_at,
                    resources=(
                        *(
                            self._resource_view(user_id, reference)
                            for reference in record.message.resource_refs
                        ),
                        *kept_bodies.get(record.sequence, ()),
                    ),
                )
                for record in records
            )
        return ConversationTranscriptResponse(
            messages=dedupe_adjacent_message_resources(messages),
            active_workflow=task.active_workflow if task else None,
            phase=task.phase if task else None,
            pending_interaction=(
                capability_confirmation_event(
                    conversation_id=conversation_id,
                    confirmation_id=pending_confirmation.confirmation_id,
                    prompt=(
                        f"{pending_confirmation.display_summary}\n"
                        + (
                            "这是删除或停用操作，请亲自确认是否执行。"
                            if pending_confirmation.capability in CONFIRMATION_SPECS
                            and CONFIRMATION_SPECS[
                                pending_confirmation.capability
                            ].requires_seal
                            else "你设置了此操作需要确认。是否执行？"
                        )
                    ),
                )
                if pending_confirmation is not None
                else resume_analysis_confirmation_event(
                    conversation_id=conversation_id,
                    analysis_id=pending_analysis.id,
                )
                if pending_analysis is not None
                and pending_analysis.status == "pending"
                else questionnaire_event(task.pending_questionnaire)
                if task is not None
                and task.pending_questionnaire is not None
                and task.pending_questionnaire.expires_at > datetime.now(timezone.utc)
                else None
            ),
            pending_interaction_body=(
                render_resume_analysis(pending_analysis.result)
                if pending_analysis is not None
                and pending_analysis.status == "pending"
                and pending_confirmation is None
                else None
            ),
        )

    def _resource_view(
        self, user_id: str, reference: ConversationResourceReference
    ) -> ConversationResourceView:
        if reference.kind == "saved_job":
            # The row keeps the display snapshot; whether the pinned JD version
            # can still be opened is decided now, so a deleted posting shows
            # as gone without rewriting the card's name.
            snapshot = self._jobs.get_snapshot(
                user_id=user_id, jd_snapshot_id=reference.resource_id
            )
            return ConversationResourceView(
                kind=reference.kind,
                resource_id=reference.resource_id,
                title=reference.title,
                description=reference.description,
                available=snapshot is not None,
            )
        if reference.kind != "resume_version":
            return ConversationResourceView(
                kind=reference.kind,
                resource_id=reference.resource_id,
                status_at_delivery=reference.status_at_delivery,
                anchored_by_other_job=reference.anchored_by_other_job,
            )
        # The row keeps the snapshot; whether the version is still reachable is
        # decided now, so a deleted resume shows as gone without rewriting
        # history.
        located = self._resumes.get_version(
            user_id=user_id, resume_version_id=reference.resource_id
        )
        return ConversationResourceView(
            kind=reference.kind,
            resource_id=reference.resource_id,
            title=reference.title,
            description=reference.description,
            available=located is not None,
            resume_id=located[0].id if located is not None else None,
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
            version_texts: dict[str, str | None] = {}
            model_summaries: dict[str, str] = {}
            for version in versions:
                document = self._resumes.read_version_document(
                    user_id=user_id, resume_version_id=version.id
                )
                version_texts[version.id] = (
                    extract_resume_text(document.document_format, document.raw_bytes)
                    if document is not None
                    else None
                )
                draft_id = self._resumes.get_tailoring_draft_id(
                    user_id=user_id, resume_version_id=version.id
                )
                if draft_id is not None:
                    draft = self._tailoring_drafts.get_for_display(
                        user_id=user_id, draft_id=draft_id
                    )
                    if draft is not None and draft.result.strategy_summary.strip():
                        model_summaries[version.id] = (
                            draft.result.strategy_summary.strip()
                        )
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
                    latest_version_id=latest.id,
                    version_count=len(versions),
                    document_format=latest.document_format,
                    byte_size=latest.byte_size,
                    updated_at=resume.updated_at,
                    versions=tuple(
                        ResumeVersionView(
                            id=version.id,
                            resume_id=version.resume_id,
                            version_number=version.version_number,
                            document_format=version.document_format,
                            byte_size=version.byte_size,
                            created_at=version.created_at,
                            change_summary=model_summaries.get(version.id)
                            or resume_version_change_summary(
                                version_texts.get(version.id),
                                (
                                    version_texts.get(versions[index + 1].id)
                                    if index + 1 < len(versions)
                                    else None
                                ),
                                has_previous=index + 1 < len(versions),
                            ),
                        )
                        for index, version in enumerate(versions)
                    ),
                )
            )
        return tuple(views)

    def resume_version_document(
        self, *, user_id: str, resume_id: str, resume_version_id: str
    ) -> tuple[StoredResumeDocument, str, ResumeVersion] | None:
        """The stored bytes of one version, with a download filename.

        Ownership is checked on the version, and the version is then checked
        against the resume it was addressed under: a version id that belongs to
        the user but to a different resume is not served.
        """
        located = self._resumes.get_version(
            user_id=user_id, resume_version_id=resume_version_id
        )
        if located is None or located[0].id != resume_id:
            return None
        resume, version = located
        document = self._resumes.read_version_document(
            user_id=user_id, resume_version_id=resume_version_id
        )
        if document is None:
            return None
        return document, resume.name, version

    def target_roles(self, *, user_id: str) -> tuple[TargetRoleView, ...]:
        return tuple(
            TargetRoleView(
                id=role.id,
                title=role.title,
                priority=role.priority,
                status=role.status,
            )
            for role in self._resumes.list_target_roles(user_id=user_id)
        )

    def create_target_role(self, *, user_id: str, title: str) -> TargetRoleView:
        normalized = title.strip()
        existing = self._resumes.list_target_roles(user_id=user_id)
        role = next(
            (item for item in existing if item.title.casefold() == normalized.casefold()),
            None,
        )
        if role is None:
            role = self._resumes.create_target_role(
                user_id=user_id,
                title=normalized,
                priority=len(existing),
            )
        return TargetRoleView(
            id=role.id,
            title=role.title,
            priority=role.priority,
            status=role.status,
        )

    def import_resume(
        self,
        *,
        user_id: str,
        content: bytes,
        document_format: str,
        name: str | None,
        resume_id: str | None,
        target_role_id: str | None,
        idempotency_key: str | None = None,
    ) -> ResumeImportResponse:
        resume, version = self._resumes.import_document(
            user_id=user_id,
            content=content,
            document_format=document_format,
            name=name,
            resume_id=resume_id,
            target_role_id=target_role_id,
            idempotency_key=idempotency_key,
        )
        return ResumeImportResponse(
            resume_id=resume.id,
            resume_version_id=version.id,
            name=resume.name,
            version_number=version.version_number,
            document_format=version.document_format,
            byte_size=version.byte_size,
        )

    def email(self, *, user_id: str, limit: int = 100) -> EmailWorkspaceResponse:
        applications = {
            item.application.id: item
            for item in self._applications.list_applications(
                user_id=user_id,
                limit=500,
            )
        }
        return EmailWorkspaceResponse(
            accounts=tuple(
                EmailAccountView(
                    id=account.id,
                    provider=account.provider,
                    email_address=account.email_address,
                    status=account.status,
                    connection_status=(
                        "connected" if account.status == "active" else account.status
                    ),
                    needs_reauthorization=account.status == "error",
                    last_synced_at=(
                        cursor.updated_at
                        if (cursor := self._email.get_cursor(account_id=account.id))
                        else None
                    ),
                )
                for account in self._email.list_accounts(user_id=user_id)
            ),
            events=tuple(
                EmailEventView(
                    id=event.id,
                    event_type=event.event_type,
                    status=event.status,
                    summary=event.summary,
                    confidence=event.confidence,
                    application_id=event.application_id,
                    application_title=(
                        applications[event.application_id].job.posting.title
                        if event.application_id in applications
                        else None
                    ),
                    company_name=(
                        applications[event.application_id].job.posting.company_name
                        if event.application_id in applications
                        else None
                    ),
                    occurred_at=event.occurred_at,
                )
                for event in self._email.list_events(user_id=user_id, limit=limit)
            ),
        )

    def calendar(
        self,
        *,
        user_id: str,
        month: str | None = None,
        timezone_name: str = "Asia/Shanghai",
    ) -> CalendarWorkspaceResponse:
        zone = ZoneInfo(timezone_name)
        selected_month = month or datetime.now(zone).strftime("%Y-%m")
        year, month_number = (int(part) for part in selected_month.split("-"))
        range_start = datetime(year, month_number, 1, tzinfo=zone)
        range_end = (
            datetime(year + 1, 1, 1, tzinfo=zone)
            if month_number == 12
            else datetime(year, month_number + 1, 1, tzinfo=zone)
        )
        links = {
            item.interview_round_id: item
            for item in self._calendar.list_links(user_id=user_id)
        }
        proposals = {
            item.interview_round_id: item
            for item in self._calendar.list_latest_proposals(user_id=user_id)
        }
        applications = {
            item.application.id: item
            for item in self._applications.list_applications(user_id=user_id, limit=500)
        }
        events = []
        for interview in self._interviews.list_scheduled_between(
            user_id=user_id,
            range_start=range_start.astimezone(timezone.utc),
            range_end=range_end.astimezone(timezone.utc),
        ):
            link = links.get(interview.id)
            proposal = proposals.get(interview.id)
            sync_status = "not_synced"
            if proposal is not None and proposal.status in {
                "pending",
                "executing",
                "reconciliation_required",
                "failed",
            }:
                sync_status = {
                    "pending": "pending_approval",
                    "executing": "syncing",
                    "reconciliation_required": "reconciliation_required",
                    "failed": "failed",
                }[proposal.status]
            elif link is not None and link.status == "active":
                sync_status = "synced"
            elif link is not None and link.status == "cancelled":
                sync_status = "cancelled"
            application = applications.get(interview.application_id)
            events.append(
                CalendarEventView(
                    id=link.id if link else interview.id,
                    interview_round_id=interview.id,
                    application_id=interview.application_id,
                    company_name=(
                        application.job.posting.company_name if application else None
                    ),
                    job_title=application.job.posting.title if application else None,
                    employer_label=interview.employer_label,
                    sequence_number=interview.sequence_number,
                    interview_status=interview.status,
                    scheduled_start=interview.scheduled_start,
                    scheduled_end=interview.scheduled_end,
                    timezone=interview.timezone or timezone_name,
                    interview_format=interview.interview_format,
                    location=interview.location,
                    meeting_url=interview.meeting_url,
                    contact_summary=interview.contact_summary,
                    sync_status=sync_status,
                    status=link.status if link else interview.status,
                    external_html_link=link.external_html_link if link else None,
                    updated_at=link.updated_at if link else interview.updated_at,
                )
            )
        return CalendarWorkspaceResponse(
            accounts=tuple(
                CalendarAccountView(
                    id=item.id,
                    provider=item.provider,
                    email_address=item.email_address,
                    calendar_id=item.calendar_id,
                    status=item.status,
                    connection_status=(
                        "connected" if item.status == "active" else item.status
                    ),
                    needs_reauthorization=item.status == "error",
                    updated_at=item.updated_at,
                )
                for item in self._calendar.list_accounts(user_id=user_id)
            ),
            events=tuple(events),
            month=selected_month,
            timezone=timezone_name,
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
            "job_analysis": self._job_analysis,
            "resume_job_match": self._resume_job_match,
            "resume_tailoring_draft": self._resume_tailoring_draft,
            DELIVERED_BODY_KIND: self._delivered_body,
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
        view = reader(user_id, resource_id)
        if view is not None and kind == "resume_job_match":
            stored = self._resume_matches.get(user_id=user_id, match_id=resource_id)
            if stored is not None:
                view = view.model_copy(update={"resume_job_match": self._match_view(user_id, stored)})
        return view

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

    def _job_analysis(self, user_id: str, analysis_id: str) -> ReportView | None:
        stored = self._jobs.get_analysis(user_id=user_id, analysis_id=analysis_id)
        if stored is None:
            return None
        result = stored.analysis.to_result()
        if result is None:
            return None
        job = self._jobs.get_job(
            user_id=user_id, job_posting_id=stored.job_posting_id
        )
        snapshot = self._jobs.get_snapshot(
            user_id=user_id, jd_snapshot_id=stored.jd_snapshot_id
        )
        subject = (
            f"{job.posting.company_name} {job.posting.title}"
            if job is not None
            else "已保存岗位"
        )
        version = f" · JD 第 {snapshot.version} 版" if snapshot is not None else ""
        return ReportView(
            kind="job_analysis",
            resource_id=stored.id,
            title="岗位 JD 分析",
            subtitle=f"{subject} · {SENIORITY_LABELS[result.seniority]}{version}",
            body=render_job_analysis(result),
            created_at=stored.created_at,
        )

    def _delivered_body(self, user_id: str, body_id: str) -> ReportView | None:
        stored = self._context.get_delivered_body(user_id, body_id)
        if stored is None:
            return None
        body = stored.body
        subtitle = "历史快照（截至生成时间）"
        availability: Literal["available", "expired"] = "available"
        if stored.retention == "snapshot":
            for dependency in stored.dependencies:
                if not self._body_dependency_exists(user_id, dependency):
                    self._context.purge_delivered_body_dependency(
                        user_id=user_id, dependency=dependency
                    )
                    return None
        elif isinstance(stored.source, SavedJobBodySource):
            job = self._jobs.get_job(
                user_id=user_id, job_posting_id=stored.source.job_posting_id
            )
            if job is None:
                return None
            body = job.snapshot.content.strip()
            subtitle = "当前岗位描述"
        elif isinstance(stored.source, ResumeAnalysisBodySource):
            now = datetime.now(timezone.utc)
            if now >= stored.source.expires_at:
                body, subtitle, availability = "", "简历分析已过期", "expired"
            else:
                analysis = self._resume_analyses.get(
                    user_id=user_id, analysis_id=stored.source.analysis_id, now=now
                )
                if analysis is None:
                    return None
                body = render_resume_analysis(analysis.result)
                subtitle = "简历分析"
        elif isinstance(stored.source, MockInterviewBodySource):
            session = self._mock_interviews.get_session(
                user_id=user_id, session_id=stored.source.session_id
            )
            if session is None:
                return None
            view = mock_interview_question_view(
                self._mock_interviews.list_turns(user_id=user_id, session_id=session.id),
                stored.source.question_number,
            )
            if view is None:
                return None
            body = render_mock_interview_question(view)
            subtitle = f"第 {stored.source.question_number} 题及追问"
        else:
            return None
        return ReportView(
            kind=DELIVERED_BODY_KIND,
            resource_id=stored.body_id,
            title=stored.title,
            subtitle=subtitle,
            body=body,
            created_at=stored.created_at,
            availability=availability,
        )

    def _body_dependency_exists(self, user_id: str, dependency: BodyDependency) -> bool:
        if dependency.kind == "job":
            return self._jobs.get_job(
                user_id=user_id, job_posting_id=dependency.resource_id
            ) is not None
        if dependency.kind == "interview_round":
            return self._interviews.get(
                user_id=user_id, interview_round_id=dependency.resource_id
            ) is not None
        if dependency.kind == "email_event":
            return self._email.get_event(
                user_id=user_id, event_id=dependency.resource_id
            ) is not None
        try:
            self._applications.get_application(
                user_id=user_id, application_id=dependency.resource_id
            )
        except ApplicationInputNotFoundError:
            return False
        return True

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

    Generated actions are derived from applications, interviews, recorded
    email events, and stored resume-tailoring gaps; none needs a worker to read. EmailTrackingService
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
        EnvironmentEmailConnectorResolver(
            secret_store=KeyringConnectorSecretStore()
        ),
        interview_service=interview_service,
    )
    return ActionCenterService(
        SQLiteActionItemStore(Path(args.action_store).expanduser()),
        application_service,
        email_tracking_service,
        interview_service,
        job_repository=job_repository,
        resume_tailoring_drafts=SQLiteResumeTailoringDraftStore(
            Path(args.resume_store).expanduser()
        ),
    )


class SavedJobDetailView(BaseModel):
    """One saved job with the JD text itself.

    Separate from ``SavedJobView`` and fetched on demand rather than folded
    into the list. A JD runs to thousands of characters, and the library lists
    up to a hundred of them; carrying every body through every refresh would
    make the page slow to serve the one card the reader actually opened.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    title: str
    company_name: str
    city: str | None = None
    salary: str | None = None
    source_name: str
    source_url: str | None = None
    availability_status: str
    pursuit_status: Literal["open", "dismissed"]
    jd_text: str
    """The stored snapshot, verbatim.

    The same bytes ``get_saved_job`` hands the reader in conversation — the
    agent returns this snapshot without a model rewriting it, so the two
    surfaces cannot disagree about what the posting says. That property is why
    both paths can exist: they are two ways to reach one document, not two
    renderings of it.
    """
    jd_version: int
    captured_at: datetime


class SavedJobSnapshotView(BaseModel):
    """One immutable JD version, as a ``saved_job`` card opens it.

    Addressed by ``jd_snapshot_id`` rather than the posting: the card in an
    old turn keeps opening the text that turn read after the posting has been
    captured again. ``latest_jd_version`` lets the card say so.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    jd_snapshot_id: str
    job_posting_id: str
    title: str
    company_name: str
    source_name: str
    source_url: str | None = None
    jd_version: int
    latest_jd_version: int
    jd_text: str
    captured_at: datetime


class AvailabilityUpdate(BaseModel):
    """What the user saw on the posting's own page.

    The employer's state, not the user's decision — the same field the
    extension writes, through a different door. The extension was never a
    separate source of truth: it reports what the person in front of the page
    saw, and this is that same report without the shortcut. One field, one
    meaning, whichever way it arrives.
    """

    model_config = ConfigDict(extra="forbid")

    availability_status: Literal["active", "closed", "unknown"]


class AvailabilityResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    job_posting_id: str
    availability_status: Literal["active", "closed", "unknown"]
    changed: bool


class PursuitStatusUpdate(BaseModel):
    """The user's own decision about a saved job. No ``user_id``: identity
    comes from the credential, like every other route."""

    model_config = ConfigDict(extra="forbid")

    pursuit_status: Literal["open", "dismissed"]


class PursuitStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    job_posting_id: str
    pursuit_status: Literal["open", "dismissed"]
    changed: bool


class JobDeletionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    job_posting_id: str
    deleted: Literal[True] = True


class ConversationDeletionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation_id: str
    deleted: Literal[True] = True

class CountResponse(BaseModel):
    count: int


class TargetRoleCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)


def build_read_router(
    action_center_factory: Callable[[], ActionCenterService],
    workspace_reader_factory: Callable[[], WorkspaceReader],
    *,
    before_conversation_delete: Callable[[str, str], None] | None = None,
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
        principal: ApiKeyPrincipal = Depends(require_scope(WORKSPACE_READ)),
        timezone: str = Query(default="Asia/Shanghai", min_length=1, max_length=100),
    ) -> DailyBriefResponse:
        # This regenerates derived action items before answering. That is a
        # write, which a GET would normally not do, but the items are a pure
        # function of the pipeline keyed by stable_key: reading a stale brief
        # would be the actual surprise. Nothing the user authored is touched.
        return DailyBriefResponse.of(
            action_center().daily_brief(user_id=principal.user_id, timezone_name=timezone)
        )

    @router.get("/applications", response_model=tuple[ApplicationView, ...])
    async def applications(
        principal: ApiKeyPrincipal = Depends(require_scope(WORKSPACE_READ)),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> tuple[ApplicationView, ...]:
        return workspace().applications(user_id=principal.user_id, limit=limit)

    @router.get(
        "/applications/{application_id}/mock-interviews",
        response_model=ApplicationMockInterviewsResponse,
    )
    async def application_mock_interviews(
        application_id: str,
        principal: ApiKeyPrincipal = Depends(require_scope(WORKSPACE_READ)),
        limit: int = Query(default=50, ge=1, le=100),
    ) -> ApplicationMockInterviewsResponse:
        try:
            return workspace().application_mock_interviews(
                user_id=principal.user_id,
                application_id=application_id,
                limit=limit,
            )
        except ApplicationInputNotFoundError as error:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "APPLICATION_NOT_FOUND",
                    "message": "没有找到这条投递记录。",
                },
            ) from error

    @router.post("/applications", response_model=ApplicationView)
    async def create_application(
        request: ApplicationCreateRequest,
        principal: ApiKeyPrincipal = Depends(require_scope(WORKSPACE_WRITE)),
    ) -> ApplicationView:
        try:
            return workspace().create_application(
                user_id=principal.user_id,
                job_posting_id=request.job_posting_id,
                resume_version_id=request.resume_version_id,
                submitted_at=request.submitted_at,
                note=request.note,
            )
        except ApplicationInputNotFoundError as error:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "APPLICATION_INPUT_NOT_FOUND",
                    "message": "没有找到对应的已保存岗位或简历版本。",
                },
            ) from error

    @router.delete("/applications", response_model=CountResponse)
    async def clear_applications(
        principal: ApiKeyPrincipal = Depends(require_scope(WORKSPACE_WRITE)),
    ) -> CountResponse:
        return CountResponse(count=workspace().clear_applications(user_id=principal.user_id))

    @router.get("/jobs", response_model=tuple[SavedJobView, ...])
    async def jobs(
        principal: ApiKeyPrincipal = Depends(require_scope(WORKSPACE_READ)),
        limit: int = Query(default=100, ge=1, le=100),
        include_dismissed: bool = Query(default=False),
    ) -> tuple[SavedJobView, ...]:
        return workspace().jobs(
            user_id=principal.user_id,
            limit=limit,
            include_dismissed=include_dismissed,
        )

    @router.get("/jobs/{job_posting_id}", response_model=SavedJobDetailView)
    async def job_detail(
        job_posting_id: str,
        principal: ApiKeyPrincipal = Depends(require_scope(WORKSPACE_READ)),
    ) -> SavedJobDetailView:
        """The JD itself, so reading a stored document does not need the agent.

        Asking a model to read back a file the user already has costs a model
        call, and until this existed it was the only way: the list carried the
        analysis but never the posting. Both routes now reach the same
        snapshot, which is what makes "ask, or go look" a real choice rather
        than one path with a detour.
        """

        detail = workspace().job_detail(
            user_id=principal.user_id, job_posting_id=job_posting_id
        )
        if detail is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "SAVED_JOB_NOT_FOUND",
                    "message": "没有找到这个已保存职位。",
                },
            )
        return detail

    @router.get(
        "/jobs/{job_posting_id}/matches", response_model=JobMatchHistoryResponse
    )
    async def job_matches(
        job_posting_id: str,
        principal: ApiKeyPrincipal = Depends(require_scope(WORKSPACE_READ)),
        limit: int = Query(default=20, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ) -> JobMatchHistoryResponse:
        view = workspace().job_matches(
            user_id=principal.user_id, job_posting_id=job_posting_id,
            limit=limit, offset=offset,
        )
        if view is None:
            raise HTTPException(status_code=404, detail="岗位不存在或不可访问。")
        return view

    @router.get(
        "/jd-snapshots/{jd_snapshot_id}", response_model=SavedJobSnapshotView
    )
    async def jd_snapshot(
        jd_snapshot_id: str,
        principal: ApiKeyPrincipal = Depends(require_scope(WORKSPACE_READ)),
    ) -> SavedJobSnapshotView:
        """The JD version a conversation card is pinned to.

        Scoped to the caller: a snapshot is only found through a posting the
        caller owns, so another user's id is a 404, not a leak. A posting that
        has since been deleted takes its snapshots with it; the card keeps its
        name from the row and reports the text as unavailable.
        """

        view = workspace().jd_snapshot(
            user_id=principal.user_id, jd_snapshot_id=jd_snapshot_id
        )
        if view is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "JD_SNAPSHOT_NOT_FOUND",
                    "message": "该岗位已删除或不可访问。",
                },
            )
        return view

    @router.put(
        "/jobs/{job_posting_id}/availability", response_model=AvailabilityResponse
    )
    async def set_job_availability(
        job_posting_id: str,
        request: AvailabilityUpdate,
        principal: ApiKeyPrincipal = Depends(require_scope(WORKSPACE_WRITE)),
    ) -> AvailabilityResponse:
        """Record a closure by hand, for when the extension cannot.

        It fails in ordinary ways — the site changes its wording, the page is
        behind a login, the extension is not installed on this machine — and
        every one of them would otherwise leave a job the user knows is gone
        sitting in the shortlist as a live candidate.

        Reversible in both directions, because "unknown" and "active" are real
        answers too: a posting can be relisted, and a misclick should cost one
        click rather than being permanent.
        """

        changed = workspace().set_job_availability(
            user_id=principal.user_id,
            job_posting_id=job_posting_id,
            status=request.availability_status,
        )
        if not changed and workspace().job_detail(
            user_id=principal.user_id, job_posting_id=job_posting_id
        ) is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "SAVED_JOB_NOT_FOUND",
                    "message": "没有找到这个已保存职位。",
                },
            )
        return AvailabilityResponse(
            job_posting_id=job_posting_id,
            availability_status=request.availability_status,
            changed=changed,
        )

    @router.put("/jobs/{job_posting_id}/pursuit", response_model=PursuitStatusResponse)
    async def set_job_pursuit(
        job_posting_id: str,
        request: PursuitStatusUpdate,
        principal: ApiKeyPrincipal = Depends(require_scope(WORKSPACE_WRITE)),
    ) -> PursuitStatusResponse:
        """Rule a saved job out of the shortlist, or put it back.

        The only decision the shortlist cannot derive. Everything else about it
        follows from what is stored — saved, applied to, closed by the employer
        — but "I looked at this and I am not going to apply" exists nowhere
        except in the user's head until they say so.

        Idempotent: setting the status it already has reports ``changed:
        false`` rather than failing, because a double-click on a card is a
        double-click, not an error.
        """

        changed = workspace().set_job_pursuit(
            user_id=principal.user_id,
            job_posting_id=job_posting_id,
            status=request.pursuit_status,
        )
        if not changed and workspace().job_detail(
            user_id=principal.user_id, job_posting_id=job_posting_id
        ) is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "SAVED_JOB_NOT_FOUND",
                    "message": "没有找到这个已保存职位。",
                },
            )
        return PursuitStatusResponse(
            job_posting_id=job_posting_id,
            pursuit_status=request.pursuit_status,
            changed=changed,
        )

    @router.delete("/jobs/{job_posting_id}", response_model=JobDeletionResponse)
    async def delete_job(
        job_posting_id: str,
        principal: ApiKeyPrincipal = Depends(require_scope(WORKSPACE_WRITE)),
    ) -> JobDeletionResponse:
        """Permanently delete a library-only job and all of its stored JDs.

        This is intentionally different from dismissal: there is no deleted
        status and no restore path.  Applied jobs are refused because their JD
        snapshot is part of the application record's historical input.
        """

        result = workspace().delete_job(
            user_id=principal.user_id, job_posting_id=job_posting_id
        )
        if result == "has_application":
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "JOB_HAS_APPLICATION",
                    "message": "这个岗位已有投递记录，不能删除其 JD。",
                },
            )
        if result == "not_found":
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "SAVED_JOB_NOT_FOUND",
                    "message": "没有找到这个已保存职位。",
                },
            )
        return JobDeletionResponse(job_posting_id=job_posting_id)

    @router.get("/conversations", response_model=tuple[ConversationView, ...])
    async def conversations(
        principal: ApiKeyPrincipal = Depends(require_scope(WORKSPACE_READ)),
        limit: int = Query(default=50, ge=1, le=100),
    ) -> tuple[ConversationView, ...]:
        return workspace().conversations(user_id=principal.user_id, limit=limit)

    @router.get(
        "/conversations/{conversation_id}/messages",
        response_model=ConversationTranscriptResponse,
    )
    async def conversation_messages(
        conversation_id: str,
        principal: ApiKeyPrincipal = Depends(require_scope(WORKSPACE_READ)),
        limit: int = Query(default=200, ge=1, le=500),
    ) -> ConversationTranscriptResponse:
        return workspace().conversation_messages(
            user_id=principal.user_id,
            conversation_id=conversation_id,
            limit=limit,
        )

    @router.delete(
        "/conversations/{conversation_id}",
        response_model=ConversationDeletionResponse,
    )
    async def delete_conversation(
        conversation_id: str,
        principal: ApiKeyPrincipal = Depends(require_scope(WORKSPACE_WRITE)),
    ) -> ConversationDeletionResponse:
        """Forget chat/session content while retaining execution audit records."""
        if before_conversation_delete is not None:
            before_conversation_delete(principal.user_id, conversation_id)
        if not workspace().delete_conversation(
            user_id=principal.user_id,
            conversation_id=conversation_id,
        ):
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "CONVERSATION_NOT_FOUND",
                    "message": "没有找到这个会话。",
                },
            )
        return ConversationDeletionResponse(conversation_id=conversation_id)

    @router.get("/resumes", response_model=tuple[ResumeView, ...])
    async def resumes(
        principal: ApiKeyPrincipal = Depends(require_scope(WORKSPACE_READ)),
    ) -> tuple[ResumeView, ...]:
        return workspace().resumes(user_id=principal.user_id)

    @router.delete("/resumes/{resume_id}", response_model=CountResponse)
    async def delete_resume(
        resume_id: str,
        principal: ApiKeyPrincipal = Depends(require_scope(WORKSPACE_WRITE)),
    ) -> CountResponse:
        if not workspace().delete_resume(user_id=principal.user_id, resume_id=resume_id):
            raise HTTPException(status_code=404, detail="简历不存在。")
        return CountResponse(count=1)

    @router.get("/target-roles", response_model=tuple[TargetRoleView, ...])
    async def target_roles(
        principal: ApiKeyPrincipal = Depends(require_scope(WORKSPACE_READ)),
    ) -> tuple[TargetRoleView, ...]:
        return workspace().target_roles(user_id=principal.user_id)

    @router.post("/target-roles", response_model=TargetRoleView)
    async def create_target_role(
        request: TargetRoleCreateRequest,
        principal: ApiKeyPrincipal = Depends(require_scope(WORKSPACE_WRITE)),
    ) -> TargetRoleView:
        try:
            return workspace().create_target_role(
                user_id=principal.user_id,
                title=request.title,
            )
        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail={"code": "INVALID_TARGET_ROLE", "message": str(error)},
            ) from error

    @router.post("/resumes/import", response_model=ResumeImportResponse)
    async def import_resume(
        file: UploadFile = File(...),
        name: str | None = Form(default=None),
        resume_id: str | None = Form(default=None),
        target_role_id: str | None = Form(default=None),
        client_upload_id: str | None = Form(default=None, min_length=1, max_length=200),
        principal: ApiKeyPrincipal = Depends(require_scope(WORKSPACE_WRITE)),
        idempotency_key: str | None = Header(
            default=None,
            alias="Idempotency-Key",
            min_length=1,
            max_length=200,
        ),
    ) -> ResumeImportResponse:
        """Store one resume version, replaying the earlier result on a retry.

        ``Idempotency-Key`` (or the ``client_upload_id`` form field for clients
        that cannot set headers) is scoped to the authenticated user. Repeating
        it with the same file and form returns the version the first call
        stored; repeating it with a different request is refused with 409.
        """
        if idempotency_key is not None and client_upload_id is not None and idempotency_key != client_upload_id:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "INVALID_RESUME_IMPORT",
                    "message": "Idempotency-Key and client_upload_id disagree.",
                },
            )
        try:
            content = await file.read(MAX_RESUME_IMPORT_BYTES + 1)
            content, document_format = validate_resume_document(
                file.filename or "",
                content,
            )
            return workspace().import_resume(
                user_id=principal.user_id,
                content=content,
                document_format=document_format,
                name=name,
                resume_id=resume_id,
                target_role_id=target_role_id,
                idempotency_key=idempotency_key or client_upload_id,
            )
        except ResumeImportConflictError as error:
            raise HTTPException(
                status_code=409,
                detail={"code": "RESUME_IMPORT_CONFLICT", "message": str(error)},
            ) from error
        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail={"code": "INVALID_RESUME_IMPORT", "message": str(error)},
            ) from error
        finally:
            await file.close()

    @router.get(
        "/resumes/{resume_id}/versions/{version_id}/document",
        response_class=Response,
        responses={
            200: {
                "content": {
                    "application/pdf": {},
                    "text/plain; charset=utf-8": {},
                    "text/markdown; charset=utf-8": {},
                }
            }
        },
    )
    async def resume_version_document(
        resume_id: str = FastAPIPath(min_length=1, max_length=200),
        version_id: str = FastAPIPath(min_length=1, max_length=200),
        download: bool = Query(default=False),
        principal: ApiKeyPrincipal = Depends(require_scope(WORKSPACE_READ)),
    ) -> Response:
        """The original bytes of one version, for the owner to view or save.

        PDF is served inline so the browser can render it; text and Markdown
        are served as plain text (never as HTML) so a resume cannot script the
        page that previews it. ``?download=1`` switches to an attachment.
        """
        located = workspace().resume_version_document(
            user_id=principal.user_id,
            resume_id=resume_id,
            resume_version_id=version_id,
        )
        if located is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "code": "RESUME_VERSION_NOT_FOUND",
                    "message": "Resume version not found or does not belong to the current user.",
                },
            )
        document, resume_name, version = located
        media_type, extension = RESUME_DOCUMENT_MEDIA_TYPES[document.document_format]
        filename = resume_document_filename(
            resume_name, version_number=version.version_number, extension=extension
        )
        disposition = "attachment" if download else "inline"
        if document.document_format == "markdown" and not download:
            # Browsers download text/markdown; an inline preview wants text.
            media_type = RESUME_DOCUMENT_MEDIA_TYPES["text"][0]
        return Response(
            content=document.raw_bytes,
            media_type=media_type,
            headers={
                "Content-Disposition": content_disposition(disposition, filename),
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "private, no-store",
            },
        )

    @router.get("/email", response_model=EmailWorkspaceResponse)
    async def email(
        principal: ApiKeyPrincipal = Depends(require_scope(WORKSPACE_READ)),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> EmailWorkspaceResponse:
        return workspace().email(user_id=principal.user_id, limit=limit)

    @router.get("/calendar", response_model=CalendarWorkspaceResponse)
    async def calendar(
        principal: ApiKeyPrincipal = Depends(require_scope(WORKSPACE_READ)),
        month: str | None = Query(default=None, pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
        timezone_name: str = Query(
            default="Asia/Shanghai", alias="timezone", min_length=1, max_length=100
        ),
    ) -> CalendarWorkspaceResponse:
        try:
            if month is None and timezone_name == "Asia/Shanghai":
                return workspace().calendar(user_id=principal.user_id)
            return workspace().calendar(
                user_id=principal.user_id,
                month=month,
                timezone_name=timezone_name,
            )
        except (ValueError, ZoneInfoNotFoundError) as error:
            raise HTTPException(status_code=422, detail="Invalid calendar month or timezone") from error

    @router.get("/company-research", response_model=tuple[CompanyResearchView, ...])
    async def company_research(
        principal: ApiKeyPrincipal = Depends(require_scope(WORKSPACE_READ)),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> tuple[CompanyResearchView, ...]:
        return workspace().research(user_id=principal.user_id, limit=limit)

    @router.delete("/company-research/{report_id}", response_model=CountResponse)
    async def delete_company_research(
        report_id: str,
        principal: ApiKeyPrincipal = Depends(require_scope(WORKSPACE_WRITE)),
    ) -> CountResponse:
        if not workspace().delete_research(user_id=principal.user_id, report_id=report_id):
            raise HTTPException(status_code=404, detail="公司研究不存在。")
        return CountResponse(count=1)

    @router.get("/reports/{kind}/{resource_id}", response_model=ReportView)
    async def report(
        kind: str,
        resource_id: str,
        principal: ApiKeyPrincipal = Depends(require_scope(WORKSPACE_READ)),
        status_at_delivery: Literal["current", "outdated", "superseded"] | None = Query(default=None),
        anchored_by_other_job: bool | None = Query(default=None),
    ) -> ReportView:
        view = workspace().report(
            user_id=principal.user_id,
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
        principal: ApiKeyPrincipal = Depends(require_scope(WORKSPACE_READ)),
        timezone: str = Query(default="Asia/Shanghai", min_length=1, max_length=100),
    ) -> DashboardResponse:
        application_items = workspace().applications(user_id=principal.user_id, limit=500)
        job_items = workspace().jobs(user_id=principal.user_id, limit=6)
        saved_job_count = workspace().job_count(user_id=principal.user_id)
        resume_items = workspace().resumes(user_id=principal.user_id)
        research_items = workspace().research(user_id=principal.user_id, limit=500)
        calendar_items = workspace().calendar(user_id=principal.user_id)
        brief = DailyBriefResponse.of(
            action_center().daily_brief(user_id=principal.user_id, timezone_name=timezone)
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
