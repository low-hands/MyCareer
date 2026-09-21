from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import sqlite3
from typing import Any, Literal
from urllib.parse import urlencode

from career_agent.agent.delivered_body_contracts import (
    BodyDependency,
    MockInterviewBodySource,
    ResumeAnalysisBodySource,
)
from career_agent.agent.input_resources import (
    saved_job_description,
    saved_job_title,
)
from career_agent.agent.mock_interview_contracts import (
    MockInterviewGraphResult,
    MockInterviewQuestionSummary,
    MockInterviewResultView,
    MockInterviewStartRequest,
)
from career_agent.agent.interview_preparation_presenter import (
    summarize_interview_preparation,
)
from career_agent.agent.conversation_span_presenter import render_conversation_span
from career_agent.agent.job_research_presenter import summarize_job_research
from career_agent.agent.summary_text import clamp, condense
from career_agent.agent.mock_interview_presenter import (
    mock_interview_question_view,
    render_mock_interview_turn,
    summarize_mock_interview_question,
    summarize_mock_interview_report,
    summarize_mock_interview_result,
)
from career_agent.agent.mock_interview_graph import (
    MockInterviewCheckpointMissingError,
    MockInterviewGraph,
    MockInterviewGraphVersionError,
    MockInterviewInputRoutingError,
)
from career_agent.agent.main_agent_contracts import (
    AnalyzeResumeToolArguments,
    ConfirmResumeAnalysisToolArguments,
    CompleteInterviewToolArguments,
    RecordInterviewRetroToolArguments,
    ReadConversationSpanToolArguments,
    ResolveClaimSourceToolArguments,
    RouteToCapabilityToolArguments,
    GetCareerMemoryDetailToolArguments,
    SearchCareerMemoryToolArguments,
    SearchCareerEpisodesToolArguments,
    SearchCareerHistoryToolArguments,
    UpdateWorkingNotesToolArguments,
    ConstraintRetirementProposal,
    FetchArchivedConstraintsToolArguments,
    ProposeConstraintRetirementToolArguments,
    ProposeMemoryTombstoneToolArguments,
    MemoryTombstoneProposal,
    ProposeMemoryAmendmentToolArguments,
    MemoryAmendmentProposal,
    CareerFactProposal,
    PENDING_PROPOSAL_TTL,
    CONFIRMATION_SPECS,
    ProposeCareerFactToolArguments,
    ConfirmCareerFactToolArguments,
    PrepareInterviewToolArguments,
    GetInterviewPreparationToolArguments,
    CreateInterviewToolArguments,
    CreateApplicationToolArguments,
    DraftResumeTailoringToolArguments,
    ExportResumeArtifactToolArguments,
    FindSavedJobsToolArguments,
    FinalizeResumeTailoringToolArguments,
    GetApplicationToolArguments,
    GetDailyBriefToolArguments,
    GetMockInterviewResultToolArguments,
    RestartMockInterviewToolArguments,
    GetInterviewToolArguments,
    GetResumeMetadataToolArguments,
    GetResumeAnalysisToolArguments,
    GetResumeJobMatchToolArguments,
    GetResumeTailoringDraftToolArguments,
    GetSavedJobToolArguments,
    AnalyzeJobToolArguments,
    ResearchJobToolArguments,
    RetryJobResearchToolArguments,
    GetJobResearchToolArguments,
    CareerProfileContext,
    HardConstraintContext,
    JobIntentUpdate,
    CompareSavedJobsToolArguments,
    ConfirmJobIntentToolArguments,
    ConfirmFreeTextPreferenceToolArguments,
    ProposeFreeTextPreferenceConfirmationToolArguments,
    FreeTextPreferenceConfirmationProposal,
    ProposeJobIntentToolArguments,
    ListResumesToolArguments,
    ListApplicationsToolArguments,
    ListActionItemsToolArguments,
    ListCalendarAccountsToolArguments,
    ListCalendarLinksToolArguments,
    ListTargetRolesToolArguments,
    ListEmailEventsToolArguments,
    ListInterviewsToolArguments,
    MatchResumeToJobToolArguments,
    OpenJobSearchToolArguments,
    ReviewResumeTailoringToolArguments,
    ReviseResumeTailoringToolArguments,
    UpdateApplicationStatusToolArguments,
    UpdateOwnerSettingsToolArguments,
    ResolveEmailEventToolArguments,
    ResolveActionItemToolArguments,
    SnoozeActionItemToolArguments,
    StartMockInterviewToolArguments,
    StartMockInterviewWorkflowInput,
    PrepareInterviewCalendarSyncToolArguments,
    GetCalendarProposalToolArguments,
    ExecuteCalendarProposalToolArguments,
    SyncApplicationEmailsToolArguments,
    UpdateInterviewToolArguments,
    ToolObservation,
    ConversationResourceReference,
    DECISION_OBSERVATION_BODY_LIMIT,
    OwnerSettingsContext,
)
from career_agent.storage.context import CareerContextStore, OwnerSettingsConflictError
from career_agent.storage.career_history import CareerHistoryStore
from career_agent.storage.episodes import SQLiteCareerEpisodeStore
from career_agent.domain.career_history import career_evidence_lineage_ref
from career_agent.agent.openai_compatible_client import (
    AgentWorkerError,
    worker_failure_reason,
)
from career_agent.agent.tool_effects import effect_for
from career_agent.harness.observability import (
    conversation_trace_key,
    record_active_trace,
)
from career_agent.connectors.email_accounts import EmailCredentialError
from career_agent.connectors.gmail_readonly import GmailAPIError
from career_agent.services.resume_analysis import (
    ResumeAnalysisNotFoundError,
    ResumeAnalysisNotPendingError,
    ResumeAnalysisService,
    ResumeAnalysisWorkerNotCommittedError,
    ResumeVersionNotFoundError,
)
from career_agent.services.intent_capture import IntentCaptureCandidate
from career_agent.services.free_text_preferences import structured_pref_scope
from career_agent.storage.intent_versions import intent_entry_id


from career_agent.services.applications import (
    ApplicationInputNotFoundError,
    ApplicationService,
    ConcurrentApplicationUpdateError,
    InvalidApplicationTransitionError,
)
from career_agent.services.action_center import (
    ActionCenterService,
    ActionItemNotFoundError,
    InvalidActionTransitionError,
)
from career_agent.connectors.calendar import CalendarConnectorError
from career_agent.services.calendar import (
    CalendarAccountNotFoundError,
    CalendarProposalConflictError,
    CalendarProposalNotFoundError,
    CalendarService,
    CalendarSyncNotAvailableError,
)
from career_agent.services.email_tracking import (
    EmailAccountNotFoundError,
    EmailEventNotFoundError,
    EmailEventResolutionError,
    EmailTrackingService,
)
from career_agent.services.interviews import (
    InterviewApplicationConflictError,
    InterviewNotFoundError,
    InterviewService,
)
from career_agent.services.interview_preparation import (
    InterviewPreparationInputNotFoundError,
    InterviewPreparationNotAvailableError,
    InterviewPreparationService,
)
from career_agent.services.job_research import (
    JobResearchExecutionError,
    JobResearchInputNotFoundError,
    JobResearchRunNotRetryableError,
    JobResearchService,
)
from career_agent.agent.job_analysis_contracts import SENIORITY_LABELS
from career_agent.services.job_analysis import (
    JobAnalysisInputNotFoundError,
    JobAnalysisService,
)
from career_agent.services.resume_job_match import (
    ResumeJobMatchAnalysisRequiredError,
    ResumeJobMatchInputNotFoundError,
    ResumeJobMatchService,
)
from career_agent.services.resume_export import (
    ResumeExportNotFoundError,
    ResumeExportService,
)
from career_agent.domain.mock_interviews import (
    MockInterviewReport,
    MockInterviewSession,
    MockInterviewTurn,
)
from career_agent.domain.resume import ResumeArtifactDelivery
from career_agent.services.resume_tailoring import (
    ResumeFinalReviewBlockedError,
    ResumeTailoringAlreadyFinalizedError,
    ResumeTailoringDraftNotFoundError,
    ResumeTailoringNotReadyError,
    ResumeTailoringReviewBlockedError,
    ResumeTailoringService,
    ResumeTailoringSupersededError,
)
from career_agent.storage.job_captures import JobCaptureStore
from career_agent.storage.jobs import JobPostingRepository
from career_agent.storage.mock_interviews import SQLiteMockInterviewStore
from career_agent.storage.context import CareerProfileStore
from career_agent.services.job_comparison import (
    JobComparisonInputNotFoundError,
    JobComparisonService,
)
from career_agent.storage.resumes import ResumeStore
from career_agent.storage.resume_tailoring import StoredResumeTailoringDraft
from career_agent.domain.memory_scope import CanonicalScope, ScopeProposal
from career_agent.services.canonical_scope import CanonicalScopeResolver
from career_agent.agent.semantic_career_retrieval import SemanticEvidenceCache
from career_agent.storage.working_notes import (
    WorkingNotesConflict,
    WorkingNotesStore,
)


logger = logging.getLogger(__name__)


MainAgentToolOutput = ToolObservation
CapabilityKind = Literal["atomic_tool", "workflow"]


# Same state as a missing proposal: to the gate both mean nothing live to
# confirm. Only the wording differs, so the model re-shows instead of asking why.
_EXPIRED_PROPOSAL_MESSAGE = (
    f"这项提案已超过 {PENDING_PROPOSAL_TTL.days} 天未确认，已经失效；"
    "请重新展示提案并等待明确确认。"
)


def _proposal_observation(
    *,
    tool_name: str,
    state: str,
    message: str,
    proposal: Any,
    payload_key: str = "proposal",
    execution_outcome: Literal["committed", "not_committed", "unknown"] | None = None,
    exclude_none: bool = False,
) -> ToolObservation:
    """Expose the exact proposal that the reducer will bind to confirmation."""
    return ToolObservation(
        tool_name=tool_name,
        state=state,
        message=message,
        payload={
            payload_key: proposal.model_dump(mode="json", exclude_none=exclude_none)
        },
        execution_outcome=execution_outcome,
    )


class MainAgentToolRegistry:
    _BOSS_CITY_CODES = {
        "全国": "100010000",
        "北京": "101010100",
        "上海": "101020100",
        "广州": "101280100",
        "深圳": "101280600",
        "杭州": "101210100",
        "成都": "101270100",
        "南京": "101190100",
        "武汉": "101200100",
        "西安": "101110100",
        "苏州": "101190400",
        "天津": "101030100",
        "重庆": "101040100",
        "长沙": "101250100",
        "厦门": "101230200",
        "beijing": "101010100",
        "shanghai": "101020100",
        "guangzhou": "101280100",
        "shenzhen": "101280600",
        "hangzhou": "101210100",
        "chengdu": "101270100",
        "nanjing": "101190100",
        "wuhan": "101200100",
        "xian": "101110100",
        "xi'an": "101110100",
    }

    def __init__(
        self,
        *,
        job_repository: JobPostingRepository | None = None,
        job_comparison_service: JobComparisonService | None = None,
        job_analysis_service: JobAnalysisService | None = None,
        career_profile_store: CareerProfileStore | None = None,
        resume_store: ResumeStore | None = None,
        resume_analysis_service: ResumeAnalysisService | None = None,
        resume_job_match_service: ResumeJobMatchService | None = None,
        resume_tailoring_service: ResumeTailoringService | None = None,
        resume_export_service: ResumeExportService | None = None,
        application_service: ApplicationService | None = None,
        email_tracking_service: EmailTrackingService | None = None,
        interview_service: InterviewService | None = None,
        interview_preparation_service: InterviewPreparationService | None = None,
        action_center_service: ActionCenterService | None = None,
        calendar_service: CalendarService | None = None,
        mock_interview_graph: MockInterviewGraph | None = None,
        mock_interview_store: SQLiteMockInterviewStore | None = None,
        job_research_service: JobResearchService | None = None,
        owner_settings_store: CareerContextStore | None = None,
        conversation_store: CareerContextStore | None = None,
        career_history_store: CareerHistoryStore | None = None,
        episode_store: SQLiteCareerEpisodeStore | None = None,
        semantic_evidence_cache: SemanticEvidenceCache | None = None,
        working_notes_store: WorkingNotesStore | None = None,
        job_capture_store: JobCaptureStore | None = None,
    ) -> None:
        self._workflow_handlers: dict[str, Callable[[dict[str, Any]], MainAgentToolOutput]] = {}
        # Workflow continuations are runtime-owned capabilities. They share the
        # harness execution path with model tools, but are deliberately absent
        # from ``schemas()`` and ``names`` so the decision model can neither see
        # nor request them.
        self._runtime_workflow_handlers: dict[
            str, Callable[[dict[str, Any]], MainAgentToolOutput]
        ] = {}
        self._atomic_handlers: dict[str, Callable[[dict[str, Any]], ToolObservation]] = {
            "route_to_capability": self._route_to_capability,
            "open_job_search": self._open_job_search,
        }
        if career_profile_store is not None:
            self._atomic_handlers.update(
                {
                    "propose_job_intent": (
                        self._propose_job_intent
                    ),
                    "confirm_job_intent": (
                        self._confirm_job_intent
                    ),
                    "confirm_free_text_preference": (
                        self._confirm_free_text_preference
                    ),
                    "propose_free_text_preference_confirmation": (
                        self._propose_free_text_preference_confirmation
                    ),
                }
            )
        self._job_repository = job_repository
        self._job_comparison_service = job_comparison_service
        self._job_analysis_service = job_analysis_service
        self._career_profile_store = career_profile_store
        self._resume_store = resume_store
        self._resume_analysis_service = resume_analysis_service
        self._resume_job_match_service = resume_job_match_service
        self._resume_tailoring_service = resume_tailoring_service
        self._resume_export_service = resume_export_service
        self._application_service = application_service
        self._email_tracking_service = email_tracking_service
        self._interview_service = interview_service
        self._interview_preparation_service = interview_preparation_service
        self._action_center_service = action_center_service
        self._calendar_service = calendar_service
        self._mock_interview_graph = mock_interview_graph
        self._mock_interview_store = mock_interview_store
        self._job_research_service = job_research_service
        self._owner_settings_store = owner_settings_store
        self._conversation_store = conversation_store
        self._career_history_store = career_history_store
        self._episode_store = episode_store
        self._semantic_evidence_cache = semantic_evidence_cache
        self._working_notes_store = working_notes_store
        self._job_capture_store = job_capture_store
        self._canonical_scope_resolver = CanonicalScopeResolver()
        if conversation_store is not None:
            self._atomic_handlers["read_conversation_span"] = (
                self._read_conversation_span
            )
            self._atomic_handlers.update(
                {
                    "fetch_archived_constraints": (
                        self._fetch_archived_constraints
                    ),
                    "propose_constraint_retirement": (
                        self._propose_constraint_retirement
                    ),
                    "confirm_constraint_retirement": (
                        self._confirm_constraint_retirement
                    ),
                }
            )
        if episode_store is not None:
            self._atomic_handlers["search_career_episodes"] = (
                self._search_career_episodes
            )
        if career_history_store is not None:
            self._atomic_handlers.update(
                {
                    "resolve_claim_source": self._resolve_claim_source,
                    "get_career_memory_detail": self._get_career_memory_detail,
                    "search_career_memory": self._search_career_memory,
                    "search_career_history": self._search_career_history,
                    "propose_memory_tombstone": self._propose_memory_tombstone,
                    "confirm_memory_tombstone": self._confirm_memory_tombstone,
                    "propose_memory_amendment": self._propose_memory_amendment,
                    "confirm_memory_amendment": self._confirm_memory_amendment,
                    "propose_career_fact": self._propose_career_fact,
                    "confirm_career_fact": self._confirm_career_fact,
                }
            )
        if working_notes_store is not None:
            self._atomic_handlers["update_working_notes"] = self._update_working_notes
        if owner_settings_store is not None:
            self._atomic_handlers["update_owner_settings"] = self._update_owner_settings
        if job_repository is not None:
            self._atomic_handlers.update(
                {
                    "find_saved_jobs": self._find_saved_jobs,
                    "get_saved_job": self._get_saved_job,
                }
            )
        if job_comparison_service is not None:
            self._atomic_handlers["compare_saved_jobs"] = self._compare_saved_jobs
        if job_analysis_service is not None:
            self._atomic_handlers["analyze_job"] = self._analyze_job
        if job_research_service is not None:
            self._workflow_handlers.update(
                {
                    "research_job": self._research_job,
                    "retry_job_research": self._retry_job_research,
                }
            )
            self._atomic_handlers["get_job_research"] = self._get_job_research
        if resume_store is not None:
            self._atomic_handlers.update(
                {
                    "list_target_roles": self._list_target_roles,
                    "list_resumes": self._list_resumes,
                    "get_resume_metadata": self._get_resume_metadata,
                }
            )
        if resume_export_service is not None:
            self._atomic_handlers["export_resume_artifact"] = (
                self._export_resume_artifact
            )
        if resume_analysis_service is not None:
            self._atomic_handlers.update(
                {
                    "analyze_resume": self._analyze_resume,
                    "get_resume_analysis": self._get_resume_analysis,
                }
            )
        if resume_job_match_service is not None:
            self._atomic_handlers.update(
                {
                    "match_resume_to_job": self._match_resume_to_job,
                    "get_resume_job_match": self._get_resume_job_match,
                }
            )
        if resume_tailoring_service is not None:
            self._atomic_handlers.update(
                {
                    "draft_resume_tailoring": self._draft_resume_tailoring,
                    "get_resume_tailoring_draft": self._get_resume_tailoring_draft,
                    "review_resume_tailoring": self._review_resume_tailoring,
                    "revise_resume_tailoring": self._revise_resume_tailoring,
                    "finalize_resume_tailoring": self._finalize_resume_tailoring,
                }
            )
        if application_service is not None:
            self._atomic_handlers.update(
                {
                    "create_application": self._create_application,
                    "update_application_status": self._update_application_status,
                    "list_applications": self._list_applications,
                    "get_application": self._get_application,
                }
            )
        if email_tracking_service is not None:
            self._workflow_handlers["sync_application_emails"] = (
                self._sync_application_emails
            )
            self._atomic_handlers.update(
                {
                    "list_email_events": self._list_email_events,
                    "resolve_email_event": self._resolve_email_event,
                }
            )
        if interview_service is not None:
            self._atomic_handlers.update(
                {
                    "list_interviews": self._list_interviews,
                    "get_interview": self._get_interview,
                    "create_interview": self._create_interview,
                    "update_interview": self._update_interview,
                    "complete_interview": self._complete_interview,
                    "record_interview_retro": self._record_interview_retro,
                }
            )
        if interview_preparation_service is not None:
            self._atomic_handlers.update(
                {
                    "prepare_interview": self._prepare_interview,
                    "get_interview_preparation": self._get_interview_preparation,
                }
            )
        if mock_interview_graph is not None and application_service is not None:
            self._workflow_handlers["start_mock_interview"] = (
                self._start_mock_interview
            )
        if mock_interview_graph is not None:
            self._runtime_workflow_handlers.update(
                {
                    "handle_mock_interview_input": (
                        self._handle_mock_interview_input
                    ),
                    "retry_mock_interview": self._retry_mock_interview,
                }
            )
        if mock_interview_store is not None:
            self._atomic_handlers["get_mock_interview_result"] = (
                self._get_mock_interview_result
            )
        if mock_interview_graph is not None and mock_interview_store is not None:
            # A workflow, not an atomic tool: it starts a run that then owns
            # subsequent turns, exactly as start_mock_interview does.
            self._workflow_handlers["restart_mock_interview"] = (
                self._restart_mock_interview
            )
        if action_center_service is not None:
            self._atomic_handlers.update(
                {
                    "get_daily_brief": self._get_daily_brief,
                    "list_action_items": self._list_action_items,
                    "complete_action_item": self._complete_action_item,
                    "dismiss_action_item": self._dismiss_action_item,
                    "snooze_action_item": self._snooze_action_item,
                }
            )
        if calendar_service is not None:
            self._atomic_handlers.update(
                {
                    "list_calendar_accounts": self._list_calendar_accounts,
                    "list_calendar_links": self._list_calendar_links,
                    "prepare_interview_calendar_sync": self._prepare_interview_calendar_sync,
                    "get_calendar_proposal": self._get_calendar_proposal,
                    "execute_calendar_proposal": self._execute_calendar_proposal,
                }
            )

    @staticmethod
    def _resource_title(*parts: str | None) -> str:
        """Join producer-selected identity fields into one bounded title."""

        title = " · ".join(part.strip() for part in parts if part and part.strip())
        return condense(title, limit=80)

    @staticmethod
    def _resource_description(text: str) -> str | None:
        description = condense(text, limit=200)
        return description or None

    def _job_display(
        self, *, user_id: str, job_posting_id: str
    ) -> tuple[str, str] | None:
        """Read formal job names when that producer dependency is available."""

        get_job = getattr(self._job_repository, "get_job", None)
        if not callable(get_job):
            return None
        job = get_job(
            user_id=user_id, job_posting_id=job_posting_id
        )
        if job is None:
            return None
        return job.posting.company_name, job.posting.title

    def _resume_display(
        self, *, user_id: str, resume_version_id: str
    ) -> str | None:
        if self._resume_store is None:
            return None
        resolved = self._resume_store.get_version(
            user_id=user_id, resume_version_id=resume_version_id
        )
        if resolved is None:
            return None
        resume, version = resolved
        return f"{resume.name} v{version.version_number}"

    def _job_resource_metadata(
        self,
        *,
        user_id: str,
        job_posting_id: str,
        resource_name: str,
        description: str,
        prefix: str | None = None,
    ) -> tuple[str, str | None]:
        job = self._job_display(user_id=user_id, job_posting_id=job_posting_id)
        title = (
            self._resource_title(prefix, job[0], job[1], resource_name)
            if job is not None
            else self._resource_title(prefix, resource_name)
        )
        return title, self._resource_description(description)

    def _job_research_metadata(
        self, *, user_id: str, report
    ) -> tuple[str, str | None]:
        job = self._job_display(
            user_id=user_id, job_posting_id=report.job_posting_id
        )
        title = self._resource_title(
            job[0] if job is not None else report.company_key,
            None if (job is not None or report.company_key) else "岗位研究报告",
        )
        anchor = f"以 {job[1]} 岗位 JD 为检索锚点。" if job is not None else ""
        description = self._resource_description(
            f"公司调研；{anchor}{report.summary}"
        )
        return title, description

    def _resume_match_metadata(
        self, *, user_id: str, stored
    ) -> tuple[str, str | None]:
        job = self._job_display(
            user_id=user_id, job_posting_id=stored.job_posting_id
        )
        resume = self._resume_display(
            user_id=user_id, resume_version_id=stored.resume_version_id
        )
        identity = (
            f"{resume} × {job[0]} · {job[1]}"
            if resume and job
            else resume or (" · ".join(job) if job else None)
        )
        title = self._resource_title(identity, "简历岗位匹配")
        return title, self._resource_description(
            f"简历与岗位匹配；整体匹配度为 {stored.result.overall_fit}。"
        )

    def _tailoring_metadata(
        self, *, user_id: str, draft: StoredResumeTailoringDraft
    ) -> tuple[str, str | None]:
        match = None
        if self._resume_job_match_service is not None:
            try:
                match = self._resume_job_match_service.get_match(
                    user_id=user_id, match_id=draft.match_id
                )
            except ResumeJobMatchInputNotFoundError:
                pass
        if match is not None:
            job = self._job_display(
                user_id=user_id, job_posting_id=match.job_posting_id
            )
            resume = self._resume_display(
                user_id=user_id, resume_version_id=match.resume_version_id
            )
        else:
            job = None
            resume = None
        identity = (
            f"{resume} → {job[0]} · {job[1]}"
            if resume and job
            else resume or (" · ".join(job) if job else None)
        )
        title = self._resource_title(
            identity, f"简历改写稿 v{draft.revision_number}"
        )
        return title, self._resource_description(draft.result.strategy_summary)

    @property
    def names(self) -> tuple[str, ...]:
        return (*self.workflow_names, *self.atomic_tool_names)

    @property
    def workflow_names(self) -> tuple[str, ...]:
        return tuple(self._workflow_handlers)

    @property
    def atomic_tool_names(self) -> tuple[str, ...]:
        return tuple(self._atomic_handlers)

    @property
    def harness_action_names(self) -> tuple[str, ...]:
        """Durable transitions callable only through bound UI interactions."""

        if self._resume_analysis_service is None:
            return ()
        return ("confirm_resume_analysis", "reject_resume_analysis")

    @property
    def runtime_workflow_names(self) -> tuple[str, ...]:
        """Workflow entries callable by the harness but never by the model."""

        return tuple(self._runtime_workflow_handlers)

    def capability_kind(self, name: str) -> CapabilityKind:
        if name in self._workflow_handlers:
            return "workflow"
        if name in self._atomic_handlers:
            return "atomic_tool"
        raise ValueError(f"Unknown main-agent capability: {name}")

    @property
    def resume_store(self) -> ResumeStore | None:
        """The store that owns resume versions, for the runtime to verify inputs against."""
        return self._resume_store

    @property
    def job_repository(self) -> JobPostingRepository | None:
        """The repository that owns saved jobs, for the runtime to verify inputs against."""
        return self._job_repository

    def schemas(self) -> tuple[dict[str, Any], ...]:
        schemas: list[dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": "route_to_capability",
                    "description": (
                        "Expose a required tool outside the current profile by "
                        "switching to its domain: job (saved jobs, comparison, company "
                        "research), resume (analysis, job match, tailoring, "
                        "export), application (applications, status, email "
                        "events), interview (rounds, preparation, retro, calendar, "
                        "mock interview) or memory (career facts, preferences, "
                        "amendments, deletions). task.tool_profile shows the "
                        "current profile and task.available_now the tools usable "
                        "in it. Core tools are shared by every profile. A tool "
                        "already offered needs no route; its missing inputs or "
                        "approval must be supplied, not bypassed by routing. "
                        "Routing only changes the offered tool set, not business "
                        "data, evidence, or authority."
                    ),
                    "parameters": RouteToCapabilityToolArguments.model_json_schema(),
                },
            }
        ]
        if self._conversation_store is not None:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "read_conversation_span",
                        "description": (
                            "The projection's through_sequence is the last message "
                            "covered by conversation_summary, and recent_from_sequence "
                            "is the first raw recent message. Read an exact inclusive "
                            "sequence span from this same conversation only when those "
                            "boundaries leave a relevant gap. For long gaps, pass focused "
                            "query terms to search message content instead of walking spans "
                            "eight rows at a time. Without query it returns the oldest rows "
                            "in the exact span. Returns at most 8 matching messages, clips "
                            "each at 4000 characters, and reports "
                            "returned/total plus clipping honestly. It never "
                            "searches another conversation or substitutes nearby rows "
                            "when the requested span is empty."
                        ),
                        "parameters": (
                            ReadConversationSpanToolArguments.model_json_schema()
                        ),
                    },
                }
            )
            schemas.extend(
                (
                    {
                        "type": "function",
                        "function": {
                            "name": "fetch_archived_constraints",
                            "description": (
                                "Read the constraints this conversation recorded "
                                "but conversation_summary is not showing. Call "
                                "this when omitted_active_constraint_count is "
                                "above zero and the reply depends on which "
                                "constraints apply; an archived constraint still "
                                "applies. Read-only."
                            ),
                            "parameters": (
                                FetchArchivedConstraintsToolArguments.model_json_schema()
                            ),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "propose_constraint_retirement",
                            "description": (
                                "Prepare to stop applying one recorded "
                                "constraint, passing its exact text from "
                                "conversation_summary.active_constraints or from "
                                "fetch_archived_constraints. Use this only when "
                                "the user says a constraint no longer holds; "
                                "never to make room for a new one, and never "
                                "because a constraint looks stale. This only "
                                "reads the target and shows a bounded proposal."
                            ),
                            "parameters": (
                                ProposeConstraintRetirementToolArguments.model_json_schema()
                            ),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "confirm_constraint_retirement",
                            "description": (
                                "Execute the exact constraint retirement already "
                                "shown to the user. Call only after explicit "
                                "agreement. The constraint stops applying and "
                                "will not return even if a later summary "
                                "rewrite re-extracts the same text."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "additionalProperties": False,
                            },
                        },
                    },
                )
            )
        if self._episode_store is not None:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "search_career_episodes",
                        "description": (
                            "Search L1 memories of completed applications, job "
                            "research, interviews, and mock interviews across "
                            "conversations, or expand one projected detail_ref. "
                            "Supports an occurred-at window and episode-type "
                            "filters. Results are compact pointers "
                            "and synopses; dereference resource_refs before using "
                            "an episode as factual evidence."
                        ),
                        "parameters": (
                            SearchCareerEpisodesToolArguments.model_json_schema()
                        ),
                    },
                }
            )
        if self._working_notes_store is not None:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "update_working_notes",
                        "description": (
                            "Replace the complete per-user agent scratchpad. It is "
                            "unconfirmed and may guide questions or response style "
                            "only; never use it as a basis for filtering, ranking, "
                            "applications, scheduling, or authoritative state."
                        ),
                        "parameters": UpdateWorkingNotesToolArguments.model_json_schema(),
                    },
                }
            )
        if self._career_history_store is not None:
            schemas.extend(
                (
                    {
                    "type": "function",
                    "function": {
                        "name": "resolve_claim_source",
                        "description": (
                            "Read the original resume evidence for a confirmed "
                            "career claim only when its projected source_ref is "
                            "relevant to the user's request. The quotation is "
                            "returned as a bounded turn-local result."
                        ),
                        "parameters": (
                            ResolveClaimSourceToolArguments.model_json_schema()
                        ),
                    },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_career_memory_detail",
                            "description": (
                                "Expand one current career claim only when its "
                                "projected detail_ref is relevant. Returns the "
                                "current revision, direct support, and correction "
                                "lineage without treating an old quotation as "
                                "support for the corrected claim."
                            ),
                            "parameters": (
                                GetCareerMemoryDetailToolArguments.model_json_schema()
                            ),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "search_career_memory",
                            "description": (
                                "Search active confirmed career claims omitted "
                                "from the bounded Tier-1 window. Use this layered "
                                "archive fetch when career_profile.memory_overflow "
                                "names this tool; paginate with the returned cursor."
                            ),
                            "parameters": (
                                SearchCareerMemoryToolArguments.model_json_schema()
                            ),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "search_career_history",
                            "description": (
                                "Search superseded or rolled-back career claims "
                                "for historical questions. Pass focused terms "
                                "expected inside the earlier claim; this is the "
                                "query-indexed history path, not source_ref lookup. "
                                "When next_cursor is returned, pass it back with "
                                "the identical query to read the next page."
                            ),
                            "parameters": (
                                SearchCareerHistoryToolArguments.model_json_schema()
                            ),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "propose_career_fact",
                            "description": (
                                "Create a quarantined career-fact candidate for "
                                "one projected career record and read the exact "
                                "claim back to the user. Use only for an explicit "
                                "user statement; it is not active until confirmed. "
                                "Supply user_quote as an exact excerpt from the "
                                "user's message or questionnaire answer to mark "
                                "user_input provenance; omit it for inference."
                            ),
                            "parameters": (
                                ProposeCareerFactToolArguments.model_json_schema()
                            ),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "confirm_career_fact",
                            "description": (
                                "Confirm the exact quarantined career-fact proposal "
                                "shown on the preceding turn. Takes no arguments."
                            ),
                            "parameters": (
                                ConfirmCareerFactToolArguments.model_json_schema()
                            ),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "propose_memory_amendment",
                            "description": (
                                "Prepare a field-level correction for one current "
                                "career claim identified by detail_ref. This only "
                                "shows the exact replacement claim and reason; it "
                                "does not write a revision."
                            ),
                            "parameters": (
                                ProposeMemoryAmendmentToolArguments.model_json_schema()
                            ),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "confirm_memory_amendment",
                            "description": (
                                "Write the exact correction proposal already shown "
                                "to the user as a new revision. Call only after "
                                "explicit agreement."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "additionalProperties": False,
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "propose_memory_tombstone",
                            "description": (
                                "Prepare an irreversible field-level deletion for one "
                                "current career claim identified by detail_ref. This "
                                "only reads the target and shows a bounded proposal; "
                                "it never deletes anything."
                            ),
                            "parameters": (
                                ProposeMemoryTombstoneToolArguments.model_json_schema()
                            ),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "confirm_memory_tombstone",
                            "description": (
                                "Execute the exact memory deletion proposal already "
                                "shown to the user. Call only after explicit agreement; "
                                "the write redacts the complete claim lineage and is "
                                "not reversible."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "additionalProperties": False,
                            },
                        },
                    },
                )
            )
        if "open_job_search" in self._atomic_handlers:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "open_job_search",
                        "description": (
                            "Open a BOSS recruitment search page when the user asks to "
                            "find new jobs. If neither the profile nor a target role supplies "
                            "a city and the user did not give one this turn, ask for the city "
                            "instead of guessing or searching nationwide. This only constructs "
                            "a safe search URL for the client; it never reads results, "
                            "automates browsing, calls BOSS APIs, or saves a job. The user "
                            "browses normally and explicitly chooses which JD to save."
                        ),
                        "parameters": OpenJobSearchToolArguments.model_json_schema(),
                    },
                }
            )
        if self._career_profile_store is not None:
            schemas.extend(
                (
                    {
                        "type": "function",
                        "function": {
                            "name": "propose_free_text_preference_confirmation",
                            "description": (
                                "Show one numbered quarantined free-text preference "
                                "from the free_text_preferences Markdown block back to "
                                "the user and ask whether it should become a lasting "
                                "active preference. This never activates it and ends "
                                "the turn waiting for the user's answer."
                            ),
                            "parameters": (
                                ProposeFreeTextPreferenceConfirmationToolArguments.model_json_schema()
                            ),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "confirm_free_text_preference",
                            "description": (
                                "Promote the exact free-text preference previously "
                                "shown by propose_free_text_preference_confirmation. "
                                "Call only on a later turn after explicit agreement."
                            ),
                            "parameters": (
                                ConfirmFreeTextPreferenceToolArguments.model_json_schema()
                            ),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "propose_job_intent",
                            "description": (
                                "Show the user what would be recorded as their stated "
                                "job intent, without saving anything. Send only the "
                                "fields they just stated in their own words; never "
                                "infer a city, salary, or bracket from a job they "
                                "looked at or from anything you concluded. Salary, "
                                "experience, and education belong to one target role "
                                "and require target_role_selection_index from "
                                "list_target_roles, because a candidate pursuing two "
                                "tracks wants different numbers for each. A city sent "
                                "without a selection index is the person's default; "
                                "sent with one it overrides that default for that role "
                                "alone. Person-level hard constraints may use only "
                                "the declared work_arrangement, work_schedule, "
                                "or company_scale "
                                "relations and must preserve the user's own wording "
                                "(for example, 必须远程 or 不接受996); never infer one. "
                                "This tool records intent only: skill and "
                                "experience claims come from the resume, never from "
                                "being told."
                            ),
                            "parameters": (
                                ProposeJobIntentToolArguments.model_json_schema()
                            ),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "confirm_job_intent",
                            "description": (
                                "Save the update the user was just shown. Call it only "
                                "after they explicitly agree to that specific readback; "
                                "continuing the conversation is not agreement."
                            ),
                            "parameters": (
                                ConfirmJobIntentToolArguments.model_json_schema()
                            ),
                        },
                    },
                )
            )
        if self._job_comparison_service is not None:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "compare_saved_jobs",
                        "description": (
                            "Lay two or more saved jobs out side by side on fixed "
                            "dimensions, using only what is already on record. It "
                            "never runs a new resume match and never scores, "
                            "weights, or ranks the jobs: a dimension the data does "
                            "not answer is reported as unknown rather than filled "
                            "in. Use it when the user asks which saved jobs to "
                            "pursue or how they differ."
                        ),
                        "parameters": CompareSavedJobsToolArguments.model_json_schema(),
                    },
                }
            )
        if self._job_repository is not None:
            schemas.extend(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "find_saved_jobs",
                            "description": "Search only the current user's previously saved or viewed jobs. Use this for historical recall, not for discovering new online jobs. Results become numbered saved-job candidates; complete JD text stays outside the decision context.",
                            "parameters": FindSavedJobsToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_saved_job",
                            "description": "Read the selected or active saved job. Pass selection_index after find_saved_jobs, or omit it to use the active job. The complete JD is delivered outside the decision context.",
                            "parameters": GetSavedJobToolArguments.model_json_schema(),
                        },
                    },
                ]
            )
        if self._job_research_service is not None:
            schemas.extend(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "research_job",
                            "description": (
                                "Optional current public-web research for the selected or "
                                "active saved job. Call only when the user explicitly asks "
                                "to research company, product-line, business, market, "
                                "competitor, or related public context; never start it "
                                "automatically during matching, tailoring, application, or "
                                "interview workflows. A generic JD cannot establish what a "
                                "specific private team works on. Results are source-grounded "
                                "and persisted. If the user explicitly agrees to investigate "
                                "business clues they reported after an interview, pass only "
                                "those relevant clues as user_provided_context; they remain "
                                "unverified until supported by public sources."
                            ),
                            "parameters": ResearchJobToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "retry_job_research",
                            "description": (
                                "Resume the active failed job-research run from its "
                                "checkpoint. Use only after a retryable research failure "
                                "and an explicit user request to retry."
                            ),
                            "parameters": RetryJobResearchToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_job_research",
                            "description": (
                                "Read a persisted job-research report without running web "
                                "research again. When a completed tool result carries a "
                                "title and reference, match the requested company to that "
                                "same result and pass its exact reference; never borrow a "
                                "reference from an older chat resource or a differently "
                                "titled result. A grounded saved-job selection_index reads "
                                "that company's latest available report, not a specific "
                                "historical version. Looking up a saved job cannot recover "
                                "the identity of a missing historical report. If no selector "
                                "identifies the requested report, explain that it cannot "
                                "be read and ask the user to supply it. "
                                "Omit both only when the user actually means the active one."
                            ),
                            "parameters": GetJobResearchToolArguments.model_json_schema(),
                        },
                    },
                ]
            )
        if self._resume_store is not None:
            schemas.extend(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "list_target_roles",
                            "description": "List the current user's resume target-role categories as numbered candidates. Never returns resume document content.",
                            "parameters": ListTargetRolesToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "list_resumes",
                            "description": "List the current user's resume families, optionally filtered with target_role_selection_index from list_target_roles. Returns numbered safe metadata only; never returns resume document content.",
                            "parameters": ListResumesToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_resume_metadata",
                            "description": "Read a resume family selected by selection_index and list its immutable versions as numbered metadata. Never returns PDF, text, Markdown, extracted content, or file paths.",
                            "parameters": GetResumeMetadataToolArguments.model_json_schema(),
                        },
                    },
                ]
            )
        if self._resume_analysis_service is not None:
            schemas.extend(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "analyze_resume",
                            "description": "Analyze the content of one resume version in isolation and extract structured career-fact candidates for review. Requires a selected resume version, or the active latest version when selection_index is omitted. Structured candidates are delivered outside the decision context and are not career facts until explicitly confirmed.",
                            "parameters": AnalyzeResumeToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_resume_analysis",
                            "description": "Retrieve the active unexpired resume analysis draft so its candidates can be reviewed before confirmation. Never returns the original resume file.",
                            "parameters": GetResumeAnalysisToolArguments.model_json_schema(),
                        },
                    },
                ]
            )
        if self._job_analysis_service is not None:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "analyze_job",
                        "description": "Analyze one saved job's complete JD on its own: core objective, inferred seniority, S/A/B/C tiered requirements with JD quotes, core competencies, implicit requirements, ATS keywords, HR / hiring-manager focus, likely interview topics, red flags, and information gaps. Pass selection_index after find_saved_jobs, or omit it to use the active job. Reads only the JD text: no resume, no preferences, no online research. Use match_resume_to_job instead when the user wants a comparison against a resume.",
                        "parameters": AnalyzeJobToolArguments.model_json_schema(),
                    },
                }
            )
        if self._resume_job_match_service is not None:
            schemas.extend(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "match_resume_to_job",
                            "description": "Compare an exact current-user resume version with one saved job's complete JD. Requires both objects and reads their source documents itself. Use resume_version_selection_index and job_selection_index to choose directly from existing candidates, or omit either selector to use its active object. Returns a grounded assessment outside the decision context; does not search online and never returns either original document.",
                            "parameters": MatchResumeToJobToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_resume_job_match",
                            "description": "Retrieve the current conversation's active persisted resume-job match. Returns only the structured assessment, never the original resume or complete JD.",
                            "parameters": GetResumeJobMatchToolArguments.model_json_schema(),
                        },
                    },
                ]
            )
        if self._resume_tailoring_service is not None:
            schemas.extend(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "draft_resume_tailoring",
                            "description": "Create a reviewable tailoring draft from the active persisted resume-job match. May accept a user tailoring goal. Grounded proposed changes are delivered outside decision context; this does not alter or create a resume version.",
                            "parameters": DraftResumeTailoringToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_resume_tailoring_draft",
                            "description": "Retrieve the active unexpired tailoring draft. Returns proposed changes for review outside decision context; it does not apply them.",
                            "parameters": GetResumeTailoringDraftToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "review_resume_tailoring",
                            "description": "Accept or reject specific 1-based change indices in an active tailoring draft. Use only decisions the user explicitly made; never infer acceptance from vague approval. Decisions are persisted and may be completed across turns. This does not create a new resume version.",
                            "parameters": ReviewResumeTailoringToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "revise_resume_tailoring",
                            "description": "Create a new child draft from the active tailoring draft using explicit user feedback. The new draft discards all prior accept/reject decisions, reruns the bounded Writer/Reviewer loop, and must be reviewed again. It never overwrites the parent draft or creates a ResumeVersion.",
                            "parameters": ReviseResumeTailoringToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "finalize_resume_tailoring",
                            "description": "Create one new immutable Markdown ResumeVersion from the explicitly accepted changes in a fully reviewed tailoring draft. Call only when the user explicitly asks to generate/save the new version after reviewing every change. Repeated calls are idempotent. Never use vague approval as authorization.",
                            "parameters": FinalizeResumeTailoringToolArguments.model_json_schema(),
                        },
                    },
                ]
            )
        if self._resume_export_service is not None:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "export_resume_artifact",
                        "description": "Prepare the active owned immutable resume version for download and return only an opaque artifact reference plus safe file metadata. Call only when the user asks to download, export, or receive the resume file. Never place file content or a local path in the conversation.",
                        "parameters": ExportResumeArtifactToolArguments.model_json_schema(),
                    },
                }
            )
        if self._application_service is not None:
            schemas.extend(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "create_application",
                            "description": "Track a real externally submitted application using one exact owned resume version and the saved job's current immutable JD snapshot. Use job_selection_index or resume_version_selection_index to override the active objects. Call only after the user explicitly reports that they actually applied; planning or preparing is not sufficient. Repeated calls for the same job return the original application.",
                            "parameters": CreateApplicationToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "update_application_status",
                            "description": "Update the active or specified application to a valid pipeline status, or add a note by supplying the unchanged status with a note. Use only status changes or facts explicitly supplied by the user.",
                            "parameters": UpdateApplicationStatusToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "list_applications",
                            "description": "List the current user's tracked applications, optionally filtered by pipeline statuses. Returns safe job and application metadata, not resume or JD contents.",
                            "parameters": ListApplicationsToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_application",
                            "description": "Read a tracked application selected by selection_index, or use the active application, including its append-only event timeline.",
                            "parameters": GetApplicationToolArguments.model_json_schema(),
                        },
                    },
                ]
            )
        if self._owner_settings_store is not None:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "update_owner_settings",
                        "description": (
                            "Propose a persistent owner setting change. The runtime "
                            "always stops this call and shows the exact change to the "
                            "owner; it takes effect only after the owner confirms the "
                            "bound interaction. Never claim it changed before confirmation."
                        ),
                        "parameters": UpdateOwnerSettingsToolArguments.model_json_schema(),
                    },
                }
            )
        if self._email_tracking_service is not None:
            schemas.extend(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "sync_application_emails",
                            "description": "Run the read-only Gmail/QQ recruiting-email synchronization workflow. It fetches metadata first, reads only candidate bodies in an isolated worker, links events to tracked applications, and returns safe summaries. Use when the user asks to check or refresh employer email progress.",
                            "parameters": SyncApplicationEmailsToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "list_email_events",
                            "description": "List safe structured recruiting-email events, optionally only those awaiting confirmation. Never returns email bodies or credentials.",
                            "parameters": ListEmailEventsToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "resolve_email_event",
                            "description": "Approve or dismiss one pending email event. Approval may explicitly correct its application link and updates the application only when the transition is valid. Use only after clear user confirmation.",
                            "parameters": ResolveEmailEventToolArguments.model_json_schema(),
                        },
                    },
                ]
            )
        if self._interview_service is not None:
            schemas.extend(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "list_interviews",
                            "description": "List real interview appointments, optionally for one application or by status. sequence_number is only the system's chronological appointment number, not an employer-confirmed round label.",
                            "parameters": ListInterviewsToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_interview",
                            "description": "Read one interview appointment and its append-only invitation, reschedule, detail-update, cancellation, and completion history.",
                            "parameters": GetInterviewToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "create_interview",
                            "description": "Create a user-reported real interview appointment for an application. employer_label may be provided only when the employer explicitly used that label; never infer 一面/二面 from sequence.",
                            "parameters": CreateInterviewToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "update_interview",
                            "description": "Apply an explicitly user-reported reschedule, added detail, or cancellation to one existing interview. A reschedule updates the same appointment rather than creating another one.",
                            "parameters": UpdateInterviewToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "complete_interview",
                            "description": "Mark one real interview completed only after the user explicitly confirms they attended it. Time passing alone is never confirmation.",
                            "parameters": CompleteInterviewToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "record_interview_retro",
                            "description": (
                                "Create a versioned post-interview report for one "
                                "user-confirmed completed real interview. Use only facts "
                                "the user just provided: preserve their source notes, "
                                "structure remembered questions and answers, and put "
                                "missing information in limitations. Never invent "
                                "interviewer feedback or present self-assessment as the "
                                "employer's decision."
                            ),
                            "parameters": RecordInterviewRetroToolArguments.model_json_schema(),
                        },
                    },
                ]
            )
        if self._interview_preparation_service is not None:
            schemas.extend(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "prepare_interview",
                            "description": "Generate or reuse a grounded preparation guide for one real upcoming interview from its exact JD snapshot, submitted resume version, confirmed evidence, and logistics. This is preparation, not a mock interview and not employer inside information.",
                            "parameters": PrepareInterviewToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_interview_preparation",
                            "description": (
                                "Read one persisted interview preparation result without "
                                "re-reading full source documents. Pass reference to "
                                "read the material a specific earlier message produced, or "
                                "omit it to use the active preparation."
                            ),
                            "parameters": GetInterviewPreparationToolArguments.model_json_schema(),
                        },
                    },
                ]
            )
        if self._mock_interview_graph is not None and self._application_service is not None:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "start_mock_interview",
                        "description": (
                            "Start one stateful mock interview for the active or numbered "
                            "application, using its exact submitted resume and immutable JD. "
                            "Optionally bind a numbered real interview appointment for context. "
                            "After the first question, user answers are routed directly to the "
                            "active mock-interview workflow; do not call this tool again to answer."
                        ),
                        "parameters": StartMockInterviewToolArguments.model_json_schema(),
                    },
                }
            )
        if self._mock_interview_graph is not None and self._mock_interview_store is not None:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "restart_mock_interview",
                        "description": (
                            "Retire a mock interview that reported "
                            "mock_interview_checkpoint_missing or "
                            "mock_interview_graph_incompatible and start a replacement "
                            "against the same application, resume version, and JD. Call "
                            "this only after the user agrees to abandon the stuck run: "
                            "its answers stay readable but it can never be finished. "
                            "Takes no arguments."
                        ),
                        "parameters": RestartMockInterviewToolArguments.model_json_schema(),
                    },
                }
            )
        if self._mock_interview_store is not None:
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": "get_mock_interview_result",
                        "description": (
                            "Read back a finished mock interview. Pass reference to "
                            "read the exact run a specific earlier message reported, "
                            "application_selection_index for the latest run on a numbered "
                            "application, or omit both for the latest run on the active "
                            "application. Without a question number this lists the "
                            "questions with their ratings; with one it returns that question, "
                            "the user's full answer, its evaluation, and any follow-ups. Use "
                            "this whenever the user asks about a past mock interview, since "
                            "the conversation only keeps a reference to the report."
                        ),
                        "parameters": GetMockInterviewResultToolArguments.model_json_schema(),
                    },
                }
            )
        if self._action_center_service is not None:
            schemas.extend(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_daily_brief",
                            "description": "Generate the current user's source-grounded daily career brief from applications, recruiting email events, real interviews, and unresolved resume-tailoring gaps. The report is computed on demand and is not stored as stale narrative memory.",
                            "parameters": GetDailyBriefToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "list_action_items",
                            "description": "Refresh and list persisted career action items such as follow-ups, pending email confirmations, interview preparation, reminders, material requests, and retrospectives.",
                            "parameters": ListActionItemsToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "complete_action_item",
                            "description": "Mark one action item completed only after the user explicitly reports completing it.",
                            "parameters": ResolveActionItemToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "dismiss_action_item",
                            "description": "Dismiss one action item only after the user explicitly says it is not applicable or should be ignored.",
                            "parameters": ResolveActionItemToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "snooze_action_item",
                            "description": "Snooze one action item until an explicit future timestamp requested by the user.",
                            "parameters": SnoozeActionItemToolArguments.model_json_schema(),
                        },
                    },
                ]
            )
        if self._calendar_service is not None:
            schemas.extend(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "list_calendar_accounts",
                            "description": "List safe Google Calendar account metadata. Credentials are never returned.",
                            "parameters": ListCalendarAccountsToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "list_calendar_links",
                            "description": "List the user's interview-to-calendar synchronization links and current sync status.",
                            "parameters": ListCalendarLinksToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "prepare_interview_calendar_sync",
                            "description": "Prepare a fixed create, update, or cancel preview for one real InterviewRound. This does not write to an external calendar and must be shown to the user for approval.",
                            "parameters": PrepareInterviewCalendarSyncToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_calendar_proposal",
                            "description": "Read one pending or historical fixed calendar-change proposal without executing it.",
                            "parameters": GetCalendarProposalToolArguments.model_json_schema(),
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "execute_calendar_proposal",
                            "description": (
                                "Execute exactly one unchanged, unexpired calendar proposal "
                                "only after the user explicitly approves that displayed "
                                "proposal. This is an external write. A failed or outcome-"
                                "unknown execution must not be repeated or claimed successful; "
                                "reconcile it, then prepare and approve a new preview."
                            ),
                            "parameters": ExecuteCalendarProposalToolArguments.model_json_schema(),
                        },
                    },
                ]
            )
        # Keep the model-facing tool prefix byte-stable across task-state
        # changes. Runtime projection below remains the capability boundary:
        # an unmet precondition produces a bounded soft refusal instead of
        # making the schema array churn from one turn to the next.
        return tuple(self._decision_tool_schema(schema) for schema in schemas)

    @staticmethod
    def _decision_tool_schema(schema: dict[str, Any]) -> dict[str, Any]:
        """Remove internal identifiers from the model-callable contract.

        Handlers still receive injected identifiers after argument projection;
        the decision model can only choose indexes already present in its task
        context or rely on the active object.
        """

        projected = deepcopy(schema)

        def scrub(node: object) -> None:
            if isinstance(node, dict):
                properties = node.get("properties")
                if isinstance(properties, dict):
                    for key in tuple(properties):
                        if key.endswith("_id"):
                            del properties[key]
                required = node.get("required")
                if isinstance(required, list):
                    node["required"] = [
                        key for key in required if not str(key).endswith("_id")
                    ]
                for value in node.values():
                    scrub(value)
            elif isinstance(node, list):
                for value in node:
                    scrub(value)

        scrub(projected["function"]["parameters"])
        return projected

    def invoke_workflow(self, name: str, arguments: dict[str, Any]) -> MainAgentToolOutput:
        handler = self._workflow_handlers.get(name)
        if handler is None:
            raise ValueError(f"Unknown main-agent workflow: {name}")
        return self._require_write_execution_outcome(name, handler(arguments))

    def invoke_runtime_workflow(
        self, name: str, arguments: dict[str, Any]
    ) -> MainAgentToolOutput:
        handler = self._runtime_workflow_handlers.get(name)
        if handler is None:
            raise ValueError(f"Unknown runtime-owned workflow: {name}")
        return self._require_write_execution_outcome(name, handler(arguments))

    @staticmethod
    def _require_write_execution_outcome(
        name: str, result: MainAgentToolOutput
    ) -> MainAgentToolOutput:
        """Reject a WRITE whose producer omitted the independent effect axis."""

        if effect_for(name) == "WRITE" and result.execution_outcome is None:
            raise ValueError(
                f"WRITE capability {name!r} returned without execution_outcome"
            )
        return result

    def _sync_application_emails(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._email_tracking_service is None:
            raise ValueError("Email tracking service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = SyncApplicationEmailsToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        try:
            result = self._email_tracking_service.sync(
                user_id=user_id,
                account_id=model_arguments.account_id,
            )
        except EmailAccountNotFoundError:
            return ToolObservation(
                tool_name="sync_application_emails",
                state="email_account_not_found",
                message="没有找到可用的邮箱账号，请先连接 Gmail 或 QQ 邮箱。",
                execution_outcome="not_committed",
            )
        except (EmailCredentialError, GmailAPIError, RuntimeError, AgentWorkerError) as error:
            return ToolObservation(
                tool_name="sync_application_emails",
                state="failed",
                message="邮箱同步暂时失败；邮箱内容、凭证和错误响应均未进入 Main Agent 上下文。",
                payload={
                    "error_code": (
                        error.code
                        if isinstance(error, AgentWorkerError)
                        else type(error).__name__
                    ),
                    "retryable": isinstance(error, (GmailAPIError, AgentWorkerError)),
                },
                # Sync persists messages and events incrementally. An error can
                # arrive after an earlier message or account was committed.
                execution_outcome="unknown",
            )
        pending = [event for event in result.events_created if event.status == "pending_confirmation"]
        return ToolObservation(
            tool_name="sync_application_emails",
            state="email_events_pending" if pending else "email_sync_complete",
            message=(
                f"已同步 {result.accounts_synced} 个邮箱，检查 {result.messages_seen} 封新邮件，"
                f"识别 {len(result.events_created)} 个求职事件，其中 {len(pending)} 个需要确认。"
            ),
            next_action=(
                "这些事件要用户逐条确认，别替他决定哪些算数。"
                if pending
                else None
            ),
            payload={
                "accounts_synced": result.accounts_synced,
                "messages_seen": result.messages_seen,
                "candidate_messages": result.candidate_messages,
                "events": [self._email_event_payload(event) for event in result.events_created],
            },
            execution_outcome="committed",
        )

    def _start_mock_interview(
        self, arguments: dict[str, Any]
    ) -> ToolObservation:
        if self._mock_interview_graph is None or self._application_service is None:
            raise ValueError("Mock interview workflow is not configured")
        workflow_input = StartMockInterviewWorkflowInput.model_validate(arguments)
        try:
            detail = self._application_service.get_application(
                user_id=workflow_input.user_id,
                application_id=workflow_input.application_id,
            )
            application = detail.application
            if workflow_input.interview_round_id is not None:
                if self._interview_service is None:
                    raise ValueError("Interview service is not configured")
                interview = self._interview_service.get_interview(
                    user_id=workflow_input.user_id,
                    interview_round_id=workflow_input.interview_round_id,
                ).interview
                if interview.application_id != application.id:
                    raise InterviewApplicationConflictError(
                        "mock interview appointment belongs to another application"
                    )
            result = self._mock_interview_graph.start(
                MockInterviewStartRequest(
                    user_id=workflow_input.user_id,
                    application_id=application.id,
                    interview_round_id=workflow_input.interview_round_id,
                    job_posting_id=application.job_posting_id,
                    jd_snapshot_id=application.jd_snapshot_id,
                    resume_version_id=application.resume_version_id,
                    interview_type=workflow_input.interview_type,
                    max_primary_questions=workflow_input.max_primary_questions,
                    max_follow_ups_per_question=(
                        workflow_input.max_follow_ups_per_question
                    ),
                )
            )
        except (ApplicationInputNotFoundError, AgentWorkerError, ValueError) as error:
            return ToolObservation(
                tool_name="start_mock_interview",
                state="failed",
                message="模拟面试暂时无法启动；请确认投递记录和对应材料仍然可用。",
                payload={
                    "error_code": (
                        error.code
                        if isinstance(error, AgentWorkerError)
                        else type(error).__name__
                    ),
                    "retryable": (
                        error.retryable
                        if isinstance(error, AgentWorkerError)
                        else False
                    ),
                },
                # Missing application input is rejected before graph start.
                # Worker failure happens after graph start has created (and
                # then cancelled) a durable session. A generic ValueError can
                # arise on either side of that boundary.
                execution_outcome=(
                    "not_committed"
                    if isinstance(error, ApplicationInputNotFoundError)
                    else "committed"
                    if isinstance(error, AgentWorkerError)
                    else "unknown"
                ),
            )
        return self._mock_interview_observation(
            result, "start_mock_interview", user_id=workflow_input.user_id
        )

    def resume_mock_interview(
        self, *, user_id: str, session_id: str, answer: str
    ) -> ToolObservation:
        """Resume the active workflow through an internal, non-model-facing path."""
        if self._mock_interview_graph is None:
            raise ValueError("Mock interview workflow is not configured")
        return self._drive_mock_interview(
            tool_name="resume_mock_interview",
            user_id=user_id,
            session_id=session_id,
            drive=lambda graph: graph.resume(
                user_id=user_id,
                session_id=session_id,
                answer=answer,
            ),
        )

    def handle_mock_interview_input(
        self, *, user_id: str, session_id: str, message: str
    ) -> ToolObservation:
        """Let the isolated workflow classify and consume one local message."""
        if self._mock_interview_graph is None:
            raise ValueError("Mock interview workflow is not configured")
        return self._drive_mock_interview(
            tool_name="handle_mock_interview_input",
            user_id=user_id,
            session_id=session_id,
            drive=lambda graph: graph.handle_input(
                user_id=user_id,
                session_id=session_id,
                message=message,
            ),
        )

    def _handle_mock_interview_input(
        self, arguments: dict[str, Any]
    ) -> ToolObservation:
        return self.handle_mock_interview_input(
            user_id=str(arguments["user_id"]),
            session_id=str(arguments["session_id"]),
            message=str(arguments["message"]),
        )

    def retry_mock_interview(
        self, *, user_id: str, session_id: str
    ) -> ToolObservation:
        """Re-drive a failed step from the answer the store already holds.

        The failure happened after the answer was persisted, so asking the
        candidate to retype it verbatim is both hostile and fragile: any
        rewording is rejected as a conflicting answer for the same turn. This
        path takes no answer argument at all, which is what makes recovery
        independent of what the candidate can remember.
        """
        if self._mock_interview_graph is None:
            raise ValueError("Mock interview workflow is not configured")
        return self._drive_mock_interview(
            tool_name="retry_mock_interview",
            user_id=user_id,
            session_id=session_id,
            drive=lambda graph: graph.retry(
                user_id=user_id,
                session_id=session_id,
            ),
        )

    def _retry_mock_interview(self, arguments: dict[str, Any]) -> ToolObservation:
        return self.retry_mock_interview(
            user_id=str(arguments["user_id"]),
            session_id=str(arguments["session_id"]),
        )

    def _drive_mock_interview(
        self,
        *,
        tool_name: str,
        user_id: str,
        session_id: str,
        drive: Callable[[Any], MockInterviewGraphResult],
    ) -> ToolObservation:
        """Run one graph advance and map its failures to a closed observation.

        ``tool_name`` is the entry that was actually called, not the capability
        that started the run. Every branch here reports it, because this string
        is what ``_tool_observation`` puts in front of the decision model: a
        hardcoded ``start_mock_interview`` told the model a call had been made
        that never happened, on the very turn its answer was being consumed.
        """
        try:
            result = drive(self._mock_interview_graph)
        except MockInterviewInputRoutingError as error:
            return ToolObservation(
                tool_name=tool_name,
                state="mock_interview_input_retry_required",
                message=(
                    "暂时无法判断这条消息是面试回答还是退出请求。"
                    "这条消息尚未保存，请重新发送。"
                ),
                next_action="这条消息没有被保存，也没有进入面试记录。请用户重发一次即可。",
                payload={
                    "session_id": session_id,
                    "error_code": error.code,
                    "retryable": error.retryable,
                },
                execution_outcome="not_committed",
            )
        except MockInterviewCheckpointMissingError:
            return ToolObservation(
                tool_name=tool_name,
                state="mock_interview_checkpoint_missing",
                message=(
                    "模拟面试的业务记录仍在，但执行断点已经丢失，当前会话无法继续。"
                ),
                next_action="这场面试已经无法恢复，重试同一个调用不会有用。只能重新开一场。",
                payload={"session_id": session_id, "retryable": False},
                execution_outcome="not_committed",
            )
        except MockInterviewGraphVersionError:
            return ToolObservation(
                tool_name=tool_name,
                state="mock_interview_graph_incompatible",
                message=(
                    "这次模拟面试由不兼容的旧版流程创建，不能用当前版本安全恢复。"
                ),
                next_action="这场面试已经无法恢复，重试同一个调用不会有用。只能重新开一场。",
                payload={"session_id": session_id, "retryable": False},
                execution_outcome="not_committed",
            )
        except (AgentWorkerError, ValueError) as error:
            return ToolObservation(
                tool_name=tool_name,
                state="failed",
                message=(
                    "这次模拟面试回答暂时无法处理。你的回答已经保存，"
                    "下一条消息会从保存的回答继续，不需要重新输入。"
                ),
                payload={
                    "session_id": session_id,
                    "error_code": (
                        error.code
                        if isinstance(error, AgentWorkerError)
                        else type(error).__name__
                    ),
                    "retryable": (
                        error.retryable
                        if isinstance(error, AgentWorkerError)
                        else False
                    ),
                },
                # Evaluation failures happen after the answer is persisted.
                # ValueError can also arise on either side of that boundary,
                # so the producer cannot safely claim no write occurred.
                execution_outcome=(
                    "committed" if isinstance(error, AgentWorkerError) else "unknown"
                ),
            )
        return self._mock_interview_observation(result, tool_name, user_id=user_id)

    def _mock_interview_observation(
        self,
        result: MockInterviewGraphResult,
        tool_name: str,
        *,
        user_id: str,
    ) -> ToolObservation:
        state = {
            "awaiting_answer": "mock_interview_answer_required",
            "running": "mock_interview_running",
            "completed": "mock_interview_completed",
            "cancelled": "mock_interview_cancelled",
        }[result.state]
        # Both strings come from the mock interview's own presenter rather than
        # being composed here: how an interview reads is not this layer's
        # concern. ``message`` is the line the durable conversation row keeps,
        # so a finished run is condensed to its headline instead of carrying the
        # whole report into every later turn's window; the full report reaches
        # the screen through the runtime presenter.
        message = (
            summarize_mock_interview_report(result.report)
            if result.state == "completed" and result.report is not None
            else render_mock_interview_turn(result)
        )
        session = (
            self._mock_interview_store.get_session(
                user_id=user_id, session_id=result.session_id
            )
            if self._mock_interview_store is not None
            else None
        )
        job = (
            self._job_display(
                user_id=user_id, job_posting_id=session.job_posting_id
            )
            if session is not None
            else None
        )
        title = self._resource_title(
            *(job or ()),
            "模拟面试报告",
        )
        description = self._resource_description(
            result.report.summary if result.report is not None else "模拟面试报告"
        )
        return ToolObservation(
            tool_name=tool_name,
            state=state,
            message=message,
            execution_outcome="committed",
            payload=result.model_dump(mode="json"),
            resource_ref=(
                # ``completed`` now guarantees a report: the graph raises rather
                # than projecting one without it, so the second half of this
                # condition could only ever have hidden that failure. The state
                # test stays because this builder also serves the running,
                # awaiting and cancelled results, which have no report at all.
                ConversationResourceReference(
                    kind="mock_interview_report",
                    resource_id=result.report_id,
                    title=title,
                    description=description,
                )
                if result.state == "completed"
                else None
            ),
        )

    def _restart_mock_interview(self, arguments: dict[str, Any]) -> ToolObservation:
        """Retire a run that cannot continue and start a fresh one in its place.

        Only reachable for the two phases that hold the slot without being able
        to advance. The store allows one unfinished run per user, so without
        retiring the stuck one first a new interview cannot be created at all.
        The replacement reuses the stuck run's own application, resume version,
        and JD snapshot, so a restart cannot silently change what is being
        practised against.
        """
        if self._mock_interview_graph is None:
            raise ValueError("Mock interview workflow is not configured")
        if self._mock_interview_store is None:
            raise ValueError("Mock interview store is not configured")
        user_id = str(arguments["user_id"])
        stuck = self._mock_interview_store.find_resumable(user_id=user_id)
        if stuck is None:
            return ToolObservation(
                tool_name="restart_mock_interview",
                state="no_mock_interview_to_restart",
                message="没有卡住的模拟面试，直接开始一场新的即可。",
                execution_outcome="not_committed",
            )
        # Cancel through the graph, not the store: the graph also deletes the
        # checkpoint thread, and for a version-incompatible run that thread is
        # the only thing still holding the old graph's state.
        self._mock_interview_graph.cancel(user_id=user_id, session_id=stuck.id)
        try:
            result = self._mock_interview_graph.start(
                MockInterviewStartRequest(
                    user_id=user_id,
                    application_id=stuck.application_id,
                    interview_round_id=stuck.interview_round_id,
                    job_posting_id=stuck.job_posting_id,
                    jd_snapshot_id=stuck.jd_snapshot_id,
                    resume_version_id=stuck.resume_version_id,
                    interview_type=stuck.interview_type,
                    max_primary_questions=stuck.max_primary_questions,
                    max_follow_ups_per_question=stuck.max_follow_ups_per_question,
                )
            )
        except (AgentWorkerError, ValueError) as error:
            # The old run is already terminal and graph.start cancels its own
            # partially-created session on failure. There is therefore nothing
            # resumable to retain in the conversation task: this is not the same
            # failure as an evaluate/report step whose answer is durable.
            return ToolObservation(
                tool_name="restart_mock_interview",
                state="mock_interview_restart_failed",
                message=(
                    "旧的模拟面试已经安全结束，但替代面试暂时启动失败。"
                    "你可以稍后重新开始一场模拟面试。"
                ),
                next_action="旧的面试已经安全结束，替代面试没起来。可以稍后重新开一场。",
                payload={
                    "error_code": (
                        error.code
                        if isinstance(error, AgentWorkerError)
                        else type(error).__name__
                    ),
                    "retryable": (
                        error.retryable
                        if isinstance(error, AgentWorkerError)
                        else False
                    ),
                },
                # The old run was durably cancelled even though its replacement
                # did not start.
                execution_outcome="committed",
            )
        return self._mock_interview_observation(
            result, "restart_mock_interview", user_id=user_id
        )

    def _get_mock_interview_result(self, arguments: dict[str, Any]) -> ToolObservation:
        """Read back a finished run the conversation only holds a reference to.

        The run's own exchanges never enter the conversation, so without this
        the full questions and answers are unreachable once the run ends.
        """
        if self._mock_interview_store is None:
            raise ValueError("Mock interview store is not configured")
        user_id = str(arguments["user_id"])
        question_number = arguments.get("question_number")
        report_id = arguments.get("report_id")
        if report_id is not None:
            # Resolved from a reference the conversation carries, so this names
            # one exact run rather than "the newest for this application".
            session_id = self._mock_interview_store.find_report_session_id(
                user_id=user_id, report_id=str(report_id)
            )
            if session_id is None:
                return ToolObservation(
                    tool_name="get_mock_interview_result",
                    state="no_mock_interview_result_found",
                    message="没有找到这次模拟面试的报告。",
                )
            session = self._mock_interview_store.get_session(
                user_id=user_id, session_id=session_id
            )
            if session is None:
                return ToolObservation(
                    tool_name="get_mock_interview_result",
                    state="no_mock_interview_result_found",
                    message="没有找到这次模拟面试的报告。",
                )
        else:
            application_id = str(arguments["application_id"])
            # Cancelled runs keep every turn they got through, so they are
            # readable too; only the report is missing. Runs still in progress
            # are excluded because the workflow, not this tool, owns a turn
            # while it is driving.
            sessions = self._mock_interview_store.list_sessions(
                user_id=user_id,
                application_id=application_id,
                statuses=("completed", "cancelled"),
                limit=1,
            )
            if not sessions:
                return ToolObservation(
                    tool_name="get_mock_interview_result",
                    state="no_mock_interview_result_found",
                    message="这个投递还没有结束过的模拟面试。",
                )
            session = sessions[0]
        report = self._mock_interview_store.get_report(
            user_id=user_id, session_id=session.id
        )
        turns = self._mock_interview_store.list_turns(
            user_id=user_id, session_id=session.id
        )
        if question_number is not None:
            return self._mock_interview_question_observation(
                turns=turns, question_number=int(question_number), session_id=session.id
            )
        return self._mock_interview_result_observation(
            user_id=user_id, session=session, report=report, turns=turns
        )

    @staticmethod
    def _mock_interview_question_observation(
        *, turns: tuple[MockInterviewTurn, ...], question_number: int,
        session_id: str,
    ) -> ToolObservation:
        """Return one exchange in full, follow-ups included.

        The verbatim text goes into the payload for the presenter rather than
        into ``message``: an answer may be twenty thousand characters, and
        ``message`` is what the durable conversation row keeps and carries into
        every later turn's recent window.
        """
        view = mock_interview_question_view(turns, question_number)
        if view is None:
            return ToolObservation(
                tool_name="get_mock_interview_result",
                state="no_mock_interview_result_found",
                message=f"这次模拟面试没有第 {question_number} 题。",
            )
        return ToolObservation(
            tool_name="get_mock_interview_result",
            state="mock_interview_question_found",
            message=summarize_mock_interview_question(view),
            payload=view.model_dump(mode="json"),
            body_source=MockInterviewBodySource(
                session_id=session_id, question_number=question_number
            ),
        )

    def _mock_interview_result_observation(
        self,
        *,
        user_id: str,
        session: MockInterviewSession,
        report: MockInterviewReport | None,
        turns: tuple[MockInterviewTurn, ...],
    ) -> ToolObservation:
        """Read a finished run back as an index of its questions.

        Everything the screen shows is projected into the payload and rendered
        by the mock interview's own presenter, so this layer composes no prose.
        ``message`` is the bounded line the transcript keeps, and the report it
        names is reachable from the same row through ``resource_ref`` — the two
        together are what let this readback survive the answer writer failing,
        which the earlier version, embedding the whole summary in ``message``
        and carrying no reference, could not.
        """
        primary = tuple(turn for turn in turns if turn.turn_type == "primary")
        questions = []
        for turn in primary:
            if turn.evaluation is not None:
                rating = turn.evaluation.rating
            elif turn.answer is None:
                # Asked and abandoned, which is not the same as answered but
                # unscored: there is nothing here to go back and read.
                rating = "未回答"
            else:
                rating = "未评价"
            questions.append(
                MockInterviewQuestionSummary(
                    plan_item_number=turn.plan_item_number,
                    question=turn.question,
                    rating=rating,
                    follow_up_count=sum(
                        1
                        for candidate in turns
                        if candidate.turn_type == "follow_up"
                        and candidate.plan_item_number == turn.plan_item_number
                    ),
                )
            )
        view = MockInterviewResultView(
            interview_type=session.interview_type,
            status=session.status,
            questions=tuple(questions),
            answered_count=sum(1 for turn in primary if turn.answer is not None),
            report_id=report.id if report is not None else None,
            report_summary=report.summary if report is not None else None,
        )
        job = self._job_display(
            user_id=user_id, job_posting_id=session.job_posting_id
        )
        title = self._resource_title(*(job or ()), "模拟面试报告")
        description = self._resource_description(
            report.summary if report is not None else "模拟面试报告"
        )
        return ToolObservation(
            tool_name="get_mock_interview_result",
            state="mock_interview_result_found",
            message=summarize_mock_interview_result(view),
            payload=view.model_dump(mode="json"),
            resource_ref=(
                ConversationResourceReference(
                    kind="mock_interview_report",
                    resource_id=report.id,
                    title=title,
                    description=description,
                )
                if report is not None
                else None
            ),
        )

    def _list_email_events(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._email_tracking_service is None:
            raise ValueError("Email tracking service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = ListEmailEventsToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        events = self._email_tracking_service.list_events(
            user_id=user_id,
            status=model_arguments.status,
            limit=model_arguments.limit,
        )
        return ToolObservation(
            tool_name="list_email_events",
            state="email_events_found" if events else "no_email_events_found",
            message=f"找到 {len(events)} 个邮件事件。" if events else "没有找到匹配的邮件事件。",
            payload={"items": [self._email_event_payload(event) for event in events]},
        )

    def _resolve_email_event(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._email_tracking_service is None:
            raise ValueError("Email tracking service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = ResolveEmailEventToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        try:
            event = self._email_tracking_service.resolve_event(
                user_id=user_id,
                event_id=model_arguments.event_id,
                approve=model_arguments.approve,
                application_id=model_arguments.application_id,
                interview_round_id=model_arguments.interview_round_id,
            )
        except EmailEventNotFoundError:
            return ToolObservation(
                tool_name="resolve_email_event",
                state="email_event_not_found",
                message="没有找到这个邮件事件，或它不属于当前用户。",
                execution_outcome="not_committed",
            )
        except EmailEventResolutionError as error:
            return ToolObservation(
                tool_name="resolve_email_event",
                state="email_event_resolution_conflict",
                message="该邮件事件暂时不能应用到投递记录。",
                payload={"reason": str(error)},
                # Applying an event spans the interview, application, and email
                # stores. A later conflict can follow an earlier local commit.
                execution_outcome="unknown",
            )
        return ToolObservation(
            tool_name="resolve_email_event",
            state="email_event_resolved",
            message="邮件事件已应用。" if event.status == "applied" else "邮件事件已忽略。",
            payload=self._email_event_payload(event),
            execution_outcome="committed",
        )

    def _list_interviews(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._interview_service is None:
            raise ValueError("Interview service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = ListInterviewsToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        interviews = self._interview_service.list_interviews(
            user_id=user_id,
            application_id=model_arguments.application_id,
            statuses=model_arguments.statuses,
            limit=model_arguments.limit,
        )
        return ToolObservation(
            tool_name="list_interviews",
            state="interviews_found" if interviews else "no_interviews_found",
            message=f"找到 {len(interviews)} 场面试。" if interviews else "没有找到匹配的面试安排。",
            payload={
                "items": [
                    {
                        "selection_index": index,
                        **self._interview_payload(interview),
                    }
                    for index, interview in enumerate(interviews, start=1)
                ]
            },
        )

    def _get_interview(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._interview_service is None:
            raise ValueError("Interview service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = GetInterviewToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.interview_round_id is None:
            raise ValueError("get_interview requires interview_round_id")
        try:
            detail = self._interview_service.get_interview(
                user_id=user_id,
                interview_round_id=model_arguments.interview_round_id,
            )
        except InterviewNotFoundError:
            return ToolObservation(
                tool_name="get_interview",
                state="interview_not_found",
                message="没有找到这场面试，或它不属于当前用户。",
            )
        return ToolObservation(
            tool_name="get_interview",
            state="interview_ready",
            message=f"已读取系统中的第 {detail.interview.sequence_number} 场面试。",
            payload={
                **self._interview_payload(detail.interview),
                "events": [
                    {
                        "event_type": event.event_type,
                        "source": event.source,
                        "email_event_id": event.email_event_id,
                        "details": event.details.model_dump(mode="json"),
                        "occurred_at": event.occurred_at.isoformat(),
                    }
                    for event in detail.events
                ],
                "retros": [self._interview_retro_payload(item) for item in detail.retros],
            },
        )

    def _create_interview(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._interview_service is None:
            raise ValueError("Interview service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = CreateInterviewToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.application_id is None:
            raise ValueError("create_interview requires application_id")
        try:
            interview = self._interview_service.create_manual(
                user_id=user_id,
                application_id=model_arguments.application_id,
                details=model_arguments.details,
            )
        except InterviewApplicationConflictError as error:
            return ToolObservation(
                tool_name="create_interview",
                state="interview_application_conflict",
                message="无法为这条投递创建面试安排。",
                payload={"reason": str(error)},
                execution_outcome="not_committed",
            )
        return ToolObservation(
            tool_name="create_interview",
            state="interview_ready",
            message=f"已记录系统中的第 {interview.sequence_number} 场面试。",
            payload=self._interview_payload(interview),
            execution_outcome="committed",
        )

    def _update_interview(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._interview_service is None:
            raise ValueError("Interview service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = UpdateInterviewToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.interview_round_id is None:
            raise ValueError("update_interview requires interview_round_id")
        try:
            interview = self._interview_service.update_manual(
                user_id=user_id,
                interview_round_id=model_arguments.interview_round_id,
                details=model_arguments.details,
            )
        except InterviewNotFoundError:
            return ToolObservation(
                tool_name="update_interview", state="interview_not_found",
                message="没有找到这场面试，或它不属于当前用户。",
                execution_outcome="not_committed",
            )
        except InterviewApplicationConflictError as error:
            return ToolObservation(
                tool_name="update_interview", state="interview_update_conflict",
                message="这场面试当前不能按该方式更新。",
                payload={"reason": str(error)},
                execution_outcome="not_committed",
            )
        return ToolObservation(
            tool_name="update_interview",
            state="interview_ready",
            message="面试安排已更新，原安排仍保留在事件历史中。",
            payload=self._interview_payload(interview),
            execution_outcome="committed",
        )

    def _complete_interview(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._interview_service is None:
            raise ValueError("Interview service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = CompleteInterviewToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.interview_round_id is None:
            raise ValueError("complete_interview requires interview_round_id")
        try:
            interview = self._interview_service.complete_interview(
                user_id=user_id,
                interview_round_id=model_arguments.interview_round_id,
                completed_at=model_arguments.completed_at,
            )
        except InterviewNotFoundError:
            return ToolObservation(
                tool_name="complete_interview", state="interview_not_found",
                message="没有找到这场面试，或它不属于当前用户。",
                execution_outcome="not_committed",
            )
        except InterviewApplicationConflictError as error:
            return ToolObservation(
                tool_name="complete_interview", state="interview_completion_conflict",
                message="这场面试当前不能标记为完成。",
                payload={"reason": str(error)},
                execution_outcome="not_committed",
            )
        return ToolObservation(
            tool_name="complete_interview",
            state="interview_ready",
            message="已将这场面试标记为完成。",
            payload=self._interview_payload(interview),
            execution_outcome="committed",
        )

    def _record_interview_retro(
        self, arguments: dict[str, Any]
    ) -> ToolObservation:
        if self._interview_service is None:
            raise ValueError("Interview service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = RecordInterviewRetroToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.interview_round_id is None:
            raise ValueError("record_interview_retro requires interview_round_id")
        try:
            report = self._interview_service.record_retro(
                user_id=user_id,
                interview_round_id=model_arguments.interview_round_id,
                source_notes=model_arguments.source_notes,
                summary=model_arguments.summary,
                questions=model_arguments.questions,
                strengths=model_arguments.strengths,
                difficulties=model_arguments.difficulties,
                interviewer_signals=model_arguments.interviewer_signals,
                next_focus=model_arguments.next_focus,
                action_items=model_arguments.action_items,
                limitations=model_arguments.limitations,
                self_assessment=model_arguments.self_assessment,
            )
        except InterviewNotFoundError:
            return ToolObservation(
                tool_name="record_interview_retro",
                state="interview_not_found",
                message="没有找到这场面试，或它不属于当前用户。",
                execution_outcome="not_committed",
            )
        except InterviewApplicationConflictError as error:
            return ToolObservation(
                tool_name="record_interview_retro",
                state="interview_retro_conflict",
                message="这场面试当前不能记录复盘报告。",
                payload={"reason": str(error)},
                execution_outcome="not_committed",
            )
        if self._action_center_service is not None:
            self._action_center_service.complete_source_action(
                user_id=user_id,
                action_type="interview_retro",
                source_id=report.interview_round_id,
            )
        job = None
        if self._application_service is not None:
            try:
                detail = self._application_service.get_application(
                    user_id=user_id, application_id=report.application_id
                )
                job = (
                    detail.job.posting.company_name,
                    detail.job.posting.title,
                )
            except ApplicationInputNotFoundError:
                pass
        interview_label = None
        if self._interview_service is not None:
            try:
                interview_label = self._interview_service.get_interview(
                    user_id=user_id,
                    interview_round_id=report.interview_round_id,
                ).interview.employer_label
            except InterviewNotFoundError:
                pass
        return ToolObservation(
            tool_name="record_interview_retro",
            state="interview_retro_recorded",
            message="真实面试复盘报告已保存；结论仅基于你的复述。",
            payload=self._interview_retro_payload(report),
            execution_outcome="committed",
            resource_ref=ConversationResourceReference(
                kind="interview_retro_report",
                resource_id=report.id,
                title=self._resource_title(
                    *(job or ()), interview_label, "面试复盘"
                ),
                description=self._resource_description(report.summary),
            ),
        )

    def _prepare_interview(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._interview_preparation_service is None:
            raise ValueError("Interview preparation service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = PrepareInterviewToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.interview_round_id is None:
            raise ValueError("prepare_interview requires interview_round_id")
        try:
            preparation = self._interview_preparation_service.prepare(
                user_id=user_id,
                interview_round_id=model_arguments.interview_round_id,
            )
        except InterviewPreparationInputNotFoundError as error:
            return ToolObservation(
                tool_name="prepare_interview",
                state="interview_preparation_input_not_found",
                message="无法读取这场面试对应的完整准备输入。",
                payload={"reason": str(error)},
                execution_outcome="not_committed",
            )
        except InterviewPreparationNotAvailableError as error:
            return ToolObservation(
                tool_name="prepare_interview",
                state="interview_preparation_not_available",
                message="当前面试状态不适合生成准备材料。",
                payload={"reason": str(error)},
                execution_outcome="not_committed",
            )
        title, description = self._job_resource_metadata(
            user_id=user_id,
            job_posting_id=preparation.job_posting_id,
            resource_name="面试准备",
            description=preparation.result.summary,
        )
        return ToolObservation(
            tool_name="prepare_interview",
            state="interview_preparation_ready",
            message=summarize_interview_preparation(preparation.result),
            payload=self._interview_preparation_payload(preparation),
            execution_outcome="committed",
            resource_ref=ConversationResourceReference(
                kind="interview_preparation",
                resource_id=preparation.id,
                title=title,
                description=description,
            ),
        )

    def _get_interview_preparation(
        self, arguments: dict[str, Any]
    ) -> ToolObservation:
        if self._interview_preparation_service is None:
            raise ValueError("Interview preparation service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = GetInterviewPreparationToolArguments.model_validate(
            {
                key: value
                for key, value in arguments.items()
                if key not in {"user_id", "interview_round_id"}
            }
        )
        # Resolved by the projection from a selection index, so it never comes
        # from the model and is not part of the model-facing schema.
        interview_round_id = arguments.get("interview_round_id")
        if model_arguments.preparation_id is None and interview_round_id is None:
            raise ValueError("get_interview_preparation requires preparation_id")
        try:
            preparation = (
                self._interview_preparation_service.get_for_interview(
                    user_id=user_id,
                    interview_round_id=str(interview_round_id),
                )
                if model_arguments.preparation_id is None
                else self._interview_preparation_service.get(
                    user_id=user_id,
                    preparation_id=model_arguments.preparation_id,
                )
            )
        except InterviewPreparationInputNotFoundError:
            return ToolObservation(
                tool_name="get_interview_preparation",
                state="interview_preparation_not_found",
                message="没有找到该面试准备结果，或它不属于当前用户。",
            )
        title, description = self._job_resource_metadata(
            user_id=user_id,
            job_posting_id=preparation.job_posting_id,
            resource_name="面试准备",
            description=preparation.result.summary,
        )
        return ToolObservation(
            tool_name="get_interview_preparation",
            state="interview_preparation_ready",
            message=summarize_interview_preparation(preparation.result),
            payload=self._interview_preparation_payload(preparation),
            resource_ref=ConversationResourceReference(
                kind="interview_preparation",
                resource_id=preparation.id,
                title=title,
                description=description,
            ),
        )

    def _get_daily_brief(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._action_center_service is None:
            raise ValueError("Action Center service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = GetDailyBriefToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        try:
            brief = self._action_center_service.daily_brief(
                user_id=user_id,
                timezone_name=model_arguments.timezone,
            )
        except ValueError:
            return ToolObservation(
                tool_name="get_daily_brief",
                state="invalid_timezone",
                message="无法识别该时区，请提供 IANA 时区名称，例如 Asia/Shanghai。",
            )
        sections = {
            "overdue": brief.overdue,
            "due_today": brief.due_today,
            "upcoming": brief.upcoming,
            "no_due_date": brief.no_due_date,
        }
        count = sum(len(items) for items in sections.values())
        return ToolObservation(
            tool_name="get_daily_brief",
            state="daily_brief_ready",
            body_dependencies=tuple(
                BodyDependency(kind=item.source_type, resource_id=item.source_id)
                for items in sections.values()
                for item in items
                if item.source_type in {"application", "interview_round", "email_event"}
            ),
            message=f"今日职业简报包含 {count} 个待办事项。" if count else "今日没有待办事项。",
            # The receipt can only carry the total without becoming the report,
            # but the split is what a follow-up turns on. ``waiting`` is an open
            # item with no due date; the domain has no separate waiting bucket.
            facts={
                "overdue": len(brief.overdue),
                "due_today": len(brief.due_today),
                "waiting": len(brief.no_due_date),
            },
            payload={
                "timezone": brief.timezone,
                "generated_at": brief.generated_at.isoformat(),
                **{
                    name: [self._action_payload(item) for item in items]
                    for name, items in sections.items()
                },
            },
        )

    def _list_action_items(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._action_center_service is None:
            raise ValueError("Action Center service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = ListActionItemsToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        try:
            items = self._action_center_service.list_actions(
                user_id=user_id,
                statuses=model_arguments.statuses,
                limit=model_arguments.limit,
                timezone_name=model_arguments.timezone,
            )
        except ValueError:
            return ToolObservation(
                tool_name="list_action_items",
                state="invalid_timezone",
                message="无法识别该时区，请提供 IANA 时区名称，例如 Asia/Shanghai。",
            )
        return ToolObservation(
            tool_name="list_action_items",
            state="action_items_found" if items else "no_action_items_found",
            message=f"找到 {len(items)} 个行动事项。" if items else "当前没有匹配的行动事项。",
            payload={
                "items": [
                    {"selection_index": index, **self._action_payload(item)}
                    for index, item in enumerate(items, start=1)
                ]
            },
        )

    def _complete_action_item(self, arguments: dict[str, Any]) -> ToolObservation:
        return self._resolve_action_item(arguments, action="complete")

    def _dismiss_action_item(self, arguments: dict[str, Any]) -> ToolObservation:
        return self._resolve_action_item(arguments, action="dismiss")

    def _resolve_action_item(
        self, arguments: dict[str, Any], *, action: Literal["complete", "dismiss"]
    ) -> ToolObservation:
        if self._action_center_service is None:
            raise ValueError("Action Center service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = ResolveActionItemToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.action_item_id is None:
            raise ValueError(f"{action}_action_item requires action_item_id")
        try:
            item = (
                self._action_center_service.complete_action(
                    user_id=user_id,
                    action_item_id=model_arguments.action_item_id,
                )
                if action == "complete"
                else self._action_center_service.dismiss_action(
                    user_id=user_id,
                    action_item_id=model_arguments.action_item_id,
                )
            )
        except ActionItemNotFoundError:
            return ToolObservation(
                tool_name=f"{action}_action_item",
                state="action_item_not_found",
                message="没有找到这个行动事项，或它不属于当前用户。",
                execution_outcome="not_committed",
            )
        return ToolObservation(
            tool_name=f"{action}_action_item",
            state="action_item_resolved",
            message="行动事项已完成。" if action == "complete" else "行动事项已忽略。",
            payload=self._action_payload(item),
            execution_outcome="committed",
        )

    def _snooze_action_item(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._action_center_service is None:
            raise ValueError("Action Center service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = SnoozeActionItemToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.action_item_id is None:
            raise ValueError("snooze_action_item requires action_item_id")
        try:
            item = self._action_center_service.snooze_action(
                user_id=user_id,
                action_item_id=model_arguments.action_item_id,
                snoozed_until=model_arguments.snoozed_until,
            )
        except ActionItemNotFoundError:
            return ToolObservation(
                tool_name="snooze_action_item",
                state="action_item_not_found",
                message="没有找到这个行动事项，或它不属于当前用户。",
                execution_outcome="not_committed",
            )
        except InvalidActionTransitionError as error:
            return ToolObservation(
                tool_name="snooze_action_item",
                state="invalid_action_transition",
                message="无法将该行动事项稍后提醒。",
                payload={"reason": str(error)},
                execution_outcome="not_committed",
            )
        return ToolObservation(
            tool_name="snooze_action_item",
            state="action_item_snoozed",
            message="行动事项已设置为稍后提醒。",
            payload=self._action_payload(item),
            execution_outcome="committed",
        )

    def _list_calendar_accounts(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._calendar_service is None:
            raise ValueError("Calendar service is not configured")
        user_id = str(arguments["user_id"])
        ListCalendarAccountsToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        accounts = self._calendar_service.list_accounts(user_id=user_id)
        return ToolObservation(
            tool_name="list_calendar_accounts",
            state="calendar_accounts_found" if accounts else "no_calendar_accounts",
            message=(
                f"找到 {len(accounts)} 个 Calendar 账户。"
                if accounts else "尚未配置 Calendar 账户。"
            ),
            payload={
                "items": [
                    {
                        "selection_index": index,
                        "calendar_account_id": account.id,
                        "provider": account.provider,
                        "email_address": account.email_address,
                        "calendar_id": account.calendar_id,
                        "status": account.status,
                    }
                    for index, account in enumerate(accounts, start=1)
                ]
            },
        )

    def _list_calendar_links(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._calendar_service is None:
            raise ValueError("Calendar service is not configured")
        user_id = str(arguments["user_id"])
        ListCalendarLinksToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        links = self._calendar_service.list_links(user_id=user_id)
        return ToolObservation(
            tool_name="list_calendar_links",
            state="calendar_links_found" if links else "no_calendar_links",
            message=(
                f"找到 {len(links)} 条面试 Calendar 同步记录。"
                if links else "当前没有面试 Calendar 同步记录。"
            ),
            payload={
                "items": [
                    {
                        "calendar_link_id": link.id,
                        "calendar_account_id": link.calendar_account_id,
                        "interview_round_id": link.interview_round_id,
                        "status": link.status,
                        "external_html_link": link.external_html_link,
                        "updated_at": link.updated_at.isoformat(),
                    }
                    for link in links
                ]
            },
        )

    def _prepare_interview_calendar_sync(
        self, arguments: dict[str, Any]
    ) -> ToolObservation:
        if self._calendar_service is None:
            raise ValueError("Calendar service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = PrepareInterviewCalendarSyncToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.interview_round_id is None:
            raise ValueError("prepare_interview_calendar_sync requires interview_round_id")
        try:
            proposal = self._calendar_service.prepare_interview_sync(
                user_id=user_id,
                interview_round_id=model_arguments.interview_round_id,
                calendar_account_id=model_arguments.calendar_account_id,
            )
        except CalendarAccountNotFoundError as error:
            return ToolObservation(
                tool_name="prepare_interview_calendar_sync",
                state="calendar_account_required",
                message="需要先配置或选择一个 Calendar 账户。",
                payload={"reason": str(error)},
                execution_outcome="not_committed",
            )
        except CalendarSyncNotAvailableError as error:
            return ToolObservation(
                tool_name="prepare_interview_calendar_sync",
                state="calendar_sync_not_available",
                message="当前面试没有需要执行的 Calendar 变更。",
                payload={"reason": str(error), "retryable": False},
                execution_outcome="not_committed",
            )
        return ToolObservation(
            tool_name="prepare_interview_calendar_sync",
            state="calendar_approval_required",
            message="Calendar 变更预览已生成；执行前需要用户明确确认。",
            payload=self._calendar_proposal_payload(proposal),
            execution_outcome="committed",
        )

    def _get_calendar_proposal(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._calendar_service is None:
            raise ValueError("Calendar service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = GetCalendarProposalToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.proposal_id is None:
            raise ValueError("get_calendar_proposal requires proposal_id")
        try:
            proposal = self._calendar_service.get_proposal(
                user_id=user_id, proposal_id=model_arguments.proposal_id
            )
        except CalendarProposalNotFoundError:
            return ToolObservation(
                tool_name="get_calendar_proposal",
                state="calendar_proposal_not_found",
                message="没有找到该 Calendar 变更预览，或它不属于当前用户。",
            )
        return ToolObservation(
            tool_name="get_calendar_proposal",
            state="calendar_proposal_ready",
            message="已读取 Calendar 变更预览。",
            payload=self._calendar_proposal_payload(proposal),
        )

    def _execute_calendar_proposal(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._calendar_service is None:
            raise ValueError("Calendar service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = ExecuteCalendarProposalToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.proposal_id is None:
            raise ValueError("execute_calendar_proposal requires proposal_id")
        try:
            execution = self._calendar_service.execute_proposal(
                user_id=user_id, proposal_id=model_arguments.proposal_id
            )
        except CalendarProposalNotFoundError:
            return ToolObservation(
                tool_name="execute_calendar_proposal",
                state="calendar_proposal_not_found",
                message="没有找到该 Calendar 变更预览，或它不属于当前用户。",
                execution_outcome="not_committed",
            )
        except CalendarProposalConflictError as error:
            return ToolObservation(
                tool_name="execute_calendar_proposal",
                state="calendar_approval_invalid",
                message="该 Calendar 批准已失效，没有执行外部写入。",
                payload={"reason": str(error)},
                execution_outcome="not_committed",
            )
        except CalendarConnectorError as error:
            return ToolObservation(
                tool_name="execute_calendar_proposal",
                state="calendar_write_failed",
                message=(
                    "Calendar 外部写入没有获得成功确认。结果不确定时，"
                    "下一份预览会先核对该执行记录，再决定是否可以重新批准。"
                ),
                payload={
                    "error_code": error.code,
                    "error_detail": str(error),
                    "retryable": False,
                },
                execution_outcome=(
                    "unknown" if error.outcome_unknown else "not_committed"
                ),
            )
        return ToolObservation(
            tool_name="execute_calendar_proposal",
            state="calendar_sync_complete",
            message="Calendar 变更已执行并获得成功确认。",
            execution_outcome="committed",
            payload={
                **self._calendar_proposal_payload(execution.proposal),
                "calendar_link_id": execution.link.id,
                "calendar_link_status": execution.link.status,
                "external_html_link": execution.link.external_html_link,
            },
        )

    @staticmethod
    def _email_event_payload(event) -> dict[str, Any]:
        return {
            "email_event_id": event.id,
            "application_id": event.application_id,
            "event_type": event.event_type,
            "status": event.status,
            "confidence": event.confidence,
            "classifier": event.classifier,
            "summary": event.summary,
            "occurred_at": event.occurred_at.isoformat(),
            "interview_details": (
                event.interview_details.model_dump(mode="json")
                if event.interview_details is not None
                else None
            ),
        }

    @staticmethod
    def _interview_payload(interview) -> dict[str, Any]:
        return {
            "interview_round_id": interview.id,
            "application_id": interview.application_id,
            "sequence_number": interview.sequence_number,
            "employer_label": interview.employer_label,
            "status": interview.status,
            "scheduled_start": (
                interview.scheduled_start.isoformat()
                if interview.scheduled_start is not None
                else None
            ),
            "scheduled_end": (
                interview.scheduled_end.isoformat()
                if interview.scheduled_end is not None
                else None
            ),
            "timezone": interview.timezone,
            "interview_format": interview.interview_format,
            "location": interview.location,
            "meeting_url": interview.meeting_url,
            "contact_summary": interview.contact_summary,
        }

    @staticmethod
    def _interview_retro_payload(report) -> dict[str, Any]:
        return {
            "retro_report_id": report.id,
            "interview_round_id": report.interview_round_id,
            "application_id": report.application_id,
            "source_notes": report.source_notes,
            "summary": report.summary,
            "questions": [item.model_dump(mode="json") for item in report.questions],
            "strengths": list(report.strengths),
            "difficulties": list(report.difficulties),
            "interviewer_signals": list(report.interviewer_signals),
            "next_focus": list(report.next_focus),
            "action_items": list(report.action_items),
            "limitations": list(report.limitations),
            "self_assessment": report.self_assessment,
            "created_at": report.created_at.isoformat(),
        }

    @staticmethod
    def _action_payload(item) -> dict[str, Any]:
        return {
            "action_item_id": item.id,
            "action_type": item.action_type,
            "source_type": item.source_type,
            "source_id": item.source_id,
            "application_id": item.application_id,
            "title": item.title,
            "summary": item.summary,
            "due_at": item.due_at.isoformat() if item.due_at is not None else None,
            "status": item.status,
            "snoozed_until": (
                item.snoozed_until.isoformat()
                if item.snoozed_until is not None
                else None
            ),
        }

    @staticmethod
    def _calendar_proposal_payload(proposal) -> dict[str, Any]:
        return {
            "proposal_id": proposal.id,
            "calendar_account_id": proposal.calendar_account_id,
            "interview_round_id": proposal.interview_round_id,
            "operation": proposal.operation,
            "status": proposal.status,
            "payload_hash": proposal.payload_hash,
            "payload": (
                proposal.payload.model_dump(mode="json")
                if proposal.payload is not None
                else None
            ),
            "created_at": proposal.created_at.isoformat(),
            "expires_at": proposal.expires_at.isoformat(),
            "executed_at": (
                proposal.executed_at.isoformat()
                if proposal.executed_at is not None
                else None
            ),
        }

    @staticmethod
    def _interview_preparation_payload(preparation) -> dict[str, Any]:
        return {
            "preparation_id": preparation.id,
            "interview_round_id": preparation.interview_round_id,
            "application_id": preparation.application_id,
            "job_posting_id": preparation.job_posting_id,
            "jd_snapshot_id": preparation.jd_snapshot_id,
            "resume_version_id": preparation.resume_version_id,
            "created_at": preparation.created_at.isoformat(),
            "preparation": preparation.result.model_dump(mode="json"),
        }

    def invoke_atomic_tool(self, name: str, arguments: dict[str, Any]) -> ToolObservation:
        handler = self._atomic_handlers.get(name)
        if handler is None:
            raise ValueError(f"Unknown main-agent atomic tool: {name}")
        return self._require_write_execution_outcome(name, handler(arguments))

    def resolve_resume_analysis_confirmation(
        self,
        *,
        user_id: str,
        analysis_id: str,
        action: Literal["confirm", "cancel"],
    ) -> ToolObservation:
        """Consume a UI-bound decision without granting that write to the LLM."""

        arguments = {"user_id": user_id, "analysis_id": analysis_id}
        if action == "confirm":
            return self._confirm_resume_analysis(arguments)
        return self._reject_resume_analysis(arguments)

    def deliver_resume_artifact(
        self, *, user_id: str, artifact_id: str
    ) -> ResumeArtifactDelivery:
        """Resolve an attachment outside the model-visible tool/state loop."""
        if self._resume_export_service is None:
            raise ValueError("Resume export service is not configured")
        return self._resume_export_service.read_artifact(
            user_id=user_id,
            artifact_id=artifact_id,
        )

    def _open_job_search(self, arguments: dict[str, Any]) -> ToolObservation:
        user_id = arguments.get("user_id")
        conversation_id = arguments.get("conversation_id")
        source_turn_id = arguments.get("source_turn_id")
        model_arguments = OpenJobSearchToolArguments.model_validate(
            {
                key: value
                for key, value in arguments.items()
                if key not in {"user_id", "conversation_id", "source_turn_id"}
            }
        )
        keyword = model_arguments.keyword.strip()
        city = model_arguments.city.strip() if model_arguments.city else None
        query = keyword
        params = {"query": query}
        city_code = self._BOSS_CITY_CODES.get(city or "") or self._BOSS_CITY_CODES.get(
            (city or "").casefold()
        )
        if city_code:
            params["city"] = city_code
        elif city:
            # Unknown city names stay visible in the search terms rather than
            # being translated into a guessed internal BOSS code.
            params["query"] = f"{city} {keyword}"
        search_url = f"https://www.zhipin.com/web/geek/job?{urlencode(params)}"
        scope = f"（{city}）" if city else ""
        client_action: dict[str, Any] = {
            "type": "open_url",
            "url": search_url,
            "label": f"在 BOSS 搜索 {keyword}",
        }
        if self._job_capture_store is not None and user_id and conversation_id:
            # The intent rides beside the URL, never inside it: the page hands
            # it to the extension over the local bridge, so BOSS never sees it
            # and a pasted link cannot impersonate this conversation.
            intent = self._job_capture_store.create_intent(
                user_id=str(user_id),
                conversation_id=str(conversation_id),
                source_turn_id=str(source_turn_id) if source_turn_id else None,
                platform="boss",
                keyword=keyword,
                city=city,
            )
            client_action["capture_intent_id"] = intent.id
            client_action["capture_intent_expires_at"] = intent.expires_at.isoformat()
        return ToolObservation(
            tool_name="open_job_search",
            state="job_search_page_ready",
            message=(
                f"已准备打开 BOSS 搜索“{keyword}”{scope}。"
                "请正常浏览，并只保存你感兴趣的岗位。"
            ),
            payload={
                "platform": "boss",
                "keyword": keyword,
                "city": city,
                "client_action": client_action,
            },
            # This capability commits the client action into the turn result;
            # it does not claim the remote page itself loaded successfully.
            execution_outcome="committed",
        )

    def _research_job(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._job_research_service is None:
            raise ValueError("Job research service is not configured")
        user_id = str(arguments["user_id"])
        pinned_snapshot_id = arguments.get("jd_snapshot_id")
        model_arguments = ResearchJobToolArguments.model_validate(
            {
                key: value
                for key, value in arguments.items()
                if key not in {"user_id", "jd_snapshot_id"}
            }
        )
        if model_arguments.job_posting_id is None:
            raise ValueError("research_job requires job_posting_id")
        try:
            result = self._job_research_service.research(
                user_id=user_id,
                job_posting_id=model_arguments.job_posting_id,
                **(
                    {"jd_snapshot_id": pinned_snapshot_id}
                    if isinstance(pinned_snapshot_id, str)
                    else {}
                ),
                focus=model_arguments.focus,
                user_provided_context=model_arguments.user_provided_context,
                max_sources=model_arguments.max_sources,
            )
        except JobResearchInputNotFoundError:
            return ToolObservation(
                tool_name="research_job",
                state="job_research_not_found",
                message="没有找到要研究的已保存岗位。",
                execution_outcome="not_committed",
            )
        except JobResearchExecutionError as error:
            return self._job_research_failure(
                tool_name="research_job",
                error=error,
                job_posting_id=model_arguments.job_posting_id,
            )
        title, description = self._job_research_metadata(
            user_id=user_id, report=result.report
        )
        return ToolObservation(
            tool_name="research_job",
            state="job_research_ready",
            # The line the durable conversation row keeps. Composed by the
            # report's own presenter so it is the same headline whether or not
            # the answer writer ran this turn; the full report reaches the
            # screen from the entity, not from here.
            message=summarize_job_research(
                result.report.summary, cached=result.cached
            ),
            facts=MainAgentToolRegistry._job_research_facts(result),
            payload=self._job_research_payload(
                result, model_arguments.job_posting_id
            ),
            resource_ref=ConversationResourceReference(
                kind="job_research_report",
                resource_id=result.report.id,
                job_posting_id=result.report.job_posting_id,
                company_key=result.report.company_key,
                title=title,
                description=description,
                status_at_delivery=result.report.status,
                anchored_by_other_job=(
                    result.report.job_posting_id
                    != model_arguments.job_posting_id
                ),
            ),
            execution_outcome="committed",
        )

    def _retry_job_research(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._job_research_service is None:
            raise ValueError("Job research service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = RetryJobResearchToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.run_id is None:
            raise ValueError("retry_job_research requires run_id")
        try:
            result = self._job_research_service.retry(
                user_id=user_id,
                run_id=model_arguments.run_id,
            )
        except (JobResearchInputNotFoundError, JobResearchRunNotRetryableError):
            return ToolObservation(
                tool_name="retry_job_research",
                state="job_research_not_retryable",
                message="当前没有可以恢复的岗位研究任务。",
                execution_outcome="not_committed",
            )
        except JobResearchExecutionError as error:
            return self._job_research_failure(
                tool_name="retry_job_research",
                error=error,
            )
        title, description = self._job_research_metadata(
            user_id=user_id, report=result.report
        )
        return ToolObservation(
            tool_name="retry_job_research",
            state="job_research_ready",
            message=summarize_job_research(result.report.summary, cached=False),
            facts=MainAgentToolRegistry._job_research_facts(result),
            payload=self._job_research_payload(result),
            resource_ref=ConversationResourceReference(
                kind="job_research_report",
                resource_id=result.report.id,
                job_posting_id=result.report.job_posting_id,
                company_key=result.report.company_key,
                title=title,
                description=description,
                status_at_delivery=result.report.status,
                anchored_by_other_job=False,
            ),
            execution_outcome="committed",
        )

    def _get_job_research(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._job_research_service is None:
            raise ValueError("Job research service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = GetJobResearchToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        try:
            result = self._job_research_service.get_report(
                user_id=user_id,
                report_id=model_arguments.report_id,
                job_posting_id=model_arguments.job_posting_id,
            )
        except JobResearchInputNotFoundError:
            return ToolObservation(
                tool_name="get_job_research",
                state="job_research_not_found",
                message="没有找到该岗位的研究报告。",
            )
        title, description = self._job_research_metadata(
            user_id=user_id, report=result.report
        )
        return ToolObservation(
            tool_name="get_job_research",
            state="job_research_ready",
            # An explicit read, not a reuse the user did not ask for, so the
            # cached wording would misdescribe it even though the service marks
            # every stored report as cached.
            message=summarize_job_research(result.report.summary, cached=False),
            facts=MainAgentToolRegistry._job_research_facts(result),
            payload=self._job_research_payload(
                result, model_arguments.job_posting_id
            ),
            resource_ref=ConversationResourceReference(
                kind="job_research_report",
                resource_id=result.report.id,
                job_posting_id=result.report.job_posting_id,
                company_key=result.report.company_key,
                title=title,
                description=description,
                status_at_delivery=result.report.status,
                anchored_by_other_job=bool(
                    self._job_research_payload(
                        result, model_arguments.job_posting_id
                    ).get("anchored_by_other_job")
                ),
            ),
        )

    @staticmethod
    def _job_research_failure(
        *,
        tool_name: str,
        error: JobResearchExecutionError,
        job_posting_id: str | None = None,
    ) -> ToolObservation:
        return ToolObservation(
            tool_name=tool_name,
            state="job_research_failed",
            message=worker_failure_reason(error),
            payload={
                "run_id": error.run_id,
                "job_posting_id": job_posting_id,
                "error_code": error.code,
                "retryable": error.retryable,
            },
            # The failed run and its retry coordinates were durably recorded.
            execution_outcome="committed",
        )

    @staticmethod
    def _job_research_facts(result) -> dict[str, bool | int | str]:
        """Whether the report was reused, how much it found, and how current.

        The receipt is the report's own headline; it cannot state a three-valued
        status precisely enough for a conditional follow-up to turn on it.
        """
        return {
            "cached": bool(result.cached),
            "finding_count": len(result.report.findings),
            "status": str(result.report.status),
        }

    @staticmethod
    def _job_research_payload(
        result, requested_job_posting_id: str | None = None
    ) -> dict[str, Any]:
        report = result.report
        # Research is reused across every saved job at one company, so a report
        # may have been anchored by a different posting's JD. Say so rather than
        # letting it read as though it were written for the job in hand.
        anchored_elsewhere = (
            requested_job_posting_id is not None
            and report.job_posting_id != requested_job_posting_id
        )
        return {
            "run_id": result.run.id,
            "report_id": report.id,
            "job_posting_id": requested_job_posting_id or report.job_posting_id,
            "anchor_job_posting_id": report.job_posting_id,
            "anchored_by_other_job": anchored_elsewhere,
            "status": report.status,
            "cached": result.cached,
            "user_provided_context": report.scope.user_provided_context,
            "research": {
                "summary": report.summary,
                "findings": [
                    finding.model_dump(mode="json") for finding in report.findings
                ],
                "open_questions": list(report.open_questions),
                "limitations": list(report.limitations),
            },
            "sources": [
                {
                    "source_key": source.source_key,
                    "url": source.url,
                    "title": source.title,
                    "publisher": source.publisher,
                    "published_at": (
                        source.published_at.isoformat()
                        if source.published_at is not None
                        else None
                    ),
                    "retrieved_at": source.retrieved_at.isoformat(),
                    "relevant_excerpt": source.relevant_excerpt,
                }
                for source in result.sources
            ],
        }

    def _find_saved_jobs(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._job_repository is None:
            raise ValueError("Saved-job repository is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = FindSavedJobsToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        items = self._job_repository.search_saved_jobs(
            user_id=user_id,
            query=model_arguments.query,
            limit=model_arguments.limit,
            # A job the user removed from the library must not come back
            # through the agent's own search: re-surfacing it is the exact
            # thing the removal was about.
            include_dismissed=False,
        )
        payload = {
            "items": [
                item.model_dump(mode="json", exclude={"jd_snapshot_id"})
                for item in items
            ],
            "query": model_arguments.query,
        }
        return ToolObservation(
            tool_name="find_saved_jobs",
            state="saved_jobs_found" if items else "no_saved_jobs_found",
            message=f"找到 {len(items)} 个已保存职位。" if items else "没有找到匹配的已保存职位。",
            payload=payload,
        )

    def _get_saved_job(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._job_repository is None:
            raise ValueError("Saved-job repository is not configured")
        user_id = str(arguments["user_id"])
        pinned_snapshot_id = arguments.get("jd_snapshot_id")
        model_arguments = GetSavedJobToolArguments.model_validate(
            {
                key: value
                for key, value in arguments.items()
                if key not in {"user_id", "jd_snapshot_id"}
            }
        )
        record = self._job_repository.get_job(user_id=user_id, job_posting_id=model_arguments.job_posting_id)
        if record is None:
            return ToolObservation(
                tool_name="get_saved_job",
                state="saved_job_not_found",
                message="没有找到这个已保存职位，或它不属于当前用户。",
                payload={"job_posting_id": model_arguments.job_posting_id},
            )
        # The conversation pinned a JD version; read that one, not the latest
        # capture, so "this job" means the text the earlier turn showed. A pin
        # that no longer resolves (or names another posting) falls back to the
        # latest rather than failing the read.
        if isinstance(pinned_snapshot_id, str) and pinned_snapshot_id != record.snapshot.id:
            pinned = self._job_repository.get_snapshot(
                user_id=user_id, jd_snapshot_id=pinned_snapshot_id
            )
            if pinned is not None and pinned.job_posting_id == record.posting.id:
                record = record.model_copy(
                    update={"snapshot": pinned, "analysis": None}
                )
        payload = {
            "job": {
                "job_posting_id": record.posting.id,
                "title": record.posting.title,
                "company_name": record.posting.company_name,
                "city": record.city,
                "salary": record.salary,
                "source_name": record.posting.source_name,
                "source_url": record.posting.source_url,
                "availability_status": record.availability_status,
            },
            "jd_snapshot": {
                "id": record.snapshot.id,
                "version": record.snapshot.version,
                "content": record.snapshot.content,
                "captured_at": record.snapshot.captured_at.isoformat(),
                "provenance": record.snapshot.provenance.model_dump(mode="json"),
            },
            "analysis": record.analysis.analysis.model_dump(mode="json") if record.analysis else None,
        }
        return ToolObservation(
            tool_name="get_saved_job",
            state="saved_job_ready",
            message=f"已读取 {record.posting.title}（{record.posting.company_name}）的完整 JD。",
            payload=payload,
            # Pinned to the snapshot, not the posting: the posting may be
            # captured again later, and this turn's card has to keep opening
            # the text this turn read. Title and description are a display
            # snapshot so the card still names the job once the posting is gone.
            resource_ref=ConversationResourceReference(
                kind="saved_job",
                resource_id=record.snapshot.id,
                job_posting_id=record.posting.id,
                title=saved_job_title(
                    record.posting.title, record.posting.company_name
                ),
                description=saved_job_description(
                    record.snapshot.version, record.posting.source_name
                ),
            ),
        )

    def _list_target_roles(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._resume_store is None:
            raise ValueError("Resume store is not configured")
        user_id = str(arguments["user_id"])
        ListTargetRolesToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        roles = self._resume_store.list_target_roles(user_id=user_id)
        return ToolObservation(
            tool_name="list_target_roles",
            state="target_roles_found" if roles else "no_target_roles_found",
            message=f"找到 {len(roles)} 个简历目标岗位分类。" if roles else "当前还没有简历目标岗位分类。",
            payload={
                "items": [
                    {
                        "selection_index": index,
                        "target_role_id": role.id,
                        "title": role.title,
                        "priority": role.priority,
                        "status": role.status,
                        "city": role.city,
                        "salary_expectation": role.salary_expectation,
                        "experience": role.experience,
                        "education": role.education,
                    }
                    for index, role in enumerate(roles, start=1)
                ]
            },
        )

    def _list_resumes(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._resume_store is None:
            raise ValueError("Resume store is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = ListResumesToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        resumes = self._resume_store.list_resumes(
            user_id=user_id,
            target_role_id=model_arguments.target_role_id,
        )
        return ToolObservation(
            tool_name="list_resumes",
            state="resumes_found" if resumes else "no_resumes_found",
            message=f"找到 {len(resumes)} 份简历。" if resumes else "没有找到匹配的简历。",
            payload={
                "target_role_id": model_arguments.target_role_id,
                "items": [
                    {
                        "selection_index": index,
                        "resume_id": resume.id,
                        "target_role_id": resume.target_role_id,
                        "name": resume.name,
                        "status": resume.status,
                        "latest_version_id": resume.latest_version_id,
                        "updated_at": resume.updated_at.isoformat(),
                    }
                    for index, resume in enumerate(resumes, start=1)
                ],
            },
        )

    def _get_resume_metadata(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._resume_store is None:
            raise ValueError("Resume store is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = GetResumeMetadataToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        resume = self._resume_store.get_resume(user_id=user_id, resume_id=model_arguments.resume_id)
        if resume is None:
            return ToolObservation(
                tool_name="get_resume_metadata",
                state="resume_not_found",
                message="没有找到这份简历，或它不属于当前用户。",
                payload={"resume_id": model_arguments.resume_id},
            )
        versions = self._resume_store.list_versions(user_id=user_id, resume_id=resume.id)
        return ToolObservation(
            tool_name="get_resume_metadata",
            state="resume_metadata_ready",
            message=f"已读取简历“{resume.name}”及其 {len(versions)} 个版本的元数据。",
            payload={
                "resume": {
                    "resume_id": resume.id,
                    "target_role_id": resume.target_role_id,
                    "name": resume.name,
                    "status": resume.status,
                    "latest_version_id": resume.latest_version_id,
                    "created_at": resume.created_at.isoformat(),
                    "updated_at": resume.updated_at.isoformat(),
                },
                "versions": [
                    {
                        "resume_version_id": version.id,
                        "version_number": version.version_number,
                        "source_type": version.source_type,
                        "document_format": version.document_format,
                        "byte_size": version.byte_size,
                        "created_at": version.created_at.isoformat(),
                    }
                    for version in versions
                ],
            },
        )

    def _analyze_resume(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._resume_analysis_service is None:
            raise ValueError("Resume analysis service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = AnalyzeResumeToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        try:
            draft = self._resume_analysis_service.analyze_version(
                user_id=user_id,
                resume_version_id=model_arguments.resume_version_id,
            )
        except ResumeVersionNotFoundError:
            return ToolObservation(
                tool_name="analyze_resume",
                state="resume_version_not_found",
                message="没有找到这个简历版本，或它不属于当前用户。",
                payload={"resume_version_id": model_arguments.resume_version_id},
                execution_outcome="not_committed",
            )
        except ResumeAnalysisWorkerNotCommittedError as error:
            return ToolObservation(
                tool_name="analyze_resume",
                state="failed",
                message=worker_failure_reason(error),
                payload={
                    "resume_version_id": model_arguments.resume_version_id,
                    "error_code": error.code,
                    "retryable": error.retryable,
                },
                execution_outcome="not_committed",
            )
        except AgentWorkerError as error:
            return ToolObservation(
                tool_name="analyze_resume",
                state="failed",
                message="简历分析结果是否已保存无法确认，请先核对再重试。",
                payload={
                    "resume_version_id": model_arguments.resume_version_id,
                    "error_code": error.code,
                    "retryable": error.retryable,
                },
                execution_outcome="unknown",
            )
        return ToolObservation(
            tool_name="analyze_resume",
            state="resume_analysis_ready",
            disposition="interaction_required",
            body_source=ResumeAnalysisBodySource(
                analysis_id=draft.id, expires_at=draft.expires_at
            ),
            message=f"已分析该简历版本，提取出 {len(draft.result.records)} 段候选经历。",
            # Whether to ask the candidate anything before confirming turns on
            # the clarification and warning counts, which the receipt cannot
            # carry without listing them.
            facts={
                "record_count": len(draft.result.records),
                "clarification_count": len(draft.result.clarification_questions),
                "has_warnings": bool(draft.result.warnings),
            },
            payload={
                "analysis_id": draft.id,
                "resume_version_id": model_arguments.resume_version_id,
                "expires_at": draft.expires_at.isoformat(),
                "records": [record.model_dump(mode="json") for record in draft.result.records],
                "clarification_questions": draft.result.clarification_questions,
                "warnings": draft.result.warnings,
            },
            execution_outcome="committed",
        )

    def _get_resume_analysis(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._resume_analysis_service is None:
            raise ValueError("Resume analysis service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = GetResumeAnalysisToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.analysis_id is None:
            raise ValueError("get_resume_analysis requires analysis_id")
        try:
            draft = self._resume_analysis_service.get_analysis(
                user_id=user_id,
                analysis_id=model_arguments.analysis_id,
            )
        except ResumeAnalysisNotFoundError:
            return ToolObservation(
                tool_name="get_resume_analysis",
                state="resume_analysis_not_found",
                message="没有找到这次简历分析，或它已经过期。",
                payload={"analysis_id": model_arguments.analysis_id},
            )
        return ToolObservation(
            tool_name="get_resume_analysis",
            state="resume_analysis_ready",
            body_source=ResumeAnalysisBodySource(
                analysis_id=draft.id, expires_at=draft.expires_at
            ),
            message=f"已读取这次简历分析，其中有 {len(draft.result.records)} 段候选经历。",
            next_action=(
                "这份分析还没确认。确认由用户在界面上完成，你不能代他确认。"
                if draft.status == "pending"
                else None
            ),
            payload={
                "analysis_id": draft.id,
                "resume_version_id": draft.resume_version_id,
                "status": draft.status,
                "expires_at": draft.expires_at.isoformat(),
                "records": [record.model_dump(mode="json") for record in draft.result.records],
                "clarification_questions": draft.result.clarification_questions,
                "warnings": draft.result.warnings,
            },
        )

    def _confirm_resume_analysis(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._resume_analysis_service is None:
            raise ValueError("Resume analysis service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = ConfirmResumeAnalysisToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.analysis_id is None:
            raise ValueError("confirm_resume_analysis requires analysis_id")
        try:
            imported = self._resume_analysis_service.confirm_analysis(
                user_id=user_id,
                analysis_id=model_arguments.analysis_id,
            )
        except (ResumeAnalysisNotFoundError, ResumeAnalysisNotPendingError):
            return ToolObservation(
                tool_name="confirm_resume_analysis",
                state="resume_analysis_decision_expired",
                message="这次简历分析已过期或已处理，不能再次确认。",
                payload={"analysis_id": model_arguments.analysis_id},
            )
        return ToolObservation(
            tool_name="confirm_resume_analysis",
            state="resume_analysis_confirmed",
            message=(
                f"已确认并保存 {len(imported.records)} 段职业经历和 "
                f"{len(imported.evidence)} 条事实证据。"
            ),
            payload={
                "analysis_id": model_arguments.analysis_id,
                "career_record_ids": [record.id for record in imported.records],
                "career_evidence_ids": [evidence.id for evidence in imported.evidence],
            },
        )

    def _reject_resume_analysis(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._resume_analysis_service is None:
            raise ValueError("Resume analysis service is not configured")
        user_id = str(arguments["user_id"])
        analysis_id = str(arguments.get("analysis_id", ""))
        if not analysis_id:
            raise ValueError("reject_resume_analysis requires analysis_id")
        try:
            self._resume_analysis_service.reject_analysis(
                user_id=user_id,
                analysis_id=analysis_id,
            )
        except (ResumeAnalysisNotFoundError, ResumeAnalysisNotPendingError):
            return ToolObservation(
                tool_name="reject_resume_analysis",
                state="resume_analysis_decision_expired",
                message="这次简历分析已过期或已处理，不能再次取消。",
                payload={"analysis_id": analysis_id},
            )
        return ToolObservation(
            tool_name="reject_resume_analysis",
            state="resume_analysis_rejected",
            message="已取消导入；这次分析不会写入职业事实库。",
            payload={"analysis_id": analysis_id},
        )

    def _propose_job_intent(self, arguments: dict[str, Any]) -> ToolObservation:
        update: JobIntentUpdate = arguments["update"]
        scope = None
        if update.is_role_scoped and self._resume_store is not None:
            role = self._resume_store.get_target_role(
                user_id=str(arguments["user_id"]),
                target_role_id=str(update.target_role_id),
            )
            if role is None:
                return ToolObservation(
                    tool_name="propose_job_intent",
                    state="target_role_not_found",
                    message="没有找到这个目标岗位，或它不属于当前用户。",
                    payload={},
                )
            scope = role.title
        return _proposal_observation(
            tool_name="propose_job_intent",
            state="job_intent_proposed",
            message=self._job_intent_readback(update, scope=scope),
            proposal=update,
            payload_key="update",
            exclude_none=True,
        )

    def _confirm_job_intent(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._career_profile_store is None:
            raise ValueError("Career profile store is not configured")
        user_id = str(arguments["user_id"])
        update: JobIntentUpdate = arguments["update"]
        conversation_id = arguments.get("conversation_id")
        intent_episode_id = (
            "intent_confirmation_"
            + hashlib.sha256(
                (
                    user_id
                    + "\0"
                    + json.dumps(
                        update.model_dump(mode="json"),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                ).encode("utf-8")
            ).hexdigest()[:32]
        )
        admitted_scopes, admitted_proposals = self._admit_job_intent_scopes(
            user_id=user_id,
            conversation_id=(
                str(conversation_id) if conversation_id is not None else None
            ),
            update=update,
        )
        layer = update.layer or (
            "transient"
            if update.timescale == "situational"
            else "contextual"
            if update.pref_scope != "global"
            else "stable"
        )
        valid_until = update.valid_until
        if update.timescale == "situational" and valid_until is None:
            valid_until = datetime.now(timezone.utc) + timedelta(days=30)
        if update.pref_scope != "global" or layer != "stable":
            versions = []
            for scope, value in admitted_scopes:
                candidate = IntentCaptureCandidate(
                    user_id=user_id,
                    scope_key=scope.scope_key,
                    pref_scope=update.pref_scope,
                    timescale=update.timescale,
                    layer=layer,
                    valid_until=valid_until,
                    value=value,
                    source="confirmed_job_intent",
                    confidence=1.0,
                )
                if update.is_role_scoped:
                    if self._resume_store is None:
                        raise ValueError("Resume store is not configured")
                    _, version = self._resume_store.capture_target_role_intent(
                        candidate
                    )
                else:
                    capture = getattr(
                        self._career_profile_store,
                        "capture_profile_intent",
                        None,
                    )
                    if not callable(capture):
                        raise ValueError(
                            "Career profile store cannot capture scoped intent"
                        )
                    _, version = capture(candidate)
                if version is not None:
                    versions.append(version)
            return ToolObservation(
                tool_name="confirm_job_intent",
                state="job_intent_recorded",
                message=self._job_intent_readback(
                    update,
                    scope=None,
                    saved=True,
                ),
                payload={
                    "intent_episode_id": intent_episode_id,
                    "pref_scope": update.pref_scope,
                    "timescale": update.timescale,
                    "versions": [
                        version.model_dump(mode="json") for version in versions
                    ],
                },
                execution_outcome="committed",
            )
        if update.is_role_scoped:
            if self._resume_store is None:
                raise ValueError("Resume store is not configured")
            role = self._resume_store.update_target_role_intent(
                user_id=user_id,
                target_role_id=str(update.target_role_id),
                city=update.city,
                salary_expectation=update.salary_expectation,
                experience=update.experience,
                education=update.education,
                source="confirmed_job_intent",
                pref_scope=update.pref_scope,
                timescale=update.timescale,
                layer=layer,
            )
            return ToolObservation(
                tool_name="confirm_job_intent",
                state="job_intent_recorded",
                message=self._job_intent_readback(update, scope=role.title, saved=True),
                payload={
                    "intent_episode_id": intent_episode_id,
                    "target_role": role.model_dump(mode="json"),
                },
                execution_outcome="committed",
            )
        # Re-read rather than trusting the projected copy: the stored profile is
        # the thing being changed, and it may have moved since the readback.
        stored = self._career_profile_store.get_profile(user_id) or (
            CareerProfileContext(user_id=user_id)
        )
        updated = update.apply_to_profile(stored)
        self._career_profile_store.upsert_profile(
            updated,
            source="confirmed_job_intent",
            pref_scope=update.pref_scope,
            timescale=update.timescale,
            layer=layer,
        )
        return ToolObservation(
            tool_name="confirm_job_intent",
            state="job_intent_recorded",
            message=self._job_intent_readback(update, scope=None, saved=True),
            payload={
                "intent_episode_id": intent_episode_id,
                "profile": updated.model_dump(mode="json"),
            },
            execution_outcome="committed",
        )

    def _confirm_free_text_preference(
        self,
        arguments: dict[str, Any],
    ) -> ToolObservation:
        if self._career_profile_store is None:
            raise ValueError("Career profile store is not configured")
        confirm = getattr(
            self._career_profile_store,
            "confirm_free_text_preference",
            None,
        )
        if not callable(confirm):
            raise ValueError("Career profile store cannot confirm free-text preferences")
        proposal = arguments.get("proposal")
        if not isinstance(proposal, FreeTextPreferenceConfirmationProposal):
            proposal = FreeTextPreferenceConfirmationProposal.model_validate(proposal)
        if proposal.base_update_id is not None:
            confirm_amendment = getattr(
                self._career_profile_store,
                "confirm_free_text_preference_amendment",
                None,
            )
            if not callable(confirm_amendment):
                raise ValueError(
                    "Career profile store cannot amend free-text preferences"
                )
            version = confirm_amendment(
                user_id=str(arguments["user_id"]),
                base_update_id=proposal.base_update_id,
                expected_content_sha256=str(
                    proposal.expected_content_sha256
                ),
                statement=proposal.statement,
            )
        else:
            version = confirm(
                user_id=str(arguments["user_id"]),
                update_id=str(arguments["update_id"]),
                conversation_id=str(arguments.get("conversation_id", "current")),
                job_posting_id=arguments.get("job_posting_id"),
                scope_choice=arguments.get("scope_choice"),
                scope_domain=arguments.get("scope_domain"),
            )
        if version is None:
            return ToolObservation(
                tool_name="confirm_free_text_preference",
                state="free_text_preference_confirmation_missing",
                message="这条待确认偏好已经变化、删除或处理过，没有重复写入。",
                payload={},
                execution_outcome="not_committed",
            )
        structured_proposal = None
        if (
            version.scope_key == "person_intent/self/company_scale"
            and version.semantic_stance in {"negative", "positive"}
        ):
            structured_proposal = JobIntentUpdate(
                pref_scope=structured_pref_scope(version.pref_scope),
                timescale=version.timescale,
                layer=version.layer,
                valid_until=version.valid_until,
                hard_constraints=(
                    HardConstraintContext(
                        relation="company_scale",
                        value=(
                            "exclude_large_companies"
                            if version.semantic_stance == "negative"
                            else "allow_large_companies"
                        ),
                    ),
                ),
            )
        message = f"已确认并启用这条长期偏好：{version.value}"
        payload: dict[str, Any] = {
            "scope_key": version.scope_key,
            "memory_entry_id": intent_entry_id(
                version.scope_key,
                version.pref_scope,
            ),
            "revision": version.revision,
        }
        if structured_proposal is not None:
            message += (
                "\n我还识别到可用于确定性筛选的结构化版本："
                f"{self._job_intent_readback(structured_proposal, scope=None)}"
                "\n是否也确认写入这条结构化偏好？"
            )
            payload["structured_proposal"] = structured_proposal.model_dump(
                mode="json", exclude_none=True
            )
        if structured_proposal is not None:
            return ToolObservation(
                tool_name="confirm_free_text_preference",
                state="free_text_preference_confirmed_structured_proposed",
                message=message,
                payload=payload,
                execution_outcome="committed",
            )
        return ToolObservation(
            tool_name="confirm_free_text_preference",
            state="free_text_preference_confirmed",
            message=message,
            payload=payload,
            execution_outcome="committed",
        )

    @staticmethod
    def _propose_free_text_preference_confirmation(
        arguments: dict[str, Any],
    ) -> ToolObservation:
        proposal = arguments["proposal"]
        if not isinstance(proposal, FreeTextPreferenceConfirmationProposal):
            proposal = FreeTextPreferenceConfirmationProposal.model_validate(proposal)
        question = (
            "这条偏好只针对这类岗位，还是以后默认都这样？"
            if proposal.needs_scope_clarification
            else "是否确认启用？"
        )
        return _proposal_observation(
            tool_name="propose_free_text_preference_confirmation",
            state="free_text_preference_confirmation_proposed",
            message=(
                "我从对话中提取到一条可能需要长期记住的偏好：\n"
                f"- {proposal.statement}\n"
                "目前它仍在隔离区，不会影响岗位推荐。"
                f"{question}"
            ),
            proposal=proposal,
        )

    def _admit_job_intent_scopes(
        self,
        *,
        user_id: str,
        conversation_id: str | None,
        update: JobIntentUpdate,
    ) -> tuple[
        tuple[tuple[CanonicalScope, str], ...], tuple[ScopeProposal, ...]
    ]:
        values = update.model_dump(exclude_none=True)
        values.pop("target_role_id", None)
        values.pop("pref_scope", None)
        values.pop("timescale", None)
        values.pop("layer", None)
        values.pop("valid_until", None)
        constraints = values.pop("hard_constraints", ())
        family = "target_role_intent" if update.is_role_scoped else "person_intent"
        subject_id = str(update.target_role_id) if update.is_role_scoped else "self"
        admitted: list[tuple[CanonicalScope, str]] = []
        proposals: list[ScopeProposal] = []
        for field, value in values.items():
            relation = (
                "default_city"
                if family == "person_intent" and field == "city"
                else field
            )
            proposal = ScopeProposal(
                user_id=user_id,
                conversation_id=conversation_id,
                family=family,
                subject_id=subject_id,
                relation=relation,
                proposed_value=str(value),
            )
            proposals.append(proposal)
            resolution = self._canonical_scope_resolver.resolve(proposal)
            if resolution.canonical_scope is None:
                raise ValueError(
                    "Job intent cannot be written without a canonical scope."
                )
            scope = resolution.canonical_scope
            admitted.append((scope, str(value)))
        for constraint in constraints:
            relation = str(constraint["relation"])
            value = str(constraint["value"])
            proposal = ScopeProposal(
                user_id=user_id,
                conversation_id=conversation_id,
                family="person_intent",
                subject_id="self",
                relation=relation,
                proposed_value=value,
            )
            proposals.append(proposal)
            resolution = self._canonical_scope_resolver.resolve(proposal)
            if resolution.canonical_scope is None:
                raise ValueError(
                    "Job intent cannot be written without a canonical scope."
                )
            scope = resolution.canonical_scope
            admitted.append((scope, value))
        return tuple(admitted), tuple(proposals)

    _JOB_INTENT_LABELS = {
        "city": "城市",
        "salary_expectation": "薪资期望",
        "experience": "经验",
        "education": "学历",
    }
    _HARD_CONSTRAINT_LABELS = {
        "work_arrangement": "办公方式硬约束",
        "work_schedule": "工作时间硬约束",
        "company_scale": "公司规模偏好",
    }

    @classmethod
    def _job_intent_readback(
        cls,
        update: JobIntentUpdate,
        *,
        scope: str | None,
        saved: bool = False,
    ) -> str:
        fields = update.model_dump(exclude_none=True)
        fields.pop("target_role_id", None)
        pref_scope = str(fields.pop("pref_scope", "global"))
        timescale = str(fields.pop("timescale", "permanent"))
        fields.pop("layer", None)
        fields.pop("valid_until", None)
        constraints = fields.pop("hard_constraints", ())
        lines = [
            f"- {cls._JOB_INTENT_LABELS[field]}：{value}"
            for field, value in fields.items()
        ]
        lines.extend(
            f"- {cls._HARD_CONSTRAINT_LABELS[item['relation']]}：{item['value']}"
            for item in constraints
        )
        body = "\n".join(lines)
        where = f"目标岗位「{scope}」" if scope else "整体求职意向"
        if pref_scope != "global":
            where += f"（仅限 {pref_scope}；{timescale}）"
        if saved:
            return f"已记录{where}：\n{body}"
        return (
            f"我准备把这些记到{where}上：\n{body}\n"
            "确认后才会保存；这只是你告诉我的意向，不是对你能力的判断。"
        )

    def _compare_saved_jobs(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._job_comparison_service is None:
            raise ValueError("Job comparison service is not configured")
        user_id = str(arguments["user_id"])
        job_posting_ids = tuple(arguments.get("job_posting_ids", ()))
        preferred_city = arguments.get("preferred_city")
        try:
            comparison = self._job_comparison_service.compare(
                user_id=user_id,
                job_posting_ids=job_posting_ids,
                preferred_city=str(preferred_city) if preferred_city else None,
            )
        except JobComparisonInputNotFoundError:
            return ToolObservation(
                tool_name="compare_saved_jobs",
                state="compare_input_not_found",
                message="其中有岗位没有找到，或它不属于当前用户。",
                payload={},
            )
        return ToolObservation(
            tool_name="compare_saved_jobs",
            state="saved_jobs_compared",
            body_dependencies=tuple(
                BodyDependency(kind="job", resource_id=row.job_posting_id)
                for row in comparison.rows
            ),
            message=f"已对比 {len(comparison.rows)} 个已保存岗位。",
            payload={"comparison": comparison.model_dump(mode="json")},
        )

    def _analyze_job(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._job_analysis_service is None:
            raise ValueError("Job analysis service is not configured")
        user_id = str(arguments["user_id"])
        pinned_snapshot_id = arguments.get("jd_snapshot_id")
        model_arguments = AnalyzeJobToolArguments.model_validate(
            {
                key: value
                for key, value in arguments.items()
                if key not in {"user_id", "jd_snapshot_id"}
            }
        )
        if model_arguments.job_posting_id is None:
            raise ValueError("analyze_job requires job_posting_id")
        try:
            stored = self._job_analysis_service.analyze(
                user_id=user_id,
                job_posting_id=model_arguments.job_posting_id,
                **(
                    {"jd_snapshot_id": pinned_snapshot_id}
                    if isinstance(pinned_snapshot_id, str)
                    else {}
                ),
            )
        except JobAnalysisInputNotFoundError as error:
            return ToolObservation(
                tool_name="analyze_job",
                state="saved_job_not_found",
                message=(
                    "没有找到这个职位对应的 JD 版本，或它不属于当前用户。"
                    if error.input_kind == "jd_snapshot"
                    else "没有找到这个已保存职位，或它不属于当前用户。"
                ),
                payload={
                    "missing_input": error.input_kind,
                    "job_posting_id": model_arguments.job_posting_id,
                },
                execution_outcome="not_committed",
            )
        except AgentWorkerError as error:
            return ToolObservation(
                tool_name="analyze_job",
                state="failed",
                message=worker_failure_reason(error),
                payload={
                    "job_posting_id": model_arguments.job_posting_id,
                    "error_code": error.code,
                    "retryable": error.retryable,
                },
                execution_outcome="not_committed",
            )
        result = stored.analysis.to_result()
        if result is None:
            raise ValueError("Job analysis service returned a legacy analysis without tiers")
        job = self._job_display(user_id=user_id, job_posting_id=stored.job_posting_id)
        title = (
            self._resource_title(job[0], job[1], "JD 分析")
            if job is not None
            else self._resource_title("岗位 JD 分析")
        )
        description = self._resource_description(
            f"仅基于 JD 文本的岗位分析；判定层级为{SENIORITY_LABELS[result.seniority]}。{result.summary}"
        )
        return ToolObservation(
            tool_name="analyze_job",
            state="job_analysis_ready",
            message=f"已完成岗位 JD 分析，判定层级为{SENIORITY_LABELS[result.seniority]}，共 {len(result.requirements)} 条分级要求。",
            payload={
                "analysis_id": stored.id,
                "job_posting_id": stored.job_posting_id,
                "jd_snapshot_id": stored.jd_snapshot_id,
                "analyzer_version": stored.analyzer_version,
                "created_at": stored.created_at.isoformat(),
                **result.model_dump(mode="json"),
            },
            resource_ref=ConversationResourceReference(
                kind="job_analysis",
                resource_id=stored.id,
                job_posting_id=stored.job_posting_id,
                title=title,
                description=description,
            ),
            execution_outcome="committed",
        )

    def _match_resume_to_job(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._resume_job_match_service is None:
            raise ValueError("Resume-job match service is not configured")
        user_id = str(arguments["user_id"])
        pinned_snapshot_id = arguments.get("jd_snapshot_id")
        model_arguments = MatchResumeToJobToolArguments.model_validate(
            {
                key: value
                for key, value in arguments.items()
                if key not in {"user_id", "jd_snapshot_id"}
            }
        )
        try:
            stored = self._resume_job_match_service.match(
                user_id=user_id,
                resume_version_id=model_arguments.resume_version_id,
                job_posting_id=model_arguments.job_posting_id,
                **(
                    {"jd_snapshot_id": pinned_snapshot_id}
                    if isinstance(pinned_snapshot_id, str)
                    else {}
                ),
            )
        except ResumeJobMatchAnalysisRequiredError as error:
            return ToolObservation(
                tool_name="match_resume_to_job",
                state="job_analysis_required",
                message="匹配前需要先完成当前 JD 版本的岗位分析。",
                next_action="先调用 analyze_job 分析当前 JD，成功后再匹配简历。",
                payload={
                    "job_posting_id": error.job_posting_id,
                    "jd_snapshot_id": error.jd_snapshot_id,
                    "retryable": False,
                },
                execution_outcome="not_committed",
            )
        except ResumeJobMatchInputNotFoundError as error:
            return ToolObservation(
                tool_name="match_resume_to_job",
                state="match_input_not_found",
                message=(
                    "没有找到这个简历版本，或它不属于当前用户。"
                    if error.input_kind == "resume_version"
                    else "没有找到这个职位对应的 JD 版本，或它不属于当前用户。"
                    if error.input_kind == "jd_snapshot"
                    else "没有找到这个已保存职位，或它不属于当前用户。"
                ),
                payload={
                    "missing_input": error.input_kind,
                    "resume_version_id": model_arguments.resume_version_id,
                    "job_posting_id": model_arguments.job_posting_id,
                },
                execution_outcome="not_committed",
            )
        except AgentWorkerError as error:
            return ToolObservation(
                tool_name="match_resume_to_job",
                state="failed",
                message=worker_failure_reason(error),
                payload={
                    "resume_version_id": model_arguments.resume_version_id,
                    "job_posting_id": model_arguments.job_posting_id,
                    "error_code": error.code,
                    "retryable": error.retryable,
                },
                execution_outcome="not_committed",
            )
        title, description = self._resume_match_metadata(
            user_id=user_id, stored=stored
        )
        return ToolObservation(
            tool_name="match_resume_to_job",
            state="resume_job_match_ready",
            message=f"已完成逐项匹配，整体匹配度为 {stored.result.overall_fit}。",
            payload={
                "match_id": stored.id,
                "resume_version_id": model_arguments.resume_version_id,
                "job_posting_id": model_arguments.job_posting_id,
                "created_at": stored.created_at.isoformat(),
                **stored.result.model_dump(mode="json"),
            },
            resource_ref=ConversationResourceReference(
                kind="resume_job_match",
                resource_id=stored.id,
                title=title,
                description=description,
            ),
            execution_outcome="committed",
        )

    def _get_resume_job_match(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._resume_job_match_service is None:
            raise ValueError("Resume-job match service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = GetResumeJobMatchToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.match_id is None:
            raise ValueError("get_resume_job_match requires match_id")
        try:
            stored = self._resume_job_match_service.get_match(
                user_id=user_id,
                match_id=model_arguments.match_id,
            )
        except ResumeJobMatchInputNotFoundError:
            return ToolObservation(
                tool_name="get_resume_job_match",
                state="resume_job_match_not_found",
                message="没有找到这次简历岗位匹配，或它不属于当前用户。",
                payload={"match_id": model_arguments.match_id},
            )
        title, description = self._resume_match_metadata(
            user_id=user_id, stored=stored
        )
        return ToolObservation(
            tool_name="get_resume_job_match",
            state="resume_job_match_ready",
            message=f"已读取匹配结果，整体匹配度为 {stored.result.overall_fit}。",
            payload={
                "match_id": stored.id,
                "resume_version_id": stored.resume_version_id,
                "job_posting_id": stored.job_posting_id,
                "created_at": stored.created_at.isoformat(),
                **stored.result.model_dump(mode="json"),
            },
            resource_ref=ConversationResourceReference(
                kind="resume_job_match",
                resource_id=stored.id,
                title=title,
                description=description,
            ),
        )

    def _draft_resume_tailoring(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._resume_tailoring_service is None:
            raise ValueError("Resume tailoring service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = DraftResumeTailoringToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.match_id is None:
            raise ValueError("draft_resume_tailoring requires match_id")
        try:
            draft = self._resume_tailoring_service.create_draft(
                user_id=user_id,
                match_id=model_arguments.match_id,
                tailoring_goal=model_arguments.tailoring_goal,
            )
        except ResumeJobMatchInputNotFoundError:
            return ToolObservation(
                tool_name="draft_resume_tailoring",
                state="resume_job_match_not_found",
                message="没有找到可用于定制的匹配结果，或它不属于当前用户。",
                payload={"match_id": model_arguments.match_id},
                execution_outcome="not_committed",
            )
        except ResumeTailoringReviewBlockedError as error:
            return ToolObservation(
                tool_name="draft_resume_tailoring",
                state="resume_tailoring_review_blocked",
                message="自动审核未能产出安全的简历修改草稿，需要调整目标或人工确认。",
                payload={"reason": str(error)},
                execution_outcome="not_committed",
            )
        except AgentWorkerError as error:
            return ToolObservation(
                tool_name="draft_resume_tailoring",
                state="failed",
                message=worker_failure_reason(error),
                payload={"error_code": error.code, "retryable": error.retryable},
                execution_outcome="not_committed",
            )
        return self._tailoring_observation(
            user_id=user_id,
            tool_name="draft_resume_tailoring",
            draft=draft,
            message=f"已生成 {len(draft.result.changes)} 条待审阅的简历修改建议。",
            execution_outcome="committed",
        )

    def _get_resume_tailoring_draft(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._resume_tailoring_service is None:
            raise ValueError("Resume tailoring service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = GetResumeTailoringDraftToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.draft_id is None:
            raise ValueError("get_resume_tailoring_draft requires draft_id")
        try:
            draft = self._resume_tailoring_service.get_draft(
                user_id=user_id,
                draft_id=model_arguments.draft_id,
            )
        except ResumeTailoringDraftNotFoundError:
            return ToolObservation(
                tool_name="get_resume_tailoring_draft",
                state="resume_tailoring_draft_not_found",
                message="没有找到这份简历定制草稿，或它已经过期。",
                payload={"draft_id": model_arguments.draft_id},
            )
        return self._tailoring_observation(
            user_id=user_id,
            tool_name="get_resume_tailoring_draft",
            draft=draft,
            message=f"已读取包含 {len(draft.result.changes)} 条修改建议的草稿。",
        )

    def _review_resume_tailoring(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._resume_tailoring_service is None:
            raise ValueError("Resume tailoring service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = ReviewResumeTailoringToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.draft_id is None:
            raise ValueError("review_resume_tailoring requires draft_id")
        try:
            draft = self._resume_tailoring_service.review_draft(
                user_id=user_id,
                draft_id=model_arguments.draft_id,
                accepted_change_indices=model_arguments.accepted_change_indices,
                rejected_change_indices=model_arguments.rejected_change_indices,
                feedback=model_arguments.feedback,
            )
        except ResumeTailoringDraftNotFoundError:
            return ToolObservation(
                tool_name="review_resume_tailoring",
                state="resume_tailoring_draft_not_found",
                message="没有找到这份简历定制草稿，或它已经过期。",
                payload={"draft_id": model_arguments.draft_id},
                execution_outcome="not_committed",
            )
        except ResumeTailoringAlreadyFinalizedError:
            return ToolObservation(
                tool_name="review_resume_tailoring",
                state="resume_tailoring_already_finalized",
                message="这份草稿已经生成了新简历版本，审阅决定不能再修改。",
                payload={"draft_id": model_arguments.draft_id},
                execution_outcome="not_committed",
            )
        except ResumeTailoringSupersededError:
            return ToolObservation(
                tool_name="review_resume_tailoring",
                state="resume_tailoring_superseded",
                message="该草稿已有更新版本，请审阅当前最新草稿。",
                payload={"draft_id": model_arguments.draft_id},
                execution_outcome="not_committed",
            )
        return self._tailoring_observation(
            user_id=user_id,
            tool_name="review_resume_tailoring",
            draft=draft,
            message=(
                "所有简历修改建议都已完成审阅。"
                if draft.status == "reviewed"
                else f"已记录审阅决定，还有 {len(draft.pending_change_indices)} 条建议待处理。"
            ),
            execution_outcome="committed",
        )

    def _revise_resume_tailoring(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._resume_tailoring_service is None:
            raise ValueError("Resume tailoring service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = ReviseResumeTailoringToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.draft_id is None:
            raise ValueError("revise_resume_tailoring requires draft_id")
        try:
            draft = self._resume_tailoring_service.revise_draft(
                user_id=user_id,
                draft_id=model_arguments.draft_id,
                feedback=model_arguments.feedback,
            )
        except ResumeTailoringDraftNotFoundError:
            return ToolObservation(
                tool_name="revise_resume_tailoring",
                state="resume_tailoring_draft_not_found",
                message="没有找到要修改的简历草稿，或它已经过期。",
                payload={"draft_id": model_arguments.draft_id},
                execution_outcome="not_committed",
            )
        except ResumeTailoringAlreadyFinalizedError:
            return ToolObservation(
                tool_name="revise_resume_tailoring",
                state="resume_tailoring_already_finalized",
                message="该草稿已经生成简历版本；如需继续修改，应基于新版本重新匹配和定制。",
                payload={"draft_id": model_arguments.draft_id},
                execution_outcome="not_committed",
            )
        except ResumeTailoringSupersededError:
            return ToolObservation(
                tool_name="revise_resume_tailoring",
                state="resume_tailoring_superseded",
                message="该草稿已有更新版本，请基于当前最新草稿继续反馈。",
                payload={"draft_id": model_arguments.draft_id},
                execution_outcome="not_committed",
            )
        except ResumeTailoringReviewBlockedError as error:
            return ToolObservation(
                tool_name="revise_resume_tailoring",
                state="resume_tailoring_review_blocked",
                message="根据用户反馈生成的新草稿未通过自动审核，原草稿保持不变。",
                payload={"reason": str(error)},
                execution_outcome="not_committed",
            )
        except AgentWorkerError as error:
            return ToolObservation(
                tool_name="revise_resume_tailoring",
                state="failed",
                message=worker_failure_reason(error),
                payload={"error_code": error.code, "retryable": error.retryable},
                execution_outcome="not_committed",
            )
        return self._tailoring_observation(
            user_id=user_id,
            tool_name="revise_resume_tailoring",
            draft=draft,
            message=(
                f"已根据反馈生成第 {draft.revision_number} 版草稿；"
                "旧审批决定未继承，请重新逐条审阅。"
            ),
            execution_outcome="committed",
        )

    def _finalize_resume_tailoring(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._resume_tailoring_service is None:
            raise ValueError("Resume tailoring service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = FinalizeResumeTailoringToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.draft_id is None:
            raise ValueError("finalize_resume_tailoring requires draft_id")
        try:
            finalized = self._resume_tailoring_service.finalize_draft(
                user_id=user_id,
                draft_id=model_arguments.draft_id,
            )
        except ResumeTailoringDraftNotFoundError:
            return ToolObservation(
                tool_name="finalize_resume_tailoring",
                state="resume_tailoring_draft_not_found",
                message="没有找到这份简历定制草稿，或它已经过期。",
                payload={"draft_id": model_arguments.draft_id},
                execution_outcome="not_committed",
            )
        except ResumeTailoringNotReadyError as error:
            return ToolObservation(
                tool_name="finalize_resume_tailoring",
                state="resume_tailoring_not_ready",
                message="必须先逐条审阅所有建议，并至少接受一条修改。",
                payload={
                    "draft_id": model_arguments.draft_id,
                    "reason": str(error),
                    "retryable": False,
                },
                execution_outcome="not_committed",
            )
        except ResumeFinalReviewBlockedError as error:
            return ToolObservation(
                tool_name="finalize_resume_tailoring",
                state="resume_final_review_blocked",
                message="最终审核发现未获批准或不安全的实质变化，因此没有创建新简历版本。",
                payload={
                    "draft_id": model_arguments.draft_id,
                    "reason": str(error),
                },
                execution_outcome="not_committed",
            )
        except AgentWorkerError as error:
            return ToolObservation(
                tool_name="finalize_resume_tailoring",
                state="failed",
                message=worker_failure_reason(error),
                payload={"error_code": error.code, "retryable": error.retryable},
                execution_outcome="not_committed",
            )
        return ToolObservation(
            tool_name="finalize_resume_tailoring",
            state="resume_tailoring_finalized",
            message=(
                "已生成新的不可变 Markdown 简历版本。"
                if finalized.created
                else "这份定制草稿已经生成过简历版本，已返回原结果。"
            ),
            payload={
                "draft_id": finalized.draft_id,
                "resume_id": finalized.resume.id,
                "resume_version_id": finalized.resume_version.id,
                "version_number": finalized.resume_version.version_number,
                "document_format": finalized.resume_version.document_format,
                "source_type": finalized.resume_version.source_type,
                "created_at": finalized.resume_version.created_at.isoformat(),
                "applied_change_indices": finalized.applied_change_indices,
                "warnings": finalized.warnings,
                "created": finalized.created,
            },
            execution_outcome="committed",
        )

    def _export_resume_artifact(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._resume_export_service is None:
            raise ValueError("Resume export service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = ExportResumeArtifactToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.resume_version_id is None:
            raise ValueError("export_resume_artifact requires resume_version_id")
        try:
            artifact = self._resume_export_service.prepare_export(
                user_id=user_id,
                resume_version_id=model_arguments.resume_version_id,
            )
        except ResumeExportNotFoundError:
            return ToolObservation(
                tool_name="export_resume_artifact",
                state="resume_version_not_found",
                message="没有找到可导出的简历版本，或它不属于当前用户。",
                payload={"resume_version_id": model_arguments.resume_version_id},
                execution_outcome="not_committed",
            )
        return ToolObservation(
            tool_name="export_resume_artifact",
            state="resume_artifact_ready",
            message=f"简历文件 {artifact.filename} 已准备好。",
            payload={
                "artifact_id": artifact.id,
                "resume_version_id": artifact.resume_version_id,
                "filename": artifact.filename,
                "media_type": artifact.media_type,
                "byte_size": artifact.byte_size,
                "created_at": artifact.created_at.isoformat(),
            },
            execution_outcome="committed",
        )

    def _create_application(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._application_service is None:
            raise ValueError("Application service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = CreateApplicationToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if (
            model_arguments.job_posting_id is None
            or model_arguments.resume_version_id is None
        ):
            raise ValueError("create_application requires job and resume version IDs")
        try:
            result = self._application_service.create_application(
                user_id=user_id,
                job_posting_id=model_arguments.job_posting_id,
                resume_version_id=model_arguments.resume_version_id,
                submitted_at=model_arguments.submitted_at,
                note=model_arguments.note,
            )
            detail = self._application_service.get_application(
                user_id=user_id,
                application_id=result.application.id,
            )
        except ApplicationInputNotFoundError as error:
            return ToolObservation(
                tool_name="create_application",
                state="application_input_not_found",
                message="没有找到对应的已保存岗位或简历版本，或它不属于当前用户。",
                payload={"missing_input": str(error)},
                execution_outcome="not_committed",
            )
        return ToolObservation(
            tool_name="create_application",
            state="application_ready",
            message=(
                "已创建投递记录。"
                if result.created
                else "这个岗位已有进行中的投递记录，已返回原记录。"
            ),
            payload={
                **self._application_payload(result.application, detail.job),
                "created": result.created,
            },
            execution_outcome="committed",
        )

    def _update_owner_settings(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._owner_settings_store is None:
            raise ValueError("Owner settings store is not configured")
        user_id = str(arguments["user_id"])
        expected_revision = int(arguments["expected_revision"])
        actor_id = str(arguments["confirmation_id"])
        model_arguments = UpdateOwnerSettingsToolArguments.model_validate(
            {
                key: value
                for key, value in arguments.items()
                if key not in {"user_id", "expected_revision", "confirmation_id"}
            }
        )
        current = self._owner_settings_store.get_owner_settings(user_id) or OwnerSettingsContext()
        desired = current.model_copy(
            update={
                "preferences": current.preferences.model_copy(
                    update={
                        "boss_search": model_arguments.boss_search
                        or current.preferences.boss_search
                    }
                ),
                "behavior_policy": current.behavior_policy.model_copy(
                    update={
                        "application_confirmation": (
                            model_arguments.application_confirmation
                            or current.behavior_policy.application_confirmation
                        ),
                        "confirm_before": (
                            current.behavior_policy.confirm_before
                            if model_arguments.confirm_before is None
                            else model_arguments.confirm_before
                        ),
                    }
                ),
            }
        )
        try:
            updated = self._owner_settings_store.update_owner_settings(
                user_id=user_id,
                desired=desired,
                expected_revision=expected_revision,
                actor_type="confirmed_agent_proposal",
                actor_id=actor_id,
            )
        except OwnerSettingsConflictError:
            return ToolObservation(
                tool_name="update_owner_settings",
                state="owner_settings_conflict",
                message="设置在确认期间已被其他入口修改；旧提议没有覆盖新设置。",
                execution_outcome="not_committed",
            )
        return ToolObservation(
            tool_name="update_owner_settings",
            state="owner_settings_updated",
            message="已按你的确认更新持久设置。",
            payload={
                "revision": updated.revision,
                "policy_revision": updated.behavior_policy.revision,
            },
            execution_outcome="committed",
        )

    def _update_application_status(
        self, arguments: dict[str, Any]
    ) -> ToolObservation:
        if self._application_service is None:
            raise ValueError("Application service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = UpdateApplicationStatusToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.application_id is None:
            raise ValueError("update_application_status requires application_id")
        try:
            application = self._application_service.update_application(
                user_id=user_id,
                application_id=model_arguments.application_id,
                status=model_arguments.status,
                note=model_arguments.note,
            )
            detail = self._application_service.get_application(
                user_id=user_id,
                application_id=application.id,
            )
        except ApplicationInputNotFoundError:
            return ToolObservation(
                tool_name="update_application_status",
                state="application_not_found",
                message="没有找到这条投递记录，或它不属于当前用户。",
                payload={"application_id": model_arguments.application_id},
                execution_outcome="not_committed",
            )
        except InvalidApplicationTransitionError as error:
            return ToolObservation(
                tool_name="update_application_status",
                state="invalid_application_transition",
                message="这次投递状态变化不符合当前流程。",
                payload={
                    "application_id": model_arguments.application_id,
                    "reason": str(error),
                },
                execution_outcome="not_committed",
            )
        except ConcurrentApplicationUpdateError:
            return ToolObservation(
                tool_name="update_application_status",
                state="application_update_conflict",
                message="这条投递记录刚刚发生了变化，请重新读取后再更新。",
                payload={"application_id": model_arguments.application_id},
                execution_outcome="not_committed",
            )
        return ToolObservation(
            tool_name="update_application_status",
            state="application_ready",
            message=f"投递状态已更新为 {application.status}。",
            payload=self._application_payload(application, detail.job),
            execution_outcome="committed",
        )

    @staticmethod
    def _career_claim_status(evidence: Any) -> str:
        return "current" if evidence.is_current else "superseded"

    def _stored_proposal_failure(
        self,
        *,
        arguments: dict[str, Any],
        tool_name: str,
        proposal: Any,
        missing_state: str,
        missing_message: str,
    ) -> ToolObservation | None:
        """Recheck the exact persisted proposal before any confirm effect."""
        slot = CONFIRMATION_SPECS[tool_name].slot
        task = (
            self._conversation_store.get_task(
                str(arguments["user_id"]), str(arguments["conversation_id"])
            )
            if self._conversation_store is not None
            else None
        )
        if task is None or getattr(task, slot) != proposal:
            return ToolObservation(
                tool_name=tool_name,
                state=missing_state,
                message=missing_message,
                execution_outcome="not_committed",
            )
        if not task.pending_proposal_is_live(slot, datetime.now(timezone.utc)):
            return ToolObservation(
                tool_name=tool_name,
                state=missing_state,
                message=_EXPIRED_PROPOSAL_MESSAGE,
                execution_outcome="not_committed",
            )
        return None

    def _propose_career_fact(
        self, arguments: dict[str, Any]
    ) -> ToolObservation:
        if self._career_history_store is None:
            raise ValueError("Career history store is not configured")
        user_id = str(arguments["user_id"])
        record_id = str(arguments["career_record_id"])
        claim = str(arguments["claim"]).strip()
        reason = str(arguments["reason"]).strip()
        origin = arguments.get("origin", "agent_inference")
        source_user_quote = arguments.get("source_user_quote")
        source_user_interaction_id = arguments.get("source_user_interaction_id")
        if origin not in {"user_input", "agent_inference"}:
            raise ValueError("career fact origin is invalid")
        if origin == "user_input" and not source_user_quote:
            raise ValueError("user_input career fact requires a verified user quote")
        if origin == "agent_inference" and (
            source_user_quote is not None or source_user_interaction_id is not None
        ):
            raise ValueError("inferred career fact cannot carry user provenance")
        pending = next(
            (
                item
                for item in self._career_history_store.list_evidence(
                    user_id=user_id,
                    career_record_id=record_id,
                    verification_status="pending",
                )
                if item.claim.strip() == claim
                and item.origin == origin
                and item.source_user_quote == source_user_quote
                and item.source_user_interaction_id == source_user_interaction_id
            ),
            None,
        )
        if pending is None:
            pending = self._career_history_store.create_evidence(
                user_id=user_id,
                career_record_id=record_id,
                claim=claim,
                origin=origin,
                source_user_quote=source_user_quote,
                source_user_interaction_id=source_user_interaction_id,
            )
        proposal = CareerFactProposal(
            career_evidence_id=pending.id,
            career_record_id=record_id,
            claim=claim,
            reason=reason,
        )
        return _proposal_observation(
            tool_name="propose_career_fact",
            state="career_fact_proposed",
            message=clamp(
                "拟将下面这条事实记入所选职业经历：\n"
                f"{claim}\n"
                f"原因：{reason}\n"
                + (
                    f"来源：用户原话“{source_user_quote}”。\n"
                    if source_user_quote else "来源：Agent 推断。\n"
                )
                + "目前仅处于隔离态；确认后才会成为长期事实。",
                limit=DECISION_OBSERVATION_BODY_LIMIT,
            ),
            proposal=proposal,
            execution_outcome="committed",
        )

    def _confirm_career_fact(
        self, arguments: dict[str, Any]
    ) -> ToolObservation:
        if self._career_history_store is None:
            raise ValueError("Career history store is not configured")
        user_id = str(arguments["user_id"])
        proposal = CareerFactProposal.model_validate(arguments["proposal"])
        refusal = self._stored_proposal_failure(
            arguments=arguments,
            tool_name="confirm_career_fact",
            proposal=proposal,
            missing_state="career_fact_confirmation_missing",
            missing_message="这条事实尚未在前一轮展示并持久化，不能直接确认。",
        )
        if refusal is not None:
            return refusal
        evidence = self._career_history_store.get_evidence(
            user_id=user_id,
            career_evidence_id=proposal.career_evidence_id,
        )
        if (
            evidence is None
            or evidence.verification_status != "pending"
            or evidence.career_record_id != proposal.career_record_id
            or evidence.claim != proposal.claim
        ):
            return ToolObservation(
                tool_name="confirm_career_fact",
                state="career_fact_candidate_missing",
                message="待确认事实已经变化或不存在，请重新提案。",
                execution_outcome="not_committed",
            )
        confirmed = self._career_history_store.confirm_evidence(
            user_id=user_id,
            career_evidence_id=evidence.id,
            reason=proposal.reason,
        )
        return ToolObservation(
            tool_name="confirm_career_fact",
            state="career_fact_confirmed",
            message="已将这条事实确认为长期职业记忆。",
            payload={
                "career_evidence_id": confirmed.id,
                "career_record_id": confirmed.career_record_id,
                "claim": confirmed.claim,
                "detail_ref": confirmed.detail_ref,
                "scope_key": confirmed.scope_key,
            },
            execution_outcome="committed",
        )

    def _propose_memory_amendment(
        self, arguments: dict[str, Any]
    ) -> ToolObservation:
        if self._career_history_store is None:
            raise ValueError("Career history store is not configured")
        user_id = str(arguments["user_id"])
        proposal = MemoryAmendmentProposal.model_validate(arguments["proposal"])
        evidence = self._career_history_store.get_evidence_by_detail_ref(
            user_id=user_id,
            detail_ref=proposal.detail_ref,
        )
        if evidence is None or not evidence.is_current:
            return ToolObservation(
                tool_name="propose_memory_amendment",
                state="memory_amendment_target_not_found",
                message="没有找到这个当前职业声明；它可能已被删除或已被更正。",
                execution_outcome="not_committed",
            )
        body = (
            f"拟将职业声明 revision {evidence.revision} 更正为：\n"
            f"{proposal.new_claim}\n"
            f"原因：{proposal.reason}\n"
            "确认后才会写入新 revision，旧版本只保留在历史审计中。"
        )
        return _proposal_observation(
            tool_name="propose_memory_amendment",
            state="memory_amendment_proposed",
            message=clamp(body, limit=DECISION_OBSERVATION_BODY_LIMIT),
            proposal=proposal,
            execution_outcome="not_committed",
        )

    @staticmethod
    def _route_to_capability(arguments: dict[str, Any]) -> ToolObservation:
        current = arguments.get("current_tool_profile")
        model_arguments = RouteToCapabilityToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "current_tool_profile"}
        )
        domain = model_arguments.domain
        if current == domain:
            return ToolObservation(
                tool_name="route_to_capability",
                state="tool_profile_unchanged",
                message=f"当前已在 {domain} 工具档。",
                next_action="直接使用 task.available_now 中的工具，不要重复路由。",
                payload={"tool_profile": domain},
            )
        return ToolObservation(
            tool_name="route_to_capability",
            state="tool_profile_switched",
            message=f"工具档已切换为 {domain}。",
            next_action="根据更新后的 task.available_now 选择下一步工具。",
            payload={"tool_profile": domain},
        )

    def _update_working_notes(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._working_notes_store is None:
            raise ValueError("Working notes store is not configured")
        result = self._working_notes_store.replace(
            user_id=str(arguments["user_id"]),
            markdown=str(arguments["markdown"]),
            expected_revision=str(arguments["expected_revision"]),
        )
        if isinstance(result, WorkingNotesConflict):
            return ToolObservation(
                tool_name="update_working_notes",
                state="working_notes_stale",
                message="工作笔记已被另一会话更新，请基于当前内容合并后重试",
                payload={
                    "current_revision": result.current.revision,
                    "current_markdown": result.current.markdown,
                },
                execution_outcome="not_committed",
            )
        return ToolObservation(
            tool_name="update_working_notes",
            state="working_notes_updated",
            message=(
                "工作笔记已清空。"
                if not result.markdown
                else f"工作笔记已更新（{len(result.markdown)} 字符）。"
            ),
            payload={
                "chars": len(result.markdown),
                "revision": result.revision,
            },
            execution_outcome="committed",
        )

    def _confirm_memory_amendment(
        self, arguments: dict[str, Any]
    ) -> ToolObservation:
        if self._career_history_store is None:
            raise ValueError("Career history store is not configured")
        user_id = str(arguments["user_id"])
        proposal = MemoryAmendmentProposal.model_validate(arguments["proposal"])
        refusal = self._stored_proposal_failure(
            arguments=arguments,
            tool_name="confirm_memory_amendment",
            proposal=proposal,
            missing_state="memory_amendment_confirmation_missing",
            missing_message="这项更正尚未在前一轮展示并持久化，不能在提案同一轮写入。",
        )
        if refusal is not None:
            return refusal
        evidence = self._career_history_store.get_evidence_by_detail_ref(
            user_id=user_id,
            detail_ref=proposal.detail_ref,
        )
        if evidence is None:
            return ToolObservation(
                tool_name="confirm_memory_amendment",
                state="memory_amendment_target_not_found",
                message="没有找到待更正的当前职业声明；请重新读取并提案。",
                execution_outcome="not_committed",
            )
        correction = self._career_history_store.correct_evidence(
            user_id=user_id,
            career_evidence_id=evidence.id,
            new_claim=proposal.new_claim,
            reason=proposal.reason,
        )
        return ToolObservation(
            tool_name="confirm_memory_amendment",
            state="career_memory_amended",
            message=f"职业声明已更正为 revision {correction.current.revision}。",
            payload={
                "detail_ref": correction.current.detail_ref,
                "revision": correction.current.revision,
                "lineage_ref": career_evidence_lineage_ref(
                    user_id=user_id,
                    scope_key=correction.current.scope_key or "",
                ),
            },
            execution_outcome="committed",
        )

    def _propose_memory_tombstone(
        self, arguments: dict[str, Any]
    ) -> ToolObservation:
        if self._career_history_store is None:
            raise ValueError("Career history store is not configured")
        user_id = str(arguments["user_id"])
        proposal = MemoryTombstoneProposal.model_validate(arguments["proposal"])
        evidence = self._career_history_store.get_evidence_by_detail_ref(
            user_id=user_id,
            detail_ref=proposal.detail_ref,
        )
        if evidence is None or not evidence.is_current:
            return ToolObservation(
                tool_name="propose_memory_tombstone",
                state="memory_tombstone_target_not_found",
                message="没有找到这个当前职业声明；它可能已被删除或已被更正。",
                execution_outcome="not_committed",
            )
        proposal = proposal.model_copy(
            update={"expected_content_sha256": evidence.content_digest}
        )
        return _proposal_observation(
            tool_name="propose_memory_tombstone",
            state="memory_tombstone_proposed",
            message=(
                f"拟永久删除这条职业声明的完整更正谱系（当前 revision "
                f"{evidence.revision}）。删除后正文、来源引文和历史版本均不可恢复；"
                "如确认，请明确同意执行。"
            ),
            proposal=proposal,
            execution_outcome="not_committed",
        )

    def _confirm_memory_tombstone(
        self, arguments: dict[str, Any]
    ) -> ToolObservation:
        if self._career_history_store is None:
            raise ValueError("Career history store is not configured")
        user_id = str(arguments["user_id"])
        proposal = MemoryTombstoneProposal.model_validate(arguments["proposal"])
        conversation_id = str(arguments["conversation_id"])
        refusal = self._stored_proposal_failure(
            arguments=arguments,
            tool_name="confirm_memory_tombstone",
            proposal=proposal,
            missing_state="memory_tombstone_confirmation_missing",
            missing_message="这项永久删除尚未在前一轮展示并持久化，不能在提案同一轮执行。",
        )
        if refusal is not None:
            return refusal
        if proposal.target_kind == "intent_preference":
            tombstone_preference = getattr(
                self._conversation_store,
                "tombstone_free_text_preference",
                None,
            )
            if not callable(tombstone_preference):
                raise ValueError(
                    "Conversation store cannot tombstone free-text preferences"
                )
            committed = tombstone_preference(
                user_id=user_id,
                scope_key=str(proposal.scope_key),
                pref_scope=str(proposal.pref_scope),
                update_id=str(proposal.update_id),
                expected_content_sha256=str(
                    proposal.expected_content_sha256
                ),
                reason=proposal.reason,
            )
            if not committed:
                already_tombstoned = getattr(
                    self._conversation_store,
                    "free_text_preference_tombstone_matches",
                    None,
                )
                if not callable(already_tombstoned) or not already_tombstoned(
                    user_id=user_id,
                    scope_key=str(proposal.scope_key),
                    pref_scope=str(proposal.pref_scope),
                    update_id=str(proposal.update_id),
                    content_digest=str(proposal.expected_content_sha256),
                ):
                    return ToolObservation(
                        tool_name="confirm_memory_tombstone",
                        state="memory_tombstone_target_changed",
                        message=(
                            "待删除的偏好在确认前已变化；"
                            "请重新读取并提案。"
                        ),
                        execution_outcome="not_committed",
                    )
            memory_entry_id = intent_entry_id(
                str(proposal.scope_key),
                str(proposal.pref_scope),
            )
            working_notes_cleared = False
            cleanup_stage = "working_notes"
            try:
                if self._working_notes_store is not None:
                    working_notes_cleared = self._working_notes_store.clear(
                        user_id=user_id
                    )
                cleanup_stage = "derived_memory"
                cleanup_scope_key = memory_entry_id
                list_versions = getattr(
                    self._conversation_store,
                    "list_profile_intent_versions",
                    None,
                )
                if callable(list_versions):
                    surviving = list_versions(
                        user_id=user_id,
                        scope_key=str(proposal.scope_key),
                    )
                    if not any(
                        item.superseded_at is None
                        and item.admission_status in {"active", "quarantined"}
                        for item in surviving
                    ):
                        cleanup_scope_key = str(proposal.scope_key)
                cleanup_counts = self._conversation_store.purge_derived_memory(
                    user_id=user_id,
                    scope_key=cleanup_scope_key,
                    update_ids=(str(proposal.update_id),),
                )
            except (OSError, sqlite3.Error, ValueError):
                logger.exception(
                    "Preference tombstone cleanup failed",
                    extra={"user_id": user_id, "scope_key": proposal.scope_key},
                )
                return ToolObservation(
                    tool_name="confirm_memory_tombstone",
                    state="memory_tombstone_cleanup_incomplete",
                    message=(
                        "这条长期偏好已删除，但派生清理本次未完成；"
                        "确认请求已保留，可直接重试。"
                    ),
                    payload={
                        "target_kind": "intent_preference",
                        "memory_entry_id": memory_entry_id,
                        "working_notes_cleared": working_notes_cleared,
                        "cleanup_incomplete": cleanup_stage,
                    },
                    execution_outcome="committed",
                )
            cleanup_counts["working_notes"] = int(working_notes_cleared)
            return ToolObservation(
                tool_name="confirm_memory_tombstone",
                state="memory_tombstoned",
                message="这条长期偏好已删除，相关派生记忆已按条目清理。",
                payload={
                    "target_kind": "intent_preference",
                    "memory_entry_id": memory_entry_id,
                    "cleanup_scope_key": cleanup_scope_key,
                    "derived_cleanup": cleanup_counts,
                },
                execution_outcome="committed",
            )
        evidence = self._career_history_store.get_evidence_by_detail_ref(
            user_id=user_id,
            detail_ref=str(proposal.detail_ref),
        )
        tombstone = None
        if evidence is None:
            tombstoned_scope = next(
                (
                    (scope_key, markers)
                    for owner_id, scope_key, markers in (
                        self._career_history_store.list_tombstoned_scopes()
                    )
                    if owner_id == user_id and proposal.detail_ref in markers
                ),
                None,
            )
            if tombstoned_scope is None:
                return ToolObservation(
                    tool_name="confirm_memory_tombstone",
                    state="memory_tombstone_target_not_found",
                    message="没有找到待删除的当前职业声明；它可能已经被处理。",
                    execution_outcome="not_committed",
                )
            scope_key, lineage_markers = tombstoned_scope
            tombstoned_update_ids = (
                self._career_history_store.list_tombstoned_update_ids(
                    user_id=user_id,
                    scope_key=scope_key,
                )
            )
            tombstone = next(
                (
                    item
                    for item in self._career_history_store.list_evidence_tombstones(
                        user_id=user_id,
                        limit=500,
                    )
                    if item.scope_key == scope_key
                ),
                None,
            )
            if tombstone is None:
                raise RuntimeError("tombstoned scope is missing its audit row")
            lineage = ()
        else:
            lineage = (
                self._career_history_store.list_evidence_lineage(
                    user_id=user_id,
                    scope_key=evidence.scope_key,
                )
                if evidence.scope_key is not None
                else ()
            )
            lineage_markers = tuple(
                dict.fromkeys(
                    (
                        career_evidence_lineage_ref(
                            user_id=user_id, scope_key=evidence.scope_key or ""
                        ),
                        *(
                            marker
                            for item in lineage
                            for marker in (item.detail_ref, item.source_ref)
                            if marker is not None
                        ),
                    )
                )
            )
            tombstoned_update_ids = tuple(
                item.update_id
                for item in lineage
                if item.update_id is not None
            )
            try:
                tombstone = self._career_history_store.tombstone_evidence(
                    user_id=user_id,
                    career_evidence_id=evidence.id,
                    reason=proposal.reason,
                    actor_type="user",
                    expected_content_sha256=proposal.expected_content_sha256,
                )
            except ValueError:
                return ToolObservation(
                    tool_name="confirm_memory_tombstone",
                    state="memory_tombstone_target_changed",
                    message="待删除的职业声明在确认前已变化；请重新读取并提案。",
                    execution_outcome="not_committed",
                )
        if lineage:
            record_active_trace(
                "memory_tombstone_observed",
                "career_evidence_tombstone",
                outcome="succeeded",
                details={
                    "conversation_key": conversation_trace_key(
                        user_id,
                        conversation_id,
                    ),
                    "p1_version_binding": True,
                    "entries": [
                        {
                            "entry_id": item.scope_key,
                            "update_id": item.update_id,
                            "content_digest": item.content_digest,
                            "revision": item.revision,
                            "lifecycle_status": "tombstoned",
                        }
                        for item in lineage
                        if item.scope_key is not None
                        and item.update_id is not None
                        and item.content_digest is not None
                        and item.revision is not None
                    ],
                },
            )
        purge = getattr(self._conversation_store, "purge_derived_memory", None)
        working_notes_cleared = False
        cleanup_stage = "working_notes"
        try:
            if self._working_notes_store is not None:
                working_notes_cleared = self._working_notes_store.clear(
                    user_id=user_id
                )
            cleanup_stage = "derived_memory"
        except (OSError, sqlite3.Error, ValueError):
            logger.exception(
                "Tombstone working-notes cleanup failed",
                extra={"user_id": user_id, "scope_key": tombstone.scope_key},
            )
            return ToolObservation(
                tool_name="confirm_memory_tombstone",
                state="memory_tombstone_cleanup_incomplete",
                message=(
                    "职业声明正文及谱系已永久删除；"
                    "工作笔记清理本次未完成，确认请求已保留。"
                ),
                payload={
                    "lineage_size": len(tombstone.evidence_ids),
                    "working_notes_cleared": False,
                    "cleanup_incomplete": cleanup_stage,
                },
                execution_outcome="committed",
            )
        if not callable(purge):
            return ToolObservation(
                tool_name="confirm_memory_tombstone",
                state="memory_tombstone_cleanup_incomplete",
                message=(
                    "职业声明正文及谱系已永久删除，但当前运行时没有派生记忆存储；"
                    "确认请求已保留，可在存储恢复后重试。"
                ),
                payload={
                    "lineage_size": len(tombstone.evidence_ids),
                    "working_notes_cleared": working_notes_cleared,
                    "cleanup_incomplete": "derived_memory",
                },
                execution_outcome="committed",
            )
        try:
            if self._semantic_evidence_cache is not None:
                self._semantic_evidence_cache.forget_evidence_ids(
                    tombstone.evidence_ids
                )
            cleanup_counts = purge(
                user_id=user_id,
                scope_key=tombstone.scope_key,
                lineage_markers=lineage_markers,
                update_ids=tombstoned_update_ids,
            )
            cleanup_counts["working_notes"] = int(working_notes_cleared)
        except (OSError, sqlite3.Error, ValueError):
            logger.exception(
                "Tombstone derived-memory cleanup failed",
                extra={
                    "user_id": user_id,
                    "scope_key": tombstone.scope_key,
                },
            )
            return ToolObservation(
                tool_name="confirm_memory_tombstone",
                state="memory_tombstone_cleanup_incomplete",
                message=(
                    "职业声明正文及谱系已永久删除；派生记忆清理本次未完成，"
                    "确认请求已保留，可直接重试。"
                ),
                payload={
                    "lineage_size": len(tombstone.evidence_ids),
                    "working_notes_cleared": working_notes_cleared,
                    "cleanup_incomplete": "derived_memory",
                },
                execution_outcome="committed",
            )
        return ToolObservation(
            tool_name="confirm_memory_tombstone",
            state="memory_tombstoned",
            message=(
                "职业声明及其完整更正谱系已永久删除；相关摘要会从未抑制的"
                "原始消息重建，无其他存活关联的情节记忆已清理。"
            ),
            payload={
                "lineage_size": len(tombstone.evidence_ids),
                "derived_cleanup": cleanup_counts,
            },
            execution_outcome="committed",
        )

    def _get_career_memory_detail(
        self, arguments: dict[str, Any]
    ) -> ToolObservation:
        if self._career_history_store is None:
            raise ValueError("Career history store is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = GetCareerMemoryDetailToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        evidence = self._career_history_store.get_evidence_by_detail_ref(
            user_id=user_id,
            detail_ref=model_arguments.detail_ref,
        )
        if evidence is None or not evidence.is_current or evidence.scope_key is None:
            return ToolObservation(
                tool_name="get_career_memory_detail",
                state="career_memory_detail_not_found",
                message="没有找到这个当前职业声明详情；历史声明请改用历史查询。",
                payload={"detail_ref": model_arguments.detail_ref},
            )
        lineage = self._career_history_store.list_evidence_lineage(
            user_id=user_id,
            scope_key=evidence.scope_key,
        )
        record = self._career_history_store.get_record(
            user_id=user_id,
            career_record_id=evidence.career_record_id,
        )
        lineage_ref = career_evidence_lineage_ref(
            user_id=user_id,
            scope_key=evidence.scope_key,
        )
        entries = []
        for item in lineage:
            status = self._career_claim_status(item)
            changed_at = item.superseded_at
            entries.append(
                {
                    "revision": item.revision,
                    "claim": item.claim,
                    "claim_status": status,
                    "valid_from": (
                        item.valid_from.isoformat() if item.valid_from else None
                    ),
                    **(
                        {"status_changed_at": changed_at.isoformat()}
                        if changed_at is not None
                        else {}
                    ),
                }
            )
        supported_by = [evidence.source_ref] if evidence.source_ref else []
        body = (
            f"当前职业声明详情（revision {evidence.revision}）\n"
            f"履历项：{record.title if record is not None else '未找到'}\n"
            f"声明：{evidence.claim}\n"
            f"直接支持：{', '.join(supported_by) if supported_by else '无直接来源引文'}\n"
            f"谱系引用：{lineage_ref}\n"
            "更正谱系：\n"
            + "\n".join(
                f"- r{item['revision']} [{item['claim_status']}] {item['claim']}"
                + (
                    f"（状态变更时间：{item['status_changed_at']}）"
                    if "status_changed_at" in item
                    else ""
                )
                for item in entries
            )
        )
        body_clipped = len(body) > DECISION_OBSERVATION_BODY_LIMIT
        if body_clipped:
            body = clamp(body, limit=DECISION_OBSERVATION_BODY_LIMIT)
        return ToolObservation(
            tool_name="get_career_memory_detail",
            state="career_memory_detail_found",
            message=(
                f"已读取当前职业声明 revision {evidence.revision}；"
                f"谱系共 {len(entries)} 条。"
            ),
            facts={
                "revision": evidence.revision or 1,
                "support_count": len(supported_by),
                "lineage_count": len(entries),
                "body_clipped": body_clipped,
            },
            payload={
                "detail_ref": model_arguments.detail_ref,
                "lineage_ref": lineage_ref,
                "supported_by": supported_by,
                "lineage": entries,
                "body": body,
                "body_clipped": body_clipped,
            },
        )

    def _search_career_episodes(
        self, arguments: dict[str, Any]
    ) -> ToolObservation:
        if self._episode_store is None:
            raise ValueError("Career episode store is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = SearchCareerEpisodesToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.detail_ref is not None:
            episode = self._episode_store.get(
                user_id=user_id,
                episode_id=model_arguments.detail_ref.removeprefix("episode:"),
            )
            episodes = (episode,) if episode is not None else ()
        else:
            episodes = self._episode_store.search(
                user_id=user_id,
                query=model_arguments.query,
                limit=model_arguments.top_k,
                start_datetime=model_arguments.start_datetime,
                end_datetime=model_arguments.end_datetime,
                kinds=model_arguments.kinds,
            )
        self._episode_store.mark_accessed(
            user_id=user_id,
            episode_ids=tuple(episode.id for episode in episodes),
        )
        items = [
            {
                "kind": episode.kind,
                "occurred_at": episode.occurred_at.isoformat(),
                "title": episode.title,
                "summary": episode.summary,
                "conversation_id": episode.conversation_id,
                "resource_refs": [
                    reference.model_dump(mode="json")
                    for reference in episode.resource_refs
                ],
            }
            for episode in episodes
        ]
        recovered_refs = []
        for episode in episodes:
            for reference in episode.resource_refs:
                try:
                    recovered_refs.append(
                        ConversationResourceReference.model_validate(
                            reference.model_dump(mode="json")
                        )
                    )
                except ValueError:
                    continue
        if not items:
            return ToolObservation(
                tool_name="search_career_episodes",
                state="career_episode_search_empty",
                message="没有找到匹配的过往求职事件。",
                payload={"items": []},
            )
        body = "过往求职事件：\n" + "\n".join(
            f"- {item['occurred_at']} [{item['kind']}] "
            f"{item['title']}：{item['summary']}"
            for item in items
        )
        body_clipped = len(body) > DECISION_OBSERVATION_BODY_LIMIT
        if body_clipped:
            body = clamp(body, limit=DECISION_OBSERVATION_BODY_LIMIT)
        return ToolObservation(
            tool_name="search_career_episodes",
            state="career_episode_search_found",
            message=f"找到 {len(items)} 条匹配的过往求职事件。",
            facts={
                "returned": len(items),
                "body_clipped": body_clipped,
            },
            payload={
                "items": items,
                "body": body,
                "body_clipped": body_clipped,
            },
            resource_refs=tuple(
                {
                    reference.resource_id: reference
                    for reference in recovered_refs
                }.values()
            ),
        )

    def _search_career_memory(
        self, arguments: dict[str, Any]
    ) -> ToolObservation:
        if self._career_history_store is None:
            raise ValueError("Career history store is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = SearchCareerMemoryToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        try:
            evidence, total, next_cursor = (
                self._career_history_store.search_current_evidence(
                    user_id=user_id,
                    query=model_arguments.query,
                    limit=model_arguments.limit,
                    cursor=model_arguments.cursor,
                )
            )
        except ValueError:
            return ToolObservation(
                tool_name="search_career_memory",
                state="invalid_input",
                message="职业记忆查询词或分页游标无效；请用聚焦词重新从第一页查询。",
                payload={"query": model_arguments.query},
            )
        items = [
            {
                "claim": item.claim,
                "revision": item.revision,
                "detail_ref": item.detail_ref,
                "source_ref": item.source_ref,
            }
            for item in evidence
        ]
        if not items:
            return ToolObservation(
                tool_name="search_career_memory",
                state="career_memory_search_empty",
                message="没有找到匹配的当前职业声明。",
                payload={
                    "query": model_arguments.query,
                    "items": [],
                    "total": total,
                },
            )
        body = "当前职业声明（来自归档层）：\n" + "\n".join(
            f"- r{item['revision']} {item['claim']}" for item in items
        )
        body_clipped = len(body) > DECISION_OBSERVATION_BODY_LIMIT
        if body_clipped:
            body = clamp(body, limit=DECISION_OBSERVATION_BODY_LIMIT)
        return ToolObservation(
            tool_name="search_career_memory",
            state="career_memory_search_found",
            message=f"从归档层找到 {len(items)}/{total} 条当前职业声明。",
            facts={
                "returned": len(items),
                "total": total,
                "body_clipped": body_clipped,
                **({"next_cursor": next_cursor} if next_cursor is not None else {}),
            },
            payload={
                "query": model_arguments.query,
                "items": items,
                "total": total,
                **({"next_cursor": next_cursor} if next_cursor is not None else {}),
                "body": body,
                "body_clipped": body_clipped,
            },
        )

    def _search_career_history(
        self, arguments: dict[str, Any]
    ) -> ToolObservation:
        if self._career_history_store is None:
            raise ValueError("Career history store is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = SearchCareerHistoryToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        try:
            evidence, total, next_cursor = (
                self._career_history_store.search_historical_evidence(
                    user_id=user_id,
                    query=model_arguments.query,
                    limit=model_arguments.limit,
                    cursor=model_arguments.cursor,
                )
            )
        except ValueError:
            return ToolObservation(
                tool_name="search_career_history",
                state="invalid_input",
                message="历史查询词或分页游标无效；请用聚焦词重新从第一页查询。",
                payload={"query": model_arguments.query},
            )
        items = []
        for item in evidence:
            status = self._career_claim_status(item)
            changed_at = item.superseded_at
            items.append(
                {
                    "claim": item.claim,
                    "revision": item.revision,
                    "claim_status": status,
                    **(
                        {"status_changed_at": changed_at.isoformat()}
                        if changed_at is not None
                        else {}
                    ),
                }
            )
        if not items:
            return ToolObservation(
                tool_name="search_career_history",
                state="career_history_empty",
                message="没有找到匹配的历史职业声明。",
                payload={
                    "query": model_arguments.query,
                    "items": [],
                    "total": total,
                },
            )
        body = "历史职业声明（不能作为当前值使用）：\n" + "\n".join(
            f"- r{item['revision']} [{item['claim_status']}] {item['claim']}"
            f"（状态变更时间：{item.get('status_changed_at', '未记录')}）"
            for item in items
        )
        body_clipped = len(body) > DECISION_OBSERVATION_BODY_LIMIT
        if body_clipped:
            body = clamp(body, limit=DECISION_OBSERVATION_BODY_LIMIT)
        return ToolObservation(
            tool_name="search_career_history",
            state="career_history_found",
            message=f"找到 {len(items)}/{total} 条匹配的历史职业声明。",
            facts={
                "returned": len(items),
                "total": total,
                "body_clipped": body_clipped,
                **({"next_cursor": next_cursor} if next_cursor is not None else {}),
            },
            payload={
                "query": model_arguments.query,
                "items": items,
                "total": total,
                **({"next_cursor": next_cursor} if next_cursor is not None else {}),
                "body": body,
                "body_clipped": body_clipped,
            },
        )

    def _resolve_claim_source(
        self, arguments: dict[str, Any]
    ) -> ToolObservation:
        if self._career_history_store is None:
            raise ValueError("Career history store is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = ResolveClaimSourceToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        evidence = self._career_history_store.get_evidence_by_source_ref(
            user_id=user_id,
            source_ref=model_arguments.source_ref,
        )
        if evidence is None:
            return ToolObservation(
                tool_name="resolve_claim_source",
                state="claim_source_not_found",
                message="没有找到这条已确认声明的来源证据。",
                payload={"source_ref": model_arguments.source_ref},
            )

        resume_display_name = "已归档简历版本"
        if (
            self._resume_store is not None
            and evidence.source_resume_version_id is not None
        ):
            source = self._resume_store.get_version(
                user_id=user_id,
                resume_version_id=evidence.source_resume_version_id,
            )
            if source is not None:
                resume, version = source
                resume_display_name = condense(
                    f"{resume.name} · 第 {version.version_number} 版",
                    limit=80,
                )

        source_quote = evidence.source_quote or ""
        heading = "原始证据引文：\n\n"
        body_clipped = (
            len(heading) + len(source_quote) > DECISION_OBSERVATION_BODY_LIMIT
        )
        if body_clipped:
            source_quote = clamp(
                source_quote,
                limit=DECISION_OBSERVATION_BODY_LIMIT - len(heading),
            )
        claim_status = self._career_claim_status(evidence)
        status_changed_at = evidence.superseded_at
        return ToolObservation(
            tool_name="resolve_claim_source",
            state="claim_source_found",
            message=(
                "已读取历史来源证据；它对应的声明已经被更正，不能作为当前声明的支持。"
                if claim_status == "superseded"
                else "已读取这条声明对应的原始来源证据。"
            ),
            facts={
                "origin": evidence.origin,
                "recorded_at": evidence.created_at.isoformat(),
                "source_locator": condense(
                    evidence.source_locator or "未提供", limit=80
                ),
                "resume_version": resume_display_name,
                "claim_status": claim_status,
                **(
                    {"status_changed_at": status_changed_at.isoformat()}
                    if status_changed_at is not None
                    else {}
                ),
                "body_clipped": body_clipped,
            },
            payload={
                "source_ref": model_arguments.source_ref,
                "source_quote": source_quote,
                "body_clipped": body_clipped,
            },
        )

    def _fetch_archived_constraints(
        self, arguments: dict[str, Any]
    ) -> ToolObservation:
        if self._conversation_store is None:
            raise ValueError("Conversation store is not configured")
        archived = self._conversation_store.list_conversation_constraints(
            user_id=str(arguments["user_id"]),
            conversation_id=str(arguments["conversation_id"]),
            statuses=("omitted",),
        )
        if not archived:
            return ToolObservation(
                tool_name="fetch_archived_constraints",
                state="no_archived_constraints",
                message=(
                    "没有被折叠的约束；conversation_summary 里显示的就是全部。"
                ),
                execution_outcome="not_committed",
            )
        # The texts go in the message, not the payload: the archive exists so
        # the model can see what the visible cap held back, and payload is not
        # part of the observation it reads.
        listed = "\n".join(f"- {row.text}" for row in archived)
        return ToolObservation(
            tool_name="fetch_archived_constraints",
            state="archived_constraints_ready",
            message=(
                f"这条对话还有 {len(archived)} 条约束因为显示上限被折叠，"
                f"它们仍然生效：\n{listed}"
            ),
            execution_outcome="not_committed",
        )

    def _propose_constraint_retirement(
        self, arguments: dict[str, Any]
    ) -> ToolObservation:
        if self._conversation_store is None:
            raise ValueError("Conversation store is not configured")
        proposal = ConstraintRetirementProposal.model_validate(
            arguments["proposal"]
        )
        live = self._conversation_store.list_conversation_constraints(
            user_id=str(arguments["user_id"]),
            conversation_id=str(arguments["conversation_id"]),
            statuses=("active", "omitted"),
        )
        if all(row.text != proposal.constraint for row in live):
            return ToolObservation(
                tool_name="propose_constraint_retirement",
                state="constraint_not_found",
                message=(
                    "这条约束不在本次对话的生效约束里；请用 "
                    "conversation_summary.active_constraints 或 "
                    "fetch_archived_constraints 给出的原文。"
                ),
                execution_outcome="not_committed",
            )
        return _proposal_observation(
            tool_name="propose_constraint_retirement",
            state="constraint_retirement_proposed",
            message=(
                f"拟停止应用这条约束：「{proposal.constraint}」。"
                "确认后它不再进入后续回答，之后的摘要重写也不会把它带回来；"
                "原始对话消息仍然保留。如确认，请明确同意执行。"
            ),
            proposal=proposal,
            execution_outcome="not_committed",
        )

    def _confirm_constraint_retirement(
        self, arguments: dict[str, Any]
    ) -> ToolObservation:
        if self._conversation_store is None:
            raise ValueError("Conversation store is not configured")
        user_id = str(arguments["user_id"])
        conversation_id = str(arguments["conversation_id"])
        proposal = ConstraintRetirementProposal.model_validate(
            arguments["proposal"]
        )
        refusal = self._stored_proposal_failure(
            arguments=arguments,
            tool_name="confirm_constraint_retirement",
            proposal=proposal,
            missing_state="constraint_retirement_confirmation_missing",
            missing_message="这项约束退休尚未在前一轮展示并持久化，不能在提案同一轮执行。",
        )
        if refusal is not None:
            return refusal
        retired = self._conversation_store.retire_conversation_constraint(
            user_id=user_id,
            conversation_id=conversation_id,
            constraint_text=proposal.constraint,
        )
        if not retired:
            return ToolObservation(
                tool_name="confirm_constraint_retirement",
                state="constraint_not_found",
                message="这条约束已经不再生效；无需重复退休。",
                execution_outcome="not_committed",
            )
        return ToolObservation(
            tool_name="confirm_constraint_retirement",
            state="constraint_retired",
            message=f"已停止应用这条约束：「{proposal.constraint}」。",
            execution_outcome="committed",
        )

    def _read_conversation_span(
        self, arguments: dict[str, Any]
    ) -> ToolObservation:
        if self._conversation_store is None:
            raise ValueError("Conversation store is not configured")
        user_id = str(arguments["user_id"])
        conversation_id = str(arguments["conversation_id"])
        model_arguments = ReadConversationSpanToolArguments.model_validate(
            {
                key: value
                for key, value in arguments.items()
                if key not in {"user_id", "conversation_id"}
            }
        )
        span_arguments = {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "from_sequence": model_arguments.from_sequence,
            "through_sequence": model_arguments.through_sequence,
        }
        if model_arguments.query is not None:
            span_arguments["query"] = model_arguments.query
        span = self._conversation_store.read_conversation_span(**span_arguments)
        body_clipped = (
            len(render_conversation_span(span)) > DECISION_OBSERVATION_BODY_LIMIT
        )
        if body_clipped:
            span = span.model_copy(update={"body_clipped": True})
        content_clipped = any(item.content_clipped for item in span.messages)
        values = {
            "tool_name": "read_conversation_span",
            "message": (
                f"已读取会话序号 {span.from_sequence}–{span.through_sequence}："
                f"返回 {span.returned}/{span.total} 条。"
            ),
            "facts": {
                "from_sequence": span.from_sequence,
                "through_sequence": span.through_sequence,
                "returned": span.returned,
                "total": span.total,
                "body_clipped": body_clipped,
                "content_clipped": content_clipped,
                "resource_ref_count": len(span.resource_refs),
                "resource_ref_total": span.resource_ref_total,
            },
            "payload": span.model_dump(mode="json"),
            "resource_refs": span.resource_refs,
        }
        if span.returned:
            return ToolObservation(state="conversation_span_found", **values)
        return ToolObservation(state="conversation_span_empty", **values)

    def _list_applications(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._application_service is None:
            raise ValueError("Application service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = ListApplicationsToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        items = self._application_service.list_applications(
            user_id=user_id,
            statuses=model_arguments.statuses,
            limit=model_arguments.limit,
        )
        return ToolObservation(
            tool_name="list_applications",
            state="applications_found" if items else "no_applications_found",
            message=f"找到 {len(items)} 条投递记录。" if items else "当前没有匹配的投递记录。",
            payload={
                "statuses": model_arguments.statuses,
                "items": [
                    {
                        "selection_index": index,
                        **self._application_payload(item.application, item.job),
                    }
                    for index, item in enumerate(items, start=1)
                ],
            },
        )

    def _get_application(self, arguments: dict[str, Any]) -> ToolObservation:
        if self._application_service is None:
            raise ValueError("Application service is not configured")
        user_id = str(arguments["user_id"])
        model_arguments = GetApplicationToolArguments.model_validate(
            {key: value for key, value in arguments.items() if key != "user_id"}
        )
        if model_arguments.application_id is None:
            raise ValueError("get_application requires application_id")
        try:
            detail = self._application_service.get_application(
                user_id=user_id,
                application_id=model_arguments.application_id,
            )
        except ApplicationInputNotFoundError:
            return ToolObservation(
                tool_name="get_application",
                state="application_not_found",
                message="没有找到这条投递记录，或它不属于当前用户。",
                payload={"application_id": model_arguments.application_id},
            )
        return ToolObservation(
            tool_name="get_application",
            state="application_ready",
            message=f"已读取 {detail.job.posting.company_name} 的投递记录。",
            payload={
                **self._application_payload(detail.application, detail.job),
                "events": [
                    {
                        "event_type": event.event_type,
                        "source": event.source,
                        "previous_status": event.previous_status,
                        "new_status": event.new_status,
                        "note": event.note,
                        "occurred_at": event.occurred_at.isoformat(),
                    }
                    for event in detail.events
                ],
            },
        )

    @staticmethod
    def _application_payload(application, job) -> dict[str, Any]:
        return {
            "application_id": application.id,
            "job_posting_id": application.job_posting_id,
            "resume_version_id": application.resume_version_id,
            "jd_snapshot_id": application.jd_snapshot_id,
            "title": job.posting.title,
            "company_name": job.posting.company_name,
            "status": application.status,
            "submitted_at": application.submitted_at.isoformat(),
            "created_at": application.created_at.isoformat(),
            "updated_at": application.updated_at.isoformat(),
        }

    def _tailoring_observation(
        self,
        *,
        user_id: str,
        tool_name: str,
        draft: StoredResumeTailoringDraft,
        message: str,
        execution_outcome: Literal["committed", "not_committed", "unknown"] | None = None,
    ) -> ToolObservation:
        title, description = self._tailoring_metadata(
            user_id=user_id, draft=draft
        )
        return ToolObservation(
            tool_name=tool_name,
            state="resume_tailoring_draft_ready",
            message=message,
            execution_outcome=execution_outcome,
            payload={
                "draft_id": draft.id,
                "match_id": draft.match_id,
                "parent_draft_id": draft.parent_draft_id,
                "revision_number": draft.revision_number,
                "status": draft.status,
                "tailoring_goal": draft.tailoring_goal,
                "expires_at": draft.expires_at.isoformat(),
                "automated_review": (
                    {
                        "status": draft.automated_review.status,
                        "attempt_count": len(draft.automated_review.attempts),
                        "warning_categories": [
                            issue.category
                            for issue in draft.automated_review.attempts[-1].result.issues
                            if issue.severity == "warning"
                        ],
                    }
                    if draft.automated_review is not None
                    else None
                ),
                "change_reviews": [
                    review.model_dump(mode="json") for review in draft.change_reviews
                ],
                "pending_change_indices": draft.pending_change_indices,
                "strategy_summary": draft.result.strategy_summary,
                "changes": [
                    {
                        "change_index": index,
                        **change.model_dump(mode="json"),
                    }
                    for index, change in enumerate(draft.result.changes, start=1)
                ],
                "preserved_strengths": draft.result.preserved_strengths,
                "unresolved_gaps": draft.result.unresolved_gaps,
                "gap_mitigations": [
                    item.model_dump(mode="json")
                    for item in draft.result.gap_mitigations
                ],
                "clarification_questions": draft.result.clarification_questions,
                "warnings": draft.result.warnings,
            },
            resource_ref=ConversationResourceReference(
                kind="resume_tailoring_draft",
                resource_id=draft.id,
                title=title,
                description=description,
            ),
        )
