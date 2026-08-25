from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Protocol

from pydantic import Field, model_validator

from career_agent.agent.job_discovery_contracts import JobDiscoveryRequest
from career_agent.agent.conversation_memory_contracts import ConversationSummaryContent
from career_agent.domain.applications import ApplicationStatus
from career_agent.domain.action_center import ActionSourceType, ActionStatus, ActionType
from career_agent.domain.email_tracking import EmailEventStatus
from career_agent.domain.interviews import InterviewDetails, InterviewStatus
from career_agent.domain.job_discovery import ContractModel


class CareerProfileContext(ContractModel):
    user_id: str
    target_roles: tuple[str, ...] = ()
    default_city: str | None = None
    salary_preference: str | None = None
    experience: str | None = None
    education: str | None = None


class AgentPreferencesContext(ContractModel):
    boss_search: Literal["explicit_request_only", "allowed"] = "explicit_request_only"


class CandidateContextItem(ContractModel):
    result_ref: str
    title: str
    company_name: str
    city: str | None = None
    salary: str | None = None


class ApplicationCandidateContextItem(ContractModel):
    application_id: str
    title: str
    company_name: str
    status: ApplicationStatus


class InterviewCandidateContextItem(ContractModel):
    interview_round_id: str
    application_id: str
    sequence_number: int
    employer_label: str | None = None
    status: InterviewStatus
    scheduled_start: datetime | None = None


class ActionCandidateContextItem(ContractModel):
    action_item_id: str
    action_type: ActionType
    source_type: ActionSourceType
    source_id: str
    title: str
    status: ActionStatus
    due_at: datetime | None = None


class CalendarAccountCandidateContextItem(ContractModel):
    calendar_account_id: str
    provider: Literal["google"]
    email_address: str
    calendar_id: str


class ConversationTaskState(ContractModel):
    active_workflow: Literal["job_discovery", "email_tracking", "none"] = "none"
    run_id: str | None = None
    phase: str | None = None
    selected_result_ref: str | None = None
    manual_search_query: str | None = None
    candidates: tuple[CandidateContextItem, ...] = ()
    active_resume_analysis_id: str | None = None
    resume_analysis_status: Literal["pending", "confirmed"] | None = None
    active_resume_job_match_id: str | None = None
    resume_job_match_status: Literal["ready"] | None = None
    active_resume_tailoring_draft_id: str | None = None
    resume_tailoring_status: Literal[
        "pending", "in_review", "reviewed", "finalized"
    ] | None = None
    active_resume_version_id: str | None = None
    active_resume_artifact_id: str | None = None
    active_job_posting_id: str | None = None
    active_application_id: str | None = None
    active_application_status: ApplicationStatus | None = None
    application_candidates: tuple[ApplicationCandidateContextItem, ...] = ()
    active_interview_round_id: str | None = None
    interview_candidates: tuple[InterviewCandidateContextItem, ...] = ()
    active_interview_preparation_id: str | None = None
    active_action_item_id: str | None = None
    action_candidates: tuple[ActionCandidateContextItem, ...] = ()
    active_calendar_proposal_id: str | None = None
    calendar_account_candidates: tuple[CalendarAccountCandidateContextItem, ...] = ()


class ConversationMessageContext(ContractModel):
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime


class CareerMemoryRecord(ContractModel):
    record_type: Literal[
        "education",
        "work",
        "internship",
        "project",
        "certification",
    ]
    organization: str | None = None
    title: str
    start_year: int | None = None
    start_month: int | None = None
    end_year: int | None = None
    end_month: int | None = None
    is_current: bool = False
    confirmed_highlights: tuple[str, ...] = ()


class CareerMemoryContext(ContractModel):
    records: tuple[CareerMemoryRecord, ...] = ()


class ToolObservation(ContractModel):
    tool_name: str
    state: str
    message: str
    next_action: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class MainAgentContext(ContractModel):
    conversation_id: str
    profile: CareerProfileContext
    preferences: AgentPreferencesContext = AgentPreferencesContext()
    task: ConversationTaskState = ConversationTaskState()
    career_memory: CareerMemoryContext = CareerMemoryContext()
    recent_messages: tuple[ConversationMessageContext, ...] = ()
    tool_observations: tuple[ToolObservation, ...] = ()
    conversation_summary: ConversationSummaryContent | None = None
    user_message: str = Field(min_length=1)

    def model_context(self) -> dict[str, Any]:
        return {
            "career_profile": {
                "target_roles": self.profile.target_roles,
                "default_city": self.profile.default_city,
                "salary_preference": self.profile.salary_preference,
                "experience": self.profile.experience,
                "education": self.profile.education,
                "records": [
                    record.model_dump(mode="json")
                    for record in self.career_memory.records
                ],
            },
            "preferences": self.preferences.model_dump(mode="json"),
            "task": {
                "active_workflow": self.task.active_workflow,
                "phase": self.task.phase,
                "manual_search_query": self.task.manual_search_query,
                "candidates": [
                    {
                        "selection_index": index,
                        "title": candidate.title,
                        "company_name": candidate.company_name,
                        "city": candidate.city,
                        "salary": candidate.salary,
                    }
                    for index, candidate in enumerate(self.task.candidates, start=1)
                ],
                "resume_analysis_status": self.task.resume_analysis_status,
                "resume_job_match_status": self.task.resume_job_match_status,
                "resume_tailoring_status": self.task.resume_tailoring_status,
                "active_application_status": self.task.active_application_status,
                "application_candidates": [
                    {
                        "selection_index": index,
                        "title": candidate.title,
                        "company_name": candidate.company_name,
                        "status": candidate.status,
                    }
                    for index, candidate in enumerate(
                        self.task.application_candidates, start=1
                    )
                ],
                "interview_candidates": [
                    {
                        "selection_index": index,
                        "sequence_number": candidate.sequence_number,
                        "employer_label": candidate.employer_label,
                        "status": candidate.status,
                        "scheduled_start": (
                            candidate.scheduled_start.isoformat()
                            if candidate.scheduled_start is not None
                            else None
                        ),
                    }
                    for index, candidate in enumerate(
                        self.task.interview_candidates, start=1
                    )
                ],
                "interview_preparation_ready": (
                    self.task.active_interview_preparation_id is not None
                ),
                "action_candidates": [
                    {
                        "selection_index": index,
                        "action_type": candidate.action_type,
                        "source_type": candidate.source_type,
                        "title": candidate.title,
                        "status": candidate.status,
                        "due_at": (
                            candidate.due_at.isoformat()
                            if candidate.due_at is not None
                            else None
                        ),
                    }
                    for index, candidate in enumerate(
                        self.task.action_candidates, start=1
                    )
                ],
                "calendar_accounts": [
                    {
                        "selection_index": index,
                        "provider": candidate.provider,
                        "email_address": candidate.email_address,
                        "calendar_id": candidate.calendar_id,
                    }
                    for index, candidate in enumerate(
                        self.task.calendar_account_candidates, start=1
                    )
                ],
            },
            "recent_messages": tuple(message.model_dump(mode="json") for message in self.recent_messages),
            "tool_observations": tuple(observation.model_dump(mode="json") for observation in self.tool_observations),
            "conversation_summary": (
                self.conversation_summary.model_dump(mode="json")
                if self.conversation_summary
                else None
            ),
            "user_message": self.user_message,
        }


class JobDiscoveryToolArguments(ContractModel):
    target_role: str | None = Field(default=None, min_length=1)
    city: str | None = Field(default=None, min_length=1)
    salary: str | None = Field(default=None, min_length=1)
    experience: str | None = Field(default=None, min_length=1)
    education: str | None = Field(default=None, min_length=1)
    selection_indices: tuple[int, ...] = ()
    selection_index: int | None = Field(default=None, ge=1)
    jd_selection_index: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_selection_indices(self) -> "JobDiscoveryToolArguments":
        if len(self.selection_indices) > 3 or len(set(self.selection_indices)) != len(self.selection_indices):
            raise ValueError("select at most three unique candidate indexes")
        return self


class FindSavedJobsToolArguments(ContractModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=20)


class GetSavedJobToolArguments(ContractModel):
    job_posting_id: str = Field(min_length=1)


class ListTargetRolesToolArguments(ContractModel):
    pass


class ListResumesToolArguments(ContractModel):
    target_role_id: str | None = Field(default=None, min_length=1)


class GetResumeMetadataToolArguments(ContractModel):
    resume_id: str = Field(min_length=1)


class AnalyzeResumeToolArguments(ContractModel):
    resume_version_id: str = Field(min_length=1)


class MatchResumeToJobToolArguments(ContractModel):
    resume_version_id: str = Field(min_length=1)
    job_posting_id: str = Field(min_length=1)


class GetResumeJobMatchToolArguments(ContractModel):
    match_id: str | None = Field(default=None, min_length=1)


class DraftResumeTailoringToolArguments(ContractModel):
    match_id: str | None = Field(default=None, min_length=1)
    tailoring_goal: str | None = Field(default=None, min_length=1, max_length=2000)


class GetResumeTailoringDraftToolArguments(ContractModel):
    draft_id: str | None = Field(default=None, min_length=1)


class ReviewResumeTailoringToolArguments(ContractModel):
    draft_id: str | None = Field(default=None, min_length=1)
    accepted_change_indices: tuple[int, ...] = Field(default=(), max_length=30)
    rejected_change_indices: tuple[int, ...] = Field(default=(), max_length=30)
    feedback: str | None = Field(default=None, min_length=1, max_length=2000)

    @model_validator(mode="after")
    def validate_decisions(self) -> ReviewResumeTailoringToolArguments:
        accepted = self.accepted_change_indices
        rejected = self.rejected_change_indices
        if not accepted and not rejected:
            raise ValueError("at least one accepted or rejected change index is required")
        if any(index < 1 for index in (*accepted, *rejected)):
            raise ValueError("change indices must be positive")
        if len(set(accepted)) != len(accepted) or len(set(rejected)) != len(rejected):
            raise ValueError("change indices must be unique")
        if set(accepted).intersection(rejected):
            raise ValueError("a change cannot be both accepted and rejected")
        return self


class FinalizeResumeTailoringToolArguments(ContractModel):
    draft_id: str | None = Field(default=None, min_length=1)


class ExportResumeArtifactToolArguments(ContractModel):
    resume_version_id: str | None = Field(default=None, min_length=1)


class CreateApplicationToolArguments(ContractModel):
    job_posting_id: str | None = Field(default=None, min_length=1)
    resume_version_id: str | None = Field(default=None, min_length=1)
    submitted_at: datetime | None = None
    note: str | None = Field(default=None, min_length=1, max_length=2000)


class UpdateApplicationStatusToolArguments(ContractModel):
    application_id: str | None = Field(default=None, min_length=1)
    selection_index: int | None = Field(default=None, ge=1)
    status: ApplicationStatus
    note: str | None = Field(default=None, min_length=1, max_length=2000)

    @model_validator(mode="after")
    def validate_application_selector(self) -> "UpdateApplicationStatusToolArguments":
        if self.application_id is not None and self.selection_index is not None:
            raise ValueError("use either application_id or selection_index")
        return self


class ListApplicationsToolArguments(ContractModel):
    statuses: tuple[ApplicationStatus, ...] = Field(default=(), max_length=8)
    limit: int = Field(default=20, ge=1, le=50)

    @model_validator(mode="after")
    def validate_statuses(self) -> "ListApplicationsToolArguments":
        if len(set(self.statuses)) != len(self.statuses):
            raise ValueError("application statuses must be unique")
        return self


class GetApplicationToolArguments(ContractModel):
    application_id: str | None = Field(default=None, min_length=1)
    selection_index: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_application_selector(self) -> "GetApplicationToolArguments":
        if self.application_id is not None and self.selection_index is not None:
            raise ValueError("use either application_id or selection_index")
        return self


class SyncApplicationEmailsToolArguments(ContractModel):
    account_id: str | None = Field(default=None, min_length=1)


class ListEmailEventsToolArguments(ContractModel):
    status: EmailEventStatus | None = None
    limit: int = Field(default=20, ge=1, le=50)


class ResolveEmailEventToolArguments(ContractModel):
    event_id: str = Field(min_length=1)
    approve: bool
    application_id: str | None = Field(default=None, min_length=1)
    interview_round_id: str | None = Field(default=None, min_length=1)


class ListInterviewsToolArguments(ContractModel):
    application_id: str | None = Field(default=None, min_length=1)
    statuses: tuple[InterviewStatus, ...] = Field(default=(), max_length=4)
    limit: int = Field(default=20, ge=1, le=50)


class GetInterviewToolArguments(ContractModel):
    interview_round_id: str | None = Field(default=None, min_length=1)
    selection_index: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_selector(self) -> "GetInterviewToolArguments":
        if self.interview_round_id is not None and self.selection_index is not None:
            raise ValueError("use either interview_round_id or selection_index")
        return self


class CreateInterviewToolArguments(ContractModel):
    application_id: str | None = Field(default=None, min_length=1)
    details: InterviewDetails


class UpdateInterviewToolArguments(ContractModel):
    interview_round_id: str | None = Field(default=None, min_length=1)
    selection_index: int | None = Field(default=None, ge=1)
    details: InterviewDetails

    @model_validator(mode="after")
    def validate_selector(self) -> "UpdateInterviewToolArguments":
        if self.interview_round_id is not None and self.selection_index is not None:
            raise ValueError("use either interview_round_id or selection_index")
        return self


class CompleteInterviewToolArguments(ContractModel):
    interview_round_id: str | None = Field(default=None, min_length=1)
    selection_index: int | None = Field(default=None, ge=1)
    completed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_selector(self) -> "CompleteInterviewToolArguments":
        if self.interview_round_id is not None and self.selection_index is not None:
            raise ValueError("use either interview_round_id or selection_index")
        return self


class PrepareInterviewToolArguments(ContractModel):
    interview_round_id: str | None = Field(default=None, min_length=1)
    selection_index: int | None = Field(default=None, ge=1)
    action_selection_index: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_selector(self) -> "PrepareInterviewToolArguments":
        selectors = (
            self.interview_round_id,
            self.selection_index,
            self.action_selection_index,
        )
        if sum(value is not None for value in selectors) > 1:
            raise ValueError("use one interview selector")
        return self


class GetInterviewPreparationToolArguments(ContractModel):
    preparation_id: str | None = Field(default=None, min_length=1)


class GetDailyBriefToolArguments(ContractModel):
    timezone: str = Field(default="Asia/Shanghai", min_length=1, max_length=100)


class ListActionItemsToolArguments(ContractModel):
    statuses: tuple[ActionStatus, ...] = Field(
        default=("open", "snoozed"), min_length=1, max_length=4
    )
    limit: int = Field(default=50, ge=1, le=100)
    timezone: str = Field(default="Asia/Shanghai", min_length=1, max_length=100)


class ResolveActionItemToolArguments(ContractModel):
    action_item_id: str | None = Field(default=None, min_length=1)
    selection_index: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_selector(self) -> "ResolveActionItemToolArguments":
        if self.action_item_id is not None and self.selection_index is not None:
            raise ValueError("use either action_item_id or selection_index")
        return self


class SnoozeActionItemToolArguments(ResolveActionItemToolArguments):
    snoozed_until: datetime


class ListCalendarAccountsToolArguments(ContractModel):
    pass


class ListCalendarLinksToolArguments(ContractModel):
    pass


class PrepareInterviewCalendarSyncToolArguments(ContractModel):
    interview_round_id: str | None = Field(default=None, min_length=1)
    interview_selection_index: int | None = Field(default=None, ge=1)
    calendar_account_id: str | None = Field(default=None, min_length=1)
    calendar_account_selection_index: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_selectors(self) -> "PrepareInterviewCalendarSyncToolArguments":
        if self.interview_round_id and self.interview_selection_index:
            raise ValueError("use one interview selector")
        if self.calendar_account_id and self.calendar_account_selection_index:
            raise ValueError("use one calendar account selector")
        return self


class GetCalendarProposalToolArguments(ContractModel):
    proposal_id: str | None = Field(default=None, min_length=1)


class ExecuteCalendarProposalToolArguments(GetCalendarProposalToolArguments):
    pass


class GetResumeAnalysisToolArguments(ContractModel):
    analysis_id: str | None = Field(default=None, min_length=1)


class ConfirmResumeAnalysisToolArguments(ContractModel):
    analysis_id: str | None = Field(default=None, min_length=1)


class JobDiscoveryWorkflowInput(ContractModel):
    user_id: str
    conversation_id: str
    task: ConversationTaskState
    user_message: str = Field(min_length=1)
    selection_indices: tuple[int, ...] = ()
    selection_index: int | None = Field(default=None, ge=1)
    jd_selection_index: int | None = Field(default=None, ge=1)
    research_request: JobDiscoveryRequest | None = None


class ToolCall(ContractModel):
    name: str
    arguments: dict[str, Any] = {}


class AgentDecision(ContractModel):
    action: Literal["ask_user", "tool_call", "final"]
    message: str | None = None
    tool_call: ToolCall | None = None


class DecisionMaker(Protocol):
    def decide(self, context: MainAgentContext, tool_specs: tuple[dict[str, Any], ...]) -> AgentDecision: ...


def project_job_discovery_arguments(context: MainAgentContext, arguments: dict[str, Any]) -> dict[str, Any]:
    forbidden = {"user_id", "conversation_id", "run_id", "result_ref", "security_id", "job_id", "jd_text"}.intersection(arguments)
    if forbidden:
        raise ValueError(f"Job Discovery tool cannot accept internal arguments: {', '.join(sorted(forbidden))}")
    model_arguments = JobDiscoveryToolArguments.model_validate(arguments)
    target_role = model_arguments.target_role
    if target_role is None and len(context.profile.target_roles) == 1:
        target_role = context.profile.target_roles[0]
    request = None
    if target_role is not None:
        request = JobDiscoveryRequest(
            user_id=context.profile.user_id,
            conversation_id=context.conversation_id,
            target_role=target_role,
            city=model_arguments.city or context.profile.default_city,
            salary=model_arguments.salary or context.profile.salary_preference,
            experience=model_arguments.experience or context.profile.experience,
            education=model_arguments.education or context.profile.education,
        )
    return JobDiscoveryWorkflowInput(
        user_id=context.profile.user_id,
        conversation_id=context.conversation_id,
        task=context.task,
        user_message=context.user_message,
        selection_indices=model_arguments.selection_indices or ((model_arguments.selection_index,) if model_arguments.selection_index else ()),
        selection_index=model_arguments.selection_index,
        jd_selection_index=model_arguments.jd_selection_index,
        research_request=request,
    ).model_dump()


def project_saved_job_arguments(context: MainAgentContext, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if "user_id" in arguments:
        raise ValueError(f"{name} cannot accept internal argument: user_id")
    if name == "find_saved_jobs":
        model_arguments = FindSavedJobsToolArguments.model_validate(arguments)
    elif name == "get_saved_job":
        model_arguments = GetSavedJobToolArguments.model_validate(arguments)
    else:
        raise ValueError(f"Unknown saved-job tool: {name}")
    return {"user_id": context.profile.user_id, **model_arguments.model_dump()}


def project_resume_arguments(context: MainAgentContext, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if "user_id" in arguments:
        raise ValueError(f"{name} cannot accept internal argument: user_id")
    if name == "list_target_roles":
        model_arguments = ListTargetRolesToolArguments.model_validate(arguments)
    elif name == "list_resumes":
        model_arguments = ListResumesToolArguments.model_validate(arguments)
    elif name == "get_resume_metadata":
        model_arguments = GetResumeMetadataToolArguments.model_validate(arguments)
    elif name == "analyze_resume":
        model_arguments = AnalyzeResumeToolArguments.model_validate(arguments)
    elif name == "match_resume_to_job":
        model_arguments = MatchResumeToJobToolArguments.model_validate(arguments)
    elif name == "get_resume_job_match":
        model_arguments = GetResumeJobMatchToolArguments.model_validate(arguments)
    elif name == "draft_resume_tailoring":
        model_arguments = DraftResumeTailoringToolArguments.model_validate(arguments)
    elif name == "get_resume_tailoring_draft":
        model_arguments = GetResumeTailoringDraftToolArguments.model_validate(arguments)
    elif name == "review_resume_tailoring":
        model_arguments = ReviewResumeTailoringToolArguments.model_validate(arguments)
    elif name == "finalize_resume_tailoring":
        model_arguments = FinalizeResumeTailoringToolArguments.model_validate(arguments)
    elif name == "export_resume_artifact":
        model_arguments = ExportResumeArtifactToolArguments.model_validate(arguments)
    elif name == "create_application":
        model_arguments = CreateApplicationToolArguments.model_validate(arguments)
    elif name == "update_application_status":
        model_arguments = UpdateApplicationStatusToolArguments.model_validate(arguments)
    elif name == "list_applications":
        model_arguments = ListApplicationsToolArguments.model_validate(arguments)
    elif name == "get_application":
        model_arguments = GetApplicationToolArguments.model_validate(arguments)
    elif name == "get_resume_analysis":
        model_arguments = GetResumeAnalysisToolArguments.model_validate(arguments)
    elif name == "confirm_resume_analysis":
        model_arguments = ConfirmResumeAnalysisToolArguments.model_validate(arguments)
    else:
        raise ValueError(f"Unknown resume tool: {name}")
    payload = model_arguments.model_dump()
    if name in {"get_resume_analysis", "confirm_resume_analysis"}:
        analysis_id = payload.get("analysis_id") or context.task.active_resume_analysis_id
        if analysis_id is None:
            raise ValueError(f"{name} requires an active resume analysis")
        payload["analysis_id"] = analysis_id
    if name == "get_resume_job_match":
        match_id = payload.get("match_id") or context.task.active_resume_job_match_id
        if match_id is None:
            raise ValueError("get_resume_job_match requires an active resume-job match")
        payload["match_id"] = match_id
    if name == "draft_resume_tailoring":
        match_id = payload.get("match_id") or context.task.active_resume_job_match_id
        if match_id is None:
            raise ValueError("draft_resume_tailoring requires an active resume-job match")
        payload["match_id"] = match_id
    if name in {
        "get_resume_tailoring_draft",
        "review_resume_tailoring",
        "finalize_resume_tailoring",
    }:
        draft_id = payload.get("draft_id") or context.task.active_resume_tailoring_draft_id
        if draft_id is None:
            raise ValueError(f"{name} requires an active tailoring draft")
        payload["draft_id"] = draft_id
    if name == "export_resume_artifact":
        resume_version_id = (
            payload.get("resume_version_id") or context.task.active_resume_version_id
        )
        if resume_version_id is None:
            raise ValueError("export_resume_artifact requires an active resume version")
        payload["resume_version_id"] = resume_version_id
    if name == "create_application":
        job_posting_id = payload.get("job_posting_id") or context.task.active_job_posting_id
        resume_version_id = (
            payload.get("resume_version_id") or context.task.active_resume_version_id
        )
        if job_posting_id is None or resume_version_id is None:
            raise ValueError(
                "create_application requires an active job and resume version"
            )
        payload["job_posting_id"] = job_posting_id
        payload["resume_version_id"] = resume_version_id
    if name in {"update_application_status", "get_application"}:
        application_id = payload.get("application_id")
        selection_index = payload.pop("selection_index", None)
        if application_id is None and selection_index is not None:
            if selection_index > len(context.task.application_candidates):
                raise ValueError("application selection index is out of range")
            application_id = context.task.application_candidates[
                selection_index - 1
            ].application_id
        application_id = application_id or context.task.active_application_id
        if application_id is None:
            raise ValueError(f"{name} requires an active application")
        payload["application_id"] = application_id
    return {"user_id": context.profile.user_id, **payload}


def project_email_arguments(
    context: MainAgentContext, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    if "user_id" in arguments:
        raise ValueError(f"{name} cannot accept internal argument: user_id")
    if name == "sync_application_emails":
        model_arguments = SyncApplicationEmailsToolArguments.model_validate(arguments)
    elif name == "list_email_events":
        model_arguments = ListEmailEventsToolArguments.model_validate(arguments)
    elif name == "resolve_email_event":
        model_arguments = ResolveEmailEventToolArguments.model_validate(arguments)
    else:
        raise ValueError(f"Unknown email tool: {name}")
    payload = model_arguments.model_dump()
    if name == "resolve_email_event" and payload.get("application_id") is None:
        payload["application_id"] = context.task.active_application_id
    if name == "resolve_email_event" and payload.get("interview_round_id") is None:
        payload["interview_round_id"] = context.task.active_interview_round_id
    return {"user_id": context.profile.user_id, **payload}


def project_interview_arguments(
    context: MainAgentContext, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    if "user_id" in arguments:
        raise ValueError(f"{name} cannot accept internal argument: user_id")
    if name == "list_interviews":
        model_arguments = ListInterviewsToolArguments.model_validate(arguments)
    elif name == "get_interview":
        model_arguments = GetInterviewToolArguments.model_validate(arguments)
    elif name == "create_interview":
        model_arguments = CreateInterviewToolArguments.model_validate(arguments)
    elif name == "update_interview":
        model_arguments = UpdateInterviewToolArguments.model_validate(arguments)
    elif name == "complete_interview":
        model_arguments = CompleteInterviewToolArguments.model_validate(arguments)
    else:
        raise ValueError(f"Unknown interview tool: {name}")
    payload = model_arguments.model_dump()
    if name == "create_interview":
        application_id = payload.get("application_id") or context.task.active_application_id
        if application_id is None:
            raise ValueError("create_interview requires an active application")
        payload["application_id"] = application_id
    if name in {"get_interview", "update_interview", "complete_interview"}:
        interview_round_id = payload.get("interview_round_id")
        selection_index = payload.pop("selection_index", None)
        if interview_round_id is None and selection_index is not None:
            if selection_index > len(context.task.interview_candidates):
                raise ValueError("interview selection index is out of range")
            interview_round_id = context.task.interview_candidates[
                selection_index - 1
            ].interview_round_id
        interview_round_id = interview_round_id or context.task.active_interview_round_id
        if interview_round_id is None:
            raise ValueError(f"{name} requires an active interview")
        payload["interview_round_id"] = interview_round_id
    return {"user_id": context.profile.user_id, **payload}


def project_interview_preparation_arguments(
    context: MainAgentContext, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    if "user_id" in arguments:
        raise ValueError(f"{name} cannot accept internal argument: user_id")
    if name == "prepare_interview":
        model_arguments = PrepareInterviewToolArguments.model_validate(arguments)
        payload = model_arguments.model_dump()
        interview_round_id = payload.get("interview_round_id")
        selection_index = payload.pop("selection_index", None)
        action_selection_index = payload.pop("action_selection_index", None)
        if interview_round_id is None and selection_index is not None:
            if selection_index > len(context.task.interview_candidates):
                raise ValueError("interview selection index is out of range")
            interview_round_id = context.task.interview_candidates[
                selection_index - 1
            ].interview_round_id
        if interview_round_id is None and action_selection_index is not None:
            if action_selection_index > len(context.task.action_candidates):
                raise ValueError("action selection index is out of range")
            action = context.task.action_candidates[action_selection_index - 1]
            if (
                action.action_type != "interview_preparation"
                or action.source_type != "interview_round"
            ):
                raise ValueError("selected action is not interview preparation")
            interview_round_id = action.source_id
        interview_round_id = interview_round_id or context.task.active_interview_round_id
        if interview_round_id is None:
            raise ValueError("prepare_interview requires an active interview")
        payload["interview_round_id"] = interview_round_id
    elif name == "get_interview_preparation":
        model_arguments = GetInterviewPreparationToolArguments.model_validate(arguments)
        payload = model_arguments.model_dump()
        preparation_id = (
            payload.get("preparation_id")
            or context.task.active_interview_preparation_id
        )
        if preparation_id is None:
            raise ValueError("get_interview_preparation requires an active preparation")
        payload["preparation_id"] = preparation_id
    else:
        raise ValueError(f"Unknown interview preparation tool: {name}")
    return {"user_id": context.profile.user_id, **payload}


def project_action_center_arguments(
    context: MainAgentContext, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    if "user_id" in arguments:
        raise ValueError(f"{name} cannot accept internal argument: user_id")
    if name == "get_daily_brief":
        model_arguments = GetDailyBriefToolArguments.model_validate(arguments)
    elif name == "list_action_items":
        model_arguments = ListActionItemsToolArguments.model_validate(arguments)
    elif name in {"complete_action_item", "dismiss_action_item"}:
        model_arguments = ResolveActionItemToolArguments.model_validate(arguments)
    elif name == "snooze_action_item":
        model_arguments = SnoozeActionItemToolArguments.model_validate(arguments)
    else:
        raise ValueError(f"Unknown action-center tool: {name}")
    payload = model_arguments.model_dump()
    if name in {"complete_action_item", "dismiss_action_item", "snooze_action_item"}:
        action_item_id = payload.get("action_item_id")
        selection_index = payload.pop("selection_index", None)
        if action_item_id is None and selection_index is not None:
            if selection_index > len(context.task.action_candidates):
                raise ValueError("action selection index is out of range")
            action_item_id = context.task.action_candidates[
                selection_index - 1
            ].action_item_id
        action_item_id = action_item_id or context.task.active_action_item_id
        if action_item_id is None:
            raise ValueError(f"{name} requires an active action item")
        payload["action_item_id"] = action_item_id
    return {"user_id": context.profile.user_id, **payload}


def project_calendar_arguments(
    context: MainAgentContext, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    if "user_id" in arguments:
        raise ValueError(f"{name} cannot accept internal argument: user_id")
    if name == "list_calendar_accounts":
        model_arguments = ListCalendarAccountsToolArguments.model_validate(arguments)
    elif name == "list_calendar_links":
        model_arguments = ListCalendarLinksToolArguments.model_validate(arguments)
    elif name == "prepare_interview_calendar_sync":
        model_arguments = PrepareInterviewCalendarSyncToolArguments.model_validate(arguments)
    elif name == "get_calendar_proposal":
        model_arguments = GetCalendarProposalToolArguments.model_validate(arguments)
    elif name == "execute_calendar_proposal":
        model_arguments = ExecuteCalendarProposalToolArguments.model_validate(arguments)
    else:
        raise ValueError(f"Unknown calendar tool: {name}")
    payload = model_arguments.model_dump()
    if name == "prepare_interview_calendar_sync":
        interview_round_id = payload.get("interview_round_id")
        interview_index = payload.pop("interview_selection_index", None)
        if interview_round_id is None and interview_index is not None:
            if interview_index > len(context.task.interview_candidates):
                raise ValueError("interview selection index is out of range")
            interview_round_id = context.task.interview_candidates[
                interview_index - 1
            ].interview_round_id
        interview_round_id = interview_round_id or context.task.active_interview_round_id
        if interview_round_id is None:
            raise ValueError("prepare_interview_calendar_sync requires an active interview")
        payload["interview_round_id"] = interview_round_id
        account_id = payload.get("calendar_account_id")
        account_index = payload.pop("calendar_account_selection_index", None)
        if account_id is None and account_index is not None:
            if account_index > len(context.task.calendar_account_candidates):
                raise ValueError("calendar account selection index is out of range")
            account_id = context.task.calendar_account_candidates[
                account_index - 1
            ].calendar_account_id
        payload["calendar_account_id"] = account_id
    if name in {"get_calendar_proposal", "execute_calendar_proposal"}:
        proposal_id = payload.get("proposal_id") or context.task.active_calendar_proposal_id
        if proposal_id is None:
            raise ValueError(f"{name} requires an active calendar proposal")
        payload["proposal_id"] = proposal_id
    return {"user_id": context.profile.user_id, **payload}
