from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Protocol

from pydantic import Field, model_validator

from career_agent.agent.job_discovery_contracts import JobDiscoveryRequest
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


class ConversationTaskState(ContractModel):
    active_workflow: Literal["job_discovery", "none"] = "none"
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
    resume_tailoring_status: Literal["pending"] | None = None


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
            },
            "recent_messages": tuple(message.model_dump(mode="json") for message in self.recent_messages),
            "tool_observations": tuple(observation.model_dump(mode="json") for observation in self.tool_observations),
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
    if name == "get_resume_tailoring_draft":
        draft_id = payload.get("draft_id") or context.task.active_resume_tailoring_draft_id
        if draft_id is None:
            raise ValueError("get_resume_tailoring_draft requires an active tailoring draft")
        payload["draft_id"] = draft_id
    return {"user_id": context.profile.user_id, **payload}
