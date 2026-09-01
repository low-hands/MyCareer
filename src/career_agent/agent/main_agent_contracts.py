from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, Protocol

from pydantic import Field, model_validator

from career_agent.agent.summary_text import condense
from career_agent.agent.conversation_memory_contracts import ConversationSummaryContent
from career_agent.domain.applications import ApplicationStatus
from career_agent.domain.action_center import ActionSourceType, ActionStatus, ActionType
from career_agent.domain.email_tracking import EmailEventStatus
from career_agent.domain.interviews import (
    InterviewDetails,
    InterviewRetroQuestion,
    InterviewSelfAssessment,
    InterviewStatus,
)
from career_agent.domain.job_discovery import ContractModel
from career_agent.domain.mock_interviews import MockInterviewType


class CareerProfileContext(ContractModel):
    """Person-level job intent only.

    Everything that varies between the roles a user is pursuing — salary band,
    the experience and education brackets they are searching within, and a city
    they would accept for one role but not another — lives on TargetRole, which
    already has identity, priority, and the resume families hanging off it.
    Keeping a second free-text role list here only guaranteed that one of the
    two would eventually be written to and the other read.
    """

    user_id: str
    default_city: str | None = None


class JobIntentUpdate(ContractModel):
    """A proposed change to stated job intent, not yet applied.

    ``target_role_id`` decides the scope. Unset, the update is about the person
    and may only carry ``city``. Set, it is about that one career track, where a
    city is an override of the person-level default.

    The scoping is a validator rather than a prompt instruction because a salary
    recorded against the person cannot be un-mixed later: the two roles it was
    meant to distinguish would already have collapsed into one number.

    Every field is optional because intent is stated a piece at a time, and an
    omitted field leaves the stored value alone rather than clearing it.
    """

    target_role_id: str | None = Field(default=None, min_length=1)
    city: str | None = Field(default=None, min_length=1, max_length=40)
    salary_expectation: str | None = Field(default=None, min_length=1, max_length=100)
    experience: str | None = Field(default=None, min_length=1, max_length=100)
    education: str | None = Field(default=None, min_length=1, max_length=100)

    @model_validator(mode="after")
    def scope_must_match_the_fields(self) -> "JobIntentUpdate":
        role_scoped = (self.salary_expectation, self.experience, self.education)
        if self.target_role_id is None and any(
            value is not None for value in role_scoped
        ):
            raise ValueError(
                "salary, experience, and education belong to a target role and "
                "need one to be selected"
            )
        if not any(
            value is not None for value in (self.city, *role_scoped)
        ):
            raise ValueError("a job intent update must change at least one field")
        return self

    @property
    def is_role_scoped(self) -> bool:
        return self.target_role_id is not None

    def apply_to_profile(self, profile: CareerProfileContext) -> CareerProfileContext:
        if self.is_role_scoped or self.city is None:
            return profile
        return profile.model_copy(update={"default_city": self.city})


class AgentPreferencesContext(ContractModel):
    boss_search: Literal["explicit_request_only", "allowed"] = "explicit_request_only"


SelectionIndex = Annotated[int, Field(ge=1)]
"""A 1-based pointer into a list the model was shown this turn.

The lower bound belongs to the type, not to each declaration. Written out
per-field it was correct twenty-nine times and missing once — on
``CompareSavedJobsToolArguments.job_selection_indices``, whose element type
carried no bound at all, so index 0 resolved through ``candidates[-1]`` to the
last saved job and a comparison ran against the wrong pair without erroring.
"""


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


class SavedJobCandidateContextItem(ContractModel):
    job_posting_id: str
    title: str
    company_name: str
    city: str | None = None
    salary: str | None = None


class TargetRoleCandidateContextItem(ContractModel):
    target_role_id: str
    title: str
    priority: int
    status: str
    # The intent recorded against this track. It rides along here rather than
    # being flattened into career_profile so the model sees which numbers belong
    # to which role instead of one blended set.
    city: str | None = None
    salary_expectation: str | None = None
    experience: str | None = None
    education: str | None = None


class ResumeCandidateContextItem(ContractModel):
    resume_id: str
    target_role_id: str
    name: str
    status: str
    latest_version_id: str | None = None


class ResumeVersionCandidateContextItem(ContractModel):
    resume_version_id: str
    version_number: int
    source_type: str
    document_format: str
    byte_size: int


class EmailEventCandidateContextItem(ContractModel):
    email_event_id: str
    event_type: str
    status: EmailEventStatus
    summary: str


class ConversationTaskState(ContractModel):
    """Durable per-conversation task state.

    ``active_workflow`` names the one multi-turn workflow that currently holds a
    suspended run, and ``run_id``/``phase``/``selected_result_ref``/
    ``manual_search_query`` are scoped to that workflow alone. Single-turn
    capabilities must not touch the slot: they finish inside one turn and have
    no run to resume, so claiming it would silently discard a workflow the user
    is still in the middle of. Use ``enter_workflow``/``leave_workflow`` rather
    than updating the fields piecemeal.
    """

    active_workflow: Literal["job_discovery", "mock_interview", "none"] = "none"
    run_id: str | None = None
    phase: str | None = None
    selected_result_ref: str | None = None
    manual_search_query: str | None = None
    candidates: tuple[CandidateContextItem, ...] = ()
    workflow_entry_message: str | None = None
    pending_job_intent_update: JobIntentUpdate | None = None
    active_resume_analysis_id: str | None = None
    resume_analysis_status: Literal["pending", "confirmed", "rejected"] | None = None
    active_resume_job_match_id: str | None = None
    resume_job_match_status: Literal["ready"] | None = None
    active_resume_tailoring_draft_id: str | None = None
    resume_tailoring_status: Literal[
        "pending", "in_review", "reviewed", "finalized", "superseded"
    ] | None = None
    active_resume_version_id: str | None = None
    active_resume_artifact_id: str | None = None
    active_job_posting_id: str | None = None
    active_job_research_run_id: str | None = None
    active_job_research_report_id: str | None = None
    job_research_status: Literal["current", "outdated", "failed"] | None = None
    active_application_id: str | None = None
    active_application_status: ApplicationStatus | None = None
    application_candidates: tuple[ApplicationCandidateContextItem, ...] = ()
    active_interview_round_id: str | None = None
    interview_candidates: tuple[InterviewCandidateContextItem, ...] = ()
    active_interview_preparation_id: str | None = None
    active_action_item_id: str | None = None
    action_candidates: tuple[ActionCandidateContextItem, ...] = ()
    active_calendar_proposal_id: str | None = None
    # Projected to the model, unlike the id beside it: whether a preview is
    # still live decides between executing it and preparing a new one, and
    # a timestamp names no object the model could act on.
    active_calendar_proposal_expires_at: datetime | None = None
    calendar_account_candidates: tuple[CalendarAccountCandidateContextItem, ...] = ()
    saved_job_candidates: tuple[SavedJobCandidateContextItem, ...] = ()
    target_role_candidates: tuple[TargetRoleCandidateContextItem, ...] = ()
    resume_candidates: tuple[ResumeCandidateContextItem, ...] = ()
    resume_version_candidates: tuple[ResumeVersionCandidateContextItem, ...] = ()
    email_event_candidates: tuple[EmailEventCandidateContextItem, ...] = ()
    email_sync_phase: str | None = None

    @model_validator(mode="after")
    def _validate_workflow_slot(self) -> "ConversationTaskState":
        if (self.active_workflow == "none") != (self.run_id is None):
            raise ValueError(
                "active_workflow and run_id must be set together: a named "
                "workflow needs a run to resume, and a run needs an owner."
            )
        return self

    def enter_workflow(
        self,
        workflow: Literal["job_discovery", "mock_interview"],
        *,
        run_id: str,
        phase: str | None = None,
        selected_result_ref: str | None = None,
        manual_search_query: str | None = None,
        candidates: tuple[CandidateContextItem, ...] | None = None,
    ) -> "ConversationTaskState":
        """Claim the workflow slot, replacing any previous occupant's scope.

        Every scoped field is written in one copy so a new workflow can never
        inherit a stale phase or selection from the one it displaced. The held
        request survives re-entry by the same workflow, which is how a run
        advances its phase, but never crosses to a different one.
        """
        return self.model_copy(
            update={
                "active_workflow": workflow,
                "run_id": run_id,
                "phase": phase,
                "selected_result_ref": selected_result_ref,
                "manual_search_query": manual_search_query,
                "candidates": self.candidates if candidates is None else candidates,
                "workflow_entry_message": (
                    self.workflow_entry_message
                    if self.active_workflow == workflow
                    else None
                ),
            }
        )

    def active_resource_flags(self) -> dict[str, bool]:
        """Whether each active object exists, without naming any of them.

        Derived from the ``active_*_id`` field names rather than listed by hand.
        Thirteen such fields existed and none reached the model: the projection
        withheld the ids, which is right, and withheld their existence with
        them, which is not. The system prompt repeatedly directs the model at
        "the active object", so a run could prepare a Calendar preview and then
        be unable to tell, on the next turn, that one was pending — the approval
        gate was unreachable rather than merely awkward.

        Deriving it means a new active object is covered the moment it is
        declared, and that a value can never leak: only ``is not None`` crosses.
        """
        return {
            f"has_{name[: -len('_id')]}": getattr(self, name) is not None
            for name in type(self).model_fields
            if name.startswith("active_") and name.endswith("_id")
        }

    def hold_entry_message(self, message: str) -> "ConversationTaskState":
        """Keep the request a multi-turn workflow has not answered yet.

        The workflow's own turns are not written to the conversation, so the
        reply to this request only exists once the run ends. Holding it here
        keeps the request and its reply in one write instead of leaving the
        conversation mid-exchange for as long as the run lasts.
        """
        return self.model_copy(update={"workflow_entry_message": message})

    def leave_workflow(self) -> "ConversationTaskState":
        """Release the slot and clear the scoped fields together.

        The held request is scoped to the occupant too. Whoever releases the
        slot has already paired it with a closing reply, so carrying it forward
        would let the next workflow answer this one's opening line.
        """
        return self.model_copy(
            update={
                "active_workflow": "none",
                "run_id": None,
                "phase": None,
                "selected_result_ref": None,
                "manual_search_query": None,
                "workflow_entry_message": None,
            }
        )


class ConversationResourceReference(ContractModel):
    """Immutable link from one historical message to its durable resource.

    A statement of what one past turn produced, so it never changes once
    written. This is the opposite of the ``active_*_id`` fields on the task,
    which track what the user is discussing now and are overwritten every time
    the focus moves. Reading back the report a turn produced needs the former;
    resolving "this report" with no antecedent needs the latter.
    """

    kind: Literal[
        "job_research_report",
        "mock_interview_report",
        "interview_preparation",
        "interview_retro_report",
        "resume_job_match",
        "resume_tailoring_draft",
    ]
    resource_id: str = Field(min_length=1)
    # Job research alone needs delivery-time render metadata because
    # ``anchored_by_other_job`` is relative to the request that produced this
    # turn and cannot be reconstructed from the report row. Its status is
    # snapshotted with that render bundle. Other kinds intentionally derive
    # current lifecycle state at read time — for example a tailoring card
    # should say that its draft has since been superseded.
    status_at_delivery: Literal["current", "outdated", "superseded"] | None = None
    anchored_by_other_job: bool | None = None

    @model_validator(mode="after")
    def scope_job_research_render_context(self) -> "ConversationResourceReference":
        values = (self.status_at_delivery, self.anchored_by_other_job)
        if self.kind == "job_research_report":
            if any(value is None for value in values):
                raise ValueError(
                    "job research references require delivery-time render context"
                )
        elif any(value is not None for value in values):
            raise ValueError(
                "delivery-time render metadata is scoped to job research; "
                "other resource kinds derive current state when read"
            )
        return self


class ConversationMessageContext(ContractModel):
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime
    resource_ref: ConversationResourceReference | None = None


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


class ToolResult(ContractModel):
    """Complete internal result used by reducers and the delivery layer.

    This object is deliberately excluded from ``MainAgentContext`` so adding a
    field to a tool handler can never silently expand the decision prompt.
    """

    tool_name: str
    state: str
    # Summary-row/message-body states use this as their sole durable receipt,
    # so an empty string would make a completed turn disappear after refresh.
    message: str = Field(min_length=1)
    next_action: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    resource_ref: ConversationResourceReference | None = None


# Compatibility name for capability handlers. New orchestration code should
# call this a result, not an observation: observations are model-facing.
ToolObservation = ToolResult


DecisionFactKey = Annotated[
    str,
    Field(pattern=r"^[a-z][a-z0-9_]{0,39}$"),
]
DecisionFactValue = (
    Annotated[bool, Field(strict=True)]
    | Annotated[int, Field(strict=True, ge=0, le=1_000_000)]
    | Annotated[str, Field(strict=True, min_length=1, max_length=80)]
)
_DECISION_FACT_KEYS_BY_STATE = {
    "daily_brief_ready": frozenset({"overdue", "due_today", "waiting"}),
    "resume_analysis_ready": frozenset(
        {"record_count", "clarification_count", "has_warnings"}
    ),
    "job_research_ready": frozenset({"cached", "finding_count", "status"}),
}


class DecisionObservation(ContractModel):
    """Closed, bounded observation visible to the Main Agent decision model."""

    tool_name: str = Field(pattern=r"^[a-z0-9_]+$", max_length=80)
    state: str = Field(pattern=r"^[a-z0-9_]+$", max_length=80)
    message: str = Field(min_length=1, max_length=600)
    facts: dict[DecisionFactKey, DecisionFactValue] = Field(
        default_factory=dict,
        max_length=8,
    )
    next_action: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9_]+$",
        max_length=80,
    )

    @model_validator(mode="after")
    def facts_cannot_name_internal_identifiers(self) -> "DecisionObservation":
        if any(key == "id" or key.endswith("_id") for key in self.facts):
            raise ValueError("decision facts cannot contain internal identifiers")
        if self.facts:
            expected = _DECISION_FACT_KEYS_BY_STATE.get(self.state)
            if expected is None or frozenset(self.facts) != expected:
                raise ValueError("decision facts must match the declared state schema")
            if self.state == "daily_brief_ready" and not all(
                type(self.facts[key]) is int
                for key in ("overdue", "due_today", "waiting")
            ):
                raise ValueError("daily brief decision facts must be integer counts")
            if self.state == "resume_analysis_ready" and not (
                type(self.facts["record_count"]) is int
                and type(self.facts["clarification_count"]) is int
                and type(self.facts["has_warnings"]) is bool
            ):
                raise ValueError("resume analysis decision facts have invalid types")
            if self.state == "job_research_ready" and not (
                type(self.facts["cached"]) is bool
                and type(self.facts["finding_count"]) is int
                and self.facts["status"] in {"current", "outdated", "superseded"}
            ):
                raise ValueError("job research decision facts have invalid values")
        return self


class MainAgentContext(ContractModel):
    conversation_id: str
    profile: CareerProfileContext
    preferences: AgentPreferencesContext = AgentPreferencesContext()
    task: ConversationTaskState = ConversationTaskState()
    career_memory: CareerMemoryContext = CareerMemoryContext()
    recent_messages: tuple[ConversationMessageContext, ...] = ()
    archived_resources: tuple[ConversationMessageContext, ...] = Field(
        default=(), max_length=12
    )
    """Reports delivered before the recent window, as a reachable catalogue.

    Without these a report becomes unreachable to the agent the moment its
    turn is summarised away: ``resource_ref`` lives only on the original
    message, and the summary carries none. The UI kept working — it reads
    the whole transcript — so the failure was asymmetric and silent, with
    the user looking at a card the model could no longer open.

    Only the reference and a bounded label cross. The report itself stays
    in its entity, which is the whole point of storing a pointer.
    """

    tool_observations: tuple[DecisionObservation, ...] = Field(default=(), max_length=3)
    conversation_summary: ConversationSummaryContent | None = None
    user_message: str = Field(min_length=1)

    def referenced_resources(self) -> tuple[ConversationResourceReference, ...]:
        """Every resource the model can name this turn, in projection order.

        Archived catalogue first, then the recent window, so an older report
        keeps a lower index than a newer one and the ordering reads the way the
        conversation happened.

        One place produces the numbering that ``model_context`` shows and that
        argument projection resolves. Numbering them twice would let the model
        pick index 2 and be handed index 3's report the moment the two loops
        drifted, which is silent and unrecoverable rather than an error.
        """
        return tuple(
            message.resource_ref
            for message in (*self.archived_resources, *self.recent_messages)
            if message.resource_ref is not None
        )

    def resolve_reference_index(
        self, *, reference_index: int, kind: str
    ) -> str:
        """Turn a turn-local index back into the internal resource id.

        The kind is checked rather than trusted: the model picks an index out of
        a mixed list, so an off-by-one would otherwise pass a preparation id to
        a report lookup and read as "not found" instead of as a bad selector.
        """
        references = self.referenced_resources()
        if not 1 <= reference_index <= len(references):
            raise ValueError("resource reference index is out of range")
        reference = references[reference_index - 1]
        if reference.kind != kind:
            raise ValueError(
                f"resource reference {reference_index} is a {reference.kind}, "
                f"not a {kind}"
            )
        return reference.resource_id

    def model_context(self) -> dict[str, Any]:
        model_messages = []
        # Numbered by walking the window in the same order and skipping the same
        # messages as ``referenced_resources``, so the index the model reads here
        # is the one it can pass back.
        archived = [
            {
                "kind": message.resource_ref.kind,
                "reference_index": index,
                "delivered_at": message.created_at.isoformat(),
                "summary": condense(message.content),
            }
            for index, message in enumerate(self.archived_resources, start=1)
            if message.resource_ref is not None
        ]
        # Continues the catalogue's numbering rather than restarting, because
        # ``referenced_resources`` concatenates the two in this same order.
        reference_index = len(archived)
        for message in self.recent_messages:
            projected = {
                "role": message.role,
                "content": message.content,
                "created_at": message.created_at.isoformat(),
            }
            if message.resource_ref is not None:
                reference_index += 1
                projected["resource"] = {
                    "kind": message.resource_ref.kind,
                    "reference_index": reference_index,
                }
            model_messages.append(projected)
        return {
            "career_profile": {
                "default_city": self.profile.default_city,
                # Role-scoped intent reaches the model through
                # target_role_candidates, which carry it per track. Flattening
                # it here would hand back the single blended profile this split
                # exists to prevent.
                "records": [
                    record.model_dump(mode="json")
                    for record in self.career_memory.records
                ],
            },
            "preferences": self.preferences.model_dump(mode="json"),
            "task": {
                **self.task.active_resource_flags(),
                "active_calendar_proposal_expires_at": (
                    self.task.active_calendar_proposal_expires_at.isoformat()
                    if self.task.active_calendar_proposal_expires_at is not None
                    else None
                ),
                "active_workflow": self.task.active_workflow,
                "phase": self.task.phase,
                "email_sync_phase": self.task.email_sync_phase,
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
                "job_research_status": self.task.job_research_status,
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
                "saved_jobs": [
                    {
                        "selection_index": index,
                        "title": candidate.title,
                        "company_name": candidate.company_name,
                        "city": candidate.city,
                        "salary": candidate.salary,
                    }
                    for index, candidate in enumerate(
                        self.task.saved_job_candidates, start=1
                    )
                ],
                "target_roles": [
                    {
                        "selection_index": index,
                        "title": candidate.title,
                        "priority": candidate.priority,
                        "status": candidate.status,
                    }
                    for index, candidate in enumerate(
                        self.task.target_role_candidates, start=1
                    )
                ],
                "resumes": [
                    {
                        "selection_index": index,
                        "name": candidate.name,
                        "status": candidate.status,
                    }
                    for index, candidate in enumerate(
                        self.task.resume_candidates, start=1
                    )
                ],
                "resume_versions": [
                    {
                        "selection_index": index,
                        "version_number": candidate.version_number,
                        "source_type": candidate.source_type,
                        "document_format": candidate.document_format,
                        "byte_size": candidate.byte_size,
                    }
                    for index, candidate in enumerate(
                        self.task.resume_version_candidates, start=1
                    )
                ],
                "email_events": [
                    {
                        "selection_index": index,
                        "event_type": candidate.event_type,
                        "status": candidate.status,
                        "summary": candidate.summary,
                    }
                    for index, candidate in enumerate(
                        self.task.email_event_candidates, start=1
                    )
                ],
            },
            # Internal resource IDs stay in durable messages for the UI and
            # projection layer. The decision model receives only turn-local
            # indexes, matching every other selectable object contract.
            "archived_reports": tuple(archived),
            "recent_messages": tuple(model_messages),
            "tool_observations": tuple(observation.model_dump(mode="json") for observation in self.tool_observations),
            "conversation_summary": (
                self.conversation_summary.model_dump(mode="json")
                if self.conversation_summary
                else None
            ),
            "user_message": self.user_message,
        }


class OpenJobSearchToolArguments(ContractModel):
    platform: Literal["boss"] = "boss"
    keyword: str = Field(min_length=1, max_length=100)
    city: str | None = Field(default=None, min_length=1, max_length=40)


class FindSavedJobsToolArguments(ContractModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=20)


class GetSavedJobToolArguments(ContractModel):
    job_posting_id: str | None = Field(default=None, min_length=1)
    selection_index: SelectionIndex | None = None


class ResearchJobToolArguments(ContractModel):
    job_posting_id: str | None = Field(default=None, min_length=1)
    selection_index: SelectionIndex | None = None
    focus: str | None = Field(default=None, min_length=1, max_length=1000)
    user_provided_context: str | None = Field(
        default=None,
        min_length=1,
        max_length=4000,
        description=(
            "Relevant business or product clues the user explicitly chose to use, "
            "for example something they heard in an interview. This is unverified "
            "user-reported context, not a public fact."
        ),
    )
    max_sources: int = Field(default=8, ge=2, le=15)


class RetryJobResearchToolArguments(ContractModel):
    run_id: str | None = Field(default=None, min_length=1)


class GetJobResearchToolArguments(ContractModel):
    """Selectors for reading back one job-research report.

    ``reference_index`` points at a report a past turn in the recent window
    produced, which is the only way to read back a report that is no longer the
    active one. The internal ids stay declared because handlers receive them
    after projection; the model-facing schema has them stripped.
    """

    report_id: str | None = Field(default=None, min_length=1)
    job_posting_id: str | None = Field(default=None, min_length=1)
    selection_index: SelectionIndex | None = None
    reference_index: SelectionIndex | None = None

    @model_validator(mode="after")
    def validate_selector(self) -> "GetJobResearchToolArguments":
        if self.reference_index is not None and self.selection_index is not None:
            raise ValueError("use either reference_index or selection_index")
        return self


class ListTargetRolesToolArguments(ContractModel):
    pass


class ListResumesToolArguments(ContractModel):
    target_role_id: str | None = Field(default=None, min_length=1)
    target_role_selection_index: SelectionIndex | None = None


class GetResumeMetadataToolArguments(ContractModel):
    resume_id: str | None = Field(default=None, min_length=1)
    selection_index: SelectionIndex | None = None


class AnalyzeResumeToolArguments(ContractModel):
    resume_version_id: str | None = Field(default=None, min_length=1)
    selection_index: SelectionIndex | None = None


class MatchResumeToJobToolArguments(ContractModel):
    resume_version_id: str | None = Field(default=None, min_length=1)
    job_posting_id: str | None = Field(default=None, min_length=1)
    resume_version_selection_index: SelectionIndex | None = None
    job_selection_index: SelectionIndex | None = None


class ProposeJobIntentToolArguments(ContractModel):
    target_role_selection_index: SelectionIndex | None = None
    city: str | None = Field(default=None, min_length=1, max_length=40)
    salary_expectation: str | None = Field(default=None, min_length=1, max_length=100)
    experience: str | None = Field(default=None, min_length=1, max_length=100)
    education: str | None = Field(default=None, min_length=1, max_length=100)


class ConfirmJobIntentToolArguments(ContractModel):
    pass


class CompareSavedJobsToolArguments(ContractModel):
    job_selection_indices: tuple[SelectionIndex, ...] = Field(
        min_length=2, max_length=10
    )


class GetResumeJobMatchToolArguments(ContractModel):
    match_id: str | None = Field(default=None, min_length=1)


class DraftResumeTailoringToolArguments(ContractModel):
    match_id: str | None = Field(default=None, min_length=1)
    tailoring_goal: str | None = Field(default=None, min_length=1, max_length=2000)


class GetResumeTailoringDraftToolArguments(ContractModel):
    draft_id: str | None = Field(default=None, min_length=1)


class ReviewResumeTailoringToolArguments(ContractModel):
    draft_id: str | None = Field(default=None, min_length=1)
    accepted_change_indices: tuple[SelectionIndex, ...] = Field(
        default=(), max_length=30
    )
    rejected_change_indices: tuple[SelectionIndex, ...] = Field(
        default=(), max_length=30
    )
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


class ReviseResumeTailoringToolArguments(ContractModel):
    draft_id: str | None = Field(default=None, min_length=1)
    feedback: str = Field(min_length=1, max_length=2000)


class FinalizeResumeTailoringToolArguments(ContractModel):
    draft_id: str | None = Field(default=None, min_length=1)


class ExportResumeArtifactToolArguments(ContractModel):
    resume_version_id: str | None = Field(default=None, min_length=1)


class CreateApplicationToolArguments(ContractModel):
    job_posting_id: str | None = Field(default=None, min_length=1)
    resume_version_id: str | None = Field(default=None, min_length=1)
    job_selection_index: SelectionIndex | None = None
    resume_version_selection_index: SelectionIndex | None = None
    submitted_at: datetime | None = None
    note: str | None = Field(default=None, min_length=1, max_length=2000)


class UpdateApplicationStatusToolArguments(ContractModel):
    application_id: str | None = Field(default=None, min_length=1)
    selection_index: SelectionIndex | None = None
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
    selection_index: SelectionIndex | None = None

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
    event_id: str | None = Field(default=None, min_length=1)
    selection_index: SelectionIndex | None = None
    approve: bool
    application_id: str | None = Field(default=None, min_length=1)
    interview_round_id: str | None = Field(default=None, min_length=1)


class ListInterviewsToolArguments(ContractModel):
    application_id: str | None = Field(default=None, min_length=1)
    statuses: tuple[InterviewStatus, ...] = Field(default=(), max_length=4)
    limit: int = Field(default=20, ge=1, le=50)


class GetInterviewToolArguments(ContractModel):
    interview_round_id: str | None = Field(default=None, min_length=1)
    selection_index: SelectionIndex | None = None

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
    selection_index: SelectionIndex | None = None
    details: InterviewDetails

    @model_validator(mode="after")
    def validate_selector(self) -> "UpdateInterviewToolArguments":
        if self.interview_round_id is not None and self.selection_index is not None:
            raise ValueError("use either interview_round_id or selection_index")
        return self


class CompleteInterviewToolArguments(ContractModel):
    interview_round_id: str | None = Field(default=None, min_length=1)
    selection_index: SelectionIndex | None = None
    completed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_selector(self) -> "CompleteInterviewToolArguments":
        if self.interview_round_id is not None and self.selection_index is not None:
            raise ValueError("use either interview_round_id or selection_index")
        return self


class RecordInterviewRetroToolArguments(ContractModel):
    interview_round_id: str | None = Field(default=None, min_length=1)
    selection_index: SelectionIndex | None = None
    source_notes: str = Field(min_length=1, max_length=20_000)
    summary: str = Field(min_length=1, max_length=5000)
    questions: tuple[InterviewRetroQuestion, ...] = Field(default=(), max_length=30)
    strengths: tuple[str, ...] = Field(default=(), max_length=20)
    difficulties: tuple[str, ...] = Field(default=(), max_length=20)
    interviewer_signals: tuple[str, ...] = Field(default=(), max_length=20)
    next_focus: tuple[str, ...] = Field(default=(), max_length=20)
    action_items: tuple[str, ...] = Field(default=(), max_length=20)
    limitations: tuple[str, ...] = Field(default=(), max_length=20)
    self_assessment: InterviewSelfAssessment = "uncertain"

    @model_validator(mode="after")
    def validate_selector(self) -> "RecordInterviewRetroToolArguments":
        if self.interview_round_id is not None and self.selection_index is not None:
            raise ValueError("use either interview_round_id or selection_index")
        return self


class PrepareInterviewToolArguments(ContractModel):
    interview_round_id: str | None = Field(default=None, min_length=1)
    selection_index: SelectionIndex | None = None
    action_selection_index: SelectionIndex | None = None

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
    """Selectors for reading back one interview preparation.

    Defaults to the active preparation. ``reference_index`` reaches one an
    earlier turn in the window produced, which the active pointer no longer
    names once the focus has moved on.
    """

    preparation_id: str | None = Field(default=None, min_length=1)
    reference_index: SelectionIndex | None = None
    interview_selection_index: SelectionIndex | None = None
    """The interview whose preparation to read, from ``list_interviews``.

    The entity-keyed route the other two readbacks already had: job research
    takes a saved job, a mock interview result takes an application. Without one
    here, a preparation fell out of reach for good once it stopped being active
    and its message left the recent window — while the report itself was still
    on disk and still visible in the transcript.
    """


class StartMockInterviewToolArguments(ContractModel):
    application_selection_index: SelectionIndex | None = None
    interview_selection_index: SelectionIndex | None = None
    interview_type: MockInterviewType = "mixed"
    max_primary_questions: int = Field(default=6, ge=1, le=20)
    max_follow_ups_per_question: int = Field(default=2, ge=0, le=5)

    @model_validator(mode="after")
    def validate_selector(self) -> "StartMockInterviewToolArguments":
        if (
            self.application_selection_index is not None
            and self.interview_selection_index is not None
        ):
            raise ValueError("use either an application or interview selector")
        return self


class RestartMockInterviewToolArguments(ContractModel):
    """No arguments: the replacement copies the stuck run's own settings.

    Letting the model restate the application or the question caps would let a
    restart quietly practise against a different resume than the run it
    replaces.
    """


class GetMockInterviewResultToolArguments(ContractModel):
    """Selectors for reading back one finished mock interview.

    Defaults to the latest finished run for the active application, which is
    what "how did I do" means in practice. ``reference_index`` instead names the
    exact run an earlier turn reported, which matters here more than elsewhere:
    one application can be practised against repeatedly, so "the latest" and
    "the one you just told me about" stop agreeing after a second run. A
    question number narrows the read to one exchange, because returning every
    answer in full would crowd out the rest of the conversation.
    """

    application_selection_index: SelectionIndex | None = None
    reference_index: SelectionIndex | None = None
    question_number: int | None = Field(default=None, ge=1, le=20)

    @model_validator(mode="after")
    def validate_selector(self) -> "GetMockInterviewResultToolArguments":
        if (
            self.reference_index is not None
            and self.application_selection_index is not None
        ):
            raise ValueError(
                "use either reference_index or application_selection_index"
            )
        return self


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
    selection_index: SelectionIndex | None = None

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
    interview_selection_index: SelectionIndex | None = None
    calendar_account_id: str | None = Field(default=None, min_length=1)
    calendar_account_selection_index: SelectionIndex | None = None

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


class StartMockInterviewWorkflowInput(ContractModel):
    user_id: str = Field(min_length=1)
    application_id: str = Field(min_length=1)
    interview_round_id: str | None = Field(default=None, min_length=1)
    interview_type: MockInterviewType
    max_primary_questions: int = Field(ge=1, le=20)
    max_follow_ups_per_question: int = Field(ge=0, le=5)


class ToolCall(ContractModel):
    name: str
    arguments: dict[str, Any] = {}


class AgentDecision(ContractModel):
    action: Literal["ask_user", "tool_call", "final"]
    message: str | None = None
    tool_call: ToolCall | None = None


class DecisionMaker(Protocol):
    def decide(self, context: MainAgentContext, tool_specs: tuple[dict[str, Any], ...]) -> AgentDecision: ...


def _reject_internal_identifiers(name: str, arguments: dict[str, Any]) -> None:
    forbidden = sorted(key for key in arguments if key == "user_id" or key.endswith("_id"))
    if forbidden:
        raise ValueError(
            f"{name} cannot accept internal identifiers: {', '.join(forbidden)}"
        )


def project_saved_job_arguments(context: MainAgentContext, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    _reject_internal_identifiers(name, arguments)
    if name == "find_saved_jobs":
        model_arguments = FindSavedJobsToolArguments.model_validate(arguments)
    elif name == "get_saved_job":
        model_arguments = GetSavedJobToolArguments.model_validate(arguments)
    elif name == "compare_saved_jobs":
        model_arguments = CompareSavedJobsToolArguments.model_validate(arguments)
    else:
        raise ValueError(f"Unknown saved-job tool: {name}")
    payload = model_arguments.model_dump()
    if name == "compare_saved_jobs":
        indices = payload.pop("job_selection_indices", ())
        candidates = context.task.saved_job_candidates
        # Both bounds here as well as in the schema. These guards checked only
        # the upper one, so the lower rested entirely on ``ge=1`` per field —
        # correct twenty-nine times and absent on this collection's element
        # type, where index 0 read ``candidates[-1]`` and compared a job
        # against itself without erroring.
        job_posting_ids = []
        for index in indices:
            if not 1 <= index <= len(candidates):
                raise ValueError("saved-job selection index is out of range")
            job_posting_ids.append(candidates[index - 1].job_posting_id)
        payload["job_posting_ids"] = tuple(job_posting_ids)
        payload["preferred_city"] = context.profile.default_city
    if name == "get_saved_job":
        selection_index = payload.pop("selection_index", None)
        job_posting_id = context.task.active_job_posting_id
        if selection_index is not None:
            if not 1 <= selection_index <= len(context.task.saved_job_candidates):
                raise ValueError("saved-job selection index is out of range")
            job_posting_id = context.task.saved_job_candidates[
                selection_index - 1
            ].job_posting_id
        if job_posting_id is None:
            raise ValueError("get_saved_job requires a selected or active saved job")
        payload["job_posting_id"] = job_posting_id
    return {"user_id": context.profile.user_id, **payload}


def project_job_intent_arguments(
    context: MainAgentContext,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    _reject_internal_identifiers(name, arguments)
    if name == "propose_job_intent":
        model_arguments = ProposeJobIntentToolArguments.model_validate(arguments)
        payload = model_arguments.model_dump(exclude_none=True)
        selection_index = payload.pop("target_role_selection_index", None)
        if selection_index is not None:
            candidates = context.task.target_role_candidates
            if not 1 <= selection_index <= len(candidates):
                raise ValueError("target-role selection index is out of range")
            payload["target_role_id"] = candidates[selection_index - 1].target_role_id
        update = JobIntentUpdate.model_validate(payload)
        return {
            "user_id": context.profile.user_id,
            "update": update,
            "current": context.profile,
        }
    ConfirmJobIntentToolArguments.model_validate(arguments)
    pending = context.task.pending_job_intent_update
    if pending is None:
        # Confirmation has to point at something the user was actually shown.
        raise ValueError(
            "confirm_job_intent requires a proposed update the user has seen"
        )
    return {
        "user_id": context.profile.user_id,
        "update": pending,
        "current": context.profile,
    }


def project_open_job_search_arguments(
    context: MainAgentContext, arguments: dict[str, Any]
) -> dict[str, Any]:
    _reject_internal_identifiers("open_job_search", arguments)
    model_arguments = OpenJobSearchToolArguments.model_validate(arguments)
    # Falls back to the person-level default only. A target role's city override
    # is deliberately not consulted here: task state has no notion of which role
    # the conversation is working in, so the only available rule would be "any
    # role that happens to have a city", which would silently search the wrong
    # place. Resolving it properly needs an active target role first.
    return model_arguments.model_copy(
        update={"city": model_arguments.city or context.profile.default_city}
    ).model_dump()


def project_job_research_arguments(
    context: MainAgentContext,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    _reject_internal_identifiers(name, arguments)
    if name == "research_job":
        model_arguments = ResearchJobToolArguments.model_validate(arguments)
        payload = model_arguments.model_dump()
        selection_index = payload.pop("selection_index", None)
        job_posting_id = context.task.active_job_posting_id
        if selection_index is not None:
            if not 1 <= selection_index <= len(context.task.saved_job_candidates):
                raise ValueError("saved-job selection index is out of range")
            job_posting_id = context.task.saved_job_candidates[
                selection_index - 1
            ].job_posting_id
        if job_posting_id is None:
            raise ValueError("research_job requires a selected or active saved job")
        payload["job_posting_id"] = job_posting_id
    elif name == "retry_job_research":
        RetryJobResearchToolArguments.model_validate(arguments)
        if context.task.active_job_research_run_id is None:
            raise ValueError("retry_job_research requires an active failed run")
        payload = {"run_id": context.task.active_job_research_run_id}
    elif name == "get_job_research":
        model_arguments = GetJobResearchToolArguments.model_validate(arguments)
        selection_index = model_arguments.selection_index
        if model_arguments.reference_index is not None:
            payload = {
                "report_id": context.resolve_reference_index(
                    reference_index=model_arguments.reference_index,
                    kind="job_research_report",
                )
            }
        elif selection_index is not None:
            if not 1 <= selection_index <= len(context.task.saved_job_candidates):
                raise ValueError("saved-job selection index is out of range")
            payload = {
                "job_posting_id": context.task.saved_job_candidates[
                    selection_index - 1
                ].job_posting_id
            }
        elif context.task.active_job_research_report_id is not None:
            payload = {"report_id": context.task.active_job_research_report_id}
        elif context.task.active_job_posting_id is not None:
            payload = {"job_posting_id": context.task.active_job_posting_id}
        else:
            raise ValueError("get_job_research requires an active research report or job")
    else:
        raise ValueError(f"Unknown job research tool: {name}")
    return {"user_id": context.profile.user_id, **payload}


def project_resume_arguments(context: MainAgentContext, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    _reject_internal_identifiers(name, arguments)
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
    elif name == "revise_resume_tailoring":
        model_arguments = ReviseResumeTailoringToolArguments.model_validate(arguments)
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
    else:
        raise ValueError(f"Unknown resume tool: {name}")
    payload = model_arguments.model_dump()
    if name == "list_resumes":
        selection_index = payload.pop("target_role_selection_index", None)
        if selection_index is not None:
            if not 1 <= selection_index <= len(context.task.target_role_candidates):
                raise ValueError("target-role selection index is out of range")
            payload["target_role_id"] = context.task.target_role_candidates[
                selection_index - 1
            ].target_role_id
    if name == "get_resume_metadata":
        selection_index = payload.pop("selection_index", None)
        if selection_index is not None:
            if not 1 <= selection_index <= len(context.task.resume_candidates):
                raise ValueError("resume selection index is out of range")
            payload["resume_id"] = context.task.resume_candidates[
                selection_index - 1
            ].resume_id
        if payload.get("resume_id") is None:
            raise ValueError("get_resume_metadata requires a selected resume")
    if name == "analyze_resume":
        selection_index = payload.pop("selection_index", None)
        resume_version_id = context.task.active_resume_version_id
        if selection_index is not None:
            if not 1 <= selection_index <= len(context.task.resume_version_candidates):
                raise ValueError("resume-version selection index is out of range")
            resume_version_id = context.task.resume_version_candidates[
                selection_index - 1
            ].resume_version_id
        if resume_version_id is None:
            raise ValueError("analyze_resume requires a selected or active resume version")
        payload["resume_version_id"] = resume_version_id
    if name == "match_resume_to_job":
        resume_selection_index = payload.pop(
            "resume_version_selection_index", None
        )
        job_selection_index = payload.pop("job_selection_index", None)
        resume_version_id = context.task.active_resume_version_id
        if resume_selection_index is not None:
            if not 1 <= resume_selection_index <= len(context.task.resume_version_candidates):
                raise ValueError("resume-version selection index is out of range")
            resume_version_id = context.task.resume_version_candidates[
                resume_selection_index - 1
            ].resume_version_id
        job_posting_id = context.task.active_job_posting_id
        if job_selection_index is not None:
            if not 1 <= job_selection_index <= len(context.task.saved_job_candidates):
                raise ValueError("saved-job selection index is out of range")
            job_posting_id = context.task.saved_job_candidates[
                job_selection_index - 1
            ].job_posting_id
        if resume_version_id is None or job_posting_id is None:
            raise ValueError(
                "match_resume_to_job requires selected or active resume and saved job"
            )
        payload["resume_version_id"] = resume_version_id
        payload["job_posting_id"] = job_posting_id
    if name == "get_resume_analysis":
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
        "revise_resume_tailoring",
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
        job_selection_index = payload.pop("job_selection_index", None)
        resume_selection_index = payload.pop(
            "resume_version_selection_index", None
        )
        job_posting_id = context.task.active_job_posting_id
        if job_selection_index is not None:
            if not 1 <= job_selection_index <= len(context.task.saved_job_candidates):
                raise ValueError("saved-job selection index is out of range")
            job_posting_id = context.task.saved_job_candidates[
                job_selection_index - 1
            ].job_posting_id
        resume_version_id = context.task.active_resume_version_id
        if resume_selection_index is not None:
            if not 1 <= resume_selection_index <= len(context.task.resume_version_candidates):
                raise ValueError("resume-version selection index is out of range")
            resume_version_id = context.task.resume_version_candidates[
                resume_selection_index - 1
            ].resume_version_id
        if job_posting_id is None or resume_version_id is None:
            raise ValueError(
                "create_application requires selected or active job and resume version"
            )
        payload["job_posting_id"] = job_posting_id
        payload["resume_version_id"] = resume_version_id
    if name in {"update_application_status", "get_application"}:
        application_id = payload.get("application_id")
        selection_index = payload.pop("selection_index", None)
        if application_id is None and selection_index is not None:
            if not 1 <= selection_index <= len(context.task.application_candidates):
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
    _reject_internal_identifiers(name, arguments)
    if name == "sync_application_emails":
        model_arguments = SyncApplicationEmailsToolArguments.model_validate(arguments)
    elif name == "list_email_events":
        model_arguments = ListEmailEventsToolArguments.model_validate(arguments)
    elif name == "resolve_email_event":
        model_arguments = ResolveEmailEventToolArguments.model_validate(arguments)
    else:
        raise ValueError(f"Unknown email tool: {name}")
    payload = model_arguments.model_dump()
    if name == "resolve_email_event":
        selection_index = payload.pop("selection_index", None)
        if selection_index is not None:
            if not 1 <= selection_index <= len(context.task.email_event_candidates):
                raise ValueError("email-event selection index is out of range")
            payload["event_id"] = context.task.email_event_candidates[
                selection_index - 1
            ].email_event_id
        if payload.get("event_id") is None:
            raise ValueError("resolve_email_event requires a selected email event")
    if name == "resolve_email_event" and payload.get("application_id") is None:
        payload["application_id"] = context.task.active_application_id
    if name == "resolve_email_event" and payload.get("interview_round_id") is None:
        payload["interview_round_id"] = context.task.active_interview_round_id
    return {"user_id": context.profile.user_id, **payload}


def project_interview_arguments(
    context: MainAgentContext, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    _reject_internal_identifiers(name, arguments)
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
    elif name == "record_interview_retro":
        model_arguments = RecordInterviewRetroToolArguments.model_validate(arguments)
    else:
        raise ValueError(f"Unknown interview tool: {name}")
    payload = model_arguments.model_dump()
    if name == "create_interview":
        application_id = payload.get("application_id") or context.task.active_application_id
        if application_id is None:
            raise ValueError("create_interview requires an active application")
        payload["application_id"] = application_id
    if name in {
        "get_interview",
        "update_interview",
        "complete_interview",
        "record_interview_retro",
    }:
        interview_round_id = payload.get("interview_round_id")
        selection_index = payload.pop("selection_index", None)
        if interview_round_id is None and selection_index is not None:
            if not 1 <= selection_index <= len(context.task.interview_candidates):
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
    _reject_internal_identifiers(name, arguments)
    if name == "prepare_interview":
        model_arguments = PrepareInterviewToolArguments.model_validate(arguments)
        payload = model_arguments.model_dump()
        interview_round_id = payload.get("interview_round_id")
        selection_index = payload.pop("selection_index", None)
        action_selection_index = payload.pop("action_selection_index", None)
        if interview_round_id is None and selection_index is not None:
            if not 1 <= selection_index <= len(context.task.interview_candidates):
                raise ValueError("interview selection index is out of range")
            interview_round_id = context.task.interview_candidates[
                selection_index - 1
            ].interview_round_id
        if interview_round_id is None and action_selection_index is not None:
            if not 1 <= action_selection_index <= len(context.task.action_candidates):
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
        reference_index = payload.pop("reference_index", None)
        interview_index = payload.pop("interview_selection_index", None)
        preparation_id = None
        if reference_index is not None:
            preparation_id = context.resolve_reference_index(
                reference_index=reference_index,
                kind="interview_preparation",
            )
        elif interview_index is not None:
            if not 1 <= interview_index <= len(context.task.interview_candidates):
                raise ValueError("interview selection index is out of range")
            # Resolved to the interview, not to a preparation: which preparation
            # belongs to it is the store's answer, and the newest is the one a
            # candidate means by "the prep for that interview".
            payload["interview_round_id"] = context.task.interview_candidates[
                interview_index - 1
            ].interview_round_id
        else:
            preparation_id = (
                payload.get("preparation_id")
                or context.task.active_interview_preparation_id
            )
        if preparation_id is None and "interview_round_id" not in payload:
            raise ValueError("get_interview_preparation requires an active preparation")
        payload["preparation_id"] = preparation_id
    else:
        raise ValueError(f"Unknown interview preparation tool: {name}")
    return {"user_id": context.profile.user_id, **payload}


def project_mock_interview_arguments(
    context: MainAgentContext, arguments: dict[str, Any]
) -> dict[str, Any]:
    _reject_internal_identifiers("start_mock_interview", arguments)
    model_arguments = StartMockInterviewToolArguments.model_validate(arguments)
    application_id = context.task.active_application_id
    interview_round_id: str | None = None

    if model_arguments.application_selection_index is not None:
        index = model_arguments.application_selection_index
        if not 1 <= index <= len(context.task.application_candidates):
            raise ValueError("application selection index is out of range")
        application_id = context.task.application_candidates[index - 1].application_id
    elif model_arguments.interview_selection_index is not None:
        index = model_arguments.interview_selection_index
        if not 1 <= index <= len(context.task.interview_candidates):
            raise ValueError("interview selection index is out of range")
        candidate = context.task.interview_candidates[index - 1]
        application_id = candidate.application_id
        interview_round_id = candidate.interview_round_id

    if application_id is None:
        raise ValueError("start_mock_interview requires an active application")
    if interview_round_id is None and context.task.active_interview_round_id is not None:
        active = next(
            (
                candidate
                for candidate in context.task.interview_candidates
                if candidate.interview_round_id
                == context.task.active_interview_round_id
                and candidate.application_id == application_id
            ),
            None,
        )
        if active is not None:
            interview_round_id = active.interview_round_id

    return StartMockInterviewWorkflowInput(
        user_id=context.profile.user_id,
        application_id=application_id,
        interview_round_id=interview_round_id,
        interview_type=model_arguments.interview_type,
        max_primary_questions=model_arguments.max_primary_questions,
        max_follow_ups_per_question=model_arguments.max_follow_ups_per_question,
    ).model_dump()


def project_restart_mock_interview_arguments(
    context: MainAgentContext, arguments: dict[str, Any]
) -> dict[str, Any]:
    _reject_internal_identifiers("restart_mock_interview", arguments)
    RestartMockInterviewToolArguments.model_validate(arguments)
    # The stuck run is found by user, not named by the model: there is only one
    # unfinished run per user, and naming it would mean exposing its id.
    return {"user_id": context.profile.user_id}


def project_mock_interview_result_arguments(
    context: MainAgentContext, arguments: dict[str, Any]
) -> dict[str, Any]:
    _reject_internal_identifiers("get_mock_interview_result", arguments)
    model_arguments = GetMockInterviewResultToolArguments.model_validate(arguments)
    if model_arguments.reference_index is not None:
        # A report id names one exact run, so the application is not needed and
        # must not be sent: passing both would let the handler fall back to the
        # newest run for that application and quietly answer about a different
        # interview than the turn the user pointed at.
        return {
            "user_id": context.profile.user_id,
            "report_id": context.resolve_reference_index(
                reference_index=model_arguments.reference_index,
                kind="mock_interview_report",
            ),
            "question_number": model_arguments.question_number,
        }
    application_id = context.task.active_application_id
    if model_arguments.application_selection_index is not None:
        index = model_arguments.application_selection_index
        if not 1 <= index <= len(context.task.application_candidates):
            raise ValueError("application selection index is out of range")
        application_id = context.task.application_candidates[index - 1].application_id
    if application_id is None:
        raise ValueError("get_mock_interview_result requires an active application")
    return {
        "user_id": context.profile.user_id,
        "application_id": application_id,
        "question_number": model_arguments.question_number,
    }


def project_action_center_arguments(
    context: MainAgentContext, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    _reject_internal_identifiers(name, arguments)
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
            if not 1 <= selection_index <= len(context.task.action_candidates):
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
    _reject_internal_identifiers(name, arguments)
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
            if not 1 <= interview_index <= len(context.task.interview_candidates):
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
            if not 1 <= account_index <= len(context.task.calendar_account_candidates):
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
