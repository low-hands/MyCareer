from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import json
import hashlib
from contextvars import ContextVar
from time import perf_counter
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Any, ClassVar, Literal, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.career_context import CareerContextProjector
from career_agent.agent.decision_attempts import (
    DecisionAttempt,
    observing_decision_attempts,
)
from career_agent.agent.decision_messages import decision_context_chars
from career_agent.harness.capability_steps import (
    CapabilityStep,
    observing_capability_steps,
)
from career_agent.agent.main_agent_contracts import ActiveSavedJobContextItem, AgentDecision, AttachedResumeContext, ConversationResourceReference, ConversationSpanView, ConversationTaskState, DECISION_OBSERVATION_BODY_LIMIT, DOMAIN_TOOL_PROFILES, DecisionMaker, DecisionObservation, GetCareerMemoryDetailToolArguments, MainAgentContext, MAX_DECISION_OBSERVATIONS, ReadConversationSpanToolArguments, ResolveClaimSourceToolArguments, RouteToCapabilityToolArguments, SavedJobCandidateContextItem, SearchCareerEpisodesToolArguments, SearchCareerHistoryToolArguments, SearchCareerMemoryToolArguments, ToolCall, ToolObservation, UpdateOwnerSettingsToolArguments, append_decision_observation, decision_observation_chars, project_action_center_arguments, project_calendar_arguments, project_career_fact_arguments, project_free_text_preference_arguments, project_job_intent_arguments, project_constraint_retirement_arguments, project_memory_amendment_arguments, project_working_notes_arguments, project_memory_tombstone_arguments, project_email_arguments, project_interview_arguments, project_interview_preparation_arguments, project_job_research_arguments, project_mock_interview_arguments, project_mock_interview_result_arguments, project_open_job_search_arguments, project_restart_mock_interview_arguments, project_resume_arguments, project_saved_job_arguments
from career_agent.agent.conversation_span_presenter import render_conversation_span
from career_agent.agent.conversation_span_requests import explicit_sequence_span
from career_agent.agent.summary_text import DELIVERY_SUMMARY_LIMIT, MODEL_REPLY_LIMIT, clamp
from career_agent.services.free_text_preferences import is_explicit_confirmation
from career_agent.harness.observability import (
    ACTIVE_TRACE_CONTEXT,
    EventType,
    ModelCallCategory,
    TraceRecorder,
    conversation_trace_key,
    record_active_trace,
)
from career_agent.harness.memory_telemetry import (
    memory_context_observation,
)
from career_agent.agent.delivery_policy import (
    condenses_message,
    policy_for,
    delivers_body_elsewhere,
    is_failed,
)
from career_agent.agent.tool_effects import (
    ToolEffect,
    effect_for,
    is_external_write,
    is_notes_guarded,
    is_preference_bound,
    replay_safe,
)
from career_agent.agent.working_notes_guard import (
    remembered_preference_without_authority,
    working_notes_only_tokens,
)
from career_agent.storage.capability_confirmations import (
    CapabilityConfirmationExpiredError,
    CapabilityConfirmationInProgressError,
    CapabilityConfirmationSettledError,
    SQLiteCapabilityConfirmationStore,
    arguments_hash,
)
from career_agent.agent.main_agent_reducers import reduce_task_state
from career_agent.agent.input_resources import (
    InputResourceNotFoundError,
    InputResourceRejectedError,
    resolve_input_resources,
    resolve_job_input_resources,
)
from career_agent.agent.main_agent_tools import MainAgentToolOutput, MainAgentToolRegistry
from career_agent.agent.interview_preparation_presenter import render_interview_preparation
from career_agent.agent.interview_retro_presenter import (
    InterviewRetroView,
    render_interview_retro,
)
from career_agent.agent.daily_brief_presenter import render_daily_brief
from career_agent.agent.job_comparison_presenter import render_job_comparison
from career_agent.agent.job_research_presenter import render_job_research
from career_agent.agent.mock_interview_presenter import (
    render_mock_interview_question,
    render_mock_interview_result,
    render_mock_interview_turn,
)
from career_agent.agent.resume_analysis_contracts import ResumeAnalysisResult
from career_agent.agent.resume_analysis_presenter import render_resume_analysis
from career_agent.agent.delivered_body_contracts import (
    BodyDependency,
    ResumeAnalysisBodySource,
    SavedJobBodySource,
)
from career_agent.agent.resume_job_match_contracts import ResumeJobMatchResult
from career_agent.agent.resume_job_match_presenter import render_resume_job_match
from career_agent.agent.resume_tailoring_contracts import ResumeTailoringResult
from career_agent.agent.resume_tailoring_presenter import (
    TailoringChangeReviewView,
    render_resume_tailoring,
)
from career_agent.domain.job_comparison import JobComparison
from career_agent.agent.mock_interview_contracts import (
    MockInterviewGraphResult,
    MockInterviewQuestionView,
    MockInterviewResultView,
)
from career_agent.agent.openai_compatible_client import AgentWorkerError
from career_agent.domain.interview_preparation import InterviewPreparationResult
from career_agent.domain.job_research import (
    JobResearchDraft,
    JobResearchFindingDraft,
    JobResearchSourceDraft,
)
from career_agent.domain.resume import ResumeArtifactDelivery
from career_agent.services.episode_consolidation import drafts_from_tool_results
from career_agent.services.episode_reconciliation import EpisodeReconciler
from career_agent.harness.streaming import (
    ArtifactReadyEvent,
    CapabilityCompletedEvent,
    CapabilityStartedEvent,
    ClientActionEvent,
    ContentDeltaEvent,
    InteractionOption,
    InteractionRequiredEvent,
    InteractionResponse,
    JobResourceReadyEvent,
    ProgressEvent,
    PublicStreamEvent,
    ReportReadyEvent,
    StreamEventSink,
    TurnCompletedEvent,
    TurnFailedEvent,
    TurnInputResource,
    TurnStartedEvent,
    TurnSuspendedEvent,
    interaction_id,
    iter_content_deltas,
    capability_confirmation_event,
    resume_analysis_confirmation_event,
)
from career_agent.storage.action_executions import (
    ActionExecutionAlreadyFailedError,
    ActionExecutionReconciliationRequiredError,
    RESULT_STATE_RECEIPT_KEY,
    SQLiteActionExecutionStore,
)
from career_agent.storage.context import DeliveredBodyDraft
from career_agent.storage.intent_versions import intent_entry_id
from career_agent.storage.turn_receipts import (
    REPLAYED_EVENT_TYPES,
    SQLiteTurnReceiptStore,
    TurnReceipt,
)

_STREAM_SINK: ContextVar[StreamEventSink | None] = ContextVar(
    "main_agent_stream_sink",
    default=None,
)

# Compatibility alias for focused runtime tests and callers that already bind
# the turn context directly.  The owner moved to the harness so capability
# workers can emit into the same run without importing this module.
_TRACE_CONTEXT = ACTIVE_TRACE_CONTEXT

_ACTION_INVOCATION: ContextVar[tuple[str, str | None] | None] = ContextVar(
    "main_agent_action_invocation",
    default=None,
)

ACTION_EXECUTION_POLICY_EPOCH = 1

DEFAULT_MAX_READ_CALLS = 6
_MAX_RECEIPT_KEYS = 20
_MAX_RECEIPT_VALUE_CHARS = 500

Originator = Literal["model", "user", "runtime"]
"""Who asked for a turn. Derived from its origin variant, never set beside it."""

OriginKind = Literal["model", "interaction", "workflow", "policy"]
"""The union's tag, and the single source of each variant's ``label`` prefix.

``isinstance`` is what narrows in this process. The tag exists so the prefix in
``label`` — published by the CLI and hashed into interaction ids — comes from
the class rather than from a loose string in each variant that happens to match
its class name.

It is a ``ClassVar``, so it is deliberately **not** a dataclass field: it cannot
be passed to the constructor and therefore cannot be set to disagree with the
variant it tags, for the same reason ``requested_by`` is a property. The cost is
that it does not serialize — ``fields()`` and ``asdict()`` do not see it. That
is fine while nothing crosses a process boundary as an origin: the CLI publishes
``label``, a string it builds here. Reconstructing a variant from JSON would
need a real discriminator — an instance field, a Pydantic tagged union, or an
explicit codec — and this is not one.
"""


class TurnInProgressError(RuntimeError):
    """The request's first attempt has not settled, so it cannot be replayed yet."""

    def __init__(self, request_id: str) -> None:
        super().__init__(f"request {request_id} is still running")
        self.request_id = request_id


@dataclass(frozen=True)
class ReplayedTurn:
    """A repeated request answered from its receipt; nothing executed."""

    turn_id: str
    request_id: str
    events: tuple[PublicStreamEvent, ...]

    @property
    def assistant_message(self) -> str:
        return "".join(
            event.delta for event in self.events if isinstance(event, ContentDeltaEvent)
        )


@dataclass(frozen=True)
class ModelDecision:
    """The model chose this turn's action, and this is the choice it made."""

    decision: AgentDecision

    kind: ClassVar[OriginKind] = "model"
    requested_by: ClassVar[Originator] = "model"

    @property
    def label(self) -> str:
        return f"{self.kind}:{self.decision.action}"


@dataclass(frozen=True)
class InteractionReceipt:
    """The user answered an interaction whose contract was sealed when issued.

    Named ``InteractionReceipt`` rather than ``InteractionResponse`` because
    that name is already taken in this module by the inbound UI object; this is
    what the turn became after resolving one.
    """

    scope: str
    action: str

    kind: ClassVar[OriginKind] = "interaction"
    requested_by: ClassVar[Originator] = "user"

    @property
    def label(self) -> str:
        return f"{self.kind}:{self.scope}"


@dataclass(frozen=True)
class RuntimeAction:
    """The user's message went to a workflow the runtime owns.

    The user supplied the input; the runtime, not the user and not the model,
    decided that this input belongs to this workflow. That is a different kind
    of thing from an explicit approval, and the variant is what says so —
    previously it was a second enum field beside a decision the model never made.
    """

    workflow: Literal["job_discovery", "mock_interview"]
    """The business workflow, not the handler that advanced it.

    This is published through the CLI and hashed into interaction ids, so it has
    to be a stable identity. ``handle_mock_interview_input`` and
    ``retry_mock_interview`` are two internal handlers for one workflow: naming
    them here would leak the runtime's own routing into an external surface and
    make the reported origin change when that routing is refactored. Which
    handler ran is already recoverable from the tool result and the trace, where
    an internal name belongs.
    """

    kind: ClassVar[OriginKind] = "workflow"
    requested_by: ClassVar[Originator] = "user"

    @property
    def label(self) -> str:
        return f"{self.kind}:{self.workflow}"


@dataclass(frozen=True)
class RuntimePolicyAction:
    """A deterministic policy step taken before model deliberation."""

    policy: Literal[
        "free_text_preference_confirmation",
        "free_text_preference_activation",
        "career_fact_confirmation",
        "job_intent_confirmation",
    ]

    kind: ClassVar[OriginKind] = "policy"
    requested_by: ClassVar[Originator] = "runtime"

    @property
    def label(self) -> str:
        return f"{self.kind}:{self.policy}"


TurnOrigin = ModelDecision | InteractionReceipt | RuntimeAction | RuntimePolicyAction
"""How a turn came to exist.

The ingresses are not different shapes of one decision. Typing them as one
``AgentDecision`` meant two of them had to fabricate a decision the model never
made, and every consumer of ``turn.decision`` was reading a value that might be
invented. A flag beside it (``decision_source``, later ``requested_by``) only
moved the obligation onto the reader: the CLI still had to remember to check
before publishing a tool name, and forgetting was the original incident.

A discriminated union removes the value instead of guarding it. The ~20 shared
fields stay on the envelope and need no narrowing; the three consumers that
genuinely want the model's choice narrow to ``ModelDecision`` and cannot reach a
fabricated one, because none exists.
"""

DEFAULT_MAX_WRITE_CALLS = 1
DEFAULT_MAX_EXTERNAL_WRITE_CALLS = 1
DEFAULT_MAX_PROJECTION_REFUSALS = 2
DEFAULT_MAX_AUTHORIZATION_REFUSALS = 1
DEFAULT_MAX_FAILURE_RETRIES = 2


class PendingAction(TypedDict, total=False):
    name: str
    kind: Literal["atomic_tool", "workflow"]
    runtime_owned: bool
    effect: ToolEffect
    reducer_result: MainAgentToolOutput
    arguments: dict[str, Any]
    result: MainAgentToolOutput
    synthetic_kind: Literal["projection", "authorization", "confirmation"]
    # Set only by the confirmation ingress, after a PENDING seal was consumed.
    # It is the owner's answer carried into authorization, and the reason the
    # owner rule is not re-judged for an action they have already approved.
    owner_confirmed: bool
    confirmation_id: str
    # A deterministic runtime policy chose this action; the model is not asked
    # to revise it and, once it ran, not asked to continue the turn either.
    policy_owned: bool
    # A policy-owned read that only opens the turn: after it runs the turn
    # continues at ``hydrate`` and the model decides with its observation in
    # hand, instead of ending at ``present``.
    policy_prelude: bool


class LoopControl(TypedDict, total=False):
    read_calls: int
    # Profile switches. Outside the read/write budgets: they change what the
    # next decision is offered, not any store.
    control_calls: int
    # Every write, internal or external: the durable write slot and the turn's
    # delegated write count are derived from it.
    write_calls: int
    # The subset of ``write_calls`` whose effect left this deployment. Budgeted
    # separately so a reversible local record and an external booking are not
    # competing for one slot.
    external_write_calls: int
    projection_refusals: int
    authorization_refusals: int
    fingerprints: tuple[str, ...]
    retryable_fingerprints: tuple[str, ...]
    retry_counts: dict[str, int]
    # Whether this turn has already stamped access on the episodes it projected.
    # A turn can reach ``decide`` several times as observations come back, and
    # the model sees the same projected episodes each time; the exposure is one
    # exposure.
    episodes_marked: bool


class MainAgentState(TypedDict, total=False):
    context: MainAgentContext
    decision: AgentDecision
    pending: PendingAction
    authorization_route: Literal["act", "observe", "present", "interrupt"]
    tool_results: tuple[MainAgentToolOutput, ...]
    control: LoopControl
    artifact_ids: tuple[str, ...]
    assistant_message: str
    model_message: str
    # What career memory this turn put in front of the model. Kept on the
    # state rather than read off the final context, because ``observe``
    # reloads the context after a memory write and that reload cannot know
    # what the earlier prompt already contained.
    career_memory_scope_keys: tuple[str, ...]


class MainAgentTurnResult:
    """One completed turn: how it started, and what it produced.

    ``origin`` is the only field that differs by ingress. Everything below it is
    common to all three and is read by roughly twenty consumers that have no
    reason to know which ingress ran — keeping those on the envelope is why this
    is a union inside one result rather than three parallel result types.

    Accountability lives in the variant, not beside it. ``requested_by`` is a
    property derived from ``origin``, so it cannot disagree with the shape it
    describes. There is deliberately no ``authority`` field: in this deployment
    the runtime is the sole executor and policy enforcer for every turn, which
    is exactly PCAA's *runtime* authority, so a per-turn field would be a
    constant — and a constant is worse than an absence. What the old
    ``authority`` was really trying to distinguish, an explicit human approval
    from the runtime acting on its own ownership rule, is now the difference
    between ``InteractionReceipt`` and ``RuntimeAction``.

    Attribution here is still turn-level. A model turn can produce several tool
    calls, some of them runtime-owned, and per-action provenance belongs on the
    execution records rather than on this envelope; see 071 三-7.
    """

    def __init__(self, *, origin: TurnOrigin, context: MainAgentContext, assistant_message: str, tool_result: MainAgentToolOutput | None = None, tool_results: tuple[MainAgentToolOutput, ...] = (), artifacts: tuple[ResumeArtifactDelivery, ...] = (), content_streamed: bool = False, model_message: str = "", delegated_read_count: int = 0, delegated_write_count: int = 0, career_memory_scope_keys: tuple[str, ...] = ()) -> None:
        self.origin = origin
        self.context = context
        self.assistant_message = assistant_message
        self.tool_result = tool_result
        self.tool_results = tool_results
        self.artifacts = artifacts
        self.content_streamed = content_streamed
        # The reply alone, without any presenter body appended for the screen.
        # Non-empty exactly when the model authored this turn's answer.
        self.model_message = model_message
        self.delegated_read_count = delegated_read_count
        self.delegated_write_count = delegated_write_count
        # The career scopes this turn showed the model, carried out to the
        # commit so the stored messages can be bound to them and later
        # suppressed if one of those scopes is tombstoned.
        self.career_memory_scope_keys = career_memory_scope_keys

    @property
    def requested_by(self) -> Originator:
        return self.origin.requested_by

    @property
    def model_decision(self) -> AgentDecision | None:
        """The model's choice, or ``None`` when the model made none.

        The narrowing a consumer would otherwise write by hand. It differs from
        the field it replaces in the only way that matters: a turn the model did
        not decide yields nothing, instead of an invented ``final`` or an
        invented tool name that reads as real.
        """
        return self.origin.decision if isinstance(self.origin, ModelDecision) else None


class MainAgentRuntime:
    _INTERACTION_RENDERER_STATES = frozenset(
        {
            "calendar_approval_required",
            "capability_confirmation_required",
            "email_events_pending",
            "constraint_retirement_proposed",
            "memory_amendment_proposed",
            "memory_tombstone_proposed",
            "free_text_preference_confirmation_proposed",
            "free_text_preference_confirmed_structured_proposed",
            "career_fact_proposed",
            "mock_interview_answer_required",
            "mock_interview_running",
            "resume_analysis_ready",
            "resume_final_review_blocked",
            "resume_tailoring_review_blocked",
            "resume_tailoring_superseded",
        }
    )

    @classmethod
    def _has_interaction_renderer(cls, state: str) -> bool:
        return state in cls._INTERACTION_RENDERER_STATES

    _MOCK_INTERVIEW_GRAPH_STATES = frozenset(
        {
            "mock_interview_answer_required",
            "mock_interview_running",
            "mock_interview_completed",
            "mock_interview_cancelled",
        }
    )

    def __init__(self, *, context_manager: ContextManager, decision_maker: DecisionMaker, tools: MainAgentToolRegistry, career_context_projector: CareerContextProjector | None = None, max_read_calls: int = DEFAULT_MAX_READ_CALLS, max_write_calls: int = DEFAULT_MAX_WRITE_CALLS, max_external_write_calls: int = DEFAULT_MAX_EXTERNAL_WRITE_CALLS, max_projection_refusals: int = DEFAULT_MAX_PROJECTION_REFUSALS, max_authorization_refusals: int = DEFAULT_MAX_AUTHORIZATION_REFUSALS, max_failure_retries: int = DEFAULT_MAX_FAILURE_RETRIES, owned_resources: tuple[Any, ...] = (), trace_recorder: TraceRecorder | None = None, action_execution_store: SQLiteActionExecutionStore | None = None, capability_confirmation_store: SQLiteCapabilityConfirmationStore | None = None, turn_receipt_store: SQLiteTurnReceiptStore | None = None, action_policy_epoch: int = ACTION_EXECUTION_POLICY_EPOCH, episode_reconciler: EpisodeReconciler | None = None) -> None:
        if max_read_calls < 1:
            raise ValueError("max_read_calls must be at least one")
        if max_write_calls < 1:
            raise ValueError("max_write_calls must be at least one")
        if max_external_write_calls < 1:
            raise ValueError("max_external_write_calls must be at least one")
        if max_projection_refusals < 1:
            raise ValueError("max_projection_refusals must be at least one")
        if max_authorization_refusals < 1:
            raise ValueError("max_authorization_refusals must be at least one")
        if max_failure_retries < 0:
            raise ValueError("max_failure_retries cannot be negative")
        if action_policy_epoch < 1:
            raise ValueError("action_policy_epoch must be positive")
        if (
            max_read_calls
            + max_write_calls
            + max_external_write_calls
            + max_projection_refusals
            + max_authorization_refusals
            > MAX_DECISION_OBSERVATIONS
        ):
            raise ValueError(
                "read, write, and refusal budgets must fit the observation window"
            )
        self._context_manager = context_manager
        # Exposed for the CLI's post-turn maintenance notice, which is an
        # operator concern and deliberately never reaches the decision model.
        self.context_manager = context_manager
        context_manager.on_compaction(self._announce_compaction)
        self._decision_maker = decision_maker
        self._tools = tools
        self._career_context_projector = career_context_projector
        self._episode_reconciler = episode_reconciler
        self._reconciled_users: set[str] = set()
        self._episode_reconcile_guard = Lock()
        self._episode_reconcile_locks: dict[str, Any] = {}
        self._max_read_calls = max_read_calls
        self._max_write_calls = max_write_calls
        self._max_external_write_calls = max_external_write_calls
        self._max_projection_refusals = max_projection_refusals
        self._max_authorization_refusals = max_authorization_refusals
        self._max_failure_retries = max_failure_retries
        self._trace_recorder = trace_recorder
        self._action_execution_store = action_execution_store
        self._capability_confirmation_store = capability_confirmation_store
        self._turn_receipt_store = turn_receipt_store
        self._action_policy_epoch = action_policy_epoch
        self._owned_resources = owned_resources
        self._closed = False
        request_token_usage = getattr(decision_maker, "request_token_usage", None)
        if callable(request_token_usage):
            self._decision_tool_schemas = self._tools.schemas()
            self._decision_tool_schema_chars = len(
                json.dumps(
                    self._decision_tool_schemas,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )

            def estimate_complete_request(
                context: MainAgentContext,
            ) -> tuple[int, int]:
                if self._career_context_projector is not None:
                    context = context.model_copy(
                        update={
                            "career_memory": self._career_context_projector.project(
                                user_id=context.profile.user_id,
                                query=context.user_message,
                            )
                        }
                    )
                return request_token_usage(context, self._decision_tool_schemas)

            static_request_token_usage = getattr(
                decision_maker, "static_request_token_usage", None
            )
            if not callable(static_request_token_usage):
                raise ValueError(
                    "decision makers that report request token usage must also "
                    "report static request token usage"
                )
            static_tokens, max_input_tokens = static_request_token_usage(
                self._decision_tool_schemas
            )
            self._context_manager.configure_request_token_estimator(
                estimate_complete_request,
                static_input_tokens=static_tokens,
                max_input_tokens=max_input_tokens,
            )

        graph = StateGraph(MainAgentState)
        graph.add_node("hydrate", self._hydrate_career_context)
        graph.add_node("decide", self._decide)
        graph.add_node("authorize", self._authorize)
        graph.add_node("act", self._act)
        graph.add_node("observe", self._observe)
        graph.add_node("present", self._present)
        graph.add_node("interrupt", self._interrupt)
        graph.add_conditional_edges(
            START,
            self._route_entry,
            {
                "hydrate": "hydrate",
                "authorize": "authorize",
            },
        )
        graph.add_edge("hydrate", "decide")
        graph.add_conditional_edges(
            "decide",
            self._route_decision,
            {
                "authorize": "authorize",
                "present": "present",
                "interrupt": "interrupt",
            },
        )
        graph.add_conditional_edges(
            "authorize",
            self._after_authorize,
            {
                "act": "act",
                "observe": "observe",
                "present": "present",
                "interrupt": "interrupt",
            },
        )
        graph.add_edge("act", "observe")
        graph.add_conditional_edges(
            "observe",
            self._after_observe,
            {
                "hydrate": "hydrate",
                "decide": "decide",
                "present": "present",
                "interrupt": "interrupt",
            },
        )
        graph.add_edge("present", END)
        graph.add_edge("interrupt", END)
        self._graph = graph.compile()

    @staticmethod
    def _route_entry(state: MainAgentState) -> Literal["hydrate", "authorize"]:
        """Enter at authorization when the action is already decided.

        Runtime-owned workflows, owner-confirmed actions, and deterministic
        policy actions arrive with the action in hand. They skip ``decide``:
        consulting the model would let it revise a choice already made by the
        runtime's ownership/policy rule or by a person clicking confirm.
        """

        pending = state.get("pending", {})
        return (
            "authorize"
            if pending.get("runtime_owned")
            or pending.get("owner_confirmed")
            or pending.get("policy_owned")
            else "hydrate"
        )

    def close(self) -> None:
        if self._closed:
            return
        for resource in reversed(self._owned_resources):
            close = getattr(resource, "close", None)
            if close is not None:
                close()
        self._closed = True

    @staticmethod
    def _emit(event: PublicStreamEvent) -> None:
        sink = _STREAM_SINK.get()
        if sink is None:
            return
        try:
            sink(event)
        except Exception:
            # Streaming is an observer. Losing a client or a faulty UI adapter
            # must not roll back a capability that may already have external or
            # durable effects.
            return

    @staticmethod
    def _public_capability(name: str) -> str:
        if name in {
            "open_job_search",
            "find_saved_jobs",
            "get_saved_job",
        }:
            return "job_search"
        if "job_research" in name or name in {"research_job", "retry_job_research"}:
            return "job_research"
        if "resume" in name:
            return "resume"
        if "calendar" in name:
            return "calendar"
        if "interview" in name:
            return "interview"
        if "application" in name or "email" in name:
            return "application_tracking"
        if "action" in name or "daily_brief" in name:
            return "action_center"
        return "career_task"

    _CAPABILITY_LABELS: ClassVar[dict[str, str]] = {
        "job_search": "正在处理岗位检索……",
        "job_research": "正在调研岗位相关业务信息……",
        "resume": "正在处理简历……",
        "application_tracking": "正在处理投递进展……",
        "interview": "正在处理面试任务……",
        "calendar": "正在准备日历操作……",
        "action_center": "正在整理待办事项……",
        "career_task": "正在执行职业任务……",
    }

    @classmethod
    def _emit_capability_started(cls, name: str) -> None:
        capability = cls._public_capability(name)
        cls._emit(
            CapabilityStartedEvent(
                capability=capability,
                message=cls._CAPABILITY_LABELS[capability],
            )
        )

    @classmethod
    def _emit_capability_completed(cls, name: str, state: str) -> None:
        capability = cls._public_capability(name)
        labels = {
            "job_search": "岗位检索步骤已完成。",
            "job_research": "岗位调研步骤已完成。",
            "resume": "简历处理步骤已完成。",
            "application_tracking": "投递进展处理已完成。",
            "interview": "面试处理步骤已完成。",
            "calendar": "日历准备步骤已完成。",
            "action_center": "待办整理步骤已完成。",
            "career_task": "职业任务步骤已完成。",
        }
        cls._emit(
            CapabilityCompletedEvent(
                capability=capability,
                state=state,
                message=labels[capability],
            )
        )

    def run_turn(
        self,
        *,
        user_id: str,
        conversation_id: str,
        user_message: str,
        request_id: str | None = None,
        interaction_response: InteractionResponse | None = None,
        event_sink: StreamEventSink | None = None,
        input_resources: tuple[TurnInputResource, ...] = (),
    ) -> MainAgentTurnResult | ReplayedTurn:
        """Run one committed turn and optionally publish presentation-only events.

        The sink is held outside graph state and checkpoints. A broken observer
        never gets authority to fail or mutate the business turn.

        With a receipt store, ``request_id`` identifies the turn as a whole: a
        request whose key already committed is answered from its receipt and
        returns a ``ReplayedTurn`` without executing anything.
        """

        turn_id = uuid4().hex
        if request_id is not None:
            request_id = request_id.strip()
            if not request_id or len(request_id) > 200:
                raise ValueError("request_id must contain 1 to 200 characters")
        receipt_owner: tuple[SQLiteTurnReceiptStore, str] | None = None
        if request_id is not None and self._turn_receipt_store is not None:
            existing = self._turn_receipt_store.begin(
                user_id=user_id,
                conversation_id=conversation_id,
                request_id=request_id,
                turn_id=turn_id,
            )
            if existing is not None:
                return self._replay_turn(existing, event_sink=event_sink)
            receipt_owner = (self._turn_receipt_store, request_id)
        answered: list[PublicStreamEvent] = []
        if receipt_owner is not None:
            event_sink = self._answer_recording_sink(event_sink, answered)
        sink_token = _STREAM_SINK.set(event_sink)
        action_token = _ACTION_INVOCATION.set((turn_id, request_id))
        trace_token = _TRACE_CONTEXT.set(
            (self._trace_recorder, turn_id) if self._trace_recorder is not None else None
        )
        self._emit(TurnStartedEvent(turn_id=turn_id))
        self._emit(
            ProgressEvent(
                stage="loading_context",
                message="正在读取对话和职业上下文……",
            )
        )
        reply_delivered = False

        def deliver_reply(result: MainAgentTurnResult) -> None:
            nonlocal reply_delivered
            self._deliver_reply(result=result, conversation_id=conversation_id)
            reply_delivered = True

        try:
            result = self._run_and_commit_turn(
                user_id=user_id,
                conversation_id=conversation_id,
                user_message=user_message,
                interaction_response=interaction_response,
                before_commit=deliver_reply,
                input_resources=input_resources,
            )
            self._record_turn(turn_id=turn_id, conversation_id=conversation_id, result=result)
            self._deliver_stream_events(
                result=result,
                turn_id=turn_id,
                conversation_id=conversation_id,
            )
            if receipt_owner is not None:
                self._settle_turn_receipt(
                    receipt_owner,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    answered=tuple(answered),
                    body_expires_at=min(
                        (
                            output.body_source.expires_at
                            for output in (
                                result.tool_results
                                or ((result.tool_result,) if result.tool_result else ())
                            )
                            if isinstance(output.body_source, ResumeAnalysisBodySource)
                        ),
                        default=None,
                    ),
                )
            return result
        except Exception as error:
            self._invalidate_episode_reconciliation(user_id)
            self._record_turn_failed(
                turn_id=turn_id,
                conversation_id=conversation_id,
                error=error,
                reply_delivered=reply_delivered,
            )
            if receipt_owner is not None:
                self._settle_turn_receipt(
                    receipt_owner,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                    answered=None,
                )
            if reply_delivered:
                # The reader already has the reply; what failed is keeping it.
                # Say so instead of a generic failure that reads as if the text
                # on screen were wrong.
                self._emit(
                    TurnFailedEvent(
                        turn_id=turn_id,
                        code="TURN_COMMIT_FAILED",
                        message="回复已生成，但本轮状态未能保存；刷新后这条回复可能不会保留。",
                    )
                )
            elif isinstance(error, InputResourceNotFoundError):
                self._emit(
                    TurnFailedEvent(
                        turn_id=turn_id,
                        code="INPUT_RESOURCE_NOT_FOUND",
                        message="附带的简历版本不存在或不属于当前用户，请重新选择后再发送。",
                    )
                )
            elif isinstance(error, InputResourceRejectedError):
                self._emit(
                    TurnFailedEvent(
                        turn_id=turn_id,
                        code="INPUT_RESOURCE_REJECTED",
                        message="当前模拟面试进行中，此时不会读取附带的简历。请完成或退出当前流程后再发送。",
                    )
                )
            else:
                self._emit(
                    TurnFailedEvent(
                        turn_id=turn_id,
                        code="TURN_EXECUTION_FAILED",
                        message="本轮处理失败，请稍后重试。",
                    )
                )
            raise
        finally:
            _STREAM_SINK.reset(sink_token)
            _TRACE_CONTEXT.reset(trace_token)
            _ACTION_INVOCATION.reset(action_token)

    @staticmethod
    def _answer_recording_sink(
        event_sink: StreamEventSink | None,
        answered: list[PublicStreamEvent],
    ) -> StreamEventSink:
        """Keep the events that make up the answer before handing them on.

        Recording happens ahead of the observer, so a client that disconnects
        mid-stream still leaves a complete receipt behind.
        """

        def sink(event: PublicStreamEvent) -> None:
            if isinstance(event, REPLAYED_EVENT_TYPES):
                answered.append(event)
            if event_sink is not None:
                event_sink(event)

        return sink

    @staticmethod
    def _settle_turn_receipt(
        owner: tuple[SQLiteTurnReceiptStore, str],
        *,
        user_id: str,
        conversation_id: str,
        turn_id: str,
        answered: tuple[PublicStreamEvent, ...] | None,
        body_expires_at: datetime | None = None,
    ) -> None:
        store, request_id = owner
        try:
            if answered is None:
                store.fail(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    request_id=request_id,
                    turn_id=turn_id,
                )
            else:
                store.commit(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    request_id=request_id,
                    turn_id=turn_id,
                    events=answered,
                    body_expires_at=body_expires_at,
                )
        except Exception:
            # The receipt is a convenience for a retrying client. Failing to
            # write it must not undo a turn whose effects are already durable.
            return

    def _replay_turn(
        self,
        receipt: TurnReceipt,
        *,
        event_sink: StreamEventSink | None,
    ) -> ReplayedTurn:
        """Answer a repeated request from its receipt instead of executing."""

        sink_token = _STREAM_SINK.set(event_sink)
        try:
            if receipt.status == "RUNNING":
                self._emit(
                    TurnFailedEvent(
                        turn_id=receipt.turn_id,
                        code="TURN_IN_PROGRESS",
                        message="这条请求仍在处理中；稍后重新读取对话即可看到结果。",
                    )
                )
                raise TurnInProgressError(receipt.request_id)
            self._emit(TurnStartedEvent(turn_id=receipt.turn_id))
            events = receipt.events
            if receipt.content_status != "available":
                events = (
                    ContentDeltaEvent(
                        delta=(
                            "内容已删除。"
                            if receipt.content_status == "deleted"
                            else "回执正文已过期，请查看历史对话。"
                        ),
                        delivery="synthetic",
                    ),
                    TurnCompletedEvent(turn_id=receipt.turn_id),
                )
            for event in events:
                self._emit(event)
        finally:
            _STREAM_SINK.reset(sink_token)
        return ReplayedTurn(
            turn_id=receipt.turn_id,
            request_id=receipt.request_id,
            events=events,
        )

    @staticmethod
    def _emit_trace(
        event_type: Literal["capability_failed", "presentation_degraded"],
        stage: str,
        *,
        error_code: str | None = None,
        error_detail: str | None = None,
        details: dict[str, Any] | None = None,
        recoverable: bool | None = None,
    ) -> None:
        """Record a best-effort runtime trace event under the active turn.

        The active turn's recorder + run id arrive via ``_TRACE_CONTEXT``, set in
        ``run_turn``. Static so the presenter path (``_validated``) can reach it
        without a bound instance. A write that fails must never roll back
        already-durable business effects.
        """
        MainAgentRuntime._record_trace_event(
            event_type,
            stage,
            outcome="failed",
            error_code=error_code,
            error_detail=error_detail,
            details=details,
            recoverable=recoverable,
        )

    @staticmethod
    def _record_trace_event(
        event_type: EventType,
        stage: str,
        *,
        outcome: Literal["started", "succeeded", "failed", "interrupted"],
        duration_ms: int | None = None,
        error_code: str | None = None,
        error_detail: str | None = None,
        details: dict[str, Any] | None = None,
        recoverable: bool | None = None,
        model_call_category: ModelCallCategory | None = None,
    ) -> None:
        """Best-effort event write shared by model and non-model telemetry."""
        record_active_trace(
            event_type,
            stage,
            outcome=outcome,
            duration_ms=duration_ms,
            error_code=error_code,
            error_detail=error_detail,
            details=details,
            recoverable=recoverable,
            model_call_category=model_call_category,
        )

    def _record_turn(
        self,
        *,
        turn_id: str,
        conversation_id: str,
        result: MainAgentTurnResult,
    ) -> None:
        if self._trace_recorder is None:
            return
        try:
            self._trace_recorder.record(
                turn_id,
                "turn_completed",
                "turn",
                outcome="succeeded",
                details={
                    "conversation_id": conversation_id,
                    "tool_call_count": (
                        result.delegated_read_count + result.delegated_write_count
                    ),
                    "read_call_count": result.delegated_read_count,
                    "write_call_count": result.delegated_write_count,
                    "final_state": (
                        result.tool_result.state if result.tool_result is not None else result.origin.label
                    ),
                },
            )
        except Exception:
            # Telemetry is best-effort; a recorder that cannot write must not
            # roll back a turn whose business effects are already durable.
            return

    def record_rejected_turn(self, *, user_id: str, conversation_id: str) -> None:
        """Note a turn the concurrency gate refused before it began.

        Public because the gate lives in the transport, above the loop: by the
        time a turn is rejected there is no turn id, no context and no runtime
        state — only the fact that one conversation was asked to advance twice
        at once.

        Recorded because the choice of what to replace the process-local gate
        with depends on how often that actually happens. A lease and an
        optimistic version number suit opposite contention levels, and this
        deployment has never measured which one it has. Best-effort like every
        other trace write: a rejected turn is already refused, and failing to
        record it must not turn a 409 into a 500.
        """
        if self._trace_recorder is None:
            return
        try:
            self._trace_recorder.record(
                uuid4().hex,
                "turn_rejected",
                "gate",
                outcome="interrupted",
                details={"conversation_id": conversation_id, "user_id": user_id},
            )
        except Exception:
            return

    def _record_turn_failed(
        self,
        *,
        turn_id: str,
        conversation_id: str,
        error: Exception,
        reply_delivered: bool = False,
    ) -> None:
        if self._trace_recorder is None:
            return
        error_code = getattr(error, "code", None)
        if not isinstance(error_code, str) or not error_code.strip():
            error_code = "TURN_EXECUTION_FAILED"
        retryable = getattr(error, "retryable", None)
        if not isinstance(retryable, bool):
            retryable = None
        details: dict[str, Any] = {
            "conversation_id": conversation_id,
            "error_type": type(error).__name__,
        }
        if reply_delivered:
            details["reply_delivered"] = True
        try:
            self._trace_recorder.record(
                turn_id,
                "turn_failed",
                "turn",
                outcome="failed",
                details=details,
                error_code=error_code,
                error_detail=str(error),
                recoverable=retryable,
            )
        except Exception:
            return

    def _commit_interrupted_turn(
        self, *, context: MainAgentContext, error: Exception
    ) -> None:
        """Leave a conversational trace when a turn dies after it already wrote.

        A turn runs to completion and commits afterwards, so an exception in the
        middle drops the whole conversation record — including the user's own
        message — while the domain write that already landed stays in its store.
        Three states then disagree: the applications store says it happened, the
        conversation has zero messages, and ``ConversationTaskState`` still says
        no application is active.

        The write is not lost, and a later ``list_applications`` would find it.
        The danger is narrower and worse: nothing tells the model to go look.
        ``max_write_calls = 1`` exists to make writes deliberate, and this path
        let a deliberate write go invisible at the conversation layer.

        Reachable by design, not by accident: ``_reraise_security_refusal`` hard
        throws through the whole turn for internal identifiers, extra fields and
        unknown capabilities. That boundary stays hard — this does not catch the
        error, it only records what preceded it before re-raising.

        Read from the durable ledger rather than from an in-process list. The
        list could only report writes that had already returned, because it
        appended after the call — so the case this exists for, a process dying
        during the call, left it empty, and a killed process lost it entirely.
        The ledger records intent before the call, which is what makes
        "started, outcome unknown" expressible at all.

        The two are told apart in the message on purpose. "It was written" and
        "it may have been written" ask the reader for different things, and
        conflating them either invites a duplicate or hides a real effect.

        Nothing is written when the turn opened no slot. There is then no
        disagreement to reconcile, and a clean retry is the better outcome. The
        task is committed as it was loaded: the transition this turn intended
        never finished, so claiming it did would be a second lie.
        """
        invocation = _ACTION_INVOCATION.get()
        if invocation is None or self._action_execution_store is None:
            return
        turn_id, request_id = invocation
        executions = self._action_execution_store.list_for_anchor(
            user_id=context.profile.user_id,
            conversation_id=context.conversation_id,
            anchor=request_id or turn_id,
        )
        settled = [item.tool_name for item in executions if item.status == "SUCCEEDED"]
        unsettled = [item.tool_name for item in executions if item.status == "PENDING"]
        if not settled and not unsettled:
            return
        sentences = []
        if settled:
            sentences.append(
                "以下操作已经写入：" + "、".join(dict.fromkeys(settled)) + "。"
            )
        if unsettled:
            sentences.append(
                "以下操作已经开始但没有确认结果，可能已生效也可能没有："
                + "、".join(dict.fromkeys(unsettled))
                + "。"
            )
        try:
            self._context_manager.commit_turn(
                context=context,
                task=context.task,
                assistant_message=(
                    f"本轮执行中断（{type(error).__name__}）。"
                    + "".join(sentences)
                    + "请先核对这些记录的实际状态，再决定是否重做。"
                ),
                turn_id=turn_id,
            )
        except Exception:
            # Best effort, exactly like the trace recorder: a conversation row
            # that cannot be written must not replace the original failure with
            # a less informative one.
            return

    def _reconcile_episodes(self, user_id: str) -> None:
        """Replay durable domain state once per user and after a failed turn.

        The sweep exists for the split-store failure where a domain row commits
        and its episode does not, so it has to run before anything reads L1.
        It does not belong on every turn: it is a full scan of four stores and
        one report read per mock session, and after the first pass each seam
        writes its own episode inside the conversation transaction.

        Any failed turn invalidates the guard. The next turn scans again, which
        closes the observable same-process split-store window without charging
        every healthy turn or requiring four stores to expose watermarks.
        Coordination is per user, so one owner's initial scan cannot block
        another owner's first turn.
        """

        if self._episode_reconciler is None:
            return
        user_lock = self._episode_reconcile_user_lock(user_id)
        with user_lock:
            with self._episode_reconcile_guard:
                if user_id in self._reconciled_users:
                    return
            self._episode_reconciler.reconcile_user(user_id=user_id)
            with self._episode_reconcile_guard:
                self._reconciled_users.add(user_id)

    def _invalidate_episode_reconciliation(self, user_id: str) -> None:
        if self._episode_reconciler is None:
            return
        user_lock = self._episode_reconcile_user_lock(user_id)
        with user_lock:
            with self._episode_reconcile_guard:
                self._reconciled_users.discard(user_id)

    def _episode_reconcile_user_lock(self, user_id: str):
        with self._episode_reconcile_guard:
            return self._episode_reconcile_locks.setdefault(user_id, Lock())

    def _before_commit(
        self,
        result: MainAgentTurnResult,
        hook: Callable[[MainAgentTurnResult], None] | None,
    ) -> None:
        """The run is over and its reply final; only the write remains.

        ``hook`` is where the reply goes out to the reader, ahead of the commit
        rather than after it: the commit changes nothing the reader sees, so
        waiting for it only adds the save (and any compaction) to the time the
        answer sits ready and unshown. Whether it then failed to persist is
        reported separately by the caller.
        """
        if hook is not None:
            hook(result)
        self._emit(ProgressEvent(stage="saving", message="正在保存本轮状态……"))

    def _run_and_commit_turn(
        self,
        *,
        user_id: str,
        conversation_id: str,
        user_message: str,
        interaction_response: InteractionResponse | None = None,
        before_commit: Callable[[MainAgentTurnResult], None] | None = None,
        input_resources: tuple[TurnInputResource, ...] = (),
    ) -> MainAgentTurnResult:
        self._reconcile_episodes(user_id)
        routing_task = self._context_manager.get_task(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        # Checked before any context is built, so a turn that names a version
        # the user does not own, or attaches one while a workflow is reading
        # the messages itself, fails without writing anything.
        if input_resources and self._owns_next_turn(routing_task):
            raise InputResourceRejectedError(
                f"{routing_task.active_workflow} owns this conversation; "
                "attachments are not read until it completes"
            )
        attached_resumes = (
            resolve_input_resources(
                self._tools.resume_store,
                user_id=user_id,
                resources=input_resources,
            )
            if input_resources
            else ()
        )
        attached_jobs = (
            resolve_job_input_resources(
                self._tools.job_repository,
                user_id=user_id,
                resources=input_resources,
            )
            if input_resources
            else ()
        )
        bare_confirmation_target = routing_task.bare_confirmation_target
        if bare_confirmation_target is not None:
            routing_task = self._context_manager.disarm_bare_confirmation(
                user_id=user_id,
                conversation_id=conversation_id,
                task=routing_task,
            )
        if interaction_response is not None:
            context = self._refresh_saved_job_focus(
                self._attach_input_resources(
                    self._context_manager.load_for_turn(
                        user_id=user_id,
                        conversation_id=conversation_id,
                        user_message=user_message,
                    ),
                    attached_resumes,
                    attached_jobs,
                )
            )
            try:
                result = self._run_interaction_response(
                    context=context,
                    conversation_id=conversation_id,
                    response=interaction_response,
                )
            except Exception as error:
                self._commit_interrupted_turn(context=context, error=error)
                raise
            self._before_commit(result, before_commit)
            self._context_manager.commit_turn(
                context=context,
                task=result.context.task,
                assistant_message=self._conversation_content(
                    result.tool_result,
                    screen=result.assistant_message,
                    composed=False,
                ),
                assistant_bodies=MainAgentRuntime._delivered_bodies(
                    result.tool_results
                    or ((result.tool_result,) if result.tool_result else ())
                ),
                episode_drafts=drafts_from_tool_results(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    tool_results=result.tool_results
                    or ((result.tool_result,) if result.tool_result else ()),
                ),
                memory_scope_keys=result.career_memory_scope_keys,
                turn_id=self._active_turn_id(),
            )
            return result

        if self._owns_next_turn(routing_task):
            context = self._context_manager.load_for_workflow_turn(
                user_id=user_id,
                conversation_id=conversation_id,
                task=routing_task,
            )
            try:
                result = self._run_owned_workflow_turn(
                    context=context,
                    user_message=user_message,
                )
            except Exception as error:
                self._commit_interrupted_turn(context=context, error=error)
                raise
            # One decision point for every way a run can end, so no exit path
            # can forget to leave a trace. The test is whether the workflow will
            # still be driving the next turn, not whether it still holds the
            # slot: a dead checkpoint keeps the slot to record why it died, yet
            # hands the conversation back, and that turn needs a trace too.
            self._before_commit(result, before_commit)
            if self._owns_next_turn(result.context.task):
                self._context_manager.commit_workflow_turn(
                    context=context,
                    task=result.context.task,
                )
            else:
                self._context_manager.commit_workflow_exit(
                    context=context,
                    task=result.context.task,
                    # The candidate keeps the full report on screen; the stored
                    # copy is the writer's summary of it, because only that has
                    # to fit alongside the rest of the conversation next turn.
                    assistant_message=self._conversation_content(
                        result.tool_result,
                        screen=self._durable_screen(result),
                        composed=bool(result.model_message),
                    ),
                    assistant_resource_refs=MainAgentRuntime._turn_resource_refs(
                        result.tool_results
                    ),
                    assistant_bodies=MainAgentRuntime._delivered_bodies(
                        result.tool_results
                    ),
                    turn_id=self._active_turn_id(),
                )
            return result

        context = self._refresh_saved_job_focus(
            self._attach_input_resources(
                self._context_manager.load_for_turn(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    user_message=user_message,
                ),
                attached_resumes,
                attached_jobs,
            )
        )
        try:
            result = self._run_loaded_context(
                context,
                bare_confirmation_target=bare_confirmation_target,
            )
        except Exception as error:
            self._commit_interrupted_turn(context=context, error=error)
            raise
        # This input belonged to Main Agent even when its result hands future
        # turns to a workflow. Ownership is an ingress property, not something
        # that can be inferred from the task state after execution. The reply,
        # however, did come from the workflow: it is the run's first question,
        # withheld on the same grounds as every question after it.
        self._before_commit(result, before_commit)
        if self._owns_next_turn(result.context.task):
            held = self._context_manager.commit_workflow_entry(
                context=context,
                task=result.context.task,
                episode_drafts=drafts_from_tool_results(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    tool_results=result.tool_results
                    or ((result.tool_result,) if result.tool_result else ()),
                ),
            )
            # Report the state that was stored, or the next turn would resume
            # from a task whose held request the caller never saw.
            result.context = result.context.model_copy(update={"task": held})
        else:
            self._context_manager.commit_turn(
                context=context,
                task=result.context.task,
                assistant_message=self._conversation_content(
                    result.tool_result,
                    screen=self._durable_screen(result),
                    composed=bool(result.model_message),
                ),
                assistant_resource_refs=MainAgentRuntime._turn_resource_refs(
                    result.tool_results
                ),
                assistant_bodies=MainAgentRuntime._delivered_bodies(
                    result.tool_results
                ),
                episode_drafts=drafts_from_tool_results(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    tool_results=result.tool_results
                    or ((result.tool_result,) if result.tool_result else ()),
                ),
                memory_scope_keys=result.career_memory_scope_keys,
                turn_id=self._active_turn_id(),
            )
        return result

    @staticmethod
    def _attach_input_resources(
        context: MainAgentContext,
        attached_resumes: tuple[AttachedResumeContext, ...],
        attached_jobs: tuple[SavedJobCandidateContextItem, ...] = (),
    ) -> MainAgentContext:
        """Place verified attachments on the turn and make the last one active.

        The active version is what ``analyze_resume`` and its siblings resolve
        "this resume" to, so attaching a version is the same act as choosing
        it; the stored reference on the user message is what keeps the choice
        from drifting when the resume later gains a newer version. A job
        attached the same way becomes the active posting and heads the saved
        job candidates, so ``get_saved_job`` with no selection reads exactly
        the posting the page named rather than whatever was last discussed.
        """
        if not attached_resumes and not attached_jobs:
            return context
        task_updates: dict[str, Any] = {}
        if attached_resumes:
            task_updates["active_resume_version_id"] = attached_resumes[
                -1
            ].resume_version_id
        if attached_jobs:
            attached_ids = {item.job_posting_id for item in attached_jobs}
            focused = attached_jobs[-1]
            task_updates["active_job_posting_id"] = focused.job_posting_id
            # The attachment names the posting; the pin names the JD version
            # the page handed over, so later turns keep reading that one.
            pinned = (
                ActiveSavedJobContextItem(
                    job_posting_id=focused.job_posting_id,
                    jd_snapshot_id=focused.jd_snapshot_id,
                    title=focused.title,
                    company_name=focused.company_name,
                    jd_version=focused.jd_version,
                )
                if focused.jd_snapshot_id is not None
                and focused.jd_version is not None
                else None
            )
            task_updates["active_jd_snapshot_id"] = (
                pinned.jd_snapshot_id if pinned is not None else None
            )
            task_updates["active_saved_job"] = pinned
            task_updates["saved_job_candidates"] = (
                *attached_jobs,
                *(
                    item
                    for item in context.task.saved_job_candidates
                    if item.job_posting_id not in attached_ids
                ),
            )
        return context.model_copy(
            update={
                **({"attached_resumes": attached_resumes} if attached_resumes else {}),
                "task": context.task.model_copy(update=task_updates),
            }
        )

    def _refresh_saved_job_focus(self, context: MainAgentContext) -> MainAgentContext:
        """Tell the model whether the pinned JD can still be read.

        The pin is a statement about an earlier turn; the posting may have been
        deleted since. One owned lookup per turn keeps the projection honest
        without ever putting the JD text into the context.
        """
        focus = context.task.focused_saved_job()
        if focus is None:
            return context
        repository = self._tools.job_repository
        if repository is None:
            return context
        readable = (
            repository.get_snapshot(
                user_id=context.profile.user_id,
                jd_snapshot_id=focus.jd_snapshot_id,
            )
            is not None
        )
        if readable == focus.readable:
            return context
        return context.model_copy(
            update={
                "task": context.task.focus_saved_job(
                    focus.model_copy(update={"readable": readable})
                )
            }
        )

    @staticmethod
    def _active_turn_id() -> str | None:
        invocation = _ACTION_INVOCATION.get()
        return invocation[0] if invocation is not None else None

    def _deliver_stream_events(
        self,
        *,
        result: MainAgentTurnResult,
        turn_id: str,
        conversation_id: str,
    ) -> None:
        for tool_result in result.tool_results:
            if not isinstance(tool_result, ToolObservation):
                continue
            action = tool_result.payload.get("client_action")
            if not isinstance(action, dict) or action.get("type") != "open_url":
                continue
            intent_id = action.get("capture_intent_id")
            expires_at = action.get("capture_intent_expires_at")
            self._emit(
                ClientActionEvent(
                    action="open_url",
                    url=str(action.get("url", "")),
                    label=str(action.get("label", "打开岗位搜索页")),
                    capture_intent_id=str(intent_id) if intent_id else None,
                    capture_intent_expires_at=(
                        datetime.fromisoformat(str(expires_at)) if expires_at else None
                    ),
                )
            )

        interaction = self._interaction_event(
            result=result,
            conversation_id=conversation_id,
        )
        if interaction is not None:
            self._emit(interaction)
            self._emit(
                TurnSuspendedEvent(
                    turn_id=turn_id,
                    interaction_id=interaction.interaction_id,
                )
            )
            return

        for artifact in result.artifacts:
            reference = artifact.reference
            self._emit(
                ArtifactReadyEvent(
                    artifact_id=reference.id,
                    filename=reference.filename,
                    media_type=reference.media_type,
                    byte_size=reference.byte_size,
                )
            )
        # The card is the only full-report delivery path both live and after a
        # reload. The accompanying message is the same short prose in both
        # cases; no full report is duplicated into content_delta.
        #
        # One card per report, not one per turn. Four card-backed reads fit
        # inside the read budget, so "show me the research and the match" ends a
        # turn holding two stored reports; emitting only the last one would
        # leave a durable report the reader is never handed.
        for reference in self._turn_resource_refs(result.tool_results):
            self._emit(MainAgentRuntime._resource_ready_event(reference))
        self._emit(TurnCompletedEvent(turn_id=turn_id))

    @staticmethod
    def _resource_ready_event(
        reference: ConversationResourceReference,
    ) -> ReportReadyEvent | JobResourceReadyEvent:
        if reference.kind == "saved_job":
            if reference.job_posting_id is None:
                raise ValueError("saved_job references name their posting")
            return JobResourceReadyEvent(
                resource_id=reference.resource_id,
                job_posting_id=reference.job_posting_id,
                title=reference.title,
                description=reference.description,
            )
        if reference.kind == "resume_version":
            raise ValueError("resume_version references ride on user messages")
        return ReportReadyEvent(
            kind=reference.kind,
            resource_id=reference.resource_id,
            status_at_delivery=reference.status_at_delivery,
            anchored_by_other_job=reference.anchored_by_other_job,
        )

    def _deliver_reply(
        self,
        *,
        result: MainAgentTurnResult,
        conversation_id: str,
    ) -> None:
        """Stream the reply text. Runs before the commit; see ``_before_commit``.

        Everything it sends is a function of the finished result alone, and
        nothing the commit does changes the text, so sending it first cannot
        make the live message differ from the stored one.
        """
        interaction = self._interaction_event(
            result=result,
            conversation_id=conversation_id,
        )
        if interaction is not None:
            # Resume analysis is different from ordinary questions: the user
            # must see the complete proposed evidence before the bound buttons
            # can carry meaningful consent. The card prompt is only the gate,
            # not a replacement for the analysis body.
            if interaction.scope == "resume_analysis_confirmation":
                self._emit(
                    ProgressEvent(stage="presenting", message="正在展示分析结果……")
                )
                for delta in iter_content_deltas(result.assistant_message):
                    self._emit(ContentDeltaEvent(delta=delta, delivery="synthetic"))
            return

        if not result.content_streamed:
            self._emit(
                ProgressEvent(stage="presenting", message="正在整理交付内容……")
            )
            # Compressed only when a card will render the body. For every
            # other state the message *is* the delivery, so shortening it here
            # would lose the content outright rather than move it — which is
            # what happened to the single-question mock interview readback: four
            # thousand characters the candidate had just asked for, replaced by
            # a one-line receipt with nowhere to read the rest.
            #
            # Where a card does exist this asks the same function the commit
            # asks, so the live message and the stored one stay identical.
            streamed_message = (
                self._conversation_content(
                    result.tool_result,
                    screen=self._durable_screen(result),
                    composed=bool(result.model_message),
                )
                if self._turn_is_card_backed(result.tool_results)
                else result.assistant_message
            )
            for delta in iter_content_deltas(streamed_message):
                self._emit(ContentDeltaEvent(delta=delta, delivery="synthetic"))

    @staticmethod
    def _interaction_event(
        *,
        result: MainAgentTurnResult,
        conversation_id: str,
    ) -> InteractionRequiredEvent | None:
        tool_result = result.tool_result
        prompt = result.assistant_message
        task = result.context.task
        stable_parts = (
            conversation_id,
            task.active_workflow,
            task.run_id or "",
            task.phase or "",
            tool_result.state if tool_result is not None else result.origin.label,
        )

        if tool_result is not None:
            if tool_result.state == "capability_confirmation_required":
                # Built from the sealed row's id, not from anything about this
                # turn, so a reload rebuilds the identical interaction — see the
                # same call in ``api/reads.py``.
                return capability_confirmation_event(
                    conversation_id=conversation_id,
                    confirmation_id=tool_result.payload["confirmation_id"],
                    prompt=prompt,
                )
            if (
                tool_result.state == "resume_analysis_ready"
                and task.resume_analysis_status == "pending"
                and task.active_resume_analysis_id is not None
            ):
                return resume_analysis_confirmation_event(
                    conversation_id=conversation_id,
                    analysis_id=task.active_resume_analysis_id,
                )
            if tool_result.state == "calendar_approval_required":
                return InteractionRequiredEvent(
                    interaction_id=interaction_id(*stable_parts),
                    kind="approval",
                    prompt=prompt,
                    options=(
                        InteractionOption(value="confirm", label="确认执行"),
                        InteractionOption(value="cancel", label="暂不执行"),
                    ),
                )
            if tool_result.state == "email_events_pending":
                return InteractionRequiredEvent(
                    interaction_id=interaction_id(*stable_parts),
                    kind="confirmation",
                    prompt=prompt,
                    options=(
                        InteractionOption(value="review", label="查看并确认"),
                        InteractionOption(value="later", label="稍后处理"),
                    ),
                    allow_free_text=True,
                )
            if tool_result.state in {
                "constraint_retirement_proposed",
                "memory_amendment_proposed",
                "memory_tombstone_proposed",
                "free_text_preference_confirmation_proposed",
                "free_text_preference_confirmed_structured_proposed",
                "career_fact_proposed",
                "mock_interview_answer_required",
                "mock_interview_running",
                "resume_tailoring_review_blocked",
                "resume_final_review_blocked",
                "resume_tailoring_superseded",
            }:
                return InteractionRequiredEvent(
                    interaction_id=interaction_id(*stable_parts),
                    kind="free_text",
                    prompt=prompt,
                    allow_free_text=True,
                )
        decision = result.model_decision
        if decision is not None and decision.action == "ask_user":
            # Options are grounded only in the tool result that loaded them.
            # Prompt wording is model output, not a trustworthy data-source
            # discriminator and must never change the model's decision.
            options = MainAgentRuntime._selection_options(tool_result, task)
            if options:
                return InteractionRequiredEvent(
                    interaction_id=interaction_id(*stable_parts, prompt),
                    kind="single_selection",
                    prompt=prompt,
                    options=options,
                    allow_free_text=True,
                )
            return InteractionRequiredEvent(
                interaction_id=interaction_id(*stable_parts, prompt),
                kind="free_text",
                prompt=prompt,
                allow_free_text=True,
            )
        return None

    def _run_interaction_response(
        self,
        *,
        context: MainAgentContext,
        conversation_id: str,
        response: InteractionResponse,
    ) -> MainAgentTurnResult:
        """Resolve a capability-owned UI decision before the LLM sees it."""

        if response.scope == "capability_confirmation":
            return self._run_owner_confirmation(
                context=context, conversation_id=conversation_id, response=response
            )
        task = context.task
        analysis_id = task.active_resume_analysis_id
        expected_id = (
            interaction_id(
                conversation_id,
                "resume_analysis_confirmation",
                analysis_id,
            )
            if analysis_id is not None
            else None
        )
        if (
            response.scope != "resume_analysis_confirmation"
            or analysis_id is None
            or task.resume_analysis_status != "pending"
            or response.interaction_id != expected_id
        ):
            result = ToolObservation(
                tool_name="resume_analysis_confirmation",
                state="resume_analysis_decision_expired",
                message="这项确认已过期或已处理，请重新打开当前简历分析。",
            )
            updated = context
        else:
            result = self._tools.resolve_resume_analysis_confirmation(
                user_id=context.profile.user_id,
                analysis_id=analysis_id,
                action=response.action,
            )
            updated = (
                context.model_copy(
                    update={
                        "task": reduce_task_state(
                            task, result, now=self._context_manager.now()
                        )
                    }
                )
                if result.state
                in {"resume_analysis_confirmed", "resume_analysis_rejected"}
                else context
            )
        return MainAgentTurnResult(
            # The user clicked an approval whose contract was already sealed.
            # This used to fabricate an ``AgentDecision(action="final")`` so the
            # result could be typed as a model decision; nothing read its
            # message, and everything that read its ``action`` read a fiction.
            origin=InteractionReceipt(scope=response.scope, action=response.action),
            context=updated,
            assistant_message=self._assistant_message(result),
            tool_result=result,
            tool_results=(result,),
        )

    def _run_owner_confirmation(
        self,
        *,
        context: MainAgentContext,
        conversation_id: str,
        response: InteractionResponse,
    ) -> MainAgentTurnResult:
        """Run the action the owner stopped, once, without re-asking the model.

        The model is not consulted here on purpose. It already proposed this
        action; the owner rule stopped it and the owner has now answered. Asking
        the model again would let it revise or abandon an action a person
        explicitly approved, and would make "confirm" mean "reconsider".

        Consuming the seal is what makes it exactly once: the PENDING → CONFIRMED
        transition is conditional in SQL, so a resent click, a duplicated
        request or a second tab loses the race and is told the action is
        already settled rather than running it twice.
        """

        store = self._capability_confirmation_store
        user_id = context.profile.user_id
        pending = (
            store.active_for_conversation(
                user_id=user_id, conversation_id=conversation_id
            )
            if store is not None
            else ()
        )
        confirmation = next(
            (
                candidate
                for candidate in pending
                if interaction_id(
                    conversation_id,
                    "capability_confirmation",
                    candidate.confirmation_id,
                )
                == response.interaction_id
            ),
            None,
        )
        if confirmation is None:
            return self._settled_confirmation_turn(
                context,
                "这项确认已过期或已处理，请重新提出这个操作。",
                state="capability_confirmation_expired",
            )
        if response.action == "cancel":
            cancelled = store.cancel(
                confirmation_id=confirmation.confirmation_id, user_id=user_id
            )
            if not cancelled:
                return self._settled_confirmation_turn(
                    context,
                    "这项操作已经开始执行，取消没有改写它的状态；请等待结果或进行对账。",
                    state="capability_confirmation_in_progress",
                    action="cancel",
                )
            return self._settled_confirmation_turn(
                context,
                "已按你的选择取消，没有执行这个操作。",
                state="capability_confirmation_cancelled",
                action="cancel",
            )
        if (
            confirmation.status == "PENDING"
            and
            confirmation.policy_revision
            != context.preferences.behavior_policy.revision
        ):
            store.cancel(
                confirmation_id=confirmation.confirmation_id, user_id=user_id
            )
            return self._settled_confirmation_turn(
                context,
                "你的行为规则在这项确认发出后已经变化；旧批准未执行，请重新提出操作。",
                state="capability_confirmation_expired",
            )
        try:
            sealed = store.claim(
                confirmation_id=confirmation.confirmation_id, user_id=user_id
            )
        except CapabilityConfirmationExpiredError:
            return self._settled_confirmation_turn(
                context,
                "这项确认已过期，没有执行。请重新提出这个操作。",
                state="capability_confirmation_expired",
            )
        except CapabilityConfirmationSettledError:
            return self._settled_confirmation_turn(
                context,
                "这项确认已经处理过，没有重复执行。",
                state="capability_confirmation_expired",
            )
        except CapabilityConfirmationInProgressError:
            return self._settled_confirmation_turn(
                context,
                "这项操作正在执行，没有重复启动。",
                state="capability_confirmation_in_progress",
            )
        # Re-derived rather than trusted: the arguments are read back from the
        # sealed row, and the hash they were sealed under has to still describe
        # them. A row edited underneath us is a refusal, not an execution.
        if arguments_hash(sealed.arguments) != sealed.arguments_hash:
            return self._settled_confirmation_turn(
                context,
                "这项确认的内容已经无法核对，没有执行。",
                state="capability_confirmation_expired",
            )
        invocation = _ACTION_INVOCATION.get()
        if invocation is None:
            raise RuntimeError("confirmation execution context is unavailable")
        # The durable confirmation, not the HTTP retry that delivered the
        # click, is the stable identity of the approved action. After a crash a
        # reclaimed lease therefore reaches the same action-ledger slot.
        action_token = _ACTION_INVOCATION.set(
            (invocation[0], f"confirmation:{sealed.confirmation_id}")
        )
        try:
            state = self._graph.invoke({
                "context": context,
                "career_memory_scope_keys": self._free_text_preference_scope_keys(
                    context
                ),
                "decision": AgentDecision(
                    action="tool_call",
                    tool_call=ToolCall(name=sealed.capability, arguments={}),
                ),
                "pending": {
                    "name": sealed.capability,
                    "arguments": sealed.arguments,
                    "owner_confirmed": True,
                    "confirmation_id": sealed.confirmation_id,
                },
                "artifact_ids": (),
                "tool_results": (),
                "control": {
                    "read_calls": 0,
                    "write_calls": 0,
                    "projection_refusals": 0,
                    "authorization_refusals": 0,
                    "fingerprints": (),
                    "retryable_fingerprints": (),
                    "retry_counts": {},
                },
            })
        except Exception:
            store.settle(
                confirmation_id=sealed.confirmation_id,
                user_id=user_id,
                status="RECONCILIATION_REQUIRED",
            )
            raise
        finally:
            _ACTION_INVOCATION.reset(action_token)
        last = self._last_result(state)
        confirmation_status = (
            "RECONCILIATION_REQUIRED"
            if last.state == "action_reconciliation_required"
            or last.execution_outcome == "unknown"
            else "FAILED"
            if last.execution_outcome == "not_committed"
            else "EXECUTED"
        )
        store.settle(
            confirmation_id=sealed.confirmation_id,
            user_id=user_id,
            status=confirmation_status,
        )
        control = state.get("control", {})
        return MainAgentTurnResult(
            origin=InteractionReceipt(
                scope="capability_confirmation", action=response.action
            ),
            context=state["context"],
            assistant_message=state["assistant_message"],
            tool_result=self._last_result(state),
            tool_results=state.get("tool_results", ()),
            delegated_read_count=control.get("read_calls", 0),
            delegated_write_count=control.get("write_calls", 0),
            career_memory_scope_keys=state.get("career_memory_scope_keys", ()),
        )

    def _settled_confirmation_turn(
        self,
        context: MainAgentContext,
        message: str,
        *,
        state: str,
        action: str = "confirm",
    ) -> MainAgentTurnResult:
        """A confirmation that resolved without running anything."""

        result = ToolObservation(
            tool_name="capability_confirmation", state=state, message=message
        )
        return MainAgentTurnResult(
            origin=InteractionReceipt(
                scope="capability_confirmation", action=action
            ),
            context=context,
            assistant_message=message,
            tool_result=result,
            tool_results=(result,),
        )

    @staticmethod
    def _selection_options(
        tool_result: MainAgentToolOutput | None,
        task: ConversationTaskState,
    ) -> tuple[InteractionOption, ...]:
        if not isinstance(tool_result, ToolObservation):
            return ()
        name = tool_result.tool_name
        if name == "find_saved_jobs":
            return tuple(
                InteractionOption(
                    selection_index=index,
                    label=f"{item.title}｜{item.company_name}",
                    description="，".join(
                        value for value in (item.city, item.salary) if value
                    )
                    or None,
                )
                for index, item in enumerate(task.saved_job_candidates, start=1)
            )
        if name == "list_target_roles":
            return tuple(
                InteractionOption(
                    selection_index=index,
                    label=item.title,
                    description=f"优先级 {item.priority}，状态 {item.status}",
                )
                for index, item in enumerate(task.target_role_candidates, start=1)
            )
        if name == "list_resumes":
            return tuple(
                InteractionOption(
                    selection_index=index,
                    label=item.name,
                    description=f"状态：{item.status}",
                )
                for index, item in enumerate(task.resume_candidates, start=1)
            )
        if name == "get_resume_metadata":
            return tuple(
                InteractionOption(
                    selection_index=index,
                    label=f"版本 {item.version_number}",
                    description=(
                        f"{item.document_format}，{item.source_type}，"
                        f"{item.byte_size} bytes"
                    ),
                )
                for index, item in enumerate(
                    task.resume_version_candidates, start=1
                )
            )
        if name == "list_applications":
            return tuple(
                InteractionOption(
                    selection_index=index,
                    label=f"{item.title}｜{item.company_name}",
                    description=f"状态：{item.status}",
                )
                for index, item in enumerate(task.application_candidates, start=1)
            )
        if name == "list_interviews":
            return tuple(
                InteractionOption(
                    selection_index=index,
                    label=(
                        f"{item.employer_label or '面试'}｜第 {item.sequence_number} 轮"
                    ),
                    description="，".join(
                        value
                        for value in (
                            f"状态 {item.status}",
                            (
                                item.scheduled_start.isoformat()
                                if item.scheduled_start is not None
                                else None
                            ),
                        )
                        if value
                    ),
                )
                for index, item in enumerate(task.interview_candidates, start=1)
            )
        if name == "list_action_items":
            return tuple(
                InteractionOption(
                    selection_index=index,
                    label=item.title,
                    description=f"{item.action_type}，状态 {item.status}",
                )
                for index, item in enumerate(task.action_candidates, start=1)
            )
        if name == "list_calendar_accounts":
            return tuple(
                InteractionOption(
                    selection_index=index,
                    label=item.email_address,
                    description=f"{item.provider} Calendar",
                )
                for index, item in enumerate(
                    task.calendar_account_candidates, start=1
                )
            )
        if name == "list_email_events":
            return tuple(
                InteractionOption(
                    selection_index=index,
                    label=item.summary,
                    description=f"{item.event_type}，状态 {item.status}",
                )
                for index, item in enumerate(task.email_event_candidates, start=1)
            )
        return ()

    @staticmethod
    def _owns_next_turn(task: ConversationTaskState) -> bool:
        """Whether the mock interview will consume the next user message.

        Holding the workflow slot is not enough. These two phases keep it only
        to record why the run cannot continue; the run itself is unreachable, so
        the next message has to reach the decision model or the conversation
        would have no way out.
        """
        return task.active_workflow == "mock_interview" and task.phase not in {
            "mock_interview_checkpoint_missing",
            "mock_interview_graph_incompatible",
        }

    def _run_loaded_context(
        self,
        context: MainAgentContext,
        *,
        bare_confirmation_target: Literal[
            "career_fact",
            "job_intent",
            "free_text_preference",
        ] | None = None,
    ) -> MainAgentTurnResult:
        if (
            bare_confirmation_target == "career_fact"
            and context.task.pending_career_fact is not None
            and is_explicit_confirmation(context.user_message)
        ):
            return self._run_runtime_policy_tool(
                context,
                policy="career_fact_confirmation",
                tool_name="confirm_career_fact",
                arguments={},
            )
        if (
            bare_confirmation_target == "job_intent"
            and context.task.pending_job_intent_update is not None
            and is_explicit_confirmation(context.user_message)
        ):
            return self._run_runtime_policy_tool(
                context,
                policy="job_intent_confirmation",
                tool_name="confirm_job_intent",
                arguments={},
            )
        if (
            bare_confirmation_target == "free_text_preference"
            and context.task.pending_free_text_preference is not None
            and not (
                context.task.pending_free_text_preference
                .needs_scope_clarification
            )
            and is_explicit_confirmation(context.user_message)
        ):
            return self._run_runtime_policy_tool(
                context,
                policy="free_text_preference_activation",
                tool_name="confirm_free_text_preference",
                arguments={},
            )
        if (
            context.task.pending_free_text_preference is None
            and any(
                item.status == "quarantined"
                for item in context.free_text_preferences
            )
        ):
            return self._run_free_text_preference_confirmation(context)
        state = self._graph.invoke(
            {
                "context": context,
                "career_memory_scope_keys": self._free_text_preference_scope_keys(
                    context
                ),
                **self._explicit_span_prelude(context),
                "artifact_ids": (),
                "tool_results": (),
                "control": {
                    "read_calls": 0,
                    "write_calls": 0,
                    "projection_refusals": 0,
                    "authorization_refusals": 0,
                    "fingerprints": (),
                    "retryable_fingerprints": (),
                    "retry_counts": {},
                },
            }
        )
        tool_result = self._last_result(state)
        artifacts = tuple(
            self._tools.deliver_resume_artifact(
                user_id=context.profile.user_id,
                artifact_id=artifact_id,
            )
            for artifact_id in state.get("artifact_ids", ())
        )
        control = state.get("control", {})
        return MainAgentTurnResult(
            origin=ModelDecision(state["decision"]),
            context=state["context"],
            assistant_message=state["assistant_message"],
            tool_result=tool_result,
            tool_results=state.get("tool_results", ()),
            artifacts=artifacts,
            content_streamed=False,
            model_message=state.get("model_message", ""),
            delegated_read_count=control.get("read_calls", 0),
            delegated_write_count=control.get("write_calls", 0),
            career_memory_scope_keys=state.get("career_memory_scope_keys", ()),
        )

    def _explicit_span_prelude(self, context: MainAgentContext) -> MainAgentState:
        """Open the turn with the exact history read the user already specified.

        A message that names a sequence span is a complete
        ``read_conversation_span`` call; asking the model to make it let it ask
        the user to repeat the range instead. The runtime makes the call under
        the same rule the model is given — both watermarks present — and the
        model then decides with the span's observation in front of it. It is
        never a guess: an unclear or invalid range stays with the model.
        """

        if context.through_sequence < 1 or context.recent_from_sequence is None:
            return {}
        span = explicit_sequence_span(context.user_message)
        if span is None or not self._offers_tool("read_conversation_span"):
            return {}
        arguments = {
            "from_sequence": span.from_sequence,
            "through_sequence": span.through_sequence,
        }
        return {
            "decision": AgentDecision(
                action="tool_call",
                tool_call=ToolCall(name="read_conversation_span", arguments=arguments),
            ),
            "pending": {
                "name": "read_conversation_span",
                "policy_owned": True,
                "policy_prelude": True,
                "arguments": arguments,
            },
        }

    def _offers_tool(self, name: str) -> bool:
        """Whether ``name`` is among the tools the model is offered this turn."""

        schemas = getattr(self, "_decision_tool_schemas", None)
        if schemas is None:
            schemas = self._tools.schemas()
        return any(
            schema.get("function", {}).get("name") == name for schema in schemas
        )

    def _run_free_text_preference_confirmation(
        self,
        context: MainAgentContext,
    ) -> MainAgentTurnResult:
        """Deterministically surface the first relevant quarantined preference."""

        return self._run_runtime_policy_tool(
            context,
            policy="free_text_preference_confirmation",
            tool_name="propose_free_text_preference_confirmation",
            arguments={"selection_index": 1},
        )

    def _run_runtime_policy_tool(
        self,
        context: MainAgentContext,
        *,
        policy: Literal[
            "free_text_preference_confirmation",
            "free_text_preference_activation",
            "career_fact_confirmation",
            "job_intent_confirmation",
        ],
        tool_name: str,
        arguments: dict[str, Any],
    ) -> MainAgentTurnResult:
        decision = AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name=tool_name, arguments=arguments),
        )
        state = self._graph.invoke(
            {
                "context": context,
                "career_memory_scope_keys": self._free_text_preference_scope_keys(
                    context
                ),
                "decision": decision,
                "pending": {
                    "name": tool_name,
                    "policy_owned": True,
                    "arguments": arguments,
                },
                "artifact_ids": (),
                "tool_results": (),
                "control": {
                    "read_calls": 0,
                    "write_calls": 0,
                    "projection_refusals": 0,
                    "authorization_refusals": 0,
                    "fingerprints": (),
                    "retryable_fingerprints": (),
                    "retry_counts": {},
                },
            }
        )
        result = self._last_result(state)
        if result is None:
            raise RuntimeError(f"{policy} policy produced no result")
        control = state.get("control", {})
        return MainAgentTurnResult(
            origin=RuntimePolicyAction(policy=policy),
            context=state["context"],
            assistant_message=state["assistant_message"],
            tool_result=result,
            tool_results=state.get("tool_results", ()),
            delegated_read_count=control.get("read_calls", 0),
            delegated_write_count=control.get("write_calls", 0),
            career_memory_scope_keys=state.get("career_memory_scope_keys", ()),
        )

    DECISION_HEARTBEAT_SECONDS: ClassVar[float] = 15.0

    _DECISION_RETRY_REASONS: ClassVar[dict[str, str]] = {
        "MAIN_AGENT_TRANSPORT_ERROR": "上一次请求超时或连接中断",
        "MAIN_AGENT_RATE_LIMITED": "上一次请求被限流",
    }

    @classmethod
    def _decision_attempt_message(cls, attempt: DecisionAttempt) -> str | None:
        if attempt.attempt <= 1:
            return None
        code = attempt.previous_error_code or ""
        reason = cls._DECISION_RETRY_REASONS.get(
            code,
            "上一次请求被模型服务拒绝" if code.startswith("MAIN_AGENT_REJECTED_") else "上一次请求失败",
        )
        return (
            f"{reason}，正在重新判断（第 {attempt.attempt}/{attempt.max_attempts} 次，"
            f"已等待 {int(attempt.elapsed_seconds)} 秒）……"
        )

    _COMPACTION_MESSAGES: ClassVar[dict[str, tuple[str, str]]] = {
        "load": ("loading_context", "正在压缩较早的对话记录，稍后开始判断……"),
        "commit": ("saving", "正在压缩较早的对话记录，回复已送达，可以先看……"),
    }

    def _announce_compaction(self, phase: str) -> None:
        stage, message = self._COMPACTION_MESSAGES[phase]
        self._emit(ProgressEvent(stage=stage, message=message))

    def _heartbeat(
        self,
        sink: StreamEventSink | None,
        *,
        stage: str,
        describe: Callable[[int], str],
    ) -> Event:
        """Keep a blocking wait visible on the stream.

        A model request or a worker call blocks this thread, so nothing on it
        can speak until the callee returns; a helper thread posts the elapsed
        seconds to the captured sink instead. Presentation only: it never
        touches graph state, and the returned event stops it.
        """
        stop = Event()
        if sink is None:
            return stop
        interval = self.DECISION_HEARTBEAT_SECONDS
        started = perf_counter()

        def beat() -> None:
            while not stop.wait(interval):
                waited = int(perf_counter() - started)
                try:
                    sink(ProgressEvent(stage=stage, message=describe(waited)))
                except Exception:
                    return

        Thread(target=beat, name=f"{stage}-heartbeat", daemon=True).start()
        return stop

    def _decision_heartbeat(self, sink: StreamEventSink | None) -> Event:
        return self._heartbeat(
            sink,
            stage="deciding",
            describe=lambda waited: f"仍在等待模型判断（已等待 {waited} 秒）……",
        )

    _CAPABILITY_STEP_MESSAGES: ClassVar[dict[str, str]] = {
        "resume_analysis": "正在分析简历内容",
        "resume_job_match": "正在比对简历与岗位要求",
        "resume_job_match_state_audit": "正在核对简历比对结果",
        "resume_tailoring": "正在起草定制简历",
        "resume_draft_review": "正在审校简历草稿",
        "resume_finalization": "正在定稿简历",
        "resume_final_review": "正在审校定稿简历",
        "job_research": "正在调研岗位背景",
        "interview_preparation": "正在准备面试资料",
        "mock_interview_plan": "正在规划模拟面试",
        "mock_interview_input_route": "正在理解你的回答",
        "mock_interview_ask": "正在生成面试问题",
        "mock_interview_evaluate": "正在点评你的回答",
        "mock_interview_report": "正在整理面试报告",
        "email_tracking_assess": "正在识别招聘邮件",
        "email_sync.fetch": "正在读取邮箱",
        "email_sync.scan": "正在扫描邮件",
    }

    _CAPABILITY_TOOL_MESSAGES: ClassVar[dict[str, str]] = {
        "read_file": "正在阅读工作指南",
        "ls": "正在查找工作指南",
        "glob": "正在查找工作指南",
        "grep": "正在检索工作指南",
    }

    @classmethod
    def _capability_step_label(cls, step: CapabilityStep) -> str | None:
        """The user-facing name of an internal step, or nothing.

        Unknown labels stay silent rather than leaking internal names.
        """
        if step.kind == "tool":
            _, _, tool = step.stage.rpartition(".")
            return cls._CAPABILITY_TOOL_MESSAGES.get(tool)
        return cls._CAPABILITY_STEP_MESSAGES.get(step.stage)

    @staticmethod
    def _capability_step_message(label: str, step: CapabilityStep) -> str:
        if step.kind == "retry":
            return f"{label}时请求失败，正在重试……"
        if step.index is not None and step.total is not None:
            return f"{label}（第 {step.index}/{step.total} 项）……"
        if step.index is not None and step.index > 1:
            return f"{label}（第 {step.index} 次调用模型）……"
        return f"{label}……"

    def _run_capability(
        self,
        pending: PendingAction,
        run: Callable[[], MainAgentToolOutput],
    ) -> MainAgentToolOutput:
        """Execute one tool call while relaying its internal steps."""
        capability = self._public_capability(pending["name"])
        latest = self._CAPABILITY_LABELS[capability].rstrip("…")

        def on_step(step: CapabilityStep) -> None:
            nonlocal latest
            label = self._capability_step_label(step)
            if label is None:
                return
            latest = label
            self._emit(
                ProgressEvent(
                    stage="running_capability",
                    message=self._capability_step_message(label, step),
                )
            )

        heartbeat = self._heartbeat(
            _STREAM_SINK.get(),
            stage="running_capability",
            describe=lambda waited: f"{latest}（已等待 {waited} 秒）……",
        )
        try:
            with observing_capability_steps(on_step):
                return run()
        finally:
            heartbeat.set()

    def _decide(self, state: MainAgentState) -> MainAgentState:
        self._emit(ProgressEvent(stage="deciding", message="正在判断下一步操作……"))
        context = state["context"]
        control = dict(self._control(state))
        # Exposure is stamped here, where the projected episodes are about to be
        # put in front of the model, and exactly once per turn. Doing it inside
        # the projection counted every context build — pressure measurement, the
        # load itself, the reload after a memory write — so a single turn scored
        # three to five accesses and decay had nothing but bookkeeping to read.
        # Turns a workflow owns never reach here and correctly stamp nothing.
        if not control.get("episodes_marked"):
            self._context_manager.mark_episodes_projected(
                user_id=context.profile.user_id,
                context=context,
            )
            control["episodes_marked"] = True
        schemas = getattr(self, "_decision_tool_schemas", None)
        if schemas is None:
            schemas = self._tools.schemas()
        context_chars = decision_context_chars(context)
        tool_schema_chars = getattr(self, "_decision_tool_schema_chars", None)
        if tool_schema_chars is None:
            tool_schema_chars = len(
                json.dumps(schemas, ensure_ascii=False, sort_keys=True)
            )
        details = {
            "conversation_id": context.conversation_id,
            "conversation_key": conversation_trace_key(
                context.profile.user_id,
                context.conversation_id,
            ),
            "context_chars": context_chars,
            # Observation cost stays separately visible even though its
            # readable fields now use a low-authority turn-results message.
            "observation_chars": decision_observation_chars(
                context.tool_observations
            ),
            "observation_count": len(context.tool_observations),
            "offered_tool_count": len(schemas),
            "tool_schema_chars": tool_schema_chars,
        }
        cache_configuration = getattr(
            self._decision_maker, "cache_configuration", None
        )
        if callable(cache_configuration):
            details.update(cache_configuration())
        started = perf_counter()
        self._record_trace_event(
            "model_attempt",
            "main_agent_decide",
            outcome="started",
            details=details,
            model_call_category="orchestrator_decision",
        )
        def on_attempt(attempt: DecisionAttempt) -> None:
            message = self._decision_attempt_message(attempt)
            if message is not None:
                self._emit(ProgressEvent(stage="deciding", message=message))

        heartbeat = self._decision_heartbeat(_STREAM_SINK.get())
        try:
            with observing_decision_attempts(on_attempt):
                decision = self._decision_maker.decide(context, schemas)
        except Exception as error:
            self._record_trace_event(
                "memory_context_observed",
                "main_agent_decide",
                outcome="succeeded",
                details=memory_context_observation(
                    context,
                    career_memory_enabled=(
                        self._career_context_projector is not None
                    ),
                    working_notes_only_tokens=0,
                    working_notes_only_argument=False,
                ),
            )
            failure_details = dict(details)
            consume_cache_metrics = getattr(
                self._decision_maker, "consume_cache_metrics", None
            )
            if callable(consume_cache_metrics):
                failure_details.update(consume_cache_metrics())
            self._record_trace_event(
                "model_failed",
                "main_agent_decide",
                outcome="failed",
                duration_ms=int((perf_counter() - started) * 1000),
                details=failure_details,
                error_code=getattr(error, "code", "ORCHESTRATOR_DECISION_FAILED"),
                error_detail=getattr(error, "detail", None) or type(error).__name__,
                recoverable=getattr(error, "retryable", None),
                model_call_category="orchestrator_decision",
            )
            raise
        finally:
            heartbeat.set()
        note_only_tokens = self._decision_note_only_tokens(context, decision)
        self._record_trace_event(
            "memory_context_observed",
            "main_agent_decide",
            outcome="succeeded",
            details=memory_context_observation(
                context,
                career_memory_enabled=self._career_context_projector is not None,
                working_notes_only_tokens=len(note_only_tokens),
                working_notes_only_argument=bool(note_only_tokens),
            ),
        )
        decision_details = {**details, "decision_action": decision.action}
        consume_cache_metrics = getattr(
            self._decision_maker, "consume_cache_metrics", None
        )
        if callable(consume_cache_metrics):
            decision_details.update(consume_cache_metrics())
        if decision.tool_call is not None:
            decision_details.update(
                {
                    "tool_name": decision.tool_call.name,
                    # Arguments can contain user text and opaque identifiers.
                    # Persist only a stable digest; that is sufficient to spot
                    # the same call being derived again after compaction.
                    "tool_arguments_fingerprint": hashlib.sha256(
                        self._tool_call_fingerprint(decision).encode("utf-8")
                    ).hexdigest(),
                }
            )
        self._record_trace_event(
            "model_succeeded",
            "main_agent_decide",
            outcome="succeeded",
            duration_ms=int((perf_counter() - started) * 1000),
            details=decision_details,
            model_call_category="orchestrator_decision",
        )
        return {"decision": decision, "control": control}

    def _decision_note_only_tokens(
        self, context: MainAgentContext, decision: AgentDecision
    ) -> tuple[str, ...]:
        """Mirror the authorize-time notes guard for decision telemetry.

        The guard judges projected arguments, so telemetry does too; a call
        whose projection fails never reaches the guard and counts as no hit.
        An earlier gate (owner deny, budget, duplicate call) can still refuse
        first, so this measures note-derived arguments the model produced.
        """

        call = decision.tool_call
        if call is None or not is_notes_guarded(call.name):
            return ()
        try:
            arguments = (
                self._project_atomic_tool_arguments(
                    context, call.name, call.arguments
                )
                if self._tools.capability_kind(call.name) == "atomic_tool"
                else self._project_workflow_arguments(
                    context, call.name, call.arguments
                )
            )
        except ValueError:
            return ()
        return working_notes_only_tokens(arguments=arguments, context=context)

    def _run_owned_workflow_turn(
        self, *, context: MainAgentContext, user_message: str
    ) -> MainAgentTurnResult:
        """Advance an isolated workflow through the ordinary execution graph.

        Workflow ownership already determines the operation, so no model call is
        needed. The action still enters at ``authorize`` and continues through
        ``act`` and ``observe``: budgets, effect bookkeeping, task reduction,
        progress events and telemetry are therefore the same machinery used by
        model-selected tools. The raw answer stays in the private pending input
        and is never copied into a model-visible observation.
        """

        session_id = context.task.run_id
        if session_id is None:
            raise ValueError("Active mock interview has no resumable session")
        entry = (
            "retry_mock_interview"
            if context.task.phase == "failed"
            else "handle_mock_interview_input"
        )
        # Read before the graph runs: the reducer may close the workflow this
        # very turn (a cancellation sets it to "none"), and the origin has to
        # name the workflow that owned the input, not the state it left behind.
        owned_workflow = context.task.active_workflow
        if owned_workflow == "none":
            raise ValueError("owned workflow turn requires an active workflow")
        decision = AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name=entry, arguments={}),
        )
        runtime_arguments = (
            {}
            if entry == "retry_mock_interview"
            else {"message": user_message}
        )
        state = self._graph.invoke(
            {
                "context": context,
                "career_memory_scope_keys": self._free_text_preference_scope_keys(
                    context
                ),
                "decision": decision,
                # Runtime-owned workflow input is execution data, not model
                # context. It starts in pending and is replaced by projected
                # handler arguments during authorization.
                "pending": {
                    "name": entry,
                    "runtime_owned": True,
                    "arguments": runtime_arguments,
                },
                "artifact_ids": (),
                "tool_results": (),
                "control": {
                    "read_calls": 0,
                    "write_calls": 0,
                    "projection_refusals": 0,
                    "authorization_refusals": 0,
                    "fingerprints": (),
                    "retryable_fingerprints": (),
                    "retry_counts": {},
                },
            }
        )
        result = self._last_result(state)
        control = state.get("control", {})
        return MainAgentTurnResult(
            # The user supplied the input; the runtime decided it belongs to
            # the workflow it owns. The ``AgentDecision`` above is real graph
            # input — it routes ``authorize``/``act`` — but it is the runtime's
            # own construction, so it is not what this turn reports as its
            # origin.
            origin=RuntimeAction(workflow=owned_workflow),
            context=state["context"],
            assistant_message=state["assistant_message"],
            tool_result=result,
            tool_results=state.get("tool_results", ()),
            delegated_read_count=control.get("read_calls", 0),
            delegated_write_count=control.get("write_calls", 0),
            career_memory_scope_keys=state.get("career_memory_scope_keys", ()),
        )

    def _hydrate_career_context(self, state: MainAgentState) -> MainAgentState:
        context = state["context"]
        free_text_scope_keys = self._free_text_preference_scope_keys(context)
        # A prelude read that brought the turn here has been observed; the
        # model's first decision must not inherit its policy ownership.
        pending: PendingAction = {}
        if self._career_context_projector is None:
            return {
                "pending": pending,
                "career_memory_scope_keys": free_text_scope_keys,
            }
        memory = self._career_context_projector.project(
            user_id=context.profile.user_id,
            query=context.user_message,
        )
        return {
            "pending": pending,
            "context": context.model_copy(update={"career_memory": memory}),
            "career_memory_scope_keys": tuple(
                dict.fromkeys(
                    (
                        *(binding.entry_id for binding in memory.telemetry_bindings),
                        *free_text_scope_keys,
                    )
                )
            ),
        }

    @staticmethod
    def _free_text_preference_scope_keys(
        context: MainAgentContext,
    ) -> tuple[str, ...]:
        """Preference tracks whose values were exposed in this turn's prompt."""

        return tuple(
            dict.fromkeys(
                intent_entry_id(item.scope_key, item.pref_scope)
                for item in context.free_text_preferences
            )
        )

    @staticmethod
    def _tool_call_fingerprint(decision: AgentDecision) -> str:
        if decision.tool_call is None:
            return ""
        return json.dumps(
            {"name": decision.tool_call.name, "arguments": decision.tool_call.arguments},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _route_decision(
        state: MainAgentState,
    ) -> Literal["authorize", "present", "interrupt"]:
        decision = state["decision"]
        if decision.action == "tool_call":
            if decision.tool_call is None:
                raise ValueError("tool_call action requires tool_call arguments")
            return "authorize"
        if decision.action == "ask_user":
            return "interrupt"
        return "present"

    @staticmethod
    def _control(state: MainAgentState) -> LoopControl:
        return state.get("control", {})

    @staticmethod
    def _last_result(state: MainAgentState) -> MainAgentToolOutput | None:
        results = state.get("tool_results", ())
        return results[-1] if results else None

    def _authorization_refusal(
        self,
        state: MainAgentState,
        *,
        name: str,
        reason: str,
        next_action: str,
    ) -> MainAgentState:
        control = self._control(state)
        if (
            control.get("authorization_refusals", 0)
            >= self._max_authorization_refusals
        ):
            return {"authorization_route": "present"}
        result = ToolObservation(
            tool_name=name,
            state="authorization_refused",
            message=reason,
            next_action=next_action,
        )
        return {
            "authorization_route": "observe",
            "pending": {
                "name": name,
                "result": result,
                "synthetic_kind": "authorization",
                "runtime_owned": bool(
                    state.get("pending", {}).get("runtime_owned")
                ),
            },
        }

    def _authorize(self, state: MainAgentState) -> MainAgentState:
        """Project and gate one action without choosing its successor.

        Most actions are model-selected. A workflow-owned turn supplies one
        bound runtime action instead; it receives the same budgets and effect
        checks but uses a separate projector so private workflow input never
        becomes model-authored arguments.
        """

        decision = state["decision"]
        if decision.tool_call is None:
            raise ValueError("tool_call action requires tool_call arguments")
        name = decision.tool_call.name
        runtime_owned = bool(state.get("pending", {}).get("runtime_owned"))
        owner_confirmed = bool(state.get("pending", {}).get("owner_confirmed"))
        policy_owned = bool(state.get("pending", {}).get("policy_owned"))
        policy_prelude = bool(state.get("pending", {}).get("policy_prelude"))
        if runtime_owned:
            if name not in self._tools.runtime_workflow_names:
                raise ValueError(f"Unknown runtime-owned workflow: {name}")
            kind = "workflow"
        else:
            kind = self._tools.capability_kind(name)
        effect = effect_for(name)
        # Owner rules are an authority beside budgets and reachability, and a
        # confirmed seal is the owner having already exercised it: re-judging
        # here would refuse the very action they just approved, which is how
        # ``review`` degenerates into ``deny``.
        verdict = (
            "permit"
            if runtime_owned or owner_confirmed
            else state["context"].preferences.capability_verdict(name)
        )
        if verdict == "deny":
            return self._authorization_refusal(
                state,
                name=name,
                reason="你设置的偏好不允许这个操作。",
                next_action="向用户说明这条设置，不要重试这次调用。",
            )
        # ``review`` is decided here but acted on after projection: what the
        # owner approves has to be the concrete action, arguments included, and
        # those do not exist yet.
        control = self._control(state)
        bucket, used, limit = self._budget_bucket(control, name=name, effect=effect)
        if used >= limit:
            return self._authorization_refusal(
                state,
                name=name,
                reason=(
                    f"本轮 {bucket} 委派预算已经用完；请基于已有结果作答，"
                    "或说明需要下一轮继续。"
                ),
                next_action=(
                    "本轮的委派预算已经用完。请基于已有结果作答，"
                    "或者告诉用户还缺什么。"
                ),
            )

        fingerprint = self._tool_call_fingerprint(decision)
        fingerprints = control.get("fingerprints", ())
        retry_counts = dict(control.get("retry_counts", {}))
        if fingerprint in fingerprints:
            retryable = fingerprint in control.get("retryable_fingerprints", ())
            retries = retry_counts.get(fingerprint, 0)
            if not retryable:
                return self._authorization_refusal(
                    state,
                    name=name,
                    reason="相同调用已经执行过，且上次结果没有声明为可重试。",
                    next_action=(
                        "这次调用和本轮之前那次完全一样，再调一次也不会有新结果。"
                        "请用已有的观察作答，或者换一组参数。"
                    ),
                )
            if retries >= self._max_failure_retries:
                return self._authorization_refusal(
                    state,
                    name=name,
                    reason="相同失败调用已经达到本轮重试上限。",
                    next_action=(
                        "同一个失败调用已经重试到本轮上限。别再重试；"
                        "把失败讲清楚，或者问用户要不要换个做法。"
                    ),
                )
            retry_counts[fingerprint] = retries + 1
            control = {**control, "retry_counts": retry_counts}
        try:
            arguments = (
                # Taken from the seal, not re-projected. These arguments were
                # projected once, hashed, and shown to the owner; re-deriving
                # them from a decision the model did not make this turn would
                # execute something other than what was approved.
                state["pending"]["arguments"]
                if owner_confirmed
                else self._project_runtime_workflow_arguments(state, name)
                if runtime_owned
                else self._project_atomic_tool_arguments(
                    state["context"],
                    name,
                    decision.tool_call.arguments,
                )
                if kind == "atomic_tool"
                else self._project_workflow_arguments(
                    state["context"],
                    name,
                    decision.tool_call.arguments,
                )
            )
            if owner_confirmed and name == "update_owner_settings":
                arguments = {
                    **arguments,
                    "confirmation_id": state["pending"]["confirmation_id"],
                }
        except ValueError as error:
            MainAgentRuntime._reraise_security_refusal(error)
            if (
                control.get("projection_refusals", 0)
                >= self._max_projection_refusals
            ):
                return {"authorization_route": "present"}
            result = MainAgentRuntime._rejection_observation(name, error)
            return {
                "authorization_route": "observe",
                "pending": {
                    "name": name,
                    "result": result,
                    "synthetic_kind": "projection",
                    "runtime_owned": runtime_owned,
                    "policy_owned": policy_owned,
                    "policy_prelude": policy_prelude,
                },
            }
        # Runtime-owned input is execution data, not model-authored. A confirmed
        # seal already passed this guard in the turn that produced it; the
        # resumed turn has a different context, and re-judging would turn the
        # owner's approval into a refusal.
        note_only_tokens = (
            working_notes_only_tokens(arguments=arguments, context=state["context"])
            if is_notes_guarded(name) and not runtime_owned and not owner_confirmed
            else ()
        )
        notes_refusal: ToolObservation | None = None
        if note_only_tokens:
            visible_tokens = [token[:32] for token in note_only_tokens[:8]]
            notes_refusal = ToolObservation(
                tool_name=name,
                state="working_notes_derived_argument",
                message=(
                    "以下内容只出现在工作笔记、没有用户或权威记忆来源："
                    + "、".join(visible_tokens)
                    + "；请向用户确认或改用权威来源。"
                ),
                next_action=(
                    "不要换个说法重试这次调用；请向用户确认这些内容，"
                    "或改用用户消息、已确认记忆和工具结果中的权威来源。"
                ),
                payload={"tokens": visible_tokens, "tool_name": name},
                execution_outcome="not_committed",
            )
        elif (
            is_preference_bound(name)
            and not runtime_owned
            and not owner_confirmed
            and remembered_preference_without_authority(state["context"])
        ):
            notes_refusal = ToolObservation(
                tool_name=name,
                state="working_notes_derived_argument",
                message=(
                    "用户要求按“你记得的偏好”做选择，但当前没有任何已确认的偏好来源，"
                    "只有工作笔记里未确认的观察；据此比较或推荐会把猜测当作偏好。"
                ),
                next_action=(
                    "先把工作笔记里的观察原样说给用户、请用户确认或修正，"
                    "再根据确认后的偏好选择；不要先调用比较或推荐类工具。"
                ),
                payload={
                    "tokens": [],
                    "tool_name": name,
                    "referent": "remembered_preference",
                },
                execution_outcome="not_committed",
            )
        if notes_refusal is not None:
            if (
                control.get("projection_refusals", 0)
                >= self._max_projection_refusals
            ):
                return {"authorization_route": "present"}
            return {
                "authorization_route": "observe",
                "pending": {
                    "name": name,
                    "result": notes_refusal,
                    "synthetic_kind": "projection",
                    "runtime_owned": runtime_owned,
                    "policy_owned": policy_owned,
                    "policy_prelude": policy_prelude,
                },
            }
        if verdict == "review":
            return self._seal_for_owner_confirmation(
                state, name=name, arguments=arguments
            )
        return {
            "authorization_route": "act",
            "control": control,
            "pending": {
                "name": name,
                "kind": kind,
                "runtime_owned": runtime_owned,
                "owner_confirmed": owner_confirmed,
                "policy_owned": policy_owned,
                "policy_prelude": policy_prelude,
                "effect": effect,
                "arguments": arguments,
            },
        }

    def _budget_bucket(
        self, control: LoopControl, *, name: str, effect: ToolEffect
    ) -> tuple[str, int, int]:
        """Which per-turn budget this call draws on: ``(label, used, limit)``.

        Writes are two buckets, split by where the effect lives rather than by
        how the tool is named. A reversible local record and an external
        booking used to share one slot, so "record the application and put the
        interview on the calendar" could never finish in a turn; a WRITE that
        left the deployment now has its own slot and its own ceiling.
        """

        if effect == "READ":
            return "READ", control.get("read_calls", 0), self._max_read_calls
        if effect == "CONTROL":
            # A compound request legitimately routes once per domain it
            # touches; the profile count is the natural ceiling, and the
            # repeated-call fingerprint already refuses the same route twice.
            return (
                "CONTROL",
                control.get("control_calls", 0),
                len(DOMAIN_TOOL_PROFILES),
            )
        external_used = control.get("external_write_calls", 0)
        if is_external_write(name):
            return "WRITE_EXTERNAL", external_used, self._max_external_write_calls
        return (
            "WRITE",
            control.get("write_calls", 0) - external_used,
            self._max_write_calls,
        )

    def _seal_for_owner_confirmation(
        self, state: MainAgentState, *, name: str, arguments: dict[str, Any]
    ) -> MainAgentState:
        """Hold the action the owner asked to see, bound to these arguments.

        Everything else has already passed at this point — budgets, repetition,
        projection — so the sealed action is exactly the one that would have
        run. That is what makes the owner's "yes" executable next turn without
        consulting the model again: there is nothing left to decide.

        Without a store this refuses instead of silently proceeding. A rule the
        deployment cannot durably enforce must not read as permission.
        """

        external = is_external_write(name)
        rule = (
            "这个操作会写入外部系统，写入后无法由这里撤回，因此必须由你亲自确认"
            if external
            else "你设置了这个操作需要先经你确认"
        )
        if self._capability_confirmation_store is None:
            return self._authorization_refusal(
                state,
                name=name,
                reason=f"{rule}，但本次部署无法保存待确认动作。",
                next_action="告诉用户这个操作需要确认，但当前无法记录确认请求。",
            )
        context = state["context"]
        display_summary = (
            self._external_write_summary(name=name, arguments=arguments)
            if external
            else self._owner_confirmation_summary(
                context=context, name=name, arguments=arguments
            )
        )
        confirmation = self._capability_confirmation_store.seal(
            user_id=context.profile.user_id,
            conversation_id=context.conversation_id,
            capability=name,
            display_summary=display_summary,
            arguments=arguments,
            policy_revision=context.preferences.behavior_policy.revision,
        )
        if confirmation.status == "APPLYING":
            result = ToolObservation(
                tool_name=name,
                state="capability_confirmation_in_progress",
                message="同一项已批准操作正在执行，没有再次发起确认或执行。",
                next_action="告诉用户操作仍在处理中，不要重试。",
            )
            return {
                "authorization_route": "observe",
                "pending": {
                    "name": name,
                    "result": result,
                    "synthetic_kind": "confirmation",
                    "runtime_owned": False,
                },
            }
        result = ToolObservation(
            tool_name=name,
            state="capability_confirmation_required",
            message=(
                f"{display_summary}\n"
                + (
                    "这是一次外部写入，执行后无法由这里撤回。是否执行？"
                    if external
                    else "你设置了此操作需要确认。是否执行？"
                )
            ),
            # The id is deliberately absent from the message: it is a runtime
            # identifier and the model has no use for it. It travels in the
            # payload, which the harness reads and the model's observation does
            # not, because binding a button to this action is the harness's job.
            next_action="向用户说明将要执行什么并等待确认；本轮不要重试这个操作。",
            payload={"confirmation_id": confirmation.confirmation_id},
        )
        return {
            "authorization_route": "observe",
            "pending": {
                "name": name,
                "result": result,
                "synthetic_kind": "confirmation",
                "runtime_owned": False,
                "confirmation_id": confirmation.confirmation_id,
            },
        }

    @staticmethod
    def _owner_confirmation_summary(
        *, context: MainAgentContext, name: str, arguments: dict[str, Any]
    ) -> str:
        """Render only owner-readable facts; never expose sealed internal ids."""

        if name == "create_application":
            job = next(
                (
                    item
                    for item in context.task.saved_job_candidates
                    if item.job_posting_id == arguments.get("job_posting_id")
                ),
                None,
            )
            resume = next(
                (
                    item
                    for item in context.task.resume_version_candidates
                    if item.resume_version_id == arguments.get("resume_version_id")
                ),
                None,
            )
            target = (
                f"{job.company_name} · {job.title}"
                if job is not None
                else "当前选中的岗位"
            )
            version = f"，使用简历版本 v{resume.version_number}" if resume else ""
            submitted = arguments.get("submitted_at")
            when = f"，投递时间 {submitted}" if submitted is not None else ""
            return f"准备创建投递记录：{target}{version}{when}。"
        if name == "update_owner_settings":
            changes = []
            if arguments.get("boss_search") is not None:
                changes.append(f"岗位搜索偏好 → {arguments['boss_search']}")
            if arguments.get("application_confirmation") is not None:
                changes.append(
                    "投递记录确认规则 → "
                    f"{arguments['application_confirmation']}"
                )
            if arguments.get("confirm_before") is not None:
                listed = "、".join(arguments["confirm_before"]) or "（清空）"
                changes.append(f"执行前需逐项确认的操作 → {listed}")
            return "准备更新持久设置：" + "；".join(changes) + "。"
        return f"准备执行 {name}。"

    def _external_write_summary(self, *, name: str, arguments: dict[str, Any]) -> str:
        """What will land outside this deployment, in the owner's terms.

        Read live from the store the write will act on, because the owner is
        approving the concrete event and not the model's recollection of it.
        A read that fails still yields a summary: the gate must never be
        skipped because its description could not be rendered.
        """

        if name == "execute_calendar_proposal":
            try:
                proposal = self._tools.invoke_atomic_tool(
                    "get_calendar_proposal", dict(arguments)
                )
            except Exception:  # noqa: BLE001 - rendering must not block the gate
                proposal = None
            if proposal is not None and proposal.state == "calendar_proposal_ready":
                operation = proposal.payload.get("operation")
                event = proposal.payload.get("payload")
                expires_at = proposal.payload.get("expires_at")
                if isinstance(event, dict):
                    return (
                        f"准备写入外部 Calendar（{operation}）："
                        f"{event.get('title')}，"
                        f"{event.get('start_at')} → {event.get('end_at')}"
                        f"（{event.get('timezone')}），"
                        f"地点 {event.get('location') or '未提供'}；"
                        f"预览有效期至 {expires_at}。"
                    )
                return (
                    f"准备在外部 Calendar 上执行 {operation}；"
                    f"预览有效期至 {expires_at}。"
                )
            return "准备执行已预览的 Calendar 变更。"
        return f"准备向外部系统写入：{name}。"

    @staticmethod
    def _after_authorize(
        state: MainAgentState,
    ) -> Literal["act", "observe", "present", "interrupt"]:
        return state["authorization_route"]

    def _act(self, state: MainAgentState) -> MainAgentState:
        pending = state["pending"]
        name = pending["name"]
        arguments = pending["arguments"]
        self._emit_capability_started(name)
        if pending.get("effect") == "WRITE" and self._action_execution_store is not None:
            result = self._run_capability(
                pending, lambda: self._act_request_anchored_write(state)
            )
        else:
            result = self._run_capability(
                pending, lambda: self._invoke_pending(pending)
            )
        if pending.get("effect") == "WRITE" and result.execution_outcome is None:
            raise ValueError(
                f"WRITE capability {name!r} returned without execution_outcome"
            )
        self._emit_capability_completed(name, result.state)
        return {"pending": {**pending, "result": result}}

    def _invoke_pending(self, pending: PendingAction) -> MainAgentToolOutput:
        name = pending["name"]
        arguments = pending["arguments"]
        if pending.get("runtime_owned"):
            return self._tools.invoke_runtime_workflow(name, arguments)
        if pending["kind"] == "atomic_tool":
            return self._tools.invoke_atomic_tool(name, arguments)
        return self._tools.invoke_workflow(name, arguments)

    def _act_request_anchored_write(
        self, state: MainAgentState
    ) -> MainAgentToolOutput:
        """Prepare, execute, and settle every write in this turn.

        Intent is recorded before the call and the outcome after it, so a
        process that dies mid-flight leaves a row saying "this started and
        nobody knows how it ended". Recording only after the fact cannot
        express that state at all, which is the one state crash recovery cares
        about.

        Intent registration applies to every write so an interrupted effect is
        enumerable. Only PENDING replay is capability-gated; Calendar retains
        its proposal protocol underneath this correlation layer.
        """

        invocation = _ACTION_INVOCATION.get()
        if invocation is None or self._action_execution_store is None:
            raise RuntimeError("request-anchored action context is unavailable")
        turn_id, request_id = invocation
        context = state["context"]
        pending = state["pending"]
        name = pending["name"]
        arguments = pending["arguments"]
        anchor = request_id or turn_id
        fingerprint = hashlib.sha256(
            json.dumps(
                {"tool": name, "arguments": arguments},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode()
        ).hexdigest()
        try:
            execution, created = self._action_execution_store.prepare(
                user_id=context.profile.user_id,
                conversation_id=context.conversation_id,
                anchor=anchor,
                request_id=request_id,
                # Count writes, not all calls: preceding reads do not move the
                # slot, while a deliberately larger write budget gets distinct
                # durable identities instead of colliding at slot zero.
                write_slot=self._control(state).get("write_calls", 0),
                tool_name=name,
                fingerprint=fingerprint,
                policy_epoch=self._action_policy_epoch,
                replay_allowed=replay_safe(name),
            )
        except ActionExecutionReconciliationRequiredError:
            # Fed back rather than thrown. The slot holds an earlier write whose
            # outcome nobody knows, so this turn must not start a different one —
            # but killing the turn would leave the user with a crash and no way
            # to learn what is stuck. A refusal the model can explain is the
            # same treatment projection and authorization refusals already get.
            #
            # The action id stays out of the message: it is a runtime-generated
            # internal identifier, and the operator reads it from
            # ``career-agent actions reconcile`` rather than from the model.
            return ToolObservation(
                tool_name=name,
                state="action_reconciliation_required",
                message=(
                    "上一次同类操作还没有确认结果，可能已经写入，也可能没有。"
                    "在核对清楚之前不能再执行一次，否则可能重复。"
                ),
                next_action=(
                    "告诉用户有一次未确认的操作需要先核对，不要重试这次调用。"
                ),
                execution_outcome="unknown",
            )
        if not created:
            if execution.status == "SUCCEEDED":
                # The receipt repairs task state; the model sees a distinct,
                # synthetic observation and can read the durable result by its
                # returned identifiers. Reusing the original result state here
                # would invoke a presenter whose report body is intentionally
                # absent from the ledger, or recreate a one-shot interaction.
                receipt = dict(execution.output)
                replayed_state = str(
                    receipt.pop(RESULT_STATE_RECEIPT_KEY, "") or ""
                )
                state["pending"]["reducer_result"] = ToolObservation(
                    tool_name=name,
                    state=replayed_state or "failed",
                    message="持久执行回执用于修复任务状态。",
                    payload=receipt,
                )
                return ToolObservation(
                    tool_name=name,
                    state="action_execution_replayed",
                    message="这一步此前已经完成，没有再次执行。",
                    next_action="按回执中的引用或标识读取持久结果，不要重做写操作。",
                    execution_outcome="committed",
                )
            if execution.status == "FAILED":
                raise ActionExecutionAlreadyFailedError(
                    execution.error_detail
                    or "this action already ended unsuccessfully; use a new request id"
                )
            if not replay_safe(name):
                return ToolObservation(
                    tool_name=name,
                    state="action_reconciliation_required",
                    message=(
                        "上一次这个操作没有确认结果，可能已经生效，也可能没有。"
                        "这个操作重复执行无法撤销，所以在核对清楚之前不能再执行一次。"
                    ),
                    next_action="告诉用户有一次未确认的操作需要先核对，不要重试这次调用。",
                    execution_outcome="unknown",
                )

        result = self._invoke_pending(pending)
        # ``state``/``disposition`` say whether the
        # capability and the control flow failed; ``execution_outcome`` says
        # whether the side effect committed. They are independent axes, so
        # ``committed`` with a failed state is a real situation and not a
        # contradiction to reject: the external write landed and the local
        # handling, receipt parse or presentation then failed. Settling that as
        # FAILED because the state is failed would record that nothing happened
        # when something did — the same class of lie, pointed the other way, as
        # the delivery-derived ledger this replaced. Every WRITE producer must
        # declare this axis; leaving the intent PENDING and failing loudly is
        # safer than recreating state-derived settlement here.
        if result.execution_outcome is None:
            raise ValueError(
                f"WRITE capability {name!r} returned without execution_outcome"
            )
        if result.execution_outcome == "unknown":
            # The capability has returned, but the effect has not. PENDING is
            # precisely the durable representation of that ambiguity; closing
            # it as FAILED would make it disappear from reconciliation.
            return result
        if result.execution_outcome == "not_committed":
            self._action_execution_store.fail(
                action_id=execution.action_id,
                error_code=result.state.upper(),
                error_detail=result.message,
            )
            return result
        self._action_execution_store.succeed(
            action_id=execution.action_id,
            output=MainAgentRuntime._execution_receipt(result),
        )
        return result

    @staticmethod
    def _execution_receipt(
        result: MainAgentToolOutput,
    ) -> dict[str, str | int | float | bool | None]:
        """The identifiers a crashed turn would need to repair its task state.

        Scalars only, which is a rule rather than a filter on names: a crashed
        turn's reducer never ran, so ``active_*_id`` is empty and the ids are
        what repairs it, while report bodies already live in their own stores
        and a second copy here would be a second source of truth. Payloads for
        writes are flat scalar dicts of exactly those ids plus a status, so the
        rule keeps what is needed without knowing any capability's field names.

        A capability whose payload yields nothing keeps an enumerable pending
        row and no automatic repair — the coverage boundary, not a silent
        failure.
        """
        receipt: dict[str, str | int | float | bool | None] = {
            RESULT_STATE_RECEIPT_KEY: result.state
        }
        for key, value in result.payload.items():
            if len(receipt) > _MAX_RECEIPT_KEYS:
                break
            if value is not None and not isinstance(value, (str, int, float, bool)):
                continue
            if isinstance(value, str) and len(value) > _MAX_RECEIPT_VALUE_CHARS:
                continue
            receipt[key] = value
        return receipt

    @staticmethod
    def _project_runtime_workflow_arguments(
        state: MainAgentState, name: str
    ) -> dict[str, Any]:
        """Bind private workflow input to the durable owner selected at ingress."""

        context = state["context"]
        task = context.task
        if task.active_workflow != "mock_interview" or not task.run_id:
            raise ValueError("runtime-owned mock interview has no active session")
        supplied = state.get("pending", {}).get("arguments", {})
        if name == "retry_mock_interview":
            if task.phase != "failed" or supplied:
                raise ValueError("retry_mock_interview requires one failed active run")
            return {
                "user_id": context.profile.user_id,
                "session_id": task.run_id,
            }
        if name == "handle_mock_interview_input":
            if task.phase == "failed" or set(supplied) != {"message"}:
                raise ValueError(
                    "handle_mock_interview_input requires one workflow-owned message"
                )
            message = supplied["message"]
            if not isinstance(message, str):
                raise ValueError("workflow-owned mock interview message must be text")
            return {
                "user_id": context.profile.user_id,
                "session_id": task.run_id,
                "message": message,
            }
        raise ValueError(f"Unknown runtime-owned workflow: {name}")

    @staticmethod
    def _project_workflow_arguments(
        context: MainAgentContext,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        if name == "sync_application_emails":
            return project_email_arguments(context, name, arguments)
        if name in {"research_job", "retry_job_research"}:
            return project_job_research_arguments(context, name, arguments)
        if name == "start_mock_interview":
            return project_mock_interview_arguments(context, arguments)
        if name == "restart_mock_interview":
            return project_restart_mock_interview_arguments(context, arguments)
        raise ValueError(f"Unknown main-agent workflow: {name}")

    @staticmethod
    def _reraise_security_refusal(error: ValueError) -> None:
        """Keep the least-privilege boundary hard, unlike a soft refusal.

        A projection error says one of two things: the object the model named is
        not there (a wrong but ordinary choice, softened below), or the model is
        reaching for identifiers or argument shapes it must never be able to
        touch. The second must still kill the turn before it commits anything:
        softening it would turn the guard into a suggestion.
        """
        message = str(error)
        if (
            "cannot accept internal identifier" in message
            or "Extra inputs are not permitted" in message
            or message.startswith("Unknown ")
        ):
            raise error

    @staticmethod
    def _rejection_observation(name: str, error: ValueError) -> ToolObservation:
        """The soft form of a projection refusal, safe to present.

        A model-selected tool whose preconditions fail at projection used to
        raise through the whole turn, killing it with a canned failure. That
        gave the model no way to recover and the user no say. Now the refusal
        returns as an ordinary result and the model decides what to do with it —
        re-select, list what is available, or ask the user for the one thing
        only the user has.

        The state is unconditional. An earlier version chose between
        ``needs_user`` and ``invalid_input`` by looking up the capability in a
        reroute table; that decision now belongs to the model, and the counter
        in ``_authorize`` bounds how many times it may take it.
        """
        return ToolObservation(
            tool_name=name,
            state="invalid_input",
            message=f"这步暂时做不到：{error}。",
            # The one thing the state cannot say: this is not a failure to retry
            # but a selection that did not hold. Deleting ``REROUTE_FIELDS`` gave
            # the model this decision; leaving the hint empty would have given it
            # the decision without the knowledge the table used to carry.
            next_action=(
                "这不是失败，是选择或参数不成立。原样重试没有意义："
                "换一个已经在上下文里的对象，或者向用户要一个只有他才有的信息。"
            ),
        )

    def _observe(self, state: MainAgentState) -> MainAgentState:
        context = state["context"]
        pending = state["pending"]
        result = pending["result"]
        capability_name = pending["name"]
        control = dict(self._control(state))
        synthetic_kind = pending.get("synthetic_kind")
        if result.disposition == "failed":
            # The tool layer already classified this failure into an error_code
            # and a retryability flag. Copy them into the durable trace so a
            # failure that lasted one turn does not vanish with the payload.
            MainAgentRuntime._emit_trace(
                "capability_failed",
                capability_name,
                error_code=str(result.payload.get("error_code"))
                if result.payload.get("error_code") is not None
                else "CAPABILITY_FAILED",
                recoverable=(
                    bool(result.payload.get("retryable"))
                    if "retryable" in result.payload
                    else None
                ),
            )
        if synthetic_kind == "confirmation":
            # A seal is not a refusal. The action was allowed; the owner asked
            # to see it first, and the turn ends waiting for them. Counting it
            # against the refusal budget would make a rule the owner set look
            # like the model misbehaving, and would end the conversation early
            # after a few legitimate confirmations. No capability ran, so no
            # effect budget moves either.
            updated = context
        elif synthetic_kind is not None:
            updated = context
            refusal_key = (
                "projection_refusals"
                if synthetic_kind == "projection"
                else "authorization_refusals"
            )
            control[refusal_key] = control.get(refusal_key, 0) + 1
        else:
            if result.tool_name in {
                "start_mock_interview",
                "restart_mock_interview",
                "handle_mock_interview_input",
                "retry_mock_interview",
            }:
                updated = self._update_mock_interview_task(context, result)
            else:
                updated = self._update_atomic_task(
                    context,
                    pending.get("reducer_result", result),
                    now=self._context_manager.now(),
                )
            if result.state in {
                "career_memory_amended",
                "memory_tombstoned",
                "memory_tombstone_cleanup_incomplete",
                "free_text_preference_confirmed",
                "free_text_preference_confirmed_structured_proposed",
                "career_fact_confirmed",
                "working_notes_stale",
                "working_notes_updated",
            }:
                refreshed = self._context_manager.load_for_turn(
                    user_id=context.profile.user_id,
                    conversation_id=context.conversation_id,
                    # The message as sent: reloading from the prompt's clipped
                    # copy would lose the original, and the turn would store
                    # the clipped one.
                    user_message=context.stored_user_message(),
                )
                refresh_updates: dict[str, Any] = {
                    "task": updated.task,
                    "attached_resumes": context.attached_resumes,
                }
                updated = refreshed.model_copy(update=refresh_updates)
            effect = pending["effect"]
            budget_key = {
                "READ": "read_calls",
                "WRITE": "write_calls",
                "CONTROL": "control_calls",
            }[effect]
            control[budget_key] = control.get(budget_key, 0) + 1
            if effect == "WRITE" and is_external_write(pending["name"]):
                control["external_write_calls"] = (
                    control.get("external_write_calls", 0) + 1
                )
            fingerprint = self._tool_call_fingerprint(state["decision"])
            fingerprints = control.get("fingerprints", ())
            if fingerprint not in fingerprints:
                control["fingerprints"] = (*fingerprints, fingerprint)
            retryable_fingerprints = tuple(control.get("retryable_fingerprints", ()))
            if (
                result.disposition == "failed"
                and result.payload.get("retryable") is True
            ):
                if fingerprint not in retryable_fingerprints:
                    retryable_fingerprints = (*retryable_fingerprints, fingerprint)
            else:
                retryable_fingerprints = tuple(
                    item for item in retryable_fingerprints if item != fingerprint
                )
            control["retryable_fingerprints"] = retryable_fingerprints
        # The model's own arguments, not ``pending["arguments"]``. Projection
        # turns a selector into what the handler needs — including live domain
        # objects and the internal ids the projection boundary exists to keep
        # away from the model — so the projected form is neither safe to show
        # nor the thing the model would recognize as its own call.
        decision = state.get("decision")
        written = (
            decision.tool_call.arguments
            if decision is not None and decision.tool_call is not None
            else {}
        )
        observation = self._tool_observation(capability_name, result, written)
        updated = updated.model_copy(
            update={
                "tool_observations": append_decision_observation(
                    updated.tool_observations,
                    observation,
                )
            }
        )
        artifact_ids = state.get("artifact_ids", ())
        if result.state == "resume_artifact_ready":
            artifact_id = result.payload.get("artifact_id")
            if isinstance(artifact_id, str) and artifact_id not in artifact_ids:
                artifact_ids = (*artifact_ids, artifact_id)
        tool_results = state.get("tool_results", ())
        # A refusal is an internal correction the model reads and moves past; a
        # seal is this turn's actual outcome, and the interrupt path reads it
        # from here to build the interaction the owner answers.
        if synthetic_kind in (None, "confirmation"):
            tool_results = (*tool_results, result)
        career_memory_scope_keys = state.get("career_memory_scope_keys", ())
        result_scope_key = result.payload.get(
            "memory_entry_id",
            result.payload.get("scope_key"),
        )
        if (
            isinstance(result_scope_key, str)
            and result_scope_key
            and result_scope_key not in career_memory_scope_keys
        ):
            career_memory_scope_keys = (
                *career_memory_scope_keys,
                result_scope_key,
            )
        return {
            "context": updated,
            "tool_results": tool_results,
            "control": control,
            "artifact_ids": artifact_ids,
            "career_memory_scope_keys": career_memory_scope_keys,
        }

    @staticmethod
    def _after_observe(
        state: MainAgentState,
    ) -> Literal["hydrate", "decide", "present", "interrupt"]:
        pending = state["pending"]
        result = pending["result"]
        # A capability that already produced a complete, bound interaction
        # contract does not need an LLM to paraphrase or rediscover its prompt.
        if result.disposition == "interaction_required":
            return "interrupt"
        # A prelude read opened the turn before the model saw anything; the
        # ordinary turn now starts, with the read among its observations.
        if pending.get("policy_prelude"):
            return "hydrate"
        # A workflow-owned input must not fall through to the general decision
        # model after execution. Its successor is already defined by the child
        # workflow result, and the raw input was intentionally withheld from
        # Main Agent context.
        if state.get("pending", {}).get("runtime_owned"):
            return "present"
        # An owner-confirmed action is finished the moment it runs. Handing the
        # result back to the model would give it a turn it was never given: it
        # proposed this action, a rule stopped it, and a person answered. The
        # presenter reports what happened; nothing further is up for decision.
        if state.get("pending", {}).get("owner_confirmed"):
            return "present"
        if state.get("pending", {}).get("policy_owned"):
            return "present"
        # A projection refusal always returns to the model, which then re-selects,
        # asks the user, or explains — its call, not a table's.
        #
        # A ``REROUTE_FIELDS`` table used to end the turn here whenever task
        # state held no candidate list the refused tool could draw from. It
        # answered the right question ("could a re-selection possibly work?")
        # in the wrong place: whether a call is *possible* is the harness's to
        # know, but whether to retry, ask, or give up is the agent's to decide,
        # and the model is better placed than a canned presenter to say "only
        # you can tell me which one". It was also a hardcoded capability-name
        # successor table of exactly the kind this design rejects everywhere
        # else, and it had to be edited in lockstep with the menu table.
        #
        # The cost is one model call in the hopeless case. The bound is
        # ``max_projection_refusals``: every ``invalid_input`` comes from
        # ``_rejection_observation``, which only runs on the synthetic
        # projection path, so every one of them increments that counter and the
        # limit exits to ``present`` on its own.
        if pending.get("synthetic_kind") is not None:
            return "decide"
        # Failures intentionally return to the model once, with their bounded
        # observation, so it can explain, recover, or ask the user. This is not
        # the old waiting-state fallthrough.
        if result.disposition == "failed":
            return "decide"
        return "decide"

    @staticmethod
    def _present(state: MainAgentState) -> MainAgentState:
        """Deliver the model's own answer, with the presenter as the fallback.

        The model narrates now. It has seen the same presenter text the reader
        will — H puts a bounded body on the newest observation — so a post-tool
        answer is grounded rather than invented, and one message can cover a
        multi-step turn, which the presenter structurally cannot: it renders a
        single result and silently drops every earlier one.

        The presenter keeps *presentation*, and for one family it keeps the
        delivery itself. A condensed state with no card has nowhere else to put
        its body: the reader sees it here or never. The model may only introduce
        such a body, never stand in for it — it read at most a truncated 6k
        projection of it, and its reply is bounded far below that, so letting
        the reply replace the body would silently drop a complete JD, a
        comparison matrix or a mock interview readback.

        For everything else the reply is the delivery: card-backed reports reach
        the reader through the card, and a plain state's receipt is already the
        whole of it. The presenter remains the fallback for turns with no model
        prose — a control-limit exit, or a model that answered with an empty
        message.
        """

        decision = state["decision"]
        pending = state.get("pending", {})
        pending_result = pending.get("result")
        if (
            decision.action == "tool_call"
            and (
                pending.get("synthetic_kind") == "projection"
                or (
                    pending.get("synthetic_kind") == "authorization"
                    and pending.get("runtime_owned")
                )
            )
            and pending_result is not None
        ):
            # A projection refusal with no possible candidate repair is a safe,
            # user-facing explanation, but it remains outside tool_results.
            return {
                "assistant_message": MainAgentRuntime._assistant_message(
                    pending_result
                )
            }
        result = MainAgentRuntime._last_result(state)
        # ``strip()``, not truthiness: a whitespace-only message would enter this
        # branch and then clamp to "", leaving ``model_message`` empty while the
        # branch claims the model wrote the reply — and prefixing the body with a
        # blank line. The invariant below has to be true, not nearly true.
        if decision.action == "final" and (decision.message or "").strip():
            results = state.get("tool_results", ())
            body = MainAgentRuntime._undelivered_bodies(results)
            # The card ceiling means "this message is only prose about a body
            # delivered elsewhere". That holds when every delivery this turn is
            # elsewhere. It stops holding the moment the reply also has to
            # introduce a body with nowhere else to go, or cover steps the card
            # says nothing about.
            only_cards = MainAgentRuntime._turn_is_card_backed(results)
            reply = clamp(
                decision.message,
                limit=(
                    DELIVERY_SUMMARY_LIMIT if only_cards else MODEL_REPLY_LIMIT
                ),
            )
            return {
                "assistant_message": f"{reply}\n\n{body}" if body else reply,
                # Non-empty exactly when the model wrote the reply, so the
                # delivery layer needs no second flag to say the same thing.
                "model_message": reply,
            }
        if result is not None:
            return {"assistant_message": MainAgentRuntime._screen_message(result)}
        return {"assistant_message": "本轮可执行步骤已达到上限，请确认后继续。"}

    @staticmethod
    def _screen_message(result: MainAgentToolOutput) -> str:
        """The presenter text for a turn the model did not close with prose.

        A saved-job read is the one state whose rendered body is the JD itself.
        The model reads that body as its observation; the reader gets the card,
        so the screen carries the receipt rather than the text the card owns.
        """
        if result.state == "saved_job_ready" and result.resource_ref is not None:
            return result.message
        return MainAgentRuntime._assistant_message(result)

    @staticmethod
    def _turn_resource_refs(
        results: tuple[MainAgentToolOutput, ...],
    ) -> tuple[ConversationResourceReference, ...]:
        """Every stored report this turn produced, deduplicated by resource.

        Shares its judgement with card emission on purpose: the same turn must
        not hand the reader two cards while the transcript records one, and it
        must not mint two references for one report when
        ``research_job`` and ``get_job_research`` land on the same report.
        """
        references: list[ConversationResourceReference] = []
        seen: set[str] = set()
        for result in results:
            reference = result.resource_ref
            if reference is None or reference.resource_id in seen:
                continue
            seen.add(reference.resource_id)
            references.append(reference)
        return tuple(references)

    @staticmethod
    def _turn_is_card_backed(results: tuple[MainAgentToolOutput, ...]) -> bool:
        """Whether every delivery this turn goes out through a card.

        The single question three forks have to answer the same way: the reply
        may be trimmed to card-length prose, the stream may hand the body to the
        card instead of writing it out, and the row may keep the receipt — each
        of those is only safe when *nothing* this turn needs the message itself
        to carry a body.

        It is a property of the turn, never of its last result. Reading a JD and
        then researching a company gives the last result a card while the JD has
        none; answering "yes, all cards" there drops the JD from the stream and
        cuts the reply to 600 characters. Every fork asks this one function so
        the three answers cannot drift apart again.
        """
        return bool(results) and all(
            MainAgentRuntime._has_backed_card(item) for item in results
        )

    @staticmethod
    def _durable_screen(result: MainAgentTurnResult) -> str:
        """What the transcript and the live row should carry.

        ``assistant_message`` is the screen delivery, which for a card-less
        condensed state includes the presenter body. The row must not: that body
        is what the recent window exists to stay out of. When the model wrote the
        reply, the row keeps the reply.
        """
        return result.model_message or result.assistant_message

    @staticmethod
    def _undelivered_bodies(results: tuple[MainAgentToolOutput, ...]) -> str:
        """Every presenter body in this turn that no other path will carry.

        Pinned by ``test_every_card_less_body_in_the_turn_is_delivered_not_just_the_last``,
        which drives ``_present`` rather than naming this helper.

        A condensed state without a card keeps a bounded line in the transcript
        and shows the body live; nothing else ever shows it. Plain states put
        everything in the receipt, and card states deliver through the entity,
        so neither appears here.

        Deliberately every such result, not the last one. Reading a JD and then
        comparing saved jobs produces two bodies with no card behind either, and
        taking ``tool_results[-1]`` would drop the JD — reproducing one level up
        the exact failure this turn's composition was meant to fix.

        Deliberately unbounded, per body and in total: the 6k observation limit
        bounds what the *model* reads, and clamping here would re-truncate the
        full JD this path exists to deliver. The read budget is what keeps a turn
        from stacking several of them; see 070 for who owns which ceiling.
        """
        bodies: list[str] = []
        for result in results:
            policy = policy_for(result.state)
            if not policy.condensed_message or policy.delivers_body_elsewhere:
                continue
            rendered = MainAgentRuntime._assistant_message(result)
            if rendered and rendered not in bodies:
                bodies.append(rendered)
        return "\n\n".join(bodies)

    @staticmethod
    def _delivered_bodies(
        results: tuple[MainAgentToolOutput, ...],
    ) -> tuple[DeliveredBodyDraft, ...]:
        drafts: list[DeliveredBodyDraft] = []
        for result in results:
            policy = policy_for(result.state)
            if policy.body_retention == "none" or policy.body_title is None:
                continue
            if policy.body_retention == "source" and result.body_source is None:
                continue
            draft = DeliveredBodyDraft(
                kind=result.state,
                title=policy.body_title,
                retention=policy.body_retention,
                body=(
                    MainAgentRuntime._assistant_message(result)
                    if policy.body_retention == "snapshot" else ""
                ),
                source=result.body_source,
                dependencies=(
                    (*result.body_dependencies, BodyDependency(
                        kind="job", resource_id=result.body_source.job_posting_id
                    ))
                    if isinstance(result.body_source, SavedJobBodySource)
                    else result.body_dependencies
                ),
            )
            if draft not in drafts:
                drafts.append(draft)
        return tuple(drafts)

    @staticmethod
    def _interrupt(state: MainAgentState) -> MainAgentState:
        """Suspend on either a model question or a capability-owned interaction."""

        decision = state["decision"]
        if decision.action == "ask_user":
            return {"assistant_message": decision.message or "请补充下一步所需的信息。"}
        result = MainAgentRuntime._last_result(state)
        if result is None:
            raise ValueError("capability interaction requires an observed result")
        if not MainAgentRuntime._has_interaction_renderer(result.state):
            raise ValueError(
                "interaction_required result has no interaction renderer: "
                f"{result.tool_name}/{result.state}"
            )
        return {"assistant_message": MainAgentRuntime._assistant_message(result)}

    @staticmethod
    def _tool_observation(
        name: str,
        result: MainAgentToolOutput,
        arguments: dict[str, Any] | None = None,
    ) -> DecisionObservation:
        receipt = clamp(result.message) or "工具已返回，但没有提供结果摘要。"
        body = None
        if condenses_message(result.state):
            rendered = MainAgentRuntime._assistant_message(result)
            body = clamp(rendered, limit=DECISION_OBSERVATION_BODY_LIMIT) or None
        if isinstance(result, ToolObservation):
            return DecisionObservation(
                tool_name=result.tool_name,
                state=result.state,
                message=receipt,
                body=body,
                facts=dict(result.facts),
                next_action=result.next_action,
                arguments=dict(arguments or {}),
                # The handle survives the body. Clearing keeps the reference for
                # the same reason tool-result clearing keeps the tool_use record.
                resource_ref=result.resource_ref,
                resource_refs=result.resource_refs,
            )
        return DecisionObservation(
            tool_name=name,
            state=result.state,
            message=receipt,
            body=body,
            facts=dict(result.facts),
            next_action=result.next_action,
            arguments=dict(arguments or {}),
            resource_ref=result.resource_ref,
            resource_refs=result.resource_refs,
        )

    @staticmethod
    def _conversation_content(
        result: MainAgentToolOutput | None,
        *,
        screen: str,
        composed: bool,
    ) -> str:
        """Choose the durable row for each of the three delivery shapes.

        * Full row: screen and transcript keep the same text.
        * Summary row + resource card: the message is the same bounded prose
          live and after refresh; the entity-backed card owns the full body.
        * Summary row + message body: the transcript keeps the receipt.
          ``body_retention`` independently selects a snapshot card, a source
          card, or no retained body.

        Normally keyed on the state's declared policy. A missing reference is
        treated as a broken instance of a card policy and fails open to the
        full screen body, because there is then nowhere else to retrieve it.
        """
        if composed:
            # F: the model wrote this, having seen the same bounded presenter
            # text the reader gets. There is no large body left to condense —
            # the row keeps the answer, and the card still owns the report.
            #
            # Returned as-is. ``_present`` already cut the reply to the ceiling
            # this turn earns: card length when every delivery is a card, the
            # full reply ceiling otherwise. Clamping again here would apply the
            # card ceiling to a turn that did not earn it, storing 600
            # characters of an answer the reader was shown in full.
            return screen
        if result is None or not condenses_message(result.state):
            return screen
        # A policy may claim a card only if this particular observation carries
        # the reference that makes the body retrievable. Durable effects may
        # already have happened, so fail open to the full message rather than
        # raising after the fact or silently replacing it with a receipt.
        if delivers_body_elsewhere(result.state):
            if result.resource_ref is None:
                return screen
            # The stream is already cut at this number. This is a pure second
            # boundary for card-backed prose, never the summarisation strategy
            # for an ephemeral message body.
            return result.message
        return result.message

    @staticmethod
    def _assistant_message(result: MainAgentToolOutput) -> str:
        if result.state == "saved_job_ready":
            snapshot = result.payload.get("jd_snapshot")
            if isinstance(snapshot, dict):
                content = snapshot.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
        if result.state in {"conversation_span_found", "conversation_span_empty"}:
            view = MainAgentRuntime._validated(ConversationSpanView, result.payload)
            if view is not None:
                return render_conversation_span(view)
        if result.state == "claim_source_found":
            source_quote = result.payload.get("source_quote")
            if isinstance(source_quote, str) and source_quote:
                claim_status = result.facts.get("claim_status")
                if claim_status == "superseded":
                    changed_at = result.facts.get("status_changed_at")
                    return (
                        "注意：这是已被更正声明的历史引文，不能作为当前声明的"
                        f"支持（状态变更时间：{changed_at or '未记录'}）。\n\n"
                        f"原始证据引文：\n\n{source_quote}"
                    )
                return f"原始证据引文：\n\n{source_quote}"
        if result.state in {
            "career_memory_detail_found",
            "career_memory_search_found",
            "career_episode_search_found",
            "career_history_found",
        }:
            body = result.payload.get("body")
            if isinstance(body, str) and body.strip():
                return body.strip()
        if result.state in MainAgentRuntime._MOCK_INTERVIEW_GRAPH_STATES:
            # The workflow's own presenter handles every state a run can be
            # left in, including the terminal ones, so the payload is parsed
            # back into the graph's result type once and handed over whole
            # rather than picked apart here.
            graph_result = MainAgentRuntime._mock_interview_result(result)
            if graph_result is not None:
                return render_mock_interview_turn(graph_result)
        if result.state == "mock_interview_result_found":
            view = MainAgentRuntime._validated(MockInterviewResultView, result.payload)
            if view is not None:
                return render_mock_interview_result(view)
        if result.state == "mock_interview_question_found":
            view = MainAgentRuntime._validated(MockInterviewQuestionView, result.payload)
            if view is not None:
                return render_mock_interview_question(view)
        if result.state == "daily_brief_ready":
            return render_daily_brief(result.payload)
        if result.state == "resume_analysis_ready":
            analysis = MainAgentRuntime._resume_analysis_result(result)
            if analysis is not None:
                return render_resume_analysis(analysis)
        if result.state == "interview_retro_recorded":
            view = MainAgentRuntime._validated(InterviewRetroView, result.payload)
            if view is not None:
                return render_interview_retro(view)
        if result.state == "resume_job_match_ready":
            match = MainAgentRuntime._resume_job_match_result(result)
            if match is not None:
                return render_resume_job_match(match)
        if result.state == "resume_tailoring_draft_ready":
            tailoring = MainAgentRuntime._resume_tailoring_result(result)
            if tailoring is not None:
                reviews = tuple(
                    review
                    for raw in result.payload.get("change_reviews", ())
                    if isinstance(raw, dict)
                    and (
                        review := MainAgentRuntime._validated(
                            TailoringChangeReviewView, raw
                        )
                    )
                    is not None
                )
                raw_status = str(result.payload.get("status") or "pending")
                status = (
                    raw_status
                    if raw_status
                    in {
                        "pending",
                        "in_review",
                        "reviewed",
                        "finalized",
                        "superseded",
                        "expired",
                    }
                    else "pending"
                )
                raw_revision = result.payload.get("revision_number", 1)
                revision_number = (
                    raw_revision
                    if isinstance(raw_revision, int) and raw_revision >= 1
                    else 1
                )
                return render_resume_tailoring(
                    tailoring,
                    status=status,
                    revision_number=revision_number,
                    change_reviews=reviews,
                )
        if result.state == "job_research_ready":
            research = MainAgentRuntime._job_research_draft(result)
            if research is not None:
                return render_job_research(
                    research,
                    status=str(result.payload.get("status") or "current"),
                    user_provided_context=(
                        str(result.payload["user_provided_context"])
                        if result.payload.get("user_provided_context") is not None
                        else None
                    ),
                    anchored_by_other_job=bool(
                        result.payload.get("anchored_by_other_job")
                    ),
                )
        if result.state == "saved_jobs_compared":
            comparison = MainAgentRuntime._job_comparison(result)
            if comparison is not None:
                return render_job_comparison(comparison)
        if result.state == "interview_preparation_ready":
            preparation = MainAgentRuntime._interview_preparation_result(result)
            if preparation is not None:
                return render_interview_preparation(preparation)
        if result.state == "calendar_approval_required":
            payload = result.payload.get("payload")
            if isinstance(payload, dict):
                return (
                    "请确认是否执行以下 Calendar 变更：\n"
                    f"- 操作：{result.payload.get('operation')}\n"
                    f"- 标题：{payload.get('title')}\n"
                    f"- 开始：{payload.get('start_at')}\n"
                    f"- 结束：{payload.get('end_at')}\n"
                    f"- 时区：{payload.get('timezone')}\n"
                    f"- 地点：{payload.get('location') or '未提供'}\n"
                    f"- 预览失效时间：{result.payload.get('expires_at')}\n"
                    "只有你明确确认后才会写入外部 Calendar。"
                )
            return (
                "请确认是否取消这条 Calendar 事件。"
                f"预览失效时间：{result.payload.get('expires_at')}。"
            )
        return result.message

    @staticmethod
    def _validated(model, payload: object):
        """Parse a payload back into the type its presenter takes, or give up.

        A payload that will not validate means the presenter cannot be reached,
        and the caller falls back to the tool's own ``message``. Presenting
        something is always better than failing a turn whose work is already
        durable. But the degrade must not be silent: it now records a trace
        event, or the report-turned-receipt is indistinguishable from a normal
        answer and no one ever knows the payload drifted.
        """
        try:
            return model.model_validate(payload)
        except ValueError as error:
            MainAgentRuntime._emit_trace(
                "presentation_degraded",
                getattr(model, "__name__", type(model).__name__ or "presenter"),
                error_detail="validation_error",
                details={
                    "errors": getattr(error, "errors", lambda: ())() and str(error),
                },
            )
            return None

    @staticmethod
    def _mock_interview_result(
        result: MainAgentToolOutput,
    ) -> MockInterviewGraphResult | None:
        return MainAgentRuntime._validated(MockInterviewGraphResult, result.payload)

    @staticmethod
    def _has_backed_card(result: MainAgentToolOutput) -> bool:
        return delivers_body_elsewhere(result.state) and result.resource_ref is not None

    @staticmethod
    def _resume_analysis_result(
        result: MainAgentToolOutput,
    ) -> ResumeAnalysisResult | None:
        return MainAgentRuntime._validated(
            ResumeAnalysisResult,
            {
                "records": result.payload.get("records", ()),
                "clarification_questions": result.payload.get(
                    "clarification_questions", ()
                ),
                "warnings": result.payload.get("warnings", ()),
            },
        )

    @staticmethod
    def _resume_job_match_result(
        result: MainAgentToolOutput,
    ) -> ResumeJobMatchResult | None:
        return MainAgentRuntime._validated(
            ResumeJobMatchResult,
            {
                key: result.payload.get(key)
                for key in ResumeJobMatchResult.model_fields
            },
        )

    @staticmethod
    def _resume_tailoring_result(
        result: MainAgentToolOutput,
    ) -> ResumeTailoringResult | None:
        changes = []
        for raw in result.payload.get("changes", ()):
            if not isinstance(raw, dict):
                continue
            changes.append(
                {key: value for key, value in raw.items() if key != "change_index"}
            )
        return MainAgentRuntime._validated(
            ResumeTailoringResult,
            {
                "strategy_summary": result.payload.get("strategy_summary"),
                "changes": changes,
                "preserved_strengths": result.payload.get("preserved_strengths", ()),
                "unresolved_gaps": result.payload.get("unresolved_gaps", ()),
                "clarification_questions": result.payload.get(
                    "clarification_questions", ()
                ),
                "warnings": result.payload.get("warnings", ()),
            },
        )

    @staticmethod
    def _job_comparison(result: ToolObservation) -> JobComparison | None:
        raw = result.payload.get("comparison")
        if not isinstance(raw, dict):
            return None
        try:
            return JobComparison.model_validate(raw)
        except ValueError:
            return None

    @staticmethod
    def _interview_preparation_result(
        result: ToolObservation,
    ) -> InterviewPreparationResult | None:
        raw = result.payload.get("preparation")
        if not isinstance(raw, dict):
            return None
        try:
            return InterviewPreparationResult.model_validate(raw)
        except ValueError:
            return None

    @staticmethod
    def _job_research_draft(result: ToolObservation) -> JobResearchDraft | None:
        raw = result.payload.get("research")
        raw_sources = result.payload.get("sources")
        if not isinstance(raw, dict) or not isinstance(raw_sources, list):
            return None
        try:
            sources = tuple(
                JobResearchSourceDraft(
                    source_key=item["source_key"],
                    url=item["url"],
                    title=item["title"],
                    publisher=item.get("publisher"),
                    published_at=item.get("published_at"),
                    relevant_excerpt=item["relevant_excerpt"],
                )
                for item in raw_sources
                if isinstance(item, dict)
            )
            findings = tuple(
                JobResearchFindingDraft.model_validate(item)
                for item in raw.get("findings", ())
            )
            return JobResearchDraft(
                summary=raw["summary"],
                sources=sources,
                findings=findings,
                open_questions=tuple(raw.get("open_questions", ())),
                limitations=tuple(raw.get("limitations", ())),
            )
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _project_atomic_tool_arguments(context: MainAgentContext, name: str, arguments: dict[str, object]) -> dict[str, object]:
        if name == "route_to_capability":
            model_arguments = RouteToCapabilityToolArguments.model_validate(arguments)
            return {
                "current_tool_profile": context.task.tool_profile,
                **model_arguments.model_dump(),
            }
        if name == "read_conversation_span":
            model_arguments = ReadConversationSpanToolArguments.model_validate(
                arguments
            )
            return {
                "user_id": context.profile.user_id,
                "conversation_id": context.conversation_id,
                **model_arguments.model_dump(exclude_none=True),
            }
        if name == "resolve_claim_source":
            model_arguments = ResolveClaimSourceToolArguments.model_validate(arguments)
            return {
                "user_id": context.profile.user_id,
                **model_arguments.model_dump(),
            }
        if name == "get_career_memory_detail":
            model_arguments = GetCareerMemoryDetailToolArguments.model_validate(
                arguments
            )
            return {
                "user_id": context.profile.user_id,
                **model_arguments.model_dump(),
            }
        if name == "search_career_episodes":
            model_arguments = SearchCareerEpisodesToolArguments.model_validate(
                arguments
            )
            if (
                model_arguments.detail_ref is not None
                and model_arguments.detail_ref
                not in {item.detail_ref for item in context.career_episodes}
            ):
                raise ValueError(
                    "episode detail_ref was not projected in this turn"
                )
            return {
                "user_id": context.profile.user_id,
                **model_arguments.model_dump(),
            }
        if name == "search_career_memory":
            model_arguments = SearchCareerMemoryToolArguments.model_validate(arguments)
            return {
                "user_id": context.profile.user_id,
                **model_arguments.model_dump(),
            }
        if name == "search_career_history":
            model_arguments = SearchCareerHistoryToolArguments.model_validate(arguments)
            return {
                "user_id": context.profile.user_id,
                **model_arguments.model_dump(),
            }
        if name == "update_owner_settings":
            proposed = UpdateOwnerSettingsToolArguments.model_validate(arguments)
            changes = proposed.model_dump(exclude_none=True)
            if all(
                (
                    value == context.preferences.boss_search
                    if key == "boss_search"
                    else value == context.preferences.application_confirmation
                )
                for key, value in changes.items()
            ):
                raise ValueError("owner-settings proposal does not change current settings")
            return {
                "user_id": context.profile.user_id,
                "expected_revision": context.preferences.revision,
                **changes,
            }
        if name == "open_job_search":
            return project_open_job_search_arguments(context, arguments)
        if name in {
            "propose_job_intent",
            "confirm_job_intent",
        }:
            return project_job_intent_arguments(context, name, arguments)
        if name in {
            "propose_free_text_preference_confirmation",
            "confirm_free_text_preference",
        }:
            return project_free_text_preference_arguments(
                context, name, arguments
            )
        if name in {
            "propose_memory_tombstone",
            "confirm_memory_tombstone",
        }:
            return project_memory_tombstone_arguments(context, name, arguments)
        if name in {
            "propose_memory_amendment",
            "confirm_memory_amendment",
        }:
            return project_memory_amendment_arguments(context, name, arguments)
        if name == "update_working_notes":
            return project_working_notes_arguments(context, name, arguments)
        if name in {"propose_career_fact", "confirm_career_fact"}:
            return project_career_fact_arguments(context, name, arguments)
        if name in {
            "fetch_archived_constraints",
            "propose_constraint_retirement",
            "confirm_constraint_retirement",
        }:
            return project_constraint_retirement_arguments(
                context, name, arguments
            )
        if name in {"find_saved_jobs", "get_saved_job", "compare_saved_jobs"}:
            return project_saved_job_arguments(context, name, arguments)
        if name == "get_job_research":
            return project_job_research_arguments(context, name, arguments)
        if name in {"list_email_events", "resolve_email_event"}:
            return project_email_arguments(context, name, arguments)
        if name in {
            "list_interviews",
            "get_interview",
            "create_interview",
            "update_interview",
            "complete_interview",
            "record_interview_retro",
        }:
            return project_interview_arguments(context, name, arguments)
        if name in {"prepare_interview", "get_interview_preparation"}:
            return project_interview_preparation_arguments(context, name, arguments)
        if name == "get_mock_interview_result":
            return project_mock_interview_result_arguments(context, arguments)
        if name in {
            "get_daily_brief",
            "list_action_items",
            "complete_action_item",
            "dismiss_action_item",
            "snooze_action_item",
        }:
            return project_action_center_arguments(context, name, arguments)
        if name in {
            "list_calendar_accounts",
            "list_calendar_links",
            "prepare_interview_calendar_sync",
            "get_calendar_proposal",
            "execute_calendar_proposal",
        }:
            return project_calendar_arguments(context, name, arguments)
        if name in {
            "list_target_roles",
            "list_resumes",
            "get_resume_metadata",
            "analyze_resume",
            "get_resume_analysis",
            "match_resume_to_job",
            "get_resume_job_match",
            "draft_resume_tailoring",
            "get_resume_tailoring_draft",
            "review_resume_tailoring",
            "revise_resume_tailoring",
            "finalize_resume_tailoring",
            "export_resume_artifact",
            "create_application",
            "update_application_status",
            "list_applications",
            "get_application",
        }:
            return project_resume_arguments(context, name, arguments)
        return arguments

    @staticmethod
    def _update_mock_interview_task(
        context: MainAgentContext, result: ToolObservation
    ) -> MainAgentContext:
        task = context.task
        session_id = result.payload.get("session_id")
        if result.state in {
            "mock_interview_answer_required",
            "mock_interview_running",
        }:
            if not isinstance(session_id, str) or not session_id:
                raise ValueError("Mock interview result has no session_id")
            task = task.enter_workflow(
                "mock_interview",
                run_id=session_id,
                phase=result.state,
                candidates=(),
            )
        elif result.state in {
            "mock_interview_completed",
            "mock_interview_cancelled",
            "mock_interview_restart_failed",
            "no_mock_interview_to_restart",
        }:
            if task.active_workflow == "mock_interview":
                task = task.leave_workflow()
        elif result.state == "failed" and task.active_workflow == "mock_interview":
            # A persisted answer can be retried. Keep ownership instead of
            # stranding the graph after a transient Worker failure.
            task = task.enter_workflow(
                "mock_interview",
                run_id=task.run_id or str(session_id),
                phase="failed",
                candidates=(),
            )
        elif result.state in {
            "mock_interview_checkpoint_missing",
            "mock_interview_graph_incompatible",
        } and task.active_workflow == "mock_interview":
            task = task.enter_workflow(
                "mock_interview",
                run_id=task.run_id or str(session_id),
                phase=result.state,
                candidates=(),
            )
        return context.model_copy(update={"task": task})

    @staticmethod
    def _update_atomic_task(
        context: MainAgentContext,
        result: ToolObservation,
        *,
        now: datetime | None = None,
    ) -> MainAgentContext:
        return context.model_copy(
            update={"task": reduce_task_state(context.task, result, now=now)}
        )
