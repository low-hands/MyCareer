from __future__ import annotations

import json
import hashlib
from contextvars import ContextVar
from time import perf_counter
from dataclasses import dataclass
from threading import Lock
from typing import Any, ClassVar, Literal, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.career_context import CareerContextProjector
from career_agent.agent.decision_messages import decision_context_chars
from career_agent.agent.main_agent_contracts import AgentDecision, ConversationResourceReference, ConversationSpanView, ConversationTaskState, DECISION_OBSERVATION_BODY_LIMIT, DecisionMaker, DecisionObservation, GetCareerMemoryDetailToolArguments, MainAgentContext, MAX_DECISION_OBSERVATIONS, ReadConversationSpanToolArguments, ResolveClaimSourceToolArguments, SearchCareerEpisodesToolArguments, SearchCareerHistoryToolArguments, SearchCareerMemoryToolArguments, ToolCall, ToolObservation, UpdateOwnerSettingsToolArguments, append_decision_observation, decision_observation_chars, project_action_center_arguments, project_calendar_arguments, project_job_intent_arguments, project_memory_amendment_arguments, project_memory_tombstone_arguments, project_email_arguments, project_interview_arguments, project_interview_preparation_arguments, project_job_research_arguments, project_mock_interview_arguments, project_mock_interview_result_arguments, project_open_job_search_arguments, project_restart_mock_interview_arguments, project_resume_arguments, project_saved_job_arguments
from career_agent.agent.conversation_span_presenter import render_conversation_span
from career_agent.agent.summary_text import DELIVERY_SUMMARY_LIMIT, MODEL_REPLY_LIMIT, clamp
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
    delivers_body_elsewhere,
    is_failed,
)
from career_agent.agent.tool_effects import ToolEffect, effect_for, replay_safe
from career_agent.storage.capability_confirmations import (
    CapabilityConfirmationExpiredError,
    CapabilityConfirmationInProgressError,
    CapabilityConfirmationSettledError,
    SQLiteCapabilityConfirmationStore,
    arguments_hash,
)
from career_agent.agent.main_agent_reducers import reduce_task_state
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
    ProgressEvent,
    PublicStreamEvent,
    ReportReadyEvent,
    StreamEventSink,
    TurnCompletedEvent,
    TurnFailedEvent,
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

OriginKind = Literal["model", "interaction", "workflow"]
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


TurnOrigin = ModelDecision | InteractionReceipt | RuntimeAction
"""How a turn came to exist.

The three ingresses are not three shapes of one decision. Typing them as one
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


class LoopControl(TypedDict, total=False):
    read_calls: int
    write_calls: int
    projection_refusals: int
    authorization_refusals: int
    fingerprints: tuple[str, ...]
    retryable_fingerprints: tuple[str, ...]
    retry_counts: dict[str, int]


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
            "memory_amendment_proposed",
            "memory_tombstone_proposed",
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

    def __init__(self, *, context_manager: ContextManager, decision_maker: DecisionMaker, tools: MainAgentToolRegistry, career_context_projector: CareerContextProjector | None = None, max_read_calls: int = DEFAULT_MAX_READ_CALLS, max_write_calls: int = DEFAULT_MAX_WRITE_CALLS, max_projection_refusals: int = DEFAULT_MAX_PROJECTION_REFUSALS, max_authorization_refusals: int = DEFAULT_MAX_AUTHORIZATION_REFUSALS, max_failure_retries: int = DEFAULT_MAX_FAILURE_RETRIES, owned_resources: tuple[Any, ...] = (), trace_recorder: TraceRecorder | None = None, action_execution_store: SQLiteActionExecutionStore | None = None, capability_confirmation_store: SQLiteCapabilityConfirmationStore | None = None, action_policy_epoch: int = ACTION_EXECUTION_POLICY_EPOCH, episode_reconciler: EpisodeReconciler | None = None) -> None:
        if max_read_calls < 1:
            raise ValueError("max_read_calls must be at least one")
        if max_write_calls < 1:
            raise ValueError("max_write_calls must be at least one")
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
        self._decision_maker = decision_maker
        self._tools = tools
        self._career_context_projector = career_context_projector
        self._episode_reconciler = episode_reconciler
        self._reconciled_users: set[str] = set()
        self._episode_reconcile_guard = Lock()
        self._episode_reconcile_locks: dict[str, Any] = {}
        self._max_read_calls = max_read_calls
        self._max_write_calls = max_write_calls
        self._max_projection_refusals = max_projection_refusals
        self._max_authorization_refusals = max_authorization_refusals
        self._max_failure_retries = max_failure_retries
        self._trace_recorder = trace_recorder
        self._action_execution_store = action_execution_store
        self._capability_confirmation_store = capability_confirmation_store
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

        Two ingresses arrive with the action in hand: a bound runtime-owned
        workflow, and an owner confirming an action their own rule stopped. Both
        skip ``decide`` for the same reason — there is nothing left to decide,
        and consulting the model would let it revise a choice that was already
        made (by the runtime's ownership rule, or by a person clicking confirm).
        """

        pending = state.get("pending", {})
        return (
            "authorize"
            if pending.get("runtime_owned") or pending.get("owner_confirmed")
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

    @classmethod
    def _emit_capability_started(cls, name: str) -> None:
        capability = cls._public_capability(name)
        labels = {
            "job_search": "正在处理岗位检索……",
            "job_research": "正在调研岗位相关业务信息……",
            "resume": "正在处理简历……",
            "application_tracking": "正在处理投递进展……",
            "interview": "正在处理面试任务……",
            "calendar": "正在准备日历操作……",
            "action_center": "正在整理待办事项……",
            "career_task": "正在执行职业任务……",
        }
        cls._emit(
            CapabilityStartedEvent(
                capability=capability,
                message=labels[capability],
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
    ) -> MainAgentTurnResult:
        """Run one committed turn and optionally publish presentation-only events.

        The sink is held outside graph state and checkpoints. A broken observer
        never gets authority to fail or mutate the business turn.
        """

        turn_id = uuid4().hex
        if request_id is not None:
            request_id = request_id.strip()
            if not request_id or len(request_id) > 200:
                raise ValueError("request_id must contain 1 to 200 characters")
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
        try:
            result = self._run_and_commit_turn(
                user_id=user_id,
                conversation_id=conversation_id,
                user_message=user_message,
                interaction_response=interaction_response,
            )
            self._record_turn(turn_id=turn_id, conversation_id=conversation_id, result=result)
            self._deliver_stream_events(
                result=result,
                turn_id=turn_id,
                conversation_id=conversation_id,
            )
            return result
        except Exception as error:
            self._invalidate_episode_reconciliation(user_id)
            self._record_turn_failed(turn_id=turn_id, conversation_id=conversation_id, error=error)
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
    ) -> None:
        if self._trace_recorder is None:
            return
        error_code = getattr(error, "code", None)
        if not isinstance(error_code, str) or not error_code.strip():
            error_code = "TURN_EXECUTION_FAILED"
        retryable = getattr(error, "retryable", None)
        if not isinstance(retryable, bool):
            retryable = None
        try:
            self._trace_recorder.record(
                turn_id,
                "turn_failed",
                "turn",
                outcome="failed",
                details={
                    "conversation_id": conversation_id,
                    "error_type": type(error).__name__,
                },
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

    def _run_and_commit_turn(
        self,
        *,
        user_id: str,
        conversation_id: str,
        user_message: str,
        interaction_response: InteractionResponse | None = None,
    ) -> MainAgentTurnResult:
        self._reconcile_episodes(user_id)
        routing_task = self._context_manager.get_task(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if interaction_response is not None:
            context = self._context_manager.load_for_turn(
                user_id=user_id,
                conversation_id=conversation_id,
                user_message=user_message,
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
            self._emit(ProgressEvent(stage="saving", message="正在保存本轮状态……"))
            self._context_manager.commit_turn(
                context=context,
                task=result.context.task,
                assistant_message=self._conversation_content(
                    result.tool_result,
                    screen=result.assistant_message,
                    composed=False,
                ),
                episode_drafts=drafts_from_tool_results(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    tool_results=result.tool_results
                    or ((result.tool_result,) if result.tool_result else ()),
                ),
                memory_scope_keys=result.career_memory_scope_keys,
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
            self._emit(ProgressEvent(stage="saving", message="正在保存本轮状态……"))
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
                )
            return result

        context = self._context_manager.load_for_turn(
            user_id=user_id,
            conversation_id=conversation_id,
            user_message=user_message,
        )
        try:
            result = self._run_loaded_context(context)
        except Exception as error:
            self._commit_interrupted_turn(context=context, error=error)
            raise
        # This input belonged to Main Agent even when its result hands future
        # turns to a workflow. Ownership is an ingress property, not something
        # that can be inferred from the task state after execution. The reply,
        # however, did come from the workflow: it is the run's first question,
        # withheld on the same grounds as every question after it.
        self._emit(ProgressEvent(stage="saving", message="正在保存本轮状态……"))
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
                episode_drafts=drafts_from_tool_results(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    tool_results=result.tool_results
                    or ((result.tool_result,) if result.tool_result else ()),
                ),
                memory_scope_keys=result.career_memory_scope_keys,
            )
        return result

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
            self._emit(
                ClientActionEvent(
                    action="open_url",
                    url=str(action.get("url", "")),
                    label=str(action.get("label", "打开岗位搜索页")),
                )
            )

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
            self._emit(interaction)
            self._emit(
                TurnSuspendedEvent(
                    turn_id=turn_id,
                    interaction_id=interaction.interaction_id,
                )
            )
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
            self._emit(
                ReportReadyEvent(
                    kind=reference.kind,
                    resource_id=reference.resource_id,
                    status_at_delivery=reference.status_at_delivery,
                    anchored_by_other_job=reference.anchored_by_other_job,
                )
            )
        self._emit(TurnCompletedEvent(turn_id=turn_id))
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
                "memory_amendment_proposed",
                "memory_tombstone_proposed",
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
                    update={"task": reduce_task_state(task, result)}
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

    def _run_loaded_context(self, context: MainAgentContext) -> MainAgentTurnResult:
        state = self._graph.invoke(
            {
                "context": context,
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

    def _decide(self, state: MainAgentState) -> MainAgentState:
        self._emit(ProgressEvent(stage="deciding", message="正在判断下一步操作……"))
        context = state["context"]
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
        self._record_trace_event(
            "memory_context_observed",
            "main_agent_decide",
            outcome="succeeded",
            details=memory_context_observation(
                context,
                career_memory_enabled=self._career_context_projector is not None,
            ),
        )
        try:
            decision = self._decision_maker.decide(context, schemas)
        except Exception as error:
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
        return {"decision": decision}

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
        if self._career_context_projector is None:
            return {}
        context = state["context"]
        memory = self._career_context_projector.project(
            user_id=context.profile.user_id,
            query=context.user_message,
        )
        return {
            "context": context.model_copy(update={"career_memory": memory}),
            "career_memory_scope_keys": tuple(
                dict.fromkeys(
                    binding.entry_id for binding in memory.telemetry_bindings
                )
            ),
        }

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
        used = control.get("read_calls", 0) if effect == "READ" else control.get("write_calls", 0)
        limit = self._max_read_calls if effect == "READ" else self._max_write_calls
        if used >= limit:
            return self._authorization_refusal(
                state,
                name=name,
                reason=(
                    f"本轮 {effect} 委派预算已经用完；请基于已有结果作答，"
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
                "effect": effect,
                "arguments": arguments,
            },
        }

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

        if self._capability_confirmation_store is None:
            return self._authorization_refusal(
                state,
                name=name,
                reason="你设置了这个操作需要先经你确认，但本次部署无法保存待确认动作。",
                next_action="告诉用户这个操作被设置为需要确认，但当前无法记录确认请求。",
            )
        context = state["context"]
        display_summary = self._owner_confirmation_summary(
            context=context, name=name, arguments=arguments
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
            message=f"{display_summary}\n你设置了此操作需要确认。是否执行？",
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
            return "准备更新持久设置：" + "；".join(changes) + "。"
        return f"准备执行 {name}。"

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
            result = self._act_request_anchored_write(state)
        else:
            result = self._invoke_pending(pending)
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
                    context, pending.get("reducer_result", result)
                )
            if result.state in {
                "career_memory_amended",
                "memory_tombstoned",
                "memory_tombstone_cleanup_incomplete",
            }:
                refreshed = self._context_manager.load_for_turn(
                    user_id=context.profile.user_id,
                    conversation_id=context.conversation_id,
                    user_message=context.user_message,
                )
                refresh_updates: dict[str, Any] = {"task": updated.task}
                updated = refreshed.model_copy(update=refresh_updates)
            effect = pending["effect"]
            budget_key = "read_calls" if effect == "READ" else "write_calls"
            control[budget_key] = control.get(budget_key, 0) + 1
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
        return {
            "context": updated,
            "tool_results": tool_results,
            "control": control,
            "artifact_ids": artifact_ids,
        }

    @staticmethod
    def _after_observe(
        state: MainAgentState,
    ) -> Literal["decide", "present", "interrupt"]:
        pending = state["pending"]
        result = pending["result"]
        # A capability that already produced a complete, bound interaction
        # contract does not need an LLM to paraphrase or rediscover its prompt.
        if result.disposition == "interaction_required":
            return "interrupt"
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
            return {"assistant_message": MainAgentRuntime._assistant_message(result)}
        return {"assistant_message": "本轮可执行步骤已达到上限，请确认后继续。"}

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
            if not condenses_message(result.state):
                continue
            if delivers_body_elsewhere(result.state):
                continue
            rendered = MainAgentRuntime._assistant_message(result)
            # A repeated identical render adds nothing but length; two different
            # jobs rendering differently must both survive.
            if rendered and rendered not in bodies:
                bodies.append(rendered)
        return "\n\n".join(bodies)

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
        * Summary row + message body: the full text is deliberately live-only,
          while the transcript always keeps the tool's deterministic receipt.
          Daily Brief and Resume Analysis use this shape because their bodies
          have nowhere else to go but should not occupy the recent window
          indefinitely. Keeping a prefix of the body would look complete while
          silently favouring whichever sections happened to come first.

        Therefore live equality is an invariant only for the first two shapes,
        not for every call to this function. In the third shape, refreshing is
        expected to replace the ephemeral body with its bounded historical row.

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
            "propose_memory_tombstone",
            "confirm_memory_tombstone",
        }:
            return project_memory_tombstone_arguments(context, name, arguments)
        if name in {
            "propose_memory_amendment",
            "confirm_memory_amendment",
        }:
            return project_memory_amendment_arguments(context, name, arguments)
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
        context: MainAgentContext, result: ToolObservation
    ) -> MainAgentContext:
        return context.model_copy(
            update={"task": reduce_task_state(context.task, result)}
        )
