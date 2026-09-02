from __future__ import annotations

import json
from contextvars import ContextVar
from time import perf_counter
from typing import Any, Literal, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.career_context import CareerContextProjector
from career_agent.agent.answer_writer import (
    AnswerCompositionRequest,
    AnswerWriter,
)
from career_agent.agent.main_agent_contracts import AgentDecision, ConversationTaskState, DECISION_OBSERVATION_BODY_LIMIT, DecisionMaker, DecisionObservation, MainAgentContext, MAX_DECISION_OBSERVATIONS, ToolCall, ToolObservation, append_decision_observation, decision_observation_chars, project_action_center_arguments, project_calendar_arguments, project_job_intent_arguments, project_email_arguments, project_interview_arguments, project_interview_preparation_arguments, project_job_research_arguments, project_mock_interview_arguments, project_mock_interview_result_arguments, project_open_job_search_arguments, project_restart_mock_interview_arguments, project_resume_arguments, project_saved_job_arguments
from career_agent.agent.summary_text import DELIVERY_SUMMARY_LIMIT, clamp
from career_agent.harness.observability import (
    EventType,
    ModelCallCategory,
    TraceRecorder,
)
from career_agent.agent.delivery_policy import (
    condenses_message,
    delivers_body_elsewhere,
    response_type_for,
    uses_answer_writer,
)
from career_agent.agent.tool_reachability import reroutable
from career_agent.agent.tool_effects import ToolEffect, effect_for
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
    resume_analysis_confirmation_event,
)

_STREAM_SINK: ContextVar[StreamEventSink | None] = ContextVar(
    "main_agent_stream_sink",
    default=None,
)

_TRACE_CONTEXT: ContextVar[tuple[TraceRecorder, str] | None] = ContextVar(
    "main_agent_trace_context",
    default=None,
)

DEFAULT_MAX_READ_CALLS = 6
DEFAULT_MAX_WRITE_CALLS = 1
DEFAULT_MAX_PROJECTION_REFUSALS = 2
DEFAULT_MAX_AUTHORIZATION_REFUSALS = 1
DEFAULT_MAX_FAILURE_RETRIES = 2


class PendingAction(TypedDict, total=False):
    name: str
    kind: Literal["atomic_tool", "workflow"]
    effect: ToolEffect
    arguments: dict[str, Any]
    result: MainAgentToolOutput
    synthetic_kind: Literal["projection", "authorization"]


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


class MainAgentTurnResult:
    def __init__(self, *, decision: AgentDecision, context: MainAgentContext, assistant_message: str, tool_result: MainAgentToolOutput | None = None, tool_results: tuple[MainAgentToolOutput, ...] = (), artifacts: tuple[ResumeArtifactDelivery, ...] = (), content_streamed: bool = False, delegated_read_count: int = 0, delegated_write_count: int = 0) -> None:
        self.decision = decision
        self.context = context
        self.assistant_message = assistant_message
        self.tool_result = tool_result
        self.tool_results = tool_results
        self.artifacts = artifacts
        self.content_streamed = content_streamed
        self.delegated_read_count = delegated_read_count
        self.delegated_write_count = delegated_write_count


class MainAgentRuntime:
    _INTERACTION_RENDERER_STATES = frozenset(
        {
            "calendar_approval_required",
            "email_events_pending",
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

    def __init__(self, *, context_manager: ContextManager, decision_maker: DecisionMaker, tools: MainAgentToolRegistry, career_context_projector: CareerContextProjector | None = None, answer_writer: AnswerWriter | None = None, max_read_calls: int = DEFAULT_MAX_READ_CALLS, max_write_calls: int = DEFAULT_MAX_WRITE_CALLS, max_projection_refusals: int = DEFAULT_MAX_PROJECTION_REFUSALS, max_authorization_refusals: int = DEFAULT_MAX_AUTHORIZATION_REFUSALS, max_failure_retries: int = DEFAULT_MAX_FAILURE_RETRIES, owned_resources: tuple[Any, ...] = (), trace_recorder: TraceRecorder | None = None) -> None:
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
        self._answer_writer = answer_writer
        self._max_read_calls = max_read_calls
        self._max_write_calls = max_write_calls
        self._max_projection_refusals = max_projection_refusals
        self._max_authorization_refusals = max_authorization_refusals
        self._max_failure_retries = max_failure_retries
        self._trace_recorder = trace_recorder
        self._owned_resources = owned_resources
        self._closed = False

        graph = StateGraph(MainAgentState)
        graph.add_node("hydrate", self._hydrate_career_context)
        graph.add_node("decide", self._decide)
        graph.add_node("authorize", self._authorize)
        graph.add_node("act", self._act)
        graph.add_node("observe", self._observe)
        graph.add_node("present", self._present)
        graph.add_node("interrupt", self._interrupt)
        graph.add_edge(START, "hydrate")
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
        interaction_response: InteractionResponse | None = None,
        event_sink: StreamEventSink | None = None,
    ) -> MainAgentTurnResult:
        """Run one committed turn and optionally publish presentation-only events.

        The sink is held outside graph state and checkpoints. A broken observer
        never gets authority to fail or mutate the business turn.
        """

        turn_id = uuid4().hex
        sink_token = _STREAM_SINK.set(event_sink)
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
        context = _TRACE_CONTEXT.get()
        if context is None:
            return
        recorder, run_id = context
        try:
            recorder.record(
                run_id,
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
        except Exception:
            return

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
                        result.tool_result.state if result.tool_result is not None else result.decision.action
                    ),
                },
            )
        except Exception:
            # Telemetry is best-effort; a recorder that cannot write must not
            # roll back a turn whose business effects are already durable.
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

    def _run_and_commit_turn(
        self,
        *,
        user_id: str,
        conversation_id: str,
        user_message: str,
        interaction_response: InteractionResponse | None = None,
    ) -> MainAgentTurnResult:
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
            result = self._run_interaction_response(
                context=context,
                conversation_id=conversation_id,
                response=interaction_response,
            )
            self._emit(ProgressEvent(stage="saving", message="正在保存本轮状态……"))
            self._context_manager.commit_turn(
                context=context,
                task=result.context.task,
                assistant_message=self._conversation_content(
                    result.tool_result,
                    screen=result.assistant_message,
                    composed=False,
                ),
            )
            return result

        if self._owns_next_turn(routing_task):
            context = self._context_manager.load_for_workflow_turn(
                user_id=user_id,
                conversation_id=conversation_id,
                task=routing_task,
            )
            self._emit_capability_started("start_mock_interview")
            result = self._run_active_mock_interview(
                context=context,
                user_message=user_message,
            )
            if result.tool_result is not None:
                self._emit_capability_completed(
                    "start_mock_interview", result.tool_result.state
                )
            self._stream_answer_if_eligible(
                result=result,
                conversation_id=conversation_id,
                user_request=user_message,
            )
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
                        screen=result.assistant_message,
                        composed=result.content_streamed,
                    ),
                    assistant_resource_ref=(
                        result.tool_result.resource_ref
                        if result.tool_result is not None
                        else None
                    ),
                )
            return result

        context = self._context_manager.load_for_turn(
            user_id=user_id,
            conversation_id=conversation_id,
            user_message=user_message,
        )
        result = self._run_loaded_context(context)
        self._stream_answer_if_eligible(
            result=result,
            conversation_id=conversation_id,
            user_request=user_message,
        )
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
                    screen=result.assistant_message,
                    composed=result.content_streamed,
                ),
                assistant_resource_ref=(
                    result.tool_result.resource_ref
                    if result.tool_result is not None
                    else None
                ),
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
                    screen=result.assistant_message,
                    composed=False,
                )
                if result.tool_result is not None
                and self._has_backed_card(result.tool_result)
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
        if result.tool_result is not None and result.tool_result.resource_ref is not None:
            self._emit(
                ReportReadyEvent(
                    kind=result.tool_result.resource_ref.kind,
                    resource_id=result.tool_result.resource_ref.resource_id,
                    status_at_delivery=(
                        result.tool_result.resource_ref.status_at_delivery
                    ),
                    anchored_by_other_job=(
                        result.tool_result.resource_ref.anchored_by_other_job
                    ),
                )
            )
        self._emit(TurnCompletedEvent(turn_id=turn_id))

    def _stream_answer_if_eligible(
        self,
        *,
        result: MainAgentTurnResult,
        conversation_id: str,
        user_request: str,
    ) -> None:
        if self._answer_writer is None or not self._should_use_answer_writer(result):
            return
        if self._interaction_event(result=result, conversation_id=conversation_id):
            return

        # Report-shaped turns deliver their body through the card, so the
        # writer is asked for a bounded delivery summary instead of a rewrite.
        # The instruction alone is not enough — it is a request to a model about
        # its own output length — so the stream is cut at the same number below.
        limit = (
            DELIVERY_SUMMARY_LIMIT
            if result.tool_result is not None
            and self._has_backed_card(result.tool_result)
            else None
        )
        request = AnswerCompositionRequest(
            response_type=self._answer_response_type(result),
            user_request=user_request,
            grounded_draft=result.assistant_message,
            max_chars=limit,
            # "Preserve every factual value" cannot be obeyed alongside a
            # character ceiling when grounded_draft is a whole report. Asking
            # for both left the writer to choose which instruction to break.
            required_rules=(
                (
                    "Do not add facts that are absent from grounded_draft.",
                    "Every fact you do state must be accurate and keep the "
                    "uncertainty grounded_draft gave it.",
                    "Keep any warning, limitation, or staleness notice.",
                )
                if limit is not None
                else (
                    "Preserve every factual value and all explicit uncertainty from grounded_draft.",
                    "Do not add facts that are absent from grounded_draft.",
                )
            ),
        )
        self._emit(
            ProgressEvent(stage="presenting", message="正在生成最终回答……")
        )
        chunks: list[str] = []
        written = 0
        trace_details = {
            "grounded_draft_chars": len(request.grounded_draft),
            "user_request_chars": len(request.user_request),
            "max_chars": request.max_chars,
            "response_type": request.response_type,
        }
        started = perf_counter()
        self._record_trace_event(
            "model_attempt",
            "answer_writer",
            outcome="started",
            details=trace_details,
            model_call_category="writer",
        )
        try:
            for delta in self._answer_writer.stream(request):
                if not delta:
                    continue
                # The ellipsis is part of the budget, not an addition to it:
                # counting only the content let an answer that landed exactly
                # on the limit overflow it by the marker's own character.
                if limit is not None and written + len(delta) + 1 > limit:
                    # Cut on the delta that would cross, and mark the cut in the
                    # stream rather than only in storage: the reader has to see
                    # the same text the transcript will keep.
                    remainder = delta[: max(0, limit - written - 1)]
                    if remainder:
                        chunks.append(remainder)
                        self._emit(ContentDeltaEvent(delta=remainder))
                    chunks.append("…")
                    self._emit(ContentDeltaEvent(delta="…"))
                    break
                chunks.append(delta)
                written += len(delta)
                self._emit(ContentDeltaEvent(delta=delta))
        except AgentWorkerError as error:
            self._record_trace_event(
                "model_failed",
                "answer_writer",
                outcome="failed",
                duration_ms=int((perf_counter() - started) * 1000),
                details={**trace_details, "emitted_chars": sum(map(len, chunks))},
                error_code=error.code,
                error_detail=error.detail or type(error).__name__,
                recoverable=error.retryable,
                model_call_category="writer",
            )
            if chunks:
                # Some text has already reached the user. Falling back now
                # would append a second, contradictory answer to that prefix.
                raise
            return
        except Exception as error:
            self._record_trace_event(
                "model_failed",
                "answer_writer",
                outcome="failed",
                duration_ms=int((perf_counter() - started) * 1000),
                details={**trace_details, "emitted_chars": sum(map(len, chunks))},
                error_code="ANSWER_WRITER_FAILED",
                error_detail=type(error).__name__,
                model_call_category="writer",
            )
            raise
        self._record_trace_event(
            "model_succeeded",
            "answer_writer",
            outcome="succeeded",
            duration_ms=int((perf_counter() - started) * 1000),
            details={**trace_details, "emitted_chars": sum(map(len, chunks))},
            model_call_category="writer",
        )
        if not chunks:
            return
        result.assistant_message = "".join(chunks)
        result.content_streamed = True

    @staticmethod
    def _should_use_answer_writer(result: MainAgentTurnResult) -> bool:
        """Whether this turn's deliverable is one the writer should restate.

        Artifacts and questions are excluded here rather than in the registry:
        both are properties of the turn, not of the state it ended in. What the
        state decides — is this report-shaped — is a single registry lookup, so
        it can no longer diverge from the response type or from the durable-row
        rule that assumes the same answer.
        """
        if result.artifacts or result.decision.action == "ask_user":
            return False
        tool_result = result.tool_result
        if tool_result is None:
            return result.decision.action == "final" and bool(result.assistant_message)
        return uses_answer_writer(tool_result.state)

    @staticmethod
    def _answer_response_type(result: MainAgentTurnResult) -> str:
        tool_result = result.tool_result
        if tool_result is None:
            return "general"
        return response_type_for(tool_result.state)

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
            tool_result.state if tool_result is not None else result.decision.action,
        )

        if tool_result is not None:
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
        if result.decision.action == "ask_user":
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
        decision = AgentDecision(action="final", message=result.message)
        return MainAgentTurnResult(
            decision=decision,
            context=updated,
            assistant_message=self._assistant_message(result),
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
            decision=state["decision"],
            context=state["context"],
            assistant_message=state["assistant_message"],
            tool_result=tool_result,
            tool_results=state.get("tool_results", ()),
            artifacts=artifacts,
            delegated_read_count=control.get("read_calls", 0),
            delegated_write_count=control.get("write_calls", 0),
        )

    def _decide(self, state: MainAgentState) -> MainAgentState:
        self._emit(ProgressEvent(stage="deciding", message="正在判断下一步操作……"))
        context = state["context"]
        schemas = self._tools.schemas(context)
        context_chars = len(
            json.dumps(context.model_context(), ensure_ascii=False, sort_keys=True)
        )
        tool_schema_chars = len(
            json.dumps(schemas, ensure_ascii=False, sort_keys=True)
        )
        details = {
            "context_chars": context_chars,
            # This helper serializes the same exclude-none projection used by
            # model_context; the metric is the actual dynamic prompt growth.
            "observation_chars": decision_observation_chars(
                context.tool_observations
            ),
            "observation_count": len(context.tool_observations),
            "offered_tool_count": len(schemas),
            "tool_schema_chars": tool_schema_chars,
        }
        started = perf_counter()
        self._record_trace_event(
            "model_attempt",
            "main_agent_decide",
            outcome="started",
            details=details,
            model_call_category="orchestrator_decision",
        )
        try:
            decision = self._decision_maker.decide(context, schemas)
        except Exception as error:
            self._record_trace_event(
                "model_failed",
                "main_agent_decide",
                outcome="failed",
                duration_ms=int((perf_counter() - started) * 1000),
                details=details,
                error_code=getattr(error, "code", "ORCHESTRATOR_DECISION_FAILED"),
                error_detail=getattr(error, "detail", None) or type(error).__name__,
                recoverable=getattr(error, "retryable", None),
                model_call_category="orchestrator_decision",
            )
            raise
        self._record_trace_event(
            "model_succeeded",
            "main_agent_decide",
            outcome="succeeded",
            duration_ms=int((perf_counter() - started) * 1000),
            details={**details, "decision_action": decision.action},
            model_call_category="orchestrator_decision",
        )
        return {"decision": decision}

    def _run_active_mock_interview(
        self, *, context: MainAgentContext, user_message: str
    ) -> MainAgentTurnResult:
        session_id = context.task.run_id
        if session_id is None:
            raise ValueError("Active mock interview has no resumable session")
        if context.task.phase == "failed":
            # The answer for this turn is already durable; the step after it
            # failed. Re-drive from the store rather than treating this message
            # as a new answer, which the turn would reject as conflicting and
            # leave the candidate unable to leave the failed phase at all.
            result = self._tools.retry_mock_interview(
                user_id=context.profile.user_id,
                session_id=session_id,
            )
        else:
            result = self._tools.handle_mock_interview_input(
                user_id=context.profile.user_id,
                session_id=session_id,
                message=user_message,
            )
        updated = self._update_mock_interview_task(context, result)
        updated = updated.model_copy(
            update={
                "tool_observations": append_decision_observation(
                    updated.tool_observations,
                    self._tool_observation("start_mock_interview", result),
                )
            }
        )
        decision = AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name="start_mock_interview", arguments={}),
        )
        return MainAgentTurnResult(
            decision=decision,
            context=updated,
            assistant_message=self._assistant_message(result),
            tool_result=result,
            tool_results=(result,),
        )

    def _hydrate_career_context(self, state: MainAgentState) -> MainAgentState:
        if self._career_context_projector is None:
            return {}
        context = state["context"]
        memory = self._career_context_projector.project(
            user_id=context.profile.user_id,
            query=context.user_message,
        )
        return {"context": context.model_copy(update={"career_memory": memory})}

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
            },
        }

    def _authorize(self, state: MainAgentState) -> MainAgentState:
        """Project and gate one model-selected action without choosing its successor."""

        decision = state["decision"]
        if decision.tool_call is None:
            raise ValueError("tool_call action requires tool_call arguments")
        name = decision.tool_call.name
        kind = self._tools.capability_kind(name)
        effect = effect_for(name)
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
                next_action="finish_or_ask_to_continue",
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
                    next_action="use_existing_observation_or_change_arguments",
                )
            if retries >= self._max_failure_retries:
                return self._authorization_refusal(
                    state,
                    name=name,
                    reason="相同失败调用已经达到本轮重试上限。",
                    next_action="explain_failure_or_ask_to_continue",
                )
            retry_counts[fingerprint] = retries + 1
            control = {**control, "retry_counts": retry_counts}
        try:
            arguments = (
                self._project_atomic_tool_arguments(
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
                },
            }
        return {
            "authorization_route": "act",
            "control": control,
            "pending": {
                "name": name,
                "kind": kind,
                "effect": effect,
                "arguments": arguments,
            },
        }

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
        result = (
            self._tools.invoke_atomic_tool(name, arguments)
            if pending["kind"] == "atomic_tool"
            else self._tools.invoke_workflow(name, arguments)
        )
        self._emit_capability_completed(name, result.state)
        return {"pending": {**pending, "result": result}}

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
        returns as an ordinary result, and which way the turn goes from there is
        told by whether any re-routable candidates exist in task state:

        - ``needs_user`` when no candidates can possibly resolve the refusal —
          nothing to list, nothing to point at; only the user can supply it.
        - ``invalid_input`` when candidates for the named capability exist, so
          the model can list or re-select them in the same turn instead of
          bouncing the user.
        """
        return ToolObservation(
            tool_name=name,
            state="invalid_input",
            message=f"这步暂时做不到：{error}。",
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
        if synthetic_kind is not None:
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
            }:
                updated = self._update_mock_interview_task(context, result)
            else:
                updated = self._update_atomic_task(context, result)
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
        observation = self._tool_observation(capability_name, result)
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
        if synthetic_kind is None:
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
        # Projection refusals normally return to the model so it can re-select
        # or ask the user. The second refusal is a control-limit exit, not a
        # semantic successor chosen from a capability-name table.
        if result.state == "invalid_input":
            if not reroutable(pending["name"], state["context"].task):
                return "present"
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
        """Deliver a final decision without letting model prose replace evidence."""

        decision = state["decision"]
        pending = state.get("pending", {})
        pending_result = pending.get("result")
        if (
            decision.action == "tool_call"
            and pending.get("synthetic_kind") == "projection"
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
        if result is not None:
            # Deliberately retain the authoritative presenter override. Letting
            # a post-tool model message coexist with evidence is the separate
            # presentation-composition item (070/F), not part of loop routing.
            return {"assistant_message": MainAgentRuntime._assistant_message(result)}
        if decision.action == "final" and decision.message:
            return {"assistant_message": decision.message}
        return {"assistant_message": "本轮可执行步骤已达到上限，请确认后继续。"}

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
    def _tool_observation(name: str, result: MainAgentToolOutput) -> DecisionObservation:
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
                facts=MainAgentRuntime._decision_facts(result),
                next_action=result.next_action,
            )
        return DecisionObservation(
            tool_name=name,
            state=result.state,
            message=receipt,
            body=body,
            facts=MainAgentRuntime._decision_facts(result),
            next_action=result.next_action,
        )

    @staticmethod
    def _decision_facts(result: MainAgentToolOutput) -> dict[str, bool | int | str]:
        """Project only explicitly approved scalar facts into the model context.

        This is deliberately state-keyed instead of accepting a handler-owned
        ``facts`` dict. Adding payload fields must never silently expand the
        decision prompt. ``waiting`` currently means an open Daily Brief item
        without a due date; the domain has no separate waiting bucket yet.
        """
        payload = result.payload
        if (
            isinstance(result, ToolObservation)
            and result.disposition == "failed"
            and type(payload.get("retryable")) is bool
        ):
            return {"retryable": payload["retryable"]}
        if result.state == "daily_brief_ready":
            buckets = {
                key: payload.get(key)
                for key in ("overdue", "due_today", "no_due_date")
            }
            if all(isinstance(value, (list, tuple)) for value in buckets.values()):
                return {
                    "overdue": len(buckets["overdue"]),
                    "due_today": len(buckets["due_today"]),
                    "waiting": len(buckets["no_due_date"]),
                }
        if result.state == "resume_analysis_ready":
            records = payload.get("records")
            clarifications = payload.get("clarification_questions")
            warnings = payload.get("warnings")
            if all(
                isinstance(value, (list, tuple))
                for value in (records, clarifications, warnings)
            ):
                return {
                    "record_count": len(records),
                    "clarification_count": len(clarifications),
                    "has_warnings": bool(warnings),
                }
        if result.state == "job_research_ready":
            research = payload.get("research")
            findings = research.get("findings") if isinstance(research, dict) else None
            cached = payload.get("cached")
            status = payload.get("status")
            if (
                isinstance(cached, bool)
                and isinstance(findings, (list, tuple))
                and status in {"current", "outdated", "superseded"}
            ):
                return {
                    "cached": cached,
                    "finding_count": len(findings),
                    "status": status,
                }
        return {}

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
            return clamp(screen) if composed else result.message
        return result.message

    @staticmethod
    def _assistant_message(result: MainAgentToolOutput) -> str:
        if result.state == "saved_job_ready":
            snapshot = result.payload.get("jd_snapshot")
            if isinstance(snapshot, dict):
                content = snapshot.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
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
        if name == "open_job_search":
            return project_open_job_search_arguments(context, arguments)
        if name in {
            "propose_job_intent",
            "confirm_job_intent",
        }:
            return project_job_intent_arguments(context, name, arguments)
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
