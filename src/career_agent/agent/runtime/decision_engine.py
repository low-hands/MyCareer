from __future__ import annotations

from threading import Event
from time import perf_counter
from collections.abc import Callable
from typing import Any, Literal

import hashlib
import json

from career_agent.agent.context.manager import ContextManager
from career_agent.agent.runtime.decision_attempts import (
    DecisionAttempt,
    observing_decision_attempts,
)
from career_agent.agent.runtime.decision_messages import decision_context_chars
from career_agent.agent.capabilities.catalog import ToolProfile
from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.contracts.decisions import (
    AgentDecision,
    DecisionMaker,
)
from career_agent.agent.contracts.observations import decision_observation_chars
from career_agent.agent.contracts.task_state import ConversationTaskState
from career_agent.agent.capabilities.profiles import profile_schemas, profile_tools
from career_agent.agent.capabilities.reachability import STATE_GATED_TOOLS, reachable
from career_agent.agent.capabilities.effects import is_notes_guarded
from career_agent.agent.runtime.turn_coordinator import STREAM_SINK
from career_agent.agent.middleware.working_notes import working_notes_only_tokens
from career_agent.harness.memory_telemetry import memory_context_observation
from career_agent.harness.observability import conversation_trace_key
from career_agent.harness.graph_routing import GraphRoutingPolicy
from career_agent.harness.streaming import ProgressEvent, PublicStreamEvent, StreamEventSink
from career_agent.agent.capabilities.registry import MainAgentToolRegistry
from career_agent.agent.runtime.state import MainAgentState


class DecisionEngine:
    """Offer reachable schemas, invoke the orchestrator model, and trace its choice."""

    _RETRY_REASONS: dict[str, str] = {
        "MAIN_AGENT_TRANSPORT_ERROR": "上一次请求超时或连接中断",
        "MAIN_AGENT_RATE_LIMITED": "上一次请求被限流",
    }

    def __init__(
        self,
        *,
        emit: Callable[[PublicStreamEvent], None],
        decision_heartbeat: Callable[[StreamEventSink | None], Event],
        record_trace_event: Callable[..., None],
        project_atomic_tool_arguments: Callable[
            [MainAgentContext, str, dict[str, Any]], dict[str, Any]
        ],
        project_workflow_arguments: Callable[
            [MainAgentContext, str, dict[str, Any]], dict[str, Any]
        ],
        context_manager: ContextManager,
        decision_maker_provider: Callable[[], DecisionMaker],
        tools: MainAgentToolRegistry,
        career_memory_enabled: bool,
    ) -> None:
        self._emit = emit
        self._decision_heartbeat = decision_heartbeat
        self._record_trace_event = record_trace_event
        self._project_atomic_tool_arguments = project_atomic_tool_arguments
        self._project_workflow_arguments = project_workflow_arguments
        self._context_manager = context_manager
        self._decision_maker_provider = decision_maker_provider
        self._tools = tools
        self._career_memory_enabled = career_memory_enabled
        self._registered_tool_schemas: tuple[dict[str, Any], ...] | None = None
        self._profile_tool_schemas: dict[
            tuple[ToolProfile, tuple[str, ...] | None],
            tuple[dict[str, Any], ...],
        ] = {}

    def registered_schemas(self) -> tuple[dict[str, Any], ...]:
        if self._registered_tool_schemas is None:
            self._registered_tool_schemas = tuple(self._tools.schemas())
        return self._registered_tool_schemas

    def tool_schemas(
        self,
        profile: ToolProfile,
        task: ConversationTaskState | None = None,
    ) -> tuple[dict[str, Any], ...]:
        registered = self.registered_schemas()
        reachable_names = (
            tuple(
                sorted(
                    str(schema.get("function", {}).get("name"))
                    for schema in registered
                    if schema.get("function", {}).get("name")
                    in profile_tools(profile)
                    and task is not None
                    and (
                        str(schema.get("function", {}).get("name"))
                        not in STATE_GATED_TOOLS
                        or reachable(
                            str(schema.get("function", {}).get("name")), task
                        )
                    )
                )
            )
            if task is not None
            else None
        )
        key = (profile, reachable_names)
        cached = self._profile_tool_schemas.get(key)
        if cached is None:
            cached = profile_schemas(profile, registered, task)
            self._profile_tool_schemas[key] = cached
        return cached

    def decide(self, state: MainAgentState) -> MainAgentState:
        self._emit(
            ProgressEvent(stage="deciding", message="正在判断下一步操作……")
        )
        context = state["context"]
        control = dict(state.get("control", {}))
        if not control.get("episodes_marked"):
            self._context_manager.mark_episodes_projected(
                user_id=context.profile.user_id,
                context=context,
            )
            control["episodes_marked"] = True

        schemas = self.tool_schemas(context.task.tool_profile, context.task)
        decision_maker = self._decision_maker_provider()
        details = self._trace_details(context, schemas, decision_maker)
        started = perf_counter()
        self._record_trace_event(
            "model_attempt",
            "main_agent_decide",
            outcome="started",
            details=details,
            model_call_category="orchestrator_decision",
        )

        def on_attempt(attempt: DecisionAttempt) -> None:
            message = self._attempt_message(attempt)
            if message is not None:
                self._emit(ProgressEvent(stage="deciding", message=message))

        heartbeat = self._decision_heartbeat(STREAM_SINK.get())
        try:
            with observing_decision_attempts(on_attempt):
                decision = decision_maker.decide(context, schemas)
        except Exception as error:
            self._record_memory_context(context, ())
            failure_details = self._with_cache_metrics(details, decision_maker)
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

        note_only_tokens = self.note_only_tokens(context, decision)
        self._record_memory_context(context, note_only_tokens)
        decision_details = self._with_cache_metrics(
            {**details, "decision_action": decision.action},
            decision_maker,
        )
        if decision.tool_call is not None:
            decision_details.update(
                {
                    "tool_name": decision.tool_call.name,
                    "tool_arguments_fingerprint": hashlib.sha256(
                        self.tool_call_fingerprint(decision).encode("utf-8")
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

    def note_only_tokens(
        self, context: MainAgentContext, decision: AgentDecision
    ) -> tuple[str, ...]:
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

    @staticmethod
    def tool_call_fingerprint(decision: AgentDecision) -> str:
        if decision.tool_call is None:
            return ""
        return json.dumps(
            {
                "name": decision.tool_call.name,
                "arguments": decision.tool_call.arguments,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def route(
        state: MainAgentState,
    ) -> Literal["authorize", "present", "interrupt"]:
        return GraphRoutingPolicy.route_decision(state)

    @classmethod
    def _attempt_message(cls, attempt: DecisionAttempt) -> str | None:
        if attempt.attempt <= 1:
            return None
        code = attempt.previous_error_code or ""
        reason = cls._RETRY_REASONS.get(
            code,
            "上一次请求被模型服务拒绝"
            if code.startswith("MAIN_AGENT_REJECTED_")
            else "上一次请求失败",
        )
        return (
            f"{reason}，正在重新判断（第 {attempt.attempt}/{attempt.max_attempts} 次，"
            f"已等待 {int(attempt.elapsed_seconds)} 秒）……"
        )

    def _trace_details(
        self,
        context: MainAgentContext,
        schemas: tuple[dict[str, Any], ...],
        decision_maker: DecisionMaker,
    ) -> dict[str, Any]:
        details = {
            "conversation_id": context.conversation_id,
            "conversation_key": conversation_trace_key(
                context.profile.user_id,
                context.conversation_id,
            ),
            "context_chars": decision_context_chars(context),
            "observation_chars": decision_observation_chars(
                context.tool_observations
            ),
            "observation_count": len(context.tool_observations),
            "tool_profile": context.task.tool_profile,
            "offered_tool_count": len(schemas),
            "tool_schema_chars": len(
                json.dumps(schemas, ensure_ascii=False, sort_keys=True)
            ),
        }
        cache_configuration = getattr(
            decision_maker, "cache_configuration", None
        )
        if callable(cache_configuration):
            details.update(cache_configuration())
        return details

    @staticmethod
    def _with_cache_metrics(
        details: dict[str, Any], decision_maker: DecisionMaker
    ) -> dict[str, Any]:
        enriched = dict(details)
        consume_cache_metrics = getattr(
            decision_maker, "consume_cache_metrics", None
        )
        if callable(consume_cache_metrics):
            enriched.update(consume_cache_metrics())
        return enriched

    def _record_memory_context(
        self,
        context: MainAgentContext,
        note_only_tokens: tuple[str, ...],
    ) -> None:
        self._record_trace_event(
            "memory_context_observed",
            "main_agent_decide",
            outcome="succeeded",
            details=memory_context_observation(
                context,
                career_memory_enabled=self._career_memory_enabled,
                working_notes_only_tokens=len(note_only_tokens),
                working_notes_only_argument=bool(note_only_tokens),
            ),
        )
