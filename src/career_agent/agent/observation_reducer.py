from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    DecisionObservation,
    MainAgentContext,
    ToolObservation,
    append_decision_observation,
)
from career_agent.agent.main_agent_tools import MainAgentToolOutput
from career_agent.agent.main_state import LoopControl, MainAgentState, PendingAction
from career_agent.agent.tool_effects import is_external_write


_REFRESH_STATES = frozenset(
    {
        "career_memory_amended",
        "memory_tombstoned",
        "memory_tombstone_cleanup_incomplete",
        "free_text_preference_confirmed",
        "free_text_preference_confirmed_structured_proposed",
        "career_fact_confirmed",
        "working_notes_stale",
        "working_notes_updated",
    }
)

_MOCK_INTERVIEW_TOOLS = frozenset(
    {
        "start_mock_interview",
        "restart_mock_interview",
        "handle_mock_interview_input",
        "retry_mock_interview",
    }
)

_MOCK_INTERVIEW_SELECTION_STATES = frozenset(
    {
        "mock_interview_resume_choice_required",
        "mock_interview_job_choice_required",
    }
)


class ObservationReductionHost(Protocol):
    """Runtime-owned reducers and renderers used by observation reduction."""

    def _emit_trace(
        self,
        event_type: str,
        stage: str,
        *,
        error_code: str | None = None,
        recoverable: bool | None = None,
    ) -> None: ...

    def _update_mock_interview_task(
        self, context: MainAgentContext, result: ToolObservation
    ) -> MainAgentContext: ...

    def _update_atomic_task(
        self,
        context: MainAgentContext,
        result: ToolObservation,
        *,
        now: datetime | None = None,
    ) -> MainAgentContext: ...

    def _tool_call_fingerprint(self, decision: AgentDecision) -> str: ...

    def _tool_observation(
        self,
        name: str,
        result: MainAgentToolOutput,
        arguments: dict[str, Any] | None = None,
    ) -> DecisionObservation: ...


class ObservationReducer:
    """Fold one capability result into graph control and conversation context."""

    def __init__(
        self,
        *,
        host: ObservationReductionHost,
        context_manager: ContextManager,
    ) -> None:
        self._host = host
        self._context_manager = context_manager

    def reduce(self, state: MainAgentState) -> MainAgentState:
        context = state["context"]
        pending = state["pending"]
        result = pending["result"]
        capability_name = pending["name"]
        control = dict(state.get("control", {}))
        synthetic_kind = pending.get("synthetic_kind")

        if result.disposition == "failed":
            self._host._emit_trace(
                "capability_failed",
                capability_name,
                error_code=(
                    str(result.payload.get("error_code"))
                    if result.payload.get("error_code") is not None
                    else "CAPABILITY_FAILED"
                ),
                recoverable=(
                    bool(result.payload.get("retryable"))
                    if "retryable" in result.payload
                    else None
                ),
            )

        if synthetic_kind == "confirmation":
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
            updated = self._reduce_task(context, pending, result)
            if result.state in _REFRESH_STATES:
                refreshed = self._context_manager.load_for_turn(
                    user_id=context.profile.user_id,
                    conversation_id=context.conversation_id,
                    user_message=context.stored_user_message(),
                )
                updated = refreshed.model_copy(
                    update={
                        "task": updated.task,
                        "attached_resumes": context.attached_resumes,
                        "attached_jobs": context.attached_jobs,
                    }
                )
            self._account_for_call(state, pending, result, control)

        decision = state.get("decision")
        written = (
            decision.tool_call.arguments
            if decision is not None and decision.tool_call is not None
            else {}
        )
        observation = self._host._tool_observation(
            capability_name, result, written
        )
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
            career_memory_scope_keys = (*career_memory_scope_keys, result_scope_key)

        return {
            "context": updated,
            "tool_results": tool_results,
            "control": control,
            "artifact_ids": artifact_ids,
            "career_memory_scope_keys": career_memory_scope_keys,
        }

    def _reduce_task(
        self,
        context: MainAgentContext,
        pending: PendingAction,
        result: MainAgentToolOutput,
    ) -> MainAgentContext:
        if (
            result.tool_name in _MOCK_INTERVIEW_TOOLS
            and result.state not in _MOCK_INTERVIEW_SELECTION_STATES
        ):
            return self._host._update_mock_interview_task(context, result)
        return self._host._update_atomic_task(
            context,
            pending.get("reducer_result", result),
            now=self._context_manager.now(),
        )

    def _account_for_call(
        self,
        state: MainAgentState,
        pending: PendingAction,
        result: MainAgentToolOutput,
        control: LoopControl,
    ) -> None:
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
        if effect == "WRITE" and pending["name"] == "analyze_job":
            control["job_analysis_write_used"] = True

        fingerprint = self._host._tool_call_fingerprint(state["decision"])
        fingerprints = control.get("fingerprints", ())
        if fingerprint not in fingerprints:
            control["fingerprints"] = (*fingerprints, fingerprint)
        retryable_fingerprints = tuple(control.get("retryable_fingerprints", ()))
        if result.disposition == "failed" and result.payload.get("retryable") is True:
            if fingerprint not in retryable_fingerprints:
                retryable_fingerprints = (*retryable_fingerprints, fingerprint)
        else:
            retryable_fingerprints = tuple(
                item for item in retryable_fingerprints if item != fingerprint
            )
        control["retryable_fingerprints"] = retryable_fingerprints
