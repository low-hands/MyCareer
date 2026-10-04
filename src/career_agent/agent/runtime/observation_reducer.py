from __future__ import annotations

from collections.abc import Callable
from typing import Any

from career_agent.agent.context.manager import ContextManager
from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.contracts.decisions import AgentDecision
from career_agent.agent.contracts.observations import (
    DecisionObservation,
    ToolObservation,
    append_decision_observation,
)
from career_agent.agent.capabilities.registry import MainAgentToolOutput
from career_agent.agent.runtime.state import LoopControl, MainAgentState, PendingAction
from career_agent.agent.capabilities.effects import is_external_write
from career_agent.agent.presentation.delivery_policy import condenses_message
from career_agent.agent.presentation.result_presenter import ResultPresenter
from career_agent.agent.runtime.observability import RuntimeObservability
from career_agent.agent.support.summary_text import clamp
from career_agent.agent.contracts.observations import DECISION_OBSERVATION_BODY_LIMIT


def tool_observation(
    name: str,
    result: MainAgentToolOutput,
    arguments: dict[str, Any] | None = None,
) -> DecisionObservation:
    """Project a capability result into the model's bounded observation."""

    receipt = clamp(result.message) or "工具已返回，但没有提供结果摘要。"
    body = None
    if condenses_message(result.state):
        rendered = ResultPresenter.present(
            result,
            report_degraded=RuntimeObservability.emit_trace,
        )
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


class ObservationReducer:
    """Fold one capability result into graph control and conversation context."""

    def __init__(
        self,
        *,
        context_manager: ContextManager,
        emit_trace: Callable[..., None],
        update_mock_interview_task: Callable[
            [MainAgentContext, ToolObservation], MainAgentContext
        ],
        update_atomic_task: Callable[..., MainAgentContext],
        tool_call_fingerprint: Callable[[AgentDecision], str],
        tool_observation: Callable[
            [str, MainAgentToolOutput, dict[str, Any] | None],
            DecisionObservation,
        ],
    ) -> None:
        self._context_manager = context_manager
        self._emit_trace = emit_trace
        self._update_mock_interview_task = update_mock_interview_task
        self._update_atomic_task = update_atomic_task
        self._tool_call_fingerprint = tool_call_fingerprint
        self._tool_observation = tool_observation

    def reduce(self, state: MainAgentState) -> MainAgentState:
        context = state["context"]
        pending = state["pending"]
        result = pending["result"]
        capability_name = pending["name"]
        control = dict(state.get("control", {}))
        synthetic_kind = pending.get("synthetic_kind")

        if result.disposition == "failed":
            self._emit_trace(
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
        observation = self._tool_observation(
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
            return self._update_mock_interview_task(context, result)
        return self._update_atomic_task(
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

        fingerprint = self._tool_call_fingerprint(state["decision"])
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
