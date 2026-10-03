"""Explicit extension ports for runtime behavior that tests or hosts may replace."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from career_agent.agent.contracts.main_agent import (
    DecisionMaker,
    MainAgentContext,
    ToolObservation,
)
from career_agent.agent.middleware.argument_projection import project_atomic_arguments
from career_agent.agent.runtime.observability import RuntimeObservability
from career_agent.agent.runtime.reducers import reduce_task_state
from career_agent.agent.runtime.turn_coordinator import active_turn_id
from career_agent.agent.presentation.interaction_renderer import InteractionRenderer


AtomicArgumentProjector = Callable[
    [MainAgentContext, str, dict[str, Any]], dict[str, Any]
]
MockInterviewTaskReducer = Callable[
    [MainAgentContext, ToolObservation], MainAgentContext
]
AtomicTaskReducer = Callable[..., MainAgentContext]


class DecisionMakerSlot:
    """Explicit mutable reference used when a host replaces the decision model."""

    def __init__(self, decision_maker: DecisionMaker) -> None:
        self._decision_maker = decision_maker

    def get(self) -> DecisionMaker:
        return self._decision_maker

    def replace(self, decision_maker: DecisionMaker) -> None:
        self._decision_maker = decision_maker


def project_atomic_tool_arguments(
    context: MainAgentContext,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return project_atomic_arguments(
        context,
        name,
        arguments,
        source_turn_id=active_turn_id(),
    )


def update_mock_interview_task(
    context: MainAgentContext,
    result: ToolObservation,
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


def update_atomic_task(
    context: MainAgentContext,
    result: ToolObservation,
    *,
    now: datetime | None = None,
) -> MainAgentContext:
    return context.model_copy(
        update={"task": reduce_task_state(context.task, result, now=now)}
    )


@dataclass(frozen=True)
class RuntimePorts:
    """Supported runtime substitutions, supplied explicitly at construction."""

    project_atomic_tool_arguments: AtomicArgumentProjector = (
        project_atomic_tool_arguments
    )
    emit_capability_started: Callable[[str], None] = (
        RuntimeObservability.emit_capability_started
    )
    emit_capability_completed: Callable[[str, str], None] = (
        RuntimeObservability.emit_capability_completed
    )
    update_mock_interview_task: MockInterviewTaskReducer = (
        update_mock_interview_task
    )
    update_atomic_task: AtomicTaskReducer = update_atomic_task
    has_interaction_renderer: Callable[[str], bool] = (
        lambda state: state in InteractionRenderer.RENDERER_STATES
    )
