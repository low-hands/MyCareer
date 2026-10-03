from __future__ import annotations

from collections.abc import Callable

from career_agent.agent.capabilities.registry import MainAgentToolOutput
from career_agent.agent.presentation.engine import PresentationEngine
from career_agent.agent.presentation.interaction_renderer import InteractionRenderer
from career_agent.agent.presentation.presenter import TurnPresenter
from career_agent.agent.presentation.result_presenter import DegradedReporter, ResultPresenter
from career_agent.agent.runtime.state import MainAgentState
from career_agent.agent.contracts.turn import MainAgentTurnResult
from career_agent.harness.streaming import InteractionRequiredEvent


def _ignore_degraded(*_: object, **__: object) -> None:
    """Default reporter for presentation use outside an observed runtime."""


def build_turn_presenter(
    *, report_degraded: DegradedReporter = _ignore_degraded
) -> TurnPresenter:
    """Build the canonical presenter without depending on the runtime facade."""

    return TurnPresenter(
        render_result=lambda result: ResultPresenter.present(
            result,
            report_degraded=report_degraded,
        )
    )


def render_tool_output(
    result: MainAgentToolOutput,
    *,
    report_degraded: DegradedReporter = _ignore_degraded,
) -> str:
    return ResultPresenter.present(result, report_degraded=report_degraded)


def present_turn(
    state: MainAgentState,
    *,
    report_degraded: DegradedReporter = _ignore_degraded,
) -> MainAgentState:
    return PresentationEngine.present(
        state,
        renderer=build_turn_presenter(report_degraded=report_degraded),
    )


def build_interaction_renderer(
    *,
    active_turn_id: Callable[[], str | None] = lambda: None,
    report_degraded: DegradedReporter = _ignore_degraded,
    has_interaction_renderer: Callable[[str], bool] | None = None,
) -> InteractionRenderer:
    presenter = build_turn_presenter(report_degraded=report_degraded)
    return InteractionRenderer(
        active_turn_id=active_turn_id,
        assistant_message=presenter._assistant_message,
        has_interaction_renderer=has_interaction_renderer,
    )


def interaction_event(
    *,
    result: MainAgentTurnResult,
    conversation_id: str,
) -> InteractionRequiredEvent | None:
    """Render a completed result for callers that do not own a runtime stack."""

    return build_interaction_renderer().event(
        result=result,
        conversation_id=conversation_id,
    )
