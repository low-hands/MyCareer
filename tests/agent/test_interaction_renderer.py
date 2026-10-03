import pytest

from career_agent.agent.contracts.main_agent import (
    AgentDecision,
    CareerProfileContext,
    MainAgentContext,
    ToolCall,
    ToolObservation,
)
from career_agent.agent.presentation.interaction_renderer import InteractionRenderer
from career_agent.agent.contracts.turn import MainAgentTurnResult, ModelDecision


def _renderer(**kwargs) -> InteractionRenderer:
    return InteractionRenderer(
        active_turn_id=lambda: "turn-1",
        assistant_message=lambda result: f"rendered: {result.message}",
        **kwargs,
    )


def test_interaction_renderer_builds_free_text_event_for_model_question() -> None:
    decision = AgentDecision(action="ask_user", message="你更偏向哪个方向？")
    context = MainAgentContext(
        conversation_id="conversation-1",
        profile=CareerProfileContext(user_id="user-1"),
        user_message="帮我规划",
    )
    turn = MainAgentTurnResult(
        origin=ModelDecision(decision),
        context=context,
        assistant_message=decision.message,
    )

    event = _renderer().event(result=turn, conversation_id="conversation-1")

    assert event is not None
    assert event.kind == "free_text"
    assert event.prompt == decision.message


def test_interaction_renderer_enforces_the_configured_renderer_boundary() -> None:
    renderer = _renderer(has_interaction_renderer=lambda state: False)
    result = ToolObservation(
        tool_name="example",
        state="calendar_approval_required",
        message="需要继续。",
    )
    state = {
        "decision": AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name="example", arguments={}),
        ),
        "tool_results": (result,),
    }

    with pytest.raises(ValueError, match="has no interaction renderer"):
        renderer.interrupt(state)
