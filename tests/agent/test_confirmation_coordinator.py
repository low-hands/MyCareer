from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from career_agent.agent.runtime.interaction_coordinator import ConfirmationResolution
from career_agent.agent.contracts.main_agent import (
    CareerProfileContext,
    MainAgentContext,
    ToolObservation,
)
from career_agent.agent.contracts.turn import InteractionReceipt
from career_agent.harness.confirmation_coordinator import (
    CapabilityConfirmationCoordinator,
)
from career_agent.harness.streaming import InteractionResponse


class ResolvingInteractionCoordinator:
    def __init__(self, resolution: ConfirmationResolution | None = None) -> None:
        self.resolution = resolution
        self.calls: list[dict[str, Any]] = []

    def resolve_confirmation(self, **kwargs):
        self.calls.append(kwargs)
        if self.resolution is not None:
            return self.resolution
        graph_state = kwargs["invoke_confirmed"](
            SimpleNamespace(
                capability="confirm_memory_tombstone",
                arguments={"entry_id": "memory-1"},
                confirmation_id="confirmation-1",
            )
        )
        return ConfirmationResolution(graph_state=graph_state, action="confirm")


class RecordingAgentLoop:
    def __init__(self) -> None:
        self.invoke_calls: list[dict[str, Any]] = []
        self.result_calls: list[dict[str, Any]] = []
        self.graph_state = {"tool_results": ()}
        self.turn = object()

    def invoke(self, context, **kwargs):
        self.invoke_calls.append({"context": context, **kwargs})
        return self.graph_state

    def result_from_state(self, state, **kwargs):
        self.result_calls.append({"state": state, **kwargs})
        return self.turn


def _context() -> MainAgentContext:
    return MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        user_message="确认",
    )


def _response(action: str = "confirm") -> InteractionResponse:
    return InteractionResponse(
        interaction_id="interaction_0123456789abcdefabcd",
        scope="capability_confirmation",
        action=action,
    )


def test_owner_confirmation_replays_only_the_sealed_action() -> None:
    interactions = ResolvingInteractionCoordinator()
    loop = RecordingAgentLoop()
    coordinator = CapabilityConfirmationCoordinator(
        confirmation_store=None,
        interaction_coordinator=interactions,  # type: ignore[arg-type]
        agent_loop=loop,  # type: ignore[arg-type]
    )
    context = _context()

    turn = coordinator.run_owner_confirmation(
        context=context,
        conversation_id="c1",
        response=_response(),
    )

    assert turn is loop.turn
    assert len(loop.invoke_calls) == 1
    invocation = loop.invoke_calls[0]
    assert invocation["context"] is context
    assert invocation["decision"].tool_call.name == "confirm_memory_tombstone"
    assert invocation["decision"].tool_call.arguments == {}
    assert invocation["pending"] == {
        "name": "confirm_memory_tombstone",
        "arguments": {"entry_id": "memory-1"},
        "owner_confirmed": True,
        "confirmation_id": "confirmation-1",
    }
    assert loop.result_calls == [
        {
            "state": loop.graph_state,
            "origin": InteractionReceipt(
                scope="capability_confirmation",
                action="confirm",
            ),
        }
    ]


def test_settled_confirmation_is_returned_without_invoking_the_graph() -> None:
    interactions = ResolvingInteractionCoordinator(
        ConfirmationResolution(
            result=ToolObservation(
                tool_name="capability_confirmation",
                state="capability_confirmation_cancelled",
                message="已取消。",
            ),
            action="cancel",
        )
    )
    loop = RecordingAgentLoop()
    coordinator = CapabilityConfirmationCoordinator(
        confirmation_store=None,
        interaction_coordinator=interactions,  # type: ignore[arg-type]
        agent_loop=loop,  # type: ignore[arg-type]
    )

    turn = coordinator.run_owner_confirmation(
        context=_context(),
        conversation_id="c1",
        response=_response("cancel"),
    )

    assert loop.invoke_calls == []
    assert loop.result_calls == []
    assert turn.origin == InteractionReceipt(
        scope="capability_confirmation",
        action="cancel",
    )
    assert turn.tool_result is not None
    assert turn.tool_result.state == "capability_confirmation_cancelled"
    assert turn.assistant_message == "已取消。"
