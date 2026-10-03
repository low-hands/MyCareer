from typing import get_type_hints

from career_agent.agent.main_agent_runtime import MainAgentState as RuntimeState
from career_agent.agent.main_state import LoopControl, MainAgentState, PendingAction


def test_runtime_reexports_the_canonical_graph_state() -> None:
    assert RuntimeState is MainAgentState


def test_graph_state_has_only_declared_cross_node_channels() -> None:
    assert set(MainAgentState.__annotations__) == {
        "context",
        "decision",
        "pending",
        "authorization_route",
        "tool_results",
        "control",
        "artifact_ids",
        "assistant_message",
        "model_message",
        "career_memory_scope_keys",
    }
    annotations = get_type_hints(MainAgentState)
    assert annotations["pending"] is PendingAction
    assert annotations["control"] is LoopControl
