from typing import get_type_hints

import pytest

from career_agent.agent.contracts.main_agent import (
    CareerProfileContext,
    MainAgentContext,
)
from career_agent.agent.runtime.main_agent_runtime import MainAgentState as RuntimeState
from career_agent.agent.runtime.state import (
    LoopControl,
    MainAgentState,
    MainAgentStateValidationError,
    PendingAction,
    validate_main_agent_state,
)


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
    assert MainAgentState.__required_keys__ == set(MainAgentState.__annotations__)


def _complete_state() -> dict:
    return {
        "context": MainAgentContext(
            conversation_id="c1",
            profile=CareerProfileContext(user_id="u1"),
            user_message="继续",
        ),
        "decision": None,
        "pending": {},
        "authorization_route": None,
        "tool_results": [],
        "control": {
            "read_calls": 0,
            "write_calls": 0,
            "projection_refusals": 0,
            "authorization_refusals": 0,
            "fingerprints": [],
            "retryable_fingerprints": [],
            "retry_counts": {},
        },
        "artifact_ids": [],
        "assistant_message": "",
        "model_message": "",
        "career_memory_scope_keys": [],
    }


def test_graph_state_validation_normalizes_checkpoint_collections() -> None:
    state = validate_main_agent_state(_complete_state(), boundary="result")

    assert state["tool_results"] == ()
    assert state["artifact_ids"] == ()
    assert state["control"]["fingerprints"] == ()


@pytest.mark.parametrize(
    "mutate",
    (
        lambda state: state.pop("context"),
        lambda state: state.update({"unknown_channel": True}),
        lambda state: state["pending"].update({"unknown_pending_field": True}),
        lambda state: state["control"].update({"unknown_counter": 1}),
    ),
)
def test_graph_state_validation_rejects_missing_and_undeclared_channels(
    mutate,
) -> None:
    state = _complete_state()
    mutate(state)

    with pytest.raises(MainAgentStateValidationError):
        validate_main_agent_state(state, boundary="invoke")
