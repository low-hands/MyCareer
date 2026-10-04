from __future__ import annotations

import pytest

from career_agent.agent.contracts.decisions import (
    AgentDecision,
    ToolCall,
)
from career_agent.agent.contracts.observations import ToolObservation
from career_agent.harness.graph_routing import GraphRoutingPolicy


@pytest.mark.parametrize(
    ("pending", "expected"),
    [
        ({}, "hydrate"),
        ({"runtime_owned": True}, "authorize"),
        ({"owner_confirmed": True}, "authorize"),
        ({"policy_owned": True}, "authorize"),
    ],
)
def test_entry_skips_model_hydration_only_for_already_decided_actions(
    pending: dict,
    expected: str,
) -> None:
    assert GraphRoutingPolicy.route_entry({"pending": pending}) == expected


def test_decision_routes_by_typed_action_and_preserves_prior_failure() -> None:
    assert GraphRoutingPolicy.route_decision(
        {
            "decision": AgentDecision(
                action="tool_call",
                tool_call=ToolCall(name="find_saved_jobs", arguments={}),
            )
        }
    ) == "authorize"
    assert GraphRoutingPolicy.route_decision(
        {"decision": AgentDecision(action="ask_user", message="请补充信息。")}
    ) == "interrupt"
    assert GraphRoutingPolicy.route_decision(
        {
            "decision": AgentDecision(action="ask_user", message="请补充信息。"),
            "tool_results": (
                ToolObservation(
                    tool_name="find_saved_jobs",
                    state="failed",
                    message="读取失败。",
                    disposition="failed",
                ),
            ),
        }
    ) == "present"
    assert GraphRoutingPolicy.route_decision(
        {"decision": AgentDecision(action="final", message="完成。")}
    ) == "present"


@pytest.mark.parametrize(
    ("pending_fields", "disposition", "payload", "expected"),
    [
        ({}, "interaction_required", {}, "interrupt"),
        ({"policy_prelude": True}, "completed", {}, "hydrate"),
        ({"runtime_owned": True}, "completed", {}, "present"),
        ({"owner_confirmed": True}, "completed", {}, "present"),
        ({"policy_owned": True}, "completed", {}, "present"),
        ({}, "completed", {"turn_complete": True}, "present"),
        ({"synthetic_kind": "projection"}, "completed", {}, "decide"),
        ({}, "failed", {}, "decide"),
        ({}, "completed", {}, "decide"),
    ],
)
def test_observation_routes_by_execution_ownership_and_result_contract(
    pending_fields: dict,
    disposition: str,
    payload: dict,
    expected: str,
) -> None:
    state_name = (
        "career_fact_proposed"
        if disposition == "interaction_required"
        else "failed"
        if disposition == "failed"
        else "test_result"
    )
    result = ToolObservation(
        tool_name="find_saved_jobs",
        state=state_name,
        message="结果。",
        disposition=disposition,
        payload=payload,
    )
    state = {"pending": {**pending_fields, "result": result}}

    assert GraphRoutingPolicy.after_observe(state) == expected


def test_authorization_route_is_the_middleware_pipeline_verdict() -> None:
    assert GraphRoutingPolicy.after_authorize({"authorization_route": "observe"}) == (
        "observe"
    )
