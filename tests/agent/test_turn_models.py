from career_agent.agent.contracts.main_agent import AgentDecision
from career_agent.agent.runtime import main_agent_runtime
from career_agent.agent.contracts.turn import (
    InteractionReceipt,
    MainAgentTurnResult,
    ModelDecision,
    RuntimeAction,
    RuntimePolicyAction,
)


def test_runtime_keeps_turn_model_compatibility_exports() -> None:
    assert main_agent_runtime.MainAgentTurnResult is MainAgentTurnResult
    assert main_agent_runtime.ModelDecision is ModelDecision
    assert main_agent_runtime.InteractionReceipt is InteractionReceipt
    assert main_agent_runtime.RuntimeAction is RuntimeAction
    assert main_agent_runtime.RuntimePolicyAction is RuntimePolicyAction


def test_turn_accountability_is_derived_from_origin_variant() -> None:
    decision = AgentDecision(action="final", message="完成。")

    model_turn = MainAgentTurnResult(
        origin=ModelDecision(decision), context=None, assistant_message="完成。"
    )
    policy_turn = MainAgentTurnResult(
        origin=RuntimePolicyAction(policy="career_fact_confirmation"),
        context=None,
        assistant_message="已确认。",
    )

    assert model_turn.requested_by == "model"
    assert model_turn.model_decision is decision
    assert policy_turn.requested_by == "runtime"
    assert policy_turn.model_decision is None
