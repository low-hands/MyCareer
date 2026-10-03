from __future__ import annotations

from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    CareerProfileContext,
    DecisionObservation,
    MainAgentContext,
    ToolCall,
    ToolObservation,
)
from career_agent.agent.observation_reducer import ObservationReducer


def test_synthetic_refusal_reduces_without_runtime_host_or_task_mutation() -> None:
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        user_message="继续",
    )
    result = ToolObservation(
        tool_name="start_mock_interview",
        state="invalid_input",
        message="缺少选择。",
    )
    calls: list[str] = []

    def unexpected(*args, **kwargs):  # pragma: no cover - assertion helper
        raise AssertionError("synthetic observations must not reduce task state")

    def observation(name, tool_result, arguments):
        calls.append(name)
        return DecisionObservation(
            tool_name=name,
            state=tool_result.state,
            message=tool_result.message,
            arguments=arguments or {},
        )

    reducer = ObservationReducer(
        context_manager=object(),  # type: ignore[arg-type]
        emit_trace=lambda *args, **kwargs: calls.append("trace"),
        update_mock_interview_task=unexpected,
        update_atomic_task=unexpected,
        tool_call_fingerprint=lambda decision: "fingerprint",
        tool_observation=observation,
    )
    decision = AgentDecision(
        action="tool_call",
        tool_call=ToolCall(name="start_mock_interview", arguments={}),
    )

    update = reducer.reduce(
        {
            "context": context,
            "decision": decision,
            "pending": {
                "name": "start_mock_interview",
                "synthetic_kind": "projection",
                "result": result,
            },
            "control": {"projection_refusals": 0},
        }
    )

    assert calls == ["start_mock_interview"]
    assert update["context"].task == context.task
    assert update["context"].tool_observations[-1].state == "invalid_input"
    assert update["control"]["projection_refusals"] == 1
    assert update["tool_results"] == ()
