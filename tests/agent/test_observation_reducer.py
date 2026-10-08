from __future__ import annotations

import pytest

from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.contracts.decisions import (
    AgentDecision,
    ToolCall,
)
from career_agent.agent.contracts.observations import (
    DecisionObservation,
    ToolObservation,
)
from career_agent.agent.contracts.profile import CareerProfileContext
from career_agent.agent.contracts.task_state import ConversationTaskState
from career_agent.agent.capabilities.catalog import CAPABILITIES
from career_agent.agent.capabilities.selection_strategy import SearchStrategy
from career_agent.agent.runtime.observation_reducer import ObservationReducer


@pytest.mark.parametrize(
    ("state", "disposition", "expected_loaded"),
    (
        ("resumes_listed", "completed", True),
        ("invalid_input", "failed", False),
        ("working_notes_derived_argument", "completed", False),
    ),
)
def test_successful_tool_stays_loaded_for_later_selection(
    state, disposition, expected_loaded,
) -> None:
    class Clock:
        def now(self):
            return None

    context = MainAgentContext(
        conversation_id="c1", profile=CareerProfileContext(user_id="u1"),
        user_message="看看简历", task=ConversationTaskState(),
    )
    reducer = ObservationReducer(
        context_manager=Clock(),  # type: ignore[arg-type]
        emit_trace=lambda *args, **kwargs: None,
        update_mock_interview_task=lambda context, result: context,
        update_atomic_task=lambda context, result, now: context,
        tool_call_fingerprint=lambda decision: "fingerprint",
        tool_observation=lambda name, result, arguments: DecisionObservation(
            tool_name=name, state=result.state, message=result.message,
        ),
    )
    update = reducer.reduce({
        "context": context,
        "decision": AgentDecision(
            action="tool_call", tool_call=ToolCall(name="list_resumes", arguments={}),
        ),
        "pending": {
            "name": "list_resumes", "effect": "READ",
            "result": ToolObservation(
                tool_name="list_resumes", state=state, message="完成。",
                disposition=disposition,
            ),
        },
        "control": {},
    })
    updated = update["context"]
    assert ("list_resumes" in updated.task.loaded_capabilities) is expected_loaded
    if expected_loaded:
        next_turn = MainAgentContext.model_validate(updated.model_dump()).model_copy(
            update={"tool_observations": (), "user_message": "再看一次"},
        )
        schemas = tuple(item.tool_schema() for item in CAPABILITIES.values() if item.model_callable)
        selection = SearchStrategy().select(next_turn, schemas)
        assert "list_resumes" in selection.offered_names
        assert dict(selection.sources)["list_resumes"] == "loaded"


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


def test_w_successor_survives_observation_window_trimming() -> None:
    class Clock:
        def now(self):
            from datetime import datetime, timezone
            return datetime.now(timezone.utc)

    context = MainAgentContext(
        conversation_id="c1", profile=CareerProfileContext(user_id="u1"),
        user_message="分析后匹配简历",
        task=ConversationTaskState(
            active_job_posting_id="job-1", active_jd_snapshot_id="jd-1",
            active_job_analysis_id="analysis-1",
            active_job_analysis_jd_snapshot_id="jd-1", job_analysis_status="ready",
            active_resume_version_id="resume-1",
        ),
    )
    reducer = ObservationReducer(
        context_manager=Clock(),  # type: ignore[arg-type]
        emit_trace=lambda *args, **kwargs: None,
        update_mock_interview_task=lambda context, result: context,
        update_atomic_task=lambda context, result, now: context,
        tool_call_fingerprint=lambda decision: "fingerprint",
        tool_observation=lambda name, result, arguments: DecisionObservation(
            tool_name=name, state=result.state, message=result.message,
        ),
    )
    control = {}
    for name, state in (("analyze_job", "job_analysis_ready"), *(
        ("search_capabilities", "capabilities_found") for _ in range(11)
    )):
        effect = CAPABILITIES[name].effect
        update = reducer.reduce({
            "context": context,
            "decision": AgentDecision(
                action="tool_call", tool_call=ToolCall(name=name, arguments={}),
            ),
            "pending": {
                "name": name, "effect": effect,
                "result": ToolObservation(tool_name=name, state=state, message="完成。"),
            },
            "control": control,
        })
        context, control = update["context"], update["control"]
    assert len(context.tool_observations) == 11
    assert all(item.tool_name != "analyze_job" for item in context.tool_observations)
    assert "match_resume_to_job" in context.turn_proactive_capabilities
    restored = MainAgentContext.model_validate(context.model_dump())
    assert restored.turn_proactive_capabilities == context.turn_proactive_capabilities
    assert "turn_proactive_capabilities" not in context.model_context()
    schemas = tuple(item.tool_schema() for item in CAPABILITIES.values() if item.model_callable)
    selection = SearchStrategy().select(
        context.model_copy(update={"user_message": " "}), schemas,
    )
    assert "match_resume_to_job" in selection.offered_names
    assert dict(selection.sources)["match_resume_to_job"] == "proactive"
