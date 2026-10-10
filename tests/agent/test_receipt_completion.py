from types import SimpleNamespace

import pytest

from career_agent.agent.contracts.decisions import AgentDecision, ToolCall
from career_agent.agent.contracts.observations import ToolObservation
from career_agent.agent.presentation.receipt_completion import _DIRECT_TOOLS, completes_tool_delivery
from career_agent.harness.graph_routing import GraphRoutingPolicy


def _state(name="export_resume_artifact", result_state="resume_artifact_ready", **kwargs):
    result = ToolObservation(tool_name=name, state=result_state, message="结果已就绪", payload={"artifact_id": "artifact-1"}, **kwargs)
    return {"decision": AgentDecision(action="tool_call", tool_call=ToolCall(name=name)),
            "pending": {"name": name, "effect": "READ", "result": result},
            "context": SimpleNamespace(turn_continuation_capability=None), "tool_results": (result,)}


def test_only_export_is_a_terminal_tool():
    assert _DIRECT_TOOLS == {"export_resume_artifact"}
    assert GraphRoutingPolicy.after_observe(_state()) == "present"


@pytest.mark.parametrize("name,state", [("get_daily_brief", "daily_brief_ready"), ("list_resumes", "no_resumes_found"), ("list_interviews", "no_interviews_found"), ("create_interview", "interview_ready"), ("analyze_job", "job_analysis_ready")])
def test_other_receipts_always_return_to_model(name, state):
    assert GraphRoutingPolicy.after_observe(_state(name, state)) == "decide"


def test_export_integrity_and_prior_business_results():
    for changes in ({"disposition": "failed"}, {"next_action": "需要补充信息"}, {"state": "resume_not_found"}, {"payload": {}}):
        state = _state()
        state["pending"]["result"] = state["pending"]["result"].model_copy(update=changes)
        state["tool_results"] = (state["pending"]["result"],)
        assert not completes_tool_delivery(state)
    state = _state()
    state["tool_results"] = (ToolObservation(tool_name="list_resumes", state="resumes_found", message="查询结果"), *state["tool_results"])
    assert not completes_tool_delivery(state)
    state = _state()
    state["tool_results"] = (ToolObservation(tool_name="search_capabilities", state="capabilities_found", message="已加载"), *state["tool_results"])
    assert completes_tool_delivery(state)
    state["context"].turn_continuation_capability = "export_resume_artifact"
    assert not completes_tool_delivery(state)


def test_no_completion_metadata_in_any_schema_or_prompt():
    from career_agent.agent.providers.main_agent import _normalize_tool_specs, OpenAICompatibleMainAgentDecisionMaker
    from career_agent.evaluation.trajectory import trajectory_tool_specs
    assert "deliver_direct" not in ToolCall.model_fields
    assert "deliver_direct" not in str(_normalize_tool_specs(trajectory_tool_specs()))
    assert "deliver_direct" not in OpenAICompatibleMainAgentDecisionMaker._system_prompt()
