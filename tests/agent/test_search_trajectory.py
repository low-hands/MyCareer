"""Offline checks for search-mode trajectory fixtures and discovery hops."""

from __future__ import annotations

from career_agent.agent.capabilities.selection_strategy import SearchStrategy
from career_agent.agent.contracts.decisions import AgentDecision, ToolCall
from career_agent.agent.providers.openai_client import OpenAICompatibleAgentConfig
import career_agent.evaluation.search_trajectory as search_trajectory
from career_agent.evaluation.main_agent_scenarios import SCENARIOS
from career_agent.evaluation.search_scenarios import SEARCH_SCENARIOS
from career_agent.evaluation.search_trajectory import (
    _search_result, check_search_contract, replay_search_sample,
)
from career_agent.evaluation.trajectory import trajectory_tool_specs


def test_search_scenarios_keep_legacy_catalogue_untouched() -> None:
    assert len(SEARCH_SCENARIOS) == len(SCENARIOS) == 46
    legacy = {scenario.name: scenario for scenario in SCENARIOS}
    search = {scenario.name: scenario for scenario in SEARCH_SCENARIOS}
    assert legacy["a_core_request_routes_before_job_analysis"].steps[0].expect_tool == "route_to_capability"
    assert [step.expect_tool for step in search["a_core_request_routes_before_job_analysis"].steps] == ["analyze_job"]
    assert [step.expect_tool for step in search["tool_selection_combines_job_analysis_and_resume_match"].steps] == [
        "analyze_job", "match_resume_to_job",
    ]


def test_every_search_scenario_has_a_legal_offline_request() -> None:
    specs = trajectory_tool_specs()
    assert not [failure for scenario in SEARCH_SCENARIOS
                for failure in check_search_contract(scenario, tool_specs=specs)]


def test_search_replay_accepts_intermediate_discovery() -> None:
    scenario = next(item for item in SEARCH_SCENARIOS
                    if item.name == "a_core_request_routes_before_job_analysis")
    specs = trajectory_tool_specs()
    strategy = SearchStrategy()
    first = strategy.select(scenario.context, specs)
    loaded = _search_result(scenario.context, {"names": ["analyze_job"]})
    second = strategy.select(loaded, specs)
    assert "analyze_job" not in first.offered_names
    assert "analyze_job" in second.offered_names
    from career_agent.evaluation.trajectory import prompt_fingerprint
    responses = (
        {"scenario_step": 0, "tool_call": {
            "name": "search_capabilities", "arguments": {"names": ["analyze_job"]},
        }, "selected_schema_fingerprint": prompt_fingerprint(first.schemas, mode="search")},
        {"scenario_step": 0, "tool_call": {
            "name": "analyze_job", "arguments": {},
        }, "selected_schema_fingerprint": prompt_fingerprint(second.schemas, mode="search")},
    )
    assert replay_search_sample(scenario, tool_specs=specs, responses=responses) == ()


def test_search_recording_captures_both_decisions_and_replays(monkeypatch) -> None:
    scenario = next(item for item in SEARCH_SCENARIOS
                    if item.name == "a_core_request_routes_before_job_analysis")
    specs = trajectory_tool_specs()

    class FakeMaker:
        def __init__(self, *_args, **_kwargs):
            self.calls = 0

        def configure_tool_selection(self, _strategy):
            pass

        def decide(self, _context, _schemas):
            self.calls += 1
            name = "search_capabilities" if self.calls == 1 else "analyze_job"
            arguments = {"names": ["analyze_job"]} if self.calls == 1 else {}
            return AgentDecision(action="tool_call", tool_call=ToolCall(
                name=name, arguments=arguments,
            ))

        def consume_cache_metrics(self):
            return {"attempt_count": 1}

        def consume_decision_retry_metrics(self):
            return {"decision_retry_telemetry_version": 1, "decision_retry_events": []}

    with monkeypatch.context() as patch:
        patch.setattr(search_trajectory, "OpenAICompatibleMainAgentDecisionMaker", FakeMaker)
        sample = search_trajectory._record_sample(
            scenario, tool_specs=specs,
            config=OpenAICompatibleAgentConfig(
                endpoint="https://example.invalid/v1/chat/completions",
                api_key="test", model="test",
            ),
        )
    assert [item["tool_call"]["name"] for item in sample["steps"]] == [
        "search_capabilities", "analyze_job",
    ]
    assert all(item["decision_shape_fingerprint"] for item in sample["steps"])
    assert replay_search_sample(
        scenario, tool_specs=specs, responses=sample["steps"],
    ) == ()
