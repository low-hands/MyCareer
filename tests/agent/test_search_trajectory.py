"""Offline checks for search-mode trajectory fixtures and discovery hops."""

from __future__ import annotations

from threading import Lock
from dataclasses import replace
from types import SimpleNamespace

import pytest

from career_agent.agent.capabilities.selection_strategy import SearchStrategy
from career_agent.agent.capabilities.proactive import eligible_successors, succeeded
from career_agent.agent.contracts.observations import DecisionObservation
from career_agent.agent.contracts.observations import decision_observation_projection
from career_agent.agent.contracts.projections.memory import project_career_fact_arguments
from career_agent.agent.contracts.decisions import AgentDecision, ToolCall
from career_agent.agent.providers.openai_client import AgentWorkerError, OpenAICompatibleAgentConfig
import career_agent.evaluation.search_trajectory as search_trajectory
from career_agent.evaluation.main_agent_scenarios import SCENARIOS
from career_agent.evaluation.search_scenarios import SEARCH_SCENARIOS
from career_agent.evaluation.search_trajectory import (
    _advance_search_context, _search_result, check_search_contract, replay_search_sample,
)
from career_agent.evaluation.trajectory import TrajectoryStep, trajectory_tool_specs


def test_search_scenarios_keep_legacy_catalogue_untouched() -> None:
    assert len(SEARCH_SCENARIOS) == len(SCENARIOS) == 46
    legacy = {scenario.name: scenario for scenario in SCENARIOS}
    search = {scenario.name: scenario for scenario in SEARCH_SCENARIOS}
    assert legacy["a_core_request_routes_before_job_analysis"].steps[0].expect_tool == "route_to_capability"
    assert [step.expect_tool for step in search["a_core_request_routes_before_job_analysis"].steps] == ["analyze_job"]
    assert [step.expect_tool for step in search["tool_selection_combines_job_analysis_and_resume_match"].steps] == [
        "analyze_job", "match_resume_to_job",
    ]
    for name in (
        "intent_is_not_inferred_from_a_job_the_user_liked",
        "planning_to_apply_does_not_create_an_application",
    ):
        assert "analyze_job" in legacy[name].steps[0].forbid_tools
        assert "analyze_job" in search[name].steps[0].forbid_tools


def test_qwen_extra_job_analysis_is_marked_only_in_search_mode() -> None:
    legacy = {scenario.name: scenario for scenario in SCENARIOS}
    search = {scenario.name: scenario for scenario in SEARCH_SCENARIOS}
    for name in (
        "intent_is_not_inferred_from_a_job_the_user_liked",
        "saved_jd_body_without_language_requirement_finishes",
    ):
        assert legacy[name].known_gap is None
        assert "qwen3.7-plus" in search[name].known_gap
        assert "analyze_job" in search[name].known_gap
        assert "analyze_job" in search[name].steps[0].forbid_tools or (
            search[name].steps[0].expect_action == "final"
        )


def test_cached_research_remains_offered_after_successful_prior_call() -> None:
    name = "cached_research_routes_to_an_explicit_new_focus"
    legacy = next(item for item in SCENARIOS if item.name == name)
    search = next(item for item in SEARCH_SCENARIOS if item.name == name)
    assert "research_job" not in legacy.context.task.loaded_capabilities
    assert "research_job" in search.context.task.loaded_capabilities
    selection = SearchStrategy(
        proactive_enabled=False, intent_enabled=False,
    ).select(search.context, trajectory_tool_specs())
    assert "research_job" in selection.offered_names
    assert dict(selection.sources)["research_job"] == "loaded"
    search_cases = {item.name: item for item in SEARCH_SCENARIOS}
    assert "get_saved_job" in search_cases[
        "saved_jd_body_drives_the_next_read_step"
    ].context.task.loaded_capabilities
    assert "find_saved_jobs" not in search_cases[
        "an_authorization_refusal_is_explained_not_bypassed"
    ].context.task.loaded_capabilities
    assert "get_saved_job" not in search_cases[
        "an_invalid_selection_is_not_reconstructed"
    ].context.task.loaded_capabilities
    assert "update_working_notes" not in search_cases[
        "a_stale_working_note_is_merged_not_overwritten"
    ].context.task.loaded_capabilities


def test_search_trajectory_keeps_completed_tool_after_intervening_step() -> None:
    scenario = next(item for item in SEARCH_SCENARIOS
                    if item.name == "cached_research_routes_to_an_explicit_new_focus")
    context = _advance_search_context(
        scenario.context,
        TrajectoryStep(observation=DecisionObservation(
            tool_name="get_job_research", state="job_research_ready", message="已读取。",
        )),
    )
    context = _advance_search_context(
        context,
        TrajectoryStep(observation=DecisionObservation(
            tool_name="search_capabilities", state="capabilities_found", message="已搜索。",
        )),
    )
    assert "get_job_research" in context.task.loaded_capabilities
    assert "research_job" in context.task.loaded_capabilities


def test_uncertain_write_is_not_retained_or_used_for_successors(monkeypatch) -> None:
    import career_agent.evaluation.search_scenarios as search_scenarios

    observation = DecisionObservation(
        tool_name="research_job", state="job_research_ready", message="结果待核对。",
        disposition="completed", execution_outcome="unknown",
    )
    assert not succeeded(observation)
    assert eligible_successors(observation) == ()
    assert "execution_outcome" not in decision_observation_projection((observation,))[0]
    base = next(item for item in SCENARIOS
                if item.name == "cached_research_routes_to_an_explicit_new_focus")
    context = base.context.model_copy(update={"tool_observations": (observation,)})
    monkeypatch.setattr(search_scenarios, "SCENARIOS", (replace(base, context=context),))
    adapted = search_scenarios.search_mode_scenarios()[0]
    assert "research_job" not in adapted.context.task.loaded_capabilities
    advanced = _advance_search_context(base.context.model_copy(update={
        "tool_observations": (),
    }), TrajectoryStep(observation=observation))
    assert "research_job" not in advanced.task.loaded_capabilities


def test_every_search_scenario_has_a_legal_offline_request() -> None:
    specs = trajectory_tool_specs()
    assert not [failure for scenario in SEARCH_SCENARIOS
                for failure in check_search_contract(scenario, tool_specs=specs)]


def test_questionnaire_fact_scenario_has_a_projectable_exact_quote() -> None:
    scenario = next(item for item in SEARCH_SCENARIOS
                    if item.name == "a_questionnaire_answer_is_proposed_with_user_input_provenance")
    arguments = {
        "record_selection_index": 1,
        "claim": "日常使用 Claude Code 和 Cursor 开发内部工具。",
        "reason": "用户明确要求展示职业事实提案。",
        "user_quote": "Claude Code 和 Cursor 开发内部工具",
    }
    projected = project_career_fact_arguments(
        scenario.context, "propose_career_fact", arguments,
    )
    assert projected["career_record_id"] == "career_record_" + "a" * 32
    assert projected["origin"] == "user_input"
    assert projected["source_user_interaction_id"] == "interaction_" + "a" * 20
    with pytest.raises(ValueError, match="user_quote is absent"):
        project_career_fact_arguments(
            scenario.context, "propose_career_fact",
            {**arguments, "user_quote": "Claude Code 和 Cursor 开发内�工具"},
        )


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


def test_search_replay_stops_at_changed_intermediate_schema() -> None:
    scenario = next(item for item in SEARCH_SCENARIOS
                    if item.name == "a_core_request_routes_before_job_analysis")
    failures = replay_search_sample(
        scenario, tool_specs=trajectory_tool_specs(),
        responses=({
            "scenario_step": 0,
            "selected_schema_fingerprint": "old-schema",
            "tool_call": {"name": "analyze_job", "arguments": {}},
        },),
    )
    assert len(failures) == 1
    assert "selected schema changed during replay" in failures[0]


def test_search_recorder_replaces_cassette_with_changed_later_context(
    monkeypatch, tmp_path,
) -> None:
    scenario = SEARCH_SCENARIOS[0]
    monkeypatch.setattr(search_trajectory, "load_cassette", lambda *a, **k:
                        SimpleNamespace(recordings=((), (), ())))
    monkeypatch.setattr(search_trajectory, "search_cassette_staleness", lambda *a, **k: None)
    monkeypatch.setattr(search_trajectory, "replay_search_sample", lambda *a, **k:
                        ("case[1]: model context changed during replay",))
    monkeypatch.setattr(search_trajectory, "_record_sample", lambda *a, **k: {
        "recorded_at": "2026-10-07T00:00:00+00:00", "steps": [],
    })
    written = search_trajectory.record_search_catalogue(
        (scenario,), tool_specs=trajectory_tool_specs(),
        config=OpenAICompatibleAgentConfig(
            endpoint="https://example.invalid/v1/chat/completions",
            api_key="test", model="test",
        ), root=tmp_path, jobs=1,
    )
    assert len(written) == 1
    assert written[0].exists()


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


def test_failed_search_batch_preserves_other_samples_with_raw_retries(monkeypatch, tmp_path) -> None:
    scenario = SEARCH_SCENARIOS[0]
    lock = Lock()
    calls = 0

    def fake_record(*_args, **_kwargs):
        nonlocal calls
        with lock:
            calls += 1
            number = calls
        if number == 2:
            error = AgentWorkerError("MAIN_AGENT_INVALID_TOOL_ARGUMENTS", "invalid")
            error.recording_trace = {"decision_retry_events": [
                {"reason": "invalid_tool_arguments", "raw_output": {"content": "original"}}
            ]}
            raise error
        return {"steps": [{"decision_retry_events": [
            {"reason": "text_rejected", "raw_output": {"content": f"original-{number}"}}
        ]}]}

    monkeypatch.setattr(search_trajectory, "_record_sample", fake_record)
    with pytest.raises(AgentWorkerError) as raised:
        search_trajectory.record_search_catalogue(
            (scenario,), tool_specs=(),
            config=OpenAICompatibleAgentConfig(
                endpoint="https://example.invalid/v1/chat/completions",
                api_key="test", model="test",
            ),
            root=tmp_path, sample_count=3, jobs=3,
        )
    assert len(raised.value.recording_failures) == 1
    assert raised.value.recording_failures[0]["trace"]["decision_retry_events"][0][
        "raw_output"
    ] == {"content": "original"}
    assert len(raised.value.partial_recordings) == 2
    assert all(item["recording"]["steps"][0]["decision_retry_events"][0][
        "raw_output"
    ]["content"].startswith("original-") for item in raised.value.partial_recordings)
    assert not list(tmp_path.glob("*.json"))
