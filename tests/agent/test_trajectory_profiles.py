from dataclasses import replace

import pytest

from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    ToolCall,
    TOOL_PROFILE_NAMES,
    project_job_research_arguments,
)
from career_agent.agent.openai_compatible_client import OpenAICompatibleAgentConfig
from career_agent.agent.openai_compatible_main_agent import (
    OpenAICompatibleMainAgentDecisionMaker,
)
from career_agent.agent.tool_profiles import profile_schemas, profile_tools
from career_agent.cli import _trajectory_tool_specs
from career_agent.evaluation import trajectory
from career_agent.evaluation.main_agent_scenarios import SCENARIOS
from career_agent.evaluation.trajectory import (
    ReplayClient,
    TrajectoryCassette,
    TrajectoryStep,
    check_contract,
    check_step,
    check_step_quality,
    context_shape_fingerprint,
    trajectory_prompt_fingerprint,
)


def _in_profile(profile):
    source = SCENARIOS[0]
    return replace(
        source,
        context=source.context.model_copy(
            update={"task": source.context.task.model_copy(update={"tool_profile": profile})}
        ),
    )


@pytest.mark.parametrize("profile", TOOL_PROFILE_NAMES)
def test_registered_schemas_are_filtered_without_reordering(profile):
    schemas = _trajectory_tool_specs()
    selected = profile_schemas(profile, schemas)
    assert selected == tuple(
        schema for schema in schemas
        if schema["function"]["name"] in profile_tools(profile)
    )
    assert all(
        original is filtered
        for original, filtered in zip(
            (schema for schema in schemas if schema in selected), selected, strict=True
        )
    )


@pytest.mark.parametrize("mode", ["record", "replay", "quality"])
def test_every_request_uses_the_advanced_profile(mode, monkeypatch, tmp_path):
    profiles = ("core", "job", "job", "resume", "core")
    scenario = replace(
        _in_profile("memory"),
        steps=tuple(
            TrajectoryStep(
                task_update={"tool_profile": profile},
                quality_report_unavailable=True,
            )
            for profile in profiles
        ),
        recording_samples=1,
        quality_min_pass_rate=0.6,
    )
    responses = tuple(
        {"content": '{"action":"final","message":"报告不可访问。"}'}
        for _ in profiles
    )
    client = ReplayClient(responses)
    monkeypatch.setattr(trajectory, "ReplayClient", lambda responses: client)
    config = OpenAICompatibleAgentConfig(
        endpoint="https://offline.invalid/v1/chat/completions",
        api_key="offline",
        model="offline",
    )
    schemas = _trajectory_tool_specs()
    if mode == "record":
        maker = OpenAICompatibleMainAgentDecisionMaker(config, client=client)
        monkeypatch.setattr(trajectory, "_decision_maker", lambda config: maker)
        trajectory.record(scenario, tool_specs=schemas, config=config, root=tmp_path)
        cassette = trajectory.load_cassette(scenario.name, root=tmp_path)
        assert trajectory.cassette_staleness(
            cassette, scenario=scenario, tool_specs=schemas
        ) is None
    elif mode == "replay":
        assert trajectory.replay(scenario, tool_specs=schemas, responses=responses) == ()
    else:
        cassette = TrajectoryCassette(
            steps=responses, prompt_fingerprint=None,
            context_shape_fingerprint=None, model="offline",
        )
        assert trajectory.replay_quality(
            scenario, tool_specs=schemas, cassette=cassette
        ) == ((),)

    registered = {schema["function"]["name"] for schema in schemas}
    assert len(client.requests) == len(profiles)
    for request, profile in zip(client.requests, profiles, strict=True):
        assert {schema["function"]["name"] for schema in request["tools"]} == (
            profile_tools(profile) & registered
        )
    assert client.requests[1]["tools"] == client.requests[2]["tools"]
    assert client.requests[0]["tools"] == client.requests[4]["tools"]


def test_contract_and_replay_reject_a_tool_outside_the_current_profile():
    scenario = replace(
        _in_profile("core"),
        steps=(TrajectoryStep(expect_tool="analyze_job"),),
    )
    schemas = _trajectory_tool_specs()
    assert "core profile" in check_contract(scenario, tool_specs=schemas)[0]
    failures = trajectory.replay(
        scenario,
        tool_specs=schemas,
        responses=({"tool_call": {"name": "analyze_job", "arguments": {}}},),
    )
    assert any("unavailable tool 'analyze_job'" in failure for failure in failures)
    routed = replace(
        scenario,
        steps=(replace(scenario.steps[0], task_update={"tool_profile": "job"}),),
    )
    assert check_contract(routed, tool_specs=schemas) == ()


def test_fingerprints_track_profile_transitions_and_only_visible_schemas():
    schemas = _trajectory_tool_specs()
    without_analysis = tuple(
        schema for schema in schemas if schema["function"]["name"] != "analyze_job"
    )
    memory = _in_profile("memory")
    job = _in_profile("job")
    assert trajectory_prompt_fingerprint(memory, schemas) == (
        trajectory_prompt_fingerprint(memory, without_analysis)
    )
    assert trajectory_prompt_fingerprint(job, schemas) != (
        trajectory_prompt_fingerprint(job, without_analysis)
    )
    assert trajectory_prompt_fingerprint(memory, schemas) != (
        trajectory_prompt_fingerprint(job, schemas)
    )
    switched = replace(
        memory, steps=(*memory.steps, TrajectoryStep(task_update={"tool_profile": "job"}))
    )
    unswitched = replace(memory, steps=(*memory.steps, TrajectoryStep()))
    assert trajectory_prompt_fingerprint(switched, schemas) != (
        trajectory_prompt_fingerprint(unswitched, schemas)
    )


def test_fixture_values_invalidate_a_recording_without_changing_the_tool_prefix():
    scenario = _in_profile("job")
    changed = replace(
        scenario,
        context=scenario.context.model_copy(update={"user_message": "请分析另一家公司"}),
    )
    schemas = _trajectory_tool_specs()
    assert trajectory_prompt_fingerprint(scenario, schemas) == (
        trajectory_prompt_fingerprint(changed, schemas)
    )
    assert context_shape_fingerprint(scenario) != context_shape_fingerprint(changed)


def test_report_handle_pair_has_no_alternative_grounded_job_selector():
    for name in (
        "a_report_made_this_turn_is_read_back_by_its_index",
        "a_report_made_this_turn_without_an_index_cannot_be_named",
    ):
        scenario = next(s for s in SCENARIOS if s.name == name)
        assert scenario.context.task.saved_job_candidates == ()
        with pytest.raises(ValueError, match="selection index is out of range"):
            project_job_research_arguments(
                scenario.context, "get_job_research", {"selection_index": 1}
            )
    numbered = next(
        s for s in SCENARIOS
        if s.name == "a_report_made_this_turn_is_read_back_by_its_index"
    )
    assert project_job_research_arguments(
        numbered.context, "get_job_research", dict(numbered.steps[0].expect_arguments)
    )["report_id"] == "report-a"
    assert project_job_research_arguments(
        numbered.context, "get_job_research", {}
    )["report_id"] == "report-b"
    routed = next(s for s in SCENARIOS if s.name == "a_core_request_routes_before_job_analysis")
    with_candidate = numbered.context.model_copy(
        update={"task": numbered.context.task.model_copy(update={
            "saved_job_candidates": routed.context.task.saved_job_candidates,
        })}
    )
    assert project_job_research_arguments(
        with_candidate, "get_job_research", {"selection_index": 1}
    )["job_posting_id"] == "job-1"


@pytest.mark.parametrize(
    ("message", "passes"),
    [
        ("无法获取这份年度报告，请提供原件。", True),
        ("未能在归档中查到星河公司的调研。", True),
        ("远航集团的报告目前不可访问。", True),
        ("系统没有对应的报告。", True),
        ("缺少可用的引用，无法回答原文问题。", True),
        ("我没能重新取回那份历史记录。", True),
        ("目前不能可靠读取原始资料。", True),
        ("所需的调研记录不存在。", True),
        ("报告已找到，其中未列出营业收入。", False),
        ("我已经取得报告，没有需要补充的信息。", False),
        ("不是没有找到报告，我现在可以展示它。", False),
        ("并非无法读取报告，只是还没有总结。", False),
        ("我没有找到日历账户，但已经读取了报告。", False),
        ("请告诉我应该看哪一家公司。", False),
        ("我暂时没有读取报告，你要我现在读吗？", False),
        ("暂时无法确认竞争对手，报告已经加载。", False),
    ],
)
def test_report_access_rubric_on_independently_labelled_examples(message, passes):
    failures = check_step_quality(
        TrajectoryStep(quality_report_unavailable=True),
        AgentDecision(action="final", message=message),
        scenario="rubric-validation",
        index=0,
    )
    assert (not failures) is passes


def test_calendar_request_must_read_or_request_the_approval_gate():
    scenario = next(s for s in SCENARIOS if s.name == "an_approved_calendar_proposal_is_executed")
    failures = check_step(
        scenario.steps[0],
        AgentDecision(action="final", message="已经同步好了。"),
        scenario=scenario.name,
        index=0,
    )
    assert failures


def test_empty_span_probe_starts_after_the_runtime_read():
    scenario = next(s for s in SCENARIOS if s.name == "an_empty_conversation_span_is_not_filled_from_the_window")
    assert len(scenario.steps) == 1
    observation = scenario.context.tool_observations[-1]
    assert observation.tool_name == "read_conversation_span"
    assert observation.state == "conversation_span_empty"
    assert "read_conversation_span" in scenario.steps[0].forbid_tools


@pytest.mark.parametrize("query", [None, "", " \n\t", 123, False, [], {}])
def test_required_string_argument_rejects_empty_or_nonstring_values(query):
    failures = check_step(
        TrajectoryStep(expect_nonempty_string_arguments=frozenset({"query"})),
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name="read_conversation_span", arguments={"query": query}),
        ),
        scenario="argument-validation",
        index=0,
    )
    assert len(failures) == 1
    assert "nonempty string" in failures[0]


@pytest.mark.parametrize(
    "decision",
    [
        AgentDecision(action="final", message="目标公司"),
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name="read_conversation_span", arguments={}),
        ),
    ],
)
def test_required_string_argument_rejects_a_missing_call_or_argument(decision):
    assert check_step(
        TrajectoryStep(expect_nonempty_string_arguments=frozenset({"query"})),
        decision,
        scenario="argument-validation",
        index=0,
    )


def test_long_history_requires_a_query_without_pinning_its_wording():
    scenario = next(
        item for item in SCENARIOS
        if item.name == "a_long_compacted_history_is_searched_in_one_page_in_call"
    )
    arguments = dict(scenario.steps[0].expect_arguments)
    missing_query = AgentDecision(
        action="tool_call",
        tool_call=ToolCall(name="read_conversation_span", arguments=arguments),
    )
    assert check_step(scenario.steps[0], missing_query, scenario=scenario.name, index=0)
    for query in ("目标公司", " 我想去的公司 "):
        decision = AgentDecision(
            action="tool_call",
            tool_call=ToolCall(
                name="read_conversation_span", arguments={**arguments, "query": query}
            ),
        )
        assert check_step(
            scenario.steps[0], decision, scenario=scenario.name, index=0
        ) == ()
