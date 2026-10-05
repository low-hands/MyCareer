"""Fast offline baseline for the tool set sent to each model decision."""

from __future__ import annotations

from statistics import mean
from types import SimpleNamespace

from career_agent.agent.capabilities.catalog import CAPABILITIES
from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.contracts.profile import CareerProfileContext
from career_agent.agent.contracts.task_state import ConversationTaskState
from career_agent.agent.runtime.decision_engine import DecisionEngine
from career_agent.evaluation.main_agent_scenarios import SCENARIOS
from career_agent.evaluation.tool_selection import (
    LegacyProfileSelector,
    classify_recorded_failure,
    evaluate_tool_selection,
)
from career_agent.evaluation.tool_selection_scenarios import TOOL_SELECTION_SCENARIOS
from career_agent.evaluation.trajectory import (
    EVALUATION_BASELINE_MODEL,
    TrajectoryScenario,
    TrajectoryStep,
    advance_trajectory_context,
    cassette_staleness,
    load_cassette,
    replay_cassette,
    trajectory_tool_specs,
)


_OTHER_BEHAVIOR = {
    "working_notes_never_choose_or_rank_a_job",
    "a_report_that_scrolled_out_of_the_catalogue_is_not_faked",
    "an_empty_conversation_span_is_not_filled_from_the_window",
}
# Re-recording left these two on an older prompt fingerprint. They are not
# evaluated until re-recorded, so they must not be counted as passes either.
_STALE = {
    "stated_intent_is_proposed_before_it_is_recorded",
    "a_questionnaire_answer_is_proposed_with_user_input_provenance",
}


def test_legacy_selection_baseline_counts_and_route_gaps() -> None:
    report = evaluate_tool_selection(
        (*SCENARIOS, *TOOL_SELECTION_SCENARIOS),
        selector=LegacyProfileSelector(trajectory_tool_specs()),
    )
    assert (len(report.steps), len(report.comparable_steps)) == (64, 59)
    assert (report.demand_steps, report.covered_steps) == (31, 26)
    assert report.route_round_trips == 5
    assert len(trajectory_tool_specs()) == 71
    assert len(report.demanded_tool_names) == 22
    assert {"job", "resume", "application", "interview", "memory"} <= {
        group
        for name in report.demanded_tool_names
        for group in CAPABILITIES[name].profiles
    }
    assert {
        (step.scenario, step.index, tuple(sorted(step.missing_reasons.items())))
        for step in report.steps if step.missing_names
    } == {
        ("a_core_request_routes_before_job_analysis", 0, (("analyze_job", "not_selected"),)),
        ("tool_selection_switches_from_resume_to_job_research", 0, (("research_job", "not_selected"),)),
        ("tool_selection_combines_job_analysis_and_resume_match", 1, (("match_resume_to_job", "not_selected"),)),
        ("offline_resume_to_interview_preparation", 0, (("prepare_interview", "not_selected"),)),
        ("offline_job_to_application_creation", 0, (("create_application", "not_selected"),)),
    }
    assert all(step.schema_tokens_proxy > 0 for step in report.comparable_steps)
    assert round(mean(len(step.offered_names) for step in report.comparable_steps), 1) == 24.1


def test_existing_cross_group_gap_is_visible_before_route_and_fixed_after() -> None:
    report = evaluate_tool_selection(
        tuple(s for s in SCENARIOS if s.name == "tool_selection_combines_job_analysis_and_resume_match"),
        selector=LegacyProfileSelector(trajectory_tool_specs()),
    )
    assert [step.covered for step in report.comparable_steps] == [True, False]
    assert report.steps[1].route_round_trip
    assert report.steps[1].missing_names == {"match_resume_to_job"}
    assert report.steps[2].legacy_only
    assert not report.steps[2].has_demand
    assert "match_resume_to_job" in report.steps[2].offered_names


def test_legacy_provenance_and_ingress_keyword_promotion() -> None:
    selector = LegacyProfileSelector(trajectory_tool_specs())
    context = MainAgentContext(
        conversation_id="eval",
        profile=CareerProfileContext(user_id="eval-user"),
        user_message="请分析当前岗位的 JD。",
        task=ConversationTaskState(tool_profile="core", active_job_posting_id="job-1"),
    )
    before, _ = selector.select(context, None)
    after, _ = selector.select_ingress(context)
    assert "analyze_job" not in before.names
    assert after.sources["analyze_job"] == "profile:job"
    assert after.sources["read_conversation_span"] == "core"


def test_offline_follow_on_cases_stay_separate_from_cassette_catalogue() -> None:
    assert not ({case.name for case in TOOL_SELECTION_SCENARIOS} & {case.name for case in SCENARIOS})
    report = evaluate_tool_selection(
        TOOL_SELECTION_SCENARIOS,
        selector=LegacyProfileSelector(trajectory_tool_specs()),
    )
    assert (len(report.steps), len(report.comparable_steps)) == (10, 8)
    assert (report.demand_steps, report.covered_steps) == (8, 6)
    assert report.route_round_trips == 2


def test_unreachable_is_not_misreported_as_not_selected() -> None:
    scenario = TrajectoryScenario(
        name="approval_prerequisite_missing",
        policy="A calendar execution requires an existing proposal.",
        context=MainAgentContext(
            conversation_id="eval",
            profile=CareerProfileContext(user_id="eval-user"),
            user_message="执行日历提案",
            task=ConversationTaskState(tool_profile="interview"),
        ),
        steps=(TrajectoryStep(expect_tool="execute_calendar_proposal"),),
    )
    report = evaluate_tool_selection(
        (scenario,), selector=LegacyProfileSelector(trajectory_tool_specs())
    )
    assert report.steps[0].missing_reasons == {
        "execute_calendar_proposal": "unreachable"
    }
    assert "execute_calendar_proposal" in report.steps[0].selected_names


def test_selector_state_is_carried_between_steps_and_reset_between_scenarios() -> None:
    class StatefulSelector:
        def __init__(self) -> None:
            self.legacy = LegacyProfileSelector(trajectory_tool_specs())
            self.seen: list[tuple[int | None, str | None]] = []

        def select(self, context, prior):
            latest = (
                context.tool_observations[-1].tool_name
                if context.tool_observations else None
            )
            self.seen.append((prior, latest))
            offer, _ = self.legacy.select(context, None)
            return offer, 1 if prior is None else prior + 1

    selector = StatefulSelector()
    evaluate_tool_selection(TOOL_SELECTION_SCENARIOS[:2], selector=selector)
    assert selector.seen == [
        (None, None), (1, "route_to_capability"),
        (None, None), (1, "route_to_capability"),
    ]


def test_legacy_offer_matches_decision_engine_for_every_snapshot() -> None:
    specs = trajectory_tool_specs()
    selector = LegacyProfileSelector(specs)
    engine = DecisionEngine(
        emit=lambda _: None,
        decision_heartbeat=lambda _: None,
        record_trace_event=lambda *args, **kwargs: None,
        project_atomic_tool_arguments=lambda *args: {},
        project_workflow_arguments=lambda *args: {},
        context_manager=object(),
        decision_maker_provider=lambda: object(),
        tools=SimpleNamespace(schemas=lambda: specs),
        career_memory_enabled=False,
    )
    for scenario in (*SCENARIOS, *TOOL_SELECTION_SCENARIOS):
        context = scenario.context
        for step in scenario.steps:
            context = advance_trajectory_context(context, step)
            offer, _ = selector.select(context, None)
            actual = engine.tool_schemas(context.task.tool_profile, context.task)
            assert offer.names == {schema["function"]["name"] for schema in actual}
            assert offer.schemas == actual


def test_existing_qwen_failures_have_a_reviewed_selection_classification() -> None:
    specs = trajectory_tool_specs()
    offer_report = evaluate_tool_selection(
        SCENARIOS, selector=LegacyProfileSelector(specs)
    )
    by_scenario = {
        scenario.name: tuple(step for step in offer_report.steps if step.scenario == scenario.name)
        for scenario in SCENARIOS
    }
    categories = {}
    stale = set()
    for scenario in SCENARIOS:
        cassette = load_cassette(scenario.name)
        assert cassette is not None
        if cassette_staleness(
            cassette,
            scenario=scenario,
            tool_specs=specs,
            expected_model=EVALUATION_BASELINE_MODEL,
        ):
            stale.add(scenario.name)
            continue
        if any(replay_cassette(scenario, tool_specs=specs, cassette=cassette)):
            categories[scenario.name] = classify_recorded_failure(
                scenario,
                selection_steps=by_scenario[scenario.name],
                recordings=cassette.recordings,
            )
    assert stale == _STALE
    assert len(categories) == 10
    assert {name for name, kind in categories.items() if kind == "other_behavior"} == _OTHER_BEHAVIOR
    assert list(categories.values()).count("selection_gap_and_model_decision") == 2
    assert list(categories.values()).count("model_decision") == 5
    assert list(categories.values()).count("other_behavior") == 3
