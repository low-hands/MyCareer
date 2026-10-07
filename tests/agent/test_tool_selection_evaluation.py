"""Fast offline baseline for the tool set sent to each model decision."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
import hashlib
import json
from statistics import mean
from types import SimpleNamespace

from pydantic_core import to_jsonable_python
from pydantic import BaseModel

from career_agent.agent.capabilities.catalog import CAPABILITIES
from career_agent.agent.capabilities.reachability import reachable
from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.contracts.observations import DecisionObservation
from career_agent.agent.contracts.interactions import CONFIRMATION_SPECS
from career_agent.agent.contracts.profile import CareerProfileContext
from career_agent.agent.contracts.task_state import ConversationTaskState
from career_agent.agent.runtime.decision_engine import DecisionEngine
from career_agent.evaluation.main_agent_scenarios import SCENARIOS
from career_agent.evaluation.independent_tool_selection_holdout import SELECTION_INDEPENDENT_HOLDOUT
from career_agent.evaluation.tool_selection import (
    BatchSearchSimulationSelector,
    LegacyProfileSelector,
    ProactiveSearchSimulationSelector,
    RuntimeSearchSelector,
    SearchSimulationSelector,
    classify_recorded_failure,
    evaluate_tool_selection,
)


def test_runtime_intent_plus_w_locks_dev_offer_tradeoff_without_search_round_trips() -> None:
    specs = trajectory_tool_specs()
    w = evaluate_tool_selection(
        SELECTION_DEV, selector=RuntimeSearchSelector(specs, intent_enabled=False),
    )
    combined = evaluate_tool_selection(
        SELECTION_DEV, selector=RuntimeSearchSelector(specs, intent_enabled=True),
    )
    assert (w.covered_steps, w.demand_steps) == (19, 62)
    assert (combined.covered_steps, combined.demand_steps) == (51, 62)
    assert combined.search_round_trips == 0
    assert (
        combined.unreachable_offer_count,
        combined.waiting_reoffer_count,
        combined.unrequested_write_offer_count,
    ) == (0, 0, 30)
    assert mean(len(step.offered_names) for step in combined.comparable_steps) < 9


def test_search_simulators_lock_dev_comparison_only() -> None:
    specs = trajectory_tool_specs()
    ideal = evaluate_tool_selection(
        SELECTION_DEV, selector=SearchSimulationSelector(specs, ideal=True),
    )
    lexical = evaluate_tool_selection(
        SELECTION_DEV, selector=SearchSimulationSelector(specs, ideal=False),
    )
    assert (ideal.covered_steps, ideal.demand_steps, ideal.search_round_trips, ideal.search_failures) == (62, 62, 57, 0)
    assert (lexical.covered_steps, lexical.demand_steps, lexical.search_round_trips, lexical.search_failures) == (49, 62, 44, 13)
    assert (ideal.unreachable_offer_count, ideal.waiting_reoffer_count, ideal.unrequested_write_offer_count) == (0, 0, 0)
    assert (lexical.unreachable_offer_count, lexical.waiting_reoffer_count, lexical.unrequested_write_offer_count) == (0, 0, 27)


def test_proactive_successors_and_bound_reads_reduce_dev_search_trips() -> None:
    report = evaluate_tool_selection(
        SELECTION_DEV,
        selector=ProactiveSearchSimulationSelector(trajectory_tool_specs()),
    )
    assert (report.covered_steps, report.demand_steps) == (62, 62)
    assert (report.search_round_trips, report.search_failures) == (43, 0)
    assert (
        report.unreachable_offer_count,
        report.waiting_reoffer_count,
        report.unrequested_write_offer_count,
    ) == (0, 0, 0)


def test_batch_idealization_loads_same_turn_tools_without_crossing_user_turn() -> None:
    specs = trajectory_tool_specs()
    context = MainAgentContext(
        conversation_id="c1", profile=CareerProfileContext(user_id="u1"),
        user_message="先列简历，再分析岗位",
        task=ConversationTaskState(active_job_posting_id="job-1", active_jd_snapshot_id="jd-1"),
    )
    same_turn = TrajectoryScenario(
        name="batch_same_turn", policy="test", context=context,
        steps=(TrajectoryStep(expect_tool="list_resumes"),
               TrajectoryStep(expect_tool="analyze_job")),
    )
    separate_turn = TrajectoryScenario(
        name="batch_separate_turn", policy="test", context=context,
        steps=(TrajectoryStep(expect_tool="list_resumes"),
               TrajectoryStep(expect_tool="analyze_job", user_message="现在分析岗位")),
    )
    batched = evaluate_tool_selection(
        (same_turn,), selector=BatchSearchSimulationSelector(specs),
    )
    separate = evaluate_tool_selection(
        (separate_turn,), selector=BatchSearchSimulationSelector(specs),
    )
    assert batched.search_round_trips == 1
    assert "analyze_job" in batched.steps[0].offered_names
    assert separate.search_round_trips == 2
    assert "analyze_job" not in separate.steps[0].offered_names


def test_batch_idealization_locks_dev_lower_bound_without_holdout_scores() -> None:
    report = evaluate_tool_selection(
        SELECTION_DEV,
        selector=BatchSearchSimulationSelector(trajectory_tool_specs()),
    )
    assert (report.covered_steps, report.demand_steps, report.search_round_trips) == (
        62, 62, 39,
    )
    assert (report.unreachable_offer_count, report.waiting_reoffer_count,
            report.unrequested_write_offer_count) == (0, 0, 0)


def test_proactive_offer_uses_bound_read_and_reviewed_successor_only() -> None:
    specs = trajectory_tool_specs()
    selector = ProactiveSearchSimulationSelector(specs)
    task = ConversationTaskState(active_application_id="app-1")
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        user_message="看看投递",
        task=task,
    )
    offer, _ = selector.select(context, None)
    assert "get_application" in offer.names
    assert "create_application" not in offer.names
from career_agent.evaluation.tool_selection_scenarios import (
    FIXTURE_NOW,
    SELECTION_DEV,
    SELECTION_HOLDOUT,
    TOOL_SELECTION_SCENARIOS,
)
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
# The default historical Qwen cassettes remain stale: two use an older prompt,
# and two now expose task updates that model_copy previously dropped. None may
# be counted as a pass until that historical directory is deliberately recut.
_STALE = {
    "stated_intent_is_proposed_before_it_is_recorded",
    "a_questionnaire_answer_is_proposed_with_user_input_provenance",
    "a_repeated_call_is_not_reissued_after_an_observation",
    "tool_selection_combines_job_analysis_and_resume_match",
}


def test_selection_cases_have_balanced_coverage_and_no_cassette_names() -> None:
    cases = (*SELECTION_DEV, *SELECTION_HOLDOUT)
    assert (len(SELECTION_DEV), len(SELECTION_HOLDOUT)) == (45, 20)
    assert all(case.split == "dev" for case in SELECTION_DEV)
    assert all(case.split == "holdout" for case in SELECTION_HOLDOUT)
    assert len({case.scenario.name for case in cases}) == len(cases)
    assert {case.scenario.name for case in cases}.isdisjoint(
        {scenario.name for scenario in SCENARIOS}
    )
    counts = Counter(case.kind for case in cases)
    assert counts == {"single": 34, "cross": 14, "chain": 9, "control": 8}
    assert Counter(case.kind for case in SELECTION_DEV) == {
        "single": 27, "cross": 9, "chain": 6, "control": 3,
    }
    assert Counter(case.kind for case in SELECTION_HOLDOUT) == {
        "single": 7, "cross": 5, "chain": 3, "control": 5,
    }
    namespace_counts = Counter(namespace for case in SELECTION_DEV for namespace in case.namespaces)
    assert {descriptor.namespace for descriptor in CAPABILITIES.values() if descriptor.model_callable} == set(namespace_counts)
    assert min(namespace_counts.values()) >= 2
    for case in cases:
        assert case.split in {"dev", "holdout"}
        assert case.namespaces
        assert case.scenario.context.user_message


def test_expanded_dev_baseline_is_locked_without_tuning_on_holdout() -> None:
    report = evaluate_tool_selection(
        SELECTION_DEV, selector=LegacyProfileSelector(trajectory_tool_specs())
    )
    assert (len(report.steps), len(report.comparable_steps)) == (66, 64)
    assert (report.demand_steps, report.covered_steps, report.route_round_trips) == (62, 46, 2)
    assert len(report.demanded_tool_names) == 39
    assert (
        report.unreachable_offer_count,
        report.waiting_reoffer_count,
        report.unrequested_write_offer_count,
    ) == (339, 1, 469)
    assert {
        (step.scenario, step.index, tuple(sorted(step.missing_reasons.items())))
        for step in report.steps if step.missing_reasons
    } == {
        ("offline_resume_to_interview_preparation", 0, (("prepare_interview", "not_selected"),)),
        ("offline_job_to_application_creation", 0, (("create_application", "not_selected"),)),
        ("selection_dev_job_analysis_first", 0, (("analyze_job", "not_selected"),)),
        ("selection_dev_job_research_first", 0, (("research_job", "not_selected"),)),
        ("selection_dev_job_intent_first", 0, (("propose_job_intent", "not_selected"),)),
        ("selection_dev_interview_prep_first", 0, (("prepare_interview", "not_selected"),)),
        ("selection_dev_memory_search_first", 0, (("search_career_history", "not_selected"),)),
        ("selection_dev_memory_proposals_first", 0, (("propose_career_fact", "not_selected"),)),
        ("selection_dev_cross_compare_then_match", 2, (("match_resume_to_job", "not_selected"),)),
        ("selection_dev_cross_export_then_prepare", 1, (("prepare_interview", "not_selected"),)),
        ("selection_dev_cross_read_job_then_track", 1, (("create_application", "not_selected"),)),
        ("selection_dev_cross_memory_then_job", 1, (("search_career_episodes", "not_selected"),)),
        ("selection_dev_cross_application_then_interview", 1, (("create_interview", "not_selected"),)),
        ("selection_dev_cross_source_then_tailor", 1, (("draft_resume_tailoring", "not_selected"),)),
        ("selection_dev_chain_job_resume", 2, (("match_resume_to_job", "not_selected"),)),
        ("selection_dev_chain_job_resume", 3, (("draft_resume_tailoring", "not_selected"),)),
    }


def test_holdout_runs_without_a_locked_recall_score() -> None:
    report = evaluate_tool_selection(
        SELECTION_HOLDOUT, selector=LegacyProfileSelector(trajectory_tool_specs())
    )
    assert len(report.steps) >= len(SELECTION_HOLDOUT)
    assert all(step.split == "holdout" for step in report.steps)
    assert all(step.schema_tokens_proxy > 0 for step in report.steps)


def _holdout_manifest_digest(cases) -> str:
    def stable(value):
        if isinstance(value, DecisionObservation):
            receipt = value.model_dump(mode="python")
            # An absent internal receipt does not change the authored fixture.
            for field in ("disposition", "execution_outcome"):
                if receipt[field] is None:
                    receipt.pop(field)
            return stable(receipt)
        if isinstance(value, BaseModel):
            return stable(value.model_dump(mode="python"))
        if is_dataclass(value):
            return stable(asdict(value))
        if isinstance(value, Mapping):
            return {key: stable(item) for key, item in value.items()}
        if isinstance(value, (set, frozenset)):
            return sorted(
                (stable(item) for item in value),
                key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True),
            )
        if isinstance(value, (list, tuple)):
            return [stable(item) for item in value]
        return to_jsonable_python(value)

    manifest = [
        {
            "name": case.scenario.name,
            "split": case.split,
            "kind": case.kind,
            "namespaces": sorted(case.namespaces),
            "message": case.scenario.context.user_message,
            "task": stable(case.scenario.context.task.model_dump(
                mode="python", exclude={"loaded_capabilities"}
            )),
            "raw_turn": case.raw_turn,
            "steps": stable(case.scenario.steps),
        }
        for case in cases
    ]
    encoded = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_holdout_fixture_is_frozen_independently_of_selector_scores() -> None:
    assert _holdout_manifest_digest(SELECTION_HOLDOUT) == (
            "c2f8477110cf2591f8ccbe9d707071ef4262667fd438ff9607939d44705ef209"
    )


def test_independently_authored_holdout_is_valid_and_frozen() -> None:
    assert len(SELECTION_INDEPENDENT_HOLDOUT) == 12
    assert Counter(case.kind for case in SELECTION_INDEPENDENT_HOLDOUT) == {
        "single": 4, "cross": 3, "chain": 3, "control": 2,
    }
    report = evaluate_tool_selection(
        SELECTION_INDEPENDENT_HOLDOUT,
        selector=LegacyProfileSelector(trajectory_tool_specs()),
    )
    assert len(report.steps) == 19
    for case in SELECTION_INDEPENDENT_HOLDOUT:
        context = case.scenario.context
        for step in case.scenario.steps:
            context = advance_trajectory_context(context, step)
            assert step.expect_tool is not None
            assert reachable(step.expect_tool, context.task), (
                case.scenario.name, step.expect_tool
            )
    assert _holdout_manifest_digest(SELECTION_INDEPENDENT_HOLDOUT) == (
        "f7fcfef7322a566a80e739fd5cb1690626adf2b59123a1e53d5d9e5c70b23804"
    )


def test_overoffer_metrics_distinguish_prerequisites_waits_and_write_intent() -> None:
    cases = (
        next(case for case in SELECTION_DEV if case.scenario.name == "selection_dev_job_research_next"),
        next(case for case in SELECTION_DEV if case.scenario.name == "selection_dev_control_planning_to_apply"),
        next(case for case in SELECTION_DEV if case.scenario.name == "selection_dev_control_note_derived_filter_waits"),
    )
    report = evaluate_tool_selection(cases, selector=LegacyProfileSelector(trajectory_tool_specs()))
    by_name = {step.scenario: step for step in report.steps if step.index == 0}
    assert "research_job" in by_name["selection_dev_job_research_next"].unreachable_offered
    assert "analyze_job" in by_name["selection_dev_control_planning_to_apply"].unrequested_writes
    waiting = by_name["selection_dev_control_note_derived_filter_waits"]
    assert waiting.waiting_reoffered == {"find_saved_jobs"}
    assert waiting.required_names == frozenset()


def test_raw_cases_use_ingress_and_reference_reads_have_bound_sources() -> None:
    selector = LegacyProfileSelector(trajectory_tool_specs())
    cases = (*SELECTION_DEV, *SELECTION_HOLDOUT)
    for case in cases:
        first = case.scenario.steps[0]
        context = advance_trajectory_context(case.scenario.context, first)
        expected, _ = (
            selector.select_ingress(context)
            if case.raw_turn else selector.select(context, None)
        )
        actual = evaluate_tool_selection((case,), selector=selector).steps[0]
        assert actual.offered_names == expected.names
    for case in cases:
        if case.scenario.name.endswith("_first") and case.kind == "single":
            assert case.scenario.context.task.tool_profile == "core"

    by_name = {case.scenario.name: case.scenario.context for case in cases}
    span = by_name["selection_dev_context_next"]
    assert span.through_sequence > 0
    assert span.recent_from_sequence > span.through_sequence
    assert by_name["selection_dev_context_first"].attached_resumes
    assert by_name["selection_dev_interview_mock_next"].task.active_application_id
    claim = by_name["selection_dev_memory_search_next"].career_memory.records[0].confirmed_highlights[0]
    assert claim.detail_ref


def test_expanded_positive_demands_cover_at_least_55_model_tools_without_unreachable_targets() -> None:
    cases = (*SELECTION_DEV, *SELECTION_HOLDOUT)
    report = evaluate_tool_selection(
        cases,
        selector=LegacyProfileSelector(trajectory_tool_specs()),
    )
    assert len(report.demanded_tool_names) >= 55
    assert all("unreachable" not in step.missing_reasons.values() for step in report.steps)
    by_scenario = {
        case.scenario.name: tuple(step for step in report.steps if step.scenario == case.scenario.name)
        for case in cases
    }
    for case in cases:
        context = case.scenario.context
        for snapshot, declaration in zip(by_scenario[case.scenario.name], case.scenario.steps):
            context = advance_trajectory_context(context, declaration)
            ConversationTaskState.model_validate(context.task.model_dump(mode="python"))
            if snapshot.has_demand:
                assert any(reachable(name, context.task) for name in snapshot.required_names), (
                    case.scenario.name, snapshot.index, snapshot.required_names
                )
                for name in snapshot.required_names:
                    if name in CONFIRMATION_SPECS:
                        assert context.task.pending_proposal_is_live(
                            CONFIRMATION_SPECS[name].slot, FIXTURE_NOW
                        )
                    if name == "execute_calendar_proposal":
                        assert context.task.active_calendar_proposal_expires_at > FIXTURE_NOW


def test_legacy_selection_baseline_counts_and_route_gaps() -> None:
    report = evaluate_tool_selection(
        (*SCENARIOS, *TOOL_SELECTION_SCENARIOS),
        selector=LegacyProfileSelector(trajectory_tool_specs()),
    )
    assert (len(report.steps), len(report.comparable_steps)) == (64, 59)
    assert (report.demand_steps, report.covered_steps) == (31, 26)
    assert report.route_round_trips == 5
    assert len(trajectory_tool_specs()) == 72
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
    assert len(categories) == 9
    assert {name for name, kind in categories.items() if kind == "other_behavior"} == _OTHER_BEHAVIOR
    assert list(categories.values()).count("selection_gap_and_model_decision") == 1
    assert list(categories.values()).count("model_decision") == 5
    assert list(categories.values()).count("other_behavior") == 3
