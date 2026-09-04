"""Trajectory evaluation for the Main Agent's decision loop.

The rest of this suite scripts ``decide`` and checks the runtime. That leaves
the system prompt — 11k characters, ninety policy sentences, the largest
behavioural artifact in the project — with no coverage at all. These are the
first tests that fail when a policy stops holding.

Two levels, and the difference is the whole design:

- **Contract**, always run, no API key. Does the context we send still make the
  policy decidable, and is each expected tool exposed by the production menu?
  Hidden forbidden tools are structural reachability guarantees, tested by the
  reachability suite rather than misreported as model behaviour.
- **Replay**, run per scenario that has a cassette. Checks the decision itself.

A green contract run does NOT mean the model behaves. It means the question was
asked properly. Only a fresh recording evaluates the model, which is why
``career-agent eval trajectories --record`` exists and why cassette staleness is
reported rather than hidden.
"""


from __future__ import annotations

import json
import re

from dataclasses import replace

import pytest

from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.agent.openai_compatible_main_agent import (
    OpenAICompatibleMainAgentDecisionMaker,
)
from career_agent.agent.delivery_policy import (
    condenses_message,
    delivers_body_elsewhere,
)
from career_agent.evaluation.main_agent_scenarios import SCENARIOS
from career_agent.evaluation.trajectory import (
    TrajectoryCassette,
    cassette_staleness,
    check_contract,
    context_shape_fingerprint,
    load_cassette,
    prompt_fingerprint,
    replay,
    trajectory_prompt_fingerprint,
)


@pytest.fixture(scope="module")
def offered() -> tuple[frozenset[str], tuple[dict, ...]]:
    """Every tool the registry can offer, wired with placeholder services.

    The scenarios never execute a tool, so the services only have to exist. What
    matters is that the schema list matches production: a scenario that forbids
    a tool the model was not offered proves nothing, and this is what lets
    ``check_contract`` notice.
    """
    registry = MainAgentToolRegistry(
        **{name: object() for name in _SERVICE_PARAMETERS}
    )
    schemas = registry.schemas()
    return frozenset(spec["function"]["name"] for spec in schemas), schemas


_SERVICE_PARAMETERS = (
    "job_repository",
    "job_comparison_service",
    "career_profile_store",
    "resume_store",
    "resume_analysis_service",
    "resume_job_match_service",
    "resume_tailoring_service",
    "resume_export_service",
    "application_service",
    "email_tracking_service",
    "interview_service",
    "interview_preparation_service",
    "action_center_service",
    "calendar_service",
    "mock_interview_graph",
    "mock_interview_store",
    "job_research_service",
)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda item: item.name)
def test_a_scenario_asks_a_question_the_model_could_answer(scenario, offered) -> None:
    """The contract level: the scenario is not vacuous and not stale."""
    _, schemas = offered
    assert check_contract(scenario, tool_specs=schemas) == ()


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda item: item.name)
def test_the_recorded_decision_follows_the_policy(scenario, offered) -> None:
    """The replay level: what the model actually chose, when we have it."""
    _, schemas = offered
    cassette = load_cassette(scenario.name)
    if cassette is None:
        pytest.skip(
            f"no recording for '{scenario.name}'; run "
            "`career-agent eval trajectories --record` against a live model"
        )
    stale = cassette_staleness(
        cassette,
        scenario=scenario,
        tool_specs=schemas,
    )
    assert stale is None, f"{scenario.name}: {stale}"
    failures = replay(scenario, tool_specs=schemas, responses=cassette.steps)
    if scenario.known_gap is not None:
        # Expected to fail, and reported if it stops: the scenario is right and
        # the system is not, so a pass here means the defect was fixed and the
        # marker should be retired.
        assert failures, (
            f"{scenario.name} now passes; remove its known_gap: "
            f"{scenario.known_gap}"
        )
        pytest.xfail(scenario.known_gap)
    assert failures == ()


_FIELD_DUMP_LINE = re.compile(r"^\s*[\w ]{1,30}\s*[:：]\s*\S", re.M)


def test_a_reply_does_not_take_over_delivery_that_belongs_elsewhere() -> None:
    """The worry that kept the model silent, turned into a standing check.

    The Answer Writer existed because a model asked to close a tool-backed turn
    was expected to emit label/value scaffolding instead of a sentence. That
    assumption was removed rather than disproved once, so it needs a guard that
    every recording has to pass, not a single sample.

    Scoped to the turns the reason applies to. Structure in a reply is a defect
    when the body is delivered elsewhere — a card the reader can open, or a
    presenter render — because then the reply is duplicating, into a window that
    keeps it forever, something that already has a home. That is not true of a
    plain state: ``find_saved_jobs`` answers with ``找到 N 个已保存职位。`` and
    nothing else, with the postings themselves only in ``task.candidates``. A
    model that does not enumerate them has not answered the question, and one
    that does is carrying the only copy, not a second one.

    The guard used to apply everywhere, and flagged exactly that case. It also
    passed on three of four samples of the same behaviour, because compressing
    the same fields onto one line or prefixing them with ``-`` slipped the
    regex: the pattern measures format, the rule is about usurping delivery, and
    on a plain state the two come apart. Loosening the threshold would have
    widened that gap rather than closed it — narrowing where it applies says
    that place was never in scope, which is the true statement.

    Stays a recording-level check rather than moving to a synthetic assertion on
    ``_present``. There is nothing there to assert: ``_present`` passes
    ``decision.message`` through untouched, so a synthetic reply that duplicates
    a card's body is accepted by construction, and making it refuse one would be
    new runtime behaviour rather than the same test in a better place. The half
    that *is* synthetically checkable — which ceiling a reply is clamped to — is
    already covered three ways by ``test_a_reply_is_bounded_by_what_else_carries_
    the_delivery``. Splitting would move the covered half and leave the
    uncovered half alone.

    Already multi-sample: one recording per scenario, not one sample overall.
    The single-sample risk lives in the per-scenario behaviour assertions
    instead — ``expect_tool``, ``expect_arguments`` — and is registered there.
    """
    checked = 0
    for scenario in SCENARIOS:
        cassette = load_cassette(scenario.name)
        if cassette is None:
            continue
        for index, step in enumerate(cassette.steps):
            try:
                decision = json.loads(step["content"])
            except (TypeError, ValueError, KeyError):
                continue
            message = (decision.get("message") or "").strip()
            if decision.get("action") != "final" or not message:
                continue
            label = f"{scenario.name}[{index}]"
            # A reply is never JSON and always reads as language, whatever the
            # turn produced.
            assert not message.startswith("{"), label
            assert any(mark in message for mark in "。！？.!?"), label
            # What the turn was holding when it answered: the seeded context
            # plus every observation fed back up to and including this step.
            # Seeded ones matter — several scenarios put the card-backed result
            # in the context rather than on a step, and reading only the step
            # would exempt exactly the turns this guard is for.
            observations = (
                *scenario.context.tool_observations,
                *(
                    step_before.observation
                    for step_before in scenario.steps[: index + 1]
                    if step_before.observation is not None
                ),
            )
            if not observations:
                continue
            latest = observations[-1]
            if not (
                condenses_message(latest.state)
                or delivers_body_elsewhere(latest.state)
            ):
                continue
            checked += 1
            assert "|---" not in message, label
            assert len(_FIELD_DUMP_LINE.findall(message)) < 2, label

    assert checked, "no recorded reply on a turn that delivers a body elsewhere"


def test_every_scenario_names_the_policy_sentence_it_holds() -> None:
    """A scenario without its rule quoted cannot be judged when it fails.

    The question on a failure is always whether the policy changed or the model
    did, and that is unanswerable if the rule is not written down next to the
    assertion.
    """
    for scenario in SCENARIOS:
        assert len(scenario.policy) > 40, scenario.name
        assert scenario.steps, scenario.name


def test_a_cassette_without_the_current_prompt_fingerprint_is_stale(offered) -> None:
    _, schemas = offered
    scenario = SCENARIOS[0]
    current = trajectory_prompt_fingerprint(scenario, schemas)
    current_shape = context_shape_fingerprint(scenario)

    assert cassette_staleness(
        TrajectoryCassette(
            steps=(),
            prompt_fingerprint=None,
            context_shape_fingerprint=current_shape,
            model="test",
        ),
        scenario=scenario,
        tool_specs=schemas,
    ) == "cassette has no prompt_fingerprint; re-record it"
    assert cassette_staleness(
        TrajectoryCassette(
            steps=(),
            prompt_fingerprint="0" * len(current),
            context_shape_fingerprint=current_shape,
            model="test",
        ),
        scenario=scenario,
        tool_specs=schemas,
    ) == (
        "cassette prompt_fingerprint does not match the current dynamic "
        "prompt/tool menu; re-record it"
    )
    assert cassette_staleness(
        TrajectoryCassette(
            steps=(),
            prompt_fingerprint=current,
            context_shape_fingerprint=None,
            model="test",
        ),
        scenario=scenario,
        tool_specs=schemas,
    ) == "cassette has no context_shape_fingerprint; re-record it"
    assert cassette_staleness(
        TrajectoryCassette(
            steps=(),
            prompt_fingerprint=current,
            context_shape_fingerprint=current_shape,
            model="test",
        ),
        scenario=scenario,
        tool_specs=schemas,
    ) is None


def test_changing_the_system_prompt_changes_its_fingerprint(
    offered, monkeypatch
) -> None:
    _, schemas = offered
    before = prompt_fingerprint(schemas)
    original = OpenAICompatibleMainAgentDecisionMaker._system_prompt
    monkeypatch.setattr(
        OpenAICompatibleMainAgentDecisionMaker,
        "_system_prompt",
        staticmethod(lambda names: original(names) + " changed"),
    )

    assert prompt_fingerprint(schemas) != before


def test_changing_a_step_menu_changes_the_trajectory_fingerprint(offered) -> None:
    _, schemas = offered
    scenario = SCENARIOS[0]
    before = trajectory_prompt_fingerprint(scenario, schemas)
    context = scenario.context.model_copy(
        update={
            "task": scenario.context.task.model_copy(
                update={"active_job_posting_id": "job-1"}
            )
        }
    )

    assert trajectory_prompt_fingerprint(
        replace(scenario, context=context), schemas
    ) != before


def test_contract_rejects_an_expected_tool_hidden_on_that_step(offered) -> None:
    _, schemas = offered
    scenario = SCENARIOS[0]
    hidden_expectation = replace(
        scenario,
        steps=(
            replace(
                scenario.steps[0],
                expect_action=None,
                expect_tool="get_saved_job",
            ),
        ),
    )

    failures = check_contract(hidden_expectation, tool_specs=schemas)
    assert len(failures) == 1
    assert "hidden by the production menu" in failures[0]


def test_changing_model_context_keys_changes_the_shape_fingerprint(
    monkeypatch,
) -> None:
    scenario = SCENARIOS[0]
    before = context_shape_fingerprint(scenario)
    context_type = type(scenario.context)
    original = context_type.model_context

    def without_phase(context):
        projection = original(context)
        task = dict(projection["task"])
        task.pop("phase")
        return {**projection, "task": task}

    monkeypatch.setattr(context_type, "model_context", without_phase)

    assert context_shape_fingerprint(scenario) != before


def test_a_known_gap_is_described_well_enough_to_act_on() -> None:
    """A marker that does not name the defect is just a disabled test."""
    for scenario in SCENARIOS:
        if scenario.known_gap is not None:
            assert len(scenario.known_gap) > 80, scenario.name


def test_the_suite_is_mostly_negative_and_not_entirely_negative() -> None:
    """Both halves have to be present or the suite is satisfiable trivially.

    All-negative passes for a model that does nothing. All-positive misses the
    failures that cost money or write durable state.
    """
    positive = sum(
        1 for item in SCENARIOS if any(step.expect_tool for step in item.steps)
    )
    negative = sum(1 for item in SCENARIOS if item.tools - {
        step.expect_tool for step in item.steps
    })
    assert positive >= 3
    assert negative >= positive


def test_daily_brief_fact_pair_is_causal_and_has_fresh_model_evidence(
    offered,
) -> None:
    """The same receipt must lead elsewhere solely because facts changed."""
    _, schemas = offered
    by_name = {scenario.name: scenario for scenario in SCENARIOS}
    overdue = by_name["overdue_brief_routes_to_the_action_list"]
    clear = by_name["clear_brief_finishes_without_opening_the_action_list"]

    assert overdue.context.user_message == clear.context.user_message
    overdue_observation = overdue.context.tool_observations[0]
    clear_observation = clear.context.tool_observations[0]
    assert overdue_observation.tool_name == clear_observation.tool_name
    assert overdue_observation.state == clear_observation.state
    assert overdue_observation.message == clear_observation.message
    assert overdue_observation.facts["overdue"] > 0
    assert clear_observation.facts["overdue"] == 0

    for scenario in (overdue, clear):
        cassette = load_cassette(scenario.name)
        assert cassette is not None, f"{scenario.name} needs a live recording"
        assert cassette_staleness(
            cassette,
            scenario=scenario,
            tool_specs=schemas,
        ) is None
        assert replay(
            scenario,
            tool_specs=schemas,
            responses=cassette.steps,
        ) == ()

    overdue_cassette = load_cassette(overdue.name)
    clear_cassette = load_cassette(clear.name)
    assert overdue_cassette.steps[0].get("tool_call", {}).get("name") == (
        "list_action_items"
    )
    assert clear_cassette.steps[0].get("tool_call") is None


def test_saved_jd_body_pair_is_causal_and_has_fresh_model_evidence(
    offered,
) -> None:
    """Identical receipts diverge only when the requested JD fact is present."""
    _, schemas = offered
    by_name = {scenario.name: scenario for scenario in SCENARIOS}
    requires_rust = by_name["saved_jd_body_drives_the_next_read_step"]
    no_language = by_name[
        "saved_jd_body_without_language_requirement_finishes"
    ]

    assert requires_rust.context.user_message == no_language.context.user_message
    positive = requires_rust.context.tool_observations[0]
    negative = no_language.context.tool_observations[0]
    assert positive.tool_name == negative.tool_name
    assert positive.state == negative.state
    assert positive.message == negative.message
    assert "Rust" in (positive.body or "")
    assert "Rust" not in (negative.body or "")

    for scenario in (requires_rust, no_language):
        cassette = load_cassette(scenario.name)
        assert cassette is not None, f"{scenario.name} needs a live recording"
        assert cassette_staleness(
            cassette,
            scenario=scenario,
            tool_specs=schemas,
        ) is None
        assert replay(
            scenario,
            tool_specs=schemas,
            responses=cassette.steps,
        ) == ()

    positive_cassette = load_cassette(requires_rust.name)
    negative_cassette = load_cassette(no_language.name)
    assert positive_cassette.steps[0].get("tool_call", {}).get("name") == (
        "match_resume_to_job"
    )
    assert negative_cassette.steps[0].get("tool_call") is None


def test_in_turn_handle_pair_is_causal_and_has_fresh_model_evidence(offered) -> None:
    """Identical turns diverge only on whether the report has a number.

    Without this pair the positive scenario proves only that the model called
    ``get_job_research`` on a turn where a number happened to be present. It
    could have been choosing from ``task.has_active_job_research_report``, or
    from ``state="job_research_ready"``, and the handle would be doing nothing.

    ``decisive_facts`` cannot supply this: it checks that a path still exists in
    the projection, so a refactor that drops the field is caught, but nothing
    counterfactual is ever run. The mirror is written by hand, like the two
    before it.

    The two sides assert different things on purpose. With a number, the policy
    is the selector, and it is a correctness claim rather than a stylistic one
    only because the active report is the *other* report: calling the tool bare
    returns report-b and answers about the wrong company.

    That distinction is not hypothetical. The first version of this pair used a
    single report, so the active id equalled the target and the bare call was
    both legal — the tool's own description offers it — and correct. The model
    called it bare on both sides, and the honest reading was not "the handle is
    ignored" but "this pair cannot tell the handle apart from the default".
    A prompt sentence would have turned it green while proving nothing.

    Without a number there is no right answer to demand: a bare call returning
    the wrong report, or saying the report cannot be reached, are both
    defensible answers to an impossible request, and that open-endedness *is*
    the problem being removed. So the mirror asserts only what must hold — a
    number nobody supplied cannot appear.
    """
    _, schemas = offered
    by_name = {scenario.name: scenario for scenario in SCENARIOS}
    numbered = by_name["a_report_made_this_turn_is_read_back_by_its_index"]
    unnumbered = by_name["a_report_made_this_turn_without_an_index_cannot_be_named"]

    assert numbered.context.user_message == unnumbered.context.user_message
    assert numbered.context.task == unnumbered.context.task
    positive = numbered.context.model_context()
    negative = unnumbered.context.model_context()
    # One key differs, and inside it one field: the handle itself.
    assert [key for key in positive if positive[key] != negative[key]] == [
        "tool_observations"
    ]
    # The stored history stays, though its job has changed. Under ordinals it
    # offset the numbering so that "read the field" and "count the observations"
    # gave different answers — the only way to make an ordinal handle
    # falsifiable. A derived handle cannot be counted to at all, so the history
    # is now simply the realistic case, and the pair's discriminating power
    # comes from the handle itself.
    for position, resource_id in enumerate(("report-a", "report-b")):
        shown = positive["tool_observations"][position]
        hidden = negative["tool_observations"][position]
        handle = shown.pop("reference")
        # ``title`` and ``description`` ride on the reference, so removing one
        # removes all three. The mirror is "no reference at all", not "a
        # reference stripped of its name", which is the state a real turn
        # would be in.
        shown.pop("title")
        shown.pop("description")
        assert (
            numbered.context.resolve_reference(
                reference=handle, kind="job_research_report"
            )
            == resource_id
        )
        assert shown == hidden
    # The active report is the *other* one, which is what makes the index the
    # only correct selector rather than the tidier of two correct ones. Without
    # this the bare call returns the right report and the pair proves nothing.
    assert positive["task"]["has_active_job_research_report"] is True
    assert numbered.context.task.active_job_research_report_id == "report-b"
    assert (
        numbered.context.tool_observations[0].resource_ref.resource_id == "report-a"
    )

    for scenario in (numbered, unnumbered):
        cassette = load_cassette(scenario.name)
        assert cassette is not None, f"{scenario.name} needs a live recording"
        assert cassette_staleness(
            cassette, scenario=scenario, tool_specs=schemas
        ) is None
        failures = replay(scenario, tool_specs=schemas, responses=cassette.steps)
        if scenario.known_gap is None:
            assert failures == ()
        else:
            # The mirror is a known gap, not a passing case: the model does
            # supply a number it was never given. Asserting the failure keeps
            # the defect visible here too, so fixing it retires both markers.
            assert failures

    # The hazard the ordinal scheme sat on, now closed: under numbers, a
    # fabricated 1 named last week's report and resolved silently. There is no
    # equivalent guess here — every handle the mirror hands out is derived, and
    # the small integers the model reached for are not among them.
    for guess in ("1", "report_1", "report_000000"):
        with pytest.raises(ValueError, match="unknown resource reference"):
            unnumbered.context.resolve_reference(
                reference=guess, kind="job_research_report"
            )

    numbered_call = load_cassette(numbered.name).steps[0].get("tool_call", {})
    assert numbered_call.get("name") == "get_job_research"
    assert (
        numbered.context.resolve_reference(
            reference=numbered_call.get("arguments", {}).get("reference"),
            kind="job_research_report",
        )
        == "report-a"
    )
    # Producer-owned titles close the remaining ambiguity: the only visible
    # handles are explicitly about other companies, so the mirror no longer
    # substitutes one of them for a report it cannot name.
    unnumbered_call = load_cassette(unnumbered.name).steps[0].get("tool_call") or {}
    assert "reference" not in unnumbered_call.get("arguments", {})
