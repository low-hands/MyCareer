"""Trajectory evaluation for the Main Agent's decision loop.

The rest of this suite scripts ``decide`` and checks the runtime. That leaves
the system prompt — 11k characters, ninety policy sentences, the largest
behavioural artifact in the project — with no coverage at all. These are the
first tests that fail when a policy stops holding.

Two levels, and the difference is the whole design:

- **Contract**, always run, no API key. Does the context we send still make the
  policy decidable, and is the forbidden tool even on the menu? Catches the two
  ways an evaluation quietly stops testing anything.
- **Replay**, run per scenario that has a cassette. Checks the decision itself.

A green contract run does NOT mean the model behaves. It means the question was
asked properly. Only a fresh recording evaluates the model, which is why
``career-agent eval trajectories --record`` exists and why cassette staleness is
reported rather than hidden.
"""

from __future__ import annotations

import pytest

from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.agent.openai_compatible_main_agent import (
    OpenAICompatibleMainAgentDecisionMaker,
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
    names, _ = offered
    assert check_contract(scenario, offered_tools=names) == ()


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
    current = prompt_fingerprint(schemas)
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
        "cassette prompt_fingerprint does not match the current system prompt; "
        "re-record it"
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
