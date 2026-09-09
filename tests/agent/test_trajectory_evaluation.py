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
import threading

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

import career_agent.evaluation.trajectory as trajectory_module
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    CareerProfileBudgets,
    ConversationMessageContext,
)
from career_agent.agent.openai_compatible_client import (
    AgentWorkerError,
    OpenAICompatibleAgentConfig,
)
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
    TrajectoryStep,
    cassette_staleness,
    check_contract,
    check_step,
    check_step_quality,
    context_shape_fingerprint,
    known_gap_reproduction,
    load_cassette,
    prompt_fingerprint,
    record,
    record_catalogue,
    replay,
    replay_budget_cassette_pair,
    replay_cassette,
    replay_quality,
    minimum_detectable_regression,
    quality_shortfall,
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


def test_budget_change_validation_is_paired_by_cassette_sample(
    offered,
    monkeypatch,
) -> None:
    _, tool_specs = offered
    source = next(scenario for scenario in SCENARIOS if not scenario.known_gap)
    baseline = replace(
        source,
        context=source.context.model_copy(
            update={
                "career_profile_budgets": CareerProfileBudgets(
                    records_input_units=5_040,
                )
            }
        ),
    )
    candidate = replace(
        source,
        context=source.context.model_copy(
            update={
                "career_profile_budgets": CareerProfileBudgets(
                    records_input_units=2_800,
                )
            }
        ),
    )
    cassette = TrajectoryCassette(
        steps=({"choices": []},),
        prompt_fingerprint=None,
        context_shape_fingerprint=None,
        model="test",
    )
    monkeypatch.setattr(
        trajectory_module,
        "cassette_staleness",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        trajectory_module,
        "replay_cassette",
        lambda scenario, **kwargs: (
            ((),)
            if (
                scenario.context.career_profile_budgets.records_input_units
                == 5_040
            )
            else (("candidate regression",),)
        ),
    )
    monkeypatch.setattr(
        trajectory_module,
        "replay_quality",
        lambda *args, **kwargs: (),
    )

    result = replay_budget_cassette_pair(
        baseline_scenario=baseline,
        baseline_cassette=cassette,
        candidate_scenario=candidate,
        candidate_cassette=cassette,
        tool_specs=tool_specs,
    )

    assert result.baseline_budgets == (5_040, 800, 600)
    assert result.candidate_budgets == (2_800, 800, 600)
    assert result.pair_count == 1
    assert result.regressed_pair_count == 1
    assert result.candidate_noninferior is False


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
    "conversation_store",
)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda item: item.name)
def test_a_scenario_asks_a_question_the_model_could_answer(scenario, offered) -> None:
    """The contract level: the scenario is not vacuous and not stale."""
    _, schemas = offered
    assert check_contract(scenario, tool_specs=schemas) == ()


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda item: item.name)
def test_the_recorded_decision_follows_the_policy(scenario, offered) -> None:
    """Hard replay level: every recorded sample must preserve invariants."""
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
    sample_failures = replay_cassette(
        scenario, tool_specs=schemas, cassette=cassette
    )
    failures = tuple(
        failure for sample in sample_failures for failure in sample
    )
    if scenario.known_gap is not None:
        reproduction = known_gap_reproduction(sample_failures)
        if reproduction == "resolved":
            pytest.fail(
                f"{scenario.name} now passes all samples; remove its "
                f"known_gap: {scenario.known_gap}"
            )
        reason = scenario.known_gap
        if reproduction == "intermittent":
            failed = sum(bool(sample) for sample in sample_failures)
            reason = (
                f"{reason} (intermittent reproduction: "
                f"{failed}/{len(sample_failures)} samples failed)"
            )
        pytest.xfail(reason)
    assert all(not sample for sample in sample_failures)


@pytest.mark.parametrize(
    "scenario",
    tuple(item for item in SCENARIOS if item.has_quality_assertions),
    ids=lambda item: item.name,
)
def test_the_recorded_quality_stays_above_its_rate_floor(scenario, offered) -> None:
    """Quality replay is independent of hard-rule known-gap disposition."""
    _, schemas = offered
    cassette = load_cassette(scenario.name)
    assert cassette is not None, (
        f"no recording for '{scenario.name}'; run "
        "`career-agent eval trajectories --record` against a live model"
    )
    stale = cassette_staleness(
        cassette,
        scenario=scenario,
        tool_specs=schemas,
    )
    assert stale is None, f"{scenario.name}: {stale}"
    graded = replay_quality(scenario, tool_specs=schemas, cassette=cassette)
    shortfall = quality_shortfall(scenario, graded)
    assert shortfall is None, shortfall


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

    Every recording is inspected. Ordinary scenarios retain one sample; the
    few conclusions that depend on one exact behaviour declare three or more.
    """
    checked = 0
    for scenario in SCENARIOS:
        cassette = load_cassette(scenario.name)
        if cassette is None:
            continue
        for sample_index, recording in enumerate(cassette.recordings, start=1):
            for index, step in enumerate(recording):
                try:
                    decision = json.loads(step["content"])
                except (TypeError, ValueError, KeyError):
                    continue
                message = (decision.get("message") or "").strip()
                if decision.get("action") != "final" or not message:
                    continue
                label = f"{scenario.name}[sample={sample_index},step={index}]"
                # A reply is never JSON and always reads as language, whatever
                # the turn produced.
                assert not message.startswith("{"), label
                assert any(mark in message for mark in "。！？.!?"), label
                # What the turn was holding when it answered: the seeded
                # context plus every observation fed back through this step.
                # Seeded ones matter — several scenarios put the card-backed
                # result in the context rather than on a step.
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
        "cassette prompt_fingerprint does not match the current stable "
        "prompt/tool universe; re-record it"
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


def test_a_legacy_cassette_is_one_recording(tmp_path: Path) -> None:
    (tmp_path / "legacy.json").write_text(
        json.dumps(
            {
                "steps": [{"content": "one"}],
                "prompt_fingerprint": "prompt",
                "context_shape_fingerprint": "shape",
                "model": "test",
            }
        )
    )

    cassette = load_cassette("legacy", root=tmp_path)

    assert cassette is not None
    assert cassette.sample_count == 1
    assert cassette.recordings == (({"content": "one"},),)


def test_a_policy_critical_scenario_rejects_too_few_samples(offered) -> None:
    _, schemas = offered
    scenario = replace(SCENARIOS[0], recording_samples=3)
    cassette = TrajectoryCassette(
        steps=(),
        prompt_fingerprint=trajectory_prompt_fingerprint(scenario, schemas),
        context_shape_fingerprint=context_shape_fingerprint(scenario),
        model="test",
    )

    assert cassette_staleness(
        cassette, scenario=scenario, tool_specs=schemas
    ) == "cassette has 1 sample(s), but the scenario requires 3; re-record it"


def test_record_captures_independent_samples_and_retries_transport_errors(
    offered, monkeypatch, tmp_path: Path
) -> None:
    _, schemas = offered
    scenario = replace(SCENARIOS[0], recording_samples=3)
    calls = 0
    delays = []
    real_maker = OpenAICompatibleMainAgentDecisionMaker

    class FlakyMaker:
        _system_prompt = staticmethod(real_maker._system_prompt)

        def __init__(self, config) -> None:
            pass

        def decide(self, context, tool_specs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise AgentWorkerError(
                    "MAIN_AGENT_TRANSPORT_ERROR",
                    "temporary",
                    retryable=True,
                )
            return AgentDecision(action="ask_user", message="请补充城市。")

    monkeypatch.setattr(
        "career_agent.evaluation.trajectory."
        "OpenAICompatibleMainAgentDecisionMaker",
        FlakyMaker,
    )
    config = OpenAICompatibleAgentConfig(
        endpoint="https://example.invalid/v1/chat/completions",
        api_key="test",
        model="test",
    )

    path = record(
        scenario,
        tool_specs=schemas,
        config=config,
        root=tmp_path,
        retry_delay_seconds=0.25,
        sleeper=delays.append,
    )
    cassette = load_cassette(scenario.name, root=tmp_path)

    assert path.exists()
    assert calls == 4
    assert delays == [0.25]
    assert cassette is not None
    assert cassette.sample_count == 3
    assert [len(sample) for sample in cassette.recordings] == [1, 1, 1]
    monkeypatch.setattr(
        "career_agent.evaluation.trajectory."
        "OpenAICompatibleMainAgentDecisionMaker",
        real_maker,
    )
    assert replay_cassette(scenario, tool_specs=schemas, cassette=cassette) == (
        (),
        (),
        (),
    )


def test_record_does_not_retry_a_nonretryable_model_failure(
    offered, monkeypatch, tmp_path: Path
) -> None:
    _, schemas = offered
    scenario = SCENARIOS[0]
    calls = 0

    class BrokenMaker:
        def __init__(self, config) -> None:
            pass

        def decide(self, context, tool_specs):
            nonlocal calls
            calls += 1
            raise AgentWorkerError(
                "MAIN_AGENT_INVALID_RESPONSE",
                "invalid",
                retryable=False,
            )

    monkeypatch.setattr(
        "career_agent.evaluation.trajectory."
        "OpenAICompatibleMainAgentDecisionMaker",
        BrokenMaker,
    )
    config = OpenAICompatibleAgentConfig(
        endpoint="https://example.invalid/v1/chat/completions",
        api_key="test",
        model="test",
    )

    with pytest.raises(AgentWorkerError, match="invalid"):
        record(
            scenario,
            tool_specs=schemas,
            config=config,
            root=tmp_path,
            sleeper=lambda _: pytest.fail("must not sleep"),
        )

    assert calls == 1
    assert not (tmp_path / f"{scenario.name}.json").exists()


def test_record_runs_independent_samples_concurrently(
    offered, monkeypatch, tmp_path: Path
) -> None:
    _, schemas = offered
    scenario = replace(SCENARIOS[0], recording_samples=3)
    barrier = threading.Barrier(3)
    real_maker = OpenAICompatibleMainAgentDecisionMaker

    class GatedMaker:
        _system_prompt = staticmethod(real_maker._system_prompt)

        def __init__(self, config) -> None:
            pass

        def decide(self, context, tool_specs):
            barrier.wait(timeout=2)
            return AgentDecision(action="ask_user", message="请补充城市。")

    monkeypatch.setattr(
        "career_agent.evaluation.trajectory."
        "OpenAICompatibleMainAgentDecisionMaker",
        GatedMaker,
    )
    config = OpenAICompatibleAgentConfig(
        endpoint="https://example.invalid/v1/chat/completions",
        api_key="test",
        model="test",
    )

    path = record(
        scenario,
        tool_specs=schemas,
        config=config,
        root=tmp_path,
        max_workers=3,
        sleeper=lambda _: pytest.fail("must not sleep"),
    )
    cassette = load_cassette(scenario.name, root=tmp_path)

    assert path.exists()
    assert cassette is not None
    assert cassette.sample_count == 3


def test_record_adds_retry_jitter_when_requested(
    offered, monkeypatch, tmp_path: Path
) -> None:
    _, schemas = offered
    scenario = SCENARIOS[0]
    delays = []
    calls = 0
    real_maker = OpenAICompatibleMainAgentDecisionMaker

    class FlakyMaker:
        _system_prompt = staticmethod(real_maker._system_prompt)

        def __init__(self, config) -> None:
            pass

        def decide(self, context, tool_specs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise AgentWorkerError(
                    "MAIN_AGENT_TRANSPORT_ERROR",
                    "temporary",
                    retryable=True,
                )
            return AgentDecision(action="ask_user", message="请补充城市。")

    monkeypatch.setattr(
        "career_agent.evaluation.trajectory."
        "OpenAICompatibleMainAgentDecisionMaker",
        FlakyMaker,
    )
    monkeypatch.setattr(
        "career_agent.evaluation.trajectory.random.random",
        lambda: 1.0,
    )
    config = OpenAICompatibleAgentConfig(
        endpoint="https://example.invalid/v1/chat/completions",
        api_key="test",
        model="test",
    )

    record(
        scenario,
        tool_specs=schemas,
        config=config,
        root=tmp_path,
        retry_delay_seconds=0.25,
        sleeper=delays.append,
        jitter=True,
    )

    assert delays == [0.375]


def test_record_catalogue_skips_a_current_cassette(
    offered, monkeypatch, tmp_path: Path
) -> None:
    _, schemas = offered
    scenario = replace(SCENARIOS[0], name="current_catalogue_cassette")
    calls = 0
    real_maker = OpenAICompatibleMainAgentDecisionMaker

    class CountingMaker:
        _system_prompt = staticmethod(real_maker._system_prompt)

        def __init__(self, config) -> None:
            pass

        def decide(self, context, tool_specs):
            nonlocal calls
            calls += 1
            return AgentDecision(action="ask_user", message="请补充城市。")

    monkeypatch.setattr(
        "career_agent.evaluation.trajectory."
        "OpenAICompatibleMainAgentDecisionMaker",
        CountingMaker,
    )
    config = OpenAICompatibleAgentConfig(
        endpoint="https://example.invalid/v1/chat/completions",
        api_key="test",
        model="test",
    )
    record(
        scenario,
        tool_specs=schemas,
        config=config,
        root=tmp_path,
    )
    assert calls == 1

    record_catalogue(
        (scenario,),
        tool_specs=schemas,
        config=config,
        root=tmp_path,
        jobs=1,
    )
    assert calls == 1

    record_catalogue(
        (scenario,),
        tool_specs=schemas,
        config=config,
        root=tmp_path,
        jobs=1,
        force=True,
    )
    assert calls == 2


def test_record_catalogue_keeps_finished_neighbours_when_one_scenario_fails(
    offered, monkeypatch, tmp_path: Path
) -> None:
    _, schemas = offered
    healthy = replace(SCENARIOS[0], name="catalogue_neighbour_healthy")
    broken = replace(SCENARIOS[0], name="catalogue_neighbour_broken")
    calls = 0
    real_maker = OpenAICompatibleMainAgentDecisionMaker

    class MixedMaker:
        _system_prompt = staticmethod(real_maker._system_prompt)

        def __init__(self, config) -> None:
            pass

        def decide(self, context, tool_specs):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise AgentWorkerError(
                    "MAIN_AGENT_INVALID_RESPONSE",
                    "invalid",
                    retryable=False,
                )
            return AgentDecision(action="ask_user", message="请补充城市。")

    monkeypatch.setattr(
        "career_agent.evaluation.trajectory."
        "OpenAICompatibleMainAgentDecisionMaker",
        MixedMaker,
    )
    config = OpenAICompatibleAgentConfig(
        endpoint="https://example.invalid/v1/chat/completions",
        api_key="test",
        model="test",
    )

    with pytest.raises(AgentWorkerError, match="invalid"):
        record_catalogue(
            (healthy, broken),
            tool_specs=schemas,
            config=config,
            root=tmp_path,
            jobs=1,
        )

    assert (tmp_path / f"{healthy.name}.json").exists()
    assert not (tmp_path / f"{broken.name}.json").exists()


def test_changing_the_system_prompt_changes_its_fingerprint(
    offered, monkeypatch
) -> None:
    _, schemas = offered
    before = prompt_fingerprint(schemas)
    original = OpenAICompatibleMainAgentDecisionMaker._system_prompt
    monkeypatch.setattr(
        OpenAICompatibleMainAgentDecisionMaker,
        "_system_prompt",
        staticmethod(lambda: original() + " changed"),
    )

    assert prompt_fingerprint(schemas) != before


def test_changing_task_state_does_not_change_the_tool_prefix_fingerprint(offered) -> None:
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
    ) == before


def test_contract_accepts_a_state_gated_tool_from_the_stable_universe(offered) -> None:
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
    assert failures == ()


def test_changing_model_context_keys_changes_the_shape_fingerprint(
    monkeypatch,
) -> None:
    scenario = SCENARIOS[0]
    before = context_shape_fingerprint(scenario)
    context_type = type(scenario.context)
    original = context_type.model_context

    def without_default_city(context):
        projection = original(context)
        career_profile = dict(projection["career_profile"])
        career_profile.pop("memory/profile.md")
        return {**projection, "career_profile": career_profile}

    monkeypatch.setattr(context_type, "model_context", without_default_city)

    assert context_shape_fingerprint(scenario) != before


def test_native_chat_roles_change_the_shape_fingerprint() -> None:
    scenario = SCENARIOS[0]
    before = context_shape_fingerprint(scenario)
    context = scenario.context.model_copy(
        update={
            "recent_messages": (
                ConversationMessageContext(
                    role="assistant",
                    content="A prior native turn.",
                    created_at=datetime(2026, 9, 6, tzinfo=timezone.utc),
                ),
            )
        }
    )

    assert context_shape_fingerprint(replace(scenario, context=context)) != before


def test_a_known_gap_is_described_well_enough_to_act_on() -> None:
    """A marker that does not name the defect is just a disabled test."""
    for scenario in SCENARIOS:
        if scenario.known_gap is not None:
            assert len(scenario.known_gap) > 80, scenario.name


@pytest.mark.parametrize(
    ("sample_failures", "expected"),
    (
        ((("wrong tool",), ("wrong action",)), "stable"),
        ((("wrong tool",), ()), "intermittent"),
        (((), ()), "resolved"),
        ((), "resolved"),
    ),
)
def test_known_gap_reproduction_distinguishes_partial_recovery(
    sample_failures, expected
) -> None:
    assert known_gap_reproduction(sample_failures) == expected


def test_quality_message_facts_accept_wording_variants() -> None:
    scenario = "message-facts"
    step = TrajectoryStep(
        quality_message_contains_any=(
            frozenset({"3 份", "三份", "更早"}),
            frozenset({"未列出", "无法按引用"}),
        )
    )

    assert check_step_quality(
        step,
        AgentDecision(
            action="final",
            message="另有更早的调研未列出，当前不能直接读取。",
        ),
        scenario=scenario,
        index=0,
    ) == ()
    assert check_step_quality(
        step,
        AgentDecision(action="final", message="当前列表里没有 Shopee。"),
        scenario=scenario,
        index=0,
    )


def test_quality_contract_requires_a_rate_and_pins_its_denominator(
    offered,
) -> None:
    _, schemas = offered
    quality_step = TrajectoryStep(
        quality_message_contains_any=(frozenset({"更早"}),)
    )
    with pytest.raises(ValueError, match="quality_min_pass_rate"):
        replace(SCENARIOS[0], steps=(quality_step,))
    with pytest.raises(ValueError, match="greater than 0"):
        replace(
            SCENARIOS[0],
            steps=(quality_step,),
            quality_min_pass_rate=0,
        )

    scenario = replace(
        SCENARIOS[0],
        steps=(quality_step,),
        recording_samples=5,
        quality_min_pass_rate=0.6,
    )
    cassette = TrajectoryCassette(
        steps=(),
        prompt_fingerprint=trajectory_prompt_fingerprint(scenario, schemas),
        context_shape_fingerprint=context_shape_fingerprint(scenario),
        model="test",
        samples=(({},),) * 4,
    )
    assert "requires exactly 5" in cassette_staleness(
        cassette, scenario=scenario, tool_specs=schemas
    )
    three_sample_scenario = replace(scenario, recording_samples=3)
    config = OpenAICompatibleAgentConfig(
        endpoint="https://example.invalid/v1/chat/completions",
        api_key="test",
        model="test",
    )
    with pytest.raises(ValueError, match="exactly their declared sample count"):
        record(
            three_sample_scenario,
            tool_specs=schemas,
            config=config,
            sample_count=4,
        )


def test_quality_threshold_is_a_rate_and_reports_uncertainty() -> None:
    scenario = replace(
        SCENARIOS[0],
        steps=(
            TrajectoryStep(
                quality_message_contains_any=(frozenset({"更早"}),)
            ),
        ),
        recording_samples=5,
        quality_min_pass_rate=0.6,
    )

    assert (
        quality_shortfall(scenario, ((), (), (), ("thin",), ("thin",)))
        is None
    )
    assert "40.0%" in quality_shortfall(
        scenario, ((), (), ("thin",), ("thin",), ("thin",))
    )


def test_quality_mde_names_the_blind_spot_the_floor_cannot_see() -> None:
    assert minimum_detectable_regression(3, 0.6) == pytest.approx(1 / 3)
    assert minimum_detectable_regression(5, 0.6) == pytest.approx(0.4)
    assert minimum_detectable_regression(1, 0.6) == 0.0
    assert minimum_detectable_regression(0, 0.6) is None


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
    # With handles kept out of system control, every fresh positive sample binds
    # the matching tool-result reference rather than an older footer handle.
    assert known_gap_reproduction(
        replay_cassette(
            numbered, tool_specs=schemas, cassette=load_cassette(numbered.name)
        )
    ) == "resolved"
    # Native tool results make the positive binding reliable. The synthetic
    # mirror now also rejects every differently titled historical handle in all
    # three fresh samples, so the former intermittent gap is resolved.
    assert known_gap_reproduction(
        replay_cassette(
            unnumbered, tool_specs=schemas, cassette=load_cassette(unnumbered.name)
        )
    ) == "resolved"

    # The hazard the ordinal scheme sat on, now closed: under numbers, a
    # fabricated 1 named last week's report and resolved silently. There is no
    # equivalent guess here — every handle the mirror hands out is derived, and
    # the small integers the model reached for are not among them.
    for guess in ("1", "report_1", "report_000000"):
        with pytest.raises(ValueError, match="unknown resource reference"):
            unnumbered.context.resolve_reference(
                reference=guess, kind="job_research_report"
            )

    resolved = []
    for sample in load_cassette(numbered.name).recordings:
        numbered_call = sample[0].get("tool_call", {})
        assert numbered_call.get("name") == "get_job_research"
        resolved.append(
            numbered.context.resolve_reference(
                reference=numbered_call.get("arguments", {}).get("reference"),
                kind="job_research_report",
            )
        )
    assert resolved.count("report-a") == numbered.recording_samples
    assert resolved.count("report-h1") == 0
    # Every fresh mirror sample now uses the grounded saved-job selector; none
    # borrows an older handle despite those handles remaining visible.
    borrowed = []
    grounded_selector_count = 0
    for sample in load_cassette(unnumbered.name).recordings:
        arguments = (sample[0].get("tool_call") or {}).get("arguments", {})
        reference = arguments.get("reference")
        if reference is None:
            assert arguments.get("selection_index") == 1
            grounded_selector_count += 1
        else:
            borrowed.append(reference)
    assert borrowed == []
    assert grounded_selector_count == unnumbered.recording_samples
    assert {
        unnumbered.context.resolve_reference(
            reference=handle,
            kind="job_research_report",
        )
        for handle in borrowed
    } == set()
