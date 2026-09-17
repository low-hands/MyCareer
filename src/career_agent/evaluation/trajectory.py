"""Scenarios that pin what the Main Agent decides, not what it does after.

The evaluation runs at two levels, because only one of them can run without a
model:

**Contract level** — always runs, needs no API key. Builds the exact context a
scenario would send and checks that the decision is even *decidable* from it:
the facts the policy turns on are present in the projection, and every expected
tool exists in that step's profile. Runtime argument projection,
not prompt-shape mutation, owns task-state preconditions.

**Replay level** — runs for scenarios that have a recorded model response.
Checks the decision itself against the scenario's expectations.

The split matters and should not be blurred: passing at contract level says the
question was asked properly, not that the model answered it well. Only a fresh
recording says that. ``record`` refreshes them against the live model.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from career_agent.agent.decision_messages import project_decision_messages
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    DecisionObservation,
    MainAgentContext,
    TOOL_PROFILE_NAMES,
    append_decision_observation,
)
from career_agent.agent.openai_compatible_client import (
    AgentConfigurationError,
    AgentWorkerError,
    OpenAICompatibleAgentConfig,
)
from career_agent.agent.openai_compatible_main_agent import (
    OpenAICompatibleMainAgentDecisionMaker,
)
from career_agent.agent.tool_profiles import profile_schemas

CASSETTE_ROOT = Path(__file__).resolve().parents[3] / "evals" / "main_agent"
_MAX_RECORD_JOBS = 32
_RECORD_MAKERS = threading.local()


@dataclass(frozen=True)
class TrajectoryStep:
    """One decision the model is asked to make inside a scenario.

    Each step is a declared decision snapshot. ``observation`` and ``task_update``
    supply the result and reducer state for that snapshot, independently of the
    previous model response. Tools are not executed; these probes do not measure
    end-to-end runtime success.
    """

    expect_action: str | None = None
    expect_tool: str | None = None
    expect_tools: frozenset[str] = frozenset()
    expect_message: str | None = None
    forbid_message_contains: frozenset[str] = frozenset()
    """Substrings the reply must not carry, for delivery the runtime owns.

    A card-backed report reaches the reader through its entity. Restating the
    body in the reply duplicates it into a window that will keep the row for
    every later turn, and puts a second, drifting copy next to the one the
    reader can reopen. Exact-matching prose would be brittle, so the assertion
    names only what must be absent.
    """

    forbid_final_message_contains: frozenset[str] = frozenset()
    """Substrings that may be asked about but must not be asserted as an answer."""

    quality_message_contains_any: tuple[frozenset[str], ...] = ()
    """Fragments a *good* reply carries, graded by rate rather than per sample.

    The hard assertions on this step are invariants: a single violation is a
    defect, so they are gated with ``pass^k`` — every sample must satisfy them.
    Some things a reply should do are not invariants. Volunteering that the
    catalogue is truncated makes the answer more complete; omitting it leaves
    the answer correct but thinner, and the model does it about three times in
    five.

    Treating that as a hard assertion would make the suite red on a working
    system. Deleting it would stop measuring the thing entirely — the failure
    mode this project keeps returning to, where a check is removed rather than
    re-aimed. So it is kept and graded: the scenario declares the floor its
    recorded rate must not fall below, and a regression to one-in-five is a
    failure while three-in-five is the recorded status quo.
    """

    quality_report_unavailable: bool = False
    forbid_tools: frozenset[str] = frozenset()
    expect_arguments: Mapping[str, Any] = field(default_factory=dict)
    """Argument values the call must carry, checked as a subset.

    ``expect_tool`` says which capability; this says the model selected the
    right thing with it. A subset rather than equality: the assertion is about
    the selector the policy turns on, and pinning every other argument would
    make an unrelated schema change read as a policy failure.
    """

    expect_argument_contains: Mapping[str, str] = field(default_factory=dict)
    """Required substring for a string-valued tool argument."""

    expect_nonempty_string_arguments: frozenset[str] = frozenset()

    forbid_non_null_arguments: frozenset[str] = frozenset()
    """Argument names for which the call must not invent a value.

    For a mirror scenario where the *right* behaviour is genuinely open. Take
    the handle away from an observation and the model has no good
    move left — call the read tool bare and hope the active report is the one it
    meant, or ask. Neither is wrong, so ``expect_tool`` would be asserting a
    preference rather than a policy. What must hold is narrower and real: it
    cannot produce a number it was never given.
    """

    observation: DecisionObservation | None = None
    task_update: Mapping[str, Any] = field(default_factory=dict)
    user_message: str | None = None


@dataclass(frozen=True)
class TrajectoryScenario:
    """One policy sentence, turned into something that can fail.

    ``policy`` quotes the rule from the system prompt that this scenario exists
    to hold. Keeping the quote here is deliberate: when a scenario fails, the
    thing to look at is whether the rule still says what the scenario assumed.
    """

    name: str
    policy: str
    context: MainAgentContext
    steps: tuple[TrajectoryStep, ...]
    recording_samples: int = 1
    """How many live recordings a policy-critical scenario requires."""

    quality_min_pass_rate: float | None = None
    """Observed pass-rate floor for the quality assertions.

    ``None`` when the scenario declares none. Quality scenarios pin their
    sample count, so changing the denominator cannot silently weaken this rate.

    A floor rather than a target. ``pass^k`` is the right gate for an invariant
    and the wrong one for a quality property, but so is no gate at all: a
    property nobody measures is a property that regresses unnoticed.
    """
    known_gap: str | None = None
    """A defect this scenario currently exposes, named so the suite stays green.

    Set when the scenario is right and the system is wrong. The replay is then
    expected to fail; if it starts passing, that is reported rather than
    silently absorbed, so fixing the defect retires the marker. Never set it to
    quiet a scenario whose own expectations are wrong — fix the scenario.
    """

    decisive_facts: tuple[str, ...] = ()
    """Paths into ``model_context()`` the policy turns on, as dotted strings.

    Checked for presence at contract level. A policy that says "ask for the city
    rather than guessing" is only testable if the projection actually carries
    whether a city is known; if a refactor drops that field the model starts
    guessing and no behavioural test would say why.
    """

    @property
    def tools(self) -> frozenset[str]:
        expected = {step.expect_tool for step in self.steps if step.expect_tool}
        expected.update(tool for step in self.steps for tool in step.expect_tools)
        forbidden = set().union(*(step.forbid_tools for step in self.steps))
        return frozenset(expected | forbidden)

    def __post_init__(self) -> None:
        if not 1 <= self.recording_samples <= 5:
            raise ValueError(
                "trajectory recording_samples must be between 1 and 5"
            )
        declares_quality = self.has_quality_assertions
        if declares_quality and self.quality_min_pass_rate is None:
            raise ValueError(
                "a scenario with quality assertions must declare a "
                "quality_min_pass_rate; "
                "an ungated quality assertion is one nobody is measuring"
            )
        if self.quality_min_pass_rate is not None and not (
            0 < self.quality_min_pass_rate <= 1
        ):
            raise ValueError(
                "quality_min_pass_rate must be greater than 0 and at most 1"
            )
        if self.quality_min_pass_rate is not None and not declares_quality:
            raise ValueError(
                "quality_min_pass_rate is declared but no step asserts a "
                "quality property"
            )

    @property
    def has_quality_assertions(self) -> bool:
        return any(
            step.quality_message_contains_any or step.quality_report_unavailable
            for step in self.steps
        )


class ReplayClient:
    """Serves recorded model responses and captures what was sent.

    Shaped like the OpenAI client the decision maker holds, down to the nesting,
    because the point is to exercise the real ``decide`` — its prompt assembly,
    its tool normalisation, and its response parsing — with only the network
    replaced.
    """

    def __init__(self, responses: Sequence[Mapping[str, Any]]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.completions = _ReplayCompletions(self)
        self.chat = _ReplayChat(self)

    def _next(self, kwargs: dict[str, Any]) -> Any:
        self.requests.append(kwargs)
        if not self._responses:
            raise AssertionError(
                "the cassette ran out of responses before the scenario ran out "
                "of steps; re-record it"
            )
        return _as_response(self._responses.pop(0))


class _ReplayCompletions:
    def __init__(self, client: ReplayClient) -> None:
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._next(kwargs)


class _ReplayChat:
    def __init__(self, client: ReplayClient) -> None:
        self.completions = client.completions


def _as_response(recorded: Mapping[str, Any]) -> Any:
    """Rebuild the duck-typed shape the decision maker reads off a response."""
    call = recorded.get("tool_call")
    if call is not None:
        function = type(
            "Function",
            (),
            {
                "name": call["name"],
                "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False),
            },
        )()
        tool_call = type("ToolCall", (), {"function": function})()
        message = type("Message", (), {"content": None, "tool_calls": [tool_call]})()
    else:
        message = type(
            "Message", (), {"content": recorded.get("content", ""), "tool_calls": []}
        )()
    choice = type("Choice", (), {"message": message})()
    return type("Response", (), {"choices": [choice]})()


@dataclass(frozen=True)
class StepOutcome:
    decision: AgentDecision
    failures: tuple[str, ...]


@dataclass(frozen=True)
class ScenarioOutcome:
    scenario: str
    policy: str
    replayed: bool
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class TrajectoryCassette:
    """Recorded decisions plus the prompt identity they were made under."""

    steps: tuple[dict[str, Any], ...]
    prompt_fingerprint: str | None
    context_shape_fingerprint: str | None
    model: str | None
    samples: tuple[tuple[dict[str, Any], ...], ...] = ()

    @property
    def recordings(self) -> tuple[tuple[dict[str, Any], ...], ...]:
        """Every independent recording, including legacy single-step files."""

        return self.samples or (self.steps,)

    @property
    def sample_count(self) -> int:
        return len(self.recordings)


@dataclass(frozen=True)
class PairedBudgetCassetteReplay:
    """Direct paired replay result for one explicit budget change."""

    baseline_budgets: tuple[int, int, int]
    candidate_budgets: tuple[int, int, int]
    pair_count: int
    baseline_pass_count: int
    candidate_pass_count: int
    regressed_pair_count: int
    improved_pair_count: int

    @property
    def candidate_noninferior(self) -> bool:
        return self.regressed_pair_count == 0


def cassette_path(name: str, *, root: Path | None = None) -> Path:
    return (root or CASSETTE_ROOT) / f"{name}.json"


def prompt_fingerprint(tool_specs: tuple[dict[str, Any], ...]) -> str:
    """Hash the exact system prompt and schemas sent for one decision."""
    prompt = OpenAICompatibleMainAgentDecisionMaker._system_prompt()
    encoded = json.dumps(
        {"system_prompt": prompt, "tool_specs": tool_specs},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def context_shape_fingerprint(scenario: TrajectoryScenario) -> str:
    """Hash fixture values, authority boundaries and native-message layout."""
    context = scenario.context
    step_shapes = []
    for index, step in enumerate(scenario.steps):
        context = _advance(context, step)
        projection = project_decision_messages(context)
        messages = projection.messages(
            system_prompt="[policy]", spotlight_nonce="0" * 32
        )
        step_shapes.append(
            {
                "step": index,
                "messages": messages,
                "control_paths": sorted(_key_paths(projection.control)),
                "data_paths": sorted(_key_paths(projection.data)),
                "stable_data_paths": sorted(
                    _key_paths(projection.stable_data)
                ),
                "volatile_data_paths": sorted(
                    _key_paths(projection.volatile_data)
                ),
                "turn_observation_paths": sorted(
                    _key_paths({"tool_observations": projection.turn_observations})
                ),
                "message_roles": [message["role"] for message in messages],
                "recent_resource_footers": [
                    "\n[runtime resources:" in message["content"]
                    for message in projection.recent_messages
                ],
            }
        )
    encoded = json.dumps(
        step_shapes,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _key_paths(value: Any, path: str = "$") -> set[str]:
    paths = {path}
    if isinstance(value, Mapping):
        for key, child in value.items():
            paths.update(_key_paths(child, f"{path}.{key}"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        item_path = f"{path}[]"
        paths.add(item_path)
        for child in value:
            paths.update(_key_paths(child, item_path))
    return paths


def load_cassette(
    name: str, *, root: Path | None = None
) -> TrajectoryCassette | None:
    path = cassette_path(name, root=root)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    raw_samples = payload.get("samples")
    samples = (
        tuple(
            tuple(sample["steps"] if isinstance(sample, dict) else sample)
            for sample in raw_samples
        )
        if isinstance(raw_samples, list) and raw_samples
        else ()
    )
    steps = samples[0] if samples else tuple(payload["steps"])
    return TrajectoryCassette(
        steps=steps,
        prompt_fingerprint=payload.get("prompt_fingerprint"),
        context_shape_fingerprint=payload.get("context_shape_fingerprint"),
        model=payload.get("model"),
        samples=samples,
    )


def cassette_staleness(
    cassette: TrajectoryCassette,
    *,
    scenario: TrajectoryScenario,
    tool_specs: tuple[dict[str, Any], ...],
) -> str | None:
    """Explain why a recording cannot represent the current prompt."""
    current = trajectory_prompt_fingerprint(scenario, tool_specs)
    if cassette.prompt_fingerprint is None:
        return "cassette has no prompt_fingerprint; re-record it"
    if cassette.prompt_fingerprint != current:
        return (
            "cassette prompt_fingerprint does not match the current stable "
            "prompt/tool universe; re-record it"
        )
    current_shape = context_shape_fingerprint(scenario)
    if cassette.context_shape_fingerprint is None:
        return "cassette has no context_shape_fingerprint; re-record it"
    if cassette.context_shape_fingerprint != current_shape:
        return (
            "cassette context_shape_fingerprint does not match the current "
            "model_context projection; re-record it"
        )
    if scenario.has_quality_assertions and (
        cassette.sample_count != scenario.recording_samples
    ):
        return (
            f"cassette has {cassette.sample_count} sample(s), but quality scenario "
            f"requires exactly {scenario.recording_samples}; re-record it"
        )
    if cassette.sample_count < scenario.recording_samples:
        return (
            f"cassette has {cassette.sample_count} sample(s), but the scenario "
            f"requires {scenario.recording_samples}; re-record it"
        )
    return None


def check_contract(
    scenario: TrajectoryScenario, *, tool_specs: tuple[dict[str, Any], ...]
) -> tuple[str, ...]:
    """Whether the scenario asks a question the model could answer.

    Runs without a model and catches two invalid test shapes: the context no
    longer carries the fact the policy turns on, or a step expects a tool that
    is not offered under its active profile.
    """
    failures = []
    projection = scenario.context.model_context()
    for path in scenario.decisive_facts:
        if not _has_path(projection, path):
            failures.append(
                f"{scenario.name}: the projection has no '{path}', so this "
                "policy is no longer decidable from what the model is sent"
            )
    context = scenario.context
    for index, step in enumerate(scenario.steps):
        context = _advance(context, step)
        offered = {
            spec["function"]["name"]
            for spec in profile_schemas(context.task.tool_profile, tool_specs)
        }
        expected = step.expect_tools | (
            {step.expect_tool} if step.expect_tool is not None else set()
        )
        for name in sorted(expected - offered):
            failures.append(
                f"{scenario.name}[{index}]: expected tool "
                f"'{name}' is absent from the {context.task.tool_profile} profile"
            )
    return tuple(failures)


def _has_path(payload: Any, path: str) -> bool:
    """Whether a dotted path resolves, treating an absent key as the failure.

    A present key holding ``None`` counts: "the city is not known" is itself the
    fact several policies turn on, so requiring a truthy value would reject
    exactly the contexts worth testing.
    """
    node = payload
    parts = path.split(".")
    for index, part in enumerate(parts):
        if isinstance(node, Mapping):
            remainder = ".".join(parts[index:])
            if remainder in node:
                return True
            if part not in node:
                return False
            node = node[part]
            continue
        if (
            isinstance(node, Sequence)
            and not isinstance(node, (str, bytes))
            and part.isdigit()
        ):
            index = int(part)
            if index >= len(node):
                return False
            node = node[index]
            continue
        return False
    return True


def check_step_quality(
    step: TrajectoryStep, decision: AgentDecision, *, scenario: str, index: int
) -> tuple[str, ...]:
    """Grade the properties that are graded by rate, not per sample.

    Kept apart from ``check_step`` rather than flagged inside it, so that a
    reader of either function knows which gate it feeds. Mixing them would make
    ``pass^k`` quietly apply to a property that is not an invariant.
    """
    failures = []
    message = decision.message or ""
    if step.quality_report_unavailable and not _reports_unavailability(message):
        failures.append(
            f"{scenario}[{index}]: reply did not acknowledge an unavailable report"
        )
    for alternatives in step.quality_message_contains_any:
        if not any(fragment in message for fragment in alternatives):
            failures.append(
                f"{scenario}[{index}]: reply did not mention any of "
                f"{sorted(alternatives)!r}"
            )
    return tuple(failures)


_REPORT_NOUN = r"(?:报告|调研|记录|引用|资料)"
_UNAVAILABLE_READ = (
    r"(?:(?:未能|没能|无法|不能)"
    r"(?:在[^，,。；;！？!?\n]{0,30})?"
    r"(?:直接|可靠地?|成功|重新)?"
    r"(?:找到|检索到|查到|获取|取回|读取|定位|访问)"
    r"|(?:没有|没|未)(?:在[^，,。；;！？!?\n]{0,30})?"
    r"(?:找到|检索到|查到|获取|取回))"
)
_REPORT_UNAVAILABLE = re.compile(
    rf"{_UNAVAILABLE_READ}[^，,。；;！？!?\n]{{0,32}}{_REPORT_NOUN}"
    rf"|{_REPORT_NOUN}[^，,。；;！？!?\n]{{0,20}}"
    rf"(?:{_UNAVAILABLE_READ}|找不到|不可访问|不可用|不存在|缺失)"
    rf"|(?:没有|缺少|未提供)(?:对应的?|可用的?|可访问的?|匹配的?)?{_REPORT_NOUN}"
)


def _reports_unavailability(message: str) -> bool:
    """Conservative Chinese report-access rubric, independent of company names."""
    for sentence in re.split(r"[。；;！？!?\n]", message):
        if re.search(r"(?:不是|并非).{0,6}(?:没|未|无法|不能)", sentence):
            continue
        if _REPORT_UNAVAILABLE.search(sentence):
            return True
    return False


def check_step(step: TrajectoryStep, decision: AgentDecision, *, scenario: str, index: int) -> tuple[str, ...]:
    failures = []
    label = f"{scenario}[{index}]"
    called = decision.tool_call.name if decision.tool_call is not None else None
    if step.expect_tools and called not in step.expect_tools:
        failures.append(
            f"{label}: expected one of {sorted(step.expect_tools)!r}, got {called!r}"
        )
    if step.expect_action is not None and decision.action != step.expect_action:
        failures.append(
            f"{label}: expected action '{step.expect_action}', got "
            f"'{decision.action}'" + (f" calling '{called}'" if called else "")
        )
    if step.expect_tool is not None and called != step.expect_tool:
        failures.append(
            f"{label}: expected tool '{step.expect_tool}', got "
            f"'{called or decision.action}'"
        )
    if step.expect_message is not None and decision.message != step.expect_message:
        failures.append(
            f"{label}: expected message {step.expect_message!r}, got "
            f"{decision.message!r}"
        )
    for fragment in sorted(step.forbid_message_contains):
        if fragment in (decision.message or ""):
            failures.append(
                f"{label}: reply restated runtime-owned delivery {fragment!r}"
            )
    if decision.action == "final":
        for fragment in sorted(step.forbid_final_message_contains):
            if fragment in (decision.message or ""):
                failures.append(
                    f"{label}: final reply reused unreviewed content "
                    f"{fragment!r}"
                )
    if called in step.forbid_tools:
        failures.append(f"{label}: called forbidden tool '{called}'")
    arguments = (
        decision.tool_call.arguments if decision.tool_call is not None else {}
    )
    for name, expected in sorted(step.expect_arguments.items()):
        if arguments.get(name) != expected:
            failures.append(
                f"{label}: expected argument {name}={expected!r}, got "
                f"{arguments.get(name)!r}"
            )
    for name, expected in sorted(step.expect_argument_contains.items()):
        actual = arguments.get(name)
        if not isinstance(actual, str) or expected not in actual:
            failures.append(
                f"{label}: expected argument {name} to contain {expected!r}, "
                f"got {actual!r}"
            )
    for name in sorted(step.expect_nonempty_string_arguments):
        actual = arguments.get(name)
        if not isinstance(actual, str) or not actual.strip():
            failures.append(
                f"{label}: expected argument {name} to be a nonempty string, "
                f"got {actual!r}"
            )
    for name in sorted(step.forbid_non_null_arguments):
        if arguments.get(name) is not None:
            failures.append(
                f"{label}: passed {name}={arguments[name]!r}, which the "
                "projection never offered"
            )
    return tuple(failures)


def trajectory_prompt_fingerprint(
    scenario: TrajectoryScenario,
    tool_specs: tuple[dict[str, Any], ...],
) -> str:
    """Hash the active profile and stable prompt/schema prefix at each step."""
    step_fingerprints = []
    context = scenario.context
    for index, step in enumerate(scenario.steps):
        context = _advance(context, step)
        profile = context.task.tool_profile
        step_fingerprints.append(
            {
                "step": index,
                "profile": profile,
                "fingerprint": prompt_fingerprint(profile_schemas(profile, tool_specs)),
            }
        )
    encoded = json.dumps(
        step_fingerprints,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def replay(
    scenario: TrajectoryScenario,
    *,
    tool_specs: tuple[dict[str, Any], ...],
    responses: Sequence[Mapping[str, Any]],
    config: OpenAICompatibleAgentConfig | None = None,
) -> tuple[str, ...]:
    """Run a scenario against its recording through the real decision maker.

    Each step receives its profile's schemas. Runtime tests independently cover
    argument projection, authorization and side effects.
    """
    client = ReplayClient(responses)
    maker = OpenAICompatibleMainAgentDecisionMaker(
        config
        or OpenAICompatibleAgentConfig(
            endpoint="https://replay.invalid/v1/chat/completions",
            api_key="replay",
            model="replay",
        ),
        client=client,
    )
    failures: list[str] = []
    context = scenario.context
    schemas_by_profile = {
        profile: profile_schemas(profile, tool_specs) for profile in TOOL_PROFILE_NAMES
    }
    for index, step in enumerate(scenario.steps):
        context = _advance(context, step)
        schemas = schemas_by_profile[context.task.tool_profile]
        offered_names = {spec["function"]["name"] for spec in schemas}
        decision = maker.decide(context, schemas)
        called = (
            decision.tool_call.name if decision.tool_call is not None else None
        )
        if called is not None and called not in offered_names:
            failures.append(
                f"{scenario.name}[{index}]: cassette returned unavailable "
                f"tool '{called}'"
            )
        failures.extend(check_step(step, decision, scenario=scenario.name, index=index))
    return tuple(failures)


def replay_cassette(
    scenario: TrajectoryScenario,
    *,
    tool_specs: tuple[dict[str, Any], ...],
    cassette: TrajectoryCassette,
) -> tuple[tuple[str, ...], ...]:
    """Replay every independent sample without hiding intermittent failures."""

    return tuple(
        replay(scenario, tool_specs=tool_specs, responses=responses)
        for responses in cassette.recordings
    )


def replay_quality(
    scenario: TrajectoryScenario,
    *,
    tool_specs: tuple[dict[str, Any], ...],
    cassette: TrajectoryCassette,
) -> tuple[tuple[str, ...], ...]:
    """Grade the rate-gated properties, one entry per sample.

    Separate pass over the same recordings rather than a second return value
    from ``replay_cassette``: the two tiers are read by different gates, and a
    caller that only cares about invariants should not have to know this exists.
    """
    if not scenario.has_quality_assertions:
        return ()
    schemas_by_profile = {
        profile: profile_schemas(profile, tool_specs) for profile in TOOL_PROFILE_NAMES
    }
    graded: list[tuple[str, ...]] = []
    for responses in cassette.recordings:
        client = ReplayClient(responses)
        maker = OpenAICompatibleMainAgentDecisionMaker(
            OpenAICompatibleAgentConfig(
                endpoint="https://replay.invalid/v1/chat/completions",
                api_key="replay",
                model="replay",
            ),
            client=client,
        )
        failures: list[str] = []
        context = scenario.context
        for index, step in enumerate(scenario.steps):
            context = _advance(context, step)
            decision = maker.decide(context, schemas_by_profile[context.task.tool_profile])
            failures.extend(
                check_step_quality(
                    step, decision, scenario=scenario.name, index=index
                )
            )
        graded.append(tuple(failures))
    return tuple(graded)


def replay_budget_cassette_pair(
    *,
    baseline_scenario: TrajectoryScenario,
    baseline_cassette: TrajectoryCassette,
    candidate_scenario: TrajectoryScenario,
    candidate_cassette: TrajectoryCassette,
    tool_specs: tuple[dict[str, Any], ...],
) -> PairedBudgetCassetteReplay:
    """Validate a budget change from matched cassette samples, never telemetry.

    The scenarios must differ only in ``career_profile_budgets``. Each sample
    index is one pair, so a candidate regression remains visible instead of
    being averaged into unrelated production traffic.
    """

    if baseline_scenario.known_gap or candidate_scenario.known_gap:
        raise ValueError("budget pairs cannot use known-gap scenarios")
    if (
        baseline_scenario.name != candidate_scenario.name
        or baseline_scenario.policy != candidate_scenario.policy
        or baseline_scenario.steps != candidate_scenario.steps
    ):
        raise ValueError("budget pair scenarios must share policy and steps")
    baseline_budgets = baseline_scenario.context.career_profile_budgets
    candidate_budgets = candidate_scenario.context.career_profile_budgets
    if baseline_budgets == candidate_budgets:
        raise ValueError("budget pair must compare two different configurations")
    normalized_baseline = baseline_scenario.context.model_copy(
        update={"career_profile_budgets": candidate_budgets}
    )
    if normalized_baseline != candidate_scenario.context:
        raise ValueError("budget pair scenarios may differ only in budgets")
    for label, scenario, cassette in (
        ("baseline", baseline_scenario, baseline_cassette),
        ("candidate", candidate_scenario, candidate_cassette),
    ):
        stale = cassette_staleness(
            cassette,
            scenario=scenario,
            tool_specs=tool_specs,
        )
        if stale is not None:
            raise ValueError(f"{label} budget cassette is stale: {stale}")
    if baseline_cassette.sample_count != candidate_cassette.sample_count:
        raise ValueError("budget cassette arms must have the same sample count")

    def passes(
        scenario: TrajectoryScenario,
        cassette: TrajectoryCassette,
    ) -> tuple[bool, ...]:
        invariant_failures = replay_cassette(
            scenario,
            tool_specs=tool_specs,
            cassette=cassette,
        )
        quality_failures = replay_quality(
            scenario,
            tool_specs=tool_specs,
            cassette=cassette,
        )
        if not quality_failures:
            quality_failures = tuple(() for _ in invariant_failures)
        return tuple(
            not invariant and not quality
            for invariant, quality in zip(
                invariant_failures,
                quality_failures,
                strict=True,
            )
        )

    baseline_passes = passes(baseline_scenario, baseline_cassette)
    candidate_passes = passes(candidate_scenario, candidate_cassette)
    return PairedBudgetCassetteReplay(
        baseline_budgets=(
            baseline_budgets.records_input_units,
            baseline_budgets.current_targets_input_units,
            baseline_budgets.hard_constraints_input_units,
        ),
        candidate_budgets=(
            candidate_budgets.records_input_units,
            candidate_budgets.current_targets_input_units,
            candidate_budgets.hard_constraints_input_units,
        ),
        pair_count=len(baseline_passes),
        baseline_pass_count=sum(baseline_passes),
        candidate_pass_count=sum(candidate_passes),
        regressed_pair_count=sum(
            baseline and not candidate
            for baseline, candidate in zip(
                baseline_passes,
                candidate_passes,
                strict=True,
            )
        ),
        improved_pair_count=sum(
            candidate and not baseline
            for baseline, candidate in zip(
                baseline_passes,
                candidate_passes,
                strict=True,
            )
        ),
    )


def quality_shortfall(
    scenario: TrajectoryScenario,
    graded: Sequence[Sequence[str]],
) -> str | None:
    """Whether the recorded rate has fallen below the floor the scenario keeps.

    Reports the rate either way when it returns a message, because the number is
    the finding: "3 of 5" is the status quo this project measured, and the point
    of the gate is to notice the day it becomes 1 of 5.
    """
    if scenario.quality_min_pass_rate is None:
        return None
    if len(graded) != scenario.recording_samples:
        return (
            f"{scenario.name}: quality grading requires exactly "
            f"{scenario.recording_samples} samples, got {len(graded)}"
        )
    passing = sum(1 for failures in graded if not failures)
    rate = passing / len(graded)
    if rate >= scenario.quality_min_pass_rate:
        return None
    return (
        f"{scenario.name}: quality properties held in {passing} of "
        f"{len(graded)} samples ({rate:.1%}), below the declared floor of "
        f"{scenario.quality_min_pass_rate:.1%}"
    )


def minimum_detectable_regression(
    sample_count: int,
    min_pass_rate: float,
) -> float | None:
    """Largest drop from a perfect observed rate that still meets the floor.

    The gate is a raw rate, not an interval. With n=3 and floor=0.6, 2/3 still
    passes, so a one-sample (1/3) regression is invisible. Report that blind
    spot instead of a Wilson interval that spans half of [0, 1] at this n.
    """
    if sample_count < 1 or not 0 < min_pass_rate <= 1:
        return None
    min_passes = math.ceil(min_pass_rate * sample_count - 1e-12)
    if min_passes > sample_count:
        min_passes = sample_count
    return 1.0 - (min_passes / sample_count)


def known_gap_reproduction(
    sample_failures: Sequence[Sequence[str]],
) -> str:
    """Classify whether a known defect still reproduces across its samples."""

    failed = tuple(bool(failures) for failures in sample_failures)
    if not failed or not any(failed):
        return "resolved"
    if all(failed):
        return "stable"
    return "intermittent"


def _advance(context: MainAgentContext, step: TrajectoryStep) -> MainAgentContext:
    """Apply what the runtime would have carried into this step."""
    update: dict[str, Any] = {}
    if step.user_message is not None:
        update["user_message"] = step.user_message
    if step.observation is not None:
        update["tool_observations"] = append_decision_observation(
            context.tool_observations,
            step.observation,
        )
    if step.task_update:
        update["task"] = context.task.model_copy(update=dict(step.task_update))
    return context.model_copy(update=update) if update else context


def _decision_maker(
    config: OpenAICompatibleAgentConfig,
) -> OpenAICompatibleMainAgentDecisionMaker:
    maker = getattr(_RECORD_MAKERS, "maker", None)
    cached_config = getattr(_RECORD_MAKERS, "config", None)
    if (
        maker is None
        or cached_config is not config
        or type(maker) is not OpenAICompatibleMainAgentDecisionMaker
    ):
        maker = OpenAICompatibleMainAgentDecisionMaker(config)
        _RECORD_MAKERS.maker = maker
        _RECORD_MAKERS.config = config
    return maker


def _recording_can_retry(error: AgentWorkerError) -> bool:
    """Whether one more attempt at the same sample is worth making.

    Transport errors carry their own ``retryable`` flag. An unparseable,
    malformed or empty decision is not retryable in production, where the turn
    has to fail closed, but for a recording it is one bad draw from a
    nondeterministic model: on 2026-09-11 a single such draw failed a whole
    scenario 35 cassettes into a batch. A rejected request or a configuration
    error stays fatal; retrying them changes nothing.
    """

    if error.retryable:
        return True
    return error.code in _RETRIED_MODEL_OUTPUT_CODES and not isinstance(
        error, AgentConfigurationError
    )


_RETRIED_MODEL_OUTPUT_CODES = frozenset(
    {
        "MAIN_AGENT_INVALID_RESPONSE",
        "MAIN_AGENT_INVALID_TOOL_ARGUMENTS",
        "MAIN_AGENT_EMPTY_RESPONSE",
    }
)


def _retry_wait(delay: float, attempt: int, *, jitter: bool) -> float:
    wait = delay * (2 ** (attempt - 1))
    if jitter:
        wait *= 0.5 + random.random()
    return wait


def _requested_sample_count(
    scenario: TrajectoryScenario, sample_count: int | None
) -> int:
    requested_samples = (
        scenario.recording_samples if sample_count is None else sample_count
    )
    if not 1 <= requested_samples <= 5:
        raise ValueError("trajectory sample count must be between 1 and 5")
    if scenario.has_quality_assertions and (
        requested_samples != scenario.recording_samples
    ):
        raise ValueError(
            "quality scenarios must be recorded with exactly their declared "
            "sample count"
        )
    return requested_samples


def _record_one_sample(
    scenario: TrajectoryScenario,
    *,
    tool_specs: tuple[dict[str, Any], ...],
    config: OpenAICompatibleAgentConfig,
    max_attempts: int,
    retry_delay_seconds: float,
    sleeper: Callable[[float], None],
    jitter: bool,
) -> dict[str, Any]:
    maker = _decision_maker(config)
    steps = []
    context = scenario.context
    schemas_by_profile = {
        profile: profile_schemas(profile, tool_specs) for profile in TOOL_PROFILE_NAMES
    }
    for step in scenario.steps:
        context = _advance(context, step)
        decision = None
        for attempt in range(1, max_attempts + 1):
            try:
                decision = maker.decide(context, schemas_by_profile[context.task.tool_profile])
                break
            except AgentWorkerError as error:
                if not _recording_can_retry(error) or attempt == max_attempts:
                    raise
                sleeper(_retry_wait(retry_delay_seconds, attempt, jitter=jitter))
        if decision is None:
            raise RuntimeError("trajectory record produced no decision")
        steps.append(
            {
                "tool_call": {
                    "name": decision.tool_call.name,
                    "arguments": decision.tool_call.arguments,
                }
            }
            if decision.tool_call is not None
            else {"content": decision.model_dump_json(exclude_none=True)}
        )
    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "steps": steps,
    }


def _write_cassette(
    scenario: TrajectoryScenario,
    samples: Sequence[Mapping[str, Any]],
    *,
    tool_specs: tuple[dict[str, Any], ...],
    config: OpenAICompatibleAgentConfig,
    root: Path | None,
) -> Path:
    path = cassette_path(scenario.name, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "scenario": scenario.name,
                "policy": scenario.policy,
                "model": config.model,
                "prompt_fingerprint": trajectory_prompt_fingerprint(
                    scenario, tool_specs
                ),
                "context_shape_fingerprint": context_shape_fingerprint(scenario),
                "recorded_at": samples[-1]["recorded_at"],
                # Keep the first sample under the legacy key so external readers
                # do not break while the evaluator consumes every sample below.
                "steps": samples[0]["steps"],
                "sample_count": len(samples),
                "samples": list(samples),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    return path


def scenarios_to_record(
    scenarios: Sequence[TrajectoryScenario],
    *,
    tool_specs: tuple[dict[str, Any], ...],
    root: Path | None = None,
    force: bool = False,
) -> tuple[TrajectoryScenario, ...]:
    """Scenarios whose cassettes are missing, stale, or forced to recut."""

    selected = []
    for scenario in scenarios:
        if check_contract(scenario, tool_specs=tool_specs):
            continue
        cassette = load_cassette(scenario.name, root=root)
        if force or cassette is None:
            selected.append(scenario)
            continue
        if cassette_staleness(
            cassette, scenario=scenario, tool_specs=tool_specs
        ) is not None:
            selected.append(scenario)
    return tuple(selected)


def record(
    scenario: TrajectoryScenario,
    *,
    tool_specs: tuple[dict[str, Any], ...],
    config: OpenAICompatibleAgentConfig,
    root: Path | None = None,
    sample_count: int | None = None,
    max_attempts: int = 3,
    retry_delay_seconds: float = 1.0,
    sleeper: Callable[[float], None] = time.sleep,
    max_workers: int = 1,
    jitter: bool | None = None,
) -> Path:
    """Ask the live model and write the answers down.

    Recording is the only thing that evaluates the model. Everything replay does
    afterwards evaluates this project against a decision the model already made,
    which is why a stale cassette is worth re-cutting whenever the prompt, the
    projection, or the model changes.

    Independent samples may run concurrently. Steps inside one sample stay
    sequential, because later hops depend on earlier observations.
    """
    requested_samples = _requested_sample_count(scenario, sample_count)
    if not 1 <= max_attempts <= 5:
        raise ValueError("trajectory record attempts must be between 1 and 5")
    if retry_delay_seconds < 0:
        raise ValueError("trajectory retry delay cannot be negative")
    if not 1 <= max_workers <= _MAX_RECORD_JOBS:
        raise ValueError(
            f"trajectory record jobs must be between 1 and {_MAX_RECORD_JOBS}"
        )
    use_jitter = max_workers > 1 if jitter is None else jitter
    workers = min(max_workers, requested_samples)

    def capture_sample() -> dict[str, Any]:
        return _record_one_sample(
            scenario,
            tool_specs=tool_specs,
            config=config,
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
            sleeper=sleeper,
            jitter=use_jitter,
        )

    if workers == 1:
        samples = [capture_sample() for _ in range(requested_samples)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(capture_sample) for _ in range(requested_samples)
            ]
            samples = [future.result() for future in futures]
    return _write_cassette(
        scenario,
        samples,
        tool_specs=tool_specs,
        config=config,
        root=root,
    )


def record_catalogue(
    scenarios: Sequence[TrajectoryScenario],
    *,
    tool_specs: tuple[dict[str, Any], ...],
    config: OpenAICompatibleAgentConfig,
    root: Path | None = None,
    sample_count: int | None = None,
    max_attempts: int = 3,
    retry_delay_seconds: float = 1.0,
    sleeper: Callable[[float], None] = time.sleep,
    jobs: int = 8,
    force: bool = False,
) -> tuple[Path, ...]:
    """Recut every selected cassette that is missing or stale.

    Work is one sample, not one scenario: a three-sample case can share the
    pool with other scenarios. A scenario is written only after every sample
    succeeds, so a mid-cut failure cannot leave a half cassette. Completed
    neighbours are still written, so a long recut is not all-or-nothing.
    """
    if not 1 <= jobs <= _MAX_RECORD_JOBS:
        raise ValueError(
            f"trajectory record jobs must be between 1 and {_MAX_RECORD_JOBS}"
        )
    if not 1 <= max_attempts <= 5:
        raise ValueError("trajectory record attempts must be between 1 and 5")
    if retry_delay_seconds < 0:
        raise ValueError("trajectory retry delay cannot be negative")

    to_record = scenarios_to_record(
        scenarios, tool_specs=tool_specs, root=root, force=force
    )
    if not to_record:
        return ()

    counts = {
        scenario.name: _requested_sample_count(scenario, sample_count)
        for scenario in to_record
    }
    by_name = {scenario.name: scenario for scenario in to_record}
    work = [
        (scenario, index)
        for scenario in to_record
        for index in range(counts[scenario.name])
    ]
    collected: dict[str, dict[int, dict[str, Any]]] = {
        scenario.name: {} for scenario in to_record
    }
    failed: set[str] = set()
    first_error: Exception | None = None
    written: dict[str, Path] = {}
    jitter = jobs > 1
    workers = min(jobs, len(work))

    def capture(scenario: TrajectoryScenario) -> dict[str, Any]:
        return _record_one_sample(
            scenario,
            tool_specs=tool_specs,
            config=config,
            max_attempts=max_attempts,
            retry_delay_seconds=retry_delay_seconds,
            sleeper=sleeper,
            jitter=jitter,
        )

    def accept(scenario: TrajectoryScenario, index: int, sample: dict[str, Any]) -> None:
        if scenario.name in failed:
            return
        collected[scenario.name][index] = sample
        if len(collected[scenario.name]) != counts[scenario.name]:
            return
        samples = [
            collected[scenario.name][sample_index]
            for sample_index in range(counts[scenario.name])
        ]
        written[scenario.name] = _write_cassette(
            by_name[scenario.name],
            samples,
            tool_specs=tool_specs,
            config=config,
            root=root,
        )

    if workers == 1:
        for scenario, index in work:
            if scenario.name in failed:
                continue
            try:
                sample = capture(scenario)
            except Exception as error:
                failed.add(scenario.name)
                if first_error is None:
                    first_error = error
                continue
            accept(scenario, index, sample)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(capture, scenario): (scenario, index)
                for scenario, index in work
            }
            for future in as_completed(futures):
                scenario, index = futures[future]
                try:
                    sample = future.result()
                except Exception as error:
                    failed.add(scenario.name)
                    if first_error is None:
                        first_error = error
                    continue
                accept(scenario, index, sample)

    if first_error is not None:
        raise first_error
    return tuple(written[scenario.name] for scenario in to_record if scenario.name in written)
