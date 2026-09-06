"""Scenarios that pin what the Main Agent decides, not what it does after.

The evaluation runs at two levels, because only one of them can run without a
model:

**Contract level** — always runs, needs no API key. Builds the exact context a
scenario would send and checks that the decision is even *decidable* from it:
the facts the policy turns on are present in the projection, and every expected
tool is offered by the production reachability menu at that exact step. A
forbidden tool hidden by that menu is a structural guard, not model evidence;
the reachability suite owns that guarantee.

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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Callable, Mapping, Sequence

from career_agent.agent.decision_messages import project_decision_messages
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    DecisionObservation,
    MainAgentContext,
    append_decision_observation,
)
from career_agent.agent.openai_compatible_client import (
    AgentWorkerError,
    OpenAICompatibleAgentConfig,
)
from career_agent.agent.openai_compatible_main_agent import (
    OpenAICompatibleMainAgentDecisionMaker,
)

CASSETTE_ROOT = Path(__file__).resolve().parents[3] / "evals" / "main_agent"
_MAX_RECORD_JOBS = 32
_RECORD_MAKERS = threading.local()


@dataclass(frozen=True)
class TrajectoryStep:
    """One decision the model is asked to make inside a scenario.

    A step after the first continues the same conversation with whatever the
    previous call produced. ``observation`` is what the runtime would have fed
    back, and ``task_update`` is what the reducer layer would have written into
    task state — usually the candidate list a ``list_*`` call produces. Both are
    declared rather than executed: the point is to exercise the decision loop,
    not the service stack behind it.
    """

    expect_action: str | None = None
    expect_tool: str | None = None
    expect_message: str | None = None
    forbid_message_contains: frozenset[str] = frozenset()
    """Substrings the reply must not carry, for delivery the runtime owns.

    A card-backed report reaches the reader through its entity. Restating the
    body in the reply duplicates it into a window that will keep the row for
    every later turn, and puts a second, drifting copy next to the one the
    reader can reopen. Exact-matching prose would be brittle, so the assertion
    names only what must be absent.
    """

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

    forbid_tools: frozenset[str] = frozenset()
    expect_arguments: Mapping[str, Any] = field(default_factory=dict)
    """Argument values the call must carry, checked as a subset.

    ``expect_tool`` says which capability; this says the model selected the
    right thing with it. A subset rather than equality: the assertion is about
    the selector the policy turns on, and pinning every other argument would
    make an unrelated schema change read as a policy failure.
    """

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
    The evaluator also reports a Wilson interval: five samples are useful as a
    cheap regression sentinel, not enough to claim a precise population rate.

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
        forbidden = set().union(*(step.forbid_tools for step in self.steps))
        return frozenset(expected | forbidden)

    def __post_init__(self) -> None:
        if not 1 <= self.recording_samples <= 5:
            raise ValueError(
                "trajectory recording_samples must be between 1 and 5"
            )
        declares_quality = any(
            step.quality_message_contains_any for step in self.steps
        )
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
        return any(step.quality_message_contains_any for step in self.steps)


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


def cassette_path(name: str, *, root: Path | None = None) -> Path:
    return (root or CASSETTE_ROOT) / f"{name}.json"


def prompt_fingerprint(tool_specs: tuple[dict[str, Any], ...]) -> str:
    """Hash the exact system prompt and schemas sent for one decision."""
    tool_names = tuple(spec["function"]["name"] for spec in tool_specs)
    prompt = OpenAICompatibleMainAgentDecisionMaker._system_prompt(tool_names)
    encoded = json.dumps(
        {"system_prompt": prompt, "tool_specs": tool_specs},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def context_shape_fingerprint(scenario: TrajectoryScenario) -> str:
    """Hash the authority split and native-message layout sent to the model.

    Values are deliberately excluded, but roles are not: moving prior dialogue
    back into a JSON field must stale a cassette even if a merged contract view
    still exposes the same facts. JSON sequence elements share a ``[]`` path,
    so candidate count does not make an otherwise identical shape stale.
    """
    context = scenario.context
    step_shapes = []
    for index, step in enumerate(scenario.steps):
        context = _advance(context, step)
        projection = project_decision_messages(context)
        messages = projection.messages(system_prompt="[policy]")
        step_shapes.append(
            {
                "step": index,
                "control_paths": sorted(_key_paths(projection.control)),
                "data_paths": sorted(_key_paths(projection.data)),
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
            "cassette prompt_fingerprint does not match the current dynamic "
            "prompt/tool menu; re-record it"
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
    production would hide at that exact state. A forbidden tool that is hidden
    is a structural reachability guarantee rather than a prompt-policy test; it
    is therefore not treated as vacuous model evidence here.
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
            for spec in _dynamic_schemas(tool_specs, context)
        }
        if step.expect_tool is not None and step.expect_tool not in offered:
            failures.append(
                f"{scenario.name}[{index}]: expected tool "
                f"'{step.expect_tool}' is hidden by the production menu"
            )
    return tuple(failures)


def _has_path(payload: Any, path: str) -> bool:
    """Whether a dotted path resolves, treating an absent key as the failure.

    A present key holding ``None`` counts: "the city is not known" is itself the
    fact several policies turn on, so requiring a truthy value would reject
    exactly the contexts worth testing.
    """
    node = payload
    for part in path.split("."):
        if isinstance(node, Mapping):
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
    for alternatives in step.quality_message_contains_any:
        if not any(fragment in message for fragment in alternatives):
            failures.append(
                f"{scenario}[{index}]: reply did not mention any of "
                f"{sorted(alternatives)!r}"
            )
    return tuple(failures)


def check_step(step: TrajectoryStep, decision: AgentDecision, *, scenario: str, index: int) -> tuple[str, ...]:
    failures = []
    label = f"{scenario}[{index}]"
    called = decision.tool_call.name if decision.tool_call is not None else None
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
    for name in sorted(step.forbid_non_null_arguments):
        if arguments.get(name) is not None:
            failures.append(
                f"{label}: passed {name}={arguments[name]!r}, which the "
                "projection never offered"
            )
    return tuple(failures)


def _dynamic_schemas(
    static_schemas: tuple[dict[str, Any], ...], context: MainAgentContext
) -> tuple[dict[str, Any], ...]:
    """The per-step menu the production runtime would show for ``context``.

    Kept derived from the static universe so the full-name list stays in one
    place; only the set of offered tools shrinks per step.
    """
    from career_agent.agent.tool_reachability import reachable_in_context

    return tuple(
        schema
        for schema in static_schemas
        if reachable_in_context(schema["function"]["name"], context)
    )


def trajectory_prompt_fingerprint(
    scenario: TrajectoryScenario,
    tool_specs: tuple[dict[str, Any], ...],
) -> str:
    """Hash the sequence of real per-step prompts and tool schemas.

    One static fingerprint cannot represent a dynamic menu: a reachability
    change may alter only step two, while the full registry remains identical.
    Hashing each step's actual request makes that cassette stale instead of
    silently replaying a decision produced under a tool the model no longer
    sees.
    """
    context = scenario.context
    step_fingerprints = []
    for index, step in enumerate(scenario.steps):
        context = _advance(context, step)
        step_fingerprints.append(
            {
                "step": index,
                "fingerprint": prompt_fingerprint(
                    _dynamic_schemas(tool_specs, context)
                ),
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

    Each step is evaluated against the *dynamic* menu the production runtime
    would offer for that step's state — a scenario that forbids a tool the
    model was never shown would otherwise prove nothing.
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
    for index, step in enumerate(scenario.steps):
        context = _advance(context, step)
        offered_specs = _dynamic_schemas(tool_specs, context)
        decision = maker.decide(context, offered_specs)
        offered_names = {
            spec["function"]["name"] for spec in offered_specs
        }
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
    if not any(step.quality_message_contains_any for step in scenario.steps):
        return ()
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
            decision = maker.decide(context, _dynamic_schemas(tool_specs, context))
            failures.extend(
                check_step_quality(
                    step, decision, scenario=scenario.name, index=index
                )
            )
        graded.append(tuple(failures))
    return tuple(graded)


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


def wilson_score_interval(
    passing: int,
    total: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float] | None:
    """Wilson interval for an observed binomial pass rate.

    The interval characterises uncertainty; the small fixed-sample regression
    gate above deliberately remains a point-estimate floor. Returning ``None``
    for no observations prevents callers from publishing a fictitious 0/0
    quality rate.
    """
    if total == 0:
        return None
    if not 0 <= passing <= total:
        raise ValueError(
            "passing samples must be between zero and total samples"
        )
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    rate = passing / total
    denominator = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
        / denominator
    )
    return centre - margin, centre + margin


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
    for step in scenario.steps:
        context = _advance(context, step)
        schemas = _dynamic_schemas(tool_specs, context)
        decision = None
        for attempt in range(1, max_attempts + 1):
            try:
                decision = maker.decide(context, schemas)
                break
            except AgentWorkerError as error:
                if not error.retryable or attempt == max_attempts:
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
