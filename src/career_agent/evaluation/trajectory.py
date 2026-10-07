"""Scenarios that pin what the Main Agent decides, not what it does after.

The evaluation runs at two levels, because only one of them can run without a
model:

**Contract level** — always runs, needs no API key. Builds the exact context a
scenario would send and checks that the decision is even *decidable* from it:
the facts the policy turns on are present in the projection, and every expected
tool is registered. Runtime argument projection,
not prompt-shape mutation, owns task-state preconditions.

**Replay level** — runs for scenarios that have a recorded model response.
Checks the decision itself against the scenario's expectations.

The split matters and should not be blurred: passing at contract level says the
question was asked properly, not that the model answered it well. Only a fresh
recording says that. The search trajectory recorder refreshes live samples.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel

from career_agent.agent.runtime.decision_messages import project_decision_messages
from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.contracts.task_state import ConversationTaskState
from career_agent.agent.contracts.decisions import AgentDecision
from career_agent.agent.contracts.observations import (
    DecisionObservation,
    append_decision_observation,
)
from career_agent.agent.providers.openai_client import (
    AgentConfigurationError,
    AgentWorkerError,
)
from career_agent.agent.providers.main_agent import (
    OpenAICompatibleMainAgentDecisionMaker,
)
from career_agent.agent.providers.interaction_output import interaction_schemas

CASSETTE_ROOT = Path(__file__).resolve().parents[3] / "evals" / "main_agent_search"


@dataclass(frozen=True)
class TrajectoryStep:
    """One decision the model is asked to make inside a scenario.

    Each step is a declared decision snapshot. ``observation`` and ``task_update``
    supply the result and reducer state for that snapshot, independently of the
    previous model response. Tools are not executed; these probes do not measure
    end-to-end runtime success.
    """

    expect_action: str | None = None
    forbid_actions: frozenset[str] = frozenset()
    """Actions this decision must not take, when several others are all fine.

    For a step whose right move is open (look something up first, or ask)
    but one move is the violation: answering outright with ``final``.
    """
    expect_user_input: bool = False
    """Accept either bound way to ask the user: ask_user or questionnaire."""
    expect_question_count: int | None = None
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
    if call is None:
        decision = AgentDecision.model_validate_json(recorded["content"])
        arguments = decision.model_dump(mode="json", exclude_none=True)
        action = arguments.pop("action")
        if action != "questionnaire":
            arguments.pop("questions", None)
        call = {"name": "final_response" if action == "final" else action,
                "arguments": arguments}
    function = type("Function", (), {
        "name": call["name"],
        "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False),
    })()
    tool_call = type("ToolCall", (), {"function": function})()
    message = type("Message", (), {"content": None, "tool_calls": [tool_call]})()
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
        """Every independent recording for this scenario."""

        return self.samples or (self.steps,)

    @property
    def sample_count(self) -> int:
        return len(self.recordings)


def summarize_decision_retries(cassettes: Sequence[TrajectoryCassette]) -> dict[str, Any]:
    """Summarize measured completed decisions and retry reasons."""
    steps = [step for cassette in cassettes for sample in cassette.recordings for step in sample]
    covered = [step for step in steps if step.get("decision_retry_telemetry_version") == 1]
    reason_counts: dict[str, int] = {}
    terminal_counts: dict[str, int] = {}
    for step in covered:
        for event in step.get("decision_retry_events", ()):
            counts = reason_counts if event["retried"] else terminal_counts
            reason = event["reason"]
            counts[reason] = counts.get(reason, 0) + 1

    def is_interaction(step: dict[str, Any]) -> bool:
        if "content" not in step:
            return False
        try:
            return json.loads(step["content"]).get("action") in {"ask_user", "questionnaire", "final"}
        except (ValueError, AttributeError):
            return False

    interactions = [step for step in covered if is_interaction(step)]
    def with_retries(items: Sequence[dict[str, Any]]) -> int:
        return sum(
            any(event["retried"] for event in step.get("decision_retry_events", ()))
            for step in items
        )
    return {
        "completed_decisions": len(steps),
        "covered_decisions": len(covered),
        "unknown_decisions": len(steps) - len(covered),
        "decisions_with_retries": with_retries(covered),
        "interaction_decisions": sum(is_interaction(step) for step in steps),
        "covered_interaction_decisions": len(interactions),
        "interaction_decisions_with_retries": with_retries(interactions),
        "interaction_retry_rate": with_retries(interactions) / len(interactions) if interactions else None,
        "retry_reason_counts": reason_counts,
        "terminal_rejection_reason_counts": terminal_counts,
        "note": "Counts cover completed decisions in fresh cassettes only; terminal failed recordings are reported separately.",
    }


def cassette_path(name: str, *, root: Path | None = None) -> Path:
    return (root or CASSETTE_ROOT) / f"{name}.json"


def prompt_fingerprint(tool_specs: tuple[dict[str, Any], ...]) -> str:
    """Hash the exact system prompt and schemas sent for one decision."""
    from career_agent.agent.capabilities.selection_strategy import SearchStrategy
    prompt = OpenAICompatibleMainAgentDecisionMaker._system_prompt(
        SearchStrategy().tool_policy()
    )
    encoded = json.dumps(
        {
            "system_prompt": prompt,
            "tool_specs": tool_specs + interaction_schemas(continuation=True),
            "tool_selection_mode": "search",
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# The live clock changes on every call; hashing it would make every cassette
# stale the moment it is written.
_FINGERPRINT_CLOCK = {"now": "2000-01-01T00:00:00+08:00", "timezone": "Asia/Shanghai"}


def context_shape_fingerprint(scenario: TrajectoryScenario) -> str:
    """Hash fixture values, authority boundaries and native-message layout."""
    context = scenario.context
    step_shapes = []
    for index, step in enumerate(scenario.steps):
        context = advance_trajectory_context(context, step)
        projection = project_decision_messages(context, clock=_FINGERPRINT_CLOCK)
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
    payload = json.loads(path.read_text(encoding="utf-8"))
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




def check_contract(
    scenario: TrajectoryScenario, *, tool_specs: tuple[dict[str, Any], ...]
) -> tuple[str, ...]:
    """Whether the scenario asks a question the model could answer.

    Runs without a model and catches missing decision facts or unknown tools.
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
        if step.expect_user_input and step.expect_action is not None:
            failures.append(
                f"{scenario.name}[{index}]: expect_user_input and expect_action "
                "cannot both be set"
            )
        context = advance_trajectory_context(context, step)
        registered = {spec["function"]["name"] for spec in tool_specs}
        expected = step.expect_tools | (
            {step.expect_tool} if step.expect_tool is not None else set()
        )
        for name in sorted(expected - registered):
            failures.append(
                f"{scenario.name}[{index}]: expected tool "
                f"'{name}' is not registered"
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
    if decision.action in step.forbid_actions:
        failures.append(
            f"{label}: took forbidden action '{decision.action}'"
            + (f" calling '{called}'" if called else "")
        )
    if step.expect_user_input and decision.action not in {"ask_user", "questionnaire"}:
        failures.append(
            f"{label}: expected user input action, got '{decision.action}'"
            + (f" calling '{called}'" if called else "")
        )
    if step.expect_question_count is not None and len(decision.questions) != step.expect_question_count:
        failures.append(
            f"{label}: expected {step.expect_question_count} questions, got {len(decision.questions)}"
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
    """Hash the selected schemas and stable prompt at each step."""
    step_fingerprints = []
    context = scenario.context
    from career_agent.agent.capabilities.selection_strategy import SearchStrategy
    strategy = SearchStrategy()
    for index, step in enumerate(scenario.steps):
        context = advance_trajectory_context(context, step)
        selection = strategy.select(context, tool_specs)
        step_fingerprints.append({
            "step": index,
            "fingerprint": prompt_fingerprint(selection.schemas),
        })
    encoded = json.dumps(
        step_fingerprints,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()










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


def advance_trajectory_context(context: MainAgentContext, step: TrajectoryStep) -> MainAgentContext:
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
        patch: dict[str, Any] = {}
        for name, value in step.task_update.items():
            if name == "domain_context" and isinstance(value, Mapping):
                _reject_unknown_nested_fields(
                    value, type(context.task.domain_context), name,
                )
            part = ConversationTaskState.model_validate({name: value})
            if not part.model_fields_set:
                raise ValueError(f"unknown trajectory task_update field: {name}")
            normalized = part.model_dump(mode="python", exclude_unset=True)
            for tagged_field in ("pending_interaction", "workflow"):
                if tagged_field in normalized:
                    # exclude_unset omits a discriminator when the model used
                    # its default kind; a tagged union needs the complete value.
                    normalized[tagged_field] = part.model_dump(mode="python")[
                        tagged_field
                    ]
            _merge_trajectory_patch(
                patch, normalized,
            )
        merged = context.task.model_dump(mode="python")
        _merge_trajectory_patch(merged, patch)
        for tagged_field in ("pending_interaction", "workflow"):
            if tagged_field in patch:
                merged[tagged_field] = patch[tagged_field]
        update["task"] = ConversationTaskState.model_validate(merged)
    return context.model_copy(update=update) if update else context


def _merge_trajectory_patch(target: dict[str, Any], patch: Mapping[str, Any]) -> None:
    for name, value in patch.items():
        if isinstance(value, Mapping) and isinstance(target.get(name), dict):
            _merge_trajectory_patch(target[name], value)
        else:
            target[name] = value


def _reject_unknown_nested_fields(
    raw: Mapping[str, Any], model: type[BaseModel], path: str,
) -> None:
    for name, value in raw.items():
        field = model.model_fields.get(name)
        if field is None:
            raise ValueError(f"unknown trajectory task_update field: {path}.{name}")
        nested_model = field.annotation
        if (
            isinstance(value, Mapping)
            and isinstance(nested_model, type)
            and issubclass(nested_model, BaseModel)
        ):
            _reject_unknown_nested_fields(value, nested_model, f"{path}.{name}")


def trajectory_tool_specs() -> tuple[dict[str, Any], ...]:
    """Registered model tool schemas for offline trajectory evaluation."""
    import inspect

    from career_agent.agent.capabilities.registry import MainAgentToolRegistry

    parameters = tuple(
        name
        for name in inspect.signature(MainAgentToolRegistry.__init__).parameters
        if name != "self"
    )
    return tuple(MainAgentToolRegistry(**{name: object() for name in parameters}).schemas())




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
