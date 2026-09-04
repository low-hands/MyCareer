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

import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    DecisionObservation,
    MainAgentContext,
    append_decision_observation,
)
from career_agent.agent.openai_compatible_client import OpenAICompatibleAgentConfig
from career_agent.agent.openai_compatible_main_agent import (
    OpenAICompatibleMainAgentDecisionMaker,
)

CASSETTE_ROOT = Path(__file__).resolve().parents[3] / "evals" / "main_agent"


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

    forbid_tools: frozenset[str] = frozenset()
    expect_arguments: Mapping[str, Any] = field(default_factory=dict)
    """Argument values the call must carry, checked as a subset.

    ``expect_tool`` says which capability; this says the model selected the
    right thing with it. A subset rather than equality: the assertion is about
    the selector the policy turns on, and pinning every other argument would
    make an unrelated schema change read as a policy failure.
    """

    forbid_argument_keys: frozenset[str] = frozenset()
    """Argument names the call must not carry, whatever it calls.

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
    """Hash the key paths of every model context this scenario sends.

    Values are deliberately excluded: changing a count or user sentence is a
    new example, not a projection schema change. Sequence elements share a
    ``[]`` path, so candidate count also cannot make an otherwise identical
    shape stale.
    """
    context = scenario.context
    step_shapes = []
    for index, step in enumerate(scenario.steps):
        context = _advance(context, step)
        step_shapes.append(
            {
                "step": index,
                "paths": sorted(_key_paths(context.model_context())),
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
    return TrajectoryCassette(
        steps=tuple(payload["steps"]),
        prompt_fingerprint=payload.get("prompt_fingerprint"),
        context_shape_fingerprint=payload.get("context_shape_fingerprint"),
        model=payload.get("model"),
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
    for name in sorted(step.forbid_argument_keys):
        if name in arguments:
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
    from career_agent.agent.tool_reachability import reachable

    return tuple(
        schema
        for schema in static_schemas
        if reachable(schema["function"]["name"], context.task)
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


def record(
    scenario: TrajectoryScenario,
    *,
    tool_specs: tuple[dict[str, Any], ...],
    config: OpenAICompatibleAgentConfig,
    root: Path | None = None,
) -> Path:
    """Ask the live model and write the answers down.

    Recording is the only thing that evaluates the model. Everything replay does
    afterwards evaluates this project against a decision the model already made,
    which is why a stale cassette is worth re-cutting whenever the prompt, the
    projection, or the model changes.
    """
    maker = OpenAICompatibleMainAgentDecisionMaker(config)
    steps = []
    context = scenario.context
    for step in scenario.steps:
        context = _advance(context, step)
        decision = maker.decide(context, _dynamic_schemas(tool_specs, context))
        steps.append(
            {"tool_call": {"name": decision.tool_call.name, "arguments": decision.tool_call.arguments}}
            if decision.tool_call is not None
            else {"content": decision.model_dump_json(exclude_none=True)}
        )
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
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "steps": steps,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    return path
