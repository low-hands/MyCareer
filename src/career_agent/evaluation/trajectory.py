"""Scenarios that pin what the Main Agent decides, not what it does after.

The evaluation runs at two levels, because only one of them can run without a
model:

**Contract level** — always runs, needs no API key. Builds the exact context a
scenario would send and checks that the decision is even *decidable* from it:
the facts the policy turns on are present in the projection, the tool the
scenario expects is offered, and the tools it forbids are offered too. That last
one is what keeps a scenario honest — forbidding a tool the model was never
shown proves nothing, and a suite full of vacuous scenarios is worse than none.

**Replay level** — runs for scenarios that have a recorded model response.
Checks the decision itself against the scenario's expectations.

The split matters and should not be blurred: passing at contract level says the
question was asked properly, not that the model answered it well. Only a fresh
recording says that. ``record`` refreshes them against the live model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    DecisionObservation,
    MainAgentContext,
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
    forbid_tools: frozenset[str] = frozenset()
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


def cassette_path(name: str, *, root: Path | None = None) -> Path:
    return (root or CASSETTE_ROOT) / f"{name}.json"


def load_cassette(name: str, *, root: Path | None = None) -> list[dict[str, Any]] | None:
    path = cassette_path(name, root=root)
    if not path.exists():
        return None
    return json.loads(path.read_text())["steps"]


def check_contract(
    scenario: TrajectoryScenario, *, offered_tools: frozenset[str]
) -> tuple[str, ...]:
    """Whether the scenario asks a question the model could answer.

    Runs without a model and catches the two ways an evaluation quietly stops
    testing anything: the context no longer carries the fact the policy turns
    on, or the forbidden tool is not on the menu, which makes "the model did not
    call it" true for the wrong reason.
    """
    failures = []
    projection = scenario.context.model_context()
    for path in scenario.decisive_facts:
        if not _has_path(projection, path):
            failures.append(
                f"{scenario.name}: the projection has no '{path}', so this "
                "policy is no longer decidable from what the model is sent"
            )
    for tool in scenario.tools:
        if tool not in offered_tools:
            failures.append(
                f"{scenario.name}: '{tool}' is not offered, so the scenario "
                "would pass without testing anything"
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
        if not isinstance(node, Mapping) or part not in node:
            return False
        node = node[part]
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
    if called in step.forbid_tools:
        failures.append(f"{label}: called forbidden tool '{called}'")
    return tuple(failures)


def replay(
    scenario: TrajectoryScenario,
    *,
    tool_specs: tuple[dict[str, Any], ...],
    responses: Sequence[Mapping[str, Any]],
    config: OpenAICompatibleAgentConfig | None = None,
) -> tuple[str, ...]:
    """Run a scenario against its recording through the real decision maker."""
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
        decision = maker.decide(context, tool_specs)
        failures.extend(check_step(step, decision, scenario=scenario.name, index=index))
    return tuple(failures)


def _advance(context: MainAgentContext, step: TrajectoryStep) -> MainAgentContext:
    """Apply what the runtime would have carried into this step."""
    update: dict[str, Any] = {}
    if step.user_message is not None:
        update["user_message"] = step.user_message
    if step.observation is not None:
        update["tool_observations"] = (*context.tool_observations, step.observation)[-3:]
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
        decision = maker.decide(context, tool_specs)
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
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "steps": steps,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    return path
