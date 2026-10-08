"""Offline, model-free evaluation of tools offered at each decision step."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from career_agent.agent.capabilities.catalog import CAPABILITIES
from career_agent.agent.capabilities.reachability import reachable
from career_agent.agent.capabilities.selection_strategy import SearchStrategy
from career_agent.agent.capabilities.waiting import WAITING_FOR_USER_STATES
from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.providers.token_budget import count_tokens
from career_agent.evaluation.trajectory import TrajectoryScenario, advance_trajectory_context


@dataclass(frozen=True)
class SelectionCase:
    scenario: TrajectoryScenario
    split: Literal["dev", "holdout"]
    kind: Literal["single", "cross", "chain", "control"]
    namespaces: frozenset[str]


@dataclass(frozen=True)
class ToolOffer:
    names: frozenset[str]
    selected_names: frozenset[str]
    schemas: tuple[dict[str, Any], ...]
    sources: Mapping[str, str]


class RuntimeSearchSelector:
    """Exact zero-round-trip offer from the runtime SearchStrategy."""

    def __init__(self, tool_specs: tuple[dict[str, Any], ...]) -> None:
        self._tool_specs = tool_specs
        self._strategy = SearchStrategy()

    def select(
        self, context: MainAgentContext, prior: object | None,
    ) -> tuple[ToolOffer, object | None]:
        selection = self._strategy.select(context, self._tool_specs)
        return ToolOffer(
            names=frozenset(selection.offered_names),
            selected_names=frozenset(selection.selected_names),
            schemas=selection.schemas,
            sources=dict(selection.sources),
        ), prior


@dataclass(frozen=True)
class SelectionStep:
    scenario: str
    split: Literal["dev", "holdout"] | None
    kind: Literal["single", "cross", "chain", "control"] | None
    intent_namespaces: frozenset[str] | None
    index: int
    offered_names: frozenset[str]
    selected_names: frozenset[str]
    sources: Mapping[str, str]
    required_names: frozenset[str]
    missing_reasons: Mapping[str, Literal["not_selected", "unreachable"]]
    unreachable_offered: frozenset[str]
    waiting_reoffered: frozenset[str]
    unrequested_writes: frozenset[str]
    schema_tokens_proxy: int

    @property
    def has_demand(self) -> bool:
        return bool(self.required_names)

    @property
    def covered(self) -> bool:
        return not self.has_demand or bool(self.required_names & self.offered_names)

    @property
    def missing_names(self) -> frozenset[str]:
        return frozenset(self.missing_reasons)


@dataclass(frozen=True)
class SelectionReport:
    steps: tuple[SelectionStep, ...]

    @property
    def demand_steps(self) -> int:
        return sum(step.has_demand for step in self.steps)

    @property
    def covered_steps(self) -> int:
        return sum(step.has_demand and step.covered for step in self.steps)

    @property
    def comparable_steps(self) -> tuple[SelectionStep, ...]:
        return self.steps

    @property
    def unreachable_offer_count(self) -> int:
        return sum(len(step.unreachable_offered) for step in self.comparable_steps)

    @property
    def waiting_reoffer_count(self) -> int:
        return sum(len(step.waiting_reoffered) for step in self.comparable_steps)

    @property
    def unrequested_write_offer_count(self) -> int:
        return sum(len(step.unrequested_writes) for step in self.comparable_steps)


def _expected_business_tools(scenario: TrajectoryScenario, index: int) -> frozenset[str]:
    step = scenario.steps[index]
    return frozenset(step.expect_tools or ({step.expect_tool} if step.expect_tool else set()))


def evaluate_tool_selection(
    scenarios: Sequence[TrajectoryScenario | SelectionCase], *, selector: RuntimeSearchSelector
) -> SelectionReport:
    steps: list[SelectionStep] = []
    for item in scenarios:
        case = item if isinstance(item, SelectionCase) else None
        scenario = case.scenario if case else item
        context = scenario.context
        turn_observations = list(context.tool_observations)
        prior: object | None = None
        for index, step in enumerate(scenario.steps):
            if step.user_message is not None:
                turn_observations = []
            context = advance_trajectory_context(context, step)
            if step.observation is not None and step.user_message is None:
                turn_observations.append(step.observation)
            required = _expected_business_tools(scenario, index)
            offer, prior = selector.select(context, prior)
            if not offer.names <= offer.selected_names:
                raise ValueError("selector offered a tool it did not select")
            schema_names = frozenset(schema["function"]["name"] for schema in offer.schemas)
            if schema_names != offer.names:
                raise ValueError("selector schemas do not match offered names")
            schema_json = json.dumps(
                offer.schemas, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            missing_reasons: dict[str, Literal["not_selected", "unreachable"]] = {}
            if required and not (required & offer.names):
                for name in required:
                    if name not in offer.selected_names:
                        missing_reasons[name] = "not_selected"
                    elif not reachable(name, context.task):
                        missing_reasons[name] = "unreachable"
                    else:
                        raise ValueError(f"selected reachable tool was not offered: {name}")
            unreachable_offered = frozenset(
                name for name in offer.names if not reachable(name, context.task)
            )
            waiting_reoffered = frozenset(
                observation.tool_name
                for observation in turn_observations
                if observation.state in WAITING_FOR_USER_STATES
                and observation.tool_name in offer.names
            )
            unrequested_writes = frozenset(
                name for name in offer.names
                if case is not None
                and CAPABILITIES[name].effect == "WRITE"
                and name not in required
                and CAPABILITIES[name].namespace not in case.namespaces
            )
            steps.append(SelectionStep(
                scenario=scenario.name,
                split=case.split if case else None,
                kind=case.kind if case else None,
                intent_namespaces=case.namespaces if case else None,
                index=index,
                offered_names=offer.names,
                selected_names=offer.selected_names,
                sources=offer.sources,
                required_names=required,
                missing_reasons=missing_reasons,
                unreachable_offered=unreachable_offered,
                waiting_reoffered=waiting_reoffered,
                unrequested_writes=unrequested_writes,
                schema_tokens_proxy=count_tokens(schema_json),
            ))
    return SelectionReport(tuple(steps))
