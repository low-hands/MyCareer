"""Offline, model-free evaluation of tools offered at each decision step."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol, Sequence

from career_agent.agent.capabilities.catalog import CAPABILITIES
from career_agent.agent.capabilities.profiles import CORE_TOOLS, profile_schemas, profile_tools
from career_agent.agent.capabilities.proactive import proactive_tool_names
from career_agent.agent.capabilities.reachability import reachable
from career_agent.agent.capabilities.search import search_catalog
from career_agent.agent.capabilities.selection_strategy import SearchStrategy
from career_agent.agent.capabilities.waiting import WAITING_FOR_USER_STATES
from career_agent.agent.context.turn_builder import keyword_tool_profile
from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.providers.token_budget import count_tokens
from career_agent.evaluation.trajectory import TrajectoryScenario, advance_trajectory_context


@dataclass(frozen=True)
class SelectionCase:
    scenario: TrajectoryScenario
    split: Literal["dev", "holdout"]
    kind: Literal["single", "cross", "chain", "control"]
    namespaces: frozenset[str]
    raw_turn: bool = True


@dataclass(frozen=True)
class ToolOffer:
    names: frozenset[str]
    selected_names: frozenset[str]
    schemas: tuple[dict[str, Any], ...]
    sources: Mapping[str, str]


class ToolSelector(Protocol):
    def select(
        self, context: MainAgentContext, prior: object | None
    ) -> tuple[ToolOffer, object | None]: ...


class LegacyProfileSelector:
    """The current decide-time profile/schema projection, without model calls.

    Scenario contexts are already decision snapshots, after turn ingress. Use
    ``select_ingress`` only when evaluating a raw turn before that projection.
    """

    def __init__(self, tool_specs: tuple[dict[str, Any], ...]) -> None:
        self._tool_specs = tool_specs

    def select(
        self, context: MainAgentContext, prior: object | None
    ) -> tuple[ToolOffer, object | None]:
        profile = context.task.tool_profile
        schemas = profile_schemas(profile, self._tool_specs, context.task)
        names = frozenset(schema["function"]["name"] for schema in schemas)
        registered = frozenset(schema["function"]["name"] for schema in self._tool_specs)
        selected = profile_tools(profile) & registered
        sources = {
            name: "core" if name in CORE_TOOLS else f"profile:{profile}"
            for name in names
        }
        return ToolOffer(
            names=names, selected_names=selected, schemas=schemas, sources=sources
        ), prior

    def select_ingress(
        self, context: MainAgentContext, prior: object | None = None
    ) -> tuple[ToolOffer, object | None]:
        _, offer, state = self.select_ingress_snapshot(context, prior)
        return offer, state

    def select_ingress_snapshot(
        self, context: MainAgentContext, prior: object | None = None
    ) -> tuple[MainAgentContext, ToolOffer, object | None]:
        """Apply the actual raw-turn keyword promotion before deciding."""
        profile = context.task.tool_profile
        if profile == "core":
            profile = keyword_tool_profile(context.user_message) or profile
        if profile != context.task.tool_profile:
            context = context.model_copy(update={
                "task": context.task.model_copy(update={"tool_profile": profile})
            })
        offer, state = self.select(context, prior)
        return context, offer, state


class SearchSimulationSelector:
    """Offline upper bound or lexical search using the original user wording."""

    def __init__(self, tool_specs: tuple[dict[str, Any], ...], *, ideal: bool) -> None:
        self._tool_specs = tool_specs
        self._ideal = ideal
        # Keep the historical search-only comparator after W enters runtime.
        self._strategy = SearchStrategy(proactive_enabled=False)

    def select(
        self, context: MainAgentContext, prior: object | None,
    ) -> tuple[ToolOffer, object | None]:
        loaded = tuple(prior) if isinstance(prior, tuple) else ()
        task = context.task.add_loaded_capabilities(loaded)
        selection = self._strategy.select(context.model_copy(update={"task": task}), self._tool_specs)
        names = frozenset(selection.offered_names)
        return ToolOffer(
            names=names, selected_names=frozenset(selection.selected_names),
            schemas=selection.schemas, sources=dict(selection.sources),
        ), loaded

    def select_for_demand(
        self, context: MainAgentContext, prior: object | None,
        required: frozenset[str],
    ) -> tuple[ToolOffer, object | None, bool, bool]:
        offer, loaded = self.select(context, prior)
        if not required or required & offer.names:
            return offer, loaded, False, False
        if self._ideal:
            found = tuple(name for name in CAPABILITIES if name in required)[:1]
        else:
            found = search_catalog(query=context.user_message, limit=5)
        if found:
            loaded = tuple(dict.fromkeys((*loaded, *found)))
            offer, loaded = self.select(context, loaded)
        return offer, loaded, True, not bool(required & offer.names)


class BatchSearchSimulationSelector(SearchSimulationSelector):
    """Optimistic bound: load known same-turn demands in one names call.

    The oracle sees later answer-key tools before the model would. At most ten
    names are loaded per call, matching the search tool's argument contract.
    """

    def __init__(self, tool_specs: tuple[dict[str, Any], ...]) -> None:
        super().__init__(tool_specs, ideal=True)

    def select_for_demand(
        self, context: MainAgentContext, prior: object | None,
        required: frozenset[str], *, same_turn_demands: tuple[str, ...],
    ) -> tuple[ToolOffer, object | None, bool, bool]:
        offer, loaded = self.select(context, prior)
        if not required or required & offer.names:
            return offer, loaded, False, False
        # The first name must satisfy this decision. Remaining names are the
        # optimistic look-ahead, capped by search_capabilities.names (10).
        current = next(name for name in CAPABILITIES if name in required)
        found = tuple(dict.fromkeys((current, *same_turn_demands)))[:10]
        loaded = tuple(dict.fromkeys((*loaded, *found)))
        offer, loaded = self.select(context, loaded)
        return offer, loaded, True, not bool(required & offer.names)


class ProactiveSearchSimulationSelector(SearchSimulationSelector):
    """Ideal search plus reviewed successors and bound-resource reads."""

    def __init__(self, tool_specs: tuple[dict[str, Any], ...]) -> None:
        super().__init__(tool_specs, ideal=True)

    def select(
        self, context: MainAgentContext, prior: object | None,
    ) -> tuple[ToolOffer, object | None]:
        loaded = tuple(prior) if isinstance(prior, tuple) else ()
        proactive = proactive_tool_names(context)
        task = context.task.add_loaded_capabilities((*loaded, *proactive))
        selection = self._strategy.select(
            context.model_copy(update={"task": task}), self._tool_specs,
        )
        return ToolOffer(
            names=frozenset(selection.offered_names),
            selected_names=frozenset(selection.selected_names),
            schemas=selection.schemas,
            sources={
                name: ("proactive" if name in proactive and name not in loaded else source)
                for name, source in selection.sources
            },
        ), loaded


@dataclass(frozen=True)
class SelectionStep:
    scenario: str
    split: Literal["dev", "holdout"] | None
    kind: Literal["single", "cross", "chain", "control"] | None
    intent_namespaces: frozenset[str] | None
    index: int
    profile: str
    offered_names: frozenset[str]
    selected_names: frozenset[str]
    sources: Mapping[str, str]
    required_names: frozenset[str]
    route_round_trip: bool
    legacy_only: bool
    missing_reasons: Mapping[str, Literal["not_selected", "unreachable"]]
    unreachable_offered: frozenset[str]
    waiting_reoffered: frozenset[str]
    unrequested_writes: frozenset[str]
    schema_tokens_proxy: int
    search_round_trip: bool = False
    search_failed: bool = False

    @property
    def has_demand(self) -> bool:
        return not self.legacy_only and bool(self.required_names)

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
    def route_round_trips(self) -> int:
        return sum(step.route_round_trip for step in self.steps)

    @property
    def search_round_trips(self) -> int:
        return sum(step.search_round_trip for step in self.steps)

    @property
    def search_failures(self) -> int:
        return sum(step.search_failed for step in self.steps)

    @property
    def comparable_steps(self) -> tuple[SelectionStep, ...]:
        return tuple(step for step in self.steps if not step.legacy_only)

    @property
    def demanded_tool_names(self) -> frozenset[str]:
        return frozenset().union(
            *(step.required_names for step in self.steps if step.has_demand)
        )

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
    if step.expect_tool == "route_to_capability":
        # Route is an implementation cost. Grade whether the business tool it
        # unlocks was already available before that extra decision round-trip.
        if index + 1 < len(scenario.steps):
            following = scenario.steps[index + 1]
            return frozenset(
                following.expect_tools
                or ({following.expect_tool} if following.expect_tool else set())
            ) - {"route_to_capability"}
        return frozenset()
    return frozenset(step.expect_tools or ({step.expect_tool} if step.expect_tool else set()))


def _is_legacy_only(scenario: TrajectoryScenario, index: int) -> bool:
    return index > 0 and scenario.steps[index - 1].expect_tool == "route_to_capability"


def _same_turn_demand_names(
    scenario: TrajectoryScenario, index: int,
) -> tuple[str, ...]:
    """Oracle names up to, but never across, the next user message."""
    wanted: list[str] = []
    for offset in range(index, len(scenario.steps)):
        if offset > index and scenario.steps[offset].user_message is not None:
            break
        if _is_legacy_only(scenario, offset):
            continue
        demand = _expected_business_tools(scenario, offset)
        if demand:
            wanted.append(next(name for name in CAPABILITIES if name in demand))
    return tuple(dict.fromkeys(wanted))


def evaluate_tool_selection(
    scenarios: Sequence[TrajectoryScenario | SelectionCase], *, selector: ToolSelector
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
            legacy_only = _is_legacy_only(scenario, index)
            required = (
                frozenset() if legacy_only else _expected_business_tools(scenario, index)
            )
            search_round_trip = False
            search_failed = False
            if (
                case is not None
                and case.raw_turn
                and (index == 0 or step.user_message is not None)
                and isinstance(selector, LegacyProfileSelector)
            ):
                context, offer, prior = selector.select_ingress_snapshot(context, prior)
            elif isinstance(selector, BatchSearchSimulationSelector):
                offer, prior, search_round_trip, search_failed = selector.select_for_demand(
                    context, prior, required,
                    same_turn_demands=_same_turn_demand_names(scenario, index),
                )
            elif isinstance(selector, SearchSimulationSelector):
                offer, prior, search_round_trip, search_failed = selector.select_for_demand(
                    context, prior, required,
                )
            else:
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
                profile=context.task.tool_profile,
                offered_names=offer.names,
                selected_names=offer.selected_names,
                sources=offer.sources,
                required_names=required,
                route_round_trip=(
                    isinstance(selector, LegacyProfileSelector)
                    and step.expect_tool == "route_to_capability"
                ),
                legacy_only=legacy_only,
                missing_reasons=missing_reasons,
                unreachable_offered=unreachable_offered,
                waiting_reoffered=waiting_reoffered,
                unrequested_writes=unrequested_writes,
                schema_tokens_proxy=count_tokens(schema_json),
                search_round_trip=search_round_trip,
                search_failed=search_failed,
            ))
    return SelectionReport(tuple(steps))


FailureKind = Literal[
    "selection_gap_and_model_decision", "selection_gap", "model_decision", "other_behavior"
]


def classify_recorded_failure(
    scenario: TrajectoryScenario,
    *,
    selection_steps: Sequence[SelectionStep],
    recordings: Sequence[Sequence[Mapping[str, Any]]],
) -> FailureKind:
    """Attribute a known replay failure using offers and recorded tool calls.

    The caller first checks replay assertions. This function intentionally does
    not label a final-answer or grounding failure as a tool-selection failure.
    """

    selection_gap = any(step.missing_names for step in selection_steps)
    model_decision = False
    for recording in recordings:
        for index, raw in enumerate(recording):
            if index >= len(scenario.steps):
                continue
            expected_step = scenario.steps[index]
            offered = selection_steps[index].offered_names
            raw_call = raw.get("tool_call")
            called = raw_call.get("name") if isinstance(raw_call, Mapping) else None
            expected = expected_step.expect_tools or (
                frozenset({expected_step.expect_tool})
                if expected_step.expect_tool else frozenset()
            )
            if expected and expected & offered and called not in expected:
                model_decision = True
            if called in expected_step.forbid_tools and called in offered:
                model_decision = True
    if selection_gap and model_decision:
        return "selection_gap_and_model_decision"
    if selection_gap:
        return "selection_gap"
    if model_decision:
        return "model_decision"
    return "other_behavior"
