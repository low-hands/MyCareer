"""Temporary legacy profile selector; remove after search-mode migration."""

from __future__ import annotations

from dataclasses import replace
import re
from types import MappingProxyType
from typing import Any, Mapping, TypeVar

from career_agent.agent.capabilities.catalog import (
    CAPABILITIES, TOOL_PROFILE_NAMES, ToolProfile,
)
from career_agent.agent.capabilities.reachability import (
    PRECONDITIONS, REQUIREMENTS, STATE_GATED_TOOLS, reachable,
)
from career_agent.agent.capabilities.selection import CapabilitySelection
from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.contracts.task_state import ConversationTaskState


ROUTE_TOOL = "route_to_capability"
CORE_TOOLS = frozenset(
    name for name, descriptor in CAPABILITIES.items()
    if "core" in descriptor.profiles
)
TOOL_PROFILES: Mapping[ToolProfile, frozenset[str]] = MappingProxyType({
    profile: frozenset(
        name for name, descriptor in CAPABILITIES.items()
        if descriptor.model_callable
        and ("core" in descriptor.profiles or profile in descriptor.profiles)
    )
    for profile in TOOL_PROFILE_NAMES
})
ROUTABLE_TOOLS: frozenset[str] = frozenset().union(*TOOL_PROFILES.values())
if ROUTABLE_TOOLS != frozenset(
    name for name, descriptor in CAPABILITIES.items()
    if descriptor.model_callable and descriptor.legacy_profile_exposed
):
    raise RuntimeError("every legacy-exposed capability must be reachable from a profile")
if ROUTE_TOOL not in CORE_TOOLS or CAPABILITIES[ROUTE_TOOL].effect != "CONTROL":
    raise RuntimeError("the route tool must be a core CONTROL capability")

MAX_NEXT_REQUIREMENTS = 3
_TOOL_NAME = re.compile(r"\b[a-z]+(?:_[a-z]+)+\b")
Schema = TypeVar("Schema", bound=Mapping[str, object])


def profile_tools(profile: ToolProfile) -> frozenset[str]:
    return TOOL_PROFILES[profile]


def profile_schemas(
    profile: ToolProfile,
    schemas: tuple[Schema, ...],
    task: ConversationTaskState | None = None,
) -> tuple[Schema, ...]:
    offered = profile_tools(profile)
    return tuple(
        schema for schema in schemas
        if isinstance(function := schema.get("function"), Mapping)
        and function.get("name") in offered
        and (
            task is None
            or str(function.get("name")) not in STATE_GATED_TOOLS
            or reachable(str(function.get("name")), task)
        )
    )


def project_tool_availability(task: ConversationTaskState) -> dict[str, object]:
    tools = profile_tools(task.tool_profile)
    available = sorted(name for name in tools if reachable(name, task))
    requirements: dict[str, list[str]] = {}
    domain_first = sorted(tools, key=lambda name: (name in CORE_TOOLS, name))
    for name in domain_first:
        if name in PRECONDITIONS and not reachable(name, task):
            requirement = REQUIREMENTS[name]
            requirements.setdefault(requirement, []).append(name)
    available_set = frozenset(available)
    ordered = sorted(
        requirements.items(),
        key=lambda item: not (frozenset(_TOOL_NAME.findall(item[0])) & available_set),
    )
    return {
        "tool_profile": task.tool_profile,
        "available_now": available,
        "next_requirements": [
            f"{', '.join(names)}: {requirement}"
            for requirement, names in ordered[:MAX_NEXT_REQUIREMENTS]
        ],
    }


LEGACY_TOOL_POLICY = (
    "Tools are grouped into profiles (core, job, resume, application, "
    "interview, memory). The control state's task.tool_profile is the "
    "current profile, task.available_now lists its tools usable right "
    "now, and task.next_requirements names blocked tools with their "
    "unmet preconditions, not a plan to execute. Core tools are shared "
    "by every profile. Call an offered tool directly when it is needed "
    "and its preconditions hold. Use route_to_capability only when a "
    "required tool is outside the current profile; a request's topic "
    "alone does not require routing. Reassess the next required action "
    "after each result, including for requests spanning domains. "
)


def legacy_offers_tool(
    name: str, profile: ToolProfile, task: ConversationTaskState | None,
) -> bool:
    if name not in profile_tools(profile):
        return False
    return name not in STATE_GATED_TOOLS or task is None or reachable(name, task)


class LegacyProfileStrategy:
    mode = "legacy"
    ingress_profile = True

    def __init__(self) -> None:
        self._schema_cache: dict[
            tuple[ToolProfile, tuple[str, ...] | None],
            tuple[dict[str, Any], ...],
        ] = {}

    def schemas(
        self, profile: ToolProfile, task: ConversationTaskState | None,
        registered: tuple[dict[str, Any], ...],
    ) -> tuple[dict[str, Any], ...]:
        reachable_names = (
            tuple(sorted(
                str(schema.get("function", {}).get("name"))
                for schema in registered
                if schema.get("function", {}).get("name") in profile_tools(profile)
                and task is not None
                and (
                    str(schema.get("function", {}).get("name")) not in STATE_GATED_TOOLS
                    or reachable(str(schema.get("function", {}).get("name")), task)
                )
            )) if task is not None else None
        )
        key = (profile, reachable_names)
        cached = self._schema_cache.get(key)
        if cached is None:
            cached = profile_schemas(profile, registered, task)
            self._schema_cache[key] = cached
        return cached

    def select(
        self, context: MainAgentContext,
        registered: tuple[dict[str, Any], ...],
    ) -> CapabilitySelection:
        schemas = self.schemas(context.task.tool_profile, context.task, registered)
        names = tuple(str(schema["function"]["name"]) for schema in schemas)
        selection = CapabilitySelection(
            selected_names=names, offered_names=names,
            blocked_requirements=(), schemas=schemas, mode=self.mode,
        )
        return replace(selection, tool_projection=self.tool_context(context.task, selection))

    @staticmethod
    def tool_context(task: ConversationTaskState, selection: CapabilitySelection) -> dict[str, object]:
        return project_tool_availability(task)

    @staticmethod
    def offers_tool(context: MainAgentContext, name: str) -> bool:
        return legacy_offers_tool(name, context.task.tool_profile, context.task)

    @staticmethod
    def tool_policy() -> str:
        return LEGACY_TOOL_POLICY
