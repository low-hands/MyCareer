"""Tool-profile views derived from the unified capability catalogue."""

from __future__ import annotations

import re
from types import MappingProxyType
from typing import Mapping, TypeVar

from career_agent.agent.capabilities.catalog import (
    CAPABILITIES,
    TOOL_PROFILE_NAMES,
    ToolProfile,
)
from career_agent.agent.contracts.main_agent import ConversationTaskState
from career_agent.agent.capabilities.reachability import (
    PRECONDITIONS,
    REQUIREMENTS,
    STATE_GATED_TOOLS,
    reachable,
)


ROUTE_TOOL = "route_to_capability"
CORE_TOOLS = frozenset(
    name for name, descriptor in CAPABILITIES.items()
    if "core" in descriptor.profiles
)

TOOL_PROFILES: Mapping[ToolProfile, frozenset[str]] = MappingProxyType(
    {
        profile: frozenset(
            name
            for name, descriptor in CAPABILITIES.items()
            if descriptor.model_callable
            and ("core" in descriptor.profiles or profile in descriptor.profiles)
        )
        for profile in TOOL_PROFILE_NAMES
    }
)

ROUTABLE_TOOLS: frozenset[str] = frozenset().union(*TOOL_PROFILES.values())

if ROUTABLE_TOOLS != frozenset(
    name for name, descriptor in CAPABILITIES.items() if descriptor.model_callable
):
    raise RuntimeError("every model-callable capability must be reachable from a profile")
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
        schema
        for schema in schemas
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
        key=lambda item: not (
            frozenset(_TOOL_NAME.findall(item[0])) & available_set
        ),
    )
    return {
        "tool_profile": task.tool_profile,
        "available_now": available,
        "next_requirements": [
            f"{', '.join(names)}: {requirement}"
            for requirement, names in ordered[:MAX_NEXT_REQUIREMENTS]
        ],
    }
