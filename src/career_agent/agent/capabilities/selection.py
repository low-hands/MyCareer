"""Validate a model call's selected capabilities and derive their schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from career_agent.agent.capabilities.catalog import CAPABILITIES
from career_agent.agent.capabilities.reachability import reachable
from career_agent.agent.contracts.task_state import ConversationTaskState


# Baseline candidates before task-specific loading. History readback also
# requires compressed history in SearchStrategy.
ALWAYS_OFFERED_TOOLS: tuple[str, ...] = (
    "search_capabilities",
    "load_skill",
    "read_conversation_span",
    "fetch_archived_constraints",
    "search_career_memory",
)


@dataclass(frozen=True)
class CapabilitySelection:
    """One model call's validated tool offer, including blocked prerequisites."""

    selected_names: tuple[str, ...]
    offered_names: tuple[str, ...]
    blocked_requirements: tuple[tuple[str, str], ...]
    schemas: tuple[dict[str, Any], ...]
    sources: tuple[tuple[str, str], ...] = ()
    waiting_suppressed: tuple[str, ...] = ()
    tool_projection: dict[str, object] | None = None


def prepare_capability_selection(
    selected_names: Iterable[str],
    *,
    task: ConversationTaskState,
    registered_schemas: Iterable[Mapping[str, Any]],
) -> CapabilitySelection:
    """Validate names and derive schemas, availability and execution offer once.

    The selector may name tools from several domains. A name with an unmet
    precondition stays visible as a requirement, but its schema is not offered.
    An unknown or runtime-only name is an error; callers must handle selector
    failure explicitly rather than silently exposing all registered tools.
    """

    requested = tuple(dict.fromkeys(selected_names))
    for name in requested:
        descriptor = CAPABILITIES.get(name)
        if descriptor is None or not descriptor.model_callable:
            raise ValueError(f"unknown or runtime-only capability: {name}")
    # Keep the catalogue's stable schema order even if the selector lists the
    # same capabilities in a different order on a later model call.
    requested_set = frozenset(requested)
    names = tuple(name for name in CAPABILITIES if name in requested_set)

    schemas_by_name: dict[str, dict[str, Any]] = {}
    for schema in registered_schemas:
        function = schema.get("function")
        if not isinstance(function, Mapping) or not isinstance(function.get("name"), str):
            raise ValueError("registered tool schema has no function name")
        name = function["name"]
        if name in schemas_by_name:
            raise ValueError(f"duplicate registered tool schema: {name}")
        schemas_by_name[name] = dict(schema)

    offered: list[str] = []
    blocked: list[tuple[str, str]] = []
    for name in names:
        if name not in schemas_by_name:
            raise ValueError(f"selected capability has no registered schema: {name}")
        if reachable(name, task):
            offered.append(name)
        else:
            requirement = CAPABILITIES[name].requirement
            if not requirement:
                raise ValueError(f"blocked capability has no requirement: {name}")
            blocked.append((name, requirement))

    return CapabilitySelection(
        selected_names=names,
        offered_names=tuple(offered),
        blocked_requirements=tuple(blocked),
        schemas=tuple(schemas_by_name[name] for name in offered),
    )
