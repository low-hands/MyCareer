"""Compatibility views over capability reachability metadata."""

from __future__ import annotations

from career_agent.agent.capabilities.catalog import CAPABILITIES, Precondition
from career_agent.agent.contracts.task_state import ConversationTaskState


PRECONDITIONS: dict[str, Precondition] = {
    name: descriptor.precondition
    for name, descriptor in CAPABILITIES.items()
    if descriptor.precondition is not None
}

REQUIREMENTS: dict[str, str] = {
    name: descriptor.requirement
    for name, descriptor in CAPABILITIES.items()
    if descriptor.requirement is not None
}

STATE_GATED_TOOLS = frozenset(
    name for name, descriptor in CAPABILITIES.items()
    if descriptor.schema_gated
)

_REFERENCE_READBACKS = frozenset(
    name for name, descriptor in CAPABILITIES.items()
    if descriptor.reference_readback
)


def reachable(name: str, task: ConversationTaskState) -> bool:
    precondition = PRECONDITIONS.get(name)
    return precondition(task) if precondition is not None else True
