"""Compatibility views over the unified Main Agent capability catalogue."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from career_agent.agent.capabilities.catalog import (
    CAPABILITIES,
    ApprovalPolicy,
    ToolEffect,
    capability,
)


TOOL_EFFECTS: Mapping[str, ToolEffect] = MappingProxyType(
    {name: descriptor.effect for name, descriptor in CAPABILITIES.items()}
)


def replay_safe(name: str) -> bool:
    descriptor = CAPABILITIES.get(name)
    return descriptor.replay_safe if descriptor is not None else False


def approval_policy(name: str) -> ApprovalPolicy:
    """Return the minimum approval rule enforced by the runtime."""

    return capability(name).approval_policy


def is_notes_guarded(name: str) -> bool:
    descriptor = CAPABILITIES.get(name)
    return descriptor.notes_guarded if descriptor is not None else False


def is_preference_bound(name: str) -> bool:
    descriptor = CAPABILITIES.get(name)
    return descriptor.preference_bound if descriptor is not None else False


def effect_for(name: str) -> ToolEffect:
    return capability(name).effect


def is_external_write(name: str) -> bool:
    descriptor = CAPABILITIES.get(name)
    return descriptor.external_write if descriptor is not None else False


def declared_write_capabilities() -> frozenset[str]:
    return frozenset(
        name for name, descriptor in CAPABILITIES.items()
        if descriptor.effect == "WRITE"
    )


def is_runtime_owned(name: str) -> bool:
    descriptor = CAPABILITIES.get(name)
    return descriptor.runtime_owned if descriptor is not None else False


def owner_rule_capabilities() -> frozenset[str]:
    """Capabilities that an owner ``confirm_before`` rule may name."""

    return frozenset(
        name for name, descriptor in CAPABILITIES.items()
        if descriptor.effect == "WRITE" and not descriptor.runtime_owned
    )
