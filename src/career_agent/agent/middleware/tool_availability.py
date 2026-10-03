from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from career_agent.agent.contracts.main_agent import (
    ConversationTaskState,
    ToolProfile,
)
from career_agent.agent.capabilities.registry import MainAgentToolRegistry
from career_agent.agent.middleware.contracts import AuthorizationRefusal
from career_agent.agent.capabilities.effects import ToolEffect, effect_for


@dataclass(frozen=True)
class AvailableCapability:
    kind: Literal["atomic_tool", "workflow"]
    effect: ToolEffect


class ToolAvailabilityMiddleware:
    """Resolve execution kind and enforce the active profile's visible tools."""

    def __init__(
        self,
        *,
        offers_tool: Callable[[str, ToolProfile, ConversationTaskState], bool],
        tools: MainAgentToolRegistry,
    ) -> None:
        self._offers_tool = offers_tool
        self._tools = tools

    def resolve(
        self,
        *,
        name: str,
        task: ConversationTaskState,
        runtime_owned: bool,
        owner_confirmed: bool,
        policy_owned: bool,
    ) -> AvailableCapability | AuthorizationRefusal:
        if runtime_owned:
            if name not in self._tools.runtime_workflow_names:
                raise ValueError(f"Unknown runtime-owned workflow: {name}")
            kind: Literal["atomic_tool", "workflow"] = "workflow"
        else:
            kind = self._tools.capability_kind(name)
        model_selected = not (runtime_owned or owner_confirmed or policy_owned)
        if model_selected and not self._offers_tool(
            name, task.tool_profile, task
        ):
            return AuthorizationRefusal(
                kind="out_of_profile",
                reason=f"{name} 不在当前 {task.tool_profile} 工具档内。",
                next_action=(
                    "先用 route_to_capability 切到该工具所属的领域，"
                    "再从 task.available_now 中选择工具。"
                ),
            )
        return AvailableCapability(kind=kind, effect=effect_for(name))
