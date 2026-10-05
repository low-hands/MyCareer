from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from career_agent.agent.contracts.task_state import ConversationTaskState
from career_agent.agent.capabilities.registry import MainAgentToolRegistry
from career_agent.agent.middleware.contracts import AuthorizationRefusal
from career_agent.agent.capabilities.effects import ToolEffect, effect_for
from career_agent.agent.capabilities.reachability import (
    REQUIREMENTS,
    STATE_GATED_TOOLS,
    reachable,
)


@dataclass(frozen=True)
class AvailableCapability:
    kind: Literal["atomic_tool", "workflow"]
    effect: ToolEffect


class ToolAvailabilityMiddleware:
    """Resolve execution kind and enforce the model's offered tool set."""

    def __init__(
        self,
        *,
        tools: MainAgentToolRegistry,
    ) -> None:
        self._tools = tools

    def resolve(
        self,
        *,
        name: str,
        task: ConversationTaskState,
        offered_tool_names: tuple[str, ...],
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
        if model_selected and name in STATE_GATED_TOOLS and not reachable(name, task):
            requirement = REQUIREMENTS[name]
            return AuthorizationRefusal(
                kind="precondition",
                reason=f"{name} 当前不可执行：{requirement}。",
                next_action=requirement,
            )
        if model_selected and name not in offered_tool_names:
            return AuthorizationRefusal(
                kind="not_offered",
                reason=f"{name} 未在本次模型调用中提供。",
                next_action=(
                    "请从本次提供的工具中选择；如果所需工具不在其中，"
                    "先使用 route_to_capability 切换当前工具档。"
                ),
            )
        return AvailableCapability(kind=kind, effect=effect_for(name))
