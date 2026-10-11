from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Literal

from career_agent.agent.contracts.task_state import ConversationTaskState
from career_agent.agent.capabilities.catalog import CAPABILITIES
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

    def __init__(self, *, tools: MainAgentToolRegistry) -> None:
        self._tools = tools

    def resolve(
        self,
        *,
        name: str,
        task: ConversationTaskState,
        offered_tool_names: tuple[str, ...],
        waiting_tool_names: Collection[str] = (),
        runtime_owned: bool,
        owner_confirmed: bool,
        policy_owned: bool,
    ) -> AvailableCapability | AuthorizationRefusal:
        if runtime_owned:
            if name not in self._tools.runtime_workflow_names:
                raise ValueError(f"Unknown runtime-owned workflow: {name}")
            kind: Literal["atomic_tool", "workflow"] = "workflow"
        elif name not in CAPABILITIES:
            # Keep unknown names as hard errors. The refusal path below is
            # reserved for catalogued capabilities whose current state blocks
            # them before execution.
            raise ValueError(f"Unknown main-agent capability: {name}")
        model_selected = not (runtime_owned or owner_confirmed or policy_owned)
        if model_selected and name in waiting_tool_names:
            return AuthorizationRefusal(
                kind="waiting_for_user",
                reason=f"{name} 正在等待用户确认或回答，本轮不能再次调用。",
                next_action="等待用户回应，不要重新加载或重复调用这个工具。",
            )
        if (
            model_selected
            and (name in STATE_GATED_TOOLS or name not in offered_tool_names)
            and not reachable(name, task)
        ):
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
                next_action="请从本次提供的工具中选择；如果所需工具不在其中，先调用 search_capabilities。",
            )
        if not runtime_owned:
            kind = self._tools.capability_kind(name)
        return AvailableCapability(kind=kind, effect=effect_for(name))
