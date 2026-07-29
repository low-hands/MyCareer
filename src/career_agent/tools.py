from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool

from career_agent.contracts import ToolPermission


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    tool: BaseTool
    permission: ToolPermission
    timeout_seconds: float = 30.0


class ToolRegistry:
    """Central inventory used to apply policy before tools reach the model."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        tool: BaseTool,
        *,
        permission: ToolPermission = ToolPermission.READ,
        timeout_seconds: float = 30.0,
    ) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._tools[tool.name] = RegisteredTool(
            tool=tool,
            permission=permission,
            timeout_seconds=timeout_seconds,
        )

    def allowed(self, permissions: set[ToolPermission]) -> list[BaseTool]:
        return [item.tool for item in self.allowed_entries(permissions)]

    def allowed_entries(self, permissions: set[ToolPermission]) -> tuple[RegisteredTool, ...]:
        return tuple(item for item in self._tools.values() if item.permission in permissions)

    async def execute(
        self,
        *,
        name: str,
        arguments: dict[str, Any],
        permissions: set[ToolPermission],
    ) -> str:
        """Execute an allowed tool with its configured timeout."""

        registered = self._tools.get(name)
        if registered is None:
            return self._error_payload("unknown_tool", f"Tool is not registered: {name}")
        if registered.permission not in permissions:
            return self._error_payload(
                "permission_denied",
                f"Tool requires '{registered.permission.value}' permission",
            )

        try:
            result = await asyncio.wait_for(
                registered.tool.ainvoke(arguments),
                timeout=registered.timeout_seconds,
            )
        except TimeoutError:
            return self._error_payload(
                "timeout",
                f"Tool exceeded {registered.timeout_seconds:g} seconds",
            )
        except Exception as exc:
            return self._error_payload("tool_error", str(exc))

        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, default=str)

    @staticmethod
    def _error_payload(code: str, message: str) -> str:
        return json.dumps(
            {"ok": False, "error": {"code": code, "message": message}},
            ensure_ascii=False,
        )

    def describe(self) -> tuple[RegisteredTool, ...]:
        return tuple(self._tools.values())
