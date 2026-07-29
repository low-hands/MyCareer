from __future__ import annotations

from typing import Protocol

from career_agent.contracts import AgentRequest, AgentResult


class AgentRuntime(Protocol):
    """Stable boundary implemented by every agent framework adapter."""

    async def run(self, request: AgentRequest) -> AgentResult: ...
