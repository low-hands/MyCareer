from __future__ import annotations

from typing import Protocol

from career_agent.contracts import AgentRequest, PreparedContext
from career_agent.prompts import CORE_AGENT_SYSTEM_PROMPT, render_system_prompt


class ContextManager(Protocol):
    async def prepare(self, request: AgentRequest) -> PreparedContext: ...


class BasicContextManager:
    """Initial context policy; replaceable without changing the runtime contract."""

    def __init__(self, system_prompt: str = CORE_AGENT_SYSTEM_PROMPT) -> None:
        self._system_prompt = system_prompt

    async def prepare(self, request: AgentRequest) -> PreparedContext:
        facts = [str(item) for item in request.metadata.get("facts", [])]
        return PreparedContext(
            system_prompt=render_system_prompt(
                base_prompt=self._system_prompt,
                facts=facts,
            ),
            facts=facts,
        )
