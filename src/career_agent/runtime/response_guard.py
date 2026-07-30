"""Terminal response guard contracts used inside the agent runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from career_agent.contracts import AgentRequest


@dataclass(frozen=True, slots=True)
class ToolCallEvidence:
    """One completed tool call observed during the current user turn."""

    name: str
    succeeded: bool
    output: str


@dataclass(frozen=True, slots=True)
class TurnEvidence:
    """Evidence available when reviewing a proposed terminal response."""

    available_tool_names: tuple[str, ...] = ()
    tool_calls: tuple[ToolCallEvidence, ...] = ()


@dataclass(frozen=True, slots=True)
class GuardDecision:
    """Decision returned before a terminal response can leave the runtime."""

    accepted: bool
    code: str = ""
    feedback: str = ""
    required_tool: str = ""
    safe_output: str = ""


class TerminalResponseGuard(Protocol):
    """Review a proposed terminal answer using current-turn execution evidence."""

    async def review(
        self,
        *,
        request: AgentRequest,
        response_text: str,
        evidence: TurnEvidence,
    ) -> GuardDecision: ...


class AllowAllResponseGuard:
    """Default no-op policy for runtimes without a domain response guard."""

    async def review(
        self,
        *,
        request: AgentRequest,
        response_text: str,
        evidence: TurnEvidence,
    ) -> GuardDecision:
        return GuardDecision(accepted=True)
