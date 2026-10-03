from __future__ import annotations

from typing import Any, Protocol

from career_agent.agent.main_state import MainAgentState
from career_agent.agent.middleware.contracts import AuthorizationRefusalKind


class MiddlewareTracingHost(Protocol):
    def _record_trace_event(
        self,
        event_type: str,
        stage: str,
        *,
        outcome: str,
        details: dict[str, Any] | None = None,
        recoverable: bool | None = None,
    ) -> None: ...


class MiddlewareTracing:
    """Emit policy telemetry without coupling policy decisions to the runtime."""

    def __init__(self, *, host: MiddlewareTracingHost) -> None:
        self._host = host

    def authorization_refused(
        self,
        state: MainAgentState,
        *,
        name: str,
        kind: AuthorizationRefusalKind,
        capped: bool,
    ) -> None:
        self._host._record_trace_event(
            "authorization_refused",
            "authorize",
            outcome="failed",
            details={
                "tool_name": name,
                "refusal_kind": kind,
                "tool_profile": state["context"].task.tool_profile,
                "capped": capped,
            },
            recoverable=not capped,
        )
