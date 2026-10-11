from __future__ import annotations

from collections.abc import Callable

from career_agent.agent.runtime.state import MainAgentState
from career_agent.agent.middleware.contracts import AuthorizationRefusalKind


class MiddlewareTracing:
    """Emit policy telemetry without coupling policy decisions to the runtime."""

    def __init__(self, *, record_trace_event: Callable[..., None]) -> None:
        self._record_trace_event = record_trace_event

    def authorization_refused(
        self,
        state: MainAgentState,
        *,
        name: str,
        kind: AuthorizationRefusalKind,
        capped: bool,
    ) -> None:
        self._record_trace_event(
            "authorization_refused",
            "authorize",
            outcome="failed",
            details={
                "tool_name": name,
                "refusal_kind": kind,
                "capped": capped,
            },
            recoverable=not capped,
        )
