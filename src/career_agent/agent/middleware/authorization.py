from __future__ import annotations

from typing import Literal

from career_agent.agent.main_agent_contracts import ToolObservation
from career_agent.agent.main_state import MainAgentState
from career_agent.agent.middleware.contracts import AuthorizationRefusal
from career_agent.agent.middleware.tracing import MiddlewareTracing


class AuthorizationMiddleware:
    """Apply owner policy and render all authorization-stage refusals."""

    def __init__(
        self, *, tracing: MiddlewareTracing, max_refusals: int
    ) -> None:
        self._tracing = tracing
        self._max_refusals = max_refusals

    @staticmethod
    def verdict(
        state: MainAgentState,
        *,
        name: str,
        runtime_owned: bool,
        owner_confirmed: bool,
    ) -> Literal["permit", "review"] | AuthorizationRefusal:
        verdict = (
            "permit"
            if runtime_owned or owner_confirmed
            else state["context"].preferences.capability_verdict(name)
        )
        if verdict == "deny":
            return AuthorizationRefusal(
                kind="preference_deny",
                reason="你设置的偏好不允许这个操作。",
                next_action="向用户说明这条设置，不要重试这次调用。",
            )
        return verdict

    def refuse(
        self,
        state: MainAgentState,
        *,
        name: str,
        refusal: AuthorizationRefusal,
    ) -> MainAgentState:
        control = state.get("control", {})
        capped = control.get("authorization_refusals", 0) >= self._max_refusals
        self._tracing.authorization_refused(
            state,
            name=name,
            kind=refusal.kind,
            capped=capped,
        )
        if capped:
            return {"authorization_route": "present"}
        return {
            "authorization_route": "observe",
            "pending": {
                "name": name,
                "result": ToolObservation(
                    tool_name=name,
                    state="authorization_refused",
                    message=refusal.reason,
                    next_action=refusal.next_action,
                ),
                "synthetic_kind": "authorization",
                "runtime_owned": bool(
                    state.get("pending", {}).get("runtime_owned")
                ),
            },
        }
