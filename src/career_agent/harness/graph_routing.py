from __future__ import annotations

from typing import Literal

from career_agent.agent.runtime.state import MainAgentState


class GraphRoutingPolicy:
    """Pure transition rules for the fixed main-agent graph."""

    @staticmethod
    def route_entry(
        state: MainAgentState,
    ) -> Literal["hydrate", "authorize"]:
        pending = state.get("pending", {})
        return (
            "authorize"
            if pending.get("runtime_owned")
            or pending.get("owner_confirmed")
            or pending.get("policy_owned")
            else "hydrate"
        )

    @staticmethod
    def route_decision(
        state: MainAgentState,
    ) -> Literal["authorize", "present", "interrupt"]:
        decision = state["decision"]
        if decision.action == "tool_call":
            if decision.tool_call is None:
                raise ValueError("tool_call action requires tool_call arguments")
            return "authorize"
        if decision.action in {"ask_user", "questionnaire"}:
            if any(
                item.disposition == "failed"
                for item in state.get("tool_results", ())
            ):
                return "present"
            return "interrupt"
        return "present"

    @staticmethod
    def after_authorize(
        state: MainAgentState,
    ) -> Literal["act", "observe", "present", "interrupt"]:
        return state["authorization_route"]

    @staticmethod
    def after_observe(
        state: MainAgentState,
    ) -> Literal["hydrate", "decide", "present", "interrupt"]:
        pending = state["pending"]
        result = pending["result"]
        if result.disposition == "interaction_required":
            return "interrupt"
        if pending.get("policy_prelude"):
            return "hydrate"
        if (
            pending.get("runtime_owned")
            or pending.get("owner_confirmed")
            or pending.get("policy_owned")
        ):
            return "present"
        if (
            result.disposition != "failed"
            and result.payload.get("turn_complete") is True
        ):
            return "present"
        return "decide"
