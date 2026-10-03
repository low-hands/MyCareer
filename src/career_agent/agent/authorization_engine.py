from __future__ import annotations

from typing import Protocol

from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.agent.main_state import LoopControl, MainAgentState
from career_agent.agent.middleware.approval import ApprovalMiddleware
from career_agent.agent.middleware.argument_projection import (
    ArgumentProjectionHost,
    ArgumentProjectionMiddleware,
    ProjectionRefusal,
)
from career_agent.agent.middleware.authorization import (
    AuthorizationMiddleware,
)
from career_agent.agent.middleware.budget import BudgetMiddleware
from career_agent.agent.middleware.contracts import AuthorizationRefusal
from career_agent.agent.middleware.idempotency import (
    IdempotencyAccepted,
    IdempotencyMiddleware,
)
from career_agent.agent.middleware.tool_availability import (
    AvailableCapability,
    ToolAvailabilityHost,
    ToolAvailabilityMiddleware,
)
from career_agent.agent.middleware.tracing import (
    MiddlewareTracing,
    MiddlewareTracingHost,
)
from career_agent.agent.middleware.working_notes_guard import (
    WorkingNotesGuardMiddleware,
)
from career_agent.agent.tool_effects import ToolEffect
from career_agent.storage.capability_confirmations import (
    SQLiteCapabilityConfirmationStore,
)


class AuthorizationHost(
    ArgumentProjectionHost,
    MiddlewareTracingHost,
    ToolAvailabilityHost,
    Protocol,
):
    """Runtime hooks consumed by the authorization middleware pipeline."""


class AuthorizationEngine:
    """Run the ordered policy pipeline behind LangGraph's authorize node.

    Each middleware owns one policy axis. This coordinator only preserves their
    ordering and converts an accepted request into the graph's pending action.
    """

    def __init__(
        self,
        *,
        host: AuthorizationHost,
        tools: MainAgentToolRegistry,
        confirmation_store: SQLiteCapabilityConfirmationStore | None,
        max_read_calls: int,
        max_write_calls: int,
        max_external_write_calls: int,
        max_projection_refusals: int,
        max_authorization_refusals: int,
        max_failure_retries: int,
    ) -> None:
        self._availability = ToolAvailabilityMiddleware(host=host, tools=tools)
        self._tracing = MiddlewareTracing(host=host)
        self._authorization = AuthorizationMiddleware(
            tracing=self._tracing,
            max_refusals=max_authorization_refusals,
        )
        self._budget = BudgetMiddleware(
            max_read_calls=max_read_calls,
            max_write_calls=max_write_calls,
            max_external_write_calls=max_external_write_calls,
        )
        self._idempotency = IdempotencyMiddleware(
            max_failure_retries=max_failure_retries
        )
        self._argument_projection = ArgumentProjectionMiddleware(
            host=host,
            max_projection_refusals=max_projection_refusals,
        )
        self._working_notes_guard = WorkingNotesGuardMiddleware(
            max_projection_refusals=max_projection_refusals
        )
        self._approval = ApprovalMiddleware(
            tools=tools,
            confirmation_store=confirmation_store,
        )

    def _refuse(
        self,
        state: MainAgentState,
        *,
        name: str,
        refusal: AuthorizationRefusal,
    ) -> MainAgentState:
        return self._authorization.refuse(
            state,
            name=name,
            refusal=refusal,
        )

    def authorize(self, state: MainAgentState) -> MainAgentState:
        decision = state["decision"]
        if decision.tool_call is None:
            raise ValueError("tool_call action requires tool_call arguments")
        name = decision.tool_call.name
        pending = state.get("pending", {})
        runtime_owned = bool(pending.get("runtime_owned"))
        owner_confirmed = bool(pending.get("owner_confirmed"))
        policy_owned = bool(pending.get("policy_owned"))
        policy_prelude = bool(pending.get("policy_prelude"))

        capability = self._availability.resolve(
            name=name,
            task=state["context"].task,
            runtime_owned=runtime_owned,
            owner_confirmed=owner_confirmed,
            policy_owned=policy_owned,
        )
        if isinstance(capability, AuthorizationRefusal):
            return self._refuse(state, name=name, refusal=capability)
        assert isinstance(capability, AvailableCapability)

        verdict = self._authorization.verdict(
            state,
            name=name,
            runtime_owned=runtime_owned,
            owner_confirmed=owner_confirmed,
        )
        if isinstance(verdict, AuthorizationRefusal):
            return self._refuse(state, name=name, refusal=verdict)

        control = state.get("control", {})
        budget_refusal = self._budget.check(
            control,
            name=name,
            effect=capability.effect,
        )
        if budget_refusal is not None:
            return self._refuse(state, name=name, refusal=budget_refusal)

        replay = self._idempotency.check(control, decision=decision)
        if isinstance(replay, AuthorizationRefusal):
            return self._refuse(state, name=name, refusal=replay)
        assert isinstance(replay, IdempotencyAccepted)
        control = replay.control

        projection = self._argument_projection.project(
            state,
            name=name,
            kind=capability.kind,
            runtime_owned=runtime_owned,
            owner_confirmed=owner_confirmed,
            policy_owned=policy_owned,
            policy_prelude=policy_prelude,
        )
        if isinstance(projection, ProjectionRefusal):
            return projection.state_update
        arguments = projection.arguments

        notes_refusal = self._working_notes_guard.check(
            state,
            name=name,
            arguments=arguments,
            runtime_owned=runtime_owned,
            owner_confirmed=owner_confirmed,
            policy_owned=policy_owned,
            policy_prelude=policy_prelude,
        )
        if notes_refusal is not None:
            return notes_refusal

        if verdict == "review":
            approval = self._approval.seal(
                state,
                name=name,
                arguments=arguments,
            )
            if isinstance(approval, AuthorizationRefusal):
                return self._refuse(state, name=name, refusal=approval)
            return approval

        return {
            "authorization_route": "act",
            "control": control,
            "pending": {
                "name": name,
                "kind": capability.kind,
                "runtime_owned": runtime_owned,
                "owner_confirmed": owner_confirmed,
                "policy_owned": policy_owned,
                "policy_prelude": policy_prelude,
                "effect": capability.effect,
                "arguments": arguments,
            },
        }

    def budget_bucket(
        self, control: LoopControl, *, name: str, effect: ToolEffect
    ) -> tuple[str, int, int]:
        """Compatibility surface for callers that inspect budget classes."""

        return self._budget.bucket(control, name=name, effect=effect)
