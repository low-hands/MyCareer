from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from career_agent.agent.execution.action_ledger import ActionLedger
from career_agent.agent.main_agent_tools import MainAgentToolOutput, MainAgentToolRegistry
from career_agent.agent.main_state import MainAgentState, PendingAction
from career_agent.storage.action_executions import SQLiteActionExecutionStore


class CapabilityExecutionHost(Protocol):
    """Presentation hooks the executor needs from the graph runtime."""

    def _emit_capability_started(self, name: str) -> None: ...

    def _emit_capability_completed(self, name: str, state: str) -> None: ...

    def _run_capability(
        self,
        pending: PendingAction,
        run: Callable[[], MainAgentToolOutput],
    ) -> MainAgentToolOutput: ...


class CapabilityExecutor:
    """Dispatch one authorized capability through progress and ledger hooks."""

    def __init__(
        self,
        *,
        host: CapabilityExecutionHost,
        tools: MainAgentToolRegistry,
        action_execution_store: SQLiteActionExecutionStore | None,
        action_policy_epoch: int,
    ) -> None:
        self._host = host
        self._tools = tools
        self._action_ledger = ActionLedger(
            store=action_execution_store,
            policy_epoch=action_policy_epoch,
        )

    def act(self, state: MainAgentState) -> MainAgentState:
        pending = state["pending"]
        name = pending["name"]
        self._host._emit_capability_started(name)
        if pending.get("effect") == "WRITE" and self._action_ledger.enabled:
            result = self._host._run_capability(
                pending,
                lambda: self._act_request_anchored_write(state),
            )
        else:
            result = self._host._run_capability(
                pending,
                lambda: self._invoke_pending(pending),
            )
        if pending.get("effect") == "WRITE" and result.execution_outcome is None:
            raise ValueError(
                f"WRITE capability {name!r} returned without execution_outcome"
            )
        self._host._emit_capability_completed(name, result.state)
        return {"pending": {**pending, "result": result}}

    def _invoke_pending(self, pending: PendingAction) -> MainAgentToolOutput:
        name = pending["name"]
        arguments = pending["arguments"]
        if pending.get("runtime_owned"):
            return self._tools.invoke_runtime_workflow(name, arguments)
        if pending["kind"] == "atomic_tool":
            return self._tools.invoke_atomic_tool(name, arguments)
        return self._tools.invoke_workflow(name, arguments)

    def _act_request_anchored_write(
        self, state: MainAgentState
    ) -> MainAgentToolOutput:
        return self._action_ledger.execute(state, self._invoke_pending)

    execution_receipt = staticmethod(ActionLedger.execution_receipt)
