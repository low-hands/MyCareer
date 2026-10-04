from __future__ import annotations

from collections.abc import Callable

from career_agent.agent.execution.operation_journal import OperationJournal
from career_agent.agent.capabilities.registry import MainAgentToolOutput, MainAgentToolRegistry
from career_agent.agent.runtime.state import MainAgentState, PendingAction
from career_agent.storage.operation_journal import SQLiteActionExecutionStore


class CapabilityExecutor:
    """Dispatch one authorized capability through progress and ledger hooks."""

    def __init__(
        self,
        *,
        tools: MainAgentToolRegistry,
        action_execution_store: SQLiteActionExecutionStore | None,
        action_policy_epoch: int,
        emit_capability_started: Callable[[str], None],
        emit_capability_completed: Callable[[str, str], None],
        run_capability: Callable[
            [PendingAction, Callable[[], MainAgentToolOutput]],
            MainAgentToolOutput,
        ],
    ) -> None:
        self._tools = tools
        self._emit_capability_started = emit_capability_started
        self._emit_capability_completed = emit_capability_completed
        self._run_capability = run_capability
        self._operation_journal = OperationJournal(
            store=action_execution_store,
            policy_epoch=action_policy_epoch,
        )

    def act(self, state: MainAgentState) -> MainAgentState:
        pending = state["pending"]
        name = pending["name"]
        self._emit_capability_started(name)
        if pending.get("effect") == "WRITE" and self._operation_journal.enabled:
            result = self._run_capability(
                pending,
                lambda: self._act_request_anchored_write(state),
            )
        else:
            result = self._run_capability(
                pending,
                lambda: self._invoke_pending(pending),
            )
        if pending.get("effect") == "WRITE" and result.execution_outcome is None:
            raise ValueError(
                f"WRITE capability {name!r} returned without execution_outcome"
            )
        self._emit_capability_completed(name, result.state)
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
        return self._operation_journal.execute(state, self._invoke_pending)

    execution_receipt = staticmethod(OperationJournal.execution_receipt)
