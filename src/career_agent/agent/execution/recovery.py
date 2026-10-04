from __future__ import annotations

from collections.abc import Callable, Mapping

from career_agent.agent.capabilities.catalog import capability
from career_agent.agent.capabilities.registry import MainAgentToolOutput
from career_agent.agent.execution.operation_journal import OperationJournal
from career_agent.storage.operation_journal import (
    ActionExecution,
    SQLiteActionExecutionStore,
)


OperationReconciler = Callable[[ActionExecution], MainAgentToolOutput]


class OperationReconcilerRegistry:
    """Capability-keyed recovery handlers for effects with an uncertain outcome."""

    def __init__(self, handlers: Mapping[str, OperationReconciler] | None = None) -> None:
        self._handlers = dict(handlers or {})
        for name in self._handlers:
            descriptor = capability(name)
            if descriptor.effect != "WRITE" or descriptor.recovery_policy != "reconcile":
                raise ValueError(f"capability {name!r} is not reconcilable")

    def recover(
        self,
        *,
        store: SQLiteActionExecutionStore,
        user_id: str,
    ) -> tuple[ActionExecution, ...]:
        recovered: list[ActionExecution] = []
        for operation in store.list_incomplete(user_id=user_id):
            if (
                operation.status != "PENDING"
                or operation.phase not in {"RUNNING", "RECONCILIATION_REQUIRED"}
                or operation.recovery_policy != "reconcile"
            ):
                continue
            handler = self._handlers.get(operation.capability)
            if handler is None:
                continue
            try:
                result = handler(operation)
            except Exception:
                # Recovery is opportunistic at turn ingress. The durable row is
                # intentionally left visible for the next retry or an operator.
                continue
            if result.execution_outcome == "committed":
                receipt = OperationJournal.execution_receipt(result)
                references = OperationJournal.references(receipt)
                recovered.append(
                    store.succeed(
                        action_id=operation.operation_id,
                        output=receipt,
                        output_references=references,
                        external_reference=OperationJournal.external_reference(
                            references
                        ),
                    )
                )
            elif result.execution_outcome == "not_committed":
                recovered.append(
                    store.fail(
                        action_id=operation.operation_id,
                        error_code=result.state.upper(),
                        error_detail=result.message,
                    )
                )
        return tuple(recovered)
