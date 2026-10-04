"""Canonical storage API for durable capability operations.

The physical table and compatibility names remain in ``action_executions`` so
existing installations upgrade in place instead of copying execution history.
"""

from career_agent.storage.action_executions import (
    ActionExecution,
    ActionExecutionAlreadyFailedError,
    ActionExecutionConflictError,
    ActionExecutionReconciliationRequiredError,
    ActionExecutionStatus,
    OperationPhase,
    OperationRecord,
    RESULT_STATE_RECEIPT_KEY,
    SQLiteActionExecutionStore,
    SQLiteOperationJournal,
)

__all__ = [
    "ActionExecution",
    "ActionExecutionAlreadyFailedError",
    "ActionExecutionConflictError",
    "ActionExecutionReconciliationRequiredError",
    "ActionExecutionStatus",
    "OperationPhase",
    "OperationRecord",
    "RESULT_STATE_RECEIPT_KEY",
    "SQLiteActionExecutionStore",
    "SQLiteOperationJournal",
]
