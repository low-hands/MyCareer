from __future__ import annotations

from collections.abc import Callable

import hashlib
import json

from career_agent.agent.contracts.observations import ToolObservation
from career_agent.agent.capabilities.registry import MainAgentToolOutput
from career_agent.agent.runtime.state import MainAgentState, PendingAction
from career_agent.agent.capabilities.effects import replay_safe
from career_agent.agent.runtime.turn_coordinator import ACTION_INVOCATION
from career_agent.storage.action_executions import (
    ActionExecutionAlreadyFailedError,
    ActionExecutionReconciliationRequiredError,
    RESULT_STATE_RECEIPT_KEY,
    SQLiteActionExecutionStore,
)


_MAX_RECEIPT_KEYS = 20
_MAX_RECEIPT_VALUE_CHARS = 500


class ActionLedger:
    """Prepare, replay, and settle request-anchored durable writes."""

    def __init__(
        self,
        *,
        store: SQLiteActionExecutionStore | None,
        policy_epoch: int,
    ) -> None:
        self._store = store
        self._policy_epoch = policy_epoch

    @property
    def enabled(self) -> bool:
        return self._store is not None

    def execute(
        self,
        state: MainAgentState,
        invoke: Callable[[PendingAction], MainAgentToolOutput],
    ) -> MainAgentToolOutput:
        invocation = ACTION_INVOCATION.get()
        if invocation is None or self._store is None:
            raise RuntimeError("request-anchored action context is unavailable")
        turn_id, request_id = invocation
        context = state["context"]
        pending = state["pending"]
        name = pending["name"]
        arguments = pending["arguments"]
        anchor = request_id or turn_id
        fingerprint = hashlib.sha256(
            json.dumps(
                {"tool": name, "arguments": arguments},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode()
        ).hexdigest()
        try:
            execution, created = self._store.prepare(
                user_id=context.profile.user_id,
                conversation_id=context.conversation_id,
                anchor=anchor,
                request_id=request_id,
                write_slot=state.get("control", {}).get("write_calls", 0),
                tool_name=name,
                fingerprint=fingerprint,
                policy_epoch=self._policy_epoch,
                replay_allowed=replay_safe(name),
            )
        except ActionExecutionReconciliationRequiredError:
            return ToolObservation(
                tool_name=name,
                state="action_reconciliation_required",
                message=(
                    "上一次同类操作还没有确认结果，可能已经写入，也可能没有。"
                    "在核对清楚之前不能再执行一次，否则可能重复。"
                ),
                next_action="告诉用户有一次未确认的操作需要先核对，不要重试这次调用。",
                execution_outcome="unknown",
            )
        if not created:
            if execution.status == "SUCCEEDED":
                receipt = dict(execution.output)
                replayed_state = str(receipt.pop(RESULT_STATE_RECEIPT_KEY, "") or "")
                state["pending"]["reducer_result"] = ToolObservation(
                    tool_name=name,
                    state=replayed_state or "failed",
                    message="持久执行回执用于修复任务状态。",
                    payload=receipt,
                )
                return ToolObservation(
                    tool_name=name,
                    state="action_execution_replayed",
                    message="这一步此前已经完成，没有再次执行。",
                    next_action="按回执中的引用或标识读取持久结果，不要重做写操作。",
                    execution_outcome="committed",
                )
            if execution.status == "FAILED":
                raise ActionExecutionAlreadyFailedError(
                    execution.error_detail
                    or "this action already ended unsuccessfully; use a new request id"
                )
            if not replay_safe(name):
                return ToolObservation(
                    tool_name=name,
                    state="action_reconciliation_required",
                    message=(
                        "上一次这个操作没有确认结果，可能已经生效，也可能没有。"
                        "这个操作重复执行无法撤销，所以在核对清楚之前不能再执行一次。"
                    ),
                    next_action="告诉用户有一次未确认的操作需要先核对，不要重试这次调用。",
                    execution_outcome="unknown",
                )

        result = invoke(pending)
        if result.execution_outcome is None:
            raise ValueError(
                f"WRITE capability {name!r} returned without execution_outcome"
            )
        if result.execution_outcome == "unknown":
            return result
        if result.execution_outcome == "not_committed":
            self._store.fail(
                action_id=execution.action_id,
                error_code=result.state.upper(),
                error_detail=result.message,
            )
            return result
        self._store.succeed(
            action_id=execution.action_id,
            output=self.execution_receipt(result),
        )
        return result

    @staticmethod
    def execution_receipt(
        result: MainAgentToolOutput,
    ) -> dict[str, str | int | float | bool | None]:
        """Keep only scalar identifiers needed to repair reducer state."""

        receipt: dict[str, str | int | float | bool | None] = {
            RESULT_STATE_RECEIPT_KEY: result.state
        }
        for key, value in result.payload.items():
            if len(receipt) > _MAX_RECEIPT_KEYS:
                break
            if value is not None and not isinstance(value, (str, int, float, bool)):
                continue
            if isinstance(value, str) and len(value) > _MAX_RECEIPT_VALUE_CHARS:
                continue
            receipt[key] = value
        return receipt

