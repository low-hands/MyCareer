from __future__ import annotations

from threading import Lock
from typing import Any

from career_agent.agent.context.manager import ContextManager
from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.runtime.turn_coordinator import ACTION_INVOCATION
from career_agent.services.episode_reconciliation import EpisodeReconciler
from career_agent.storage.action_executions import SQLiteActionExecutionStore


class ReconciliationCoordinator:
    """Repair durable seams without putting recovery state in graph state."""

    def __init__(
        self,
        *,
        context_manager: ContextManager | None,
        action_execution_store: SQLiteActionExecutionStore | None,
        episode_reconciler: EpisodeReconciler | None,
        reconciled_users: set[str] | None = None,
        reconcile_guard: Any | None = None,
        user_locks: dict[str, Any] | None = None,
    ) -> None:
        self._context_manager = context_manager
        self._action_execution_store = action_execution_store
        self._episode_reconciler = episode_reconciler
        self._reconciled_users = (
            set() if reconciled_users is None else reconciled_users
        )
        self._reconcile_guard = reconcile_guard or Lock()
        self._user_locks = {} if user_locks is None else user_locks

    def commit_interrupted_turn(
        self,
        *,
        context: MainAgentContext,
        error: Exception,
    ) -> None:
        """Leave a receipt when a failed turn may already have written."""

        invocation = ACTION_INVOCATION.get()
        if (
            invocation is None
            or self._action_execution_store is None
            or self._context_manager is None
        ):
            return
        turn_id, request_id = invocation
        executions = self._action_execution_store.list_for_anchor(
            user_id=context.profile.user_id,
            conversation_id=context.conversation_id,
            anchor=request_id or turn_id,
        )
        settled = [item.tool_name for item in executions if item.status == "SUCCEEDED"]
        unsettled = [item.tool_name for item in executions if item.status == "PENDING"]
        if not settled and not unsettled:
            return
        sentences: list[str] = []
        if settled:
            sentences.append(
                "以下操作已经写入：" + "、".join(dict.fromkeys(settled)) + "。"
            )
        if unsettled:
            sentences.append(
                "以下操作已经开始但没有确认结果，可能已生效也可能没有："
                + "、".join(dict.fromkeys(unsettled))
                + "。"
            )
        try:
            self._context_manager.commit_turn(
                context=context,
                task=context.task,
                assistant_message=(
                    f"本轮执行中断（{type(error).__name__}）。"
                    + "".join(sentences)
                    + "请先核对这些记录的实际状态，再决定是否重做。"
                ),
                turn_id=turn_id,
            )
        except Exception:
            # This receipt is best effort and must not replace the original
            # execution failure with a secondary conversation-store failure.
            return

    def reconcile_episodes(self, user_id: str) -> None:
        """Replay durable domain state once per user and after a failed turn."""

        if self._episode_reconciler is None:
            return
        user_lock = self.user_lock(user_id)
        with user_lock:
            with self._reconcile_guard:
                if user_id in self._reconciled_users:
                    return
            self._episode_reconciler.reconcile_user(user_id=user_id)
            with self._reconcile_guard:
                self._reconciled_users.add(user_id)

    def invalidate_episode_reconciliation(self, user_id: str) -> None:
        if self._episode_reconciler is None:
            return
        user_lock = self.user_lock(user_id)
        with user_lock:
            with self._reconcile_guard:
                self._reconciled_users.discard(user_id)

    def user_lock(self, user_id: str):
        with self._reconcile_guard:
            return self._user_locks.setdefault(user_id, Lock())
