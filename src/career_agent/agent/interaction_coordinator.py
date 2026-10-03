from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import json

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import (
    ConversationTaskState,
)
from career_agent.agent.main_agent_tools import MainAgentToolOutput, MainAgentToolRegistry
from career_agent.agent.main_state import MainAgentState
from career_agent.agent.questionnaire_contracts import PendingQuestionnaire
from career_agent.agent.turn_coordinator import ACTION_INVOCATION
from career_agent.harness.streaming import (
    interaction_id,
)
from career_agent.storage.capability_confirmations import (
    CapabilityConfirmationExpiredError,
    CapabilityConfirmationInProgressError,
    CapabilityConfirmationSettledError,
    SQLiteCapabilityConfirmationStore,
    arguments_hash,
)


_QUESTIONNAIRE_FAILURE_MESSAGES = {
    "questionnaire_not_pending": "这份问卷已提交或不再有效，请刷新会话后重新发起当前任务。",
    "questionnaire_expired": "这份问卷已过期，请重新发起当前任务。",
    "questionnaire_invalid_answers": "问卷答案与当前题目不一致，请刷新后重新填写。",
    "questionnaire_resource_binding_changed": "当前任务绑定的简历或岗位已变化，请重新发起当前任务。",
    "questionnaire_resume_unavailable": "问卷绑定的简历版本已不可用，请重新选择简历后发起任务。",
    "questionnaire_job_unavailable": "问卷绑定的岗位已不可用，请重新选择岗位后发起任务。",
    "questionnaire_jd_unavailable": "问卷绑定的岗位描述版本已不可用，请重新选择岗位后发起任务。",
}


class QuestionnaireContinuationError(ValueError):
    """A safe, deterministic failure before questionnaire continuation begins."""

    def __init__(self, reason: str) -> None:
        if reason not in _QUESTIONNAIRE_FAILURE_MESSAGES:
            raise ValueError("unknown questionnaire failure reason")
        super().__init__(reason)
        self.code = reason.upper()
        self.user_message = _QUESTIONNAIRE_FAILURE_MESSAGES[reason]


@dataclass(frozen=True)
class ConfirmationResolution:
    graph_state: MainAgentState | None = None
    result: MainAgentToolOutput | None = None
    action: str = "confirm"


class InteractionCoordinator:
    """Validate questionnaire continuations and durable confirmations."""

    def __init__(
        self,
        *,
        context_manager: ContextManager,
        tools: MainAgentToolRegistry,
        confirmation_store: SQLiteCapabilityConfirmationStore | None,
    ) -> None:
        self._context_manager = context_manager
        self._tools = tools
        self._confirmation_store = confirmation_store

    def prepare_questionnaire_continuation(
        self,
        *,
        user_id: str,
        conversation_id: str,
        response: Any,
        task: ConversationTaskState,
    ) -> Any:
        pending = task.pending_questionnaire
        if pending is None or pending.interaction_id != response.interaction_id:
            raise QuestionnaireContinuationError("questionnaire_not_pending")
        if datetime.now(timezone.utc) >= pending.expires_at:
            raise QuestionnaireContinuationError("questionnaire_expired")
        try:
            pending.validate_answers(response.answers)
        except ValueError as error:
            raise QuestionnaireContinuationError(
                "questionnaire_invalid_answers"
            ) from error
        if (
            task.active_workflow != pending.active_workflow
            or task.active_resume_version_id != pending.resume_version_id
            or task.active_job_posting_id != pending.job_posting_id
            or task.active_jd_snapshot_id != pending.jd_snapshot_id
        ):
            raise QuestionnaireContinuationError(
                "questionnaire_resource_binding_changed"
            )
        self._validate_questionnaire_resources(user_id=user_id, pending=pending)

        answers = []
        visible_answers = []
        for question, answer in zip(pending.questions, response.answers):
            selected = {option.value: option for option in question.options}
            labels = [selected[value].label for value in answer.selected_values]
            answers.append(
                {
                    "question": question.prompt,
                    "answer": (
                        "跳过"
                        if answer.skipped
                        else {
                            "selected": labels,
                            "free_text": answer.free_text,
                        }
                    ),
                }
            )
            visible_answer = "跳过" if answer.skipped else "、".join(labels)
            if answer.free_text:
                visible_answer = (
                    f"{visible_answer}；{answer.free_text}"
                    if visible_answer
                    else answer.free_text
                )
            visible_answers.append(f"{question.prompt}：{visible_answer}")
        message = (
            "以下是用户一次提交的当前任务问卷答案。仅用于当前绑定任务；跳过或选择“无”"
            "不构成永久职业事实；泛泛的技能回答不能扩写成项目、年限或成果。"
            "请先完成原任务。完成后，若某条非跳过答案是跨岗位稳定事实，且能明确归入"
            "现有职业记录，可以调用 propose_career_fact 展示一条 origin=user_input 的"
            "隔离态提案；当前岗位的强调偏好不得持久化，归属不明确时先询问归属，"
            "不得猜测或直接写入。提案不得替代或打断原任务。\n"
            + json.dumps(answers, ensure_ascii=False)
        )
        context = self._context_manager.load_for_turn(
            user_id=user_id,
            conversation_id=conversation_id,
            user_message=message,
        )
        return context.model_copy(
            update={
                "task": context.task.model_copy(
                    update={"pending_questionnaire": None}
                ),
                "user_message_source": "已提交当前任务问卷：\n"
                + "\n".join(visible_answers),
                "user_interaction_id": pending.interaction_id,
            }
        )

    def resolve_confirmation(
        self,
        *,
        context: Any,
        conversation_id: str,
        response: Any,
        invoke_confirmed: Callable[[Any], MainAgentState],
    ) -> ConfirmationResolution:
        if response.scope != "capability_confirmation":
            raise ValueError(
                f"no capability owns interaction scope {response.scope!r}"
            )
        store = self._confirmation_store
        user_id = context.profile.user_id
        pending = (
            store.active_for_conversation(
                user_id=user_id,
                conversation_id=conversation_id,
            )
            if store is not None
            else ()
        )
        confirmation = next(
            (
                candidate
                for candidate in pending
                if interaction_id(
                    conversation_id,
                    "capability_confirmation",
                    candidate.confirmation_id,
                )
                == response.interaction_id
            ),
            None,
        )
        if confirmation is None:
            return self._settled(
                "这项确认已过期或已处理，请重新提出这个操作。",
                state="capability_confirmation_expired",
            )
        if response.action == "cancel":
            cancelled = store.cancel(
                confirmation_id=confirmation.confirmation_id,
                user_id=user_id,
            )
            if not cancelled:
                return self._settled(
                    "这项操作已经开始执行，取消没有改写它的状态；请等待结果或进行对账。",
                    state="capability_confirmation_in_progress",
                    action="cancel",
                )
            return self._settled(
                "已按你的选择取消，没有执行这个操作。",
                state="capability_confirmation_cancelled",
                action="cancel",
            )
        if (
            confirmation.status == "PENDING"
            and confirmation.policy_revision
            != context.preferences.behavior_policy.revision
        ):
            store.cancel(
                confirmation_id=confirmation.confirmation_id,
                user_id=user_id,
            )
            return self._settled(
                "你的行为规则在这项确认发出后已经变化；旧批准未执行，请重新提出操作。",
                state="capability_confirmation_expired",
            )
        try:
            sealed = store.claim(
                confirmation_id=confirmation.confirmation_id,
                user_id=user_id,
            )
        except CapabilityConfirmationExpiredError:
            return self._settled(
                "这项确认已过期，没有执行。请重新提出这个操作。",
                state="capability_confirmation_expired",
            )
        except CapabilityConfirmationSettledError:
            return self._settled(
                "这项确认已经处理过，没有重复执行。",
                state="capability_confirmation_expired",
            )
        except CapabilityConfirmationInProgressError:
            return self._settled(
                "这项操作正在执行，没有重复启动。",
                state="capability_confirmation_in_progress",
            )
        if arguments_hash(sealed.arguments) != sealed.arguments_hash:
            return self._settled(
                "这项确认的内容已经无法核对，没有执行。",
                state="capability_confirmation_expired",
            )

        invocation = ACTION_INVOCATION.get()
        if invocation is None:
            raise RuntimeError("confirmation execution context is unavailable")
        action_token = ACTION_INVOCATION.set(
            (invocation[0], f"confirmation:{sealed.confirmation_id}")
        )
        try:
            graph_state = invoke_confirmed(sealed)
        except Exception:
            store.settle(
                confirmation_id=sealed.confirmation_id,
                user_id=user_id,
                status="RECONCILIATION_REQUIRED",
            )
            raise
        finally:
            ACTION_INVOCATION.reset(action_token)
        results = graph_state.get("tool_results", ())
        last = results[-1] if results else None
        if last is None:
            raise RuntimeError("confirmed action produced no result")
        confirmation_status = (
            "RECONCILIATION_REQUIRED"
            if last.state == "action_reconciliation_required"
            or last.execution_outcome == "unknown"
            else "FAILED"
            if last.execution_outcome == "not_committed"
            else "EXECUTED"
        )
        store.settle(
            confirmation_id=sealed.confirmation_id,
            user_id=user_id,
            status=confirmation_status,
        )
        return ConfirmationResolution(graph_state=graph_state, action=response.action)

    def _validate_questionnaire_resources(
        self, *, user_id: str, pending: PendingQuestionnaire
    ) -> None:
        if pending.resume_version_id is not None:
            store = self._tools.resume_store
            if store is None or store.read_version_document(
                user_id=user_id,
                resume_version_id=pending.resume_version_id,
            ) is None:
                raise QuestionnaireContinuationError(
                    "questionnaire_resume_unavailable"
                )
        if pending.job_posting_id is not None:
            jobs = self._tools.job_repository
            if jobs is None or jobs.get_job(
                user_id=user_id,
                job_posting_id=pending.job_posting_id,
            ) is None:
                raise QuestionnaireContinuationError("questionnaire_job_unavailable")
        if pending.jd_snapshot_id is not None:
            jobs = self._tools.job_repository
            if jobs is None or jobs.get_snapshot(
                user_id=user_id,
                jd_snapshot_id=pending.jd_snapshot_id,
            ) is None:
                raise QuestionnaireContinuationError("questionnaire_jd_unavailable")

    @staticmethod
    def _settled(
        message: str,
        *,
        state: str,
        action: str = "confirm",
    ) -> ConfirmationResolution:
        return ConfirmationResolution(
            result=MainAgentToolOutput(
                tool_name="capability_confirmation",
                state=state,
                message=message,
            ),
            action=action,
        )
