from __future__ import annotations

from typing import Any

from pydantic_core import to_jsonable_python

from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.contracts.interactions import (
    CONFIRMATION_SPECS,
    confirmation_arguments_snapshot,
)
from career_agent.agent.contracts.observations import ToolObservation
from career_agent.agent.capabilities.registry import MainAgentToolRegistry
from career_agent.agent.runtime.state import MainAgentState
from career_agent.agent.middleware.contracts import AuthorizationRefusal
from career_agent.agent.capabilities.catalog import capability
from career_agent.storage.capability_confirmations import (
    SQLiteCapabilityConfirmationStore,
)


class ApprovalMiddleware:
    """Seal owner-review actions and render their confirmation contract."""

    def __init__(
        self,
        *,
        tools: MainAgentToolRegistry,
        confirmation_store: SQLiteCapabilityConfirmationStore | None,
    ) -> None:
        self._tools = tools
        self._confirmation_store = confirmation_store

    def seal(
        self, state: MainAgentState, *, name: str, arguments: dict[str, Any]
    ) -> MainAgentState | AuthorizationRefusal:
        descriptor = capability(name)
        external = descriptor.external_write
        if external:
            rule = "这个操作会写入外部系统，写入后无法由这里撤回，因此必须由你亲自确认"
            prompt = "这是一次外部写入，执行后无法由这里撤回。是否执行？"
        elif descriptor.destructive:
            rule = "这是无法撤销的破坏性操作，系统要求必须由你亲自确认"
            prompt = "这是无法撤销的破坏性操作，系统强制要求确认。是否执行？"
        else:
            rule = "你设置了这个操作需要先经你确认"
            prompt = "你设置了此操作需要确认。是否执行？"
        if self._confirmation_store is None:
            return AuthorizationRefusal(
                kind="seal_unavailable",
                reason=f"{rule}，但本次部署无法保存待确认动作。",
                next_action="告诉用户这个操作需要确认，但当前无法记录确认请求。",
            )
        context = state["context"]
        display_summary = (
            self._external_write_summary(name=name, arguments=arguments)
            if external
            else self._owner_confirmation_summary(
                context=context, name=name, arguments=arguments
            )
        )
        sealed_arguments = (
            confirmation_arguments_snapshot(
                context.task,
                name,
                user_id=context.profile.user_id,
                conversation_id=context.conversation_id,
            )
            if name in CONFIRMATION_SPECS
            else to_jsonable_python(arguments)
        )
        if name == "confirm_free_text_preference":
            sealed_arguments.update(
                {
                    key: value
                    for key, value in arguments.items()
                    if key in {"scope_choice", "scope_domain", "job_posting_id"}
                }
            )
        confirmation = self._confirmation_store.seal(
            user_id=context.profile.user_id,
            conversation_id=context.conversation_id,
            capability=name,
            display_summary=display_summary,
            arguments=sealed_arguments,
            policy_revision=context.preferences.behavior_policy.revision,
        )
        if confirmation.status == "APPLYING":
            return {
                "authorization_route": "observe",
                "pending": {
                    "name": name,
                    "result": ToolObservation(
                        tool_name=name,
                        state="capability_confirmation_in_progress",
                        message=(
                            "同一项已批准操作正在执行，没有再次发起确认或执行。"
                        ),
                        next_action="告诉用户操作仍在处理中，不要重试。",
                    ),
                    "synthetic_kind": "confirmation",
                    "runtime_owned": False,
                },
            }
        return {
            "authorization_route": "observe",
            "pending": {
                "name": name,
                "result": ToolObservation(
                    tool_name=name,
                    state="capability_confirmation_required",
                    message=(
                        f"{display_summary}\n"
                        + prompt
                    ),
                    next_action=(
                        "向用户说明将要执行什么并等待确认；本轮不要重试这个操作。"
                    ),
                    payload={"confirmation_id": confirmation.confirmation_id},
                ),
                "synthetic_kind": "confirmation",
                "runtime_owned": False,
                "confirmation_id": confirmation.confirmation_id,
            },
        }

    @staticmethod
    def _owner_confirmation_summary(
        *, context: MainAgentContext, name: str, arguments: dict[str, Any]
    ) -> str:
        if name == "create_application":
            job = next(
                (
                    item
                    for item in context.task.saved_job_candidates
                    if item.job_posting_id == arguments.get("job_posting_id")
                ),
                None,
            )
            resume = next(
                (
                    item
                    for item in context.task.resume_version_candidates
                    if item.resume_version_id == arguments.get("resume_version_id")
                ),
                None,
            )
            target = (
                f"{job.company_name} · {job.title}"
                if job is not None
                else "当前选中的岗位"
            )
            version = f"，使用简历版本 v{resume.version_number}" if resume else ""
            submitted = arguments.get("submitted_at")
            when = f"，投递时间 {submitted}" if submitted is not None else ""
            return f"准备创建投递记录：{target}{version}{when}。"
        if name == "resolve_email_event":
            event = next((item for item in context.task.email_event_candidates
                          if item.email_event_id == arguments.get("event_id")), None)
            target = event.summary if event is not None else "当前选中的邮件事件"
            action = "应用到投递／面试记录" if arguments.get("approve") else "忽略"
            return f"准备{action}：{target}。处理后不能恢复为待确认。"
        if name == "restart_mock_interview":
            return "准备取消当前卡住的模拟面试、删除旧检查点并新建替代练习；旧会话将无法恢复。"
        if name == "update_owner_settings":
            changes = []
            if arguments.get("boss_search") is not None:
                changes.append(f"岗位搜索偏好 → {arguments['boss_search']}")
            if arguments.get("application_confirmation") is not None:
                changes.append(
                    "投递记录确认规则 → "
                    f"{arguments['application_confirmation']}"
                )
            if arguments.get("confirm_before") is not None:
                listed = "、".join(arguments["confirm_before"]) or "（清空）"
                changes.append(f"执行前需逐项确认的操作 → {listed}")
            return "准备更新持久设置：" + "；".join(changes) + "。"
        if name == "confirm_memory_tombstone":
            proposal = arguments.get("proposal", {})
            if hasattr(proposal, "model_dump"):
                proposal = proposal.model_dump(mode="json")
            return (
                "准备永久删除刚才提案的职业声明。"
                f"原因：{str(proposal.get('reason', '用户要求删除'))[:200]}"
            )
        if name == "confirm_constraint_retirement":
            proposal = arguments.get("proposal", {})
            if hasattr(proposal, "model_dump"):
                proposal = proposal.model_dump(mode="json")
            return f"准备停用对话约束：「{proposal.get('constraint', '当前提案')}」。"
        return f"准备执行 {name}。"

    def _external_write_summary(
        self, *, name: str, arguments: dict[str, Any]
    ) -> str:
        if name == "execute_calendar_proposal":
            try:
                proposal = self._tools.invoke_atomic_tool(
                    "get_calendar_proposal", dict(arguments)
                )
            except Exception:  # noqa: BLE001 - rendering must not block the gate
                proposal = None
            if proposal is not None and proposal.state == "calendar_proposal_ready":
                operation = proposal.payload.get("operation")
                event = proposal.payload.get("payload")
                expires_at = proposal.payload.get("expires_at")
                if isinstance(event, dict):
                    return (
                        f"准备写入外部 Calendar（{operation}）："
                        f"{event.get('title')}，"
                        f"{event.get('start_at')} → {event.get('end_at')}"
                        f"（{event.get('timezone')}），"
                        f"地点 {event.get('location') or '未提供'}；"
                        f"预览有效期至 {expires_at}。"
                    )
                return (
                    f"准备在外部 Calendar 上执行 {operation}；"
                    f"预览有效期至 {expires_at}。"
                )
            return "准备执行已预览的 Calendar 变更。"
        return f"准备向外部系统写入：{name}。"
