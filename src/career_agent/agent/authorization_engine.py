from __future__ import annotations

from typing import Any, Literal, Protocol

import json

from pydantic_core import to_jsonable_python

from career_agent.agent.main_agent_contracts import (
    CONFIRMATION_SPECS,
    DOMAIN_TOOL_PROFILES,
    AgentDecision,
    MainAgentContext,
    ToolObservation,
    confirmation_arguments_snapshot,
)
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.agent.tool_effects import (
    ToolEffect,
    effect_for,
    is_external_write,
    is_notes_guarded,
    is_preference_bound,
)
from career_agent.agent.working_notes_guard import (
    remembered_preference_without_authority,
    working_notes_only_tokens,
)
from career_agent.storage.capability_confirmations import (
    SQLiteCapabilityConfirmationStore,
)


AuthorizationRefusalKind = Literal[
    "out_of_profile",
    "preference_deny",
    "budget_exhausted",
    "duplicate_call",
    "retry_limit",
    "seal_unavailable",
]


class AuthorizationHost(Protocol):
    """Runtime services needed by authorization, without owning the runtime."""

    def _offers_tool(self, name: str, profile: str, task: Any) -> bool: ...

    def _project_runtime_workflow_arguments(
        self, state: dict[str, Any], name: str
    ) -> dict[str, Any]: ...

    def _project_atomic_tool_arguments(
        self, context: MainAgentContext, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]: ...

    def _project_workflow_arguments(
        self, context: MainAgentContext, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]: ...

    def _record_trace_event(
        self,
        event_type: str,
        stage: str,
        *,
        outcome: str,
        details: dict[str, Any] | None = None,
        recoverable: bool | None = None,
    ) -> None: ...

    def _reraise_security_refusal(self, error: ValueError) -> None: ...

    def _rejection_observation(
        self, name: str, error: ValueError
    ) -> ToolObservation: ...


class AuthorizationEngine:
    """Projects and authorizes one proposed action.

    The engine owns policy decisions, refusal accounting, confirmation seals,
    and per-turn delegation budgets. It deliberately does not choose graph
    successors or execute tools; those remain LangGraph/runtime concerns.
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
        self._host = host
        self._tools = tools
        self._confirmation_store = confirmation_store
        self._max_read_calls = max_read_calls
        self._max_write_calls = max_write_calls
        self._max_external_write_calls = max_external_write_calls
        self._max_projection_refusals = max_projection_refusals
        self._max_authorization_refusals = max_authorization_refusals
        self._max_failure_retries = max_failure_retries

    @staticmethod
    def _control(state: dict[str, Any]) -> dict[str, Any]:
        return state.get("control", {})

    @staticmethod
    def _tool_call_fingerprint(decision: AgentDecision) -> str:
        if decision.tool_call is None:
            return ""
        return json.dumps(
            {"name": decision.tool_call.name, "arguments": decision.tool_call.arguments},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _refuse(
        self,
        state: dict[str, Any],
        *,
        name: str,
        kind: AuthorizationRefusalKind,
        reason: str,
        next_action: str,
    ) -> dict[str, Any]:
        control = self._control(state)
        capped = (
            control.get("authorization_refusals", 0)
            >= self._max_authorization_refusals
        )
        self._host._record_trace_event(
            "authorization_refused",
            "authorize",
            outcome="failed",
            details={
                "tool_name": name,
                "refusal_kind": kind,
                "tool_profile": state["context"].task.tool_profile,
                "capped": capped,
            },
            recoverable=not capped,
        )
        if capped:
            return {"authorization_route": "present"}
        result = ToolObservation(
            tool_name=name,
            state="authorization_refused",
            message=reason,
            next_action=next_action,
        )
        return {
            "authorization_route": "observe",
            "pending": {
                "name": name,
                "result": result,
                "synthetic_kind": "authorization",
                "runtime_owned": bool(state.get("pending", {}).get("runtime_owned")),
            },
        }

    def authorize(self, state: dict[str, Any]) -> dict[str, Any]:
        decision = state["decision"]
        if decision.tool_call is None:
            raise ValueError("tool_call action requires tool_call arguments")
        name = decision.tool_call.name
        runtime_owned = bool(state.get("pending", {}).get("runtime_owned"))
        owner_confirmed = bool(state.get("pending", {}).get("owner_confirmed"))
        policy_owned = bool(state.get("pending", {}).get("policy_owned"))
        policy_prelude = bool(state.get("pending", {}).get("policy_prelude"))
        if runtime_owned:
            if name not in self._tools.runtime_workflow_names:
                raise ValueError(f"Unknown runtime-owned workflow: {name}")
            kind = "workflow"
        else:
            kind = self._tools.capability_kind(name)
        effect = effect_for(name)
        model_selected = not (runtime_owned or owner_confirmed or policy_owned)
        tool_profile = state["context"].task.tool_profile
        if model_selected and not self._host._offers_tool(
            name, tool_profile, state["context"].task
        ):
            return self._refuse(
                state,
                name=name,
                kind="out_of_profile",
                reason=f"{name} 不在当前 {tool_profile} 工具档内。",
                next_action=(
                    "先用 route_to_capability 切到该工具所属的领域，"
                    "再从 task.available_now 中选择工具。"
                ),
            )
        verdict = (
            "permit"
            if runtime_owned or owner_confirmed
            else state["context"].preferences.capability_verdict(name)
        )
        if verdict == "deny":
            return self._refuse(
                state,
                name=name,
                kind="preference_deny",
                reason="你设置的偏好不允许这个操作。",
                next_action="向用户说明这条设置，不要重试这次调用。",
            )
        control = self._control(state)
        bucket, used, limit = self.budget_bucket(control, name=name, effect=effect)
        if used >= limit:
            return self._refuse(
                state,
                name=name,
                kind="budget_exhausted",
                reason=(
                    f"本轮 {bucket} 委派预算已经用完；请基于已有结果作答，"
                    "或说明需要下一轮继续。"
                ),
                next_action=(
                    "本轮的委派预算已经用完。请基于已有结果作答，"
                    "或者告诉用户还缺什么。"
                ),
            )

        fingerprint = self._tool_call_fingerprint(decision)
        fingerprints = control.get("fingerprints", ())
        retry_counts = dict(control.get("retry_counts", {}))
        if fingerprint in fingerprints:
            retryable = fingerprint in control.get("retryable_fingerprints", ())
            retries = retry_counts.get(fingerprint, 0)
            if not retryable:
                return self._refuse(
                    state,
                    name=name,
                    kind="duplicate_call",
                    reason="相同调用已经执行过，且上次结果没有声明为可重试。",
                    next_action=(
                        "这次调用和本轮之前那次完全一样，再调一次也不会有新结果。"
                        "请用已有的观察作答，或者换一组参数。"
                    ),
                )
            if retries >= self._max_failure_retries:
                return self._refuse(
                    state,
                    name=name,
                    kind="retry_limit",
                    reason="相同失败调用已经达到本轮重试上限。",
                    next_action=(
                        "同一个失败调用已经重试到本轮上限。别再重试；"
                        "把失败讲清楚，或者问用户要不要换个做法。"
                    ),
                )
            retry_counts[fingerprint] = retries + 1
            control = {**control, "retry_counts": retry_counts}
        try:
            arguments = (
                state["pending"]["arguments"]
                if owner_confirmed
                else self._host._project_runtime_workflow_arguments(state, name)
                if runtime_owned
                else self._host._project_atomic_tool_arguments(
                    state["context"], name, decision.tool_call.arguments
                )
                if kind == "atomic_tool"
                else self._host._project_workflow_arguments(
                    state["context"], name, decision.tool_call.arguments
                )
            )
            if owner_confirmed and name == "update_owner_settings":
                arguments = {
                    **arguments,
                    "confirmation_id": state["pending"]["confirmation_id"],
                }
        except ValueError as error:
            self._host._reraise_security_refusal(error)
            if control.get("projection_refusals", 0) >= self._max_projection_refusals:
                return {"authorization_route": "present"}
            result = self._host._rejection_observation(name, error)
            return {
                "authorization_route": "observe",
                "pending": {
                    "name": name,
                    "result": result,
                    "synthetic_kind": "projection",
                    "runtime_owned": runtime_owned,
                    "policy_owned": policy_owned,
                    "policy_prelude": policy_prelude,
                },
            }

        note_only_tokens = (
            working_notes_only_tokens(arguments=arguments, context=state["context"])
            if is_notes_guarded(name) and not runtime_owned and not owner_confirmed
            else ()
        )
        notes_refusal: ToolObservation | None = None
        if note_only_tokens:
            visible_tokens = [token[:32] for token in note_only_tokens[:8]]
            notes_refusal = ToolObservation(
                tool_name=name,
                state="working_notes_derived_argument",
                message=(
                    "以下内容只出现在工作笔记、没有用户或权威记忆来源："
                    + "、".join(visible_tokens)
                    + "；请向用户确认或改用权威来源。"
                ),
                next_action=(
                    "不要换个说法重试这次调用；请向用户确认这些内容，"
                    "或改用用户消息、已确认记忆和工具结果中的权威来源。"
                ),
                payload={"tokens": visible_tokens, "tool_name": name},
                execution_outcome="not_committed",
            )
        elif (
            is_preference_bound(name)
            and not runtime_owned
            and not owner_confirmed
            and remembered_preference_without_authority(state["context"])
        ):
            notes_refusal = ToolObservation(
                tool_name=name,
                state="working_notes_derived_argument",
                message=(
                    "用户要求按“你记得的偏好”做选择，但当前没有任何已确认的偏好来源，"
                    "只有工作笔记里未确认的观察；据此比较或推荐会把猜测当作偏好。"
                ),
                next_action=(
                    "先把工作笔记里的观察原样说给用户、请用户确认或修正，"
                    "再根据确认后的偏好选择；不要先调用比较或推荐类工具。"
                ),
                payload={
                    "tokens": [],
                    "tool_name": name,
                    "referent": "remembered_preference",
                },
                execution_outcome="not_committed",
            )
        if notes_refusal is not None:
            if control.get("projection_refusals", 0) >= self._max_projection_refusals:
                return {"authorization_route": "present"}
            return {
                "authorization_route": "observe",
                "pending": {
                    "name": name,
                    "result": notes_refusal,
                    "synthetic_kind": "projection",
                    "runtime_owned": runtime_owned,
                    "policy_owned": policy_owned,
                    "policy_prelude": policy_prelude,
                },
            }
        if verdict == "review":
            return self._seal_for_owner_confirmation(
                state, name=name, arguments=arguments
            )
        return {
            "authorization_route": "act",
            "control": control,
            "pending": {
                "name": name,
                "kind": kind,
                "runtime_owned": runtime_owned,
                "owner_confirmed": owner_confirmed,
                "policy_owned": policy_owned,
                "policy_prelude": policy_prelude,
                "effect": effect,
                "arguments": arguments,
            },
        }

    def budget_bucket(
        self, control: dict[str, Any], *, name: str, effect: ToolEffect
    ) -> tuple[str, int, int]:
        if effect == "READ":
            return "READ", control.get("read_calls", 0), self._max_read_calls
        if effect == "CONTROL":
            return "CONTROL", control.get("control_calls", 0), len(DOMAIN_TOOL_PROFILES)
        external_used = control.get("external_write_calls", 0)
        if is_external_write(name):
            return "WRITE_EXTERNAL", external_used, self._max_external_write_calls
        limit = self._max_write_calls
        if name == "match_resume_to_job" and control.get(
            "job_analysis_write_used", False
        ):
            limit += 1
        return "WRITE", control.get("write_calls", 0) - external_used, limit

    def _seal_for_owner_confirmation(
        self, state: dict[str, Any], *, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        external = is_external_write(name)
        rule = (
            "这个操作会写入外部系统，写入后无法由这里撤回，因此必须由你亲自确认"
            if external
            else "你设置了这个操作需要先经你确认"
        )
        if self._confirmation_store is None:
            return self._refuse(
                state,
                name=name,
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
            result = ToolObservation(
                tool_name=name,
                state="capability_confirmation_in_progress",
                message="同一项已批准操作正在执行，没有再次发起确认或执行。",
                next_action="告诉用户操作仍在处理中，不要重试。",
            )
            return {
                "authorization_route": "observe",
                "pending": {
                    "name": name,
                    "result": result,
                    "synthetic_kind": "confirmation",
                    "runtime_owned": False,
                },
            }
        result = ToolObservation(
            tool_name=name,
            state="capability_confirmation_required",
            message=(
                f"{display_summary}\n"
                + (
                    "这是一次外部写入，执行后无法由这里撤回。是否执行？"
                    if external
                    else "你设置了此操作需要确认。是否执行？"
                )
            ),
            next_action="向用户说明将要执行什么并等待确认；本轮不要重试这个操作。",
            payload={"confirmation_id": confirmation.confirmation_id},
        )
        return {
            "authorization_route": "observe",
            "pending": {
                "name": name,
                "result": result,
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

    def _external_write_summary(self, *, name: str, arguments: dict[str, Any]) -> str:
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
