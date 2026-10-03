from __future__ import annotations

from typing import Any

from career_agent.agent.contracts.main_agent import ToolObservation
from career_agent.agent.runtime.state import MainAgentState
from career_agent.agent.capabilities.effects import is_notes_guarded, is_preference_bound
from career_agent.agent.middleware.working_notes import (
    remembered_preference_without_authority,
    working_notes_only_tokens,
)


class WorkingNotesGuardMiddleware:
    """Prevent unconfirmed working notes from becoming execution authority."""

    def __init__(self, *, max_projection_refusals: int) -> None:
        self._max_projection_refusals = max_projection_refusals

    def check(
        self,
        state: MainAgentState,
        *,
        name: str,
        arguments: dict[str, Any],
        runtime_owned: bool,
        owner_confirmed: bool,
        policy_owned: bool,
        policy_prelude: bool,
    ) -> MainAgentState | None:
        note_only_tokens = (
            working_notes_only_tokens(
                arguments=arguments, context=state["context"]
            )
            if is_notes_guarded(name) and not runtime_owned and not owner_confirmed
            else ()
        )
        refusal: ToolObservation | None = None
        if note_only_tokens:
            visible_tokens = [token[:32] for token in note_only_tokens[:8]]
            refusal = ToolObservation(
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
            refusal = ToolObservation(
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
        if refusal is None:
            return None
        if (
            state.get("control", {}).get("projection_refusals", 0)
            >= self._max_projection_refusals
        ):
            return {"authorization_route": "present"}
        return {
            "authorization_route": "observe",
            "pending": {
                "name": name,
                "result": refusal,
                "synthetic_kind": "projection",
                "runtime_owned": runtime_owned,
                "policy_owned": policy_owned,
                "policy_prelude": policy_prelude,
            },
        }
