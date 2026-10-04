from __future__ import annotations

from career_agent.agent.capabilities.catalog import DOMAIN_TOOL_PROFILES
from career_agent.agent.runtime.state import LoopControl
from career_agent.agent.middleware.contracts import AuthorizationRefusal
from career_agent.agent.capabilities.effects import ToolEffect, is_external_write


class BudgetMiddleware:
    """Classify and enforce per-turn delegation budgets."""

    def __init__(
        self,
        *,
        max_read_calls: int,
        max_write_calls: int,
        max_external_write_calls: int,
    ) -> None:
        self._max_read_calls = max_read_calls
        self._max_write_calls = max_write_calls
        self._max_external_write_calls = max_external_write_calls

    def bucket(
        self, control: LoopControl, *, name: str, effect: ToolEffect
    ) -> tuple[str, int, int]:
        if effect == "READ":
            return "READ", control.get("read_calls", 0), self._max_read_calls
        if effect == "CONTROL":
            return (
                "CONTROL",
                control.get("control_calls", 0),
                len(DOMAIN_TOOL_PROFILES),
            )
        external_used = control.get("external_write_calls", 0)
        if is_external_write(name):
            return (
                "WRITE_EXTERNAL",
                external_used,
                self._max_external_write_calls,
            )
        limit = self._max_write_calls
        if name == "match_resume_to_job" and control.get(
            "job_analysis_write_used", False
        ):
            limit += 1
        return "WRITE", control.get("write_calls", 0) - external_used, limit

    def check(
        self, control: LoopControl, *, name: str, effect: ToolEffect
    ) -> AuthorizationRefusal | None:
        bucket, used, limit = self.bucket(control, name=name, effect=effect)
        if used < limit:
            return None
        return AuthorizationRefusal(
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
