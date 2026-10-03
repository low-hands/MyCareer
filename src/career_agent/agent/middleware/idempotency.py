from __future__ import annotations

from dataclasses import dataclass
import json

from career_agent.agent.main_agent_contracts import AgentDecision
from career_agent.agent.main_state import LoopControl
from career_agent.agent.middleware.contracts import AuthorizationRefusal


@dataclass(frozen=True)
class IdempotencyAccepted:
    control: LoopControl


class IdempotencyMiddleware:
    """Reject duplicate calls and bound explicitly retryable failures."""

    def __init__(self, *, max_failure_retries: int) -> None:
        self._max_failure_retries = max_failure_retries

    @staticmethod
    def fingerprint(decision: AgentDecision) -> str:
        if decision.tool_call is None:
            return ""
        return json.dumps(
            {
                "name": decision.tool_call.name,
                "arguments": decision.tool_call.arguments,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def check(
        self, control: LoopControl, *, decision: AgentDecision
    ) -> IdempotencyAccepted | AuthorizationRefusal:
        fingerprint = self.fingerprint(decision)
        if fingerprint not in control.get("fingerprints", ()):
            return IdempotencyAccepted(control=control)
        if fingerprint not in control.get("retryable_fingerprints", ()):
            return AuthorizationRefusal(
                kind="duplicate_call",
                reason="相同调用已经执行过，且上次结果没有声明为可重试。",
                next_action=(
                    "这次调用和本轮之前那次完全一样，再调一次也不会有新结果。"
                    "请用已有的观察作答，或者换一组参数。"
                ),
            )
        retry_counts = dict(control.get("retry_counts", {}))
        retries = retry_counts.get(fingerprint, 0)
        if retries >= self._max_failure_retries:
            return AuthorizationRefusal(
                kind="retry_limit",
                reason="相同失败调用已经达到本轮重试上限。",
                next_action=(
                    "同一个失败调用已经重试到本轮上限。别再重试；"
                    "把失败讲清楚，或者问用户要不要换个做法。"
                ),
            )
        retry_counts[fingerprint] = retries + 1
        return IdempotencyAccepted(
            control={**control, "retry_counts": retry_counts}
        )
