"""Grounding policy for the Job Discovery Agent."""

from __future__ import annotations

import re

from career_agent.contracts import AgentRequest
from career_agent.runtime.response_guard import (
    GuardDecision,
    TerminalResponseGuard,
    TurnEvidence,
)

_SEARCH_TOOL = "career_search_jobs"
_DETAIL_TOOL = "career_get_job_detail"

_DETAIL_PATTERNS = (
    re.compile(r"(?:完整\s*(?:jd|职位描述)|jd\s*详情|岗位详情|职位详情|工作职责|岗位职责|任职要求)", re.I),
    re.compile(r"第\s*(?:\d+|[一二三四五六七八九十]+)\s*(?:个|条|项|份)?\s*(?:岗位|职位|jd)?", re.I),
    re.compile(r"(?:想看|看看|查看|打开|展开).{0,30}(?:那个|这个|这条|｜|\|)", re.I),
)
_SEARCH_PATTERNS = (
    re.compile(r"(?:搜|搜索|查找|找|推荐|看看|看下|有没有).{0,24}(?:岗位|职位|工作|招聘)", re.I),
    re.compile(r"(?:岗位|职位|工作|招聘).{0,12}(?:推荐|搜索|检索)", re.I),
)


class JobGroundingGuard(TerminalResponseGuard):
    """Require grounded tool evidence for explicit listing-data requests.

    Intent detection is deliberately conservative. This guard is a safety net
    for explicit search/detail requests, not a general conversation router.
    """

    async def review(
        self,
        *,
        request: AgentRequest,
        response_text: str,
        evidence: TurnEvidence,
    ) -> GuardDecision:
        required_tool = _required_tool(request.message)
        if not required_tool:
            return GuardDecision(accepted=True)

        successful_tools = {tool_call.name for tool_call in evidence.tool_calls if tool_call.succeeded}
        if required_tool in successful_tools:
            return GuardDecision(accepted=True)

        available = required_tool in evidence.available_tool_names
        availability_note = (
            f"`{required_tool}` is available and must be called."
            if available
            else f"`{required_tool}` is unavailable, so you must not invent an answer."
        )
        return GuardDecision(
            accepted=False,
            code="missing_grounding_tool",
            required_tool=required_tool,
            feedback=(
                "The proposed answer is not grounded in the required job tool. "
                f"{availability_note} Do not answer from memory. "
                "Use the tool result, including no_results, ambiguous, or not_found, "
                "before producing the final user-facing response."
            ),
            safe_output=("无法在没有岗位工具证据的情况下回答这个请求。请稍后重试，或检查岗位搜索服务是否可用。"),
        )


def _required_tool(message: str) -> str:
    normalized = " ".join(message.strip().split())
    if any(pattern.search(normalized) for pattern in _DETAIL_PATTERNS):
        return _DETAIL_TOOL
    if any(pattern.search(normalized) for pattern in _SEARCH_PATTERNS):
        return _SEARCH_TOOL
    return ""
