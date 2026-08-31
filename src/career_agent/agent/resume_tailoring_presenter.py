from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from career_agent.agent.resume_tailoring_contracts import ResumeTailoringResult


TailoringDisplayStatus = Literal[
    "pending", "in_review", "reviewed", "finalized", "superseded", "expired"
]

_STATUS_NOTICES = {
    "pending": "> 这是只读草稿预览；请在对话中明确接受、拒绝或要求修改。",
    "in_review": "> 这份草稿仍在审阅中；卡片本身不执行任何修改。",
    "reviewed": "> 修改建议已完成取舍；卡片保留当时的草稿内容。",
    "finalized": "> 这份草稿已经定稿；卡片仅供回看。",
    "superseded": "> 这份草稿已被更新的修订替代；请勿按此旧版本继续操作。",
    "expired": "> 这份草稿已经过期；内容仍可回看，但不能再审核或定稿。",
}


class TailoringChangeReviewView(BaseModel):
    """Only the historical decision fields the read-only presenter needs."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    change_index: int
    decision: Literal["accepted", "rejected"]


def render_resume_tailoring(
    result: ResumeTailoringResult,
    *,
    status: TailoringDisplayStatus,
    revision_number: int,
    change_reviews: tuple[TailoringChangeReviewView, ...] = (),
) -> str:
    """Render a historical, explicitly read-only tailoring draft."""
    decisions = {item.change_index: item.decision for item in change_reviews}
    blocks = [
        f"# 简历定制草稿 · 修订 {revision_number}",
        _STATUS_NOTICES[status],
        result.strategy_summary,
    ]
    if result.changes:
        changes = []
        for index, change in enumerate(result.changes, start=1):
            decision = decisions.get(index)
            decision_label = {
                "accepted": " · 已接受",
                "rejected": " · 已拒绝",
            }.get(decision, "")
            lines = [
                f"### {index}. {change.target_locator}{decision_label}",
                "",
            ]
            if change.original_quote:
                lines.extend((f"原文：{change.original_quote}", ""))
            lines.extend((f"建议：{change.proposed_text}", "", f"理由：{change.rationale}"))
            if change.addresses_requirements:
                lines.extend(
                    (
                        "",
                        "对应岗位要求：",
                        *[f"- {item}" for item in change.addresses_requirements],
                    )
                )
            lines.extend(("", "事实依据："))
            lines.extend(
                f"- {item.source_locator}：{item.source_quote}"
                for item in change.support_evidence
            )
            changes.append("\n".join(lines))
        blocks.append("## 修改建议\n\n" + "\n\n".join(changes))
    for title, items in (
        ("保留的优势", result.preserved_strengths),
        ("仍未解决的差距", result.unresolved_gaps),
        ("需要你补充", result.clarification_questions),
        ("注意事项", result.warnings),
    ):
        if items:
            blocks.append(f"## {title}\n\n" + "\n".join(f"- {item}" for item in items))
    return "\n\n".join(blocks)
