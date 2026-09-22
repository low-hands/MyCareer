from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from career_agent.agent.resume_tailoring_contracts import (
    GapMitigation,
    ResumeTailoringResult,
)


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

_EVIDENCE_QUALITY_LABELS = {
    "exact": "已核对原文",
    "normalized": "已归一化核对",
    "ocr_unverified": "待用户确认（OCR 未验证）",
}


class TailoringChangeReviewView(BaseModel):
    """Only the historical decision fields the read-only presenter needs."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    change_index: int
    decision: Literal["accepted", "rejected"]


def _render_gap_mitigation(item: GapMitigation, index: int) -> str:
    adjacent_experience = getattr(item, "adjacent_experience", ())
    alternative_evidence = getattr(item, "alternative_evidence", ())
    learning_plan = getattr(item, "learning_plan", None)
    clarification_question = getattr(item, "clarification_question", None)
    type_label = {
        "hard_blocker": "硬性 blocker",
        "strengthenable": "可补强项",
    }[item.gap_type]
    mode_label = {
        "clarify": "澄清招聘方 / 补充信息",
        "provide_evidence": "补充已有证据",
        "build_artifact": "制作计划产物",
        "learn": "学习并达到可演示水平",
        None: "旧版未分类",
    }[item.resolution_mode]
    lines = [
        f"### {index}. [{item.priority}] {item.gap or '未命名缺口'}",
        "",
        f"分类：{type_label}",
        "",
        f"处理方式：{mode_label}",
        "",
        f"判断：{item.rationale}",
        "",
        f"下一步：{item.next_action}",
    ]
    if adjacent_experience:
        lines.extend(("", "相邻经验："))
        lines.extend(
            f"- {evidence.source_locator}：{evidence.source_quote}（{evidence.relevance}；"
            f"{_EVIDENCE_QUALITY_LABELS[evidence.evidence_quality]}"
            + (f"；第 {evidence.page} 页" if evidence.page is not None else "")
            + ")"
            for evidence in adjacent_experience
        )
    if alternative_evidence:
        lines.extend(("", "可替代证据："))
        for evidence in alternative_evidence:
            if isinstance(evidence, str):
                lines.append(f"- 旧版未分类：{evidence}")
            elif evidence.status == "existing":
                lines.append(
                    f"- 已有材料：{evidence.description}"
                    f"（{evidence.source_locator}：{evidence.source_quote}；"
                    f"{_EVIDENCE_QUALITY_LABELS[evidence.evidence_quality]}"
                    + (f"；第 {evidence.page} 页" if evidence.page is not None else "")
                    + ")"
                )
            else:
                lines.append(
                    f"- 计划产物：{evidence.description}"
                    f"（验收标准：{evidence.acceptance_criteria}）"
                )
    if learning_plan is not None:
        plan = learning_plan
        lines.extend(
            (
                "",
                "学习路线：",
                f"- 目标：{plan.objective}",
                "- 资源方向：" + "；".join(plan.resource_directions),
                f"- 最低可接受水平：{plan.minimum_acceptable_level}",
            )
        )
        if plan.estimated_effort is not None:
            lines.append(f"- 预计投入：{plan.estimated_effort}")
    talking_point = item.interview_talking_point
    if talking_point is not None:
        lines.extend(("", "面试诚实表达：", f"- 承认缺口：{talking_point.acknowledge_gap}"))
        if talking_point.bridge_to_evidence is not None:
            lines.append(f"- 连接经验：{talking_point.bridge_to_evidence}")
        lines.append(f"- 补强动作：{talking_point.close_with_action}")
    if clarification_question is not None:
        lines.extend(("", f"澄清问题：{clarification_question}"))
    return "\n".join(lines)


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
                f"（{_EVIDENCE_QUALITY_LABELS[item.evidence_quality]}"
                + (f"；第 {item.page} 页" if item.page is not None else "")
                + ")"
                for item in change.support_evidence
            )
            changes.append("\n".join(lines))
        blocks.append("## 修改建议\n\n" + "\n\n".join(changes))
    if result.gap_mitigations:
        ordered = sorted(
            result.gap_mitigations,
            key=lambda item: (
                {"P0": 0, "P1": 1, "P2": 2}[item.priority],
                item.gap or "",
            ),
        )
        blocks.append(
            "## 缺口缓解与学习路线\n\n"
            + "\n\n".join(
                _render_gap_mitigation(item, index)
                for index, item in enumerate(ordered, start=1)
            )
        )
    for title, items in (
        ("保留的优势", result.preserved_strengths),
        (
            "仍未解决的差距",
            result.unresolved_gaps if not result.gap_mitigations else (),
        ),
        ("需要你补充", result.clarification_questions),
        ("注意事项", result.warnings),
    ):
        if items:
            blocks.append(f"## {title}\n\n" + "\n".join(f"- {item}" for item in items))
    return "\n\n".join(blocks)
