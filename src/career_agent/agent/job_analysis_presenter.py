from __future__ import annotations

from career_agent.agent.job_analysis_contracts import (
    SENIORITY_LABELS,
    TIER_LABELS,
    TIER_ORDER,
    JobAnalysisResult,
    QuotedFinding,
)

_KIND_LABELS = {"fact": "事实", "inference": "推断"}


def _quoted(items: tuple[QuotedFinding, ...]) -> str:
    return "\n".join(f"- {item.text}\n  > JD：{item.jd_quote}" for item in items)


def render_job_analysis(result: JobAnalysisResult) -> str:
    """Render the JD-only analysis, tier by tier, without any resume context."""
    blocks = [
        "# 岗位 JD 分析",
        f"核心目标：{result.core_objective}",
        f"层级判定：{SENIORITY_LABELS[result.seniority]}",
        result.summary,
    ]
    if result.requirements:
        sections = []
        for tier in TIER_ORDER:
            items = result.requirements_in_tier(tier)
            if not items:
                continue
            rows = "\n".join(
                f"- **{item.text}**（{_KIND_LABELS[item.kind]}）\n  > JD：{item.jd_quote}"
                for item in items
            )
            sections.append(f"### {TIER_LABELS[tier]}\n\n{rows}")
        blocks.append("## 分级要求\n\n" + "\n\n".join(sections))
    if result.core_competencies:
        blocks.append(
            "## 核心能力\n\n" + "\n".join(f"- {item}" for item in result.core_competencies)
        )
    if result.implicit_requirements:
        blocks.append("## 隐性要求\n\n" + _quoted(result.implicit_requirements))
    for title, items in (
        ("ATS 关键词", result.ats_keywords),
        ("HR 关注点", result.hr_focus),
        ("Hiring Manager 关注点", result.hiring_manager_focus),
        ("可能的面试话题", result.likely_interview_topics),
    ):
        if items:
            blocks.append(f"## {title}\n\n" + "\n".join(f"- {item}" for item in items))
    if result.red_flags:
        blocks.append("## 红旗信号\n\n" + _quoted(result.red_flags))
    if result.information_gaps:
        blocks.append(
            "## 待向 HR 确认\n\n"
            + "\n".join(f"- {item}" for item in result.information_gaps)
        )
    return "\n\n".join(blocks)
