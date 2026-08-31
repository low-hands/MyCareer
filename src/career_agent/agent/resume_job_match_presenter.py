from __future__ import annotations

from career_agent.agent.resume_job_match_contracts import ResumeJobMatchResult


_FIT_LABELS = {
    "strong": "强匹配",
    "moderate": "中等匹配",
    "weak": "弱匹配",
    "insufficient_evidence": "证据不足",
}

_STATUS_LABELS = {
    "matched": "匹配",
    "partial": "部分匹配",
    "missing": "未体现",
    "unclear": "不明确",
}


def render_resume_job_match(result: ResumeJobMatchResult) -> str:
    """Render the grounded requirement-by-requirement comparison."""
    blocks = [
        "# 简历与岗位匹配",
        f"整体判断：{_FIT_LABELS.get(result.overall_fit, result.overall_fit)}",
        result.summary,
    ]
    if result.requirements:
        rows = []
        for index, item in enumerate(result.requirements, start=1):
            lines = [
                f"### {index}. [{_STATUS_LABELS.get(item.status, item.status)}] {item.requirement}",
                "",
                f"> JD：{item.jd_quote}",
                "",
                item.rationale,
            ]
            if item.resume_evidence:
                lines.extend(("", "简历证据："))
                lines.extend(
                    f"- {evidence.source_locator}：{evidence.source_quote}"
                    for evidence in item.resume_evidence
                )
            rows.append("\n".join(lines))
        blocks.append("## 要求逐项核对\n\n" + "\n\n".join(rows))
    for title, items in (
        ("建议", result.recommendations),
        ("需要补充的信息", result.clarification_questions),
        ("分析限制", result.limitations),
    ):
        if items:
            blocks.append(f"## {title}\n\n" + "\n".join(f"- {item}" for item in items))
    return "\n\n".join(blocks)
