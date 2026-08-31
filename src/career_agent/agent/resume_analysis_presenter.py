from __future__ import annotations

from career_agent.agent.resume_analysis_contracts import ResumeAnalysisResult


_RECORD_LABELS = {
    "education": "教育经历",
    "work": "工作经历",
    "internship": "实习经历",
    "project": "项目经历",
    "certification": "证书",
}


def _period(record) -> str:
    def point(year: int | None, month: int | None) -> str:
        if year is None:
            return "时间未知"
        return f"{year}.{month:02d}" if month is not None else str(year)

    end = "至今" if record.is_current else point(record.end_year, record.end_month)
    return f"{point(record.start_year, record.start_month)} – {end}"


def render_resume_analysis(result: ResumeAnalysisResult) -> str:
    """Render every extracted candidate so the user can verify it before import."""
    blocks = ["# 简历分析结果"]
    if not result.records:
        blocks.append("没有提取到可确认的职业经历。")
    for index, record in enumerate(result.records, start=1):
        heading = " · ".join(
            part
            for part in (
                record.title,
                record.organization,
                _RECORD_LABELS.get(record.record_type, record.record_type),
            )
            if part
        )
        lines = [
            f"## {index}. {heading}",
            "",
            f"- 时间：{_period(record)}",
            f"- 来源位置：{record.source_locator}",
            f"- 原文：{record.source_quote}",
        ]
        if record.evidence:
            lines.extend(("", "事实候选："))
            lines.extend(
                f"- {item.claim}（{item.source_locator}：{item.source_quote}）"
                for item in record.evidence
            )
        blocks.append("\n".join(lines))
    if result.clarification_questions:
        blocks.append(
            "## 需要你确认\n\n"
            + "\n".join(f"- {item}" for item in result.clarification_questions)
        )
    if result.warnings:
        blocks.append(
            "## 注意事项\n\n" + "\n".join(f"- {item}" for item in result.warnings)
        )
    blocks.append("> 以上内容尚未写入职业事实库；只有你明确确认后才会保存。")
    return "\n\n".join(blocks)
