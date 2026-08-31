from __future__ import annotations

from career_agent.domain.job_research import JobResearchDraft


def render_job_research(
    draft: JobResearchDraft,
    *,
    status: str,
    user_provided_context: str | None = None,
    anchored_by_other_job: bool = False,
) -> str:
    """Render a source-grounded report without asking Main Agent to rewrite it."""
    blocks = ["# 公司调研", draft.summary]
    if anchored_by_other_job:
        blocks.append(
            "> 这份调研的主体是公司，内容来自你在同一家公司保存的另一个岗位所触发的"
            "检索。公司层面的结论对这个岗位同样适用，但检索锚点来自那份 JD，"
            "因此它未必覆盖这个岗位特有的产品线。需要针对性内容时可以指定 focus "
            "重新调研。"
        )
    if status == "outdated":
        blocks.append(
            "> 这份报告已超过当前时效窗口；引用仍可查看，但建议重新研究。"
        )
    if user_provided_context is not None:
        context_quote = "\n".join(
            f"> {line}" for line in user_provided_context.splitlines()
        )
        blocks.append(
            "## 用户提供的检索线索\n\n"
            f"{context_quote}\n\n"
            "该内容来自用户陈述，仅作为检索线索；除非下方公开来源独立支持，否则不视为事实。"
        )

    if draft.findings:
        rendered = []
        labels = {"fact": "事实", "inference": "推断", "unknown": "未知"}
        for finding in draft.findings:
            citations = (
                " ".join(f"[{key}]" for key in finding.source_keys)
                if finding.source_keys
                else ""
            )
            rendered.append(
                f"### {finding.topic}\n\n"
                f"{finding.statement} {citations}".rstrip()
                + f"\n\n- 类型：{labels[finding.evidence_type]}"
                f"\n- 置信度：{finding.confidence}"
            )
        blocks.append("## 研究结论\n\n" + "\n\n".join(rendered))

    if draft.open_questions:
        blocks.append(
            "## 面试前仍需确认\n\n"
            + "\n".join(f"- {item}" for item in draft.open_questions)
        )
    if draft.limitations:
        blocks.append(
            "## 信息限制\n\n"
            + "\n".join(f"- {item}" for item in draft.limitations)
        )
    if draft.sources:
        source_lines = []
        for source in draft.sources:
            publisher = f" — {source.publisher}" if source.publisher else ""
            published = (
                f"（{source.published_at.isoformat()}）"
                if source.published_at is not None
                else ""
            )
            title = source.title.replace("[", "\\[").replace("]", "\\]")
            safe_url = source.url.replace("<", "%3C").replace(">", "%3E")
            excerpt = "\n".join(
                f"> {line}" for line in source.relevant_excerpt.splitlines()
            )
            source_lines.append(
                f"### [{source.source_key}] [{title}](<{safe_url}>)"
                f"{publisher}{published}\n\n{excerpt}"
            )
        blocks.append("## 来源\n\n" + "\n\n".join(source_lines))
    return "\n\n".join(blocks)
