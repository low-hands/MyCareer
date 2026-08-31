from __future__ import annotations

from collections.abc import Iterable

from career_agent.agent.summary_text import condense
from career_agent.domain.interview_preparation import InterviewPreparationResult


def _bullets(items: Iterable[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _section(title: str, body: str) -> str:
    return f"## {title}\n\n{body}"


def render_interview_preparation(result: InterviewPreparationResult) -> str:
    """Render the authoritative structured brief without another model call."""
    sections = ["# 面试准备", result.summary]

    if result.focus_areas:
        sections.append(
            _section(
                "准备重点",
                "\n\n".join(
                    (
                        f"### {item.topic}（{item.priority}）\n\n"
                        f"{item.rationale}\n\n"
                        f"> JD：{item.jd_quote}"
                    )
                    for item in result.focus_areas
                ),
            )
        )

    if result.evidence_stories:
        sections.append(
            _section(
                "可准备的经历证据",
                "\n\n".join(
                    (
                        f"### {item.theme}\n\n"
                        f"- 简历位置：{item.resume_locator}\n"
                        f"- 简历原句：{item.resume_quote}\n"
                        f"- 准备提示：{item.preparation_prompt}"
                    )
                    for item in result.evidence_stories
                ),
            )
        )

    if result.likely_questions:
        question_blocks = []
        for index, item in enumerate(result.likely_questions, start=1):
            lines = [f"### {index}. {item.question}", "", item.rationale]
            if item.answer_outline:
                lines.extend(("", "回答提纲：", _bullets(item.answer_outline)))
            if item.follow_ups:
                lines.extend(("", "可能追问：", _bullets(item.follow_ups)))
            question_blocks.append("\n".join(lines))
        sections.append(_section("可能的问题", "\n\n".join(question_blocks)))

    if result.gaps:
        sections.append(
            _section(
                "需要诚实处理的差距",
                "\n\n".join(
                    (
                        f"### {item.gap}\n\n"
                        f"> JD：{item.jd_quote}\n\n"
                        f"应对策略：{item.honest_response_strategy}"
                    )
                    for item in result.gaps
                ),
            )
        )

    if result.questions_to_ask:
        sections.append(
            _section(
                "可以反问面试官",
                "\n\n".join(
                    f"- {item.question}\n  - 原因：{item.rationale}"
                    for item in result.questions_to_ask
                ),
            )
        )

    if result.checklist:
        sections.append(_section("面试前检查", _bullets(result.checklist)))
    if result.limitations:
        sections.append(_section("信息限制", _bullets(result.limitations)))

    return "\n\n".join(sections)


def summarize_interview_preparation(result: InterviewPreparationResult) -> str:
    """State the brief's outcome in the one line the transcript keeps.

    Deterministic on purpose: this is the copy the durable row falls back to
    when the answer writer did not run, and the writer cannot cover its own
    failure.
    """
    headline = condense(result.summary)
    counts = (
        f"（{len(result.likely_questions)} 个可能问题，"
        f"{len(result.focus_areas)} 个准备重点）"
    )
    return f"面试准备材料已生成。{headline}{counts}" if headline else (
        f"面试准备材料已生成。{counts}"
    )
