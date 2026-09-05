from __future__ import annotations

import re

from career_agent.agent.mock_interview_contracts import (
    MockInterviewGraphResult,
    MockInterviewQuestionView,
    MockInterviewResultView,
)
from career_agent.agent.summary_text import condense
from career_agent.domain.mock_interviews.models import MockInterviewReport


_MARKDOWN_LINK_BRACKET = re.compile(r"([\[\]])")
_MARKDOWN_HEADING_PREFIX = re.compile(r"(?m)^([ \t]*)(#)(?=#*\s|$)")
_MARKDOWN_BLOCKQUOTE_PREFIX = re.compile(r"(?m)^([ \t]*)(>)")
_MARKDOWN_SETEXT_HEADING = re.compile(r"(?m)^([ \t]*)([=-])(?=\2{2,}[ \t]*$)")
_MARKDOWN_HTTP_URL = re.compile(r"(?i)\b(https?):(?=//)")
_MARKDOWN_WWW_URL = re.compile(r"(?i)\b(www\.)(?=[^\s]+)")


def _escape_markdown_text(value: str) -> str:
    """Allow useful emphasis/code/lists but not links, headings, or raw HTML."""
    escaped_html = value.replace("&", "&amp;").replace("<", "&lt;")
    escaped_links = _MARKDOWN_LINK_BRACKET.sub(r"\\\1", escaped_html)
    escaped_headings = _MARKDOWN_HEADING_PREFIX.sub(r"\1\\\2", escaped_links)
    escaped_headings = _MARKDOWN_BLOCKQUOTE_PREFIX.sub(
        r"\1\\\2", escaped_headings
    )
    escaped_headings = _MARKDOWN_SETEXT_HEADING.sub(
        r"\1\\\2", escaped_headings
    )
    escaped_urls = _MARKDOWN_HTTP_URL.sub(r"\1:" + "\u200b", escaped_headings)
    return _MARKDOWN_WWW_URL.sub(r"\1" + "\u200b", escaped_urls)


def render_mock_interview_report(report: MockInterviewReport) -> str:
    """Render a finished mock interview for the screen.

    Kept out of the tool observation's ``message`` so the durable conversation
    row can stay one line: the report itself is an entity the UI reads back
    through the message's resource reference.
    """
    sections = (
        ("总结", (report.summary,)),
        ("表现亮点", report.strengths),
        ("待提升", report.development_areas),
        ("练习建议", report.practice_actions),
    )
    blocks = [
        title
        + "\n"
        + (
            "\n".join(f"- {_escape_markdown_text(item)}" for item in items)
            or "- 暂无"
        )
        for title, items in sections
    ]
    rating_labels = {
        "strong": "表现突出",
        "adequate": "达到要求",
        "weak": "需要加强",
        "insufficient_evidence": "信息不足",
    }
    question_sections = "\n\n".join(
        (
            f"### 第 {item.plan_item_number} 题 · "
            f"{rating_labels.get(item.final_rating, item.final_rating)}\n\n"
            f"**问题**：{_escape_markdown_text(item.question)}\n\n"
            f"{_escape_markdown_text(item.summary)}\n\n"
            f"追问次数：{item.follow_up_count}"
        )
        for item in report.question_results
    )
    if question_sections:
        blocks.append(f"## 每题反馈\n\n{question_sections}")
    return "\n\n".join(("模拟面试完成。", *blocks))


def render_mock_interview_turn(result: MockInterviewGraphResult) -> str:
    """Render one advance of a mock interview run for the screen.

    Every state a run can be left in is handled here, in the same module as the
    report renderer, because they are one audience's view of one workflow. This
    previously lived in two places outside the workflow — the tool layer built
    the ``awaiting_answer`` copy and hardcoded the terminal one-liners, while
    the runtime's presenter dispatch re-parsed the serialized payload back into
    this very type just to reach ``render_mock_interview_report``. Both layers
    are generic; neither had a reason to know how an interview reads.

    The graph's own ``message`` is internal status text, so it is used only for
    ``running``, the one state with nothing else to say.
    """
    if result.state == "completed":
        if result.report is not None:
            return render_mock_interview_report(result.report)
        # A completed run with no report in hand: the session is closed either
        # way, so say so rather than falling through to internal status text.
        return "模拟面试完成。"
    if result.state == "cancelled":
        return "模拟面试已取消。"
    if result.state == "awaiting_answer" and result.question is not None:
        blocks = []
        if result.evaluation is not None:
            blocks.append(
                "上一题反馈：\n"
                f"{_escape_markdown_text(result.evaluation.summary)}\n"
                "下一步原因："
                f"{_escape_markdown_text(result.evaluation.next_action_reason)}"
            )
        blocks.append(f"模拟面试题：\n{_escape_markdown_text(result.question)}")
        return "\n\n".join(blocks)
    return result.message


def summarize_mock_interview_report(report: MockInterviewReport) -> str:
    """State a finished run's outcome in the one line the transcript keeps."""
    headline = _escape_markdown_text(condense(report.summary))
    return f"模拟面试完成。{headline}" if headline else "模拟面试完成。"


INTERVIEW_TYPE_LABELS = {
    "technical": "技术面",
    "role_specific": "专业面",
    "behavioral": "行为面",
    "hr": "HR 面",
    "mixed": "综合面",
}


def _result_headline(view: MockInterviewResultView) -> str:
    """Name the run by what it covered and how much of it was attempted.

    Asked and answered are different numbers once a run can stop early, so an
    abandoned question is still a row in the index. Reporting only the row count
    would present it as an attempted one.
    """
    label = INTERVIEW_TYPE_LABELS.get(view.interview_type, view.interview_type)
    parts = [f"{label}，{len(view.questions)} 题"]
    if view.answered_count < len(view.questions):
        parts.append(f"答了 {view.answered_count} 题")
    if view.status == "cancelled":
        parts.append("中途取消")
    return "模拟面试（" + "，".join(parts) + "）"


def render_mock_interview_result(view: MockInterviewResultView) -> str:
    """Render a finished run's index for the screen.

    An index rather than a transcript: naming a question number reads that
    exchange in full, so a long run costs one short screen instead of every
    answer at once.
    """
    lines = [
        f"{item.plan_item_number}. [{item.rating}"
        + (f"，追问 {item.follow_up_count} 次" if item.follow_up_count else "")
        + f"] {_escape_markdown_text(item.question[:60])}"
        for item in view.questions
    ]
    blocks = [
        _result_headline(view),
        "题目\n" + ("\n".join(lines) if lines else "暂无"),
    ]
    if view.report_summary is not None:
        blocks.append(f"总结\n{_escape_markdown_text(view.report_summary)}")
    blocks.append("要看某题的完整问答，说题号。")
    return "\n\n".join(blocks)


def summarize_mock_interview_result(view: MockInterviewResultView) -> str:
    """State a readback's outcome in the one line the transcript keeps.

    The index itself is not kept: it is reproducible from the store on demand,
    and the report it points at is reachable through the message's resource
    reference, so the row only has to say which run was opened.
    """
    headline = _result_headline(view)
    if view.report_summary is None:
        return f"已读取{headline}。"
    return f"已读取{headline}。{_escape_markdown_text(condense(view.report_summary))}"


def render_mock_interview_question(view: MockInterviewQuestionView) -> str:
    """Render one exchange for the screen, follow-ups included."""
    blocks = []
    for exchange in view.exchanges:
        label = "追问" if exchange.turn_type == "follow_up" else "主问题"
        lines = [f"{label}\n{_escape_markdown_text(exchange.question)}"]
        if exchange.answer is not None:
            lines.append(f"你的回答\n{_escape_markdown_text(exchange.answer)}")
        if exchange.evaluation_summary is not None:
            lines.append(
                f"评价（{exchange.rating}）\n"
                f"{_escape_markdown_text(exchange.evaluation_summary)}"
            )
        blocks.append("\n\n".join(lines))
    header = f"第 {view.question_number} 题："
    return header + "\n\n" + "\n\n---\n\n".join(blocks)


def summarize_mock_interview_question(view: MockInterviewQuestionView) -> str:
    """State which exchange was opened, without quoting it.

    The answer this reads back can run to twenty thousand characters, so the
    verbatim text stays on the screen it was asked for and the row records only
    that it was asked. Reading it again is one tool call away.
    """
    follow_ups = sum(
        1 for exchange in view.exchanges if exchange.turn_type == "follow_up"
    )
    suffix = f"，含 {follow_ups} 次追问" if follow_ups else ""
    return (
        f"已读取模拟面试第 {view.question_number} 题的完整问答{suffix}。"
        f"{_escape_markdown_text(condense(view.exchanges[0].question))}"
    )
