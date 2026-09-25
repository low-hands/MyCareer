from __future__ import annotations

from datetime import datetime
import re
from typing import Any

from career_agent.agent.mock_interview_contracts import (
    MockInterviewExchange,
    MockInterviewGraphResult,
    MockInterviewQuestionView,
    MockInterviewResultView,
)
from career_agent.agent.summary_text import condense
from career_agent.agent.mock_interview_company_styles import company_style_by_heading
from career_agent.domain.mock_interviews.models import (
    MockInterviewPlan,
    MockInterviewReport,
    MockInterviewSession,
    MockInterviewTurn,
)


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


def mock_interview_question_view(
    turns: tuple[MockInterviewTurn, ...], question_number: int
) -> MockInterviewQuestionView | None:
    matching = tuple(turn for turn in turns if turn.plan_item_number == question_number)
    if not matching:
        return None
    return MockInterviewQuestionView(
        question_number=question_number,
        exchanges=tuple(
            MockInterviewExchange(
                turn_type=turn.turn_type,
                question=turn.question,
                answer=turn.answer,
                rating=turn.evaluation.rating if turn.evaluation else None,
                evaluation_summary=turn.evaluation.summary if turn.evaluation else None,
            )
            for turn in matching
        ),
    )


def render_mock_interview_report(
    report: MockInterviewReport, plan=None, *, lead: bool = True
) -> str:
    """Render a finished mock interview for the screen.

    Kept out of the tool observation's ``message`` so the durable conversation
    row can stay one line: the report itself is an entity the UI reads back
    through the message's resource reference. Section titles are real
    headings, so the report card styles them like every other report; the
    "模拟面试完成。" lead is for the chat reply and is left off inside the card
    (``lead=False``), whose own header already names the report.
    """
    sections = (
        ("总结", (report.summary,)),
        ("表现亮点", report.strengths),
        ("待提升", report.development_areas),
        ("练习建议", report.practice_actions),
    )
    blocks = [
        f"## {title}"
        + "\n\n"
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
    question_type_labels = {
        "introduction": "自我介绍", "knowledge": "基础知识", "problem_solving": "问题解决",
        "system_design": "系统设计", "project_deep_dive": "项目深挖", "role_scenario": "岗位场景",
        "behavioral": "行为", "motivation": "动机", "career_planning": "职业规划",
    }
    def render_question(item):
        plan_item = next((candidate for candidate in (plan.items if plan is not None else ()) if candidate.sequence_number == item.plan_item_number), None)
        question_type = f" · {question_type_labels.get(plan_item.question_type, plan_item.question_type)}" if plan_item is not None else ""
        evidence = f"本题依据：{_escape_markdown_text('；'.join(plan_item.resume_quotes))}\n\n" if plan_item is not None and plan_item.resume_quotes else ""
        return (f"### 第 {item.plan_item_number} 题{question_type} · {rating_labels.get(item.final_rating, item.final_rating)}\n\n"
                f"**问题**：{_escape_markdown_text(item.question)}\n\n"
                f"{_escape_markdown_text(item.summary)}\n\n{evidence}追问次数：{item.follow_up_count}")
    question_sections = "\n\n".join(render_question(item) for item in report.question_results)
    if question_sections:
        blocks.append(f"## 每题反馈\n\n{question_sections}")
    return "\n\n".join((("模拟面试完成。",) if lead else ()) + tuple(blocks))


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
        if result.resume_basis is not None:
            blocks.append(_escape_markdown_text(result.resume_basis))
        # The bare question, as the transcript shows it after a reload.
        blocks.append(_escape_markdown_text(result.question))
        return "\n\n".join(blocks)
    return result.message


def practice_basis_line(
    *,
    resume: tuple[str, int] | None,
    job: str | None = None,
    company: str | None = None,
    research_at: datetime | None = None,
    style: str | None = None,
    style_inferred: bool = False,
) -> str:
    """Say up front what the questions come from, and what they do not.

    ``resume`` is (name, version number); ``job`` is "公司 · 岗位" for a chosen
    saved job, ``company`` the employer named without one.
    """
    basis = [f"简历《{resume[0]}》v{resume[1]}"] if resume is not None else []
    if job is not None:
        basis.append(f"岗位《{job}》的 JD")
    elif company is not None:
        basis.append(f"公司「{company}」（没有 JD）")
    notes = []
    if style is not None:
        notes.append(f"面试风格参考{style}" + ("（由公司名推断）" if style_inferred else ""))
    if research_at is not None:
        notes.append(f"业务背景参考你 {research_at:%Y-%m-%d} 的公司研究")
    if not basis:
        return "本场不参考简历，只考察通用题和专业基础。"
    joined = "、".join(basis)
    # A space between a trailing Latin token ("v2", "JD") and the Chinese verb.
    gap = " " if joined[-1].isascii() and joined[-1].isalnum() else ""
    line = ("本场不参考简历，" if resume is None else "本场") + f"基于{joined}{gap}出题"
    return line + "".join(f"；{note}" for note in notes) + "。"


def practice_basis(
    session: MockInterviewSession,
    *,
    resumes: Any,
    jobs: Any = None,
    research_at: datetime | None = None,
    plan: MockInterviewPlan | None = None,
) -> str:
    """Resolve a session's pinned resume and job for the line.

    Every id is read as pinned at start, so a reload describes the run exactly
    as it was based, even after a newer JD version. ``research_at`` is when the
    pinned company research report was written, looked up by the caller.
    """
    located = (
        resumes.get_version(user_id=session.user_id, resume_version_id=session.resume_version_id)
        if session.resume_version_id is not None and resumes is not None
        else None
    )
    record = (
        jobs.get_job(user_id=session.user_id, job_posting_id=session.job_posting_id)
        if session.job_posting_id is not None and jobs is not None
        else None
    )
    style = company_style_by_heading(plan.company_style_profile if plan is not None else None)
    return practice_basis_line(
        resume=(located[0].name, located[1].version_number) if located is not None else None,
        job=(
            f"{record.posting.company_name} · {record.posting.title}"
            if record is not None
            else None
        ),
        company=session.target_company,
        research_at=research_at,
        style=style.display_name if style is not None else None,
        style_inferred=plan.company_style_inferred if plan is not None else False,
    )


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
