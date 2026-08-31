from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from career_agent.domain.interviews import (
    InterviewRetroQuestion,
    InterviewRetroReport,
    InterviewSelfAssessment,
)


_ASSESSMENT_LABELS = {
    "strong": "表现较好",
    "mixed": "有好有坏",
    "weak": "表现不足",
    "uncertain": "暂不确定",
}


def _bullets(items: tuple[str, ...]) -> str:
    return "\n".join(f"- {item}" for item in items)


class InterviewRetroView(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    source_notes: str
    summary: str
    questions: tuple[InterviewRetroQuestion, ...] = ()
    strengths: tuple[str, ...] = ()
    difficulties: tuple[str, ...] = ()
    interviewer_signals: tuple[str, ...] = ()
    next_focus: tuple[str, ...] = ()
    action_items: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    self_assessment: InterviewSelfAssessment = "uncertain"

    @classmethod
    def of(cls, report: InterviewRetroReport) -> "InterviewRetroView":
        return cls.model_validate(report.model_dump(mode="python"))


def render_interview_retro(report: InterviewRetroReport | InterviewRetroView) -> str:
    """Render one immutable real-interview retrospective from its entity."""
    blocks = [
        "# 真实面试复盘",
        report.summary,
        f"自我判断：{_ASSESSMENT_LABELS.get(report.self_assessment, report.self_assessment)}",
    ]
    if report.questions:
        questions = []
        for index, item in enumerate(report.questions, start=1):
            lines = [f"### {index}. {item.question}"]
            if item.answer_summary:
                lines.append(f"回答回忆：{item.answer_summary}")
            lines.append(
                "自评："
                + _ASSESSMENT_LABELS.get(item.self_assessment, item.self_assessment)
            )
            if item.notes:
                lines.append(f"备注：{item.notes}")
            questions.append("\n\n".join(lines))
        blocks.append("## 问答回忆\n\n" + "\n\n".join(questions))
    blocks.append("## 你的原始复述\n\n" + report.source_notes)
    for title, items in (
        ("表现亮点", report.strengths),
        ("困难与不足", report.difficulties),
        ("面试官信号", report.interviewer_signals),
        ("下一步准备重点", report.next_focus),
        ("行动项", report.action_items),
        ("信息限制", report.limitations),
    ):
        if items:
            blocks.append(f"## {title}\n\n{_bullets(items)}")
    blocks.append("> 本报告只依据你的复述，不推断面试官评价或招聘结果。")
    return "\n\n".join(blocks)
