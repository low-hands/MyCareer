from career_agent.agent.interview_preparation_presenter import (
    render_interview_preparation,
)
from career_agent.domain.interview_preparation import (
    EvidenceStory,
    GapPreparation,
    InterviewFocusArea,
    InterviewPreparationResult,
    LikelyQuestion,
    QuestionToAsk,
)


def test_presenter_renders_every_available_section_without_internal_metadata() -> None:
    rendered = render_interview_preparation(InterviewPreparationResult(
        summary="重点准备可靠性和个人贡献。",
        focus_areas=(InterviewFocusArea(
            topic="可靠性",
            priority="high",
            rationale="这是岗位核心要求。",
            jd_quote="Build reliable systems.",
        ),),
        evidence_stories=(EvidenceStory(
            theme="检索评估",
            resume_locator="project.1",
            resume_quote="Built a retrieval evaluation suite.",
            preparation_prompt="补充真实使用的指标。",
        ),),
        likely_questions=(LikelyQuestion(
            question="你如何定义可靠？",
            rationale="验证指标意识。",
            answer_outline=("先定义目标", "再说明监控"),
            follow_ups=("失败时如何降级？",),
        ),),
        gaps=(GapPreparation(
            gap="未体现线上规模",
            jd_quote="Operate at scale.",
            honest_response_strategy="明确实际规模并说明迁移思路。",
        ),),
        questions_to_ask=(QuestionToAsk(
            question="团队如何衡量检索质量？",
            rationale="确认岗位的成功标准。",
        ),),
        checklist=("确认会议链接",),
        limitations=("没有招聘方确认的面试题。",),
    ))

    for heading in (
        "## 准备重点",
        "## 可准备的经历证据",
        "## 可能的问题",
        "## 需要诚实处理的差距",
        "## 可以反问面试官",
        "## 面试前检查",
        "## 信息限制",
    ):
        assert heading in rendered
    assert "失败时如何降级？" in rendered
    assert "application_id" not in rendered
    assert "source_notes" not in rendered
