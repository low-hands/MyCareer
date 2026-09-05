from __future__ import annotations

from datetime import datetime, timezone

import pytest

from career_agent.agent.interview_preparation_presenter import (
    summarize_interview_preparation,
)
from career_agent.agent.job_research_presenter import summarize_job_research
from career_agent.agent.mock_interview_contracts import MockInterviewGraphResult
from career_agent.agent.mock_interview_presenter import (
    render_mock_interview_report,
    render_mock_interview_turn,
    summarize_mock_interview_report,
)
from career_agent.agent.summary_text import SUMMARY_LIMIT, condense
from career_agent.domain.interview_preparation import (
    InterviewFocusArea,
    InterviewPreparationResult,
    LikelyQuestion,
)
from career_agent.domain.mock_interviews import (
    MockInterviewAnswerEvaluation,
    MockInterviewQuestionResult,
    MockInterviewReport,
    MockInterviewScoreDimension,
)

NOW = datetime(2026, 8, 31, tzinfo=timezone.utc)


def _report(*, summary: str = "整体表现稳定。") -> MockInterviewReport:
    return MockInterviewReport(
        id="report-1",
        session_id="session-1",
        completion_reason="plan_completed",
        summary=summary,
        question_results=(
            MockInterviewQuestionResult(
                plan_item_number=1,
                question="讲一个你做过的检索评估。",
                final_rating="adequate",
                summary="结构清晰，缺量化。",
                follow_up_count=1,
            ),
        ),
        strengths=("表达清晰",),
        development_areas=("量化不足",),
        practice_actions=("补一组指标",),
        created_at=NOW,
    )


def _graph_result(**overrides) -> MockInterviewGraphResult:
    payload = {
        "session_id": "session-1",
        "state": "running",
        "message": "Mock interview is processing the current turn.",
    }
    payload.update(overrides)
    return MockInterviewGraphResult.model_validate(payload)


class TestCondense:
    def test_keeps_a_short_first_line_verbatim(self) -> None:
        assert condense("整体表现稳定。\n更多细节在后面。") == "整体表现稳定。"

    def test_bounds_a_long_line_so_the_recent_window_stays_small(self) -> None:
        """A report's summary can run to thousands of characters.

        The condensed line is carried into every later turn's recent window, so
        an unbounded first line would defeat the point of not storing the report.
        """
        condensed = condense("回" * 4000)
        assert len(condensed) == SUMMARY_LIMIT
        assert condensed.endswith("…")

    def test_an_empty_summary_produces_nothing_rather_than_whitespace(self) -> None:
        assert condense("   \n  ") == ""


class TestMockInterviewTurn:
    def test_a_finished_run_renders_the_full_report(self) -> None:
        rendered = render_mock_interview_turn(
            _graph_result(state="completed", report=_report(), report_id="report-1")
        )
        assert rendered.startswith("模拟面试完成。")
        assert "量化不足" in rendered
        assert "## 每题反馈" in rendered
        assert "讲一个你做过的检索评估。" in rendered

    def test_a_finished_run_without_a_report_still_states_it_is_over(self) -> None:
        """The session is closed either way.

        Falling through to the graph's ``message`` here would put internal
        English status text on the candidate's screen.
        """
        assert render_mock_interview_turn(_graph_result(state="completed")) == (
            "模拟面试完成。"
        )

    def test_model_authored_question_feedback_cannot_add_markdown(self) -> None:
        report = _report().model_copy(
            update={
                "question_results": (
                    _report().question_results[0].model_copy(
                        update={
                            "question": "[打开](javascript:alert(1))",
                            "summary": "<script>alert(1)</script>\n# 注入标题",
                        }
                    ),
                )
            }
        )

        rendered = render_mock_interview_report(report)

        assert r"\[打开\](javascript:alert(1))" in rendered
        assert "&lt;script>alert(1)&lt;/script>" in rendered
        assert r"\# 注入标题" in rendered

    def test_all_model_authored_report_sections_are_escaped_without_noise(self) -> None:
        report = _report(summary="# 总结\nC++ 提升 1.5 倍").model_copy(
            update={
                "strengths": ("- 亮点",),
                "development_areas": ("[待提升](javascript:alert(1))",),
                "practice_actions": ("> 练习建议",),
            }
        )

        rendered = render_mock_interview_report(report)

        assert r"\# 总结" in rendered
        assert "C++ 提升 1.5 倍" in rendered
        assert "- - 亮点" in rendered
        assert r"\[待提升\](javascript:alert(1))" in rendered
        assert r"- \> 练习建议" in rendered

    def test_a_cancelled_run_says_so(self) -> None:
        assert render_mock_interview_turn(_graph_result(state="cancelled")) == (
            "模拟面试已取消。"
        )

    def test_a_waiting_run_leads_with_the_previous_answer_s_feedback(self) -> None:
        rendered = render_mock_interview_turn(
            _graph_result(
                state="awaiting_answer",
                question="再讲一个线上故障。",
                turn_id="turn-2",
                evaluation=MockInterviewAnswerEvaluation(
                    rating="adequate",
                    summary="结构清晰，缺量化。",
                    dimensions=(
                        MockInterviewScoreDimension(
                            dimension="specificity",
                            score=3,
                            feedback="缺少指标。",
                        ),
                    ),
                    next_action="next_question",
                    next_action_reason="这一项已覆盖。",
                ),
            )
        )
        assert rendered.index("上一题反馈") < rendered.index("模拟面试题")
        assert "再讲一个线上故障。" in rendered

    def test_a_waiting_run_uses_restricted_markdown_for_model_text(self) -> None:
        rendered = render_mock_interview_turn(
            _graph_result(
                state="awaiting_answer",
                question=(
                    "## 题目\n> 不应成为引用块\n请比较 `recall` 和 **precision**，"
                    "不要点 [链接](https://evil.example) 或 https://evil.example"
                ),
                turn_id="turn-2",
                evaluation=MockInterviewAnswerEvaluation(
                    rating="adequate",
                    summary="# 上一题反馈",
                    dimensions=(
                        MockInterviewScoreDimension(
                            dimension="specificity", score=3, feedback="具体"
                        ),
                    ),
                    next_action="next_question",
                    next_action_reason="见 www.evil.example",
                ),
            )
        )

        assert r"\## 题目" in rendered
        assert r"\> 不应成为引用块" in rendered
        assert "`recall`" in rendered
        assert "**precision**" in rendered
        assert r"\[链接\]" in rendered
        assert "https://evil.example" not in rendered
        assert "www.evil.example" not in rendered

    def test_the_first_question_has_no_feedback_block(self) -> None:
        rendered = render_mock_interview_turn(
            _graph_result(
                state="awaiting_answer",
                question="先自我介绍。",
                turn_id="turn-1",
            )
        )
        assert "上一题反馈" not in rendered
        assert rendered == "模拟面试题：\n先自我介绍。"

    def test_an_in_flight_run_falls_back_to_the_graph_s_own_status(self) -> None:
        assert render_mock_interview_turn(_graph_result()) == (
            "Mock interview is processing the current turn."
        )


class TestDurableRowSummaries:
    """The line each report-producing turn leaves in the transcript.

    Every one of these is deterministic. The case they exist for is the answer
    writer not running — a failed or absent model call — so a writer cannot be
    the thing that produces them.
    """

    def test_a_mock_interview_row_is_bounded_by_the_headline(self) -> None:
        summary = summarize_mock_interview_report(_report(summary="回" * 3000))
        assert summary.startswith("模拟面试完成。")
        assert len(summary) <= len("模拟面试完成。") + SUMMARY_LIMIT

    def test_a_reused_research_row_says_it_was_reused(self) -> None:
        assert summarize_job_research("字节跳动在做基础模型。", cached=True) == (
            "已复用仍在有效期内的岗位研究报告。字节跳动在做基础模型。"
        )

    def test_a_fresh_research_row_does_not(self) -> None:
        assert summarize_job_research("字节跳动在做基础模型。", cached=False) == (
            "岗位研究已完成。字节跳动在做基础模型。"
        )

    def test_a_preparation_row_carries_the_counts_the_body_would_show(self) -> None:
        result = InterviewPreparationResult(
            summary="重点准备检索评估。",
            focus_areas=(
                InterviewFocusArea(
                    topic="检索评估",
                    priority="high",
                    rationale="JD 明确要求。",
                    jd_quote="负责检索质量评估",
                ),
            ),
            likely_questions=(
                LikelyQuestion(
                    question="你如何度量召回率？",
                    rationale="直接对应 JD。",
                ),
            ),
        )
        assert summarize_interview_preparation(result) == (
            "面试准备材料已生成。重点准备检索评估。（1 个可能问题，1 个准备重点）"
        )


def test_the_presenter_is_the_only_place_that_names_a_run_s_states() -> None:
    """Guards the consolidation this module exists for.

    The terminal copy and the awaiting-answer assembly used to live in the
    generic tool layer, and the runtime's dispatch re-parsed the serialized
    payload back into ``MockInterviewGraphResult`` just to reach the report
    renderer. Both are now one call into this presenter; if that copy migrates
    back out, this fails.
    """
    import inspect

    from career_agent.agent import main_agent_tools

    source = inspect.getsource(main_agent_tools)
    assert "模拟面试已取消。" not in source
    assert "模拟面试题：" not in source
    assert "上一题反馈" not in source


@pytest.mark.parametrize(
    "state",
    ["awaiting_answer", "running", "completed", "cancelled"],
)
def test_every_state_the_graph_can_return_renders_something(state: str) -> None:
    """No state can reach the screen empty.

    The graph's ``state`` is a closed literal, so this is the whole surface the
    presenter has to cover.
    """
    result = _graph_result(
        state=state,
        question="讲一个项目。" if state == "awaiting_answer" else None,
        turn_id="turn-1" if state == "awaiting_answer" else None,
        report=_report() if state == "completed" else None,
        report_id="report-1" if state == "completed" else None,
    )
    assert render_mock_interview_turn(result).strip()
