from __future__ import annotations

from collections.abc import Callable
from typing import Any

from career_agent.agent.conversation_span_presenter import render_conversation_span
from career_agent.agent.daily_brief_presenter import render_daily_brief
from career_agent.agent.interview_preparation_presenter import (
    render_interview_preparation,
)
from career_agent.agent.interview_retro_presenter import (
    InterviewRetroView,
    render_interview_retro,
)
from career_agent.agent.job_analysis_contracts import JobAnalysisResult
from career_agent.agent.job_analysis_presenter import render_job_analysis
from career_agent.agent.job_comparison_presenter import render_job_comparison
from career_agent.agent.job_research_presenter import render_job_research
from career_agent.agent.main_agent_contracts import ConversationSpanView, ToolObservation
from career_agent.agent.main_agent_tools import MainAgentToolOutput
from career_agent.agent.mock_interview_contracts import (
    MockInterviewGraphResult,
    MockInterviewQuestionView,
    MockInterviewResultView,
)
from career_agent.agent.mock_interview_presenter import (
    render_mock_interview_question,
    render_mock_interview_result,
    render_mock_interview_turn,
)
from career_agent.agent.resume_job_match_contracts import ResumeJobMatchResult
from career_agent.agent.resume_job_match_presenter import render_resume_job_match
from career_agent.agent.resume_tailoring_contracts import ResumeTailoringResult
from career_agent.agent.resume_tailoring_presenter import (
    TailoringChangeReviewView,
    render_resume_tailoring,
)
from career_agent.domain.interview_preparation import InterviewPreparationResult
from career_agent.domain.job_comparison import JobComparison
from career_agent.domain.job_research import (
    JobResearchDraft,
    JobResearchFindingDraft,
    JobResearchSourceDraft,
)


DegradedReporter = Callable[..., None]


class ResultPresenter:
    """Parse capability payloads and dispatch them to domain presenters."""

    _MOCK_INTERVIEW_GRAPH_STATES = frozenset(
        {
            "mock_interview_answer_required",
            "mock_interview_running",
            "mock_interview_completed",
            "mock_interview_cancelled",
        }
    )

    @classmethod
    def present(
        cls,
        result: MainAgentToolOutput,
        *,
        report_degraded: DegradedReporter,
    ) -> str:
        if result.state == "saved_job_ready":
            snapshot = result.payload.get("jd_snapshot")
            if isinstance(snapshot, dict):
                content = snapshot.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
        if result.state in {"conversation_span_found", "conversation_span_empty"}:
            view = cls.validated(
                ConversationSpanView,
                result.payload,
                report_degraded=report_degraded,
            )
            if view is not None:
                return render_conversation_span(view)
        if result.state == "claim_source_found":
            source_quote = result.payload.get("source_quote")
            if isinstance(source_quote, str) and source_quote:
                claim_status = result.facts.get("claim_status")
                if claim_status == "superseded":
                    changed_at = result.facts.get("status_changed_at")
                    return (
                        "注意：这是已被更正声明的历史引文，不能作为当前声明的"
                        f"支持（状态变更时间：{changed_at or '未记录'}）。\n\n"
                        f"原始证据引文：\n\n{source_quote}"
                    )
                return f"原始证据引文：\n\n{source_quote}"
        if result.state in {
            "career_memory_detail_found",
            "career_memory_search_found",
            "career_episode_search_found",
            "career_history_found",
            "skill_loaded",
        }:
            body = result.payload.get("body")
            if isinstance(body, str) and body.strip():
                return body.strip()
        if result.state in cls._MOCK_INTERVIEW_GRAPH_STATES:
            graph_result = cls.mock_interview_result(
                result, report_degraded=report_degraded
            )
            if graph_result is not None:
                return render_mock_interview_turn(graph_result)
        if result.state == "mock_interview_result_found":
            view = cls.validated(
                MockInterviewResultView,
                result.payload,
                report_degraded=report_degraded,
            )
            if view is not None:
                return render_mock_interview_result(view)
        if result.state == "mock_interview_question_found":
            view = cls.validated(
                MockInterviewQuestionView,
                result.payload,
                report_degraded=report_degraded,
            )
            if view is not None:
                return render_mock_interview_question(view)
        if result.state == "daily_brief_ready":
            return render_daily_brief(result.payload)
        if result.state == "interview_retro_recorded":
            view = cls.validated(
                InterviewRetroView,
                result.payload,
                report_degraded=report_degraded,
            )
            if view is not None:
                return render_interview_retro(view)
        if result.state == "resume_job_match_ready":
            match = cls.resume_job_match_result(
                result, report_degraded=report_degraded
            )
            if match is not None:
                return render_resume_job_match(match)
        if result.state == "job_analysis_ready":
            analysis = cls.job_analysis_result(
                result, report_degraded=report_degraded
            )
            if analysis is not None:
                return render_job_analysis(analysis)
        if result.state == "resume_tailoring_draft_ready":
            rendered = cls._resume_tailoring_message(
                result, report_degraded=report_degraded
            )
            if rendered is not None:
                return rendered
        if result.state == "job_research_ready":
            research = cls.job_research_draft(result)
            if research is not None:
                return render_job_research(
                    research,
                    status=str(result.payload.get("status") or "current"),
                    user_provided_context=(
                        str(result.payload["user_provided_context"])
                        if result.payload.get("user_provided_context") is not None
                        else None
                    ),
                    anchored_by_other_job=bool(
                        result.payload.get("anchored_by_other_job")
                    ),
                )
        if result.state == "saved_jobs_compared":
            comparison = cls.job_comparison(result)
            if comparison is not None:
                return render_job_comparison(comparison)
        if result.state == "interview_preparation_ready":
            preparation = cls.interview_preparation_result(result)
            if preparation is not None:
                return render_interview_preparation(preparation)
        if result.state == "calendar_approval_required":
            return cls._calendar_approval_message(result)
        return result.message

    @staticmethod
    def validated(
        model: Any,
        payload: object,
        *,
        report_degraded: DegradedReporter,
    ) -> Any | None:
        try:
            return model.model_validate(payload)
        except ValueError as error:
            report_degraded(
                "presentation_degraded",
                getattr(model, "__name__", type(model).__name__ or "presenter"),
                error_detail="validation_error",
                details={
                    "errors": getattr(error, "errors", lambda: ())() and str(error),
                },
            )
            return None

    @classmethod
    def mock_interview_result(
        cls,
        result: MainAgentToolOutput,
        *,
        report_degraded: DegradedReporter,
    ) -> MockInterviewGraphResult | None:
        return cls.validated(
            MockInterviewGraphResult,
            result.payload,
            report_degraded=report_degraded,
        )

    @classmethod
    def resume_job_match_result(
        cls,
        result: MainAgentToolOutput,
        *,
        report_degraded: DegradedReporter,
    ) -> ResumeJobMatchResult | None:
        return cls.validated(
            ResumeJobMatchResult,
            {
                key: result.payload.get(key)
                for key in ResumeJobMatchResult.model_fields
            },
            report_degraded=report_degraded,
        )

    @classmethod
    def job_analysis_result(
        cls,
        result: MainAgentToolOutput,
        *,
        report_degraded: DegradedReporter,
    ) -> JobAnalysisResult | None:
        return cls.validated(
            JobAnalysisResult,
            {
                key: result.payload.get(key)
                for key in JobAnalysisResult.model_fields
            },
            report_degraded=report_degraded,
        )

    @classmethod
    def resume_tailoring_result(
        cls,
        result: MainAgentToolOutput,
        *,
        report_degraded: DegradedReporter,
    ) -> ResumeTailoringResult | None:
        changes = []
        for raw in result.payload.get("changes", ()):
            if not isinstance(raw, dict):
                continue
            changes.append(
                {key: value for key, value in raw.items() if key != "change_index"}
            )
        return cls.validated(
            ResumeTailoringResult,
            {
                "strategy_summary": result.payload.get("strategy_summary"),
                "changes": changes,
                "preserved_strengths": result.payload.get("preserved_strengths", ()),
                "unresolved_gaps": result.payload.get("unresolved_gaps", ()),
                "gap_mitigations": result.payload.get("gap_mitigations", ()),
                "clarification_questions": result.payload.get(
                    "clarification_questions", ()
                ),
                "warnings": result.payload.get("warnings", ()),
            },
            report_degraded=report_degraded,
        )

    @staticmethod
    def job_comparison(result: ToolObservation) -> JobComparison | None:
        raw = result.payload.get("comparison")
        if not isinstance(raw, dict):
            return None
        try:
            return JobComparison.model_validate(raw)
        except ValueError:
            return None

    @staticmethod
    def interview_preparation_result(
        result: ToolObservation,
    ) -> InterviewPreparationResult | None:
        raw = result.payload.get("preparation")
        if not isinstance(raw, dict):
            return None
        try:
            return InterviewPreparationResult.model_validate(raw)
        except ValueError:
            return None

    @staticmethod
    def job_research_draft(result: ToolObservation) -> JobResearchDraft | None:
        raw = result.payload.get("research")
        raw_sources = result.payload.get("sources")
        if not isinstance(raw, dict) or not isinstance(raw_sources, list):
            return None
        try:
            sources = tuple(
                JobResearchSourceDraft(
                    source_key=item["source_key"],
                    url=item["url"],
                    title=item["title"],
                    publisher=item.get("publisher"),
                    published_at=item.get("published_at"),
                    relevant_excerpt=item["relevant_excerpt"],
                )
                for item in raw_sources
                if isinstance(item, dict)
            )
            findings = tuple(
                JobResearchFindingDraft.model_validate(item)
                for item in raw.get("findings", ())
            )
            return JobResearchDraft(
                summary=raw["summary"],
                sources=sources,
                findings=findings,
                open_questions=tuple(raw.get("open_questions", ())),
                limitations=tuple(raw.get("limitations", ())),
            )
        except (KeyError, TypeError, ValueError):
            return None

    @classmethod
    def _resume_tailoring_message(
        cls,
        result: MainAgentToolOutput,
        *,
        report_degraded: DegradedReporter,
    ) -> str | None:
        tailoring = cls.resume_tailoring_result(
            result, report_degraded=report_degraded
        )
        if tailoring is None:
            return None
        reviews = tuple(
            review
            for raw in result.payload.get("change_reviews", ())
            if isinstance(raw, dict)
            and (
                review := cls.validated(
                    TailoringChangeReviewView,
                    raw,
                    report_degraded=report_degraded,
                )
            )
            is not None
        )
        raw_status = str(result.payload.get("status") or "pending")
        status = (
            raw_status
            if raw_status
            in {
                "pending",
                "in_review",
                "reviewed",
                "finalized",
                "superseded",
                "expired",
            }
            else "pending"
        )
        raw_revision = result.payload.get("revision_number", 1)
        revision_number = (
            raw_revision
            if isinstance(raw_revision, int) and raw_revision >= 1
            else 1
        )
        return render_resume_tailoring(
            tailoring,
            status=status,
            revision_number=revision_number,
            change_reviews=reviews,
        )

    @staticmethod
    def _calendar_approval_message(result: MainAgentToolOutput) -> str:
        payload = result.payload.get("payload")
        if isinstance(payload, dict):
            return (
                "请确认是否执行以下 Calendar 变更：\n"
                f"- 操作：{result.payload.get('operation')}\n"
                f"- 标题：{payload.get('title')}\n"
                f"- 开始：{payload.get('start_at')}\n"
                f"- 结束：{payload.get('end_at')}\n"
                f"- 时区：{payload.get('timezone')}\n"
                f"- 地点：{payload.get('location') or '未提供'}\n"
                f"- 预览失效时间：{result.payload.get('expires_at')}\n"
                "只有你明确确认后才会写入外部 Calendar。"
            )
        return (
            "请确认是否取消这条 Calendar 事件。"
            f"预览失效时间：{result.payload.get('expires_at')}。"
        )
