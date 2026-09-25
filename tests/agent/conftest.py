"""Shared mock-interview doubles.

The graph needs a worker and a source loader to reach any state at all, so every
test that drives a real graph needs both. They live here rather than in one test
module because `tests` is not a package: importing across test files raises
`ModuleNotFoundError` under a bare `pytest` invocation, while conftest is loaded
by path and stays importable.
"""

from __future__ import annotations

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import (
    ConversationTaskState,
    ToolProfile,
)
from career_agent.agent.mock_interview_contracts import (
    MockInterviewFollowUpDecision,
    MockInterviewInputDecision,
    MockInterviewPlanDraft,
    MockInterviewQuestionDraft,
    MockInterviewReportDraft,
)
from career_agent.agent.mock_interview_graph import MockInterviewSources
from career_agent.agent.interview_preparation_contracts import InterviewPreparationContext
from career_agent.domain.mock_interviews import (
    MockInterviewAnswerEvaluation,
    MockInterviewPlanItem,
    MockInterviewQuestionResult,
    MockInterviewScoreDimension,
)
from career_agent.storage.context import CareerContextStore
from career_agent.storage.resumes import StoredResumeDocument


def enter_tool_profile(
    manager: ContextManager | CareerContextStore,
    profile: ToolProfile,
    *,
    user_id: str = "u1",
    conversation_id: str = "c1",
) -> None:
    """Persist the conversation's tool profile as if the model had routed there.

    Domain tools are refused outside their profile, so a test whose decision
    maker starts straight at a domain tool declares the profile up front instead
    of spending a decision on ``route_to_capability``. Only the profile changes;
    any other seeded task state is kept.
    """

    store = manager._store if isinstance(manager, ContextManager) else manager
    task = store.get_task(user_id, conversation_id) or ConversationTaskState()
    store.upsert_task(
        user_id=user_id,
        conversation_id=conversation_id,
        task=task.model_copy(update={"tool_profile": profile}),
    )


def evaluation(
    rating: str, next_action: str = "next_question"
) -> MockInterviewAnswerEvaluation:
    return MockInterviewAnswerEvaluation(
        rating=rating,
        summary=f"{rating} 的回答。",
        dimensions=(
            MockInterviewScoreDimension(
                dimension="specificity", score=3, feedback="还行"
            ),
        ),
        next_action=next_action,
        next_action_reason="够了" if next_action == "next_question" else "要追问",
        follow_up_question="再具体点？" if next_action == "follow_up" else None,
    )


class FixedSources:
    """Fixed resume and JD, so the graph has something to plan against."""

    def load(self, *, session):
        return MockInterviewSources(
            document=StoredResumeDocument(
                resume_version_id=session.resume_version_id,
                document_format="markdown",
                raw_bytes="# 简历\n负责检索系统。".encode(),
            ),
            context=InterviewPreparationContext(
                jd_text="招 RAG 工程师。",
                company_name="示例科技",
                role_title="RAG 工程师",
            ),
        )


class OneQuestionWorker:
    """One planned question, no follow-up, one score, one report: enough to finish."""

    def route_input(self, **kwargs):
        return MockInterviewInputDecision(action="answer")

    def plan(
        self,
        *,
        session,
        document,
        context,
    ):
        return MockInterviewPlanDraft(
            summary="一题",
            items=(
                MockInterviewPlanItem(
                    sequence_number=1,
                    question_type="project_deep_dive",
                    difficulty="intermediate",
                    focus="检索可靠性",
                    rationale="JD 要求",
                    jd_quotes=("RAG",),
                    resume_locators=("简历",),
                    resume_quotes=("负责检索系统",),
                    question="介绍一个你负责的检索改进。",
                ),
            ),
        )

    def ask(
        self,
        *,
        session,
        plan,
        plan_item,
        prior_turns=(),
        document,
        jd_text,
        company_name="",
        role_title="",
        confirmed_facts=(),
    ):
        return MockInterviewQuestionDraft(question="介绍一个你负责的检索改进。")

    def decide_follow_up(
        self,
        *,
        session,
        plan_item,
        turns,
        follow_ups_remaining,
        document,
        jd_text,
    ):
        return MockInterviewFollowUpDecision(next_action="next_question")

    def evaluate(
        self,
        *,
        session,
        plan_item,
        turns,
        document,
        jd_text,
        confirmed_facts=(),
    ):
        return evaluation("adequate")

    def report(
        self,
        *,
        session,
        plan,
        turns,
        completion_reason,
        document,
        jd_text,
        company_name="",
        role_title="",
        confirmed_facts=(),
    ):
        return MockInterviewReportDraft(
            summary="完成了限定的练习计划。",
            question_results=(
                MockInterviewQuestionResult(
                    plan_item_number=1,
                    question="介绍一个你负责的检索改进。",
                    final_rating="adequate",
                    summary="还行",
                    follow_up_count=0,
                ),
            ),
        )
