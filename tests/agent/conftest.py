"""Shared mock-interview doubles.

The graph needs a worker and a source loader to reach any state at all, so every
test that drives a real graph needs both. They live here rather than in one test
module because `tests` is not a package: importing across test files raises
`ModuleNotFoundError` under a bare `pytest` invocation, while conftest is loaded
by path and stays importable.
"""

from __future__ import annotations

import pytest

from career_agent.agent.context.manager import ContextManager
from career_agent.agent.capabilities.catalog import CAPABILITIES
from career_agent.agent.capabilities.registry import MainAgentToolRegistry
from career_agent.agent.contracts.task_state import ConversationTaskState
from career_agent.agent.workflows.mock_interview.contracts import (
    MockInterviewFollowUpDecision,
    MockInterviewInputDecision,
    MockInterviewPlanDraft,
    MockInterviewQuestionDraft,
    MockInterviewReportDraft,
)
from career_agent.agent.workflows.mock_interview.graph import MockInterviewSources
from career_agent.agent.contracts.interview_preparation import InterviewPreparationContext
from career_agent.evaluation.trajectory import trajectory_tool_specs
from career_agent.domain.mock_interviews import (
    MockInterviewAnswerEvaluation,
    MockInterviewPlanItem,
    MockInterviewQuestionResult,
    MockInterviewScoreDimension,
)
from career_agent.storage.context import CareerContextStore
from career_agent.storage.resumes import StoredResumeDocument


_EXECUTION_TEST_MODULES = frozenset({
    "test_capture_source_turn", "test_job_capture_continuation",
    "test_main_agent_applications", "test_main_agent_email_tracking",
    "test_main_agent_interview_preparation", "test_main_agent_interviews",
    "test_main_agent_job_comparison", "test_main_agent_job_research",
    "test_main_agent_loop_evaluation", "test_main_agent_mock_interview",
    "test_main_agent_resume_tools", "test_main_agent_runtime",
    "test_resume_job_match", "test_trace_wiring_guard",
    "test_working_notes_guard",
})


@pytest.fixture(autouse=True)
def _isolate_runtime_execution_from_tool_discovery(request, monkeypatch):
    """Scripted execution tests receive their registered tools directly.

    Selection and reachability are exercised in their dedicated suites. These
    fixtures test execution, budgets, persistence, and presentation after a
    model has already chosen a tool.
    """
    if request.module.__name__.split(".")[-1] not in _EXECUTION_TEST_MODULES:
        return
    from career_agent.agent.capabilities import selection, selection_strategy
    from career_agent.agent.middleware import tool_availability

    monkeypatch.setattr(
        selection_strategy, "ALWAYS_OFFERED_TOOLS",
        tuple(name for name, descriptor in CAPABILITIES.items() if descriptor.model_callable),
    )
    if request.node.name == "test_internal_and_external_writes_draw_on_separate_budgets":
        fixture_reachable = lambda name, task: name not in {
            "execute_calendar_proposal", "get_calendar_proposal",
        }
    else:
        fixture_reachable = lambda name, task: True
    monkeypatch.setattr(selection, "reachable", fixture_reachable)
    monkeypatch.setattr(tool_availability, "reachable", fixture_reachable)


def enter_tool_profile(
    manager: ContextManager | CareerContextStore,
    profile: str,
    *,
    user_id: str = "u1",
    conversation_id: str = "c1",
) -> None:
    """Load a test fixture's named capability family before its scripted call."""

    store = manager._store if isinstance(manager, ContextManager) else manager
    task = store.get_task(user_id, conversation_id) or ConversationTaskState()
    store.upsert_task(
        user_id=user_id,
        conversation_id=conversation_id,
        task=task.add_loaded_capabilities(tuple(
            name for name, descriptor in CAPABILITIES.items()
            if descriptor.model_callable and descriptor.namespace is not None
            and descriptor.namespace.split(".", 1)[0] == profile
        )),
    )


class CatalogSchemaRegistry(MainAgentToolRegistry):
    """A registry double that offers every catalogue schema, as production does.

    Runtime tests fake handlers without wiring the services that install their
    schemas. Execution is limited to the schemas offered to the model, so a
    double with handlers but no schemas has every call refused as not offered.
    Offering the full catalogue lets the decide-time profile filter choose the
    offer exactly as a fully wired registry would.
    """

    def schemas(self) -> tuple[dict, ...]:
        return trajectory_tool_specs()


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
