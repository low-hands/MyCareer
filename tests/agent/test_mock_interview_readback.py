"""Reading back a finished mock interview.

The run's own exchanges never enter the conversation, and the turn that closed it
stores one line plus a reference rather than the report, so this tool is the only
path from the main thread back to what was actually asked and answered.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from career_agent.agent.main_agent_contracts import (
    ApplicationCandidateContextItem,
    CareerProfileContext,
    ConversationTaskState,
    MainAgentContext,
    project_mock_interview_result_arguments,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.agent.mock_interview_contracts import MockInterviewStartRequest
from career_agent.agent.mock_interview_graph import MockInterviewGraph
from career_agent.domain.mock_interviews import (
    MockInterviewPlan,
    MockInterviewPlanItem,
    MockInterviewQuestionResult,
    MockInterviewReport,
)
from career_agent.storage.mock_interviews import SQLiteMockInterviewStore
from conftest import FixedSources, OneQuestionWorker, evaluation

NOW = datetime(2026, 8, 29, tzinfo=timezone.utc)


def _screen(observation):
    """What the readback puts on the screen, as the runtime renders it.

    Separate from ``observation.message``, which is the bounded line the
    transcript keeps. The two used to be one string; asserting content here
    and boundedness there is what keeps them from collapsing back together.
    """
    return MainAgentRuntime._assistant_message(observation)


def _planned_session(store, *, interview_type="technical"):
    """A started session with a two-question plan behind it."""
    session = store.create_session(
        user_id="u1",
        application_id="app-1",
        job_posting_id="job-1",
        jd_snapshot_id="jd-1",
        resume_version_id="rv-1",
        interview_type=interview_type,
    )
    store.save_plan(
        session=session,
        plan=MockInterviewPlan(
            session_id=session.id,
            summary="两题",
            items=tuple(
                MockInterviewPlanItem(
                    sequence_number=number,
                    question_type=question_type,
                    difficulty="intermediate",
                    focus=f"focus {number}",
                    rationale="r",
                    jd_quotes=("q",),
                    resume_locators=("l",),
                    resume_quotes=("rq",),
                )
                for number, question_type in (
                    (1, "project_deep_dive"),
                    (2, "system_design"),
                )
            ),
            created_at=NOW,
        ),
    )
    return store.start(session=session)


def _completed_store(tmp_path: Path) -> SQLiteMockInterviewStore:
    """Two answered questions, the first of which drew one follow-up."""
    store = SQLiteMockInterviewStore(tmp_path / "mock.sqlite3")
    session = _planned_session(store)
    session = _answer_first_question(store, session)
    session = _answer_second_question(store, session)
    return _complete(store, session)


def _answer_first_question(store, session):
    session, turn = store.ask(
        session=session,
        plan_item_number=1,
        question_type="project_deep_dive",
        question="介绍一个你负责的 RAG 可靠性改进。",
    )
    session, turn = store.record_answer(
        session=session,
        turn=turn,
        answer="我负责设计离线评估集，召回从 61% 到 78%。",
    )
    session, turn = store.record_evaluation(
        session=session, turn=turn, evaluation=evaluation("adequate", "follow_up")
    )
    session, follow_up = store.ask(
        session=session,
        plan_item_number=1,
        question_type="project_deep_dive",
        question="评估集怎么标注的？",
        turn_type="follow_up",
        parent_turn_id=turn.id,
    )
    session, follow_up = store.record_answer(
        session=session, turn=follow_up, answer="人工标 300 条，双人交叉。"
    )
    session, _ = store.record_evaluation(
        session=session, turn=follow_up, evaluation=evaluation("strong")
    )
    return session


def _answer_second_question(store, session):
    session, turn = store.ask(
        session=session,
        plan_item_number=2,
        question_type="system_design",
        question="怎么设计检索层的降级？",
    )
    session, turn = store.record_answer(
        session=session, turn=turn, answer="双路召回加超时熔断。"
    )
    session, _ = store.record_evaluation(
        session=session, turn=turn, evaluation=evaluation("weak")
    )
    return session


def _complete(store, session, *, report_id="rep-1", summary="项目深度可以，系统设计偏弱。"):
    store.complete(
        session=session,
        report=MockInterviewReport(
            id=report_id,
            session_id=session.id,
            completion_reason="plan_completed",
            summary=summary,
            question_results=(
                MockInterviewQuestionResult(
                    plan_item_number=1,
                    question="q1",
                    final_rating="strong",
                    summary="好",
                    follow_up_count=1,
                ),
                MockInterviewQuestionResult(
                    plan_item_number=2,
                    question="q2",
                    final_rating="weak",
                    summary="弱",
                    follow_up_count=0,
                ),
            ),
            strengths=("能讲清个人职责",),
            development_areas=("补降级细节",),
            practice_actions=("重写系统设计回答",),
            created_at=NOW,
        ),
    )
    return store


def _invoke(store, **arguments):
    tools = MainAgentToolRegistry(mock_interview_store=store)
    return tools.invoke_atomic_tool(
        "get_mock_interview_result",
        {"user_id": "u1", "application_id": "app-1", **arguments},
    )


def test_the_tool_is_offered_only_when_a_store_is_configured(tmp_path) -> None:
    def names(registry):
        return {schema["function"]["name"] for schema in registry.schemas()}

    assert "get_mock_interview_result" not in names(MainAgentToolRegistry())
    assert "get_mock_interview_result" in names(
        MainAgentToolRegistry(mock_interview_store=_completed_store(tmp_path))
    )


def test_the_default_read_indexes_the_questions_without_their_answers(tmp_path) -> None:
    """An index, not a transcript: a long run costs one short message."""
    observation = _invoke(_completed_store(tmp_path), question_number=None)

    assert observation.state == "mock_interview_result_found"
    screen = _screen(observation)
    assert "1. [adequate，追问 1 次]" in screen
    assert "2. [weak]" in screen
    assert "项目深度可以" in screen
    # The answers stay out until a question is named.
    assert "离线评估集" not in screen
    assert "双路召回" not in screen
    # The row keeps a line and a way back to the report, not the index.
    assert "1. [" not in observation.message
    assert observation.resource_ref is not None
    assert observation.resource_ref.resource_id == "rep-1"


def test_naming_a_question_returns_it_with_its_follow_ups(tmp_path) -> None:
    observation = _invoke(_completed_store(tmp_path), question_number=1)

    assert observation.state == "mock_interview_question_found"
    screen = _screen(observation)
    assert "我负责设计离线评估集，召回从 61% 到 78%。" in screen
    # The follow-up belongs to the same question, so it comes back with it.
    assert "评估集怎么标注的？" in screen
    assert "人工标 300 条，双人交叉。" in screen
    # Another question's answer does not.
    assert "双路召回" not in screen
    # An answer runs to 20k characters, so none of it reaches the row.
    assert "我负责设计离线评估集" not in observation.message
    assert "第 1 题" in observation.message


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ({"question_number": 9}, "没有第 9 题"),
        ({"application_id": "app-9"}, "还没有结束过的模拟面试"),
    ],
)
def test_a_missing_result_says_so_instead_of_failing(
    tmp_path, arguments, expected
) -> None:
    observation = _invoke(_completed_store(tmp_path), **arguments)

    assert observation.state == "no_mock_interview_result_found"
    assert expected in observation.message


def test_a_cancelled_run_is_readable_and_says_it_stopped_early(tmp_path) -> None:
    """Cancelling keeps the turns, so what was answered is still readable.

    Only the report is missing, and the count has to read as what the run got
    through rather than as a finished interview.
    """
    store = SQLiteMockInterviewStore(tmp_path / "mock.sqlite3")
    session = store.create_session(
        user_id="u1",
        application_id="app-1",
        job_posting_id="job-1",
        jd_snapshot_id="jd-1",
        resume_version_id="rv-1",
        interview_type="technical",
    )
    store.save_plan(
        session=session,
        plan=MockInterviewPlan(
            session_id=session.id,
            summary="两题",
            items=(
                MockInterviewPlanItem(
                    sequence_number=1,
                    question_type="project_deep_dive",
                    difficulty="intermediate",
                    focus="focus 1",
                    rationale="r",
                    jd_quotes=("q",),
                    resume_locators=("l",),
                    resume_quotes=("rq",),
                ),
            ),
            created_at=NOW,
        ),
    )
    session = store.start(session=session)
    session = _answer_first_question(store, session)
    store.cancel(session=session)

    observation = _invoke(store, question_number=None)
    assert observation.state == "mock_interview_result_found"
    assert "1 题" in observation.message
    assert "中途取消" in observation.message
    # Asked and answered agree here, so no second count is claimed.
    assert "答了" not in observation.message
    # No report was written, so there is nothing for the row to point at.
    assert observation.resource_ref is None
    # The answer is still reachable by number.
    assert "离线评估集" in _screen(_invoke(store, question_number=1))


def test_a_question_abandoned_before_answering_is_not_counted_as_answered(
    tmp_path,
) -> None:
    """Asked and answered are different numbers once a run can stop early."""
    store = SQLiteMockInterviewStore(tmp_path / "mock.sqlite3")
    graph = MockInterviewGraph(store=store, worker=OneQuestionWorker(), sources=FixedSources())
    graph.start(
        MockInterviewStartRequest(
            user_id="u1",
            application_id="app-1",
            job_posting_id="job-1",
            jd_snapshot_id="jd-1",
            resume_version_id="rv-1",
            interview_type="technical",
            max_primary_questions=1,
            max_follow_ups_per_question=0,
        )
    )
    session = store.find_resumable(user_id="u1")
    graph.cancel(user_id="u1", session_id=session.id)

    observation = _invoke(store, question_number=None)
    assert "1 题，答了 0 题" in observation.message
    # Nothing to go back and read, which "未评价" would not convey.
    assert "[未回答]" in _screen(observation)


def test_a_run_the_real_graph_finished_is_readable_through_the_tool(tmp_path) -> None:
    """The tool reads the same file the workflow wrote, not a parallel copy.

    The other tests here build rows directly, so they would still pass if the
    tool and the graph disagreed about where turns live.
    """
    store = SQLiteMockInterviewStore(tmp_path / "mock.sqlite3")
    graph = MockInterviewGraph(store=store, worker=OneQuestionWorker(), sources=FixedSources())
    first = graph.start(
        MockInterviewStartRequest(
            user_id="u1",
            application_id="app-1",
            job_posting_id="job-1",
            jd_snapshot_id="jd-1",
            resume_version_id="rv-1",
            interview_type="technical",
            max_primary_questions=1,
            max_follow_ups_per_question=0,
        )
    )
    result = graph.resume(
        user_id="u1", session_id=first.session_id, answer="我负责离线评估集的设计。"
    )
    assert result.state == "completed"

    session = store.get_session(user_id="u1", session_id=first.session_id)
    tools = MainAgentToolRegistry(mock_interview_store=store)

    def read(question_number):
        return tools.invoke_atomic_tool(
            "get_mock_interview_result",
            {
                "user_id": "u1",
                "application_id": session.application_id,
                "question_number": question_number,
            },
        )

    index = read(None)
    assert index.state == "mock_interview_result_found"
    assert "1. [" in _screen(index)
    assert "我负责离线评估集的设计。" in _screen(read(1))


def test_projection_resolves_the_application_by_index_and_rejects_internal_ids() -> None:
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        task=ConversationTaskState(
            active_application_id="app-1",
            application_candidates=(
                ApplicationCandidateContextItem(
                    application_id="app-2",
                    title="RAG Engineer",
                    company_name="Acme",
                    status="submitted",
                ),
            ),
        ),
        user_message="我上次模拟面试第一题答得咋样",
    )

    # No selector falls back to the application the conversation is already on.
    assert (
        project_mock_interview_result_arguments(context, {})["application_id"] == "app-1"
    )
    projected = project_mock_interview_result_arguments(
        context, {"application_selection_index": 1, "question_number": 2}
    )
    assert projected["application_id"] == "app-2"
    assert projected["question_number"] == 2

    with pytest.raises(ValueError, match="internal identifiers"):
        project_mock_interview_result_arguments(context, {"application_id": "app-2"})
    with pytest.raises(ValueError, match="out of range"):
        project_mock_interview_result_arguments(
            context, {"application_selection_index": 9}
        )


def test_a_report_id_reads_that_exact_run_even_when_a_later_one_exists(
    tmp_path,
) -> None:
    """A reference has to survive a second interview on the same application.

    The application path deliberately returns the newest finished run, so once
    the user practises twice it stops naming the run an earlier turn reported.
    Reading by report id is what keeps that earlier reference answerable.
    """
    store = _completed_store(tmp_path)
    second = _planned_session(store, interview_type="behavioral")
    second = _answer_first_question(store, second)
    second = _answer_second_question(store, second)
    _complete(
        store,
        second,
        report_id="rep-2",
        summary="第二次练习，系统设计有进步。",
    )
    tools = MainAgentToolRegistry(mock_interview_store=store)

    def read(**arguments):
        return tools.invoke_atomic_tool(
            "get_mock_interview_result", {"user_id": "u1", **arguments}
        )

    # Without a report id the newest run wins, which is why the id is needed.
    assert "第二次练习" in read(application_id="app-1").message
    older = read(report_id="rep-1")
    assert older.state == "mock_interview_result_found"
    assert "项目深度可以" in _screen(older)
    assert "第二次练习" not in _screen(older)
    # The row points at the run that was named, not the newest one.
    assert older.resource_ref.resource_id == "rep-1"
    # The run's turns come from the referenced session, not the newest one.
    assert "离线评估集" in _screen(read(report_id="rep-1", question_number=1))


def test_a_report_belonging_to_another_user_is_not_readable(tmp_path) -> None:
    """The id alone is not authorization: the owner filter is the check."""
    store = _completed_store(tmp_path)
    tools = MainAgentToolRegistry(mock_interview_store=store)

    observation = tools.invoke_atomic_tool(
        "get_mock_interview_result", {"user_id": "u2", "report_id": "rep-1"}
    )

    assert observation.state == "no_mock_interview_result_found"
    assert "项目深度可以" not in observation.message


def test_an_unknown_report_id_says_so_instead_of_failing(tmp_path) -> None:
    tools = MainAgentToolRegistry(mock_interview_store=_completed_store(tmp_path))

    observation = tools.invoke_atomic_tool(
        "get_mock_interview_result", {"user_id": "u1", "report_id": "rep-missing"}
    )

    assert observation.state == "no_mock_interview_result_found"
