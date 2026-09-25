"""A mock interview's exchanges appear in the transcript, not in the model's context.

The run's turns were kept out of the stored conversation so they would not
crowd Main Agent's context, but that also hid them from the reader: answered
questions showed as empty bubbles, a reload lost the whole exchange, and a run
that failed before its closing reply left nothing at all.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from career_agent.agent.context_manager import ContextManager
from career_agent.api.reads import WorkspaceReader
from career_agent.domain.mock_interviews import MockInterviewPlan, MockInterviewPlanItem
from career_agent.storage.context import CareerContextStore
from career_agent.storage.mock_interviews import SQLiteMockInterviewStore
from tests.api.test_clear_applications import _paths

# The opening question says what it is based on; these runs use no resume.
NO_RESUME = "本场不参考简历，只考察通用题和专业基础。"


def _plan(session_id: str) -> MockInterviewPlan:
    return MockInterviewPlan(
        session_id=session_id,
        summary="行为面",
        items=(
            MockInterviewPlanItem(
                sequence_number=1,
                question_type="behavioral",
                difficulty="intermediate",
                focus="跨团队协作",
                rationale="考察推动能力",
                question="讲一次你推动跨团队交付的经历。",
            ),
        ),
        created_at=datetime.now(timezone.utc),
    )


def test_a_running_interview_shows_its_request_exchanges_and_open_question(tmp_path) -> None:
    paths = _paths(tmp_path)
    manager = ContextManager(CareerContextStore(Path(paths.context_store)))
    context = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="我想做一场行为面模拟"
    )
    # As in a real turn: the run plans and asks its first question while the
    # turn that started it is still executing, and only then is the request held.
    mock = SQLiteMockInterviewStore(Path(paths.mock_interview_store))
    session = mock.create_session(user_id="u1", interview_type="behavioral", conversation_id="c1")
    mock.save_plan(session=session, plan=_plan(session.id))
    session = mock.start(session=session)
    session, primary = mock.ask(
        session=session, plan_item_number=1, question_type="behavioral",
        question="讲一次你推动跨团队交付的经历。",
    )
    manager.commit_workflow_entry(
        context=context,
        task=context.task.enter_workflow("mock_interview", run_id="run", phase="awaiting_answer"),
    )
    session, primary = mock.record_answer(session=session, turn=primary, answer="我拉齐了工程和设计。")
    session = mock.settle_answer(session=session, turn=primary)
    mock.ask(
        session=session, plan_item_number=1, question_type="behavioral",
        question="你怎么判断优先级？", turn_type="follow_up", parent_turn_id=primary.id,
    )

    transcript = WorkspaceReader(paths).conversation_messages(user_id="u1", conversation_id="c1")

    assert [(message.role, message.content) for message in transcript.messages] == [
        ("user", "我想做一场行为面模拟"),
        ("assistant", f"{NO_RESUME}\n\n讲一次你推动跨团队交付的经历。"),
        ("user", "我拉齐了工程和设计。"),
        # Waiting for its answer; after a reload nothing else would show it.
        ("assistant", "你怎么判断优先级？"),
    ]
    # The model's view of the conversation is unchanged: none of it is there.
    reloaded = manager.load_for_turn(user_id="u1", conversation_id="c1", user_message="下一步")
    assert all("跨团队交付" not in message.content for message in reloaded.recent_messages)


def test_a_finished_interview_reads_request_exchanges_then_report(tmp_path) -> None:
    paths = _paths(tmp_path)
    manager = ContextManager(CareerContextStore(Path(paths.context_store)))
    context = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="我想做一场行为面模拟"
    )
    held = manager.commit_workflow_entry(
        context=context,
        task=context.task.enter_workflow("mock_interview", run_id="run", phase="awaiting_answer"),
    )
    mock = SQLiteMockInterviewStore(Path(paths.mock_interview_store))
    session = mock.create_session(user_id="u1", interview_type="behavioral", conversation_id="c1")
    mock.save_plan(session=session, plan=_plan(session.id))
    session = mock.start(session=session)
    session, turn = mock.ask(
        session=session, plan_item_number=1, question_type="behavioral",
        question="讲一次你推动跨团队交付的经历。",
    )
    session, turn = mock.record_answer(session=session, turn=turn, answer="我拉齐了工程和设计。")
    mock.settle_answer(session=session, turn=turn)

    # The run ends: the held request and the closing reply are written together.
    closing = manager.load_for_workflow_turn(user_id="u1", conversation_id="c1", task=held)
    manager.commit_workflow_exit(
        context=closing, task=held.leave_workflow(), assistant_message="模拟面试完成，报告见卡片。"
    )

    transcript = WorkspaceReader(paths).conversation_messages(user_id="u1", conversation_id="c1")

    assert [message.content for message in transcript.messages] == [
        "我想做一场行为面模拟",
        f"{NO_RESUME}\n\n讲一次你推动跨团队交付的经历。",
        "我拉齐了工程和设计。",
        "模拟面试完成，报告见卡片。",
    ]


def test_the_message_that_stops_a_run_stays_in_the_transcript(tmp_path) -> None:
    paths = _paths(tmp_path)
    manager = ContextManager(CareerContextStore(Path(paths.context_store)))
    context = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="我想做一场行为面模拟"
    )
    held = manager.commit_workflow_entry(
        context=context,
        task=context.task.enter_workflow("mock_interview", run_id="run", phase="awaiting_answer"),
    )
    mock = SQLiteMockInterviewStore(Path(paths.mock_interview_store))
    session = mock.create_session(user_id="u1", interview_type="behavioral", conversation_id="c1")
    mock.save_plan(session=session, plan=_plan(session.id))
    session = mock.start(session=session)
    session, _ = mock.ask(
        session=session, plan_item_number=1, question_type="behavioral",
        question="讲一次你推动跨团队交付的经历。",
    )
    mock.cancel(session=session, message="不练了")
    closing = manager.load_for_workflow_turn(user_id="u1", conversation_id="c1", task=held)
    manager.commit_workflow_exit(
        context=closing, task=held.leave_workflow(), assistant_message="模拟面试已取消。"
    )

    transcript = WorkspaceReader(paths).conversation_messages(user_id="u1", conversation_id="c1")

    assert [(message.role, message.content) for message in transcript.messages] == [
        ("user", "我想做一场行为面模拟"),
        ("assistant", f"{NO_RESUME}\n\n讲一次你推动跨团队交付的经历。"),
        ("user", "不练了"),
        ("assistant", "模拟面试已取消。"),
    ]


def test_a_conversation_whose_only_message_is_held_by_a_run_is_listed(tmp_path) -> None:
    paths = _paths(tmp_path)
    store = CareerContextStore(Path(paths.context_store))
    manager = ContextManager(store)
    context = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="我想做一场行为面模拟"
    )
    manager.commit_workflow_entry(
        context=context,
        task=context.task.enter_workflow("mock_interview", run_id="run", phase="awaiting_answer"),
    )

    [listed] = store.list_conversations(user_id="u1")

    assert listed.conversation_id == "c1"
    assert listed.title == "我想做一场行为面模拟"
