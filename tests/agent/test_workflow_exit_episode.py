from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import ConversationTaskState
from career_agent.storage.context import CareerContextStore
from career_agent.storage.episodes import SQLiteCareerEpisodeStore


class RecordingSummaryWorker:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def summarize(self, *, previous, messages):
        self.calls.append((previous, messages))
        raise AssertionError("two seam messages must not satisfy a four-message batch")


def test_mock_exit_writes_l1_even_when_summary_batch_does_not_run(tmp_path) -> None:
    path = tmp_path / "context.sqlite3"
    worker = RecordingSummaryWorker()
    manager = ContextManager(
        CareerContextStore(path),
        summary_worker=worker,
        summary_batch_size=4,
    )
    entry_context = manager.load_for_turn(
        user_id="u1",
        conversation_id="conversation-1",
        user_message="开始模拟面试",
    )
    held = manager.commit_workflow_entry(
        context=entry_context,
        task=ConversationTaskState(
            active_workflow="mock_interview",
            run_id="mock-1",
            phase="mock_interview_answer_required",
        ),
    )
    workflow_context = manager.load_for_workflow_turn(
        user_id="u1",
        conversation_id="conversation-1",
        task=held,
    )

    manager.commit_workflow_exit(
        context=workflow_context,
        task=ConversationTaskState(),
        assistant_message="模拟面试已完成；STAR 回答需要补清楚结果。",
    )

    episode = SQLiteCareerEpisodeStore(path).get_by_source(
        user_id="u1",
        kind="mock_interview",
        source_run_id="mock-1",
    )
    assert episode is not None
    assert episode.conversation_id == "conversation-1"
    assert "STAR" in episode.summary
    assert worker.calls == []
    assert manager._store.get_conversation_summary(
        user_id="u1",
        conversation_id="conversation-1",
    ) is None


def test_adopted_mock_exit_without_a_held_entry_still_writes_l1(tmp_path) -> None:
    path = tmp_path / "context.sqlite3"
    manager = ContextManager(CareerContextStore(path))
    workflow_context = manager.load_for_workflow_turn(
        user_id="u1",
        conversation_id="conversation-1",
        task=ConversationTaskState(
            active_workflow="mock_interview",
            run_id="mock-adopted",
            phase="mock_interview_answer_required",
        ),
    )

    manager.commit_workflow_exit(
        context=workflow_context,
        task=ConversationTaskState(),
        assistant_message="已结束接管的模拟面试。",
    )

    episode = SQLiteCareerEpisodeStore(path).get_by_source(
        user_id="u1",
        kind="mock_interview",
        source_run_id="mock-adopted",
    )
    assert episode is not None
    assert episode.summary == "已结束接管的模拟面试。"
