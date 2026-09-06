import pytest

from career_agent.agent.conversation_memory_contracts import ConversationSummaryContent
from career_agent.agent.context_manager import ContextManager
from career_agent.agent.openai_compatible_client import AgentWorkerError
from career_agent.agent.main_agent_contracts import AgentPreferencesContext, CareerProfileContext, ConversationResourceReference, ConversationTaskState
from career_agent.harness.observability import (
    ACTIVE_TRACE_CONTEXT,
    InMemoryTraceRecorder,
    conversation_trace_key,
)
from career_agent.storage.context import CareerContextStore


def manager(
    tmp_path,
    *,
    limit: int = 4,
    summary_worker=None,
    summary_batch_size: int = 2,
    max_recent_context_chars: int = 16000,
    compact_occupancy_threshold: float = 0.75,
    compacted_message_warning_threshold: int = 200,
) -> ContextManager:
    return ContextManager(
        CareerContextStore(tmp_path / "context.sqlite3"),
        summary_worker=summary_worker,
        recent_message_limit=limit,
        summary_batch_size=summary_batch_size,
        max_message_chars=32,
        max_recent_context_chars=max_recent_context_chars,
        compact_occupancy_threshold=compact_occupancy_threshold,
        compacted_message_warning_threshold=compacted_message_warning_threshold,
    )


class RecordingSummaryWorker:
    def __init__(self) -> None:
        self.calls = []

    def summarize(self, *, previous, messages):
        self.calls.append((previous, messages))
        decisions = previous.confirmed_decisions if previous else ()
        # Keep the newest few. A real worker consolidates rather than appends,
        # and the contract caps this tuple, so an ever-growing double would fail
        # on its own accumulation instead of on the behaviour under test.
        decisions = decisions[-4:]
        return ConversationSummaryContent(
            user_goals=("Maintain conversation continuity",),
            confirmed_decisions=(*decisions, f"covered-through-{messages[-1].sequence}"),
            unresolved_questions=(),
            active_constraints=("Do not promote this summary to career facts",),
        )


@pytest.mark.parametrize("threshold", [0.69, 0.91])
def test_compaction_occupancy_threshold_stays_in_the_measured_band(
    tmp_path, threshold
) -> None:
    with pytest.raises(ValueError, match="between 0.7 and 0.9"):
        manager(tmp_path, compact_occupancy_threshold=threshold)


def test_loads_profile_preferences_task_and_bounded_history(tmp_path) -> None:
    context_manager = manager(tmp_path)
    context_manager.upsert_profile(CareerProfileContext(user_id="u1", default_city="Shanghai"))
    context_manager.upsert_preferences(user_id="u1", preferences=AgentPreferencesContext(boss_search="allowed"))
    initial = context_manager.load_for_turn(user_id="u1", conversation_id="c1", user_message="First message")
    context_manager.commit_turn(context=initial, task=ConversationTaskState(active_workflow="job_discovery", run_id="run-1", phase="selection_required"), assistant_message="First response")

    loaded = context_manager.load_for_turn(user_id="u1", conversation_id="c1", user_message="Second message")

    assert loaded.profile.default_city == "Shanghai"
    assert loaded.preferences.boss_search == "allowed"
    assert loaded.task.run_id == "run-1"
    assert [message.content for message in loaded.recent_messages] == ["First message", "First response"]
    assert loaded.through_sequence == 0
    assert loaded.recent_from_sequence == 1
    assert "through_sequence" not in loaded.model_context()
    assert "recent_from_sequence" not in loaded.model_context()
    assert loaded.user_message == "Second message"


def test_workflow_turn_updates_routing_without_loading_or_writing_main_memory(
    tmp_path,
) -> None:
    worker = RecordingSummaryWorker()
    context_manager = manager(tmp_path, limit=2, summary_worker=worker)
    initial = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="Main Agent message"
    )
    context_manager.commit_turn(
        context=initial,
        task=ConversationTaskState(),
        assistant_message="Main Agent response",
    )
    task = ConversationTaskState(
        active_workflow="mock_interview",
        run_id="mock-session-1",
        phase="mock_interview_answer_required",
    )

    workflow_context = context_manager.load_for_workflow_turn(
        user_id="u1",
        conversation_id="c1",
        task=task,
    )
    context_manager.commit_workflow_turn(
        context=workflow_context,
        task=task.model_copy(update={"phase": "mock_interview_running"}),
    )

    assert workflow_context.recent_messages == ()
    assert workflow_context.conversation_summary is None
    assert "Private interview answer" not in workflow_context.user_message
    assert worker.calls == []
    loaded = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="Back to main"
    )
    assert [message.content for message in loaded.recent_messages] == [
        "Main Agent message",
        "Main Agent response",
    ]
    assert loaded.task.phase == "mock_interview_running"


def test_context_isolated_by_user_and_conversation(tmp_path) -> None:
    context_manager = manager(tmp_path)
    context_manager.upsert_profile(CareerProfileContext(user_id="u1"))
    first = context_manager.load_for_turn(user_id="u1", conversation_id="same", user_message="u1")
    context_manager.commit_turn(context=first, task=ConversationTaskState(active_workflow="job_discovery", run_id="run-u1"), assistant_message="done")

    other_user = context_manager.load_for_turn(user_id="u2", conversation_id="same", user_message="u2")
    other_conversation = context_manager.load_for_turn(user_id="u1", conversation_id="other", user_message="other")

    assert other_user.profile.default_city is None
    assert other_user.task.run_id is None
    assert other_user.recent_messages == ()
    assert other_conversation.task.run_id is None
    assert other_conversation.recent_messages == ()


def test_commit_trims_messages_and_survives_manager_rebuild(tmp_path) -> None:
    first = manager(tmp_path, limit=2)
    for index in range(2):
        context = first.load_for_turn(user_id="u1", conversation_id="c1", user_message=f"user-{index}")
        first.commit_turn(context=context, task=ConversationTaskState(), assistant_message=f"assistant-{index}")

    rebuilt = manager(tmp_path, limit=2)
    loaded = rebuilt.load_for_turn(user_id="u1", conversation_id="c1", user_message="next")

    assert [message.content for message in loaded.recent_messages] == ["user-1", "assistant-1"]


def test_messages_are_truncated_without_profile_mutation(tmp_path) -> None:
    context_manager = manager(tmp_path)
    context_manager.upsert_profile(CareerProfileContext(user_id="u1", default_city="Shanghai"))
    context = context_manager.load_for_turn(user_id="u1", conversation_id="c1", user_message="x" * 100)
    context_manager.commit_turn(context=context, task=ConversationTaskState(), assistant_message="y" * 100)

    loaded = context_manager.load_for_turn(user_id="u1", conversation_id="c1", user_message="next")

    assert [len(message.content) for message in loaded.recent_messages] == [32, 32]
    assert loaded.profile.default_city == "Shanghai"


def test_short_chat_below_occupancy_keeps_raw_messages_without_summary(
    tmp_path,
) -> None:
    worker = RecordingSummaryWorker()
    context_manager = manager(tmp_path, limit=4, summary_worker=worker)
    for index in range(2):
        context = context_manager.load_for_turn(
            user_id="u1",
            conversation_id="c1",
            user_message=f"user-{index}",
        )
        context_manager.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message=f"assistant-{index}",
        )

    assert worker.calls == []
    assert context_manager._store.get_conversation_summary(
        user_id="u1", conversation_id="c1"
    ) is None
    assert len(
        context_manager._store.list_messages_after(
            user_id="u1",
            conversation_id="c1",
            after_sequence=0,
            limit=10,
        )
    ) == 4


def test_rolls_old_messages_into_structured_summary_and_keeps_recent_raw_window(
    tmp_path,
) -> None:
    worker = RecordingSummaryWorker()
    context_manager = manager(
        tmp_path,
        limit=4,
        summary_worker=worker,
        max_recent_context_chars=60,
    )
    for index in range(3):
        context = context_manager.load_for_turn(
            user_id="u1",
            conversation_id="c1",
            user_message=f"user-{index}",
        )
        context_manager.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message=f"assistant-{index}",
        )

    loaded = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="next"
    )
    stored = context_manager._store.get_conversation_summary(
        user_id="u1", conversation_id="c1"
    )

    assert stored is not None
    assert stored.through_sequence == 2
    assert loaded.conversation_summary == stored.content
    assert loaded.through_sequence == 2
    assert loaded.recent_from_sequence == 3
    assert loaded.model_context()["through_sequence"] == 2
    assert loaded.model_context()["recent_from_sequence"] == 3
    assert [message.content for message in loaded.recent_messages] == [
        "user-1",
        "assistant-1",
        "user-2",
        "assistant-2",
    ]
    assert [message.sequence for message in worker.calls[0][1]] == [1, 2]
    # Summarised rows are skipped, not deleted. A summary is a model output, so
    # while it is the only thing read, it must not be the only thing kept: if it
    # loses or distorts a turn, the original is the only way to find out.
    remaining = context_manager._store.list_messages_after(
        user_id="u1",
        conversation_id="c1",
        after_sequence=0,
        limit=10,
    )
    assert [message.sequence for message in remaining] == [1, 2, 3, 4, 5, 6]


def test_occupancy_compaction_records_its_trigger_without_raw_arguments(
    tmp_path,
) -> None:
    worker = RecordingSummaryWorker()
    context_manager = manager(
        tmp_path,
        limit=4,
        summary_worker=worker,
        max_recent_context_chars=60,
    )
    recorder = InMemoryTraceRecorder()
    token = ACTIVE_TRACE_CONTEXT.set((recorder, "turn-1"))
    try:
        for index in range(2):
            context = context_manager.load_for_turn(
                user_id="u1",
                conversation_id="c1",
                user_message=f"user-{index}",
            )
            context_manager.commit_turn(
                context=context,
                task=ConversationTaskState(),
                assistant_message="a" * 20,
            )
    finally:
        ACTIVE_TRACE_CONTEXT.reset(token)

    compacted = [
        event
        for event in recorder.snapshot("turn-1").events
        if event.event_type == "context_compacted"
    ]
    assert len(compacted) == 1
    assert compacted[0].details == {
        "conversation_id": "c1",
        "conversation_key": conversation_trace_key("u1", "c1"),
        "trigger": "occupancy",
        "through_sequence": 2,
        "occupancy": 52 / 60,
        "projection_overflow": False,
        "batch_size": 2,
    }


def test_default_short_chat_compacts_when_unsummarized_rows_leave_projection(
    tmp_path,
) -> None:
    worker = RecordingSummaryWorker()
    context_manager = ContextManager(
        CareerContextStore(tmp_path / "context.sqlite3"),
        summary_worker=worker,
    )
    recorder = InMemoryTraceRecorder()
    token = ACTIVE_TRACE_CONTEXT.set((recorder, "turn-1"))
    try:
        for _ in range(6):
            context = context_manager.load_for_turn(
                user_id="u1",
                conversation_id="c1",
                user_message="ok",
            )
            context_manager.commit_turn(
                context=context,
                task=ConversationTaskState(),
                assistant_message="ok",
            )
    finally:
        ACTIVE_TRACE_CONTEXT.reset(token)

    summary = context_manager._store.get_conversation_summary(
        user_id="u1", conversation_id="c1"
    )
    assert summary is not None
    assert summary.through_sequence == 4
    compacted = [
        event
        for event in recorder.snapshot("turn-1").events
        if event.event_type == "context_compacted"
    ]
    assert compacted[-1].details["trigger"] == "projection_overflow"
    assert compacted[-1].details["projection_overflow"] is True
    assert compacted[-1].details["occupancy"] < 0.01


def test_workflow_exit_compacts_at_a_low_occupancy_seam(tmp_path) -> None:
    worker = RecordingSummaryWorker()
    context_manager = manager(tmp_path, limit=4, summary_worker=worker)
    entry_context = context_manager.load_for_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="开始模拟面试",
    )
    held = context_manager.commit_workflow_entry(
        context=entry_context,
        task=ConversationTaskState(
            active_workflow="mock_interview",
            run_id="mock-1",
            phase="mock_interview_answer_required",
        ),
    )
    workflow_context = context_manager.load_for_workflow_turn(
        user_id="u1",
        conversation_id="c1",
        task=held,
    )

    context_manager.commit_workflow_exit(
        context=workflow_context,
        task=ConversationTaskState(),
        assistant_message="模拟面试已完成。",
    )

    summary = context_manager._store.get_conversation_summary(
        user_id="u1", conversation_id="c1"
    )
    assert summary is not None
    assert summary.through_sequence == 2
    assert len(worker.calls) == 1


def test_report_delivery_does_not_create_a_low_occupancy_seam(tmp_path) -> None:
    worker = RecordingSummaryWorker()
    context_manager = manager(tmp_path, limit=4, summary_worker=worker)
    context = context_manager.load_for_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="研究这个岗位",
    )

    context_manager.commit_turn(
        context=context,
        task=ConversationTaskState(),
        assistant_message="岗位调研报告已生成。",
        assistant_resource_refs=(
            ConversationResourceReference(
                kind="job_research_report",
                resource_id="report-1",
                status_at_delivery="current",
                anchored_by_other_job=False,
            ),
        ),
    )

    assert worker.calls == []
    assert context_manager._store.get_conversation_summary(
        user_id="u1", conversation_id="c1"
    ) is None


def test_a_conversation_without_a_summary_worker_still_keeps_every_message(
    tmp_path,
) -> None:
    """Retention must not depend on whether an optional worker was wired.

    Without a summary worker the read window is already limited to
    ``recent_message_limit`` messages, so pruning the table deleted only rows
    that could never be read again. It also took their resource references
    along, which is what a later turn scans to name an old report.
    """
    context_manager = manager(tmp_path, limit=4)
    window_sizes = []
    for index in range(20):
        context = context_manager.load_for_turn(
            user_id="u1", conversation_id="c1", user_message=f"user-{index}"
        )
        window_sizes.append(len(context.recent_messages))
        context_manager.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message=f"assistant-{index}",
        )

    assert max(window_sizes) <= 4
    stored = context_manager._store.list_messages_after(
        user_id="u1", conversation_id="c1", after_sequence=0, limit=1000
    )
    assert len(stored) == 40
    assert stored[0].content == "user-0"


def test_the_read_window_stays_bounded_while_the_table_keeps_growing(tmp_path) -> None:
    """What the model reads is bounded; what the file stores is not.

    These are separate properties now. Bounding the file too would mean deleting
    originals on the agent's own schedule, which is what the operator has to be
    able to decide instead.
    """
    worker = RecordingSummaryWorker()
    context_manager = manager(
        tmp_path,
        limit=4,
        summary_worker=worker,
        max_recent_context_chars=32,
    )
    window_sizes = []
    for index in range(40):
        context = context_manager.load_for_turn(
            user_id="u1", conversation_id="c1", user_message=f"user-{index}"
        )
        window_sizes.append(len(context.recent_messages))
        context_manager.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message=f"assistant-{index}",
        )

    assert max(window_sizes) <= 6
    stored = context_manager._store.list_messages_after(
        user_id="u1", conversation_id="c1", after_sequence=0, limit=1000
    )
    assert len(stored) == 80
    # Everything past the read window is reclaimable, and nothing inside it is.
    reclaimable, byte_size = context_manager._store.count_compacted_messages(
        user_id="u1", conversation_id="c1"
    )
    summary = context_manager._store.get_conversation_summary(
        user_id="u1", conversation_id="c1"
    )
    assert summary is not None
    assert reclaimable == summary.through_sequence
    assert len(stored) - reclaimable <= 6
    assert byte_size > 0
    summary = context_manager._store.get_conversation_summary(
        user_id="u1", conversation_id="c1"
    )
    assert summary is not None
    assert summary.through_sequence >= 70
    # The recent window is still served from raw rows, not from the summary.
    loaded = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="next"
    )
    assert [message.content for message in loaded.recent_messages] == [
        "user-39",
        "assistant-39",
    ]


def test_rolling_summary_merges_previous_summary_and_is_session_scoped(tmp_path) -> None:
    worker = RecordingSummaryWorker()
    context_manager = manager(
        tmp_path,
        limit=2,
        summary_worker=worker,
        max_recent_context_chars=32,
    )
    for index in range(3):
        context = context_manager.load_for_turn(
            user_id="u1", conversation_id="c1", user_message=f"user-{index}"
        )
        context_manager.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message=f"assistant-{index}",
        )

    summary = context_manager._store.get_conversation_summary(
        user_id="u1", conversation_id="c1"
    )
    assert summary is not None
    assert summary.through_sequence == 4
    assert worker.calls[0][0] is None
    assert worker.calls[1][0].confirmed_decisions == ("covered-through-2",)
    assert summary.content.confirmed_decisions == (
        "covered-through-2",
        "covered-through-4",
    )
    assert context_manager.load_for_turn(
        user_id="u2", conversation_id="c1", user_message="other user"
    ).conversation_summary is None
    assert context_manager.load_for_turn(
        user_id="u1", conversation_id="other", user_message="other session"
    ).conversation_summary is None


class FailingSummaryWorker:
    def summarize(self, **kwargs):
        raise AgentWorkerError(
            "SUMMARY_FAILED", "temporary summary failure", retryable=True
        )


def test_summary_worker_failure_preserves_recent_conversation(tmp_path) -> None:
    context_manager = manager(
        tmp_path,
        limit=2,
        summary_worker=FailingSummaryWorker(),
        max_recent_context_chars=32,
    )
    for index in range(2):
        context = context_manager.load_for_turn(
            user_id="u1", conversation_id="c1", user_message=f"user-{index}"
        )
        context_manager.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message=f"assistant-{index}",
        )

    loaded = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="next"
    )
    assert loaded.conversation_summary is None
    assert [message.content for message in loaded.recent_messages][-2:] == [
        "user-1",
        "assistant-1",
    ]


def test_recent_message_projection_obeys_total_character_budget(tmp_path) -> None:
    context_manager = manager(
        tmp_path,
        limit=4,
        max_recent_context_chars=40,
    )
    for _ in range(2):
        context = context_manager.load_for_turn(
            user_id="u1", conversation_id="c1", user_message="x" * 32
        )
        context_manager.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message="y" * 32,
        )

    loaded = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="next"
    )
    assert sum(len(message.content) for message in loaded.recent_messages) == 40
    assert loaded.recent_messages[-1].content == "y" * 20
    assert loaded.through_sequence == 0
    assert loaded.recent_from_sequence == 3


def test_conversation_span_is_exact_owned_bounded_and_reports_full_count(
    tmp_path,
) -> None:
    context_manager = manager(tmp_path, limit=4)
    for index in range(15):
        context = context_manager.load_for_turn(
            user_id="u1", conversation_id="c1", user_message=f"user-{index}"
        )
        context_manager.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message=f"assistant-{index}",
        )
    other = context_manager.load_for_turn(
        user_id="u1", conversation_id="c2", user_message="other-conversation"
    )
    context_manager.commit_turn(
        context=other,
        task=ConversationTaskState(),
        assistant_message="other-answer",
    )

    span = context_manager._store.read_conversation_span(
        user_id="u1",
        conversation_id="c1",
        from_sequence=1,
        through_sequence=30,
    )
    outside = context_manager._store.read_conversation_span(
        user_id="u1",
        conversation_id="c1",
        from_sequence=100,
        through_sequence=110,
    )

    assert span.returned == 8
    assert span.total == 30
    assert [message.sequence for message in span.messages] == list(range(1, 9))
    assert all("other" not in message.content for message in span.messages)
    assert outside.returned == outside.total == 0
    assert outside.messages == ()


def test_full_stored_message_is_clipped_only_for_summary_input(tmp_path) -> None:
    worker = RecordingSummaryWorker()
    context_manager = ContextManager(
        CareerContextStore(tmp_path / "context.sqlite3"),
        summary_worker=worker,
        recent_message_limit=2,
        summary_batch_size=2,
        max_message_chars=32000,
        max_recent_context_chars=10000,
    )
    context = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="u" * 8000
    )
    context_manager.commit_turn(
        context=context,
        task=ConversationTaskState(),
        assistant_message="a" * 8000,
    )
    second = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="second"
    )
    context_manager.commit_turn(
        context=second,
        task=ConversationTaskState(),
        assistant_message="second answer",
    )

    # Triggering and retrying summary must not fail the conversation load.
    context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="next"
    )
    stored = context_manager._store.list_messages(
        "u1", "c1", limit=10
    )
    assert len(stored[0].content) == 8000
    assert [len(item.content) for item in worker.calls[0][1]] == [4000, 4000]
