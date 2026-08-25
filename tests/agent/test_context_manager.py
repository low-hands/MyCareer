from career_agent.agent.conversation_memory_contracts import ConversationSummaryContent
from career_agent.agent.context_manager import ContextManager
from career_agent.agent.openai_compatible_client import AgentWorkerError
from career_agent.agent.main_agent_contracts import AgentPreferencesContext, CareerProfileContext, ConversationTaskState
from career_agent.storage.context import CareerContextStore


def manager(
    tmp_path,
    *,
    limit: int = 4,
    summary_worker=None,
    summary_batch_size: int = 2,
    max_recent_context_chars: int = 16000,
) -> ContextManager:
    return ContextManager(
        CareerContextStore(tmp_path / "context.sqlite3"),
        summary_worker=summary_worker,
        recent_message_limit=limit,
        summary_batch_size=summary_batch_size,
        max_message_chars=32,
        max_recent_context_chars=max_recent_context_chars,
    )


class RecordingSummaryWorker:
    def __init__(self) -> None:
        self.calls = []

    def summarize(self, *, previous, messages):
        self.calls.append((previous, messages))
        decisions = previous.confirmed_decisions if previous else ()
        return ConversationSummaryContent(
            user_goals=("Maintain conversation continuity",),
            confirmed_decisions=(*decisions, f"covered-through-{messages[-1].sequence}"),
            unresolved_questions=(),
            active_constraints=("Do not promote this summary to career facts",),
        )


def test_loads_profile_preferences_task_and_bounded_history(tmp_path) -> None:
    context_manager = manager(tmp_path)
    context_manager.upsert_profile(CareerProfileContext(user_id="u1", target_roles=("AI Engineer",), default_city="Shanghai"))
    context_manager.upsert_preferences(user_id="u1", preferences=AgentPreferencesContext(boss_search="allowed"))
    initial = context_manager.load_for_turn(user_id="u1", conversation_id="c1", user_message="First message")
    context_manager.commit_turn(context=initial, task=ConversationTaskState(active_workflow="job_discovery", run_id="run-1", phase="selection_required"), assistant_message="First response")

    loaded = context_manager.load_for_turn(user_id="u1", conversation_id="c1", user_message="Second message")

    assert loaded.profile.default_city == "Shanghai"
    assert loaded.preferences.boss_search == "allowed"
    assert loaded.task.run_id == "run-1"
    assert [message.content for message in loaded.recent_messages] == ["First message", "First response"]
    assert loaded.user_message == "Second message"


def test_context_isolated_by_user_and_conversation(tmp_path) -> None:
    context_manager = manager(tmp_path)
    context_manager.upsert_profile(CareerProfileContext(user_id="u1", target_roles=("AI Engineer",)))
    first = context_manager.load_for_turn(user_id="u1", conversation_id="same", user_message="u1")
    context_manager.commit_turn(context=first, task=ConversationTaskState(active_workflow="job_discovery", run_id="run-u1"), assistant_message="done")

    other_user = context_manager.load_for_turn(user_id="u2", conversation_id="same", user_message="u2")
    other_conversation = context_manager.load_for_turn(user_id="u1", conversation_id="other", user_message="other")

    assert other_user.profile.target_roles == ()
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
    context_manager.upsert_profile(CareerProfileContext(user_id="u1", target_roles=("AI Engineer",), default_city="Shanghai"))
    context = context_manager.load_for_turn(user_id="u1", conversation_id="c1", user_message="x" * 100)
    context_manager.commit_turn(context=context, task=ConversationTaskState(), assistant_message="y" * 100)

    loaded = context_manager.load_for_turn(user_id="u1", conversation_id="c1", user_message="next")

    assert [len(message.content) for message in loaded.recent_messages] == [32, 32]
    assert loaded.profile.default_city == "Shanghai"


def test_rolls_old_messages_into_structured_summary_and_keeps_recent_raw_window(
    tmp_path,
) -> None:
    worker = RecordingSummaryWorker()
    context_manager = manager(tmp_path, limit=4, summary_worker=worker)
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
    assert [message.content for message in loaded.recent_messages] == [
        "user-1",
        "assistant-1",
        "user-2",
        "assistant-2",
    ]
    assert [message.sequence for message in worker.calls[0][1]] == [1, 2]
    assert len(
        context_manager._store.list_messages_after(
            user_id="u1",
            conversation_id="c1",
            after_sequence=0,
            limit=10,
        )
    ) == 6


def test_rolling_summary_merges_previous_summary_and_is_session_scoped(tmp_path) -> None:
    worker = RecordingSummaryWorker()
    context_manager = manager(tmp_path, limit=2, summary_worker=worker)
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
    assert loaded.recent_messages[-1].content == "y" * 32
