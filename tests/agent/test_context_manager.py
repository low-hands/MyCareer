from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import AgentPreferencesContext, CareerProfileContext, ConversationTaskState
from career_agent.storage.context import CareerContextStore


def manager(tmp_path, *, limit: int = 4) -> ContextManager:
    return ContextManager(CareerContextStore(tmp_path / "context.sqlite3"), recent_message_limit=limit, max_message_chars=32)


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
