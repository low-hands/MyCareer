from datetime import datetime, timezone
import sqlite3

import pytest

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import ConversationTaskState
from career_agent.agent.session_manager import SessionManager
from career_agent.storage.context import CareerContextStore


def test_session_is_created_and_reused_by_context_manager(tmp_path) -> None:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    manager = ContextManager(store)

    first = manager.load_for_turn(user_id="u1", conversation_id="c1", user_message="hello")
    second = manager.load_for_turn(user_id="u1", conversation_id="c1", user_message="again")
    session = SessionManager(store).get_or_create(user_id="u1", session_id="c1")

    assert first.conversation_id == second.conversation_id == "c1"
    assert session.user_id == "u1"
    assert session.status == "active"


def test_v3_sessions_gain_one_persisted_spotlight_nonce(tmp_path) -> None:
    path = tmp_path / "context.sqlite3"
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE schema_versions(component TEXT PRIMARY KEY, "
            "version INTEGER NOT NULL, updated_at TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO schema_versions VALUES ('agent_context', 3, ?)",
            (now,),
        )
        connection.execute(
            "CREATE TABLE sessions(session_id TEXT NOT NULL, user_id TEXT NOT NULL, "
            "status TEXT NOT NULL, created_at TEXT NOT NULL, "
            "last_active_at TEXT NOT NULL, PRIMARY KEY(user_id, session_id))"
        )
        connection.execute(
            "INSERT INTO sessions VALUES ('c1', 'u1', 'active', ?, ?)",
            (now, now),
        )

    session = CareerContextStore(path).get_session("u1", "c1")

    assert session is not None
    assert len(session.spotlight_nonce) == 32


def test_same_session_id_is_isolated_by_user(tmp_path) -> None:
    manager = SessionManager(CareerContextStore(tmp_path / "context.sqlite3"))
    first = manager.get_or_create(user_id="u1", session_id="c1")
    second = manager.get_or_create(user_id="u2", session_id="c1")

    assert first.user_id == "u1"
    assert second.user_id == "u2"


def test_closed_session_cannot_be_reopened(tmp_path) -> None:
    manager = SessionManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.get_or_create(user_id="u1", session_id="c1")
    closed = manager.close(user_id="u1", session_id="c1")

    assert closed.status == "closed"
    with pytest.raises(ValueError, match="closed"):
        manager.get_or_create(user_id="u1", session_id="c1")


def test_conversation_listing_uses_persisted_first_and_last_messages(tmp_path) -> None:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    manager = ContextManager(store)
    context = manager.load_for_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="帮我分析这个岗位",
    )
    manager.commit_turn(
        context=context,
        task=ConversationTaskState(),
        assistant_message="已经完成岗位分析。",
    )
    manager.load_for_turn(
        user_id="u1",
        conversation_id="empty-session",
        user_message="尚未提交",
    )

    conversations = store.list_conversations(user_id="u1")

    assert len(conversations) == 1
    assert conversations[0].conversation_id == "c1"
    assert conversations[0].title == "帮我分析这个岗位"
    assert conversations[0].last_message_preview == "已经完成岗位分析。"
    assert conversations[0].message_count == 2
    assert store.list_conversations(user_id="u2") == ()


def test_delete_conversation_removes_only_the_owners_session_memory(tmp_path) -> None:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    manager = ContextManager(store)
    for user_id in ("u1", "u2"):
        context = manager.load_for_turn(
            user_id=user_id,
            conversation_id="shared-id",
            user_message=f"{user_id} message",
        )
        manager.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message=f"{user_id} reply",
        )

    assert store.delete_conversation(user_id="u1", conversation_id="shared-id")
    assert store.get_session("u1", "shared-id") is None
    assert store.get_task("u1", "shared-id") is None
    assert store.list_messages("u1", "shared-id", limit=10) == ()
    assert store.get_session("u2", "shared-id") is not None
    assert len(store.list_messages("u2", "shared-id", limit=10)) == 2
    assert not store.delete_conversation(user_id="u1", conversation_id="shared-id")
