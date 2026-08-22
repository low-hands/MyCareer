from datetime import datetime, timezone

import pytest

from career_agent.agent.context_manager import ContextManager
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
