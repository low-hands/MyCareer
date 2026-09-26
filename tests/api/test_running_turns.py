"""A conversation whose turn is still executing says so.

The client may leave a running conversation (open another one, start a task
from a page); the server keeps executing and commits the turn. When the
reader comes back, or looks at the sidebar, they need to know the reply is
still coming rather than see a conversation that looks finished or missing.
"""

from __future__ import annotations

from pathlib import Path

from career_agent.agent.context_manager import ContextManager
from career_agent.api.reads import WorkspaceReader
from career_agent.storage.context import CareerContextStore
from career_agent.storage.turn_receipts import SQLiteTurnReceiptStore
from tests.api.test_clear_applications import _paths


def test_a_running_turn_is_reported_in_the_list_and_the_transcript(tmp_path) -> None:
    paths = _paths(tmp_path)
    store = CareerContextStore(Path(paths.context_store))
    manager = ContextManager(store)
    # One finished conversation, and a new one whose first turn is executing.
    done = manager.load_for_turn(user_id="u1", conversation_id="done", user_message="你好")
    manager.commit_turn(context=done, task=done.task, assistant_message="你好！")
    manager.load_for_turn(user_id="u1", conversation_id="busy", user_message="帮我研究公司")
    receipts = SQLiteTurnReceiptStore(Path(paths.context_store))
    receipts.begin(user_id="u1", conversation_id="busy", request_id="r1", turn_id="t1")
    reader = WorkspaceReader(paths)

    listed = {item.id: item for item in reader.conversations(user_id="u1")}
    assert listed["busy"].turn_running is True
    assert listed["done"].turn_running is False
    assert reader.conversation_messages(user_id="u1", conversation_id="busy").turn_running

    receipts.commit(
        user_id="u1", conversation_id="busy", request_id="r1", turn_id="t1", events=()
    )
    assert not reader.conversation_messages(user_id="u1", conversation_id="busy").turn_running
    # Another user's running turn is not ours.
    receipts.begin(user_id="u2", conversation_id="done", request_id="r2", turn_id="t2")
    assert not reader.conversation_messages(user_id="u1", conversation_id="done").turn_running
