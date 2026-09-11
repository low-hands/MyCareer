import hashlib
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
)
from threading import Event

import pytest

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.decision_messages import project_decision_messages
from career_agent.agent.main_agent_contracts import (
    CareerProfileContext,
    MainAgentContext,
    WorkingNotesContext,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.harness.observability import (
    ACTIVE_TRACE_CONTEXT,
    InMemoryTraceRecorder,
)
from career_agent.storage.context import CareerContextStore
from career_agent.storage.working_notes import (
    WorkingNotesConflict,
    WorkingNotesSnapshot,
    WorkingNotesStore,
)


def test_working_notes_are_replaced_and_injected_as_untrusted_scratchpad(tmp_path) -> None:
    context = CareerContextStore(tmp_path / "context.sqlite3")
    notes = WorkingNotesStore(tmp_path / "working-notes")
    registry = MainAgentToolRegistry(working_notes_store=notes)

    result = registry.invoke_atomic_tool(
        "update_working_notes",
        {
            "user_id": "private-user-id",
            "markdown": "- 喜欢先看结论\n- 可能不想去大厂，下次应询问",
            "expected_revision": "empty",
        },
    )

    assert result.state == "working_notes_updated"
    assert result.payload["revision"] == hashlib.sha256(
        "- 喜欢先看结论\n- 可能不想去大厂，下次应询问".encode()
    ).hexdigest()[:12]
    loaded = ContextManager(
        context,
        working_notes_store=notes,
    ).load_for_turn(
        user_id="private-user-id",
        conversation_id="c1",
        user_message="推荐几个岗位",
    )
    projected = loaded.model_context()
    assert projected["working_notes"] == {
        "revision": result.payload["revision"],
        "markdown": "- 喜欢先看结论\n- 可能不想去大厂，下次应询问",
    }
    decision_projection = project_decision_messages(loaded)
    assert decision_projection.volatile_data["working_notes"] == projected[
        "working_notes"
    ]
    assert all("private-user-id" not in path.name for path in notes.root.iterdir())


def test_oversized_file_is_clipped_without_failing_the_turn(tmp_path) -> None:
    context = CareerContextStore(tmp_path / "context.sqlite3")
    notes = WorkingNotesStore(tmp_path / "working-notes")
    original = "x" * 2001
    notes._path("u1").write_text(original, encoding="utf-8")
    recorder = InMemoryTraceRecorder()
    token = ACTIVE_TRACE_CONTEXT.set((recorder, "turn-oversize"))
    try:
        loaded = ContextManager(context, working_notes_store=notes).load_for_turn(
            user_id="u1",
            conversation_id="c1",
            user_message="继续",
        )
    finally:
        ACTIVE_TRACE_CONTEXT.reset(token)

    projected = loaded.model_context()["working_notes"]
    assert projected == {
        "revision": hashlib.sha256(original.encode()).hexdigest()[:12],
        "markdown": "x" * 2000,
        "clipped": True,
    }
    event = next(
        event
        for event in recorder.snapshot("turn-oversize").events
        if event.event_type == "working_notes_oversize"
    )
    assert event.details == {"chars": 2000}


def test_non_utf8_file_projects_as_empty_without_failing_the_turn(tmp_path) -> None:
    context = CareerContextStore(tmp_path / "context.sqlite3")
    notes = WorkingNotesStore(tmp_path / "working-notes")
    notes._path("u1").write_bytes(b"\xff\xfe\xfa")

    loaded = ContextManager(context, working_notes_store=notes).load_for_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="继续",
    )

    assert loaded.model_context()["working_notes"] == {
        "revision": "empty",
        "markdown": "",
    }


def test_matching_revision_replaces_notes_and_returns_disk_revision(tmp_path) -> None:
    notes = WorkingNotesStore(tmp_path / "working-notes")

    result = notes.replace(
        user_id="u1",
        markdown="first",
        expected_revision="empty",
    )

    assert isinstance(result, WorkingNotesSnapshot)
    assert result == notes.read(user_id="u1")
    assert result.revision == hashlib.sha256(b"first").hexdigest()[:12]
    assert notes._path("u1").read_text(encoding="utf-8") == "first"


def test_crossed_writes_return_current_notes_then_merge_succeeds(tmp_path) -> None:
    notes = WorkingNotesStore(tmp_path / "working-notes")
    registry = MainAgentToolRegistry(working_notes_store=notes)
    revision_a = notes.read(user_id="u1").revision
    revision_b = notes.read(user_id="u1").revision

    first = registry.invoke_atomic_tool(
        "update_working_notes",
        {
            "user_id": "u1",
            "markdown": "- 会话 A 的关键片段",
            "expected_revision": revision_a,
        },
    )
    stale = registry.invoke_atomic_tool(
        "update_working_notes",
        {
            "user_id": "u1",
            "markdown": "- 会话 B 的内容",
            "expected_revision": revision_b,
        },
    )

    assert first.state == "working_notes_updated"
    assert stale.state == "working_notes_stale"
    assert stale.execution_outcome == "not_committed"
    assert stale.payload == {
        "current_revision": first.payload["revision"],
        "current_markdown": "- 会话 A 的关键片段",
    }
    merged = registry.invoke_atomic_tool(
        "update_working_notes",
        {
            "user_id": "u1",
            "markdown": "- 会话 A 的关键片段\n- 会话 B 的内容",
            "expected_revision": stale.payload["current_revision"],
        },
    )
    assert merged.state == "working_notes_updated"
    assert notes.read(user_id="u1").markdown == (
        "- 会话 A 的关键片段\n- 会话 B 的内容"
    )


def test_simultaneous_writes_allow_exactly_one_revision_winner(tmp_path) -> None:
    notes = WorkingNotesStore(tmp_path / "working-notes")

    def write(markdown):
        return notes.replace(
            user_id="u1",
            markdown=markdown,
            expected_revision="empty",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(write, ("first", "second")))

    winners = tuple(
        result for result in results if isinstance(result, WorkingNotesSnapshot)
    )
    conflicts = tuple(
        result for result in results if isinstance(result, WorkingNotesConflict)
    )
    assert len(winners) == len(conflicts) == 1
    assert conflicts[0].current == winners[0]
    assert notes.read(user_id="u1") == winners[0]


def test_clear_waits_for_replace_lock_then_removes_the_new_file(
    tmp_path,
    monkeypatch,
) -> None:
    notes = WorkingNotesStore(tmp_path / "working-notes")
    initial = notes.replace(
        user_id="u1",
        markdown="before",
        expected_revision="empty",
    )
    assert isinstance(initial, WorkingNotesSnapshot)
    original_read = notes._read_path
    read_completed = Event()
    allow_replace_to_finish = Event()
    clear_started = Event()

    def paused_read(path):
        snapshot = original_read(path)
        read_completed.set()
        assert allow_replace_to_finish.wait(timeout=1)
        return snapshot

    def clear():
        clear_started.set()
        return notes.clear(user_id="u1")

    monkeypatch.setattr(notes, "_read_path", paused_read)
    with ThreadPoolExecutor(max_workers=2) as executor:
        replacement = executor.submit(
            notes.replace,
            user_id="u1",
            markdown="replacement",
            expected_revision=initial.revision,
        )
        assert read_completed.wait(timeout=1)
        clearing = executor.submit(clear)
        assert clear_started.wait(timeout=1)
        try:
            with pytest.raises(FuturesTimeoutError):
                clearing.result(timeout=0.05)
        finally:
            allow_replace_to_finish.set()
        assert isinstance(replacement.result(timeout=1), WorkingNotesSnapshot)
        assert clearing.result(timeout=1) is True

    assert notes.read(user_id="u1").revision == "empty"


def test_empty_nonempty_empty_keeps_empty_revision_at_both_ends(tmp_path) -> None:
    notes = WorkingNotesStore(tmp_path / "working-notes")
    initial = notes.read(user_id="u1")

    nonempty = notes.replace(
        user_id="u1",
        markdown="temporary",
        expected_revision=initial.revision,
    )
    assert isinstance(nonempty, WorkingNotesSnapshot)
    emptied = notes.replace(
        user_id="u1",
        markdown="",
        expected_revision=nonempty.revision,
    )

    assert initial.revision == "empty"
    assert isinstance(emptied, WorkingNotesSnapshot)
    assert emptied.revision == "empty"
    assert notes.read(user_id="u1") == emptied


@pytest.mark.parametrize(
    "arguments",
    (
        {"markdown": "new"},
        {"markdown": "new", "expected_revision": "wrong"},
        {"markdown": "new", "expected_revision": "A" * 12},
    ),
)
def test_invalid_expected_revision_is_rejected_during_projection(
    tmp_path,
    arguments,
) -> None:
    notes = WorkingNotesStore(tmp_path / "working-notes")
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        working_notes=WorkingNotesContext(
            markdown="",
            revision="empty",
        ),
        user_message="更新笔记",
    )

    with pytest.raises(ValueError):
        MainAgentRuntime._project_atomic_tool_arguments(
            context,
            "update_working_notes",
            arguments,
        )

    assert notes.read(user_id="u1").revision == "empty"


def test_user_isolation_and_clear_are_preserved(tmp_path) -> None:
    notes = WorkingNotesStore(tmp_path / "working-notes")
    first = notes.replace(
        user_id="u1",
        markdown="one",
        expected_revision="empty",
    )
    second = notes.replace(
        user_id="u2",
        markdown="two",
        expected_revision="empty",
    )

    assert isinstance(first, WorkingNotesSnapshot)
    assert isinstance(second, WorkingNotesSnapshot)
    assert notes.read(user_id="u1").markdown == "one"
    assert notes.read(user_id="u2").markdown == "two"
    assert notes.clear(user_id="u1") is True
    assert notes.clear(user_id="u1") is False
    assert notes.read(user_id="u1").revision == "empty"
    assert notes.read(user_id="u2").markdown == "two"


def test_store_rejects_more_than_2000_characters_before_writing(tmp_path) -> None:
    notes = WorkingNotesStore(tmp_path / "working-notes")

    with pytest.raises(ValueError, match="2000"):
        notes.replace(
            user_id="u1",
            markdown="x" * 2001,
            expected_revision="empty",
        )

    assert notes.read(user_id="u1").revision == "empty"


def test_store_conflict_type_carries_current_snapshot(tmp_path) -> None:
    notes = WorkingNotesStore(tmp_path / "working-notes")
    first = notes.replace(
        user_id="u1",
        markdown="winner",
        expected_revision="empty",
    )
    assert isinstance(first, WorkingNotesSnapshot)

    conflict = notes.replace(
        user_id="u1",
        markdown="loser",
        expected_revision="empty",
    )

    assert isinstance(conflict, WorkingNotesConflict)
    assert conflict.current == first
