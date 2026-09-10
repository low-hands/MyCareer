from career_agent.agent.context_manager import ContextManager
from career_agent.agent.decision_messages import project_decision_messages
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.storage.context import CareerContextStore
from career_agent.storage.working_notes import WorkingNotesStore


def test_working_notes_are_replaced_and_injected_as_untrusted_scratchpad(tmp_path) -> None:
    context = CareerContextStore(tmp_path / "context.sqlite3")
    notes = WorkingNotesStore(tmp_path / "working-notes")
    registry = MainAgentToolRegistry(working_notes_store=notes)

    result = registry.invoke_atomic_tool(
        "update_working_notes",
        {
            "user_id": "private-user-id",
            "markdown": "- 喜欢先看结论\n- 可能不想去大厂，下次应询问",
        },
    )

    assert result.state == "working_notes_updated"
    loaded = ContextManager(
        context,
        working_notes_store=notes,
    ).load_for_turn(
        user_id="private-user-id",
        conversation_id="c1",
        user_message="推荐几个岗位",
    )
    projected = loaded.model_context()
    assert projected["working_notes"].startswith("- 喜欢先看结论")
    decision_projection = project_decision_messages(loaded)
    assert decision_projection.volatile_data["working_notes"].startswith(
        "- 喜欢先看结论"
    )
    assert all("private-user-id" not in path.name for path in notes.root.iterdir())


def test_working_notes_have_a_hard_2000_character_ceiling(tmp_path) -> None:
    notes = WorkingNotesStore(tmp_path / "working-notes")
    notes.replace(user_id="u1", markdown="x" * 2000)

    try:
        notes.replace(user_id="u1", markdown="x" * 2001)
    except ValueError as error:
        assert "2000" in str(error)
    else:
        raise AssertionError("oversized working notes were accepted")
