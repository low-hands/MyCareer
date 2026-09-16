from career_agent.agent.career_context import CareerContextProjector
from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    MemoryTombstoneProposal,
    ToolCall,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.services.memory_review import MemoryReviewService
from career_agent.storage.career_history import CareerHistoryStore
from career_agent.storage.context import CareerContextStore
from career_agent.storage.working_notes import WorkingNotesStore
from conftest import enter_tool_profile


class SequenceDecisionMaker:
    def __init__(self, *decisions: AgentDecision) -> None:
        self.decisions = list(decisions)
        self.contexts = []

    def decide(self, context, tool_specs):
        self.contexts.append(context)
        return self.decisions.pop(0)


def _tool(name: str, arguments: dict | None = None) -> AgentDecision:
    return AgentDecision(
        action="tool_call",
        tool_call=ToolCall(name=name, arguments=arguments or {}),
    )


def _final() -> AgentDecision:
    return AgentDecision(action="final", message="")


def _runtime(
    tmp_path,
    *decisions,
    project_career_memory: bool = False,
    profile=None,
):
    context = CareerContextStore(tmp_path / "context.sqlite3")
    history = CareerHistoryStore(tmp_path / "career.sqlite3")
    notes = WorkingNotesStore(tmp_path / "working-notes")
    if profile is not None:
        enter_tool_profile(context, profile)
    return (
        MainAgentRuntime(
            context_manager=ContextManager(context, working_notes_store=notes),
            decision_maker=SequenceDecisionMaker(*decisions),
            tools=MainAgentToolRegistry(
                career_history_store=history,
                conversation_store=context,
                working_notes_store=notes,
            ),
            career_context_projector=(
                CareerContextProjector(history) if project_career_memory else None
            ),
        ),
        context,
        history,
    )


def test_a_turn_binds_the_career_memory_it_showed_the_model(tmp_path) -> None:
    """A committed turn must record which claims its prompt exposed.

    Deletion finds affected transcript text through that binding alone. If the
    commit records nothing, ``purge_derived_memory`` has nothing to match on
    and a tombstoned claim survives verbatim in the conversation.
    """

    _, _, history = _runtime(tmp_path)
    record = history.create_record(
        user_id="u1",
        record_type="internship",
        organization="Private Corp.",
        title="AI Intern",
        is_current=False,
    )
    history.confirm_evidence(
        user_id="u1",
        career_evidence_id=history.create_evidence(
            user_id="u1",
            career_record_id=record.id,
            claim="Built a private ranking prototype.",
            origin="user_input",
        ).id,
    )

    runtime, context_store, _ = _runtime(
        tmp_path,
        _final(),
        project_career_memory=True,
    )
    result = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="Tell me about the ranking prototype.",
    )

    exposed = result.career_memory_scope_keys
    assert exposed, "the turn showed career memory but recorded no exposure"

    context_store.purge_derived_memory(user_id="u1", scope_key=exposed[0])
    assert context_store.list_messages_after(
        user_id="u1",
        conversation_id="c1",
        after_sequence=0,
        limit=10,
    ) == ()


def test_tombstone_requires_readback_then_cleans_derived_memory(tmp_path) -> None:
    runtime, _, history = _runtime(tmp_path)
    record = history.create_record(
        user_id="u1",
        record_type="internship",
        organization="Private Corp.",
        title="AI Intern",
        is_current=False,
    )
    evidence = history.confirm_evidence(
        user_id="u1",
        career_evidence_id=history.create_evidence(
            user_id="u1",
            career_record_id=record.id,
            claim="Built a private ranking prototype.",
            origin="user_input",
        ).id,
    )
    context = CareerContextStore(tmp_path / "context.sqlite3")
    _, export_id, _ = MemoryReviewService(
        context_store=context,
        career_history_store=history,
    ).export(user_id="u1")
    notes = WorkingNotesStore(tmp_path / "working-notes")
    notes.replace(
        user_id="u1",
        markdown="- Built a private ranking prototype.",
        expected_revision="empty",
    )

    runtime, _, _ = _runtime(
        tmp_path,
        _tool(
            "propose_memory_tombstone",
            {
                "detail_ref": evidence.detail_ref,
                "reason": "Remove this internship detail permanently.",
            },
        ),
        _final(),
        profile="memory",
    )
    proposed = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="Delete that internship detail.",
    )

    assert history.get_evidence(
        user_id="u1", career_evidence_id=evidence.id
    ) is not None
    assert proposed.context.task.pending_memory_tombstone is not None

    runtime, _, history = _runtime(
        tmp_path,
        _tool("confirm_memory_tombstone"),
        _final(),
        profile="memory",
    )
    completed = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="I confirm the permanent deletion.",
    )

    assert history.get_evidence(
        user_id="u1", career_evidence_id=evidence.id
    ) is None
    assert completed.context.task.pending_memory_tombstone is None
    mutation = completed.tool_results[0]
    assert mutation.state == "memory_tombstoned"
    assert mutation.payload["lineage_size"] == 1
    assert mutation.payload["derived_cleanup"]["memory_review_items"] == 1
    assert mutation.payload["derived_cleanup"]["working_notes"] == 1
    retained = context.get_memory_review_export(
        user_id="u1",
        export_id=export_id,
    )
    assert retained is not None
    assert all(item["update_id"] != evidence.update_id for item in retained)
    assert all(item["value"] != evidence.claim for item in retained)
    assert notes.read(user_id="u1").markdown == ""
    assert "private ranking prototype" not in str(
        completed.context.model_context()
    ).casefold()


def test_amendment_requires_readback_and_refreshes_current_revision(tmp_path) -> None:
    _, _, history = _runtime(tmp_path)
    record = history.create_record(
        user_id="u1",
        record_type="project",
        title="Retriever",
    )
    original = history.confirm_evidence(
        user_id="u1",
        career_evidence_id=history.create_evidence(
            user_id="u1",
            career_record_id=record.id,
            claim="Assisted with ranking evaluation.",
            origin="user_input",
        ).id,
    )
    runtime, _, _ = _runtime(
        tmp_path,
        _tool(
            "propose_memory_amendment",
            {
                "detail_ref": original.detail_ref,
                "new_claim": "Led ranking evaluation.",
                "reason": "Correct ownership.",
            },
        ),
        _final(),
        profile="memory",
    )
    proposed = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="Change assisted to led.",
    )
    assert proposed.context.task.pending_memory_amendment is not None
    assert history.get_current_evidence(
        user_id="u1",
        scope_key=original.scope_key,
    ).claim == original.claim

    runtime, _, history = _runtime(
        tmp_path,
        _tool("confirm_memory_amendment"),
        _final(),
        profile="memory",
    )
    completed = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="Confirm that correction.",
    )

    current = history.get_current_evidence(
        user_id="u1",
        scope_key=original.scope_key,
    )
    assert current is not None
    assert current.claim == "Led ranking evaluation."
    assert current.revision == 2
    assert completed.context.task.pending_memory_amendment is None
    assert "Led ranking evaluation." in str(completed.context.model_context())


def test_tombstone_cannot_execute_in_the_turn_that_prepared_it(tmp_path) -> None:
    _, _, history = _runtime(tmp_path)
    record = history.create_record(
        user_id="u1",
        record_type="project",
        title="Private",
    )
    evidence = history.confirm_evidence(
        user_id="u1",
        career_evidence_id=history.create_evidence(
            user_id="u1",
            career_record_id=record.id,
            claim="Private claim.",
            origin="user_input",
        ).id,
    )
    runtime, _, history = _runtime(
        tmp_path,
        _tool(
            "propose_memory_tombstone",
            {
                "detail_ref": evidence.detail_ref,
                "reason": "Delete.",
            },
        ),
        _tool("confirm_memory_tombstone"),
        _final(),
        profile="memory",
    )
    premature = runtime._tools.invoke_atomic_tool(
        "confirm_memory_tombstone",
        {
            "user_id": "u1",
            "conversation_id": "c1",
            "proposal": MemoryTombstoneProposal(
                target_kind="career_evidence",
                detail_ref=evidence.detail_ref,
                reason="Delete.",
            ),
        },
    )
    assert premature.state == "memory_tombstone_confirmation_missing"

    result = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="Delete this claim.",
    )

    assert history.get_evidence(
        user_id="u1", career_evidence_id=evidence.id
    ) is not None
    assert [item.state for item in result.tool_results] == [
        "memory_tombstone_proposed"
    ]


def test_cleanup_failure_keeps_confirmation_for_idempotent_retry(
    tmp_path, monkeypatch
) -> None:
    _, _, history = _runtime(tmp_path)
    record = history.create_record(
        user_id="u1",
        record_type="project",
        title="Private",
    )
    evidence = history.confirm_evidence(
        user_id="u1",
        career_evidence_id=history.create_evidence(
            user_id="u1",
            career_record_id=record.id,
            claim="Private claim.",
            origin="user_input",
        ).id,
    )
    runtime, _, _ = _runtime(
        tmp_path,
        _tool(
            "propose_memory_tombstone",
            {"detail_ref": evidence.detail_ref, "reason": "Delete."},
        ),
        _final(),
        profile="memory",
    )
    runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="Delete this claim.",
    )

    runtime, context, _ = _runtime(
        tmp_path,
        _tool("confirm_memory_tombstone"),
        _final(),
        profile="memory",
    )

    def fail_cleanup(**_kwargs):
        raise OSError("simulated cleanup failure")

    monkeypatch.setattr(context, "purge_derived_memory", fail_cleanup)
    result = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="I confirm.",
    )

    assert result.tool_results[0].state == "memory_tombstone_cleanup_incomplete"
    assert result.context.task.pending_memory_tombstone is not None
    assert context.get_task("u1", "c1").pending_memory_tombstone is not None
    assert result.context.recent_messages
    assert result.context.user_message == "I confirm."

    retry_runtime, retry_context, _ = _runtime(
        tmp_path,
        _tool("confirm_memory_tombstone"),
        _final(),
        profile="memory",
    )
    retried = retry_runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="Retry the cleanup.",
    )

    assert retried.tool_results[0].state == "memory_tombstoned"
    assert retried.context.task.pending_memory_tombstone is None
    assert retry_context.get_task("u1", "c1").pending_memory_tombstone is None


def test_working_notes_unlink_failure_is_a_retriable_cleanup_state(
    tmp_path,
    monkeypatch,
) -> None:
    _, _, history = _runtime(tmp_path)
    record = history.create_record(
        user_id="u1",
        record_type="project",
        title="Private project",
    )
    evidence = history.confirm_evidence(
        user_id="u1",
        career_evidence_id=history.create_evidence(
            user_id="u1",
            career_record_id=record.id,
            claim="Built a private prototype.",
            origin="user_input",
        ).id,
    )
    runtime, _, _ = _runtime(
        tmp_path,
        _tool(
            "propose_memory_tombstone",
            {"detail_ref": evidence.detail_ref, "reason": "Delete it."},
        ),
        _final(),
        profile="memory",
    )
    runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="Delete the private prototype.",
    )

    def fail_clear(self, *, user_id):
        raise OSError(f"cannot unlink notes for {user_id}")

    monkeypatch.setattr(WorkingNotesStore, "clear", fail_clear)
    runtime, context, _ = _runtime(
        tmp_path,
        _tool("confirm_memory_tombstone"),
        _final(),
        profile="memory",
    )
    result = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="I confirm.",
    )

    mutation = result.tool_results[0]
    assert mutation.state == "memory_tombstone_cleanup_incomplete"
    assert mutation.payload["working_notes_cleared"] is False
    assert mutation.payload["cleanup_incomplete"] == "working_notes"
    assert context.get_task("u1", "c1").pending_memory_tombstone is not None
