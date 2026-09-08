from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    MemoryTombstoneProposal,
    ToolCall,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.storage.career_history import CareerHistoryStore
from career_agent.storage.context import CareerContextStore


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


def _runtime(tmp_path, *decisions):
    context = CareerContextStore(tmp_path / "context.sqlite3")
    history = CareerHistoryStore(tmp_path / "career.sqlite3")
    return (
        MainAgentRuntime(
            context_manager=ContextManager(context),
            decision_maker=SequenceDecisionMaker(*decisions),
            tools=MainAgentToolRegistry(
                career_history_store=history,
                conversation_store=context,
            ),
        ),
        context,
        history,
    )


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
    assert mutation.payload["cleanup_status"] == "completed"
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


def test_cleanup_failure_clears_the_consumed_confirmation(
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
    )

    def fail_cleanup(**_kwargs):
        raise OSError("simulated cleanup failure")

    monkeypatch.setattr(context, "purge_derived_memory", fail_cleanup)
    result = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="I confirm.",
    )

    assert result.tool_results[0].state == "memory_tombstone_cleanup_pending"
    assert result.context.task.pending_memory_tombstone is None
    assert context.get_task("u1", "c1").pending_memory_tombstone is None
    assert result.context.recent_messages
    assert result.context.user_message == "I confirm."
