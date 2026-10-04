from career_agent.agent.contracts.observations import ToolObservation
from career_agent.agent.execution.recovery import OperationReconcilerRegistry
from career_agent.storage.operation_journal import SQLiteOperationJournal


def test_reconciler_settles_an_uncertain_external_operation(tmp_path) -> None:
    store = SQLiteOperationJournal(tmp_path / "operations.sqlite3")
    operation, _ = store.prepare(
        user_id="user-1",
        conversation_id="conversation-1",
        turn_id="turn-1",
        anchor="request-1",
        request_id="request-1",
        write_slot=0,
        tool_name="execute_calendar_proposal",
        fingerprint="fingerprint-1",
        policy_epoch=1,
        replay_allowed=True,
        recovery_policy="reconcile",
        input_references={"proposal_id": "proposal-1"},
    )
    store.mark_running(action_id=operation.operation_id)
    store.require_reconciliation(
        action_id=operation.operation_id,
        error_code="TIMEOUT",
        error_detail="provider outcome unknown",
    )
    seen = []

    def reconcile(pending):
        seen.append(pending)
        return ToolObservation(
            tool_name="execute_calendar_proposal",
            state="calendar_sync_complete",
            message="reconciled",
            payload={
                "calendar_link_id": "link-1",
                "external_event_id": "event-1",
            },
            execution_outcome="committed",
        )

    recovered = OperationReconcilerRegistry(
        {"execute_calendar_proposal": reconcile}
    ).recover(store=store, user_id="user-1")

    assert [item.operation_id for item in recovered] == [operation.operation_id]
    assert seen[0].input_references == {"proposal_id": "proposal-1"}
    assert recovered[0].phase == "EFFECT_COMMITTED"
    assert recovered[0].output_references == {
        "calendar_link_id": "link-1",
        "external_event_id": "event-1",
    }
    assert recovered[0].external_reference == "event-1"


def test_reconciler_leaves_unknown_outcome_open(tmp_path) -> None:
    store = SQLiteOperationJournal(tmp_path / "operations.sqlite3")
    operation, _ = store.prepare(
        user_id="user-1",
        conversation_id="conversation-1",
        turn_id="turn-1",
        anchor="request-1",
        request_id="request-1",
        write_slot=0,
        tool_name="execute_calendar_proposal",
        fingerprint="fingerprint-1",
        policy_epoch=1,
        recovery_policy="reconcile",
        input_references={"proposal_id": "proposal-1"},
    )
    store.mark_running(action_id=operation.operation_id)

    registry = OperationReconcilerRegistry(
        {
            "execute_calendar_proposal": lambda pending: ToolObservation(
                tool_name=pending.capability,
                state="calendar_write_failed",
                message="still unknown",
                execution_outcome="unknown",
            )
        }
    )

    assert registry.recover(store=store, user_id="user-1") == ()
    assert store.get(action_id=operation.operation_id).status == "PENDING"
