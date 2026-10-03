from career_agent.agent.capabilities.executor import CapabilityExecutor as LegacyExecutor
from career_agent.agent.execution.action_ledger import ActionLedger
from career_agent.agent.execution.capability_executor import CapabilityExecutor
from career_agent.agent.contracts.main_agent import ToolObservation
from career_agent.storage.action_executions import RESULT_STATE_RECEIPT_KEY


def test_legacy_capability_executor_import_reexports_execution_class() -> None:
    assert LegacyExecutor is CapabilityExecutor


def test_action_ledger_receipt_keeps_only_bounded_scalar_reducer_data() -> None:
    result = ToolObservation(
        tool_name="create_application",
        state="application_ready",
        message="已创建投递记录。",
        execution_outcome="committed",
        payload={
            "application_id": "application-1",
            "created": True,
            "nested": {"secret": "not a receipt"},
            "large": "x" * 501,
        },
    )

    receipt = ActionLedger.execution_receipt(result)

    assert receipt == {
        RESULT_STATE_RECEIPT_KEY: "application_ready",
        "application_id": "application-1",
        "created": True,
    }


def test_action_ledger_is_disabled_without_a_store() -> None:
    assert ActionLedger(store=None, policy_epoch=1).enabled is False
