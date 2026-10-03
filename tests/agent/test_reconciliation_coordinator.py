from career_agent.agent.execution.reconciliation import ReconciliationCoordinator


class CountingReconciler:
    def __init__(self) -> None:
        self.calls = []

    def reconcile_user(self, *, user_id: str):
        self.calls.append(user_id)


def test_reconciliation_coordinator_sweeps_once_until_invalidated() -> None:
    reconciler = CountingReconciler()
    coordinator = ReconciliationCoordinator(
        context_manager=None,
        action_execution_store=None,
        episode_reconciler=reconciler,
    )

    coordinator.reconcile_episodes("user-1")
    coordinator.reconcile_episodes("user-1")
    coordinator.invalidate_episode_reconciliation("user-1")
    coordinator.reconcile_episodes("user-1")

    assert reconciler.calls == ["user-1", "user-1"]


def test_reconciliation_coordinator_is_a_noop_without_recovery_stores() -> None:
    coordinator = ReconciliationCoordinator(
        context_manager=None,
        action_execution_store=None,
        episode_reconciler=None,
    )

    coordinator.reconcile_episodes("user-1")
    coordinator.invalidate_episode_reconciliation("user-1")
