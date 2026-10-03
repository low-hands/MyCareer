from types import SimpleNamespace

from career_agent.agent.runtime.checkpoint_lifecycle import CheckpointLifecycle


class RecordingCheckpointer:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.deleted: list[str] = []

    def delete_thread(self, thread_id: str) -> None:
        if self.fail:
            raise RuntimeError("checkpoint store unavailable")
        self.deleted.append(thread_id)


def _result(*, questionnaire: object | None):
    return SimpleNamespace(
        context=SimpleNamespace(
            conversation_id="c1",
            profile=SimpleNamespace(user_id="u1"),
            task=SimpleNamespace(pending_questionnaire=questionnaire),
        )
    )


def test_questionnaire_checkpoint_is_retained_until_resume() -> None:
    checkpointer = RecordingCheckpointer()
    lifecycle = CheckpointLifecycle(
        checkpointer=checkpointer,
        thread_id=lambda **kwargs: "thread-1",
    )

    lifecycle.settle(_result(questionnaire=object()))

    assert checkpointer.deleted == []


def test_settled_checkpoint_is_deleted_by_stable_thread_id() -> None:
    checkpointer = RecordingCheckpointer()
    captured = []
    lifecycle = CheckpointLifecycle(
        checkpointer=checkpointer,
        thread_id=lambda **kwargs: captured.append(kwargs) or "thread-1",
    )

    lifecycle.settle(_result(questionnaire=None))

    assert captured == [{"user_id": "u1", "conversation_id": "c1"}]
    assert checkpointer.deleted == ["thread-1"]


def test_checkpoint_cleanup_failure_does_not_fail_a_committed_turn(caplog) -> None:
    lifecycle = CheckpointLifecycle(
        checkpointer=RecordingCheckpointer(fail=True),
        thread_id=lambda **kwargs: "thread-1",
    )

    lifecycle.settle(_result(questionnaire=None))

    assert "failed to delete settled main graph checkpoint" in caplog.text
