"""A suspended workflow run can be traced back to the conversation holding it."""

from __future__ import annotations

from career_agent.agent.main_agent_contracts import ConversationTaskState
from career_agent.storage.context import CareerContextStore


def _store_with_runs(tmp_path) -> CareerContextStore:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    store.upsert_task(
        user_id="u1",
        conversation_id="c-interview",
        task=ConversationTaskState().enter_workflow(
            "mock_interview", run_id="mock-1", phase="mock_interview_answer_required"
        ),
    )
    store.upsert_task(
        user_id="u1",
        conversation_id="c-jobs",
        task=ConversationTaskState().enter_workflow(
            "job_discovery", run_id="mock-1", phase="searching"
        ),
    )
    store.upsert_task(
        user_id="u2",
        conversation_id="c-other-owner",
        task=ConversationTaskState().enter_workflow(
            "mock_interview", run_id="mock-2", phase="mock_interview_answer_required"
        ),
    )
    return store


def test_run_resolves_to_the_conversation_that_holds_it(tmp_path) -> None:
    store = _store_with_runs(tmp_path)

    assert (
        store.conversation_owning_run(
            user_id="u1", workflow="mock_interview", run_id="mock-1"
        )
        == "c-interview"
    )


def test_run_is_not_found_across_workflows_or_owners(tmp_path) -> None:
    store = _store_with_runs(tmp_path)

    assert (
        store.conversation_owning_run(
            user_id="u1", workflow="mock_interview", run_id="mock-2"
        )
        is None
    )
    assert (
        store.conversation_owning_run(
            user_id="u1", workflow="job_discovery", run_id="mock-9"
        )
        is None
    )


def test_a_conversation_that_left_the_workflow_no_longer_owns_the_run(tmp_path) -> None:
    store = _store_with_runs(tmp_path)
    store.upsert_task(
        user_id="u1",
        conversation_id="c-interview",
        task=store.get_task("u1", "c-interview").leave_workflow(),
    )

    assert (
        store.conversation_owning_run(
            user_id="u1", workflow="mock_interview", run_id="mock-1"
        )
        is None
    )
