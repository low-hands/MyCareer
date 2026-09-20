import argparse
from datetime import datetime, timedelta, timezone

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import OwnerSettingsContext
from career_agent.agent.questionnaire_contracts import PendingQuestionnaire, UserQuestion
from career_agent.api.reads import WorkspaceReader
from career_agent.storage.capability_confirmations import SQLiteCapabilityConfirmationStore
from career_agent.storage.context import CareerContextStore


def _args(tmp_path):
    return argparse.Namespace(
        job_store=str(tmp_path / "jobs.sqlite3"),
        resume_store=str(tmp_path / "resumes.sqlite3"),
        application_store=str(tmp_path / "applications.sqlite3"),
        calendar_store=str(tmp_path / "calendar.sqlite3"),
        job_research_store=str(tmp_path / "research.sqlite3"),
        context_store=str(tmp_path / "context.sqlite3"),
        mock_interview_store=str(tmp_path / "mock.sqlite3"),
    )


def test_reload_hides_a_pending_confirmation_from_an_old_policy_revision(tmp_path):
    args = _args(tmp_path)
    context_store = CareerContextStore(tmp_path / "context.sqlite3")
    manager = ContextManager(context_store)
    manager.load_for_turn(user_id="u1", conversation_id="c1", user_message="开始")
    initial = OwnerSettingsContext()
    current = context_store.update_owner_settings(
        user_id="u1",
        desired=initial.model_copy(
            update={
                "behavior_policy": initial.behavior_policy.model_copy(
                    update={"application_confirmation": "always_ask"}
                )
            }
        ),
        expected_revision=0,
        actor_type="cli",
        actor_id="test",
    )
    confirmations = SQLiteCapabilityConfirmationStore(tmp_path / "context.sqlite3")
    stale = confirmations.seal(
        user_id="u1", conversation_id="c1", capability="create_application",
        display_summary="旧批准", arguments={"job": 1}, policy_revision=0,
    )

    transcript = WorkspaceReader(args).conversation_messages(
        user_id="u1", conversation_id="c1"
    )

    assert transcript.pending_interaction is None
    assert confirmations.get(stale.confirmation_id).status == "PENDING"
    assert current.behavior_policy.revision == 1


def test_reload_restores_the_same_pending_questionnaire(tmp_path):
    args = _args(tmp_path)
    context_store = CareerContextStore(tmp_path / "context.sqlite3")
    manager = ContextManager(context_store)
    manager.load_for_turn(user_id="u1", conversation_id="c1", user_message="优化简历")
    now = datetime.now(timezone.utc)
    pending = PendingQuestionnaire(
        interaction_id="interaction_1234567890abcdef1234",
        prompt="请补充信息",
        questions=(
            UserQuestion(question_id="q1", prompt="问题一", kind="free_text"),
            UserQuestion(question_id="q2", prompt="问题二", kind="free_text"),
        ),
        created_at=now, expires_at=now + timedelta(days=7), active_workflow="none",
    )
    task = manager.get_task(user_id="u1", conversation_id="c1")
    context_store.upsert_task(
        user_id="u1", conversation_id="c1",
        task=task.model_copy(update={"pending_questionnaire": pending}),
    )
    first = WorkspaceReader(args).conversation_messages(user_id="u1", conversation_id="c1")
    second = WorkspaceReader(args).conversation_messages(user_id="u1", conversation_id="c1")
    assert first.pending_interaction == second.pending_interaction
    assert first.pending_interaction is not None
    assert first.pending_interaction.kind == "questionnaire"
    assert first.pending_interaction.interaction_id == pending.interaction_id
    assert WorkspaceReader(args).conversation_messages(
        user_id="u2", conversation_id="c1"
    ).pending_interaction is None
    assert WorkspaceReader(args).conversation_messages(
        user_id="u1", conversation_id="c2"
    ).pending_interaction is None
