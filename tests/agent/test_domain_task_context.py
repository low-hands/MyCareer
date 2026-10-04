from career_agent.agent.contracts.candidates import (
    ApplicationCandidateContextItem,
    EmailEventCandidateContextItem,
)
from career_agent.agent.contracts.task_state import ConversationTaskState


def test_task_state_has_only_orchestration_and_grouped_domain_fields() -> None:
    assert tuple(ConversationTaskState.model_fields) == (
        "workflow",
        "tool_profile",
        "pending_interaction",
        "domain_context",
    )


def test_legacy_application_fields_migrate_into_the_domain_context() -> None:
    candidate = ApplicationCandidateContextItem(
        application_id="application-1",
        title="AI Engineer",
        company_name="Acme",
        status="submitted",
    )
    email = EmailEventCandidateContextItem(
        email_event_id="email-1",
        event_type="interview_invitation",
        status="pending_confirmation",
        summary="Interview invitation",
    )

    task = ConversationTaskState(
        active_application_id="application-1",
        active_application_status="submitted",
        application_candidates=(candidate,),
        email_event_candidates=(email,),
        email_sync_phase="email_sync_complete",
    )

    assert task.active_application_id == "application-1"
    assert task.active_application_status == "submitted"
    assert task.application_candidates == (candidate,)
    assert task.email_event_candidates == (email,)
    assert task.email_sync_phase == "email_sync_complete"
    payload = task.model_dump(mode="json")
    assert payload["domain_context"]["application"]["active_id"] == "application-1"
    assert "active_application_id" not in payload
    assert task.active_resource_flags()["has_active_application"] is True


def test_application_updates_preserve_other_task_and_domain_state() -> None:
    task = ConversationTaskState(active_job_posting_id="job-1")

    updated = task.update_application_context(
        active_id="application-1",
        active_status="interviewing",
    )

    assert updated.active_job_posting_id == "job-1"
    assert updated.active_application_id == "application-1"
    assert updated.active_application_status == "interviewing"
    assert task.active_application_id is None


def test_new_domain_context_wins_over_stale_flat_compatibility_values() -> None:
    task = ConversationTaskState.model_validate(
        {
            "domain_context": {
                "application": {
                    "active_id": "application-new",
                    "active_status": "interviewing",
                }
            },
            "active_application_id": "application-old",
            "active_application_status": "submitted",
        }
    )

    assert task.active_application_id == "application-new"
    assert task.active_application_status == "interviewing"


def test_legacy_interview_calendar_and_action_fields_migrate_together() -> None:
    task = ConversationTaskState(
        active_interview_round_id="interview-1",
        active_interview_preparation_id="preparation-1",
        active_calendar_proposal_id="proposal-1",
        active_action_item_id="action-1",
    )

    assert task.domain_context.interview.active_round_id == "interview-1"
    assert task.domain_context.interview.active_preparation_id == "preparation-1"
    assert task.domain_context.interview.active_calendar_proposal_id == "proposal-1"
    assert task.domain_context.action_center.active_id == "action-1"
    flags = task.active_resource_flags()
    assert flags["has_active_interview_round"] is True
    assert flags["has_active_interview_preparation"] is True
    assert flags["has_active_calendar_proposal"] is True
    assert flags["has_active_action_item"] is True
    payload = task.model_dump(mode="json")
    assert "active_interview_round_id" not in payload
    assert "active_action_item_id" not in payload


def test_legacy_job_and_resume_fields_migrate_into_owned_contexts() -> None:
    task = ConversationTaskState(
        active_job_posting_id="job-1",
        active_jd_snapshot_id="snapshot-1",
        active_job_analysis_id="analysis-1",
        active_job_research_run_id="research-run-1",
        active_job_research_report_id="research-report-1",
        active_resume_job_match_id="match-1",
        active_resume_tailoring_draft_id="draft-1",
        active_resume_version_id="version-1",
        active_resume_artifact_id="artifact-1",
    )

    assert task.domain_context.job.active_posting_id == "job-1"
    assert task.domain_context.job.active_jd_snapshot_id == "snapshot-1"
    assert task.domain_context.job.active_analysis_id == "analysis-1"
    assert task.domain_context.job.active_research_run_id == "research-run-1"
    assert task.domain_context.job.active_research_report_id == "research-report-1"
    assert task.domain_context.resume.active_job_match_id == "match-1"
    assert task.domain_context.resume.active_tailoring_draft_id == "draft-1"
    assert task.domain_context.resume.active_version_id == "version-1"
    assert task.domain_context.resume.active_artifact_id == "artifact-1"
    payload = task.model_dump(mode="json")
    assert "active_job_posting_id" not in payload
    assert "active_resume_version_id" not in payload
