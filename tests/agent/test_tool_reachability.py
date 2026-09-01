"""The reachability table that decides which tools are offered this turn.

These tests hold two things:

1. The table can never hide a tool that has no object to require, and never
   names a tool the registry does not actually expose.
2. The one class that must stay offered no matter what the task state says is
   the reference-index readback, whose reachability lives in the conversation
   window rather than in task state.
"""

from __future__ import annotations

from career_agent.agent.main_agent_contracts import ConversationTaskState
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.agent.tool_reachability import (
    PRECONDITIONS,
    REROUTE_FIELDS,
    _REFERENCE_READBACKS,
    reachable,
    reroutable,
)


def _registry() -> MainAgentToolRegistry:
    return MainAgentToolRegistry(
        job_repository=object(),
        career_profile_store=object(),
        resume_store=object(),
        resume_analysis_service=object(),
        resume_job_match_service=object(),
        resume_tailoring_service=object(),
        resume_export_service=object(),
        application_service=object(),
        email_tracking_service=object(),
        interview_service=object(),
        interview_preparation_service=object(),
        action_center_service=object(),
        calendar_service=object(),
        mock_interview_graph=object(),
        mock_interview_store=object(),
        job_research_service=object(),
        job_comparison_service=object(),
    )


def test_every_precondition_names_a_real_tool() -> None:
    known = set(_registry().names)
    assert set(PRECONDITIONS) <= known
    assert _REFERENCE_READBACKS <= known


def test_a_cold_task_offers_only_the_tools_with_no_requirement() -> None:
    task = ConversationTaskState()
    # Always-on tools stay on.
    assert reachable("open_job_search", task)
    assert reachable("find_saved_jobs", task)
    assert reachable("list_action_items", task)
    assert reachable("get_daily_brief", task)
    # A tool whose object is absent is hidden.
    assert not reachable("get_saved_job", task)
    assert not reachable("research_job", task)
    assert not reachable("match_resume_to_job", task)
    assert not reachable("execute_calendar_proposal", task)
    assert not reachable("confirm_job_intent", task)


def test_a_reference_readback_is_never_hidden_by_task_state() -> None:
    task = ConversationTaskState()
    # Reachability lives in the conversation window, which task state does not
    # carry. Hiding these on a cold turn would strand the readback path.
    assert reachable("get_job_research", task)
    assert reachable("get_interview_preparation", task)
    assert reachable("get_mock_interview_result", task)


def test_a_selected_candidate_makes_the_detail_tool_reachable() -> None:
    task = ConversationTaskState(
        saved_job_candidates=(
            __import__("career_agent.agent.main_agent_contracts", fromlist=["SavedJobCandidateContextItem"]).SavedJobCandidateContextItem(
                job_posting_id="job-1",
                title="AI Engineer",
                company_name="Acme",
            ),
        ),
    )
    assert reachable("get_saved_job", task)
    assert reachable("research_job", task)


def test_restart_mock_interview_only_after_a_stuck_run() -> None:
    assert not reachable(
        "restart_mock_interview",
        ConversationTaskState(active_workflow="mock_interview", run_id="r", phase="mock_interview_running"),
    )
    assert reachable(
        "restart_mock_interview",
        ConversationTaskState(active_workflow="mock_interview", run_id="r", phase="mock_interview_checkpoint_missing"),
    )


def test_retry_research_requires_the_active_run_its_projection_will_name() -> None:
    assert not reachable("retry_job_research", ConversationTaskState())
    assert not reachable(
        "retry_job_research",
        ConversationTaskState(job_research_status="failed"),
    )
    assert reachable(
        "retry_job_research",
        ConversationTaskState(
            active_job_research_run_id="run-1",
            job_research_status="current",
        ),
    )
    assert reachable(
        "retry_job_research",
        ConversationTaskState(
            active_job_research_run_id="run-1",
            job_research_status="failed",
        ),
    )


def test_reroutable_requires_candidates_but_not_active_objects() -> None:
    """Recovery draws from candidate lists; active pointers are not enough."""
    task = ConversationTaskState()
    assert not reroutable("create_interview", task)
    assert not reroutable("get_saved_job", task)
    # An active pointer alone cannot recover an out-of-range selector.
    active_only = ConversationTaskState(active_application_id="app-1")
    assert not reroutable("create_interview", active_only)


def test_that_a_selector_list_makes_the_tool_reroutable() -> None:
    from career_agent.agent.main_agent_contracts import (
        ApplicationCandidateContextItem,
        SavedJobCandidateContextItem,
        InterviewCandidateContextItem,
    )

    job_task = ConversationTaskState(
        saved_job_candidates=(
            SavedJobCandidateContextItem(
                job_posting_id="job-1",
                title="AI Engineer",
                company_name="Acme",
            ),
        ),
    )
    assert reroutable("get_saved_job", job_task)
    assert reroutable("research_job", job_task)

    interview_task = ConversationTaskState(
        interview_candidates=(
            InterviewCandidateContextItem(
                interview_round_id="round-1",
                application_id="app-1",
                sequence_number=1,
                status="scheduled",
            ),
        ),
    )
    assert reroutable("get_interview", interview_task)
    assert reroutable("start_mock_interview", interview_task)

    app_task = ConversationTaskState(
        application_candidates=(
            ApplicationCandidateContextItem(
                application_id="app-1",
                title="算法",
                company_name="Acme",
                status="submitted",
            ),
        ),
    )
    assert reroutable("create_interview", app_task)


def test_selector_reroute_covers_the_historically_missing_entries() -> None:
    """The four tools whose reroute paths were absent before the merge.

    Each resolves a ``*_selection_index`` against a candidate list in its
    projection, so an out-of-range selector has a real recovery path. Kept as
    direct assertions: a source-scanning derivation kept false-positiving over
    sibling ``name in {…}`` branches, and four explicit entries cost less than
    the scanner's maintenance.
    """
    assert REROUTE_FIELDS["propose_job_intent"] == ("target_role_candidates",)
    assert REROUTE_FIELDS["list_resumes"] == ("target_role_candidates",)
    assert set(REROUTE_FIELDS["start_mock_interview"]) == {
        "application_candidates",
        "interview_candidates",
    }
    assert REROUTE_FIELDS["get_mock_interview_result"] == (
        "application_candidates",
    )
    # The entity-keyed path has no candidates to re-select from, and the
    # readback gets its recovery from the same saved-job list its selector uses.
    assert REROUTE_FIELDS["get_job_research"] == ("saved_job_candidates",)


def test_reroute_fields_only_name_candidates_consumed_by_the_projector() -> None:
    assert REROUTE_FIELDS["create_application"] == (
        "saved_job_candidates",
        "resume_version_candidates",
    )
    assert REROUTE_FIELDS["get_interview_preparation"] == (
        "interview_candidates",
    )
