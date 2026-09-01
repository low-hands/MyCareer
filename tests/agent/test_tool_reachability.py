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
    _REFERENCE_READBACKS,
    reachable,
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