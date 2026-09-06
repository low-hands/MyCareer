"""Which tools should be on the menu this turn, from the current task state.

This is menu, not capability. Hiding a tool here only removes it from what the
decision model is *offered* this turn; it is never removed from the registry,
whose handler still exists and still enforces its own preconditions through
argument projection. So hiding too little is harmless — the projection guard
fires with a soft result later — while hiding too much silently strands a legal
path. The table below therefore errs toward offering: a tool is hidden only when
there is provably no object it could name right now.

The one rule with no such argument is the reference-index readbacks. Their
reachability lives in the conversation window (``recent_messages`` /
``archived_resources``), not in ``ConversationTaskState``, so the task state can
neither prove nor disprove that some report is still reachable. They are never
hidden, even on a cold turn.
"""

from __future__ import annotations

from collections.abc import Callable

from career_agent.agent.main_agent_contracts import MainAgentContext, ConversationTaskState

Precondition = Callable[[ConversationTaskState], bool]


# Readbacks reached through a resource handle: the handle is derived per conversation
# into the conversation's resource references, which live outside task state.
_REFERENCE_READBACKS = frozenset(
    {
        "get_job_research",
        "get_interview_preparation",
        "get_mock_interview_result",
    }
)


def _reachable_via_job(task: ConversationTaskState) -> bool:
    return bool(task.active_job_posting_id or task.saved_job_candidates)


def _reachable_via_resume_version(task: ConversationTaskState) -> bool:
    return bool(task.active_resume_version_id or task.resume_version_candidates)


def _reachable_via_application(task: ConversationTaskState) -> bool:
    return bool(task.active_application_id or task.application_candidates)


def _reachable_via_interview(task: ConversationTaskState) -> bool:
    return bool(task.active_interview_round_id or task.interview_candidates)


def _reachable_via_action_item(task: ConversationTaskState) -> bool:
    return bool(task.active_action_item_id or task.action_candidates)


PRECONDITIONS: dict[str, Precondition] = {
    # Saved jobs and their derived runs.
    "get_saved_job": _reachable_via_job,
    "research_job": _reachable_via_job,
    "compare_saved_jobs": lambda t: bool(t.saved_job_candidates),
    # Resumes and their immutable versions.
    "get_resume_metadata": lambda t: bool(t.resume_candidates),
    "analyze_resume": _reachable_via_resume_version,
    "export_resume_artifact": lambda t: bool(t.active_resume_version_id),
    # Active-object-only analysis / match / tailoring chain.
    "get_resume_analysis": lambda t: bool(t.active_resume_analysis_id),
    "get_resume_job_match": lambda t: bool(t.active_resume_job_match_id),
    "draft_resume_tailoring": lambda t: bool(t.active_resume_job_match_id),
    "get_resume_tailoring_draft": lambda t: bool(
        t.active_resume_tailoring_draft_id
    ),
    "review_resume_tailoring": lambda t: bool(
        t.active_resume_tailoring_draft_id
    ),
    "revise_resume_tailoring": lambda t: bool(
        t.active_resume_tailoring_draft_id
    ),
    "finalize_resume_tailoring": lambda t: bool(
        t.active_resume_tailoring_draft_id
    ),
    # Matching and applying need a job *and* a resume version.
    "match_resume_to_job": lambda t: (
        _reachable_via_job(t) and _reachable_via_resume_version(t)
    ),
    "create_application": lambda t: (
        _reachable_via_job(t) and _reachable_via_resume_version(t)
    ),
    # Applications.
    "get_application": _reachable_via_application,
    "update_application_status": _reachable_via_application,
    "create_interview": _reachable_via_application,
    # Interviews.
    "get_interview": _reachable_via_interview,
    "update_interview": _reachable_via_interview,
    "complete_interview": _reachable_via_interview,
    "record_interview_retro": _reachable_via_interview,
    "prepare_interview": lambda t: bool(
        _reachable_via_interview(t) or t.action_candidates
    ),
    # Email events.
    "resolve_email_event": lambda t: bool(t.email_event_candidates),
    # Stated intent can only be confirmed once a proposal has been read back.
    "confirm_job_intent": lambda t: t.pending_job_intent_update is not None,
    # Action center items.
    "complete_action_item": _reachable_via_action_item,
    "dismiss_action_item": _reachable_via_action_item,
    "snooze_action_item": _reachable_via_action_item,
    # Calendar.
    "prepare_interview_calendar_sync": _reachable_via_interview,
    "get_calendar_proposal": lambda t: bool(t.active_calendar_proposal_id),
    "execute_calendar_proposal": lambda t: bool(t.active_calendar_proposal_id),
    # Job research retry.
    "retry_job_research": lambda t: bool(t.active_job_research_run_id),
    # Mock interview.
    "start_mock_interview": lambda t: bool(
        _reachable_via_application(t) or _reachable_via_interview(t)
    ),
    "restart_mock_interview": lambda t: (
        t.active_workflow == "mock_interview"
        and t.phase
        in {
            "mock_interview_checkpoint_missing",
            "mock_interview_graph_incompatible",
        }
    ),
}


def reachable(name: str, task: ConversationTaskState) -> bool:
    """Whether ``name`` should be offered given ``task``.

    A tool with no declared precondition is always offered. Reference-index
    readbacks are always offered because their reachability lives in the
    conversation window rather than in task state.
    """
    if name in _REFERENCE_READBACKS:
        return True
    precondition = PRECONDITIONS.get(name)
    return precondition(task) if precondition is not None else True


def reachable_in_context(name: str, context: MainAgentContext) -> bool:
    """Context-aware reachability for capabilities whose anchor is not task state."""
    if name == "read_conversation_span":
        return bool(
            context.through_sequence
            or (
                context.recent_from_sequence is not None
                and context.recent_from_sequence > 1
            )
        )
    return reachable(name, context.task)
