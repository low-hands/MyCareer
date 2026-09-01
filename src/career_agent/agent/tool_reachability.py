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

from career_agent.agent.main_agent_contracts import ConversationTaskState

Precondition = Callable[[ConversationTaskState], bool]


# Readbacks reached through ``reference_index``: the handle is a turn-local index
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


# Whether a refused projection can be resolved by re-selecting from a candidate
# list. Kept beside the menu preconditions so both questions share one
# declaration of what a tool's selector can draw from: adding a tool here vs
# above without the other is the drift this single file exists to prevent.
# Reference readbacks are deliberately absent from the menu table (they are
# never hidden) but can still recover from an out-of-range selector, so they
# appear here alone.
REROUTE_FIELDS: dict[str, tuple[str, ...]] = {
    "get_job_research": ("saved_job_candidates",),
    "get_saved_job": ("saved_job_candidates",),
    "research_job": ("saved_job_candidates",),
    "compare_saved_jobs": ("saved_job_candidates",),
    "get_resume_metadata": ("resume_candidates",),
    "analyze_resume": ("resume_version_candidates",),
    "match_resume_to_job": ("saved_job_candidates", "resume_version_candidates"),
    "create_application": (
        "saved_job_candidates",
        "resume_version_candidates",
    ),
    "propose_job_intent": ("target_role_candidates",),
    "list_resumes": ("target_role_candidates",),
    "get_application": ("application_candidates",),
    "update_application_status": ("application_candidates",),
    "create_interview": ("application_candidates",),
    "get_interview": ("interview_candidates",),
    "update_interview": ("interview_candidates",),
    "complete_interview": ("interview_candidates",),
    "record_interview_retro": ("interview_candidates",),
    "prepare_interview": ("interview_candidates", "action_candidates"),
    "resolve_email_event": ("email_event_candidates",),
    "complete_action_item": ("action_candidates",),
    "dismiss_action_item": ("action_candidates",),
    "snooze_action_item": ("action_candidates",),
    "prepare_interview_calendar_sync": (
        "calendar_account_candidates",
        "interview_candidates",
    ),
    "start_mock_interview": ("application_candidates", "interview_candidates"),
    "get_mock_interview_result": ("application_candidates",),
    "get_interview_preparation": ("interview_candidates",),
}


def reroutable(name: str, task: ConversationTaskState) -> bool:
    """Whether a refused ``name`` can recover by drawing from current candidates.

    Tools absent from ``REROUTE_FIELDS`` have no candidate path: only the user
    can supply the object, so a refusal must end the turn. Reference readbacks
    may still be listed — their menu presence is unconditional, but an
    out-of-range selector can be fixed against a candidate list if one exists.
    """
    fields = REROUTE_FIELDS.get(name, ())
    return any(bool(getattr(task, field, ())) for field in fields)


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
