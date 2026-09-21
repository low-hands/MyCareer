"""Task-state reducers for atomic Main Agent tools.

Each reducer owns one tool's effect on ``ConversationTaskState`` and is keyed by
tool name in ``ATOMIC_TASK_REDUCERS``, so adding a capability means registering
an entry rather than extending a branch chain inside the runtime. A reducer only
runs when the result state is one the entry accepts, which keeps the
"did this tool actually succeed" check next to the fields it guards.
"""

from __future__ import annotations

from datetime import datetime, timezone

from typing import Any, Callable

from career_agent.agent.main_agent_contracts import (
    ActionCandidateContextItem,
    ActiveSavedJobContextItem,
    ApplicationCandidateContextItem,
    CalendarAccountCandidateContextItem,
    CareerFactProposal,
    ConstraintRetirementProposal,
    FreeTextPreferenceConfirmationProposal,
    JobIntentUpdate,
    MemoryAmendmentProposal,
    MemoryTombstoneProposal,
    ConversationTaskState,
    EmailEventCandidateContextItem,
    InterviewCandidateContextItem,
    ResumeCandidateContextItem,
    ResumeVersionCandidateContextItem,
    SavedJobCandidateContextItem,
    TargetRoleCandidateContextItem,
    TOOL_PROFILE_NAMES,
    ToolResult,
)

TaskReducer = Callable[[ConversationTaskState, ToolResult], ConversationTaskState]


class ReducerEntry:
    """One tool's reducer plus the result states that may trigger it."""

    __slots__ = ("states", "reduce")

    def __init__(self, states: frozenset[str], reduce: TaskReducer) -> None:
        self.states = states
        self.reduce = reduce

    def applies_to(self, state: str) -> bool:
        # An empty set means the tool has a single meaningful outcome and the
        # reducer is safe for any state it reports.
        return not self.states or state in self.states


def _items(result: ToolResult, key: str = "items") -> tuple[dict[str, Any], ...]:
    raw = result.payload.get(key, ())
    return tuple(item for item in raw if isinstance(item, dict))


def _action_candidate(item: dict[str, Any]) -> ActionCandidateContextItem:
    return ActionCandidateContextItem(
        action_item_id=item["action_item_id"],
        action_type=item["action_type"],
        source_type=item["source_type"],
        source_id=item["source_id"],
        title=item["title"],
        status=item["status"],
        due_at=item.get("due_at"),
    )


def _sync_application_emails(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    # Email sync finishes in one turn and has no run to resume, so it records its
    # own phase instead of taking the workflow slot away from a job-discovery run
    # the user has not finished yet.
    return task.model_copy(update={"email_sync_phase": result.state})


def _find_saved_jobs(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    return task.model_copy(
        update={
            "saved_job_candidates": tuple(
                SavedJobCandidateContextItem(
                    job_posting_id=item["job_posting_id"],
                    title=item["title"],
                    company_name=item["company_name"],
                    city=item.get("city"),
                    salary=item.get("salary"),
                )
                for item in _items(result)
            )
        }
    )


def _get_saved_job(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    job = result.payload.get("job", {})
    snapshot = result.payload.get("jd_snapshot", {})
    focus = (
        ActiveSavedJobContextItem(
            job_posting_id=job["job_posting_id"],
            jd_snapshot_id=snapshot["id"],
            title=str(job.get("title") or "")[:200] or "未命名岗位",
            company_name=str(job.get("company_name") or "")[:200] or "未知公司",
            jd_version=snapshot["version"],
        )
        if isinstance(job, dict)
        and isinstance(snapshot, dict)
        and job.get("job_posting_id")
        and snapshot.get("id")
        and snapshot.get("version")
        else None
    )
    if focus is None:
        return task.model_copy(
            update={
                "active_job_posting_id": job.get("job_posting_id")
                if isinstance(job, dict)
                else None
            }
        )
    return task.focus_saved_job(focus)


def _job_research_ready(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    return task.model_copy(
        update={
            "active_job_posting_id": result.payload.get("job_posting_id"),
            "active_job_research_run_id": result.payload.get("run_id"),
            "active_job_research_report_id": result.payload.get("report_id"),
            "job_research_status": result.payload.get("status") or "current",
        }
    )


def _job_research_failed(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    return task.model_copy(
        update={
            "active_job_posting_id": result.payload.get("job_posting_id")
            or task.active_job_posting_id,
            "active_job_research_run_id": result.payload.get("run_id"),
            "job_research_status": "failed",
        }
    )


def _job_research_result(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    if result.state == "job_research_failed":
        return _job_research_failed(task, result)
    return _job_research_ready(task, result)


def _list_target_roles(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    return task.model_copy(
        update={
            "target_role_candidates": tuple(
                TargetRoleCandidateContextItem(
                    target_role_id=item["target_role_id"],
                    title=item["title"],
                    priority=item["priority"],
                    status=item["status"],
                    city=item.get("city"),
                    salary_expectation=item.get("salary_expectation"),
                    experience=item.get("experience"),
                    education=item.get("education"),
                )
                for item in _items(result)
            )
        }
    )


def _list_resumes(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    return task.model_copy(
        update={
            "resume_candidates": tuple(
                ResumeCandidateContextItem(
                    resume_id=item["resume_id"],
                    target_role_id=item["target_role_id"],
                    name=item["name"],
                    status=item["status"],
                    latest_version_id=item.get("latest_version_id"),
                )
                for item in _items(result)
            )
        }
    )


def _get_resume_metadata(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    # Listing versions offers a choice; it does not make any of them active.
    return task.model_copy(
        update={
            "resume_version_candidates": tuple(
                ResumeVersionCandidateContextItem(
                    resume_version_id=item["resume_version_id"],
                    version_number=item["version_number"],
                    source_type=item["source_type"],
                    document_format=item["document_format"],
                    byte_size=item["byte_size"],
                )
                for item in _items(result, "versions")
            )
        }
    )


def _list_email_events(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    return task.model_copy(
        update={
            "email_event_candidates": tuple(
                EmailEventCandidateContextItem(
                    email_event_id=item["email_event_id"],
                    event_type=item["event_type"],
                    status=item["status"],
                    summary=item["summary"],
                )
                for item in _items(result)
            )
        }
    )


def _resolve_email_event(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    # The model picks events by index, so a resolved one must leave the list.
    # Keeping it would advertise a stale status and let the same event be
    # resolved twice under a shifted index.
    event_id = result.payload.get("email_event_id")
    return task.model_copy(
        update={
            "email_event_candidates": tuple(
                candidate
                for candidate in task.email_event_candidates
                if candidate.email_event_id != event_id
            )
        }
    )


def _list_calendar_accounts(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    return task.model_copy(
        update={
            "calendar_account_candidates": tuple(
                CalendarAccountCandidateContextItem(
                    calendar_account_id=item["calendar_account_id"],
                    provider=item["provider"],
                    email_address=item["email_address"],
                    calendar_id=item["calendar_id"],
                )
                for item in _items(result)
            )
        }
    )


_PENDING_CALENDAR_PROPOSAL_STATES = frozenset(
    {"calendar_approval_required", "calendar_proposal_ready"}
)
"""The only states that leave a preview waiting for the user."""


def _calendar_proposal(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    """Track the one Calendar preview that is still waiting for approval.

    Set on the states that produce or re-read a pending preview, cleared on
    every state that ends one. The distinction only started to matter once the
    existence flag reached the model: before that a stale id was invisible, and
    now it would tell the model a preview is pending when the store will refuse
    to execute it.

    ``calendar_write_failed`` clears too. The service marks the proposal
    ``failed`` before raising, and ``execute_proposal`` requires ``pending``, so
    a retry cannot succeed — leaving the flag up would send the model back at a
    door that is already locked instead of preparing a new preview.
    """
    interview_round_id = (
        result.payload.get("interview_round_id") or task.active_interview_round_id
    )
    if result.state not in _PENDING_CALENDAR_PROPOSAL_STATES:
        return task.model_copy(
            update={
                "active_calendar_proposal_id": None,
                "active_calendar_proposal_expires_at": None,
                "active_interview_round_id": interview_round_id,
            }
        )
    expires_at = result.payload.get("expires_at")
    return task.model_copy(
        update={
            "active_calendar_proposal_id": result.payload.get("proposal_id"),
            # Carried alongside the id because the model is shown the expiry and
            # not the id: a lapsed preview has to be prepared again rather than
            # executed, and without this the flag would read as "pending" long
            # after the preview stopped being executable.
            "active_calendar_proposal_expires_at": (
                datetime.fromisoformat(expires_at)
                if isinstance(expires_at, str)
                else None
            ),
            "active_interview_round_id": interview_round_id,
        }
    )


def _get_daily_brief(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    items = tuple(
        item
        for section in ("overdue", "due_today", "upcoming", "no_due_date")
        for item in _items(result, section)
    )
    return task.model_copy(
        update={"action_candidates": tuple(_action_candidate(item) for item in items)}
    )


def _list_action_items(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    return task.model_copy(
        update={
            "action_candidates": tuple(
                _action_candidate(item) for item in _items(result)
            )
        }
    )


def _resolve_action_item(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    action_item_id = result.payload.get("action_item_id")
    return task.model_copy(
        update={
            "active_action_item_id": action_item_id,
            "action_candidates": tuple(
                candidate
                for candidate in task.action_candidates
                if candidate.action_item_id != action_item_id
            ),
        }
    )


def _list_interviews(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    return task.model_copy(
        update={
            "interview_candidates": tuple(
                InterviewCandidateContextItem(
                    interview_round_id=item["interview_round_id"],
                    application_id=item["application_id"],
                    sequence_number=item["sequence_number"],
                    employer_label=item.get("employer_label"),
                    status=item["status"],
                    scheduled_start=item.get("scheduled_start"),
                )
                for item in _items(result)
            )
        }
    )


def _interview_ready(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    return task.model_copy(
        update={
            "active_interview_round_id": result.payload.get("interview_round_id"),
            "active_application_id": result.payload.get("application_id"),
        }
    )


def _interview_preparation_ready(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    return task.model_copy(
        update={
            "active_interview_preparation_id": result.payload.get("preparation_id"),
            "active_interview_round_id": result.payload.get("interview_round_id"),
            "active_application_id": result.payload.get("application_id"),
            "active_job_posting_id": result.payload.get("job_posting_id"),
            "active_resume_version_id": result.payload.get("resume_version_id"),
        }
    )


def _analyze_resume(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    return task.model_copy(
        update={
            "active_resume_analysis_id": result.payload.get("analysis_id"),
            "resume_analysis_status": "pending",
            "active_resume_version_id": result.payload.get("resume_version_id"),
        }
    )


def _get_resume_analysis(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    status = result.payload.get("status")
    return task.model_copy(
        update={
            "active_resume_analysis_id": result.payload.get("analysis_id"),
            "resume_analysis_status": (
                status if status in {"pending", "confirmed", "rejected"} else None
            ),
        }
    )


def _confirm_resume_analysis(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    return task.model_copy(update={"resume_analysis_status": "confirmed"})


def _reject_resume_analysis(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    return task.model_copy(update={"resume_analysis_status": "rejected"})


def _resume_job_match_ready(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    return task.model_copy(
        update={
            "active_resume_job_match_id": result.payload.get("match_id"),
            "resume_job_match_status": "ready",
            "active_job_posting_id": result.payload.get("job_posting_id"),
            "active_resume_version_id": result.payload.get("resume_version_id"),
        }
    )


def _job_analysis_ready(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    snapshot_id = result.payload.get("jd_snapshot_id")
    job_posting_id = result.payload.get("job_posting_id")
    focus = task.active_saved_job
    if (
        focus is not None
        and (
            focus.job_posting_id != job_posting_id
            or focus.jd_snapshot_id != snapshot_id
        )
    ):
        focus = None
    return task.model_copy(
        update={
            "active_job_analysis_id": result.payload.get("analysis_id"),
            "active_job_analysis_jd_snapshot_id": snapshot_id,
            "job_analysis_status": "ready",
            "active_job_posting_id": job_posting_id,
            "active_jd_snapshot_id": snapshot_id,
            "active_saved_job": focus,
        }
    )


def _tailoring_draft_ready(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    return task.model_copy(
        update={
            "active_resume_tailoring_draft_id": result.payload.get("draft_id"),
            "resume_tailoring_status": result.payload.get("status"),
        }
    )


def _finalize_resume_tailoring(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    return task.model_copy(
        update={
            "resume_tailoring_status": "finalized",
            "active_resume_version_id": result.payload.get("resume_version_id"),
        }
    )


def _export_resume_artifact(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    return task.model_copy(
        update={
            "active_resume_version_id": result.payload.get("resume_version_id"),
            "active_resume_artifact_id": result.payload.get("artifact_id"),
        }
    )


def _create_application(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    return task.model_copy(
        update={
            "active_application_id": result.payload.get("application_id"),
            "active_application_status": result.payload.get("status"),
            "active_job_posting_id": result.payload.get("job_posting_id"),
            "active_resume_version_id": result.payload.get("resume_version_id"),
        }
    )


def _application_ready(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    # Reading or restatusing an application says nothing about which job posting
    # or resume version the user is working on, so those stay put.
    return task.model_copy(
        update={
            "active_application_id": result.payload.get("application_id"),
            "active_application_status": result.payload.get("status"),
        }
    )


def _list_applications(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    return task.model_copy(
        update={
            "application_candidates": tuple(
                ApplicationCandidateContextItem(
                    application_id=item["application_id"],
                    title=item["title"],
                    company_name=item["company_name"],
                    status=item["status"],
                )
                for item in _items(result)
            )
        }
    )


def _entry(states: tuple[str, ...], reduce: TaskReducer) -> ReducerEntry:
    return ReducerEntry(frozenset(states), reduce)


def _fanout(
    names: tuple[str, ...], states: tuple[str, ...], reduce: TaskReducer
) -> dict[str, ReducerEntry]:
    """Register one reducer under several tool names sharing an outcome."""
    entry = _entry(states, reduce)
    return {name: entry for name in names}


def _route_to_capability(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    profile = result.payload.get("tool_profile")
    if profile not in TOOL_PROFILE_NAMES:
        return task
    return task.model_copy(update={"tool_profile": profile})


def _propose_job_intent(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    raw = result.payload.get("update")
    if not isinstance(raw, dict):
        return task
    return task.model_copy(
        update={
            "pending_job_intent_update": JobIntentUpdate.model_validate(raw),
            "bare_confirmation_target": "job_intent",
        }
    )


def _confirm_job_intent(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    # Clearing on success is what stops one confirmation from being reusable by
    # a later turn that the user never saw a readback for.
    return task.model_copy(
        update={
            "pending_job_intent_update": None,
            "bare_confirmation_target": None,
        }
    )


def _propose_free_text_preference_confirmation(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    raw = result.payload.get("proposal")
    if not isinstance(raw, dict):
        return task
    return task.model_copy(
        update={
            "pending_free_text_preference": (
                FreeTextPreferenceConfirmationProposal.model_validate(raw)
            ),
            "bare_confirmation_target": "free_text_preference",
        }
    )


def _confirm_free_text_preference(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    structured = result.payload.get("structured_proposal")
    return task.model_copy(
        update={
            "pending_free_text_preference": None,
            "pending_job_intent_update": (
                JobIntentUpdate.model_validate(structured)
                if isinstance(structured, dict)
                else task.pending_job_intent_update
            ),
            "bare_confirmation_target": (
                "job_intent" if isinstance(structured, dict) else None
            ),
        }
    )


def _propose_memory_tombstone(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    raw = result.payload.get("proposal")
    if not isinstance(raw, dict):
        return task
    return task.model_copy(
        update={
            "pending_memory_tombstone": MemoryTombstoneProposal.model_validate(
                raw
            )
        }
    )


def _propose_memory_amendment(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    raw = result.payload.get("proposal")
    if not isinstance(raw, dict):
        return task
    return task.model_copy(
        update={
            "pending_memory_amendment": MemoryAmendmentProposal.model_validate(
                raw
            )
        }
    )


def _confirm_memory_amendment(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    return task.model_copy(
        update={"pending_memory_amendment": None}
    )


def _confirm_memory_tombstone(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    return task.model_copy(
        update={"pending_memory_tombstone": None}
    )


def _propose_career_fact(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    raw = result.payload.get("proposal")
    if not isinstance(raw, dict):
        return task
    return task.model_copy(
        update={
            "pending_career_fact": CareerFactProposal.model_validate(raw),
            "bare_confirmation_target": "career_fact",
        }
    )


def _confirm_career_fact(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    return task.model_copy(
        update={
            "pending_career_fact": None,
            "bare_confirmation_target": None,
        }
    )


def _propose_constraint_retirement(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    raw = result.payload.get("proposal")
    if not isinstance(raw, dict):
        return task
    return task.model_copy(
        update={
            "pending_constraint_retirement": (
                ConstraintRetirementProposal.model_validate(raw)
            )
        }
    )


def _confirm_constraint_retirement(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    return task.model_copy(update={"pending_constraint_retirement": None})


ATOMIC_TASK_REDUCERS: dict[str, ReducerEntry] = {
    "propose_job_intent": _entry(
        ("job_intent_proposed",), _propose_job_intent
    ),
    "confirm_job_intent": _entry(
        ("job_intent_recorded",), _confirm_job_intent
    ),
    "propose_free_text_preference_confirmation": _entry(
        ("free_text_preference_confirmation_proposed",),
        _propose_free_text_preference_confirmation,
    ),
    "confirm_free_text_preference": _entry(
        (
            "free_text_preference_confirmed",
            "free_text_preference_confirmed_structured_proposed",
        ),
        _confirm_free_text_preference,
    ),
    "propose_memory_tombstone": _entry(
        ("memory_tombstone_proposed",), _propose_memory_tombstone
    ),
    "propose_memory_amendment": _entry(
        ("memory_amendment_proposed",), _propose_memory_amendment
    ),
    "confirm_memory_amendment": _entry(
        ("career_memory_amended",), _confirm_memory_amendment
    ),
    "confirm_memory_tombstone": _entry(
        ("memory_tombstoned",),
        _confirm_memory_tombstone,
    ),
    "propose_career_fact": _entry(
        ("career_fact_proposed",), _propose_career_fact
    ),
    "confirm_career_fact": _entry(
        ("career_fact_confirmed",), _confirm_career_fact
    ),
    "propose_constraint_retirement": _entry(
        ("constraint_retirement_proposed",), _propose_constraint_retirement
    ),
    "confirm_constraint_retirement": _entry(
        ("constraint_retired",), _confirm_constraint_retirement
    ),
    "route_to_capability": _entry(("tool_profile_switched",), _route_to_capability),
    "sync_application_emails": _entry((), _sync_application_emails),
    "find_saved_jobs": _entry(
        ("saved_jobs_found", "no_saved_jobs_found"), _find_saved_jobs
    ),
    "get_saved_job": _entry(("saved_job_ready",), _get_saved_job),
    "analyze_job": _entry(("job_analysis_ready",), _job_analysis_ready),
    **_fanout(
        ("research_job", "retry_job_research", "get_job_research"),
        ("job_research_ready", "job_research_failed"),
        _job_research_result,
    ),
    "list_target_roles": _entry(
        ("target_roles_found", "no_target_roles_found"), _list_target_roles
    ),
    "list_resumes": _entry(("resumes_found", "no_resumes_found"), _list_resumes),
    "get_resume_metadata": _entry(("resume_metadata_ready",), _get_resume_metadata),
    "list_email_events": _entry(
        ("email_events_found", "no_email_events_found"), _list_email_events
    ),
    "resolve_email_event": _entry(("email_event_resolved",), _resolve_email_event),
    "list_calendar_accounts": _entry(
        ("calendar_accounts_found", "no_calendar_accounts"), _list_calendar_accounts
    ),
    "get_daily_brief": _entry(("daily_brief_ready",), _get_daily_brief),
    "list_action_items": _entry(
        ("action_items_found", "no_action_items_found"), _list_action_items
    ),
    "list_interviews": _entry(
        ("interviews_found", "no_interviews_found"), _list_interviews
    ),
    "analyze_resume": _entry(("resume_analysis_ready",), _analyze_resume),
    "get_resume_analysis": _entry(("resume_analysis_ready",), _get_resume_analysis),
    "confirm_resume_analysis": _entry(
        ("resume_analysis_confirmed",), _confirm_resume_analysis
    ),
    "reject_resume_analysis": _entry(
        ("resume_analysis_rejected",), _reject_resume_analysis
    ),
    "finalize_resume_tailoring": _entry(
        ("resume_tailoring_finalized",), _finalize_resume_tailoring
    ),
    "export_resume_artifact": _entry(
        ("resume_artifact_ready",), _export_resume_artifact
    ),
    "create_application": _entry(("application_ready",), _create_application),
    "list_applications": _entry(
        ("applications_found", "no_applications_found"), _list_applications
    ),
    **_fanout(
        (
            "prepare_interview_calendar_sync",
            "get_calendar_proposal",
            "execute_calendar_proposal",
        ),
        (
            "calendar_approval_required",
            "calendar_proposal_ready",
            "calendar_sync_complete",
            # Registered so the slot is released: these were unhandled, which
            # was harmless while nothing was projected and is not now.
            "calendar_proposal_not_found",
            "calendar_approval_invalid",
            "calendar_write_failed",
        ),
        _calendar_proposal,
    ),
    **_fanout(
        ("complete_action_item", "dismiss_action_item", "snooze_action_item"),
        ("action_item_resolved", "action_item_snoozed"),
        _resolve_action_item,
    ),
    **_fanout(
        (
            "get_interview",
            "create_interview",
            "update_interview",
            "complete_interview",
            "record_interview_retro",
        ),
        ("interview_ready", "interview_retro_recorded"),
        _interview_ready,
    ),
    **_fanout(
        ("prepare_interview", "get_interview_preparation"),
        ("interview_preparation_ready",),
        _interview_preparation_ready,
    ),
    **_fanout(
        ("match_resume_to_job", "get_resume_job_match"),
        ("resume_job_match_ready",),
        _resume_job_match_ready,
    ),
    **_fanout(
        (
            "draft_resume_tailoring",
            "get_resume_tailoring_draft",
            "review_resume_tailoring",
            "revise_resume_tailoring",
        ),
        ("resume_tailoring_draft_ready",),
        _tailoring_draft_ready,
    ),
    **_fanout(
        ("update_application_status", "get_application"),
        ("application_ready",),
        _application_ready,
    ),
}


def reduce_task_state(
    task: ConversationTaskState,
    result: ToolResult,
    *,
    now: datetime | None = None,
) -> ConversationTaskState:
    """Apply the registered reducer for ``result``, if any.

    An unregistered tool is not an error: read-only tools whose output the user
    consumes directly have no task-state effect to record.
    """
    entry = ATOMIC_TASK_REDUCERS.get(result.tool_name)
    if entry is None or not entry.applies_to(result.state):
        return task
    # Stamped here rather than in each propose reducer, so a reducer added later
    # cannot put a proposal in a slot without starting its expiry clock.
    return entry.reduce(task, result).stamp_new_proposals(
        task, now or datetime.now(timezone.utc)
    )
