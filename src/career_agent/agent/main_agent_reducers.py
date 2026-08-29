"""Task-state reducers for atomic Main Agent tools.

Each reducer owns one tool's effect on ``ConversationTaskState`` and is keyed by
tool name in ``ATOMIC_TASK_REDUCERS``, so adding a capability means registering
an entry rather than extending a branch chain inside the runtime. A reducer only
runs when the result state is one the entry accepts, which keeps the
"did this tool actually succeed" check next to the fields it guards.
"""

from __future__ import annotations

from typing import Any, Callable

from career_agent.agent.main_agent_contracts import (
    ActionCandidateContextItem,
    ApplicationCandidateContextItem,
    CalendarAccountCandidateContextItem,
    ConversationTaskState,
    EmailEventCandidateContextItem,
    InterviewCandidateContextItem,
    ResumeCandidateContextItem,
    ResumeVersionCandidateContextItem,
    SavedJobCandidateContextItem,
    TargetRoleCandidateContextItem,
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
    return task.model_copy(
        update={"active_job_posting_id": job.get("job_posting_id")}
    )


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


def _calendar_proposal(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    return task.model_copy(
        update={
            "active_calendar_proposal_id": result.payload.get("proposal_id"),
            "active_interview_round_id": result.payload.get("interview_round_id")
            or task.active_interview_round_id,
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
                status if status in {"pending", "confirmed"} else None
            ),
        }
    )


def _confirm_resume_analysis(
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    return task.model_copy(update={"resume_analysis_status": "confirmed"})


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


ATOMIC_TASK_REDUCERS: dict[str, ReducerEntry] = {
    "sync_application_emails": _entry((), _sync_application_emails),
    "find_saved_jobs": _entry(
        ("saved_jobs_found", "no_saved_jobs_found"), _find_saved_jobs
    ),
    "get_saved_job": _entry(("saved_job_ready",), _get_saved_job),
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
    task: ConversationTaskState, result: ToolResult
) -> ConversationTaskState:
    """Apply the registered reducer for ``result``, if any.

    An unregistered tool is not an error: read-only tools whose output the user
    consumes directly have no task-state effect to record.
    """
    entry = ATOMIC_TASK_REDUCERS.get(result.tool_name)
    if entry is None or not entry.applies_to(result.state):
        return task
    return entry.reduce(task, result)
