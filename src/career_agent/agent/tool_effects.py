from __future__ import annotations

from types import MappingProxyType
from typing import Literal, Mapping


ToolEffect = Literal["READ", "WRITE"]


_WRITE_CAPABILITIES = frozenset(
    {
        "analyze_resume",
        "complete_action_item",
        "complete_interview",
        "confirm_job_intent",
        "create_application",
        "create_interview",
        "dismiss_action_item",
        "draft_resume_tailoring",
        "execute_calendar_proposal",
        "export_resume_artifact",
        "finalize_resume_tailoring",
        "handle_mock_interview_input",
        "match_resume_to_job",
        "open_job_search",
        "prepare_interview",
        "prepare_interview_calendar_sync",
        "record_interview_retro",
        "research_job",
        "resolve_email_event",
        "retry_job_research",
        "review_resume_tailoring",
        "revise_resume_tailoring",
        "snooze_action_item",
        "start_mock_interview",
        "sync_application_emails",
        "update_application_status",
        "update_interview",
        "update_owner_settings",
        "restart_mock_interview",
        "retry_mock_interview",
    }
)

_READ_CAPABILITIES = frozenset(
    {
        "compare_saved_jobs",
        "find_saved_jobs",
        "get_application",
        "get_calendar_proposal",
        "get_daily_brief",
        "get_interview",
        "get_interview_preparation",
        "get_job_research",
        "get_mock_interview_result",
        "get_resume_analysis",
        "get_resume_job_match",
        "get_resume_metadata",
        "get_resume_tailoring_draft",
        "get_saved_job",
        "list_action_items",
        "list_applications",
        "list_calendar_accounts",
        "list_calendar_links",
        "list_email_events",
        "list_interviews",
        "list_resumes",
        "list_target_roles",
        "propose_job_intent",
        "read_conversation_span",
    }
)

if _READ_CAPABILITIES & _WRITE_CAPABILITIES:
    raise RuntimeError("a Main Agent capability cannot be both READ and WRITE")

TOOL_EFFECTS: Mapping[str, ToolEffect] = MappingProxyType(
    {
        **dict.fromkeys(_READ_CAPABILITIES, "READ"),
        **dict.fromkeys(_WRITE_CAPABILITIES, "WRITE"),
    }
)


_REPLAY_SAFE_CAPABILITIES = frozenset(
    {
        # UNIQUE(user_id, job_posting_id): a second call for the same posting
        # returns the existing application rather than creating another.
        "create_application",
        # Carries its own idempotency key and a reconciliation path; the service
        # treats an unsettled attempt as recovery rather than as a new write.
        "execute_calendar_proposal",
        # The handler keys the applied change by the bound confirmation id;
        # a retry returns the audit event's recorded after-state.
        "update_owner_settings",
    }
)
"""Writes that may be invoked again for a slot whose outcome is unknown.

Declared, never inferred. All writes record intent, but re-running one requires
something downstream that collapses a second call into the original effect.

``research_job`` is deliberately absent although it looks eligible. Its unique
index is ``WHERE status = 'running'`` — a guard against two concurrent runs, not
a deduplication of results. What actually collapses a repeat is the freshness
cache (``find_completed(created_after=now - freshness)``), which holds for a
crash-retry seconds later and stops holding once the window passes. Safety that
depends on how long the operator took to retry is not the kind that belongs in
this set.

"""


def replay_safe(name: str) -> bool:
    """Whether re-invoking this capability cannot produce a second effect."""

    return name in _REPLAY_SAFE_CAPABILITIES


def effect_for(name: str) -> ToolEffect:
    try:
        return TOOL_EFFECTS[name]
    except KeyError as error:
        raise ValueError(f"Main Agent capability has no declared effect: {name}") from error
