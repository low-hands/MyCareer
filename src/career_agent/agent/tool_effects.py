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


def effect_for(name: str) -> ToolEffect:
    try:
        return TOOL_EFFECTS[name]
    except KeyError as error:
        raise ValueError(f"Main Agent capability has no declared effect: {name}") from error
