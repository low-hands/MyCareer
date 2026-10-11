"""Safety boundaries for the future per-call selector, before it is wired in."""

from __future__ import annotations

import pytest

from career_agent.agent.capabilities.catalog import CAPABILITIES
from career_agent.agent.capabilities.selection import (
    ALWAYS_OFFERED_TOOLS,
    prepare_capability_selection,
)
from career_agent.agent.contracts.task_state import ConversationTaskState


def _schemas():
    return tuple(
        descriptor.tool_schema()
        for descriptor in CAPABILITIES.values()
        if descriptor.model_callable
    )


def test_only_small_discovery_set_is_always_offered() -> None:
    assert ALWAYS_OFFERED_TOOLS == (
        "search_capabilities", "load_skill", "read_conversation_span",
        "fetch_archived_constraints", "search_career_memory",
    )


def test_cross_domain_selection_uses_one_offer_and_reports_blocked_step() -> None:
    task = ConversationTaskState(
        active_job_posting_id="job-1",
        active_resume_version_id="resume-1",
    )
    selection = prepare_capability_selection(
        ("match_resume_to_job", "analyze_job", "analyze_job"),
        task=task,
        registered_schemas=_schemas(),
    )

    assert selection.selected_names == ("analyze_job", "match_resume_to_job")
    assert selection.offered_names == ("analyze_job",)
    assert tuple(schema["function"]["name"] for schema in selection.schemas) == (
        "analyze_job",
    )
    assert selection.blocked_requirements[0][0] == "match_resume_to_job"

    ready = prepare_capability_selection(
        ("analyze_job", "match_resume_to_job"),
        task=ConversationTaskState(
            active_job_posting_id="job-1",
            active_resume_version_id="resume-1",
            active_jd_snapshot_id="jd-1",
            active_job_analysis_id="analysis-1",
            active_job_analysis_jd_snapshot_id="jd-1",
            job_analysis_status="ready",
        ),
        registered_schemas=_schemas(),
    )
    assert ready.offered_names == ("analyze_job", "match_resume_to_job")
    assert ready.blocked_requirements == ()


@pytest.mark.parametrize("name", ("not_a_tool", "handle_mock_interview_input"))
def test_unregistered_or_runtime_only_selection_fails_closed(name: str) -> None:
    with pytest.raises(ValueError, match="unknown or runtime-only"):
        prepare_capability_selection(
            (name,), task=ConversationTaskState(), registered_schemas=_schemas()
        )


def test_missing_registered_schema_does_not_expand_offer() -> None:
    with pytest.raises(ValueError, match="no registered schema"):
        prepare_capability_selection(
            ("list_resumes",), task=ConversationTaskState(), registered_schemas=()
        )
