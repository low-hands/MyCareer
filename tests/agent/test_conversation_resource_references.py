"""Reading back a report an earlier turn produced.

A resource-backed message stores one line and a reference, not the report, so
the reference is the only path from the conversation to what the report said.
The model must be able to follow it without ever seeing an internal id, which
is what a turn-local index buys and what these tests pin down.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from career_agent.agent.main_agent_contracts import (
    ApplicationCandidateContextItem,
    CareerProfileContext,
    ConversationMessageContext,
    ConversationResourceReference,
    ConversationTaskState,
    GetInterviewPreparationToolArguments,
    GetJobResearchToolArguments,
    GetMockInterviewResultToolArguments,
    MainAgentContext,
    SavedJobCandidateContextItem,
    project_interview_preparation_arguments,
    project_job_research_arguments,
    project_mock_interview_result_arguments,
)
from career_agent.agent.main_agent_tools import MainAgentToolRegistry

NOW = datetime(2026, 8, 31, tzinfo=timezone.utc)


def _message(content: str, *, kind: str | None = None, resource_id: str = "") -> ConversationMessageContext:
    return ConversationMessageContext(
        role="assistant",
        content=content,
        created_at=NOW,
        resource_ref=(
            ConversationResourceReference(
                kind=kind,
                resource_id=resource_id,
                status_at_delivery=("current" if kind == "job_research_report" else None),
                anchored_by_other_job=(False if kind == "job_research_report" else None),
            )
            if kind is not None
            else None
        ),
    )


def _context(*messages: ConversationMessageContext, task: ConversationTaskState | None = None) -> MainAgentContext:
    return MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        task=task or ConversationTaskState(),
        recent_messages=messages,
        user_message="那份报告里怎么说的",
    )


def _mixed_context() -> MainAgentContext:
    """Three reports of different kinds, with plain turns interleaved."""
    return _context(
        _message("岗位研究已完成。", kind="job_research_report", resource_id="report-1"),
        _message("好的。"),
        _message(
            "面试准备材料已生成。",
            kind="interview_preparation",
            resource_id="prep-1",
        ),
        _message("模拟面试完成。", kind="mock_interview_report", resource_id="rep-1"),
    )


def test_the_index_the_model_reads_is_the_index_it_can_pass_back() -> None:
    """Projection and resolution number the same list the same way.

    Numbering them in two independent loops would be a silent defect rather than
    an error: the model would ask for index 2 and be handed index 3's report.
    """
    context = _mixed_context()

    projected = context.model_context()["recent_messages"]
    assert [item.get("resource") for item in projected] == [
        {"kind": "job_research_report", "reference_index": 1},
        None,
        {"kind": "interview_preparation", "reference_index": 2},
        {"kind": "mock_interview_report", "reference_index": 3},
    ]
    # Plain turns are skipped in the numbering, not counted and hidden.
    assert [reference.resource_id for reference in context.referenced_resources()] == [
        "report-1",
        "prep-1",
        "rep-1",
    ]


def test_internal_ids_never_reach_the_model() -> None:
    context = _mixed_context()

    serialized = repr(context.model_context())
    for resource_id in ("report-1", "prep-1", "rep-1"):
        assert resource_id not in serialized


def test_job_research_reference_keeps_delivery_time_render_context() -> None:
    reference = ConversationResourceReference(
        kind="job_research_report",
        resource_id="report-1",
        status_at_delivery="outdated",
        anchored_by_other_job=True,
    )

    assert reference.model_dump(mode="json") == {
        "kind": "job_research_report",
        "resource_id": "report-1",
        "status_at_delivery": "outdated",
        "anchored_by_other_job": True,
    }
    # The model learns only that a selectable report exists; presentation
    # metadata and the internal id remain outside its prompt.
    projected = _context(
        ConversationMessageContext(
            role="assistant",
            content="岗位研究已完成。",
            created_at=NOW,
            resource_ref=reference,
        )
    ).model_context()["recent_messages"][0]
    assert projected["resource"] == {
        "kind": "job_research_report",
        "reference_index": 1,
    }


def test_resolution_checks_the_kind_rather_than_trusting_the_index() -> None:
    """A mistyped index is an error, not a lookup for the wrong entity.

    The model picks out of one mixed list, so without this an off-by-one hands a
    preparation id to a report lookup, which reads back as "not found" and
    invites a retry that cannot succeed.
    """
    context = _mixed_context()

    assert (
        context.resolve_reference_index(
            reference_index=1, kind="job_research_report"
        )
        == "report-1"
    )
    with pytest.raises(ValueError, match="is a interview_preparation, not a"):
        context.resolve_reference_index(
            reference_index=2, kind="job_research_report"
        )


@pytest.mark.parametrize("reference_index", [0, 4])
def test_an_index_outside_the_window_is_rejected(reference_index) -> None:
    """Out of range rather than clamped: the window slides as turns age out.

    A report the recent window no longer covers is genuinely unreachable this
    way, and silently resolving to the nearest reference would answer about a
    different report than the user asked about.
    """
    with pytest.raises(ValueError, match="out of range"):
        _mixed_context().resolve_reference_index(
            reference_index=reference_index, kind="job_research_report"
        )


def test_no_references_leaves_every_index_out_of_range() -> None:
    context = _context(_message("你好。"))

    assert context.referenced_resources() == ()
    assert context.model_context()["recent_messages"][0].get("resource") is None
    with pytest.raises(ValueError, match="out of range"):
        context.resolve_reference_index(
            reference_index=1, kind="job_research_report"
        )


def test_job_research_reads_back_the_referenced_report_not_the_active_one() -> None:
    """A reference outranks the active pointer, which is the whole point.

    ``active_job_research_report_id`` follows the current focus, so by the time
    the user asks about an earlier report it names a different one. Without the
    reference winning here, "那份报告" would silently return the newest.
    """
    context = _context(
        _message("岗位研究已完成。", kind="job_research_report", resource_id="report-1"),
        task=ConversationTaskState(
            active_job_research_report_id="report-9",
            active_job_posting_id="job-9",
        ),
    )

    projected = project_job_research_arguments(
        context, "get_job_research", {"reference_index": 1}
    )
    assert projected == {"user_id": "u1", "report_id": "report-1"}
    # With no reference the active pointer still applies.
    assert project_job_research_arguments(context, "get_job_research", {}) == {
        "user_id": "u1",
        "report_id": "report-9",
    }


def test_job_research_rejects_two_selectors_at_once() -> None:
    """One selector names one report; two disagree with no way to choose."""
    context = _context(
        _message("岗位研究已完成。", kind="job_research_report", resource_id="report-1"),
        task=ConversationTaskState(
            saved_job_candidates=(
                SavedJobCandidateContextItem(
                    job_posting_id="job-1", title="RAG Engineer", company_name="Acme"
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match="either reference_index or selection_index"):
        project_job_research_arguments(
            context,
            "get_job_research",
            {"reference_index": 1, "selection_index": 1},
        )


def test_interview_preparation_reads_back_an_earlier_result() -> None:
    context = _context(
        _message(
            "面试准备材料已生成。",
            kind="interview_preparation",
            resource_id="prep-1",
        ),
        task=ConversationTaskState(active_interview_preparation_id="prep-9"),
    )

    projected = project_interview_preparation_arguments(
        context, "get_interview_preparation", {"reference_index": 1}
    )
    assert projected == {"user_id": "u1", "preparation_id": "prep-1"}
    assert project_interview_preparation_arguments(
        context, "get_interview_preparation", {}
    ) == {"user_id": "u1", "preparation_id": "prep-9"}


def test_a_referenced_mock_interview_is_read_by_report_not_by_application() -> None:
    """The reference names one run, so the application must not be sent too.

    One application can be practised against repeatedly. The handler falls back
    to the newest finished run for an application, so passing both would answer
    about a later interview than the turn the user pointed at.
    """
    context = _context(
        _message("模拟面试完成。", kind="mock_interview_report", resource_id="rep-1"),
        task=ConversationTaskState(
            active_application_id="app-1",
            application_candidates=(
                ApplicationCandidateContextItem(
                    application_id="app-2",
                    title="RAG Engineer",
                    company_name="Acme",
                    status="submitted",
                ),
            ),
        ),
    )

    projected = project_mock_interview_result_arguments(
        context, {"reference_index": 1, "question_number": 2}
    )
    assert projected == {
        "user_id": "u1",
        "report_id": "rep-1",
        "question_number": 2,
    }
    assert "application_id" not in projected

    with pytest.raises(
        ValueError, match="either reference_index or application_selection_index"
    ):
        project_mock_interview_result_arguments(
            context, {"reference_index": 1, "application_selection_index": 1}
        )


@pytest.mark.parametrize(
    ("contract", "internal_ids"),
    [
        (GetJobResearchToolArguments, ("report_id", "job_posting_id")),
        (GetInterviewPreparationToolArguments, ("preparation_id",)),
        (GetMockInterviewResultToolArguments, ()),
    ],
)
def test_every_read_back_contract_offers_the_reference_selector(
    contract, internal_ids
) -> None:
    """All three, not two: the mechanism is only usable where it is wired.

    A reference the model can see but not act on for one kind is worse than
    none, because the report is visibly there and unreachable. The internal ids
    stay declared for the projected handler call and are checked here to be
    absent from what the model is offered.
    """
    schema = MainAgentToolRegistry._decision_tool_schema(
        {"function": {"parameters": contract.model_json_schema()}}
    )
    properties = schema["function"]["parameters"]["properties"]

    assert "reference_index" in properties
    for internal_id in internal_ids:
        assert internal_id in contract.model_json_schema()["properties"]
        assert internal_id not in properties
