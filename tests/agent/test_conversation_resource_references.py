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
    DecisionObservation,
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
        resource_refs=(
            (
                ConversationResourceReference(
                    kind=kind,
                    resource_id=resource_id,
                    status_at_delivery=(
                        "current" if kind == "job_research_report" else None
                    ),
                    anchored_by_other_job=(
                        False if kind == "job_research_report" else None
                    ),
                ),
            )
            if kind is not None
            else ()
        ),
    )


def _handle(reference: ConversationResourceReference, *, conversation_id: str) -> str:
    """The handle the projection derives, derived the same way the code does.

    Asserted as a derivation rather than as a pasted literal: what these tests
    mean is "the name for this resource", and a literal would keep passing while
    silently meaning something else.
    """
    return MainAgentContext(
        conversation_id=conversation_id,
        profile=CareerProfileContext(user_id="u1"),
        user_message="x",
    ).reference_handle(reference)


def _only_handle(context: MainAgentContext) -> str:
    """The one handle a single-resource context hands out."""
    handles = tuple(context.reference_handles())
    assert len(handles) == 1
    return handles[0]


def _context(*messages: ConversationMessageContext, task: ConversationTaskState | None = None, conversation_id: str = "c1") -> MainAgentContext:
    return MainAgentContext(
        conversation_id=conversation_id,
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


def test_the_handle_the_model_reads_is_the_handle_it_can_pass_back() -> None:
    """Projection and resolution derive the same name for the same resource.

    Deriving them separately would be a silent defect rather than an error: the
    model would send back a name the resolver does not know, or worse, one it
    knows as a different resource.
    """
    context = _mixed_context()

    projected = context.model_context()["recent_messages"]
    shown = [item.get("resources") for item in projected]
    assert [
        None if group is None else [item["kind"] for item in group]
        for group in shown
    ] == [
        ["job_research_report"],
        None,
        ["interview_preparation"],
        ["mock_interview_report"],
    ]
    # Every name shown resolves, as its own kind, to the resource it was shown
    # for — the property, rather than the strings the derivation happens to make.
    for group, resource_id in zip(
        (shown[0], shown[2], shown[3]), ("report-1", "prep-1", "rep-1")
    ):
        item = group[0]
        assert (
            context.resolve_reference(
                reference=item["reference"], kind=item["kind"]
            )
            == resource_id
        )
    # A handle carries its kind in the prefix, so a mistake reads as a mismatch
    # rather than as "not found".
    assert shown[0][0]["reference"].startswith("report_")
    assert shown[2][0]["reference"].startswith("preparation_")
    assert shown[3][0]["reference"].startswith("mock_")
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
        # Untitled here on purpose: legacy rows are still readable, while the
        # projection omits absent metadata rather than showing a blank field.
        "title": None,
        "description": None,
    }
    # The model learns only that a selectable report exists; presentation
    # metadata and the internal id remain outside its prompt.
    projected = _context(
        ConversationMessageContext(
            role="assistant",
            content="岗位研究已完成。",
            created_at=NOW,
            resource_refs=(reference,),
        )
    ).model_context()["recent_messages"][0]
    assert projected["resources"] == [
        {
            "kind": "job_research_report",
            "reference": _handle(reference, conversation_id="c1"),
        }
    ]


def test_resolution_checks_the_kind_rather_than_trusting_the_prefix() -> None:
    """The prefix is a hint to the model; the server verifies against its own map.

    ``preparation_...`` tells the model what it is holding without pairing the
    handle with a separate field. But it is text the model produced, so a handle
    used against the wrong reader has to be refused by what was actually handed
    out, not by reading the string.
    """
    context = _mixed_context()
    handles = context.reference_handles()
    report = next(h for h, rid in handles.items() if rid == "report-1")
    preparation = next(h for h, rid in handles.items() if rid == "prep-1")

    assert (
        context.resolve_reference(reference=report, kind="job_research_report")
        == "report-1"
    )
    with pytest.raises(ValueError, match="is a interview_preparation, not a"):
        context.resolve_reference(
            reference=preparation, kind="job_research_report"
        )


@pytest.mark.parametrize(
    "fabricated", ["report_1", "report_000000", "report_deadbeef", "1"]
)
def test_a_handle_that_was_never_handed_out_is_rejected(fabricated) -> None:
    """The reason for the whole scheme, stated as a test.

    Under ordinals the model wrote ``reference_index=1`` on a turn where no
    index had been offered — recorded behaviour, not a worry — and it resolved,
    because 1 was a real number of a real report of the right kind. Nothing
    could tell "the number I was given" from "the number I counted to".

    A derived handle has no counting order to land on. Every plausible guess
    here — a small integer, a zeroed suffix, a well-formed hex string — is
    simply not in the map.
    """
    with pytest.raises(ValueError, match="unknown resource reference"):
        _mixed_context().resolve_reference(
            reference=fabricated, kind="job_research_report"
        )


def test_a_handle_from_another_conversation_does_not_resolve() -> None:
    """The salt is the conversation, so names do not travel between them.

    Scope hygiene rather than a security boundary: it keeps a handle quoted from
    elsewhere from silently naming some local resource.
    """
    message = _message(
        "岗位研究已完成。", kind="job_research_report", resource_id="report-1"
    )
    here = _context(message, conversation_id="c1")
    elsewhere = _context(message, conversation_id="c2")
    foreign = next(iter(elsewhere.reference_handles()))

    assert foreign not in here.reference_handles()
    with pytest.raises(ValueError, match="unknown resource reference"):
        here.resolve_reference(reference=foreign, kind="job_research_report")


def test_no_references_leaves_every_handle_unknown() -> None:
    context = _context(_message("你好。"))

    assert context.referenced_resources() == ()
    assert context.reference_handles() == {}
    assert not context.model_context()["recent_messages"][0].get("resources")
    with pytest.raises(ValueError, match="unknown resource reference"):
        context.resolve_reference(
            reference="report_abc123", kind="job_research_report"
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
        context, "get_job_research", {"reference": _only_handle(context)}
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

    with pytest.raises(ValueError, match="either reference or selection_index"):
        project_job_research_arguments(
            context,
            "get_job_research",
            {"reference": _only_handle(context), "selection_index": 1},
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
        context, "get_interview_preparation", {"reference": _only_handle(context)}
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
        context, {"reference": _only_handle(context), "question_number": 2}
    )
    assert projected == {
        "user_id": "u1",
        "report_id": "rep-1",
        "question_number": 2,
    }
    assert "application_id" not in projected

    with pytest.raises(
        ValueError, match="either reference or application_selection_index"
    ):
        project_mock_interview_result_arguments(
            context, {"reference": _only_handle(context), "application_selection_index": 1}
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

    assert "reference" in properties
    for internal_id in internal_ids:
        assert internal_id in contract.model_json_schema()["properties"]
        assert internal_id not in properties


def test_every_projection_path_carries_the_label_beside_the_handle() -> None:
    """The catalogue, the window and this turn's observations, or none of them.

    Two reports produced in one turn is the case the handle was added for, and
    it reaches the model through ``tool_observations`` — the one path that was
    left unlabelled when labels were added to the other two. Without it the two
    observations differ only in an opaque suffix, so a model asked about the
    first has nothing to match on, which is the failure labels exist to remove.
    """
    first = ConversationResourceReference(
        kind="job_research_report",
        resource_id="report-a",
        title="示例科技",
        description="企业搜索产品调研。",
        status_at_delivery="current",
        anchored_by_other_job=False,
    )
    second = ConversationResourceReference(
        kind="job_research_report",
        resource_id="report-b",
        title="另一家科技",
        description="推荐系统产品调研。",
        status_at_delivery="current",
        anchored_by_other_job=False,
    )
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        user_message="示例科技那份怎么说",
        archived_resources=(
            ConversationMessageContext(
                role="assistant",
                content="更早的一份。",
                created_at=NOW,
                resource_refs=(first,),
            ),
        ),
        recent_messages=(
            ConversationMessageContext(
                role="assistant",
                content="上一轮的一份。",
                created_at=NOW,
                resource_refs=(second,),
            ),
        ),
        tool_observations=(
            DecisionObservation(
                tool_name="research_job",
                state="job_research_ready",
                message="已完成示例科技的岗位研究。",
                resource_ref=first,
            ),
        ),
    )

    projected = context.model_context()

    archived = projected["archived_reports"]["items"][0]
    assert archived["title"] == "示例科技"
    assert archived["description"] == "企业搜索产品调研。"
    assert projected["recent_messages"][0]["resources"][0]["title"] == "另一家科技"
    assert projected["recent_messages"][0]["resources"][0]["description"] == "推荐系统产品调研。"
    assert projected["tool_observations"][0]["title"] == "示例科技"
    assert projected["tool_observations"][0]["description"] == "企业搜索产品调研。"


def test_an_unlabelled_resource_shows_no_empty_label_anywhere() -> None:
    """Absent, not blank: a producer that has nothing to say says nothing."""
    reference = ConversationResourceReference(
        kind="interview_preparation", resource_id="prep-1"
    )
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        user_message="继续",
        recent_messages=(
            ConversationMessageContext(
                role="assistant",
                content="面试准备已生成。",
                created_at=NOW,
                resource_refs=(reference,),
            ),
        ),
        tool_observations=(
            DecisionObservation(
                tool_name="prepare_interview",
                state="interview_preparation_ready",
                message="已生成。",
                resource_ref=reference,
            ),
        ),
    )

    projected = context.model_context()

    assert "title" not in projected["recent_messages"][0]["resources"][0]
    assert "description" not in projected["recent_messages"][0]["resources"][0]
    assert "title" not in projected["tool_observations"][0]
    assert "description" not in projected["tool_observations"][0]
