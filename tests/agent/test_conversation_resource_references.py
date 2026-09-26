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
from career_agent.domain.job_research.models import company_key

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
        # Untitled and unbound here on purpose: legacy rows are still readable,
        # while the projection omits absent metadata rather than showing a
        # blank field.
        "job_posting_id": None,
        "company_key": None,
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
    # With no reference the active pointer still applies, and carries the
    # request so the read can check it is about the same company.
    assert project_job_research_arguments(context, "get_job_research", {}) == {
        "user_id": "u1",
        "report_id": "report-9",
        "implicit_request": context.user_message,
    }


def _titled_report_context(user_message: str) -> MainAgentContext:
    """One stored report titled after 历史科技甲, one saved job at 示例科技."""
    message = _message("上周的调研好了。", kind="job_research_report", resource_id="report-h1")
    titled = message.model_copy(
        update={
            "resource_refs": (
                message.resource_refs[0].model_copy(update={"title": "历史科技甲"}),
            )
        }
    )
    return _context(
        titled,
        task=ConversationTaskState(
            saved_job_candidates=(
                SavedJobCandidateContextItem(
                    job_posting_id="job-1", title="算法工程师", company_name="示例科技"
                ),
            ),
        ),
    ).model_copy(update={"user_message": user_message})


def test_a_report_handle_titled_for_another_company_is_refused() -> None:
    """The handle was issued, but not for the company the user named.

    Reading it would answer about 历史科技甲 while the user asked about 示例科技,
    and nothing downstream could tell. The refusal names the grounded route.
    """
    context = _titled_report_context("示例科技那份调研里，他们的主要竞争对手是谁？")

    with pytest.raises(ValueError, match="selection_index 1（示例科技）"):
        project_job_research_arguments(
            context, "get_job_research", {"reference": _only_handle(context)}
        )


@pytest.mark.parametrize(
    "user_message",
    (
        "那份报告里怎么说的",
        "历史科技甲那份调研里竞争对手是谁",
        "把示例科技和历史科技甲的调研放一起看",
    ),
)
def test_a_report_handle_is_kept_when_its_company_is_named_or_none_is(
    user_message: str,
) -> None:
    context = _titled_report_context(user_message)

    projected = project_job_research_arguments(
        context, "get_job_research", {"reference": _only_handle(context)}
    )
    assert projected == {"user_id": "u1", "report_id": "report-h1"}


def _bound_report_context(
    user_message: str,
    *,
    report_job: str = "job-h1",
    report_company: str = "历史科技甲",
    candidates: tuple[tuple[str, str], ...] = (("job-1", "字节跳动"),),
    searched_this_turn: bool = False,
) -> MainAgentContext:
    """One stored report bound to its company entity, plus saved jobs.

    The reference carries ``job_posting_id``/``company_key`` the way every new
    producer writes them; its title is deliberately a different spelling so a
    test cannot pass by matching prose.
    """
    message = _message("上周的调研好了。", kind="job_research_report", resource_id="report-h1")
    bound = message.model_copy(
        update={
            "resource_refs": (
                message.resource_refs[0].model_copy(
                    update={
                        "title": f"{report_company}（研究）",
                        "job_posting_id": report_job,
                        "company_key": company_key(report_company),
                    }
                ),
            )
        }
    )
    observations = (
        (
            DecisionObservation(
                tool_name="find_saved_jobs",
                state="saved_jobs_found",
                message=f"找到 {len(candidates)} 个已保存岗位。",
            ),
        )
        if searched_this_turn
        else ()
    )
    return _context(
        bound,
        task=ConversationTaskState(
            saved_job_candidates=tuple(
                SavedJobCandidateContextItem(
                    job_posting_id=job_id, title="算法工程师", company_name=company
                )
                for job_id, company in candidates
            ),
        ),
    ).model_copy(
        update={"user_message": user_message, "tool_observations": observations}
    )


def test_a_report_handle_for_another_company_is_refused_when_the_user_wrote_an_alias() -> None:
    """The user wrote 字节, the job is saved as 字节跳动, the report is about 历史科技甲.

    No spelling of the company appears verbatim anywhere, so a text check would
    let the wrong report through. The search the model ran this turn resolved
    字节 to the saved job, and the report's company key says it is not that
    employer.
    """
    context = _bound_report_context(
        "字节那份调研里，他们的主要竞争对手是谁？", searched_this_turn=True
    )

    with pytest.raises(ValueError, match="selection_index 1（字节跳动）"):
        project_job_research_arguments(
            context, "get_job_research", {"reference": _only_handle(context)}
        )


def test_an_alias_written_without_a_search_still_refuses_another_companys_report() -> None:
    """No find_saved_jobs ran this turn, so nothing resolved 字节 to an entity.

    The leading part of a saved job's company name is still enough to know the
    request is about that company and not the report's; the refusal points at
    the saved job so the model resolves the alias instead of guessing a title.
    """
    context = _bound_report_context("字节那份调研里，他们的主要竞争对手是谁？")

    with pytest.raises(ValueError, match="find_saved_jobs"):
        project_job_research_arguments(
            context, "get_job_research", {"reference": _only_handle(context)}
        )


def test_a_short_name_heading_several_saved_companies_is_refused_as_ambiguous() -> None:
    """"中国" heads both 中国移动 and 中国银行; the report being about one of them
    does not make it the one the user meant."""
    context = _bound_report_context(
        "中国那份调研怎么说",
        report_company="中国银行",
        candidates=(("job-1", "中国移动"), ("job-2", "中国银行")),
    )

    with pytest.raises(ValueError, match="more than one saved company"):
        project_job_research_arguments(
            context, "get_job_research", {"reference": _only_handle(context)}
        )


def test_a_short_name_shared_by_the_held_report_and_a_saved_job_is_ambiguous() -> None:
    """腾讯 heads both the report's 腾讯 and the saved job's 腾讯音乐, so neither is
    picked; the model has to ask."""
    context = _bound_report_context(
        "腾讯那份调研怎么说",
        report_company="腾讯",
        candidates=(("job-1", "腾讯音乐"),),
    )

    with pytest.raises(ValueError, match="more than one saved company"):
        project_job_research_arguments(
            context, "get_job_research", {"reference": _only_handle(context)}
        )


def test_an_ascii_short_name_needs_three_characters() -> None:
    """"AB" is too short to name "ABC Robotics", so the handle is left to the
    model; "abc" is enough and sends it to the saved job."""
    kept = _bound_report_context(
        "AB 那份调研怎么说", candidates=(("job-1", "ABC Robotics"),)
    )
    projected = project_job_research_arguments(
        kept, "get_job_research", {"reference": _only_handle(kept)}
    )
    assert projected == {"user_id": "u1", "report_id": "report-h1"}

    refused = _bound_report_context(
        "abc 那份调研怎么说", candidates=(("job-1", "ABC Robotics"),)
    )
    with pytest.raises(ValueError, match="selection_index 1（ABC Robotics）"):
        project_job_research_arguments(
            refused, "get_job_research", {"reference": _only_handle(refused)}
        )


def test_a_report_handle_is_kept_when_bound_to_the_company_the_alias_resolved_to() -> None:
    """Same alias, but the held report is about the company the search found.

    Its title spells the company differently from the saved job; identity, not
    title text, is what keeps the read grounded.
    """
    context = _bound_report_context(
        "字节那份调研里，他们的主要竞争对手是谁？",
        report_job="job-other-role",
        report_company="BYTEDANCE  LTD",
        candidates=(("job-1", "ByteDance Ltd"),),
        searched_this_turn=True,
    )
    assert "ByteDance Ltd" not in (
        context.recent_messages[0].resource_refs[0].title or ""
    )

    projected = project_job_research_arguments(
        context, "get_job_research", {"reference": _only_handle(context)}
    )
    assert projected == {"user_id": "u1", "report_id": "report-h1"}


def test_a_report_handle_is_kept_when_it_anchors_the_job_the_alias_resolved_to() -> None:
    """A report anchored by the very saved job is about that job's company,
    whatever its stored company key says."""
    context = _bound_report_context(
        "字节那份调研里怎么说的",
        report_job="job-1",
        report_company="旧名字",
        searched_this_turn=True,
    )

    projected = project_job_research_arguments(
        context, "get_job_research", {"reference": _only_handle(context)}
    )
    assert projected == {"user_id": "u1", "report_id": "report-h1"}


def test_a_mixed_search_result_does_not_pin_the_request_to_one_company() -> None:
    """A role search returning several employers says nothing about which one
    the user meant, so the handle is left to the model."""
    context = _bound_report_context(
        "那份调研里怎么说的",
        candidates=(("job-1", "字节跳动"), ("job-2", "示例科技")),
        searched_this_turn=True,
    )

    projected = project_job_research_arguments(
        context, "get_job_research", {"reference": _only_handle(context)}
    )
    assert projected == {"user_id": "u1", "report_id": "report-h1"}


def test_a_bound_report_is_refused_on_identity_even_when_its_title_echoes_the_name() -> None:
    """A title containing the asked-for name cannot vouch for a reference whose
    identity says it is about someone else."""
    context = _bound_report_context("字节跳动那份调研怎么说", report_company="字节跳动前员工创业公司")
    reference = context.recent_messages[0].resource_refs[0]
    assert "字节跳动" in (reference.title or "")

    with pytest.raises(ValueError, match="selection_index 1（字节跳动）"):
        project_job_research_arguments(
            context, "get_job_research", {"reference": _only_handle(context)}
        )


def test_company_identity_on_a_reference_is_scoped_to_job_research() -> None:
    with pytest.raises(ValueError, match="scoped to job research"):
        ConversationResourceReference(
            kind="mock_interview_report", resource_id="rep-1", company_key="acme"
        )


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
