from __future__ import annotations

import json

from career_agent.agent.career_context import CareerContextProjector
from career_agent.agent.main_agent_contracts import (
    CareerProfileBudgets,
    CareerProfileContext,
    CurrentTargetContext,
    HardConstraintContext,
    MainAgentContext,
    ToolObservation,
)
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.storage.resumes import ResumeStore
from career_agent.storage.career_history import CareerHistoryStore


def _decode_tier_one(projected: dict[str, object]) -> list[dict[str, object]]:
    records_table = projected["records"]
    claims_table = projected["claims"]
    assert isinstance(records_table, dict)
    assert isinstance(claims_table, dict)
    records = [
        dict(zip(records_table["fields"], row, strict=True))
        for row in records_table["rows"]
    ]
    for record in records:
        record["confirmed_highlights"] = []
    for row in claims_table["rows"]:
        claim = dict(zip(claims_table["fields"], row, strict=True))
        record_index = claim.pop("record")
        records[record_index]["confirmed_highlights"].append(claim)
    return records


def _confirmed_highlight(
    store: CareerHistoryStore,
    *,
    user_id: str,
    career_record_id: str,
    claim: str,
):
    evidence = store.create_evidence(
        user_id=user_id,
        career_record_id=career_record_id,
        claim=claim,
        origin="user_input",
    )
    return store.confirm_evidence(
        user_id=user_id,
        career_evidence_id=evidence.id,
    )


def test_historical_source_presenter_never_renders_a_missing_timestamp() -> None:
    result = ToolObservation(
        tool_name="resolve_claim_source",
        state="claim_source_found",
        message="历史证据。",
        facts={"claim_status": "superseded"},
        payload={"source_quote": "Historical quote"},
    )

    rendered = MainAgentRuntime._assistant_message(result)

    assert "状态变更时间：未记录" in rendered
    assert "None" not in rendered


def test_projector_injects_only_bounded_confirmed_career_memory(tmp_path) -> None:
    store = CareerHistoryStore(tmp_path / "career.sqlite3")
    relevant = store.create_record(
        user_id="u1",
        record_type="project",
        title="RAG Evaluation Platform",
        start_year=2025,
    )
    other = store.create_record(
        user_id="u1",
        record_type="work",
        organization="Example Inc.",
        title="Backend Engineer",
        start_year=2024,
    )
    _confirmed_highlight(
        store,
        user_id="u1",
        career_record_id=relevant.id,
        claim="Built a RAG evaluation pipeline",
    )
    pending = store.create_evidence(
        user_id="u1",
        career_record_id=relevant.id,
        claim="PRIVATE UNCONFIRMED CLAIM",
        origin="user_input",
    )
    assert pending.verification_status == "pending"
    _confirmed_highlight(
        store,
        user_id="u1",
        career_record_id=other.id,
        claim="Maintained payment services",
    )

    memory = CareerContextProjector(store, candidate_record_limit=1).project(
        user_id="u1",
        query="帮我匹配一个 RAG 岗位",
    )

    assert len(memory.records) == 1
    assert memory.records_total == 2
    assert memory.claims_total == 2
    assert memory.records[0].title == "RAG Evaluation Platform"
    assert len(memory.records[0].confirmed_highlights) == 1
    highlight = memory.records[0].confirmed_highlights[0]
    assert highlight.claim == "Built a RAG evaluation pipeline"
    assert highlight.origin == "user_input"
    assert highlight.source_ref is None
    serialized = memory.model_dump_json()
    assert "PRIVATE UNCONFIRMED CLAIM" not in serialized
    assert "career_record_" not in serialized
    assert "career_evidence_" not in serialized


def test_main_agent_model_context_contains_compact_provenance_without_quotes(
    tmp_path,
) -> None:
    store = CareerHistoryStore(tmp_path / "career.sqlite3")
    record = store.create_record(
        user_id="u1",
        record_type="work",
        organization="Example Inc.",
        title="Product Manager",
        start_year=2022,
        is_current=True,
    )
    _confirmed_highlight(
        store,
        user_id="u1",
        career_record_id=record.id,
        claim="Led knowledge-base planning",
    )
    memory = CareerContextProjector(store).project(user_id="u1", query="职业规划")
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        career_memory=memory,
        user_message="帮我做职业规划",
    )

    projected = context.model_context()["career_profile"]

    records = _decode_tier_one(projected)
    assert records[0]["title"] == "Product Manager"
    highlight = records[0]["confirmed_highlights"][0]
    assert highlight["claim"] == "Led knowledge-base planning"
    assert highlight["origin"] == "user_input"
    assert "recorded_at" in highlight
    assert highlight["source_ref"] is None
    assert highlight["revision"] == 1
    assert str(highlight["detail_ref"]).startswith("detail_")
    assert "supported_by" not in str(projected)
    assert "superseded_by" not in str(projected)
    assert "lineage_ref" not in str(projected)
    assert "source_quote" not in str(projected)
    assert "source_locator" not in str(projected)
    assert "user_id" not in str(projected)


def test_projector_exposes_only_the_current_evidence_revision(tmp_path) -> None:
    store = CareerHistoryStore(tmp_path / "career.sqlite3")
    record = store.create_record(
        user_id="u1",
        record_type="project",
        title="Retrieval Evaluation",
    )
    original = _confirmed_highlight(
        store,
        user_id="u1",
        career_record_id=record.id,
        claim="Assisted with retrieval evaluation",
    )
    corrected = store.correct_evidence(
        user_id="u1",
        career_evidence_id=original.id,
        new_claim="Led retrieval evaluation",
        reason="User corrected ownership",
    )

    memory = CareerContextProjector(store).project(
        user_id="u1", query="retrieval evaluation"
    )
    claims = [
        item.claim
        for projected_record in memory.records
        for item in projected_record.confirmed_highlights
    ]

    assert claims == [corrected.current.claim]
    assert original.claim not in memory.model_dump_json()


def test_resume_provenance_is_an_opaque_rereadable_ref_not_inline_text(
    tmp_path,
) -> None:
    path = tmp_path / "career.sqlite3"
    resumes = ResumeStore(path)
    store = CareerHistoryStore(path)
    role = resumes.create_target_role(
        user_id="u1",
        title="ML Engineer",
        priority=1,
    )
    _, version = resumes.import_document(
        user_id="u1",
        content=b"Built retrieval evaluation",
        document_format="text",
        name="Primary",
        target_role_id=role.id,
    )
    record = store.create_record(
        user_id="u1",
        record_type="project",
        title="Retrieval Evaluation",
    )
    evidence = store.create_evidence(
        user_id="u1",
        career_record_id=record.id,
        claim="Built retrieval evaluation",
        origin="resume_extraction",
        source_resume_version_id=version.id,
        source_locator="Projects / Retrieval",
        source_quote="Project detail: evaluated retrieval quality across five datasets.",
    )
    confirmed = store.confirm_evidence(
        user_id="u1",
        career_evidence_id=evidence.id,
    )
    projector = CareerContextProjector(store)

    memory = projector.project(user_id="u1", query="retrieval")
    highlight = memory.records[0].confirmed_highlights[0]
    serialized = memory.model_dump_json()

    assert highlight.recorded_at == evidence.created_at
    assert highlight.source_ref is not None
    assert evidence.id not in serialized
    assert version.id not in serialized
    assert evidence.source_quote not in serialized
    assert evidence.source_locator not in serialized
    assert projector.resolve_source_ref(
        user_id="u1",
        source_ref=highlight.source_ref,
    ) == confirmed

    tools = MainAgentToolRegistry(
        career_history_store=store,
        resume_store=resumes,
    )
    observation = tools.invoke_atomic_tool(
        "resolve_claim_source",
        {"user_id": "u1", "source_ref": highlight.source_ref},
    )
    missing = tools.invoke_atomic_tool(
        "resolve_claim_source",
        {"user_id": "u1", "source_ref": f"evidence_{'0' * 24}"},
    )

    assert observation.state == "claim_source_found"
    assert observation.payload["source_quote"] == evidence.source_quote
    assert evidence.source_quote not in str(observation.facts)
    assert observation.facts["source_locator"] == evidence.source_locator
    assert observation.facts["resume_version"] == "Primary · 第 1 版"
    assert observation.facts["claim_status"] == "current"
    assert "status_changed_at" not in observation.facts
    turn_observation = MainAgentRuntime._tool_observation(
        "resolve_claim_source",
        observation,
    )
    assert evidence.source_quote in (turn_observation.body or "")
    assert evidence.source_quote not in str(turn_observation.facts)
    assert missing.state == "claim_source_not_found"

    correction = store.correct_evidence(
        user_id="u1",
        career_evidence_id=confirmed.id,
        new_claim="Contributed to retrieval evaluation",
        reason="User corrected the level of ownership",
    )
    historical = tools.invoke_atomic_tool(
        "resolve_claim_source",
        {"user_id": "u1", "source_ref": highlight.source_ref},
    )
    historical_turn = MainAgentRuntime._tool_observation(
        "resolve_claim_source",
        historical,
    )

    assert historical.state == "claim_source_found"
    assert historical.facts["claim_status"] == "superseded"
    assert historical.facts["status_changed_at"] == (
        correction.previous.superseded_at.isoformat()
    )
    assert "不能作为当前声明的支持" in historical.message
    assert "已被更正声明的历史引文" in (historical_turn.body or "")
    assert evidence.source_quote in (historical_turn.body or "")
    corrected_memory = projector.project(user_id="u1", query="retrieval")
    corrected_highlight = corrected_memory.records[0].confirmed_highlights[0]
    assert corrected_highlight.claim == correction.current.claim
    assert corrected_highlight.source_ref is None
    detail = tools.invoke_atomic_tool(
        "get_career_memory_detail",
        {"user_id": "u1", "detail_ref": corrected_highlight.detail_ref},
    )
    history = tools.invoke_atomic_tool(
        "search_career_history",
        {"user_id": "u1", "query": "retrieval", "limit": 8},
    )
    assert detail.state == "career_memory_detail_found"
    assert detail.payload["supported_by"] == []
    assert detail.payload["lineage_ref"].startswith("lineage_")
    assert detail.payload["lineage_ref"] != highlight.source_ref
    assert [item["claim_status"] for item in detail.payload["lineage"]] == [
        "superseded",
        "current",
    ]
    assert history.state == "career_history_found"
    assert history.facts["returned"] == 1
    assert history.payload["items"][0]["claim"] == confirmed.claim
    assert history.payload["items"][0]["claim_status"] == "superseded"

    long_evidence = store.create_evidence(
        user_id="u1",
        career_record_id=record.id,
        claim="Maintained a detailed evaluation log",
        origin="resume_extraction",
        source_resume_version_id=version.id,
        source_locator="Projects / Evaluation log",
        source_quote="证" * 7_000,
    )
    long_evidence = store.confirm_evidence(
        user_id="u1",
        career_evidence_id=long_evidence.id,
    )
    long_result = tools.invoke_atomic_tool(
        "resolve_claim_source",
        {"user_id": "u1", "source_ref": long_evidence.source_ref},
    )
    long_turn_observation = MainAgentRuntime._tool_observation(
        "resolve_claim_source",
        long_result,
    )

    assert long_result.facts["body_clipped"] is True
    assert len(long_turn_observation.body or "") <= 6_000


def test_columnar_projection_is_semantically_equivalent_and_budgeted(tmp_path) -> None:
    store = CareerHistoryStore(tmp_path / "career.sqlite3")
    for record_index in range(5):
        record = store.create_record(
            user_id="u1",
            record_type="project",
            title=f"Memory Evaluation {record_index}",
        )
        for claim_index in range(3):
            _confirmed_highlight(
                store,
                user_id="u1",
                career_record_id=record.id,
                claim=(
                    f"Built memory benchmark {record_index}-{claim_index} "
                    + "x" * 80
                ),
            )
    memory = CareerContextProjector(store).project(
        user_id="u1",
        query="memory benchmark",
    )
    full = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        career_memory=memory,
        career_profile_budgets=CareerProfileBudgets(records_chars=20_000),
        user_message="memory",
    ).model_context()["career_profile"]
    decoded = _decode_tier_one(full)

    expected = [
        record.model_dump(mode="json")
        for record in memory.records
    ]
    assert decoded == expected

    memory_keys = {
        "records",
        "claims",
        "records_returned",
        "records_total",
        "claims_returned",
        "claims_total",
    }
    onto_memory = {
        key: value for key, value in full.items() if key in memory_keys
    }
    naive_m4b = {"records": expected}
    legacy_m4a = {
        "records": [
            {
                **{
                    key: value
                    for key, value in record.items()
                    if key != "confirmed_highlights"
                },
                "confirmed_highlights": [
                    {
                        key: value
                        for key, value in claim.items()
                        if key not in {"revision", "detail_ref"}
                    }
                    for claim in record["confirmed_highlights"]
                ],
            }
            for record in expected
        ]
    }
    onto_chars = len(json.dumps(onto_memory, ensure_ascii=False, sort_keys=True))
    naive_chars = len(json.dumps(naive_m4b, ensure_ascii=False, sort_keys=True))
    legacy_chars = len(json.dumps(legacy_m4a, ensure_ascii=False, sort_keys=True))
    assert onto_chars <= naive_chars * 0.8
    assert onto_chars <= legacy_chars * 1.1

    budget = 700
    bounded = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        career_memory=memory,
        career_profile_budgets=CareerProfileBudgets(records_chars=budget),
        user_message="memory",
    ).model_context()["career_profile"]
    bounded_memory = {
        key: value for key, value in bounded.items() if key in memory_keys
    }

    assert len(json.dumps(bounded_memory, ensure_ascii=False, sort_keys=True)) <= budget
    assert bounded_memory["claims_returned"] < bounded_memory["claims_total"]

    zero_records = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(
            user_id="u1",
            hard_constraints=(
                HardConstraintContext(
                    relation="work_arrangement",
                    value="必须远程",
                ),
            ),
            current_targets=(
                CurrentTargetContext(
                    title="ML Engineer",
                    priority=1,
                    salary_expectation="40-60k",
                ),
            ),
        ),
        career_memory=memory,
        career_profile_budgets=CareerProfileBudgets(records_chars=0),
        user_message="memory",
    ).model_context()["career_profile"]
    assert {
        key: zero_records[key]
        for key in memory_keys
        if key in zero_records
    } == {
        "records_returned": 0,
        "records_total": 5,
        "claims_returned": 0,
        "claims_total": 15,
    }
    assert zero_records["hard_constraints"] == [
        {"relation": "work_arrangement", "value": "必须远程"}
    ]
    assert zero_records["current_targets"]["roles"] == [
        {
            "title": "ML Engineer",
            "priority": 1,
            "salary_expectation": "40-60k",
        }
    ]

    for budget in (0, 1):
        exhausted = MainAgentContext(
            conversation_id="c1",
            profile=CareerProfileContext(
                user_id="u1",
                hard_constraints=(
                    HardConstraintContext(
                        relation="work_arrangement",
                        value="必须远程",
                    ),
                ),
                current_targets=(
                    CurrentTargetContext(
                        title="ML Engineer",
                        priority=1,
                    ),
                ),
            ),
            career_memory=memory,
            career_profile_budgets=CareerProfileBudgets(
                records_chars=budget,
                current_targets_chars=budget,
                hard_constraints_chars=budget,
            ),
            user_message="memory",
        ).model_context()["career_profile"]

        assert not {"records", "claims", "hard_constraints", "current_targets"} & set(
            exhausted
        )
        assert exhausted["records_returned"] == 0
        assert exhausted["records_total"] == 5
        assert exhausted["claims_returned"] == 0
        assert exhausted["claims_total"] == 15
        assert exhausted["hard_constraints_returned"] == 0
        assert exhausted["hard_constraints_total"] == 1
        assert exhausted["current_targets_returned"] == 0
        assert exhausted["current_targets_total"] == 1


def test_current_target_budget_reports_visible_truncation() -> None:
    targets = tuple(
        CurrentTargetContext(
            title=f"Target role {index} " + "x" * 70,
            priority=index,
            salary_expectation="40-60k " + "y" * 70,
        )
        for index in range(3)
    )
    projected = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(
            user_id="u1",
            current_targets=targets,
            current_targets_total=5,
        ),
        career_profile_budgets=CareerProfileBudgets(
            current_targets_chars=400,
        ),
        user_message="compare tracks",
    ).model_context()["career_profile"]
    target_block = projected["current_targets"]

    assert 0 < target_block["roles_returned"] < len(targets)
    assert target_block["roles_total"] == 5
    assert len(target_block["roles"]) == target_block["roles_returned"]
    assert (
        len(
            json.dumps(
                {"current_targets": target_block},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        <= 400
    )


def test_tier_one_omits_unimplemented_and_historical_fields(tmp_path) -> None:
    store = CareerHistoryStore(tmp_path / "career.sqlite3")
    first = store.create_record(
        user_id="u1",
        record_type="project",
        title="Retrieval Evaluation",
    )
    second = store.create_record(
        user_id="u1",
        record_type="work",
        title="Backend Engineer",
        organization="Example Inc.",
    )
    original = _confirmed_highlight(
        store,
        user_id="u1",
        career_record_id=first.id,
        claim="Assisted with retrieval evaluation",
    )
    store.correct_evidence(
        user_id="u1",
        career_evidence_id=original.id,
        new_claim="Led retrieval evaluation",
        reason="User corrected ownership",
    )
    _confirmed_highlight(
        store,
        user_id="u1",
        career_record_id=second.id,
        claim="Maintained payment services",
    )
    memory = CareerContextProjector(store).project(
        user_id="u1",
        query="retrieval evaluation",
    )
    projected = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        career_memory=memory,
        user_message="retrieval",
    ).model_context()["career_profile"]
    rendered = json.dumps(projected, ensure_ascii=False)
    claims = [
        dict(zip(projected["claims"]["fields"], row, strict=True))
        for row in projected["claims"]["rows"]
    ]

    assert "supported_by" not in rendered
    assert "superseded_by" not in rendered
    assert "lineage_ref" not in rendered
    assert original.claim not in rendered
    values_by_field = {
        field: {claim[field] for claim in claims}
        for field in projected["claims"]["fields"]
        if field != "record"
    }
    constant = {
        field: values
        for field, values in values_by_field.items()
        if len(values) == 1
    }
    assert "revision" not in constant
    assert "detail_ref" not in constant
    assert "claim" not in constant


def test_historical_tool_exposes_and_consumes_an_opaque_page_cursor(
    tmp_path,
) -> None:
    store = CareerHistoryStore(tmp_path / "career.sqlite3")
    record = store.create_record(
        user_id="u1",
        record_type="project",
        title="Retrieval Evaluation",
    )
    for index in range(3):
        original = _confirmed_highlight(
            store,
            user_id="u1",
            career_record_id=record.id,
            claim=f"Assisted with retrieval evaluation {index}",
        )
        store.correct_evidence(
            user_id="u1",
            career_evidence_id=original.id,
            new_claim=f"Led retrieval evaluation {index}",
            reason="User corrected ownership",
        )
    tools = MainAgentToolRegistry(career_history_store=store)

    first = tools.invoke_atomic_tool(
        "search_career_history",
        {"user_id": "u1", "query": "retrieval", "limit": 2},
    )
    second = tools.invoke_atomic_tool(
        "search_career_history",
        {
            "user_id": "u1",
            "query": "retrieval",
            "limit": 2,
            "cursor": first.facts["next_cursor"],
        },
    )
    invalid = tools.invoke_atomic_tool(
        "search_career_history",
        {
            "user_id": "u1",
            "query": "different",
            "limit": 2,
            "cursor": first.facts["next_cursor"],
        },
    )

    assert first.state == "career_history_found"
    assert first.facts["returned"] == 2
    assert first.facts["total"] == 3
    assert str(first.facts["next_cursor"]).startswith("history_")
    assert second.state == "career_history_found"
    assert second.facts == {
        "returned": 1,
        "total": 3,
        "body_clipped": False,
    }
    assert invalid.state == "invalid_input"
