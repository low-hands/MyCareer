from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import sqlite3

import pytest

from career_agent.agent.career_context import (
    CareerContextProjector,
    reciprocal_rank_fusion,
)
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
    records = projected["records"]
    assert isinstance(records, list)
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


def test_lexical_match_excludes_a_newer_current_nonmatch(tmp_path) -> None:
    store = CareerHistoryStore(tmp_path / "career.sqlite3")
    relevant = store.create_record(
        user_id="u1",
        record_type="project",
        title="Old retrieval project",
        start_year=2018,
    )
    recent = store.create_record(
        user_id="u1",
        record_type="work",
        title="Current payments role",
        start_year=2026,
        is_current=True,
    )
    _confirmed_highlight(
        store,
        user_id="u1",
        career_record_id=relevant.id,
        claim="Designed retrieval evaluation for multilingual search",
    )
    _confirmed_highlight(
        store,
        user_id="u1",
        career_record_id=recent.id,
        claim="Maintained payment reconciliation services",
    )

    memory = CareerContextProjector(store).project(
        user_id="u1",
        query="multilingual retrieval evaluation",
    )

    assert memory.records[0].title == "Old retrieval project"


def test_record_metadata_cannot_enter_the_lexical_candidate_set(tmp_path) -> None:
    store = CareerHistoryStore(tmp_path / "career.sqlite3")
    empty = store.create_record(
        user_id="u1",
        record_type="project",
        title="支付清算系统重构项目",
    )
    evidenced = store.create_record(
        user_id="u1",
        record_type="work",
        title="后端工程师",
    )
    _confirmed_highlight(
        store,
        user_id="u1",
        career_record_id=evidenced.id,
        claim="负责支付清算系统重构",
    )

    memory = CareerContextProjector(store).project(
        user_id="u1",
        query="支付清算系统重构",
    )

    assert [record.title for record in memory.records] == ["后端工程师"]
    assert empty.id not in {
        item.career_record_id
        for item in store.rank_current_evidence(
            user_id="u1", query="支付清算系统重构"
        )
    }


def test_nonmatching_current_record_does_not_consume_claim_budget(tmp_path) -> None:
    store = CareerHistoryStore(tmp_path / "career.sqlite3")
    retrieval = store.create_record(
        user_id="u1",
        record_type="project",
        title="检索项目",
    )
    current = store.create_record(
        user_id="u1",
        record_type="work",
        title="当前工作",
        is_current=True,
    )
    for index in range(5):
        _confirmed_highlight(
            store,
            user_id="u1",
            career_record_id=retrieval.id,
            claim=f"检索评估与召回优化成果 {index}",
        )
    _confirmed_highlight(
        store,
        user_id="u1",
        career_record_id=current.id,
        claim="维护支付服务",
    )

    memory = CareerContextProjector(
        store,
        tier_one_claim_limit=5,
    ).project(
        user_id="u1",
        query="检索评估召回优化",
    )

    assert [record.title for record in memory.records] == ["检索项目"]
    assert len(memory.records[0].confirmed_highlights) == 5


def test_short_cjk_query_uses_the_fts_candidate_set(tmp_path) -> None:
    store = CareerHistoryStore(tmp_path / "career.sqlite3")
    concise = store.create_record(
        user_id="u1",
        record_type="project",
        title="Concise",
    )
    verbose = store.create_record(
        user_id="u1",
        record_type="project",
        title="Verbose",
    )
    concise_evidence = _confirmed_highlight(
        store,
        user_id="u1",
        career_record_id=concise.id,
        claim="远程",
    )
    verbose_evidence = _confirmed_highlight(
        store,
        user_id="u1",
        career_record_id=verbose.id,
        claim="这个岗位支持远程办公并提供完整的跨团队协作流程",
    )

    query_terms = store.current_evidence_query_terms(user_id="u1", query="远程")
    ranked = store.rank_current_evidence(
        user_id="u1",
        query_terms=query_terms,
    )
    assert query_terms.latin == ()
    assert query_terms.cjk == ("远程",)
    assert {hit.id for hit in ranked} == {
        concise_evidence.id,
        verbose_evidence.id,
    }
    memory = CareerContextProjector(store).project(
        user_id="u1",
        query="远程",
    )

    assert {record.title for record in memory.records} == {"Concise", "Verbose"}


def test_short_latin_query_uses_the_fts_candidate_set(
    tmp_path,
) -> None:
    store = CareerHistoryStore(tmp_path / "career.sqlite3")
    concise = store.create_record(
        user_id="u1",
        record_type="project",
        title="Concise",
    )
    verbose = store.create_record(
        user_id="u1",
        record_type="project",
        title="Verbose",
    )
    _confirmed_highlight(
        store,
        user_id="u1",
        career_record_id=concise.id,
        claim="AI Go",
    )
    _confirmed_highlight(
        store,
        user_id="u1",
        career_record_id=verbose.id,
        claim="Built AI services and Go systems across several platform teams",
    )

    query_terms = store.current_evidence_query_terms(user_id="u1", query="AI Go")
    ranked = store.rank_current_evidence(
        user_id="u1",
        query_terms=query_terms,
    )
    assert query_terms.latin == ("ai", "go")
    assert query_terms.cjk == ()
    assert len(ranked) == 2
    assert {hit.claim for hit in ranked} == {
        "AI Go",
        "Built AI services and Go systems across several platform teams",
    }
    memory = CareerContextProjector(store).project(
        user_id="u1",
        query="AI Go",
    )

    assert {record.title for record in memory.records} == {"Concise", "Verbose"}


def test_domain_short_terms_are_not_silently_dropped(tmp_path) -> None:
    store = CareerHistoryStore(tmp_path / "career.sqlite3")
    record = store.create_record(
        user_id="u1",
        record_type="project",
        title="Short terms",
    )
    claims = {
        "AI算法工程师": _confirmed_highlight(
            store,
            user_id="u1",
            career_record_id=record.id,
            claim="AI算法工程师",
        ),
        "Go 后端开发": _confirmed_highlight(
            store,
            user_id="u1",
            career_record_id=record.id,
            claim="Go 后端开发",
        ),
        "C# 开发": _confirmed_highlight(
            store,
            user_id="u1",
            career_record_id=record.id,
            claim="C# 开发",
        ),
        "远程薪资面试": _confirmed_highlight(
            store,
            user_id="u1",
            career_record_id=record.id,
            claim="远程薪资面试",
        ),
    }

    expected_terms = {
        "AI算法工程师": (("ai",), ("算法工", "法工程", "工程师")),
        "Go 后端开发": (("go",), ("后端开", "端开发")),
        "C# 开发": (("c#",), ("开发",)),
        "算法工程师 AI Go": (
            ("ai", "go"),
            ("算法工", "法工程", "工程师"),
        ),
        "远程": ((), ("远程",)),
        "薪资": ((), ("薪资",)),
        "面试": ((), ("面试",)),
    }
    expected_hit = {
        "AI算法工程师": claims["AI算法工程师"].id,
        "Go 后端开发": claims["Go 后端开发"].id,
        "C# 开发": claims["C# 开发"].id,
        "算法工程师 AI Go": claims["AI算法工程师"].id,
        "远程": claims["远程薪资面试"].id,
        "薪资": claims["远程薪资面试"].id,
        "面试": claims["远程薪资面试"].id,
    }
    for query, (latin, cjk) in expected_terms.items():
        query_terms = store.current_evidence_query_terms(
            user_id="u1",
            query=query,
        )
        assert query_terms.latin == latin
        assert query_terms.cjk == cjk
        ranked = store.rank_current_evidence(
            user_id="u1",
            query_terms=query_terms,
        )
        assert ranked
        assert expected_hit[query] in {item.id for item in ranked}


def test_match_gate_ignores_bm25_population_and_keeps_scope_filters(
    tmp_path,
) -> None:
    store = CareerHistoryStore(tmp_path / "career.sqlite3")
    record = store.create_record(
        user_id="u1",
        record_type="project",
        title="Revision isolation",
    )
    current = _confirmed_highlight(
        store,
        user_id="u1",
        career_record_id=record.id,
        claim="selective AI revision 0",
    )
    for index in range(35):
        _confirmed_highlight(
            store,
            user_id="u1",
            career_record_id=record.id,
            claim=f"unrelated current evidence {index}",
        )

    def snapshot() -> tuple[str, ...]:
        query_terms = store.current_evidence_query_terms(
            user_id="u1",
            query="selective AI",
        )
        ranked = store.rank_current_evidence(
            user_id="u1",
            query_terms=query_terms,
        )
        assert len(ranked) == 1
        return tuple(item.claim for item in ranked)

    baseline = snapshot()
    with pytest.raises(ValueError, match="do not belong"):
        store.rank_current_evidence(
            user_id="u2",
            query_terms=store.current_evidence_query_terms(
                user_id="u1",
                query="selective AI",
            ),
        )
    for revision in range(1, 7):
        current = store.correct_evidence(
            user_id="u1",
            career_evidence_id=current.id,
            new_claim=f"selective AI revision {revision}",
            reason="Exercise revision isolation",
        ).current

    pending_record = store.create_record(
        user_id="u1",
        record_type="project",
        title="Pending evidence",
    )
    for index in range(6):
        store.create_evidence(
            user_id="u1",
            career_record_id=pending_record.id,
            claim=f"selective AI pending {index}",
            origin="user_input",
        )
    other_record = store.create_record(
        user_id="u2",
        record_type="project",
        title="Other user",
    )
    for index in range(6):
        _confirmed_highlight(
            store,
            user_id="u2",
            career_record_id=other_record.id,
            claim=f"selective AI other user {index}",
        )

    assert snapshot() == (current.claim,)
    with sqlite3.connect(store.path) as connection:
        indexed_superseded = connection.execute(
            """
            SELECT COUNT(*)
            FROM career_evidence_fts AS search
            JOIN career_evidence AS evidence
              ON evidence.id = search.evidence_id
            WHERE career_evidence_fts MATCH '"selective"'
              AND evidence.user_id = 'u1'
              AND evidence.superseded_by IS NOT NULL
            """
        ).fetchone()[0]
    assert indexed_superseded == 6


def test_query_terms_are_independent_of_corpus_size(tmp_path) -> None:
    empty = CareerHistoryStore(tmp_path / "empty.sqlite3")
    populated = CareerHistoryStore(tmp_path / "populated.sqlite3")
    record = populated.create_record(
        user_id="u1",
        record_type="project",
        title="Population",
    )
    for index in range(60):
        _confirmed_highlight(
            populated,
            user_id="u1",
            career_record_id=record.id,
            claim=f"unrelated filler {index}",
        )

    empty_terms = empty.current_evidence_query_terms(
        user_id="u1", query="AI 推荐系统召回"
    )
    populated_terms = populated.current_evidence_query_terms(
        user_id="u1", query="AI 推荐系统召回"
    )

    assert populated_terms == empty_terms
    assert populated_terms.latin == ("ai",)
    assert populated_terms.cjk == ("推荐系", "荐系统", "系统召", "统召回")


def test_rrf_fuses_lexical_and_semantic_ranks_without_raw_scores() -> None:
    fused = reciprocal_rank_fusion(
        ("lexical-only", "both"),
        ("both", "semantic-only"),
    )

    assert fused["both"] > fused["lexical-only"]
    assert fused["both"] > fused["semantic-only"]


def test_semantic_channel_can_rescue_a_lexical_nonmatch(tmp_path) -> None:
    store = CareerHistoryStore(tmp_path / "career.sqlite3")
    semantic_record = store.create_record(
        user_id="u1",
        record_type="project",
        title="推荐系统",
    )
    semantic_evidence = _confirmed_highlight(
        store,
        user_id="u1",
        career_record_id=semantic_record.id,
        claim="负责推荐系统召回与排序",
    )

    class SemanticRetriever:
        def rank_current_evidence_ids(self, *, user_id, query, limit):
            assert (user_id, query, limit) == (
                "u1",
                "candidate generation",
                45,
            )
            return (semantic_evidence.id,)

    memory = CareerContextProjector(
        store,
        semantic_retriever=SemanticRetriever(),
    ).project(user_id="u1", query="candidate generation")

    assert [record.title for record in memory.records] == ["推荐系统"]
    assert memory.records[0].confirmed_highlights[0].claim == (
        "负责推荐系统召回与排序"
    )


def test_query_match_outranks_fresh_current_nonmatch_end_to_end(tmp_path) -> None:
    store = CareerHistoryStore(tmp_path / "career.sqlite3")
    relevant = store.create_record(
        user_id="u1",
        record_type="project",
        title="算法工程师",
    )
    unrelated = store.create_record(
        user_id="u1",
        record_type="work",
        title="行政专员",
        is_current=True,
    )
    old_match = _confirmed_highlight(
        store,
        user_id="u1",
        career_record_id=relevant.id,
        claim="推荐系统召回推荐系统召回",
    )
    _confirmed_highlight(
        store,
        user_id="u1",
        career_record_id=unrelated.id,
        claim="负责办公用品采购与会议室排期",
    )
    old_at = datetime.now(timezone.utc) - timedelta(days=120)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE career_evidence SET created_at = ?, updated_at = ? WHERE id = ?",
            (old_at.isoformat(), old_at.isoformat(), old_match.id),
        )

    memory = CareerContextProjector(store).project(
        user_id="u1",
        query="推荐系统召回",
    )

    assert [record.title for record in memory.records] == ["算法工程师"]


def test_tier_one_claim_limit_is_depth_first_after_reranking(tmp_path) -> None:
    store = CareerHistoryStore(tmp_path / "career.sqlite3")
    retrieval = store.create_record(
        user_id="u1",
        record_type="project",
        title="Retrieval",
    )
    payments = store.create_record(
        user_id="u1",
        record_type="project",
        title="Payments",
    )
    for index in range(8):
        _confirmed_highlight(
            store,
            user_id="u1",
            career_record_id=retrieval.id,
            claim=f"Retrieval evaluation result {index}",
        )
        _confirmed_highlight(
            store,
            user_id="u1",
            career_record_id=payments.id,
            claim=f"Payment migration result {index}",
        )

    memory = CareerContextProjector(
        store,
        tier_one_claim_limit=5,
    ).project(
        user_id="u1",
        query="retrieval evaluation",
    )

    assert len(memory.records) == 1
    assert memory.records[0].title == "Retrieval"
    assert len(memory.records[0].confirmed_highlights) == 5
    assert memory.claims_total == 16


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
    memory = CareerContextProjector(store).project(
        user_id="u1", query="knowledge-base planning"
    )
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        career_memory=memory,
        user_message="帮我做职业规划",
    )

    projected = context.model_context()["career_memory"]

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
    assert memory.telemetry_inventory_complete is True
    assert {
        binding.lifecycle_status for binding in memory.telemetry_bindings
    } == {"current", "superseded"}
    assert {
        binding.update_id for binding in memory.telemetry_bindings
    } == {
        corrected.current.update_id,
        corrected.previous.update_id,
    }
    assert all(
        claim.telemetry_binding is not None
        for projected_record in memory.records
        for claim in projected_record.confirmed_highlights
    )


def test_unbound_oversized_claim_downgrades_version_inventory(tmp_path) -> None:
    store = CareerHistoryStore(tmp_path / "career.sqlite3")
    record = store.create_record(
        user_id="u1",
        record_type="project",
        title="Large evidence",
    )
    _confirmed_highlight(
        store,
        user_id="u1",
        career_record_id=record.id,
        claim="x" * 32_001,
    )

    memory = CareerContextProjector(store).project(
        user_id="u1",
        query="xxx",
    )

    assert memory.records[0].confirmed_highlights[0].telemetry_binding is None
    assert memory.telemetry_inventory_complete is False


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

    assert historical.state == "claim_source_not_found"
    assert historical_turn.body is None
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


def test_named_projection_is_semantically_equivalent_and_budgeted(tmp_path) -> None:
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
        career_profile_budgets=CareerProfileBudgets(records_input_units=20_000),
        user_message="memory",
    ).model_context()["career_memory"]
    decoded = _decode_tier_one(full)

    expected = [
        record.model_dump(mode="json")
        for record in memory.records
    ]
    assert decoded == expected

    memory_keys = {
        "records",
        "records_returned",
        "records_total",
        "claims_returned",
        "claims_total",
    }
    named_memory = {
        key: value for key, value in full.items() if key in memory_keys
    }
    assert named_memory == {"records": expected}

    budget = 700
    bounded = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        career_memory=memory,
        career_profile_budgets=CareerProfileBudgets(records_input_units=budget),
        user_message="memory",
    ).model_context()["career_memory"]
    bounded_memory = {
        key: value for key, value in bounded.items() if key in memory_keys
    }

    assert (
        CareerProfileBudgets(records_input_units=budget).estimate_tokens(
            bounded_memory
        )
        <= budget
    )
    assert bounded_memory["claims_returned"] < bounded_memory["claims_total"]

    zero_projection = MainAgentContext(
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
        career_profile_budgets=CareerProfileBudgets(records_input_units=0),
        user_message="memory",
    ).model_context()
    zero_records = zero_projection["career_memory"]
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
    assert 'Work arrangement: "必须远程"' in zero_projection[
        "career_profile"
    ]["memory/profile.md"]
    assert 'Title: "ML Engineer"' in zero_projection["career_profile"][
        "memory/current_targets.md"
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
                records_input_units=budget,
                current_targets_input_units=budget,
                hard_constraints_input_units=budget,
            ),
            user_message="memory",
        ).model_context()
        profile_files = exhausted["career_profile"]
        bounded_memory = exhausted["career_memory"]

        assert 'Work arrangement: "必须远程"' in profile_files[
            "memory/profile.md"
        ]
        assert 'Title: "ML Engineer"' in profile_files[
            "memory/current_targets.md"
        ]
        assert bounded_memory["records_returned"] == 0
        assert bounded_memory["records_total"] == 5
        assert bounded_memory["claims_returned"] == 0
        assert bounded_memory["claims_total"] == 15
        assert bounded_memory["memory_overflow"]["fetch_required"] is True
        assert {
            section["fetch_tool"]
            for section in bounded_memory["memory_overflow"]["sections"]
        } == {"search_career_memory"}


def test_current_target_projection_is_complete_and_ignores_legacy_budget() -> None:
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
            current_targets_total=3,
        ),
        career_profile_budgets=CareerProfileBudgets(
            current_targets_input_units=100,
        ),
        user_message="compare tracks",
    ).model_context()["career_profile"]
    target_file = projected["memory/current_targets.md"]

    assert target_file.count("## Target ") == 3
    assert "Target role 0" in target_file
    assert "Target role 1" in target_file
    assert "Target role 2" in target_file


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
    ).model_context()["career_memory"]
    rendered = json.dumps(projected, ensure_ascii=False)
    claims = [
        claim
        for record in projected["records"]
        for claim in record["confirmed_highlights"]
    ]

    assert "supported_by" not in rendered
    assert "superseded_by" not in rendered
    assert "lineage_ref" not in rendered
    assert original.claim not in rendered
    assert len(claims) == 1
    assert claims[0]["revision"] == 2
    assert str(claims[0]["detail_ref"]).startswith("detail_")
    assert claims[0]["claim"] == "Led retrieval evaluation"


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
    current = tools.invoke_atomic_tool(
        "search_career_memory",
        {"user_id": "u1", "query": "retrieval", "limit": 2},
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
    assert current.state == "career_memory_search_found"
    assert current.facts["returned"] == 2
    assert current.facts["total"] == 3
    assert all(
        item["detail_ref"].startswith("detail_")
        for item in current.payload["items"]
    )
