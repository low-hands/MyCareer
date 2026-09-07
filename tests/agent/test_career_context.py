from __future__ import annotations

from career_agent.agent.career_context import CareerContextProjector
from career_agent.agent.main_agent_contracts import (
    CareerProfileContext,
    MainAgentContext,
    ToolObservation,
)
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.storage.resumes import ResumeStore
from career_agent.storage.career_history import CareerHistoryStore


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

    memory = CareerContextProjector(store, max_records=1).project(
        user_id="u1",
        query="帮我匹配一个 RAG 岗位",
    )

    assert len(memory.records) == 1
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

    assert projected["records"][0]["title"] == "Product Manager"
    highlight = projected["records"][0]["confirmed_highlights"][0]
    assert highlight["claim"] == "Led knowledge-base planning"
    assert highlight["origin"] == "user_input"
    assert "recorded_at" in highlight
    assert highlight["source_ref"] is None
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
