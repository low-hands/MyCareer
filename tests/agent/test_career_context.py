from __future__ import annotations

from career_agent.agent.career_context import CareerContextProjector
from career_agent.agent.main_agent_contracts import (
    CareerProfileContext,
    MainAgentContext,
)
from career_agent.storage.career_history import CareerHistoryStore


def _confirmed_highlight(
    store: CareerHistoryStore,
    *,
    user_id: str,
    career_record_id: str,
    claim: str,
) -> None:
    evidence = store.create_evidence(
        user_id=user_id,
        career_record_id=career_record_id,
        claim=claim,
        origin="user_input",
    )
    store.confirm_evidence(user_id=user_id, career_evidence_id=evidence.id)


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
    assert memory.records[0].confirmed_highlights == (
        "Built a RAG evaluation pipeline",
    )
    serialized = memory.model_dump_json()
    assert "PRIVATE UNCONFIRMED CLAIM" not in serialized
    assert "career_record_" not in serialized
    assert "career_evidence_" not in serialized


def test_main_agent_model_context_contains_compact_memory_without_provenance(tmp_path) -> None:
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
    assert "source_quote" not in str(projected)
    assert "source_locator" not in str(projected)
    assert "user_id" not in str(projected)
