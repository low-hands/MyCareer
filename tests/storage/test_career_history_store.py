import sqlite3

import pytest

from career_agent.agent.resume_analysis_contracts import (
    ExtractedCareerEvidence,
    ExtractedCareerRecord,
    ResumeAnalysisResult,
)
from career_agent.storage.career_history import CareerHistoryStore
from career_agent.storage.resumes import ResumeStore


def build_stores(tmp_path):
    path = tmp_path / "career.sqlite3"
    resumes = ResumeStore(path)
    history = CareerHistoryStore(path)
    return history, resumes, path


def create_record(store: CareerHistoryStore, *, user_id: str = "u1"):
    return store.create_record(
        user_id=user_id,
        record_type="work",
        organization="Acme",
        title="AI Engineer",
        start_year=2023,
        start_month=7,
        is_current=True,
    )


def test_store_supports_manual_evidence_without_resume_tables(tmp_path) -> None:
    store = CareerHistoryStore(tmp_path / "career-only.sqlite3")
    record = create_record(store)

    evidence = store.create_evidence(
        user_id="u1",
        career_record_id=record.id,
        claim="Led product discovery.",
        origin="user_input",
    )

    assert store.get_evidence(user_id="u1", career_evidence_id=evidence.id) == evidence


def test_records_persist_and_are_user_scoped(tmp_path) -> None:
    store, _, path = build_stores(tmp_path)
    record = create_record(store)

    rebuilt = CareerHistoryStore(path)

    assert rebuilt.get_record(user_id="u1", career_record_id=record.id) == record
    assert rebuilt.list_records(user_id="u1") == (record,)
    assert rebuilt.get_record(user_id="u2", career_record_id=record.id) is None
    assert rebuilt.list_records(user_id="u2") == ()


def test_evidence_creation_is_scoped_and_emits_created_event(tmp_path) -> None:
    store, _, _ = build_stores(tmp_path)
    record = create_record(store)

    evidence = store.create_evidence(
        user_id="u1",
        career_record_id=record.id,
        claim="Improved answer accuracy from 62% to 81%.",
        origin="user_input",
    )

    assert evidence.verification_status == "pending"
    assert store.get_evidence(user_id="u1", career_evidence_id=evidence.id) == evidence
    assert store.get_evidence(user_id="u2", career_evidence_id=evidence.id) is None
    assert store.list_evidence(user_id="u1", career_record_id=record.id) == (evidence,)
    event = store.list_evidence_events(
        user_id="u1", career_evidence_id=evidence.id
    )[0]
    assert event.event_type == "created"
    assert event.actor_type == "user"

    with pytest.raises(ValueError, match="Career record not found"):
        store.create_evidence(
            user_id="u2",
            career_record_id=record.id,
            claim="Foreign claim",
            origin="user_input",
        )


def test_resume_extraction_requires_owned_resume_version(tmp_path) -> None:
    store, resumes, _ = build_stores(tmp_path)
    record = create_record(store)
    role = resumes.create_target_role(user_id="u1", title="AI Engineer", priority=1)
    _, version = resumes.import_document(
        user_id="u1",
        target_role_id=role.id,
        name="Base",
        content=b"Resume body",
        document_format="text",
    )
    other_role = resumes.create_target_role(
        user_id="u2", title="Product Manager", priority=1
    )
    _, other_version = resumes.import_document(
        user_id="u2",
        target_role_id=other_role.id,
        name="Other",
        content=b"Other user's resume",
        document_format="text",
    )

    evidence = store.create_evidence(
        user_id="u1",
        career_record_id=record.id,
        claim="Built an AI product.",
        origin="resume_extraction",
        source_resume_version_id=version.id,
        source_locator="line=1",
        source_quote="Built an AI product.",
    )

    assert evidence.source_resume_version_id == version.id
    assert evidence.source_ref is not None
    assert store.get_evidence_by_source_ref(
        user_id="u1", source_ref=evidence.source_ref
    ) is None
    confirmed = store.confirm_evidence(
        user_id="u1", career_evidence_id=evidence.id
    )
    assert store.get_evidence_by_source_ref(
        user_id="u1", source_ref=evidence.source_ref
    ) == confirmed
    assert store.get_evidence_by_source_ref(
        user_id="u2", source_ref=evidence.source_ref
    ) is None
    with pytest.raises(ValueError, match="Source resume version not found"):
        store.create_evidence(
            user_id="u1",
            career_record_id=record.id,
            claim="Unowned source",
            origin="resume_extraction",
            source_resume_version_id="missing-version",
            source_locator="line=1",
            source_quote="Unowned source",
        )
    with pytest.raises(ValueError, match="Source resume version not found"):
        store.create_evidence(
            user_id="u1",
            career_record_id=record.id,
            claim="Cross-user source",
            origin="resume_extraction",
            source_resume_version_id=other_version.id,
            source_locator="line=1",
            source_quote="Cross-user source",
        )


def test_v3_migration_backfills_stable_source_refs(tmp_path) -> None:
    store, resumes, path = build_stores(tmp_path)
    record = create_record(store)
    role = resumes.create_target_role(user_id="u1", title="AI Engineer", priority=1)
    _, version = resumes.import_document(
        user_id="u1",
        target_role_id=role.id,
        name="Base",
        content=b"Resume body",
        document_format="text",
    )
    evidence = store.create_evidence(
        user_id="u1",
        career_record_id=record.id,
        claim="Built an AI product.",
        origin="resume_extraction",
        source_resume_version_id=version.id,
        source_locator="line=1",
        source_quote="Built an AI product.",
    )
    with sqlite3.connect(path) as connection:
        connection.execute("DROP INDEX career_evidence_source_ref_unique_idx")
        connection.execute("ALTER TABLE career_evidence DROP COLUMN source_ref")
        connection.execute(
            "UPDATE schema_versions SET version = 2 WHERE component = 'career_history'"
        )

    migrated = CareerHistoryStore(path)
    reread = migrated.get_evidence(
        user_id="u1", career_evidence_id=evidence.id
    )

    assert reread is not None
    assert reread.source_ref == evidence.source_ref


def test_confirm_is_atomic_audited_and_idempotent(tmp_path) -> None:
    store, _, _ = build_stores(tmp_path)
    record = create_record(store)
    evidence = store.create_evidence(
        user_id="u1",
        career_record_id=record.id,
        claim="Built a RAG evaluation pipeline.",
        origin="agent_inference",
    )

    confirmed = store.confirm_evidence(
        user_id="u1",
        career_evidence_id=evidence.id,
        reason="User verified the project result.",
    )
    repeated = store.confirm_evidence(
        user_id="u1", career_evidence_id=evidence.id
    )

    assert confirmed.verification_status == "confirmed"
    assert repeated == confirmed
    events = store.list_evidence_events(
        user_id="u1", career_evidence_id=evidence.id
    )
    assert [event.event_type for event in events] == ["created", "confirmed"]
    assert events[-1].reason == "User verified the project result."
    assert store.list_evidence(
        user_id="u1", verification_status="confirmed"
    ) == (confirmed,)

    with pytest.raises(ValueError, match="Cannot change confirmed evidence"):
        store.reject_evidence(user_id="u1", career_evidence_id=evidence.id)


def test_reject_is_audited_and_cannot_be_confirmed(tmp_path) -> None:
    store, _, _ = build_stores(tmp_path)
    record = create_record(store)
    evidence = store.create_evidence(
        user_id="u1",
        career_record_id=record.id,
        claim="An incorrect extracted fact.",
        origin="user_input",
    )

    rejected = store.reject_evidence(
        user_id="u1", career_evidence_id=evidence.id
    )

    assert rejected.verification_status == "rejected"
    assert [
        event.event_type
        for event in store.list_evidence_events(
            user_id="u1", career_evidence_id=evidence.id
        )
    ] == ["created", "rejected"]
    with pytest.raises(ValueError, match="Cannot change rejected evidence"):
        store.confirm_evidence(user_id="u1", career_evidence_id=evidence.id)


def test_evidence_commands_reject_cross_user_access(tmp_path) -> None:
    store, _, _ = build_stores(tmp_path)
    record = create_record(store)
    evidence = store.create_evidence(
        user_id="u1",
        career_record_id=record.id,
        claim="Private fact",
        origin="user_input",
    )

    with pytest.raises(ValueError, match="Career evidence not found"):
        store.confirm_evidence(user_id="u2", career_evidence_id=evidence.id)
    assert store.list_evidence_events(
        user_id="u2", career_evidence_id=evidence.id
    ) == ()


def test_confirmed_resume_analysis_import_is_atomic_audited_and_idempotent(tmp_path) -> None:
    store, resumes, _ = build_stores(tmp_path)
    role = resumes.create_target_role(user_id="u1", title="Product Manager", priority=1)
    _, version = resumes.import_document(
        user_id="u1",
        target_role_id=role.id,
        name="PM Resume",
        content=b"resume",
        document_format="text",
    )
    analysis = ResumeAnalysisResult(
        records=(
            ExtractedCareerRecord(
                record_type="work",
                organization="Example Inc.",
                title="Product Manager",
                start_year=2022,
                is_current=True,
                source_locator="Experience heading",
                source_quote="Example Inc. Product Manager 2022-Present",
                evidence=(
                    ExtractedCareerEvidence(
                        claim="Led knowledge-base planning",
                        source_locator="Experience bullet 1",
                        source_quote="Led knowledge-base planning",
                    ),
                ),
            ),
        )
    )

    imported = store.import_confirmed_resume_analysis(
        user_id="u1",
        analysis_id="analysis-1",
        resume_version_id=version.id,
        result=analysis,
    )
    repeated = store.import_confirmed_resume_analysis(
        user_id="u1",
        analysis_id="analysis-1",
        resume_version_id=version.id,
        result=analysis,
    )

    assert repeated == imported
    assert len(imported.records) == 1
    assert len(imported.evidence) == 2
    assert store.list_records(user_id="u1") == imported.records
    assert {item.id for item in store.list_evidence(user_id="u1")} == {
        item.id for item in imported.evidence
    }
    assert all(item.verification_status == "confirmed" for item in imported.evidence)
    assert imported.evidence[1].source_quote == "Led knowledge-base planning"
    for evidence in imported.evidence:
        assert [
            event.event_type
            for event in store.list_evidence_events(
                user_id="u1", career_evidence_id=evidence.id
            )
        ] == ["created", "confirmed"]
