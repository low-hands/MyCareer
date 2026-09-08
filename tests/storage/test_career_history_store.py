import sqlite3

import pytest

from career_agent.agent.resume_analysis_contracts import (
    ExtractedCareerEvidence,
    ExtractedCareerRecord,
    ResumeAnalysisResult,
)
from career_agent.storage.career_history import (
    CareerEvidenceInvariantError,
    CareerHistoryStore,
)
from career_agent.storage.intent_versions import intent_content_digest
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


def test_tombstone_redacts_lineage_removes_indexes_and_blocks_rollback(
    tmp_path,
) -> None:
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
    original = store.confirm_evidence(
        user_id="u1",
        career_evidence_id=store.create_evidence(
            user_id="u1",
            career_record_id=record.id,
            claim="Built the confidential settlement platform.",
            origin="resume_extraction",
            source_resume_version_id=version.id,
            source_locator="experience:1",
            source_quote="Built the confidential settlement platform.",
        ).id,
    )
    correction = store.correct_evidence(
        user_id="u1",
        career_evidence_id=original.id,
        new_claim="Led the confidential settlement platform.",
        reason="Corrected ownership.",
    )

    tombstone = store.tombstone_evidence(
        user_id="u1",
        career_evidence_id=correction.current.id,
        reason="User requested permanent deletion.",
    )

    assert set(tombstone.evidence_ids) == {original.id, correction.current.id}
    assert store.get_evidence(
        user_id="u1", career_evidence_id=original.id
    ) is None
    assert store.get_evidence_by_source_ref(
        user_id="u1", source_ref=original.source_ref
    ) is None
    assert store.get_evidence_by_detail_ref(
        user_id="u1", detail_ref=correction.current.detail_ref
    ) is None
    assert store.list_evidence(user_id="u1", include_historical=True) == ()
    assert store.search_historical_evidence(
        user_id="u1", query="settlement"
    ) == ((), 0, None)
    with pytest.raises(ValueError, match="cannot be rolled back"):
        store.rollback_evidence_correction(
            user_id="u1",
            mutation_id=correction.snapshot.id,
            reason="Attempted restore.",
        )
    audit = store.list_evidence_tombstones(user_id="u1")
    assert len(audit) == 1
    assert audit[0].scope_key == correction.current.scope_key
    assert "confidential settlement" not in audit[0].model_dump_json()

    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            """
            SELECT id, claim, source_locator, source_quote, tombstoned_at,
                   suppression_digest
            FROM career_evidence
            WHERE scope_key = ?
            ORDER BY revision
            """,
            (correction.current.scope_key,),
        ).fetchall()
        assert len(rows) == 2
        assert all(not value for row in rows for value in row[1:4])
        assert all(row[4] and row[5].startswith("sha256:") for row in rows)
        assert connection.execute(
            """
            SELECT COUNT(*) FROM career_evidence_fts
            WHERE evidence_id IN (?, ?)
            """,
            (original.id, correction.current.id),
        ).fetchone()[0] == 0
        mutation_row = connection.execute(
            """
            SELECT status, tombstoned_at, preimage_json
            FROM career_evidence_mutations WHERE id = ?
            """,
            (correction.snapshot.id,),
        ).fetchone()
        assert mutation_row[:2] == ("tombstoned", rows[0][4])
        assert "confidential settlement" not in mutation_row[2]
        assert connection.execute(
            """
            SELECT COUNT(*) FROM career_evidence_suppressions
            WHERE career_evidence_id IN (?, ?)
            """,
            (original.id, correction.current.id),
        ).fetchone()[0] == 2
    assert store.detect_evidence_invariant_violations(user_id="u1").valid


def test_complete_cleanup_constructs_receipt_from_confirmed_tombstones(
    tmp_path,
) -> None:
    store, _, path = build_stores(tmp_path)
    record = create_record(store)
    confirmed = store.confirm_evidence(
        user_id="u1",
        career_evidence_id=store.create_evidence(
            user_id="u1",
            career_record_id=record.id,
            claim="Delete this claim.",
            origin="user_input",
        ).id,
    )
    pending = store.create_evidence(
        user_id="u1",
        career_record_id=record.id,
        claim="Unconfirmed row in the same raw scope.",
        origin="agent_inference",
    )
    tombstone = store.tombstone_evidence(
        user_id="u1",
        career_evidence_id=confirmed.id,
        reason="Delete it.",
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            UPDATE career_evidence
            SET scope_key = ?, update_id = ?, content_digest = ?, revision = 99,
                valid_from = datetime('now')
            WHERE id = ?
            """,
            (
                confirmed.scope_key,
                "career_evidence_update_" + "f" * 32,
                intent_content_digest(pending.claim),
                pending.id,
            ),
        )

    completed = store.complete_tombstone_cleanup(
        user_id="u1",
        cleanup_operation_id=tombstone.cleanup_operation_id,
    )

    assert completed.cleanup_status == "completed"
    assert completed.evidence_ids == (confirmed.id,)


def test_detector_rejects_each_tombstone_invariant_violation(tmp_path) -> None:
    store, _, path = build_stores(tmp_path)
    record = create_record(store)
    original = store.confirm_evidence(
        user_id="u1",
        career_evidence_id=store.create_evidence(
            user_id="u1",
            career_record_id=record.id,
            claim="Original private claim.",
            origin="user_input",
        ).id,
    )
    correction = store.correct_evidence(
        user_id="u1",
        career_evidence_id=original.id,
        new_claim="Corrected private claim.",
        reason="Correction.",
    )
    store.tombstone_evidence(
        user_id="u1",
        career_evidence_id=correction.current.id,
        reason="Delete.",
    )
    with sqlite3.connect(path) as connection:
        digest = connection.execute(
            "SELECT suppression_digest FROM career_evidence WHERE id = ?",
            (original.id,),
        ).fetchone()[0]

        connection.execute(
            "UPDATE career_evidence SET claim = 'leaked' WHERE id = ?",
            (original.id,),
        )
        connection.commit()
        assert "tombstone_content" in {
            item.code
            for item in store.detect_evidence_invariant_violations(
                user_id="u1"
            ).violations
        }
        connection.execute(
            "UPDATE career_evidence SET claim = '' WHERE id = ?",
            (original.id,),
        )
        connection.commit()

        connection.execute(
            """
            INSERT INTO career_evidence_fts(evidence_id, user_id, claim)
            VALUES (?, 'u1', 'leaked')
            """,
            (original.id,),
        )
        connection.commit()
        assert "tombstone_index" in {
            item.code
            for item in store.detect_evidence_invariant_violations(
                user_id="u1"
            ).violations
        }
        connection.execute(
            "DELETE FROM career_evidence_fts WHERE evidence_id = ?",
            (original.id,),
        )
        connection.commit()

        connection.execute(
            """
            DELETE FROM career_evidence_suppressions
            WHERE career_evidence_id = ?
            """,
            (original.id,),
        )
        connection.commit()
        assert "suppression_binding" in {
            item.code
            for item in store.detect_evidence_invariant_violations(
                user_id="u1"
            ).violations
        }
        connection.execute(
            """
            INSERT INTO career_evidence_suppressions(
                user_id, suppression_digest, scope_key,
                career_evidence_id, created_at
            ) VALUES ('u1', ?, ?, ?, '2026-09-07T00:00:00+00:00')
            """,
            (digest, original.scope_key, original.id),
        )
        connection.commit()

        connection.execute(
            """
            UPDATE career_evidence_mutations
            SET status = 'applied', tombstoned_at = NULL
            WHERE id = ?
            """,
            (correction.snapshot.id,),
        )
        connection.commit()
        assert "tombstone_rollback" in {
            item.code
            for item in store.detect_evidence_invariant_violations(
                user_id="u1"
            ).violations
        }


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
    assert confirmed.scope_key == f"career_evidence/{confirmed.id}/claim"
    assert confirmed.update_id is not None
    assert confirmed.content_digest == intent_content_digest(confirmed.claim)
    assert confirmed.revision == 1
    assert confirmed.valid_from == confirmed.updated_at
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


def test_resume_reextraction_skips_every_suppressed_source_triple(tmp_path) -> None:
    store, resumes, _ = build_stores(tmp_path)
    role = resumes.create_target_role(user_id="u1", title="PM", priority=1)
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
                record_type="internship",
                organization="Private Corp.",
                title="Product Intern",
                start_year=2022,
                source_locator="Experience heading",
                source_quote="Private Corp. Product Intern",
                evidence=(
                    ExtractedCareerEvidence(
                        claim="Built a confidential launch plan",
                        source_locator="Experience bullet 1",
                        source_quote="Built a confidential launch plan",
                    ),
                ),
            ),
        )
    )
    imported = store.import_confirmed_resume_analysis(
        user_id="u1",
        analysis_id="analysis-before-delete",
        resume_version_id=version.id,
        result=analysis,
    )
    for evidence in imported.evidence:
        store.tombstone_evidence(
            user_id="u1",
            career_evidence_id=evidence.id,
            reason="Delete imported internship.",
        )

    repeated_content = store.import_confirmed_resume_analysis(
        user_id="u1",
        analysis_id="analysis-after-delete",
        resume_version_id=version.id,
        result=analysis,
    )

    assert repeated_content.records == ()
    assert repeated_content.evidence == ()
    assert store.list_evidence(user_id="u1", include_historical=True) == ()


def test_correction_writes_bidirectional_lineage_snapshot_and_events(
    tmp_path,
) -> None:
    store, _, _ = build_stores(tmp_path)
    record = create_record(store)
    original = store.confirm_evidence(
        user_id="u1",
        career_evidence_id=store.create_evidence(
            user_id="u1",
            career_record_id=record.id,
            claim="Built a retrieval pipeline.",
            origin="user_input",
        ).id,
    )

    correction = store.correct_evidence(
        user_id="u1",
        career_evidence_id=original.id,
        new_claim="Led the retrieval evaluation pipeline.",
        reason="User corrected ownership and scope.",
    )

    assert correction.previous.superseded_by == correction.current.id
    assert correction.previous.superseded_at == correction.current.valid_from
    assert correction.current.supersedes_id == original.id
    assert correction.current.scope_key == original.scope_key
    assert correction.current.update_id != original.update_id
    assert correction.current.content_digest == intent_content_digest(
        correction.current.claim
    )
    assert correction.current.revision == 2
    assert correction.current.mutation_id == correction.snapshot.id
    assert correction.snapshot.preimage.active_evidence_id == original.id
    assert correction.snapshot.preimage.active_revision == 1
    assert store.list_evidence_mutations(
        user_id="u1", scope_key=original.scope_key
    ) == (correction.snapshot,)
    assert store.get_current_evidence(
        user_id="u1", scope_key=original.scope_key
    ) == correction.current
    assert store.list_evidence(
        user_id="u1", verification_status="confirmed"
    ) == (correction.current,)
    assert store.list_evidence(
        user_id="u1",
        verification_status="confirmed",
        include_historical=True,
    ) == (correction.previous, correction.current)
    assert [
        event.event_type
        for event in store.list_evidence_events(
            user_id="u1", career_evidence_id=original.id
        )
    ] == ["created", "confirmed", "superseded"]
    assert [
        event.event_type
        for event in store.list_evidence_events(
            user_id="u1", career_evidence_id=correction.current.id
        )
    ] == ["corrected"]
    assert store.detect_evidence_invariant_violations(user_id="u1").valid


def test_correction_admission_requires_current_confirmed_source_and_change(
    tmp_path,
) -> None:
    store, _, _ = build_stores(tmp_path)
    record = create_record(store)
    pending = store.create_evidence(
        user_id="u1",
        career_record_id=record.id,
        claim="Built a retrieval pipeline.",
        origin="user_input",
    )
    with pytest.raises(ValueError, match="current confirmed"):
        store.correct_evidence(
            user_id="u1",
            career_evidence_id=pending.id,
            new_claim="Led a retrieval pipeline.",
            reason="Correction",
        )
    confirmed = store.confirm_evidence(
        user_id="u1", career_evidence_id=pending.id
    )
    with pytest.raises(ValueError, match="unchanged"):
        store.correct_evidence(
            user_id="u1",
            career_evidence_id=confirmed.id,
            new_claim="Built\u3000a retrieval pipeline.",
            reason="Whitespace only",
        )
    with pytest.raises(ValueError, match="new claim and reason"):
        store.correct_evidence(
            user_id="u1",
            career_evidence_id=confirmed.id,
            new_claim="Led a retrieval pipeline.",
            reason=" ",
        )


def test_snapshot_rollback_restores_preimage_and_preserves_history(
    tmp_path,
) -> None:
    store, _, _ = build_stores(tmp_path)
    record = create_record(store)
    original = store.confirm_evidence(
        user_id="u1",
        career_evidence_id=store.create_evidence(
            user_id="u1",
            career_record_id=record.id,
            claim="Built version one.",
            origin="user_input",
        ).id,
    )
    second = store.correct_evidence(
        user_id="u1",
        career_evidence_id=original.id,
        new_claim="Built version two.",
        reason="First correction",
    )
    third = store.correct_evidence(
        user_id="u1",
        career_evidence_id=second.current.id,
        new_claim="Built version three.",
        reason="Second correction",
    )

    rolled_back = store.rollback_evidence_correction(
        user_id="u1",
        mutation_id=third.snapshot.id,
        reason="Detector requested compensation.",
    )
    repeated = store.rollback_evidence_correction(
        user_id="u1",
        mutation_id=third.snapshot.id,
        reason="Replay",
    )

    assert rolled_back.status == "rolled_back"
    assert repeated == rolled_back
    assert store.get_current_evidence(
        user_id="u1", scope_key=original.scope_key
    ).id == second.current.id
    history = store.list_evidence(
        user_id="u1",
        verification_status="confirmed",
        include_historical=True,
    )
    assert [item.revision for item in history] == [1, 2, 3]
    assert history[2].rolled_back_at is not None
    assert store.detect_evidence_invariant_violations(user_id="u1").valid

    fourth = store.correct_evidence(
        user_id="u1",
        career_evidence_id=second.current.id,
        new_claim="Built version four after rollback.",
        reason="Replacement correction",
    )
    assert fourth.current.revision == 4
    assert store.detect_evidence_invariant_violations(user_id="u1").valid


def test_detector_finds_corrupt_pointer_and_snapshot_rollback_repairs_it(
    tmp_path,
) -> None:
    store, _, path = build_stores(tmp_path)
    record = create_record(store)
    original = store.confirm_evidence(
        user_id="u1",
        career_evidence_id=store.create_evidence(
            user_id="u1",
            career_record_id=record.id,
            claim="Original claim.",
            origin="user_input",
        ).id,
    )
    correction = store.correct_evidence(
        user_id="u1",
        career_evidence_id=original.id,
        new_claim="Corrected claim.",
        reason="User correction",
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            UPDATE career_evidence
            SET superseded_at = datetime('now'), superseded_by = 'missing-evidence'
            WHERE id = ?
            """,
            (correction.current.id,),
        )

    report = store.detect_evidence_invariant_violations(user_id="u1")

    assert not report.valid
    assert {"pointer_target_missing", "active_count"} <= {
        item.code for item in report.violations
    }
    store.rollback_evidence_correction(
        user_id="u1",
        mutation_id=correction.snapshot.id,
        reason="Repair corrupt active mapping.",
    )
    assert store.detect_evidence_invariant_violations(user_id="u1").valid


def test_detector_finds_missing_lineage_event(tmp_path) -> None:
    store, _, path = build_stores(tmp_path)
    record = create_record(store)
    original = store.confirm_evidence(
        user_id="u1",
        career_evidence_id=store.create_evidence(
            user_id="u1",
            career_record_id=record.id,
            claim="Original claim.",
            origin="user_input",
        ).id,
    )
    correction = store.correct_evidence(
        user_id="u1",
        career_evidence_id=original.id,
        new_claim="Corrected claim.",
        reason="User correction",
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            DELETE FROM career_evidence_events
            WHERE event_type = 'superseded' AND mutation_id = ?
            """,
            (correction.snapshot.id,),
        )

    report = store.detect_evidence_invariant_violations(user_id="u1")

    assert "event_replay" in {item.code for item in report.violations}
    with pytest.raises(CareerEvidenceInvariantError) as error:
        CareerHistoryStore(path)
    assert "event_replay" in {
        item.code for item in error.value.report.violations
    }

    maintenance = CareerHistoryStore(path, validate_invariants=False)
    maintenance.rollback_evidence_correction(
        user_id="u1",
        mutation_id=correction.snapshot.id,
        reason="Startup detector requested compensation.",
    )
    assert CareerHistoryStore(path).detect_evidence_invariant_violations().valid


def test_v4_migration_backfills_only_confirmed_evidence_versions(
    tmp_path,
) -> None:
    store, _, path = build_stores(tmp_path)
    record = create_record(store)
    pending = store.create_evidence(
        user_id="u1",
        career_record_id=record.id,
        claim="Pending claim.",
        origin="user_input",
    )
    confirmed = store.confirm_evidence(
        user_id="u1",
        career_evidence_id=store.create_evidence(
            user_id="u1",
            career_record_id=record.id,
            claim="Confirmed claim.",
            origin="user_input",
        ).id,
    )
    with sqlite3.connect(path) as connection:
        connection.execute("DROP INDEX career_evidence_active_scope_idx")
        connection.execute("DROP INDEX career_evidence_scope_revision_idx")
        connection.execute(
            """
            UPDATE career_evidence
            SET scope_key = NULL, update_id = NULL, content_digest = NULL,
                revision = NULL, valid_from = NULL
            WHERE id = ?
            """,
            (confirmed.id,),
        )
        connection.execute(
            "UPDATE schema_versions SET version = 3 "
            "WHERE component = 'career_history'"
        )

    migrated = CareerHistoryStore(path)
    migrated_confirmed = migrated.get_evidence(
        user_id="u1", career_evidence_id=confirmed.id
    )
    migrated_pending = migrated.get_evidence(
        user_id="u1", career_evidence_id=pending.id
    )

    assert migrated_confirmed.revision == 1
    assert migrated_confirmed.scope_key == (
        f"career_evidence/{confirmed.id}/claim"
    )
    assert migrated_confirmed.valid_from is not None
    assert migrated_pending.revision is None
    assert migrated_pending.scope_key is None
    assert migrated.detect_evidence_invariant_violations(user_id="u1").valid


def test_v4_migration_rebuilds_the_event_log_for_lineage_events(
    tmp_path,
) -> None:
    store, _, path = build_stores(tmp_path)
    record = create_record(store)
    evidence = store.confirm_evidence(
        user_id="u1",
        career_evidence_id=store.create_evidence(
            user_id="u1",
            career_record_id=record.id,
            claim="Confirmed claim.",
            origin="user_input",
        ).id,
    )
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute("DROP INDEX career_evidence_events_evidence_idx")
        connection.execute(
            "ALTER TABLE career_evidence_events "
            "RENAME TO career_evidence_events_v4"
        )
        connection.execute(
            """
            CREATE TABLE career_evidence_events (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                career_evidence_id TEXT NOT NULL REFERENCES career_evidence(id),
                event_type TEXT NOT NULL CHECK (
                    event_type IN ('created', 'confirmed', 'rejected')
                ),
                previous_status TEXT,
                new_status TEXT NOT NULL,
                actor_type TEXT NOT NULL,
                reason TEXT,
                occurred_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO career_evidence_events
            SELECT id, user_id, career_evidence_id, event_type,
                   previous_status, new_status, actor_type, reason, occurred_at
            FROM career_evidence_events_v4
            """
        )
        connection.execute("DROP TABLE career_evidence_events_v4")
        connection.execute(
            """
            CREATE INDEX career_evidence_events_evidence_idx
            ON career_evidence_events(career_evidence_id, occurred_at)
            """
        )
        connection.execute(
            "UPDATE schema_versions SET version = 3 "
            "WHERE component = 'career_history'"
        )

    migrated = CareerHistoryStore(path)
    correction = migrated.correct_evidence(
        user_id="u1",
        career_evidence_id=evidence.id,
        new_claim="Corrected claim.",
        reason="Migration accepted lineage event",
    )

    assert correction.current.revision == 2
    assert migrated.detect_evidence_invariant_violations(user_id="u1").valid


def test_v5_migration_backfills_detail_refs_and_history_index(tmp_path) -> None:
    store, _, path = build_stores(tmp_path)
    record = create_record(store)
    original = store.confirm_evidence(
        user_id="u1",
        career_evidence_id=store.create_evidence(
            user_id="u1",
            career_record_id=record.id,
            claim="Assisted with retrieval evaluation",
            origin="user_input",
        ).id,
    )
    correction = store.correct_evidence(
        user_id="u1",
        career_evidence_id=original.id,
        new_claim="Led retrieval evaluation",
        reason="User corrected ownership",
    )
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            DROP TRIGGER career_evidence_fts_insert;
            DROP TRIGGER career_evidence_fts_delete;
            DROP TRIGGER career_evidence_fts_update;
            DROP TABLE career_evidence_fts;
            DROP INDEX career_evidence_detail_ref_unique_idx;
            UPDATE career_evidence SET detail_ref = NULL;
            UPDATE schema_versions SET version = 4
            WHERE component = 'career_history';
            """
        )

    migrated = CareerHistoryStore(path)
    current = migrated.get_evidence(
        user_id="u1",
        career_evidence_id=correction.current.id,
    )
    history, total, cursor = migrated.search_historical_evidence(
        user_id="u1",
        query="retrieval",
    )

    assert current is not None
    assert current.detail_ref.startswith("detail_")
    assert migrated.get_evidence_by_detail_ref(
        user_id="u1",
        detail_ref=current.detail_ref,
    ) == current
    assert [item.id for item in history] == [original.id]
    assert total == 1
    assert cursor is None


def test_historical_search_is_indexed_bounded_and_cursor_paginated(
    tmp_path,
) -> None:
    store, _, path = build_stores(tmp_path)
    record = create_record(store)
    originals = []
    for index in range(3):
        original = store.confirm_evidence(
            user_id="u1",
            career_evidence_id=store.create_evidence(
                user_id="u1",
                career_record_id=record.id,
                claim=f"Assisted with retrieval evaluation {index}",
                origin="user_input",
            ).id,
        )
        originals.append(original)
        store.correct_evidence(
            user_id="u1",
            career_evidence_id=original.id,
            new_claim=f"Led retrieval evaluation {index}",
            reason="User corrected ownership",
        )

    first, total, cursor = store.search_historical_evidence(
        user_id="u1",
        query="retrieval",
        limit=2,
    )
    assert len(first) == 2
    assert total == 3
    assert cursor is not None

    second, second_total, next_cursor = store.search_historical_evidence(
        user_id="u1",
        query="retrieval",
        limit=2,
        cursor=cursor,
    )
    assert len(second) == 1
    assert second_total == total
    assert next_cursor is None
    assert {item.id for item in (*first, *second)} == {
        item.id for item in originals
    }
    current, current_total, current_cursor = store.search_current_evidence(
        user_id="u1",
        query="retrieval",
        limit=2,
    )
    assert len(current) == 2
    assert current_total == 3
    assert current_cursor is not None
    current_rest, _, final_cursor = store.search_current_evidence(
        user_id="u1",
        query="retrieval",
        limit=2,
        cursor=current_cursor,
    )
    assert len(current_rest) == 1
    assert final_cursor is None
    assert all(item.is_current for item in (*current, *current_rest))
    with pytest.raises(ValueError, match="does not match this query"):
        store.search_historical_evidence(
            user_id="u1",
            query="different",
            cursor=cursor,
        )

    with sqlite3.connect(path) as connection:
        plan = [
            str(row[3])
            for row in connection.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT evidence.id
                FROM career_evidence_fts AS search
                JOIN career_evidence AS evidence
                  ON evidence.id = search.evidence_id
                WHERE search.user_id = ?
                  AND career_evidence_fts MATCH ?
                  AND evidence.verification_status = 'confirmed'
                  AND (
                      evidence.superseded_by IS NOT NULL
                      OR evidence.rolled_back_at IS NOT NULL
                  )
                ORDER BY bm25(career_evidence_fts), evidence.created_at DESC
                LIMIT ? OFFSET ?
                """,
                ("u1", '"retrieval"', 2, 0),
            )
        ]
    assert any("VIRTUAL TABLE INDEX" in step for step in plan)
    assert not any("SCAN evidence" in step for step in plan)
