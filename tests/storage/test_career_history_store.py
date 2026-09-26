import sqlite3

import pytest

from career_agent.storage.career_history import CareerHistoryStore
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


def test_tombstone_redacts_the_complete_linked_lineage_and_indexes(
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
    audit = store.list_evidence_tombstones(user_id="u1")
    assert len(audit) == 1
    assert audit[0].scope_key == correction.current.scope_key
    assert "confidential settlement" not in audit[0].model_dump_json()

    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            """
            SELECT id, claim, source_locator, source_quote, tombstoned_at
            FROM career_evidence
            WHERE scope_key = ?
            ORDER BY revision
            """,
            (correction.current.scope_key,),
        ).fetchall()
        assert len(rows) == 2
        assert all(not value for row in rows for value in row[1:4])
        assert all(row[4] for row in rows)
        assert connection.execute(
            """
            SELECT COUNT(*) FROM career_evidence_fts
            WHERE evidence_id IN (?, ?)
            """,
            (original.id, correction.current.id),
        ).fetchone()[0] == 0
        removed_tables = {
            row[0]
            for row in connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'table' AND name IN (
                    'career_evidence_mutations',
                    'career_evidence_suppressions',
                    'career_memory_deletion_operations'
                )
                """
            )
        }
        assert removed_tables == set()


def test_tombstone_digest_rejects_a_stale_delete_precondition(tmp_path) -> None:
    store, _, _ = build_stores(tmp_path)
    record = create_record(store)
    evidence = store.confirm_evidence(
        user_id="u1",
        career_evidence_id=store.create_evidence(
            user_id="u1",
            career_record_id=record.id,
            claim="Keep this current claim.",
            origin="user_input",
        ).id,
    )

    with pytest.raises(ValueError, match="changed after deletion was proposed"):
        store.tombstone_evidence(
            user_id="u1",
            career_evidence_id=evidence.id,
            reason="Delete.",
            expected_content_sha256="sha256:" + "0" * 64,
        )

    assert store.get_evidence(
        user_id="u1", career_evidence_id=evidence.id
    ) == evidence


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


def test_existing_history_adds_user_interaction_provenance_columns(tmp_path) -> None:
    store, _, path = build_stores(tmp_path)
    record = create_record(store)
    evidence = store.create_evidence(
        user_id="u1", career_record_id=record.id,
        claim="Built an internal tool.", origin="agent_inference",
    )
    with sqlite3.connect(path) as connection:
        connection.execute("ALTER TABLE career_evidence DROP COLUMN source_user_quote")
        connection.execute("ALTER TABLE career_evidence DROP COLUMN source_user_interaction_id")

    migrated = CareerHistoryStore(path)
    reread = migrated.get_evidence(user_id="u1", career_evidence_id=evidence.id)
    assert reread is not None
    assert reread.claim == evidence.claim
    assert reread.source_user_quote is None
    assert reread.source_user_interaction_id is None
    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(career_evidence)")}
    assert {"source_user_quote", "source_user_interaction_id"} <= columns


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


def test_correction_appends_a_linked_revision_and_events(
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


def test_v8_migration_backfills_the_short_term_fts_column(tmp_path) -> None:
    store, _, path = build_stores(tmp_path)
    record = create_record(store)
    evidence = store.confirm_evidence(
        user_id="u1",
        career_evidence_id=store.create_evidence(
            user_id="u1",
            career_record_id=record.id,
            claim="C# 开发",
            origin="user_input",
        ).id,
    )
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            DROP TRIGGER career_evidence_fts_insert;
            DROP TRIGGER career_evidence_fts_delete;
            DROP TRIGGER career_evidence_fts_update;
            DROP TABLE career_evidence_fts;
            DROP TABLE career_evidence_short_fts;
            ALTER TABLE career_evidence DROP COLUMN short_terms;
            CREATE VIRTUAL TABLE career_evidence_fts USING fts5(
                evidence_id UNINDEXED,
                user_id UNINDEXED,
                claim,
                tokenize='trigram'
            );
            INSERT INTO career_evidence_fts(evidence_id, user_id, claim)
            SELECT id, user_id, claim FROM career_evidence
            WHERE tombstoned_at IS NULL;
            UPDATE schema_versions SET version = 7
            WHERE component = 'career_history';
            """
        )

    migrated = CareerHistoryStore(path)
    query_terms = migrated.current_evidence_query_terms(
        user_id="u1",
        query="C# 开发",
    )
    ranked = migrated.rank_current_evidence(
        user_id="u1",
        query_terms=query_terms,
    )

    assert query_terms.latin == ("c#",)
    assert query_terms.cjk == ("开发",)
    assert [hit.id for hit in ranked] == [evidence.id]
    with sqlite3.connect(path) as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(career_evidence)")
        }
        claim_fts_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'career_evidence_fts'"
        ).fetchone()[0]
        short_fts_sql = connection.execute(
            """
            SELECT sql FROM sqlite_master
            WHERE name = 'career_evidence_short_fts'
            """
        ).fetchone()[0]
    assert "short_terms" in columns
    assert "short_terms" not in claim_fts_sql
    assert "short_terms" in short_fts_sql


def test_short_term_index_tracks_corrections_and_tombstones(tmp_path) -> None:
    store = CareerHistoryStore(tmp_path / "career.sqlite3")
    record = create_record(store)
    original = store.confirm_evidence(
        user_id="u1",
        career_evidence_id=store.create_evidence(
            user_id="u1",
            career_record_id=record.id,
            claim="C# 开发",
            origin="user_input",
        ).id,
    )
    correction = store.correct_evidence(
        user_id="u1",
        career_evidence_id=original.id,
        new_claim="Go 后端",
        reason="Corrected technology",
    )

    assert store.rank_current_evidence(user_id="u1", query="C#") == ()
    assert [
        hit.id
        for hit in store.rank_current_evidence(user_id="u1", query="Go")
    ] == [correction.current.id]

    store.tombstone_evidence(
        user_id="u1",
        career_evidence_id=correction.current.id,
        reason="Remove the corrected claim",
    )

    assert store.rank_current_evidence(user_id="u1", query="Go") == ()
    with sqlite3.connect(store.path) as connection:
        for table in ("career_evidence_fts", "career_evidence_short_fts"):
            assert connection.execute(
                f"SELECT COUNT(*) FROM {table} WHERE evidence_id IN (?, ?)",
                (original.id, correction.current.id),
            ).fetchone()[0] == 0


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
                  AND evidence.superseded_by IS NOT NULL
                ORDER BY bm25(career_evidence_fts), evidence.created_at DESC
                LIMIT ? OFFSET ?
                """,
                ("u1", '"retrieval"', 2, 0),
            )
        ]
    assert any("VIRTUAL TABLE INDEX" in step for step in plan)
    assert not any("SCAN evidence" in step for step in plan)
