from datetime import datetime, timedelta, timezone
import sqlite3


from career_agent.services.intent_capture import IntentCaptureCandidate
from career_agent.storage.intent_versions import (
    append_intent_version,
    apply_intent_version_schema,
    capture_intent_version,
    list_intent_versions,
)


NOW = datetime(2026, 9, 8, tzinfo=timezone.utc)
SCOPE = "person_intent/self/default_city"


def connection() -> sqlite3.Connection:
    result = sqlite3.connect(":memory:")
    apply_intent_version_schema(result)
    return result


def test_independent_named_scopes_each_start_at_revision_one() -> None:
    with connection() as database:
        global_value = append_intent_version(
            database,
            user_id="u1",
            scope_key=SCOPE,
            pref_scope="global",
            value="上海",
            source="test",
            valid_from=NOW,
        )
        interview_value = append_intent_version(
            database,
            user_id="u1",
            scope_key=SCOPE,
            pref_scope="interview",
            value="北京",
            source="test",
            valid_from=NOW,
        )

        assert global_value.revision == interview_value.revision == 1
        assert {
            (item.pref_scope, item.value)
            for item in list_intent_versions(
                database,
                user_id="u1",
                active_only=True,
            )
        } == {("global", "上海"), ("interview", "北京")}


def test_flat_v1_schema_rebuilds_with_scoped_uniqueness_and_backfill() -> None:
    database = sqlite3.connect(":memory:")
    database.executescript(
        """
        CREATE TABLE career_intent_versions (
            update_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            scope_key TEXT NOT NULL,
            value TEXT NOT NULL,
            content_digest TEXT NOT NULL,
            revision INTEGER NOT NULL,
            valid_from TEXT NOT NULL,
            superseded_at TEXT,
            superseded_by TEXT,
            source TEXT NOT NULL,
            UNIQUE(user_id, scope_key, revision)
        );
        INSERT INTO career_intent_versions
        VALUES (
            'intent_update_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
            'u1',
            'person_intent/self/default_city',
            '上海',
            'sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
            1,
            '2026-01-01T00:00:00+00:00',
            NULL,
            NULL,
            'legacy'
        );
        """
    )

    apply_intent_version_schema(database)
    scoped = append_intent_version(
        database,
        user_id="u1",
        scope_key=SCOPE,
        pref_scope="interview",
        value="北京",
        source="test",
        valid_from=NOW,
    )
    migrated = list_intent_versions(
        database,
        user_id="u1",
        pref_scope="global",
    )[0]

    assert scoped.revision == 1
    assert migrated.pref_scope == "global"
    assert migrated.last_corroborated_at == migrated.valid_from
    assert migrated.capture_action == "retain"


def test_unambiguous_revision_supersedes_a_stable_value_in_one_write() -> None:
    with connection() as database:
        append_intent_version(
            database,
            user_id="u1",
            scope_key=SCOPE,
            value="上海",
            source="test",
            valid_from=NOW,
            layer="stable",
        )

        decision, revised = capture_intent_version(
            database,
            candidate=IntentCaptureCandidate(
                user_id="u1",
                scope_key=SCOPE,
                value="杭州",
                source="user",
                observed_at=NOW + timedelta(days=1),
            ),
        )

        assert decision.reason == "ordinary_revision"
        assert revised is not None and revised.admission_status == "active"
        assert [
            item.value
            for item in list_intent_versions(
                database,
                user_id="u1",
                active_only=True,
            )
        ] == ["杭州"]


def test_abstention_does_not_write() -> None:
    with connection() as database:
        abstained, version = capture_intent_version(
            database,
            candidate=IntentCaptureCandidate(
                user_id="u1",
                scope_key=SCOPE,
                value="聊天内容但不是偏好",
                source="extractor",
                contains_preference_signal=False,
            ),
        )

        assert abstained.reason == "abstained" and version is None
        assert list_intent_versions(database, user_id="u1") == ()


def test_admitting_a_held_value_closes_its_quarantine_row() -> None:
    with connection() as database:
        _, held = capture_intent_version(
            database,
            candidate=IntentCaptureCandidate(
                user_id="u1",
                scope_key=SCOPE,
                value="杭州",
                source="extractor",
                ambiguous=True,
                observed_at=NOW,
            ),
        )
        decision, admitted = capture_intent_version(
            database,
            candidate=IntentCaptureCandidate(
                user_id="u1",
                scope_key=SCOPE,
                value="杭州",
                source="user",
                observed_at=NOW + timedelta(days=1),
            ),
        )

        assert held is not None and admitted is not None
        assert decision.action == "add"
        closed = next(
            item
            for item in list_intent_versions(database, user_id="u1")
            if item.update_id == held.update_id
        )
        assert closed.superseded_by == admitted.update_id


def test_a_quarantined_value_is_held_out_of_the_active_track() -> None:
    with connection() as database:
        _, quarantined = capture_intent_version(
            database,
            candidate=IntentCaptureCandidate(
                user_id="u1",
                scope_key=SCOPE,
                value="也许杭州",
                source="extractor",
                ambiguous=True,
                observed_at=NOW,
            ),
        )

        assert quarantined is not None
        assert quarantined.admission_status == "quarantined"
        # Held, not deleted: it stays readable for review and waits for the
        # user to confirm or correct it rather than expiring on a clock.
        assert list_intent_versions(
            database,
            user_id="u1",
            active_only=True,
        ) == ()
        assert len(list_intent_versions(database, user_id="u1")) == 1


def test_same_digest_without_explicit_corroboration_is_a_noop() -> None:
    with connection() as database:
        first = append_intent_version(
            database,
            user_id="u1",
            scope_key=SCOPE,
            value="上海",
            source="test",
            valid_from=NOW,
        )
        database.execute(
            """
            UPDATE career_intent_versions SET last_corroborated_at = ?
            WHERE update_id = ?
            """,
            ((NOW - timedelta(days=14)).isoformat(), first.update_id),
        )
        unchanged = append_intent_version(
            database,
            user_id="u1",
            scope_key=SCOPE,
            value="上海",
            source="migration:career_profile_context",
            valid_from=NOW,
        )
        corroborated = append_intent_version(
            database,
            user_id="u1",
            scope_key=SCOPE,
            value="上海",
            source="user",
            valid_from=NOW,
            last_corroborated_at=NOW,
        )

        assert unchanged.last_corroborated_at == NOW - timedelta(days=14)
        assert corroborated.last_corroborated_at == NOW
