import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from career_agent.agent.conversation_memory_contracts import (
    ConversationSummaryContent,
)
from career_agent.agent.main_agent_contracts import CareerProfileContext
from career_agent.storage.calendar import SQLiteCalendarStore
from career_agent.storage.action_executions import SQLiteActionExecutionStore
from career_agent.storage.career_history import CareerHistoryStore
from career_agent.storage.context import CareerContextStore
from career_agent.storage.interview_preparations import (
    SQLiteInterviewPreparationStore,
)
from career_agent.storage.resume_analysis import SQLiteResumeAnalysisDraftStore
from career_agent.storage.resume_artifacts import SQLiteResumeArtifactStore
from career_agent.storage.resume_job_matches import SQLiteResumeJobMatchStore
from career_agent.storage.resume_tailoring import SQLiteResumeTailoringDraftStore
from career_agent.storage.resumes import ResumeStore
from career_agent.storage.schema import (
    SchemaVersionError,
    apply_schema,
    check_schema_version,
    record_schema_version,
)


SHARED_STORES = (
    ResumeStore,
    CareerHistoryStore,
    SQLiteResumeJobMatchStore,
    SQLiteResumeArtifactStore,
    SQLiteResumeAnalysisDraftStore,
    SQLiteResumeTailoringDraftStore,
    SQLiteInterviewPreparationStore,
)


def test_every_owner_of_the_shared_file_records_its_own_version(tmp_path: Path) -> None:
    path = tmp_path / "resumes.sqlite3"
    for store in SHARED_STORES:
        store(path)

    with sqlite3.connect(path) as connection:
        recorded = dict(
            connection.execute("SELECT component, version FROM schema_versions")
        )

    # PRAGMA user_version is one integer per file, so it can only ever describe
    # one of these seven. Each owner needs its own row.
    assert set(recorded) == {
        "resumes",
        "career_history",
        "resume_job_matches",
        "resume_artifacts",
        "resume_analysis",
        "resume_tailoring",
        "interview_preparations",
    }
    assert all(version >= 1 for version in recorded.values())


# Every component and the version its code claims. The runtime cannot derive these
# — the numbers predate the registry — so raising one has to be a deliberate edit
# here as well, which is the moment to notice a migration was never written.
DECLARED_VERSIONS = {
    "resumes": 9,
    "career_history": 8,
    "action_center": 2,
    "action_executions": 1,
    "turn_receipts": 2,
    "applications": 1,
    "interviews": 2,
    "calendar": 3,
    "email_tracking": 1,
    "oauth_flows": 2,
    "mock_interviews": 3,
    "resume_analysis": 1,
    "resume_tailoring": 1,
    "resume_artifacts": 1,
    "resume_job_matches": 1,
    "interview_preparations": 1,
    "agent_context": 17,
    "api_keys": 3,
    "capability_confirmations": 2,
    "job_postings": 2,
    "job_research": 2,
    "run_events": 2,
    "career_episodes": 8,
}


def test_no_component_declares_a_version_this_table_does_not_know_about() -> None:
    """Bumping a version without writing the migration fails here.

    ``apply_schema`` takes the version as an argument because these numbers are
    historical, so nothing at runtime can tell a real upgrade from a typo. This
    table is the second place the number has to change.
    """
    sources = Path("src/career_agent/storage").glob("*.py")
    declared: dict[str, int] = {}
    for source in sources:
        for component, version in re.findall(
            r"apply_schema\(\s*connection,\s*\"([a-z_]+)\",\s*(\d+)", source.read_text()
        ):
            declared[component] = int(version)

    assert declared == DECLARED_VERSIONS


def test_agent_context_v13_migrates_free_text_scope_and_expiry(
    tmp_path: Path,
) -> None:
    path = tmp_path / "context.sqlite3"
    store = CareerContextStore(path)
    pending = store.capture_free_text_preference_from_message(
        user_id="u1",
        conversation_id="c1",
        message="我不去大厂。",
    )
    assert pending is not None
    old_scope = "person_intent/self/employer_scale_preference"
    new_scope = "person_intent/self/company_scale"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE career_intent_versions SET scope_key = ?",
            (old_scope,),
        )
        connection.execute(
            """
            INSERT INTO conversation_message_memory_bindings(
                user_id, conversation_id, sequence, scope_key, created_at
            ) VALUES ('u1', 'c1', 1, ?, ?)
            """,
            (old_scope, datetime.now(timezone.utc).isoformat()),
        )
        connection.execute(
            """
            UPDATE schema_versions SET version = 12
            WHERE component = 'agent_context'
            """
        )

    CareerContextStore(path)

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            """
            SELECT scope_key, valid_until
            FROM career_intent_versions
            """
        ).fetchone() == (new_scope, None)
        assert connection.execute(
            """
            SELECT scope_key FROM conversation_message_memory_bindings
            """
        ).fetchone() == (new_scope,)
        assert connection.execute(
            """
            SELECT topic_key, statement
            FROM free_text_preferences_fts
            """
        ).fetchone() == ("employer_scale", "我不去大厂。")


def test_agent_context_v10_drops_removed_scope_queue_schema(tmp_path: Path) -> None:
    path = tmp_path / "context.sqlite3"
    CareerContextStore(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE scope_resolution_queue(id TEXT PRIMARY KEY, value TEXT)"
        )
        connection.execute(
            "CREATE TABLE scope_resolution_events(id TEXT PRIMARY KEY, value TEXT)"
        )
        connection.execute(
            """
            INSERT INTO schema_versions(component, version)
            VALUES ('memory_scope', 1)
            """
        )
        connection.execute(
            """
            UPDATE schema_versions SET version = 9
            WHERE component = 'agent_context'
            """
        )

    CareerContextStore(path)

    with sqlite3.connect(path) as connection:
        queue_tables = connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table'
              AND name IN ('scope_resolution_queue', 'scope_resolution_events')
            """
        ).fetchall()
        legacy_version = connection.execute(
            """
            SELECT 1 FROM schema_versions WHERE component = 'memory_scope'
            """
        ).fetchone()
    assert queue_tables == []
    assert legacy_version is None


def test_agent_context_v11_seeds_the_constraint_ledger_from_stored_summaries(
    tmp_path: Path,
) -> None:
    """A pre-v11 conversation keeps its constraints and gains an exit path."""

    path = tmp_path / "context.sqlite3"
    store = CareerContextStore(path)
    store.compact_conversation_summary(
        user_id="u1",
        conversation_id="c1",
        expected_previous_through_sequence=0,
        content=ConversationSummaryContent(
            active_constraints=("不接受 996", "不经批准不发邮件")
        ),
        through_sequence=2,
    )
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE conversation_constraint_archive")
        connection.execute(
            "UPDATE schema_versions SET version = 10 "
            "WHERE component = 'agent_context'"
        )

    reopened = CareerContextStore(path)

    ledger = reopened.list_conversation_constraints(
        user_id="u1", conversation_id="c1"
    )
    assert tuple((row.text, row.status) for row in ledger) == (
        ("不接受 996", "active"),
        ("不经批准不发邮件", "active"),
    )
    assert reopened.retire_conversation_constraint(
        user_id="u1", conversation_id="c1", constraint_text="不接受 996"
    )


def test_career_history_v7_drops_suppression_digest_column(tmp_path: Path) -> None:
    path = tmp_path / "resumes.sqlite3"
    CareerHistoryStore(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "ALTER TABLE career_evidence ADD COLUMN suppression_digest TEXT"
        )
        connection.execute(
            """
            UPDATE schema_versions SET version = 6
            WHERE component = 'career_history'
            """
        )

    CareerHistoryStore(path)

    with sqlite3.connect(path) as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(career_evidence)")
        }
    assert "suppression_digest" not in columns


def test_no_store_writes_the_file_wide_version_the_registry_replaced() -> None:
    """The registry has to be the only source of truth, not merely the better one.

    ``PRAGMA user_version`` is one integer per *file*, and seven stores own
    ``resumes.sqlite3``. While any store still wrote it, the file carried two
    version records that nothing reconciled: bumping a component in
    ``schema_versions`` left the PRAGMA behind, and no code path noticed. Nothing
    ever read it — so the fix is to stop writing it, and this test is what keeps
    a future store from reintroducing the second record.
    """
    offenders = [
        source.name
        for source in Path("src/career_agent/storage").glob("*.py")
        if re.search(r"PRAGMA\s+user_version\s*=", source.read_text())
    ]
    assert offenders == []


def test_the_shared_file_carries_component_rows_and_no_file_wide_version(
    tmp_path: Path,
) -> None:
    path = tmp_path / "resumes.sqlite3"
    for store in SHARED_STORES:
        store(path)

    with sqlite3.connect(path) as connection:
        file_wide = connection.execute("PRAGMA user_version").fetchone()[0]
        components = connection.execute(
            "SELECT COUNT(*) FROM schema_versions"
        ).fetchone()[0]

    # Untouched at its default. A number here would describe one of seven owners
    # and misdescribe the other six.
    assert file_wide == 0
    assert components == len(SHARED_STORES)


def test_opening_a_newer_schema_fails_instead_of_writing(tmp_path: Path) -> None:
    path = tmp_path / "resumes.sqlite3"
    ResumeStore(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE schema_versions SET version = 99 WHERE component = 'resumes'"
        )

    # An older build would find its CREATE TABLE IF NOT EXISTS calls satisfied
    # and go on to write rows the newer build considers malformed.
    with pytest.raises(SchemaVersionError) as error:
        ResumeStore(path)
    assert error.value.found == 99


def test_context_v5_backfills_current_default_city(tmp_path: Path) -> None:
    path = tmp_path / "context.sqlite3"
    store = CareerContextStore(path)
    store.upsert_profile(CareerProfileContext(user_id="u1", default_city="上海"))
    with sqlite3.connect(path) as connection:
        connection.execute(
            "DELETE FROM career_intent_versions WHERE user_id = 'u1'"
        )
        connection.execute(
            "UPDATE schema_versions SET version = 4 "
            "WHERE component = 'agent_context'"
        )

    reopened = CareerContextStore(path)
    versions = reopened.list_profile_intent_versions(user_id="u1")

    assert [(item.scope_key, item.value, item.revision) for item in versions] == [
        ("person_intent/self/default_city", "上海", 1)
    ]
    assert versions[0].source == "migration:career_profile_context"


def test_resumes_v5_backfills_each_current_role_intent_field(
    tmp_path: Path,
) -> None:
    path = tmp_path / "resumes.sqlite3"
    store = ResumeStore(path)
    role = store.create_target_role(user_id="u1", title="Agent", priority=0)
    store.update_target_role_intent(
        user_id="u1",
        target_role_id=role.id,
        city="北京",
        salary_expectation="40K",
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            "DELETE FROM career_intent_versions WHERE user_id = 'u1'"
        )
        connection.execute(
            "UPDATE schema_versions SET version = 4 WHERE component = 'resumes'"
        )

    reopened = ResumeStore(path)
    versions = reopened.list_target_role_intent_versions(user_id="u1")

    assert [(item.scope_key, item.value, item.revision) for item in versions] == [
        (f"target_role_intent/{role.id}/city", "北京", 1),
        (
            f"target_role_intent/{role.id}/salary_expectation",
            "40K",
            1,
        ),
    ]
    assert {item.source for item in versions} == {"migration:target_roles"}


def test_context_reopen_does_not_refresh_intent_decay_clock(tmp_path: Path) -> None:
    path = tmp_path / "context.sqlite3"
    store = CareerContextStore(path)
    store.upsert_profile(CareerProfileContext(user_id="u1", default_city="上海"))
    stale = datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat()
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            UPDATE career_intent_versions SET last_corroborated_at = ?
            WHERE user_id = 'u1'
            """,
            (stale,),
        )

    reopened = CareerContextStore(path)

    assert reopened.list_profile_intent_versions(
        user_id="u1"
    )[0].last_corroborated_at.isoformat() == stale


def test_resume_reopen_does_not_refresh_intent_decay_clock(tmp_path: Path) -> None:
    path = tmp_path / "resumes.sqlite3"
    store = ResumeStore(path)
    role = store.create_target_role(user_id="u1", title="Agent", priority=0)
    store.update_target_role_intent(
        user_id="u1",
        target_role_id=role.id,
        city="北京",
    )
    stale = datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat()
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            UPDATE career_intent_versions SET last_corroborated_at = ?
            WHERE user_id = 'u1'
            """,
            (stale,),
        )

    reopened = ResumeStore(path)

    assert reopened.list_target_role_intent_versions(
        user_id="u1"
    )[0].last_corroborated_at.isoformat() == stale


def test_registration_is_idempotent_and_reports_the_previous_version(
    tmp_path: Path,
) -> None:
    with sqlite3.connect(tmp_path / "x.sqlite3") as connection:
        assert check_schema_version(connection, "demo", 1) == 0
        record_schema_version(connection, "demo", 1)
        assert check_schema_version(connection, "demo", 1) == 1
        # A later version upgrades in place, which is how a migration detects
        # that it has work to do.
        assert check_schema_version(connection, "demo", 2) == 1


def test_the_version_is_checked_before_any_migration_statement_runs(
    tmp_path: Path,
) -> None:
    """Order matters: a newer file must raise SchemaVersionError, not OperationalError.

    If the migration ran first it could trip over a column that changed meaning,
    and the failure would look like a corrupt database rather than a stale build.
    """
    ran: list[str] = []

    def migrate(connection: sqlite3.Connection) -> None:
        ran.append("baseline")

    with sqlite3.connect(tmp_path / "x.sqlite3") as connection:
        apply_schema(connection, "demo", 1, migrate)
        assert ran == ["baseline"]
        connection.execute("UPDATE schema_versions SET version = 5")
        with pytest.raises(SchemaVersionError):
            apply_schema(connection, "demo", 1, migrate)
    assert ran == ["baseline"]


def test_the_version_is_recorded_only_after_the_migration_succeeds(
    tmp_path: Path,
) -> None:
    """A half-applied schema marked current would never be repaired."""

    def failing(connection: sqlite3.Connection) -> None:
        raise sqlite3.OperationalError("migration blew up")

    path = tmp_path / "x.sqlite3"
    with sqlite3.connect(path) as connection:
        with pytest.raises(sqlite3.OperationalError):
            apply_schema(connection, "demo", 1, failing)
    with sqlite3.connect(path) as connection:
        assert check_schema_version(connection, "demo", 1) == 0


def test_a_one_way_upgrade_runs_only_for_a_file_older_than_it(tmp_path: Path) -> None:
    """The baseline is idempotent and always runs; backfills are not and must not."""
    calls: list[str] = []

    def baseline(connection: sqlite3.Connection) -> None:
        calls.append("baseline")

    upgrades = {2: lambda connection: calls.append("backfill")}

    path = tmp_path / "x.sqlite3"
    with sqlite3.connect(path) as connection:
        # A new component is created directly at the current shape. Replaying a
        # historical ALTER/backfill over that shape would be wrong.
        apply_schema(connection, "demo", 1, baseline)
    assert calls == ["baseline"]

    with sqlite3.connect(path) as connection:
        apply_schema(connection, "demo", 2, baseline, upgrades)
    assert calls == ["baseline", "baseline", "backfill"]

    with sqlite3.connect(path) as connection:
        apply_schema(connection, "demo", 2, baseline, upgrades)
    # Re-opening repairs the shape but does not replay the backfill.
    assert calls == ["baseline", "baseline", "backfill", "baseline"]


def test_an_upgrade_can_use_a_table_the_same_version_baseline_adds(
    tmp_path: Path,
) -> None:
    """Baseline runs first, so a version may add a table and backfill it at once.

    The other order fails on `no such table`, which would force splitting one
    logical migration across two releases for no reason.
    """

    def baseline(connection: sqlite3.Connection) -> None:
        connection.execute("CREATE TABLE IF NOT EXISTS notes(id TEXT PRIMARY KEY)")
        connection.execute("CREATE TABLE IF NOT EXISTS note_tags(id TEXT, tag TEXT)")

    def backfill(connection: sqlite3.Connection) -> None:
        connection.execute(
            "INSERT INTO note_tags(id, tag) SELECT id, 'untagged' FROM notes"
        )

    path = tmp_path / "x.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE notes(id TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO notes(id) VALUES ('n1')")
        apply_schema(connection, "demo", 1, lambda c: None)

    with sqlite3.connect(path) as connection:
        apply_schema(connection, "demo", 2, baseline, {2: backfill})
        assert connection.execute("SELECT tag FROM note_tags").fetchall() == [
            ("untagged",)
        ]


def test_finalization_runs_after_upgrades_add_the_columns_it_uses(
    tmp_path: Path,
) -> None:
    """Dependent indexes must never run before their column upgrade."""

    def baseline(connection: sqlite3.Connection) -> None:
        connection.execute("CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY)")

    def add_company_key(connection: sqlite3.Connection) -> None:
        connection.execute("ALTER TABLE records ADD COLUMN company_key TEXT")

    def finalize(connection: sqlite3.Connection) -> None:
        connection.execute(
            "CREATE INDEX IF NOT EXISTS records_company_idx "
            "ON records(company_key)"
        )

    path = tmp_path / "x.sqlite3"
    with sqlite3.connect(path) as connection:
        apply_schema(connection, "demo", 1, baseline)
    with sqlite3.connect(path) as connection:
        apply_schema(
            connection,
            "demo",
            2,
            baseline,
            {2: add_company_key},
            finalize=finalize,
        )
        assert connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'index' "
            "AND name = 'records_company_idx'"
        ).fetchone() == (1,)


def test_an_existing_component_cannot_advance_with_a_missing_upgrade(
    tmp_path: Path,
) -> None:
    path = tmp_path / "x.sqlite3"
    with sqlite3.connect(path) as connection:
        apply_schema(connection, "demo", 1, lambda c: None)
    with sqlite3.connect(path) as connection:
        with pytest.raises(ValueError, match="missing upgrades for versions \\(2,\\)"):
            apply_schema(connection, "demo", 2, lambda c: None)


def test_an_upgrade_above_the_declared_version_is_rejected(tmp_path: Path) -> None:
    """Otherwise the recorded number would understate what the file contains."""
    with sqlite3.connect(tmp_path / "x.sqlite3") as connection:
        with pytest.raises(ValueError, match="understate"):
            apply_schema(
                connection, "demo", 1, lambda c: None, {2: lambda c: None}
            )


def test_calendar_v1_store_upgrades_with_an_execution_ledger(tmp_path: Path) -> None:
    path = tmp_path / "calendar.sqlite3"
    SQLiteCalendarStore(path)
    with sqlite3.connect(path) as connection:
        connection.execute("DROP INDEX calendar_executions_user_status_idx")
        connection.execute("DROP TABLE calendar_operation_executions")
        connection.execute(
            "UPDATE schema_versions SET version = 1 WHERE component = 'calendar'"
        )

    SQLiteCalendarStore(path)

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT version FROM schema_versions WHERE component = 'calendar'"
        ).fetchone() == (3,)
        assert connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'calendar_operation_executions'"
        ).fetchone() == (1,)


def test_calendar_v2_store_backfills_the_original_policy_epoch(tmp_path: Path) -> None:
    path = tmp_path / "calendar.sqlite3"
    SQLiteCalendarStore(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "ALTER TABLE calendar_change_proposals DROP COLUMN policy_epoch"
        )
        connection.execute(
            "UPDATE schema_versions SET version = 2 WHERE component = 'calendar'"
        )

    SQLiteCalendarStore(path)

    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT version FROM schema_versions WHERE component = 'calendar'"
        ).fetchone() == (3,)
        columns = {
            row[1]: row for row in connection.execute(
                "PRAGMA table_info(calendar_change_proposals)"
            )
        }
        assert columns["policy_epoch"][3] == 1
        assert columns["policy_epoch"][4] == "1"
