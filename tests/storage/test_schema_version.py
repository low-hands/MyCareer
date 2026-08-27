import re
import sqlite3
from pathlib import Path

import pytest

from career_agent.storage.career_history import CareerHistoryStore
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
    "resumes": 3,
    "career_history": 2,
    "action_center": 2,
    "applications": 1,
    "interviews": 1,
    "calendar": 1,
    "email_tracking": 1,
    "mock_interviews": 2,
    "resume_analysis": 1,
    "resume_tailoring": 1,
    "resume_artifacts": 1,
    "resume_job_matches": 1,
    "interview_preparations": 1,
    "agent_context": 1,
    "job_postings": 1,
    "job_discovery_runs": 1,
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
