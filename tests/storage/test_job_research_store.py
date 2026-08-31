from __future__ import annotations

import sqlite3

from career_agent.storage.job_research import SQLiteJobResearchStore


def test_a_version_one_database_gains_the_company_key_without_losing_reports(
    tmp_path,
) -> None:
    """The baseline runs before upgrades, so it may not assume the column exists.

    An index on company_key placed in the baseline alone fails to open exactly
    the files that still need the upgrade, which is every existing install.
    """
    path = tmp_path / "research.sqlite3"
    SQLiteJobResearchStore(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE schema_versions SET version = 1 WHERE component = 'job_research'"
        )
        connection.execute(
            "DROP INDEX IF EXISTS job_research_reports_user_company_idx"
        )
        for table in ("job_research_runs", "job_research_reports"):
            connection.execute(f"ALTER TABLE {table} DROP COLUMN company_key")

    SQLiteJobResearchStore(path)

    with sqlite3.connect(path) as connection:
        version = connection.execute(
            "SELECT version FROM schema_versions WHERE component = 'job_research'"
        ).fetchone()[0]
        runs = {
            row[1]
            for row in connection.execute("PRAGMA table_info(job_research_runs)")
        }
        reports = {
            row[1]
            for row in connection.execute("PRAGMA table_info(job_research_reports)")
        }
    assert version == 2
    assert "company_key" in runs
    assert "company_key" in reports


def test_a_fresh_database_is_adopted_at_the_current_version(tmp_path) -> None:
    """Fresh files skip upgrades entirely, so the baseline has to be complete."""
    path = tmp_path / "fresh.sqlite3"
    SQLiteJobResearchStore(path)

    with sqlite3.connect(path) as connection:
        index = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'index' "
            "AND name = 'job_research_reports_user_company_idx'"
        ).fetchone()
    assert index is not None

