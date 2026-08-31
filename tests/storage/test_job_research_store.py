from __future__ import annotations

import sqlite3

import pytest

from career_agent.storage.job_research import SQLiteJobResearchStore


def test_a_version_one_development_database_requires_a_rebuild(tmp_path) -> None:
    """Pre-release schemas are deliberately not migrated in place."""
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

    with pytest.raises(ValueError, match="missing upgrades for versions \\(2,\\)"):
        SQLiteJobResearchStore(path)


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
