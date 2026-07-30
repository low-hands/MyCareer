"""SQLite implementation of the search-run snapshot repository."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from career_agent.jobs.search_models import SearchCriteria, SearchRun
from career_agent.jobs.search_repository import DuplicateSearchRunError


class SQLiteSearchRunRepository:
    """Store immutable search-run snapshots in a shared SQLite database.

    Two tables are used:

    - ``search_runs`` — one row per SearchRun (metadata only)
    - ``search_run_items`` — ordered job_id references per run

    Full job data lives exclusively in ``job_postings``; this repository
    never duplicates posting content.
    """

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize_schema(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            job_table = connection.execute(
                """
                SELECT 1
                FROM sqlite_master
                WHERE type = 'table' AND name = 'job_postings'
                """
            ).fetchone()
            if job_table is None:
                raise RuntimeError(
                    "job_postings schema is missing; initialize "
                    "SQLiteJobPostingRepository before SQLiteSearchRunRepository"
                )
            connection.executescript(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_job_postings_tenant_job
                ON job_postings (tenant_id, job_id);

                CREATE TABLE IF NOT EXISTS search_runs (
                    run_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    user_query TEXT NOT NULL,
                    criteria_json TEXT NOT NULL,
                    candidate_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (run_id, tenant_id)
                );

                CREATE TABLE IF NOT EXISTS search_run_items (
                    run_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    job_id TEXT NOT NULL,
                    PRIMARY KEY (run_id, position),
                    UNIQUE (run_id, job_id),
                    FOREIGN KEY (run_id, tenant_id)
                        REFERENCES search_runs (run_id, tenant_id),
                    FOREIGN KEY (tenant_id, job_id)
                        REFERENCES job_postings (tenant_id, job_id)
                );

                CREATE INDEX IF NOT EXISTS idx_search_runs_tenant_thread
                ON search_runs (tenant_id, thread_id, created_at DESC);
                """
            )

    # ------------------------------------------------------------------
    # SearchRunRepository implementation
    # ------------------------------------------------------------------

    def create(self, run: SearchRun) -> SearchRun:
        """Persist a new immutable SearchRun.

        Validates that every referenced job_id belongs to the same tenant.
        Returns the run unchanged on success.
        """
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            duplicate = connection.execute(
                "SELECT 1 FROM search_runs WHERE run_id = ?",
                (run.run_id,),
            ).fetchone()
            if duplicate is not None:
                raise DuplicateSearchRunError(f"Search run already exists: {run.run_id}")

            # Verify every job_id belongs to the run's tenant
            placeholders = ",".join("?" for _ in run.job_ids)
            rows = connection.execute(
                f"""
                SELECT job_id
                FROM job_postings
                WHERE tenant_id = ? AND job_id IN ({placeholders})
                """,
                (run.tenant_id, *run.job_ids),
            ).fetchall()
            found_ids = {row["job_id"] for row in rows}
            missing = set(run.job_ids) - found_ids
            if missing:
                raise ValueError(f"job_ids not found for tenant {run.tenant_id!r}: {sorted(missing)}")

            connection.execute(
                """
                INSERT INTO search_runs (
                    run_id, tenant_id, thread_id, user_query,
                    criteria_json, candidate_count, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.tenant_id,
                    run.thread_id,
                    run.user_query,
                    run.criteria.model_dump_json(),
                    run.candidate_count,
                    run.created_at.isoformat(),
                ),
            )

            for position, job_id in enumerate(run.job_ids, start=1):
                connection.execute(
                    """
                    INSERT INTO search_run_items (
                        run_id, tenant_id, position, job_id
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (run.run_id, run.tenant_id, position, job_id),
                )

            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

        return run

    def get(self, tenant_id: str, run_id: str) -> SearchRun | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM search_runs
                WHERE tenant_id = ? AND run_id = ?
                """,
                (tenant_id, run_id),
            ).fetchone()
            if row is None:
                return None

            items = connection.execute(
                """
                SELECT job_id
                FROM search_run_items
                WHERE run_id = ? AND tenant_id = ?
                ORDER BY position
                """,
                (run_id, tenant_id),
            ).fetchall()

        return self._row_to_run(row, items)

    def get_latest(self, tenant_id: str, thread_id: str) -> SearchRun | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM search_runs
                WHERE tenant_id = ? AND thread_id = ?
                ORDER BY created_at DESC, rowid DESC
                LIMIT 1
                """,
                (tenant_id, thread_id),
            ).fetchone()
            if row is None:
                return None

            items = connection.execute(
                """
                SELECT job_id
                FROM search_run_items
                WHERE run_id = ? AND tenant_id = ?
                ORDER BY position
                """,
                (row["run_id"], tenant_id),
            ).fetchall()

        return self._row_to_run(row, items)

    def list_by_thread(
        self,
        tenant_id: str,
        thread_id: str,
        *,
        limit: int = 20,
    ) -> list[SearchRun]:
        if limit < 1:
            raise ValueError("limit must be positive")

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM search_runs
                WHERE tenant_id = ? AND thread_id = ?
                ORDER BY created_at DESC, rowid DESC
                LIMIT ?
                """,
                (tenant_id, thread_id, limit),
            ).fetchall()

            runs: list[SearchRun] = []
            for row in rows:
                items = connection.execute(
                    """
                    SELECT job_id
                    FROM search_run_items
                    WHERE run_id = ? AND tenant_id = ?
                    ORDER BY position
                    """,
                    (row["run_id"], tenant_id),
                ).fetchall()
                runs.append(self._row_to_run(row, items))

        return runs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_run(row: sqlite3.Row, items: list[sqlite3.Row]) -> SearchRun:
        return SearchRun(
            run_id=row["run_id"],
            tenant_id=row["tenant_id"],
            thread_id=row["thread_id"],
            user_query=row["user_query"],
            criteria=SearchCriteria.model_validate_json(row["criteria_json"]),
            job_ids=tuple(item["job_id"] for item in items),
            candidate_count=row["candidate_count"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )
