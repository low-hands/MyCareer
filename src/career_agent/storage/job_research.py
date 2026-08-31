from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import sqlite3

from career_agent.domain.job_research import (
    JobResearchReport,
    JobResearchRun,
    JobResearchSource,
)
from career_agent.storage.schema import apply_schema


class SQLiteJobResearchStore:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            apply_schema(
                connection,
                "job_research",
                2,
                self._migrate,
                finalize=self._finalize_schema,
            )
        os.chmod(self.path, 0o600)

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS job_research_runs (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                company_key TEXT,
                job_posting_id TEXT NOT NULL,
                jd_snapshot_id TEXT NOT NULL,
                scope_json TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('running', 'completed', 'failed', 'cancelled')),
                input_fingerprint TEXT NOT NULL,
                worker_version TEXT NOT NULL,
                report_id TEXT,
                error_code TEXT,
                error_detail TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS job_research_one_active_input_idx
            ON job_research_runs(user_id, input_fingerprint, worker_version)
            WHERE status = 'running'
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS job_research_runs_user_job_idx
            ON job_research_runs(user_id, job_posting_id, updated_at DESC)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS job_research_sources (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                source_key TEXT NOT NULL,
                url TEXT NOT NULL,
                normalized_url TEXT NOT NULL,
                title TEXT NOT NULL,
                publisher TEXT,
                published_at TEXT,
                retrieved_at TEXT NOT NULL,
                relevant_excerpt TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                UNIQUE(run_id, source_key),
                UNIQUE(run_id, normalized_url),
                FOREIGN KEY(run_id) REFERENCES job_research_runs(id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS job_research_reports (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL UNIQUE,
                user_id TEXT NOT NULL,
                company_key TEXT,
                job_posting_id TEXT NOT NULL,
                jd_snapshot_id TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('current', 'outdated', 'superseded')),
                report_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES job_research_runs(id)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS job_research_reports_user_job_idx
            ON job_research_reports(user_id, job_posting_id, created_at DESC)
            """
        )

    @staticmethod
    def _finalize_schema(connection: sqlite3.Connection) -> None:
        """Create objects that depend on the complete v2 column set."""
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS job_research_reports_user_company_idx
            ON job_research_reports(user_id, company_key, created_at DESC)
            """
        )

    def create_run(self, run: JobResearchRun) -> JobResearchRun:
        with self._connect() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO job_research_runs(
                        id, user_id, company_key, job_posting_id, jd_snapshot_id,
                        scope_json, status, input_fingerprint, worker_version,
                        report_id, error_code, error_detail, started_at,
                        completed_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run.id,
                        run.user_id,
                        run.company_key,
                        run.job_posting_id,
                        run.jd_snapshot_id,
                        run.scope.model_dump_json(),
                        run.status,
                        run.input_fingerprint,
                        run.worker_version,
                        run.report_id,
                        run.error_code,
                        run.error_detail,
                        run.started_at.isoformat(),
                        run.completed_at.isoformat() if run.completed_at else None,
                        run.updated_at.isoformat(),
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError("an equivalent job research run is already active") from error
        return run

    def get_run(self, *, user_id: str, run_id: str) -> JobResearchRun | None:
        with self._connect() as connection:
            row = connection.execute(
                self._RUN_SELECT + " WHERE user_id = ? AND id = ?",
                (user_id, run_id),
            ).fetchone()
        return self._run(row) if row else None

    def find_completed(
        self,
        *,
        user_id: str,
        company_key: str,
        input_fingerprint: str,
        worker_version: str,
        created_after: datetime,
    ) -> JobResearchReport | None:
        """Find a fresh report to reuse for this company.

        The company is matched as well as the fingerprint, even though the
        fingerprint already derives from it. It costs nothing, and it means a
        later change to how the fingerprint is computed can never make one
        company's research answerable with another's.
        """
        with self._connect() as connection:
            row = connection.execute(
                self._REPORT_SELECT
                + " JOIN job_research_runs r ON r.id = p.run_id "
                "WHERE p.user_id = ? AND p.company_key = ? "
                "AND r.input_fingerprint = ? "
                "AND r.worker_version = ? AND r.status = 'completed' "
                "AND p.status = 'current' AND p.created_at >= ? "
                "ORDER BY p.created_at DESC LIMIT 1",
                (
                    user_id,
                    company_key,
                    input_fingerprint,
                    worker_version,
                    created_after.isoformat(),
                ),
            ).fetchone()
        return self._report(row) if row else None

    def mark_running(self, *, run: JobResearchRun, updated_at: datetime) -> JobResearchRun:
        if run.status != "failed":
            raise ValueError("only failed research runs can be retried")
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE job_research_runs
                SET status = 'running', error_code = NULL, error_detail = NULL,
                    updated_at = ?
                WHERE id = ? AND user_id = ? AND status = 'failed'
                """,
                (updated_at.isoformat(), run.id, run.user_id),
            )
        current = self.get_run(user_id=run.user_id, run_id=run.id)
        if current is None or current.status != "running":
            raise RuntimeError("failed to resume job research run")
        return current

    def fail(
        self,
        *,
        run: JobResearchRun,
        error_code: str,
        error_detail: str,
        updated_at: datetime,
    ) -> JobResearchRun:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE job_research_runs
                SET status = 'failed', error_code = ?, error_detail = ?, updated_at = ?
                WHERE id = ? AND user_id = ? AND status = 'running'
                """,
                (
                    error_code,
                    error_detail,
                    updated_at.isoformat(),
                    run.id,
                    run.user_id,
                ),
            )
        current = self.get_run(user_id=run.user_id, run_id=run.id)
        if current is None:
            raise RuntimeError("failed to persist job research failure")
        return current

    def complete(
        self,
        *,
        run: JobResearchRun,
        sources: tuple[JobResearchSource, ...],
        report: JobResearchReport,
    ) -> JobResearchReport:
        if run.status != "running" or report.run_id != run.id:
            raise ValueError("research completion does not match an active run")
        if any(source.run_id != run.id for source in sources):
            raise ValueError("research source belongs to a different run")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT status FROM job_research_runs WHERE id = ? AND user_id = ?",
                (run.id, run.user_id),
            ).fetchone()
            if current is None or current[0] != "running":
                raise ValueError("research run is no longer active")
            connection.execute(
                """
                UPDATE job_research_reports
                SET status = 'superseded'
                WHERE user_id = ? AND company_key = ? AND status = 'current'
                  AND run_id IN (
                      SELECT id
                      FROM job_research_runs
                      WHERE user_id = ?
                        AND input_fingerprint = ?
                        AND worker_version = ?
                  )
                """,
                (
                    run.user_id,
                    run.company_key,
                    run.user_id,
                    run.input_fingerprint,
                    run.worker_version,
                ),
            )
            for source in sources:
                connection.execute(
                    """
                    INSERT INTO job_research_sources(
                        id, run_id, source_key, url, normalized_url, title,
                        publisher, published_at, retrieved_at, relevant_excerpt,
                        content_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source.id,
                        source.run_id,
                        source.source_key,
                        source.url,
                        source.normalized_url,
                        source.title,
                        source.publisher,
                        source.published_at.isoformat() if source.published_at else None,
                        source.retrieved_at.isoformat(),
                        source.relevant_excerpt,
                        source.content_sha256,
                    ),
                )
            connection.execute(
                """
                INSERT INTO job_research_reports(
                    id, run_id, user_id, company_key, job_posting_id,
                    jd_snapshot_id, status, report_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.id,
                    report.run_id,
                    report.user_id,
                    report.company_key,
                    report.job_posting_id,
                    report.jd_snapshot_id,
                    report.status,
                    report.model_dump_json(),
                    report.created_at.isoformat(),
                ),
            )
            connection.execute(
                """
                UPDATE job_research_runs
                SET status = 'completed', report_id = ?, completed_at = ?, updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (
                    report.id,
                    report.created_at.isoformat(),
                    report.created_at.isoformat(),
                    run.id,
                    run.user_id,
                ),
            )
        return report

    def get_report(
        self,
        *,
        user_id: str,
        report_id: str,
        outdated_before: datetime | None = None,
    ) -> JobResearchReport | None:
        with self._connect() as connection:
            row = connection.execute(
                self._REPORT_SELECT + " WHERE p.user_id = ? AND p.id = ?",
                (user_id, report_id),
            ).fetchone()
        report = self._report(row) if row else None
        if (
            report is not None
            and report.status == "current"
            and outdated_before is not None
            and report.created_at < outdated_before
        ):
            return report.model_copy(update={"status": "outdated"})
        return report

    def latest_report(
        self,
        *,
        user_id: str,
        job_posting_id: str,
        outdated_before: datetime | None = None,
    ) -> JobResearchReport | None:
        with self._connect() as connection:
            row = connection.execute(
                self._REPORT_SELECT
                + " WHERE p.user_id = ? AND p.job_posting_id = ? "
                "ORDER BY p.created_at DESC LIMIT 1",
                (user_id, job_posting_id),
            ).fetchone()
        if row is None:
            return None
        report = self._report(row)
        if (
            report.status == "current"
            and outdated_before is not None
            and report.created_at < outdated_before
        ):
            return report.model_copy(update={"status": "outdated"})
        return report

    def latest_company_report(
        self,
        *,
        user_id: str,
        company_key: str,
        outdated_before: datetime | None = None,
    ) -> JobResearchReport | None:
        with self._connect() as connection:
            row = connection.execute(
                self._REPORT_SELECT
                + " WHERE p.user_id = ? AND p.company_key = ? "
                "ORDER BY CASE WHEN p.status = 'current' THEN 0 ELSE 1 END, "
                "p.created_at DESC LIMIT 1",
                (user_id, company_key),
            ).fetchone()
        if row is None:
            return None
        report = self._report(row)
        if (
            report.status == "current"
            and outdated_before is not None
            and report.created_at < outdated_before
        ):
            return report.model_copy(update={"status": "outdated"})
        return report

    def list_reports(
        self,
        *,
        user_id: str,
        limit: int = 50,
    ) -> tuple[JobResearchReport, ...]:
        if limit < 1:
            raise ValueError("limit must be positive")
        with self._connect() as connection:
            rows = connection.execute(
                self._REPORT_SELECT
                + " WHERE p.user_id = ? "
                "ORDER BY CASE WHEN p.status = 'current' THEN 0 ELSE 1 END, "
                "p.created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return tuple(self._report(row) for row in rows)

    def list_sources(
        self,
        *,
        user_id: str,
        report_id: str,
    ) -> tuple[JobResearchSource, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                self._SOURCE_SELECT
                + " JOIN job_research_reports p ON p.run_id = s.run_id "
                "WHERE p.user_id = ? AND p.id = ? ORDER BY s.source_key",
                (user_id, report_id),
            ).fetchall()
        return tuple(self._source(row) for row in rows)

    _RUN_SELECT = (
        "SELECT id, user_id, company_key, job_posting_id, jd_snapshot_id, "
        "scope_json, status, "
        "input_fingerprint, worker_version, report_id, error_code, error_detail, "
        "started_at, completed_at, updated_at FROM job_research_runs"
    )
    _REPORT_SELECT = (
        "SELECT p.report_json, p.status FROM job_research_reports p"
    )
    _SOURCE_SELECT = (
        "SELECT s.id, s.run_id, s.source_key, s.url, s.normalized_url, s.title, "
        "s.publisher, s.published_at, s.retrieved_at, s.relevant_excerpt, "
        "s.content_sha256 FROM job_research_sources s"
    )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @staticmethod
    def _run(row) -> JobResearchRun:
        from career_agent.domain.job_research import JobResearchScope

        return JobResearchRun(
            id=row[0],
            user_id=row[1],
            company_key=row[2],
            job_posting_id=row[3],
            jd_snapshot_id=row[4],
            scope=JobResearchScope.model_validate_json(row[5]),
            status=row[6],
            input_fingerprint=row[7],
            worker_version=row[8],
            report_id=row[9],
            error_code=row[10],
            error_detail=row[11],
            started_at=row[12],
            completed_at=row[13],
            updated_at=row[14],
        )

    @staticmethod
    def _report(row) -> JobResearchReport:
        report = JobResearchReport.model_validate_json(row[0])
        return report.model_copy(update={"status": row[1]})

    @staticmethod
    def _source(row) -> JobResearchSource:
        return JobResearchSource(
            id=row[0],
            run_id=row[1],
            source_key=row[2],
            url=row[3],
            normalized_url=row[4],
            title=row[5],
            publisher=row[6],
            published_at=row[7],
            retrieved_at=row[8],
            relevant_excerpt=row[9],
            content_sha256=row[10],
        )
