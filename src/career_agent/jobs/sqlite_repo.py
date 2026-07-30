"""SQLite implementation of the job posting repository."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from career_agent.jobs.models import JobPosting
from career_agent.jobs.repository import JobPostingNotFoundError

_TRACKING_QUERY_KEYS = {"fbclid", "gclid"}


class SQLiteJobPostingRepository:
    """Store normalized postings in a shared SQLite database by tenant."""

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
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS job_postings (
                    job_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    identity_key TEXT NOT NULL,
                    source TEXT NOT NULL,
                    external_id TEXT NOT NULL DEFAULT '',
                    detail_locator TEXT NOT NULL DEFAULT '',
                    source_url TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL,
                    company_name TEXT NOT NULL,
                    location TEXT NOT NULL DEFAULT '',
                    employment_type TEXT NOT NULL DEFAULT '',
                    salary TEXT NOT NULL DEFAULT '',
                    experience_required TEXT NOT NULL DEFAULT '',
                    education_required TEXT NOT NULL DEFAULT '',
                    company_industry TEXT NOT NULL DEFAULT '',
                    company_size TEXT NOT NULL DEFAULT '',
                    company_stage TEXT NOT NULL DEFAULT '',
                    description TEXT NOT NULL DEFAULT '',
                    detail_level TEXT NOT NULL DEFAULT 'summary'
                        CHECK (detail_level IN ('summary', 'full')),
                    fetched_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (tenant_id, source, identity_key)
                );

                CREATE INDEX IF NOT EXISTS idx_job_postings_tenant_updated
                ON job_postings (tenant_id, updated_at DESC);
                """
            )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(job_postings)").fetchall()}
            if "detail_locator" not in columns:
                connection.execute("ALTER TABLE job_postings ADD COLUMN detail_locator TEXT NOT NULL DEFAULT ''")

    @staticmethod
    def _row_to_posting(row: sqlite3.Row) -> JobPosting:
        return JobPosting(
            job_id=row["job_id"],
            tenant_id=row["tenant_id"],
            source=row["source"],
            external_id=row["external_id"],
            detail_locator=row["detail_locator"],
            source_url=row["source_url"],
            title=row["title"],
            company_name=row["company_name"],
            location=row["location"],
            employment_type=row["employment_type"],
            salary=row["salary"],
            experience_required=row["experience_required"],
            education_required=row["education_required"],
            company_industry=row["company_industry"],
            company_size=row["company_size"],
            company_stage=row["company_stage"],
            description=row["description"],
            detail_level=row["detail_level"],
            fetched_at=datetime.fromisoformat(row["fetched_at"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def upsert(self, posting: JobPosting) -> JobPosting:
        """Insert a posting or refresh the existing source identity."""

        identity_key = _posting_identity_key(posting)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO job_postings (
                    job_id,
                    tenant_id,
                    identity_key,
                    source,
                    external_id,
                    detail_locator,
                    source_url,
                    title,
                    company_name,
                    location,
                    employment_type,
                    salary,
                    experience_required,
                    education_required,
                    company_industry,
                    company_size,
                    company_stage,
                    description,
                    detail_level,
                    fetched_at,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (tenant_id, source, identity_key)
                DO UPDATE SET
                    external_id = CASE
                        WHEN excluded.external_id <> '' THEN excluded.external_id
                        ELSE job_postings.external_id
                    END,
                    detail_locator = CASE
                        WHEN excluded.detail_locator <> ''
                        THEN excluded.detail_locator
                        ELSE job_postings.detail_locator
                    END,
                    source_url = CASE
                        WHEN excluded.source_url <> '' THEN excluded.source_url
                        ELSE job_postings.source_url
                    END,
                    title = excluded.title,
                    company_name = excluded.company_name,
                    location = CASE
                        WHEN excluded.location <> '' THEN excluded.location
                        ELSE job_postings.location
                    END,
                    employment_type = CASE
                        WHEN excluded.employment_type <> '' THEN excluded.employment_type
                        ELSE job_postings.employment_type
                    END,
                    salary = CASE
                        WHEN excluded.salary <> '' THEN excluded.salary
                        ELSE job_postings.salary
                    END,
                    experience_required = CASE
                        WHEN excluded.experience_required <> ''
                        THEN excluded.experience_required
                        ELSE job_postings.experience_required
                    END,
                    education_required = CASE
                        WHEN excluded.education_required <> ''
                        THEN excluded.education_required
                        ELSE job_postings.education_required
                    END,
                    company_industry = CASE
                        WHEN excluded.company_industry <> ''
                        THEN excluded.company_industry
                        ELSE job_postings.company_industry
                    END,
                    company_size = CASE
                        WHEN excluded.company_size <> '' THEN excluded.company_size
                        ELSE job_postings.company_size
                    END,
                    company_stage = CASE
                        WHEN excluded.company_stage <> '' THEN excluded.company_stage
                        ELSE job_postings.company_stage
                    END,
                    description = CASE
                        WHEN excluded.detail_level = 'full' THEN excluded.description
                        ELSE job_postings.description
                    END,
                    detail_level = CASE
                        WHEN job_postings.detail_level = 'full'
                          OR excluded.detail_level = 'full'
                        THEN 'full'
                        ELSE 'summary'
                    END,
                    fetched_at = excluded.fetched_at,
                    updated_at = excluded.updated_at
                """,
                (
                    posting.job_id,
                    posting.tenant_id,
                    identity_key,
                    posting.source,
                    posting.external_id,
                    posting.detail_locator,
                    posting.source_url,
                    posting.title,
                    posting.company_name,
                    posting.location,
                    posting.employment_type,
                    posting.salary,
                    posting.experience_required,
                    posting.education_required,
                    posting.company_industry,
                    posting.company_size,
                    posting.company_stage,
                    posting.description,
                    posting.detail_level,
                    posting.fetched_at.isoformat(),
                    posting.created_at.isoformat(),
                    posting.updated_at.isoformat(),
                ),
            )
            row = connection.execute(
                """
                SELECT *
                FROM job_postings
                WHERE tenant_id = ? AND source = ? AND identity_key = ?
                """,
                (posting.tenant_id, posting.source, identity_key),
            ).fetchone()

        if row is None:  # pragma: no cover - defensive check after write
            raise RuntimeError(f"Upserted job posting cannot be reloaded: {posting.job_id}")
        return self._row_to_posting(row)

    def get(self, tenant_id: str, job_id: str) -> JobPosting | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM job_postings
                WHERE tenant_id = ? AND job_id = ?
                """,
                (tenant_id, job_id),
            ).fetchone()
        return self._row_to_posting(row) if row is not None else None

    def get_many(
        self,
        tenant_id: str,
        job_ids: Sequence[str],
    ) -> list[JobPosting]:
        if not job_ids:
            return []

        placeholders = ", ".join("?" for _ in job_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT *
                FROM job_postings
                WHERE tenant_id = ? AND job_id IN ({placeholders})
                """,
                [tenant_id, *job_ids],
            ).fetchall()

        postings_by_id = {posting.job_id: posting for posting in (self._row_to_posting(row) for row in rows)}
        return [postings_by_id[job_id] for job_id in job_ids if job_id in postings_by_id]

    def save_full_detail(
        self,
        *,
        tenant_id: str,
        job_id: str,
        description: str,
        fetched_at: datetime | None = None,
    ) -> JobPosting:
        normalized_description = description.strip()
        if not normalized_description:
            raise ValueError("description must not be blank")

        refreshed_at = fetched_at or datetime.now(UTC)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE job_postings
                SET description = ?,
                    detail_level = 'full',
                    fetched_at = ?,
                    updated_at = ?
                WHERE tenant_id = ? AND job_id = ?
                """,
                (
                    normalized_description,
                    refreshed_at.isoformat(),
                    refreshed_at.isoformat(),
                    tenant_id,
                    job_id,
                ),
            )
            if cursor.rowcount == 0:
                raise JobPostingNotFoundError(f"Job posting not found for tenant {tenant_id!r}: {job_id}")

        posting = self.get(tenant_id, job_id)
        if posting is None:  # pragma: no cover - defensive check after write
            raise RuntimeError(f"Updated job posting cannot be reloaded: {job_id}")
        return posting


def _posting_identity_key(posting: JobPosting) -> str:
    external_id = posting.external_id.strip()
    if external_id:
        return f"external:{external_id}"

    source_url = _normalize_source_url(posting.source_url)
    if source_url:
        return f"url:{source_url}"

    content = "\n".join(
        _normalize_fingerprint_text(value)
        for value in (
            posting.title,
            posting.company_name,
            posting.location,
            posting.description,
        )
    )
    return f"content:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"


def _normalize_source_url(value: str) -> str:
    value = value.strip()
    if not value:
        return ""

    parsed = urlsplit(value)
    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()
    if not scheme or not hostname:
        return value.rstrip("/")

    netloc = hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"

    query = urlencode(
        sorted(
            (key, query_value)
            for key, query_value in parse_qsl(parsed.query, keep_blank_values=True)
            if not key.lower().startswith("utm_") and key.lower() not in _TRACKING_QUERY_KEYS
        )
    )
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((scheme, netloc, path, query, ""))


def _normalize_fingerprint_text(value: str) -> str:
    return " ".join(value.casefold().split())
