from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from career_agent.domain.job_discovery import (
    JDSnapshot,
    JobDetail,
    JobPosting,
    company_title_fingerprint,
    content_fingerprint,
    jd_content_hash,
    new_id,
    normalize_jd,
    validate_job_detail,
)


AvailabilityStatus = Literal["active", "closed", "unknown"]


class StoredJobRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    posting: JobPosting
    snapshot: JDSnapshot
    city: str | None = None
    salary: str | None = None
    availability_status: AvailabilityStatus
    last_checked_at: datetime
    closed_at: datetime | None = None


class StoredJobSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    job_posting_id: str
    title: str
    company_name: str
    city: str | None = None
    salary: str | None = None
    source_name: str
    source_url: str | None = None
    availability_status: AvailabilityStatus
    captured_at: datetime
    last_checked_at: datetime


class JobPostingRepository(Protocol):
    def save_detail(
        self,
        *,
        user_id: str,
        run_id: str,
        result_ref: str,
        selection_index: int,
        detail: JobDetail,
    ) -> StoredJobRecord: ...

    def list_jobs(self, *, user_id: str, limit: int = 20) -> tuple[StoredJobSummary, ...]: ...

    def search_saved_jobs(self, *, user_id: str, query: str, limit: int = 20) -> tuple[StoredJobSummary, ...]: ...

    def get_job(self, *, user_id: str, job_posting_id: str) -> StoredJobRecord | None: ...

    def get_for_run(self, *, user_id: str, run_id: str, selection_index: int) -> StoredJobRecord | None: ...

    def mark_availability(self, *, user_id: str, job_posting_id: str, status: AvailabilityStatus, checked_at: datetime | None = None) -> bool: ...


class SQLiteJobPostingRepository:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS job_postings (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    source_identity TEXT NOT NULL,
                    source_job_id TEXT,
                    source_url TEXT,
                    title TEXT NOT NULL,
                    company_name TEXT NOT NULL,
                    city TEXT,
                    salary TEXT,
                    availability_status TEXT NOT NULL CHECK(availability_status IN ('active', 'closed', 'unknown')),
                    persisted_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    last_checked_at TEXT NOT NULL,
                    closed_at TEXT,
                    latest_snapshot_id TEXT NOT NULL,
                    company_title_fingerprint TEXT NOT NULL,
                    content_fingerprint TEXT NOT NULL,
                    UNIQUE(user_id, source_name, source_identity)
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS job_postings_user_recent_idx ON job_postings(user_id, last_checked_at DESC)")
            connection.execute("CREATE INDEX IF NOT EXISTS job_postings_user_status_idx ON job_postings(user_id, availability_status)")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jd_snapshots (
                    id TEXT PRIMARY KEY,
                    job_posting_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    captured_at TEXT NOT NULL,
                    provenance_json TEXT NOT NULL,
                    normalizer_version TEXT NOT NULL,
                    UNIQUE(job_posting_id, content_hash),
                    UNIQUE(job_posting_id, version),
                    FOREIGN KEY(job_posting_id) REFERENCES job_postings(id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS job_run_links (
                    user_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    result_ref TEXT NOT NULL,
                    selection_index INTEGER NOT NULL,
                    job_posting_id TEXT NOT NULL,
                    PRIMARY KEY(user_id, run_id, result_ref),
                    UNIQUE(user_id, run_id, selection_index),
                    FOREIGN KEY(job_posting_id) REFERENCES job_postings(id)
                )
                """
            )
            connection.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS job_posting_fts USING fts5(job_posting_id UNINDEXED, user_id UNINDEXED, title, company_name, city, jd_content, tokenize='unicode61')"
            )
        os.chmod(self.path, 0o600)

    def save_detail(
        self,
        *,
        user_id: str,
        run_id: str,
        result_ref: str,
        selection_index: int,
        detail: JobDetail,
    ) -> StoredJobRecord:
        if not user_id or not run_id or not result_ref or selection_index < 1:
            raise ValueError("user_id, run_id, result_ref, and a positive selection_index are required")
        normalized = validate_job_detail(detail)
        now = datetime.now(timezone.utc)
        source_identity = self._source_identity(detail, normalized)
        content_hash = jd_content_hash(normalized)

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT id, persisted_at FROM job_postings WHERE user_id = ? AND source_name = ? AND source_identity = ?",
                (user_id, detail.source_name, source_identity),
            ).fetchone()
            posting_id = row[0] if row else new_id("job")
            persisted_at = datetime.fromisoformat(row[1]) if row else now
            snapshot_row = connection.execute(
                "SELECT id, version, content, content_hash, captured_at, provenance_json, normalizer_version FROM jd_snapshots WHERE job_posting_id = ? AND content_hash = ?",
                (posting_id, content_hash),
            ).fetchone()
            if snapshot_row is None:
                version = connection.execute(
                    "SELECT COALESCE(MAX(version), 0) + 1 FROM jd_snapshots WHERE job_posting_id = ?",
                    (posting_id,),
                ).fetchone()[0]
                snapshot = JDSnapshot(
                    id=new_id("jd"),
                    job_posting_id=posting_id,
                    version=version,
                    content=normalized,
                    content_hash=content_hash,
                    captured_at=detail.captured_at,
                    provenance=detail.provenance,
                    normalizer_version="jd-text-v1",
                )
            else:
                snapshot = self._snapshot_from_row(posting_id, snapshot_row)

            posting = JobPosting(
                id=posting_id,
                user_id=user_id,
                title=detail.title.strip(),
                company_name=detail.company_name.strip(),
                source_name=detail.source_name,
                source_job_id=detail.source_job_id,
                source_url=detail.source_url,
                external_status="active",
                persisted_at=persisted_at,
                last_seen_at=now,
                latest_snapshot_id=snapshot.id,
                company_title_fingerprint=company_title_fingerprint(detail.title, detail.company_name),
                content_fingerprint=content_fingerprint(detail.title, detail.company_name, normalized),
            )
            connection.execute(
                """
                INSERT INTO job_postings(
                    id, user_id, source_name, source_identity, source_job_id, source_url,
                    title, company_name, city, salary, availability_status, persisted_at,
                    last_seen_at, last_checked_at, closed_at, latest_snapshot_id,
                    company_title_fingerprint, content_fingerprint
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, NULL, ?, ?, ?)
                ON CONFLICT(user_id, source_name, source_identity) DO UPDATE SET
                    source_job_id=excluded.source_job_id,
                    source_url=excluded.source_url,
                    title=excluded.title,
                    company_name=excluded.company_name,
                    city=excluded.city,
                    salary=excluded.salary,
                    availability_status='active',
                    last_seen_at=excluded.last_seen_at,
                    last_checked_at=excluded.last_checked_at,
                    closed_at=NULL,
                    latest_snapshot_id=excluded.latest_snapshot_id,
                    company_title_fingerprint=excluded.company_title_fingerprint,
                    content_fingerprint=excluded.content_fingerprint
                """,
                (
                    posting.id, user_id, detail.source_name, source_identity, detail.source_job_id,
                    detail.source_url, posting.title, posting.company_name, detail.city, detail.salary,
                    posting.persisted_at.isoformat(), now.isoformat(), now.isoformat(), snapshot.id,
                    posting.company_title_fingerprint, posting.content_fingerprint,
                ),
            )
            if snapshot_row is None:
                connection.execute(
                    "INSERT INTO jd_snapshots(id, job_posting_id, version, content, content_hash, captured_at, provenance_json, normalizer_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        snapshot.id, posting_id, snapshot.version, snapshot.content, snapshot.content_hash,
                        snapshot.captured_at.isoformat(), snapshot.provenance.model_dump_json(), snapshot.normalizer_version,
                    ),
                )
            connection.execute(
                "INSERT INTO job_run_links(user_id, run_id, result_ref, selection_index, job_posting_id) VALUES (?, ?, ?, ?, ?) ON CONFLICT(user_id, run_id, result_ref) DO UPDATE SET selection_index=excluded.selection_index, job_posting_id=excluded.job_posting_id",
                (user_id, run_id, result_ref, selection_index, posting_id),
            )
            connection.execute("DELETE FROM job_posting_fts WHERE job_posting_id = ?", (posting_id,))
            connection.execute(
                "INSERT INTO job_posting_fts(job_posting_id, user_id, title, company_name, city, jd_content) VALUES (?, ?, ?, ?, ?, ?)",
                (posting_id, user_id, posting.title, posting.company_name, detail.city or "", normalized),
            )
        os.chmod(self.path, 0o600)
        return StoredJobRecord(
            posting=posting,
            snapshot=snapshot,
            city=detail.city,
            salary=detail.salary,
            availability_status="active",
            last_checked_at=now,
        )

    def list_jobs(self, *, user_id: str, limit: int = 20) -> tuple[StoredJobSummary, ...]:
        self._validate_limit(limit)
        with self._connect() as connection:
            rows = connection.execute(self._SUMMARY_SELECT + " WHERE p.user_id = ? ORDER BY p.last_checked_at DESC LIMIT ?", (user_id, limit)).fetchall()
        return tuple(self._summary_from_row(row) for row in rows)

    def search_saved_jobs(self, *, user_id: str, query: str, limit: int = 20) -> tuple[StoredJobSummary, ...]:
        self._validate_limit(limit)
        normalized_query = query.strip()
        if not normalized_query:
            return self.list_jobs(user_id=user_id, limit=limit)
        like = f"%{normalized_query}%"
        tokens = tuple(dict.fromkeys(re.findall(r"[\w+#.-]+", normalized_query, flags=re.UNICODE)))
        fts_query = " OR ".join(json.dumps(token, ensure_ascii=False) for token in tokens)
        if fts_query:
            sql = (
                "WITH fts_hits AS ("
                "SELECT job_posting_id, bm25(job_posting_fts, 0.0, 0.0, 8.0, 6.0, 3.0, 1.0) AS relevance "
                "FROM job_posting_fts WHERE user_id = ? AND job_posting_fts MATCH ?"
                ") "
                + self._SUMMARY_SELECT
                + " LEFT JOIN fts_hits f ON f.job_posting_id = p.id"
                + " WHERE p.user_id = ? AND (p.title LIKE ? OR p.company_name LIKE ? OR COALESCE(p.city, '') LIKE ? OR s.content LIKE ? OR f.job_posting_id IS NOT NULL)"
                + " ORDER BY CASE WHEN lower(p.title) = lower(?) THEN 0 WHEN p.title LIKE ? THEN 1 WHEN p.company_name LIKE ? THEN 2 WHEN COALESCE(p.city, '') LIKE ? THEN 3 ELSE 4 END,"
                + " COALESCE(f.relevance, 1000000.0) ASC, p.last_checked_at DESC LIMIT ?"
            )
            params: tuple[object, ...] = (
                user_id, fts_query, user_id, like, like, like, like,
                normalized_query, like, like, like, limit,
            )
        else:
            sql = (
                self._SUMMARY_SELECT
                + " WHERE p.user_id = ? AND (p.title LIKE ? OR p.company_name LIKE ? OR COALESCE(p.city, '') LIKE ? OR s.content LIKE ?)"
                + " ORDER BY p.last_checked_at DESC LIMIT ?"
            )
            params = (user_id, like, like, like, like, limit)
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return tuple(self._summary_from_row(row) for row in rows)

    def get_job(self, *, user_id: str, job_posting_id: str) -> StoredJobRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                self._RECORD_SELECT + " WHERE p.user_id = ? AND p.id = ?",
                (user_id, job_posting_id),
            ).fetchone()
        return self._record_from_row(row) if row else None

    def get_for_run(self, *, user_id: str, run_id: str, selection_index: int) -> StoredJobRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                self._RECORD_SELECT
                + " JOIN job_run_links l ON l.job_posting_id = p.id WHERE p.user_id = ? AND l.user_id = ? AND l.run_id = ? AND l.selection_index = ?",
                (user_id, user_id, run_id, selection_index),
            ).fetchone()
        return self._record_from_row(row) if row else None

    def mark_availability(self, *, user_id: str, job_posting_id: str, status: AvailabilityStatus, checked_at: datetime | None = None) -> bool:
        if status not in {"active", "closed", "unknown"}:
            raise ValueError("Unsupported availability status")
        timestamp = checked_at or datetime.now(timezone.utc)
        closed_at = timestamp.isoformat() if status == "closed" else None
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE job_postings SET availability_status = ?, last_checked_at = ?, closed_at = ? WHERE user_id = ? AND id = ?",
                (status, timestamp.isoformat(), closed_at, user_id, job_posting_id),
            )
        return cursor.rowcount == 1

    @staticmethod
    def _validate_limit(limit: int) -> None:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")

    @staticmethod
    def _source_identity(detail: JobDetail, normalized: str) -> str:
        if detail.source_job_id:
            return f"source_job_id:{detail.source_job_id}"
        if detail.source_url:
            return "source_url:" + hashlib.sha256(detail.source_url.encode("utf-8")).hexdigest()
        return "content:" + content_fingerprint(detail.title, detail.company_name, normalized)

    @staticmethod
    def _snapshot_from_row(posting_id: str, row: tuple) -> JDSnapshot:
        from career_agent.domain.job_discovery import Provenance

        return JDSnapshot(
            id=row[0], job_posting_id=posting_id, version=row[1], content=row[2], content_hash=row[3],
            captured_at=row[4], provenance=Provenance.model_validate_json(row[5]), normalizer_version=row[6],
        )

    @staticmethod
    def _posting_from_row(row: tuple) -> JobPosting:
        return JobPosting(
            id=row[0], user_id=row[1], source_name=row[2], source_job_id=row[3], source_url=row[4],
            title=row[5], company_name=row[6], external_status=row[7], persisted_at=row[8],
            last_seen_at=row[9], latest_snapshot_id=row[10], company_title_fingerprint=row[11], content_fingerprint=row[12],
        )

    @classmethod
    def _record_from_row(cls, row: tuple) -> StoredJobRecord:
        posting = cls._posting_from_row(row[:13])
        snapshot = cls._snapshot_from_row(posting.id, row[17:])
        return StoredJobRecord(
            posting=posting,
            snapshot=snapshot,
            city=row[13],
            salary=row[14],
            availability_status=posting.external_status,
            last_checked_at=row[15],
            closed_at=row[16],
        )

    @staticmethod
    def _summary_from_row(row: tuple) -> StoredJobSummary:
        return StoredJobSummary(
            job_posting_id=row[0], title=row[1], company_name=row[2], city=row[3], salary=row[4],
            source_name=row[5], source_url=row[6], availability_status=row[7], captured_at=row[8], last_checked_at=row[9],
        )

    _SUMMARY_SELECT = """
        SELECT p.id, p.title, p.company_name, p.city, p.salary, p.source_name,
               p.source_url, p.availability_status, s.captured_at, p.last_checked_at
        FROM job_postings p
        JOIN jd_snapshots s ON s.id = p.latest_snapshot_id
    """
    _RECORD_SELECT = """
        SELECT p.id, p.user_id, p.source_name, p.source_job_id, p.source_url,
               p.title, p.company_name, p.availability_status, p.persisted_at,
               p.last_seen_at, p.latest_snapshot_id, p.company_title_fingerprint,
               p.content_fingerprint, p.city, p.salary, p.last_checked_at, p.closed_at,
               s.id, s.version, s.content, s.content_hash,
               s.captured_at, s.provenance_json, s.normalizer_version
        FROM job_postings p
        JOIN jd_snapshots s ON s.id = p.latest_snapshot_id
    """

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection
