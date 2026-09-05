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
from career_agent.storage.schema import apply_schema


AvailabilityStatus = Literal["active", "closed", "unknown"]
"""Whether the posting is still live **on the platform**."""

PursuitStatus = Literal["open", "dismissed"]
"""Whether the user is still considering this posting.

Deliberately separate from ``availability_status``: one is the employer's
state, the other is the user's decision, and they move for unrelated reasons.
A live posting the user ruled out and a closed posting they still want to
remember are both real, and one field cannot say both.

Only *exclusion* is stated. Every saved job is under consideration until it is
either applied to or dismissed, because that is what a saved job already means —
requiring an "I might apply" mark would ask the user to re-state something they
said by saving it. What the derivation cannot know is when they stopped
considering one, so that is the single thing they have to tell us. Without it
the list only grows, which is how a shortlist stops being read at all.
"""


class JDAnalysisPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    job_summary: str
    responsibilities: tuple[str, ...] = ()
    required_skills: tuple[str, ...] = ()
    preferred_qualifications: tuple[str, ...] = ()
    clarification_questions: tuple[str, ...] = ()


class StoredJDAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    job_posting_id: str
    jd_snapshot_id: str
    analyzer_version: str
    analysis: JDAnalysisPayload
    created_at: datetime


class StoredJobRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    posting: JobPosting
    snapshot: JDSnapshot
    city: str | None = None
    salary: str | None = None
    availability_status: AvailabilityStatus
    pursuit_status: PursuitStatus
    last_checked_at: datetime
    closed_at: datetime | None = None
    analysis: StoredJDAnalysis | None = None


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
    pursuit_status: PursuitStatus = "open"
    captured_at: datetime
    last_checked_at: datetime


class JobPostingRepository(Protocol):
    def save_captured_detail(
        self,
        *,
        user_id: str,
        detail: JobDetail,
    ) -> StoredJobRecord: ...

    def save_detail(
        self,
        *,
        user_id: str,
        run_id: str,
        result_ref: str,
        selection_index: int,
        detail: JobDetail,
    ) -> StoredJobRecord: ...

    def list_jobs(self, *, user_id: str, limit: int = 20, include_dismissed: bool) -> tuple[StoredJobSummary, ...]: ...

    def set_pursuit_status(self, *, user_id: str, job_posting_id: str, status: PursuitStatus) -> bool: ...

    def delete_job(self, *, user_id: str, job_posting_id: str) -> bool: ...

    def count_jobs(self, *, user_id: str, include_dismissed: bool) -> int: ...

    def stale_open_job_stats(
        self,
        *,
        user_id: str,
        checked_before: datetime,
        excluded_job_posting_ids: frozenset[str] = frozenset(),
    ) -> tuple[int, datetime | None]: ...

    def search_saved_jobs(self, *, user_id: str, query: str, limit: int = 20, include_dismissed: bool) -> tuple[StoredJobSummary, ...]: ...

    def get_job(self, *, user_id: str, job_posting_id: str) -> StoredJobRecord | None: ...

    def get_snapshot(self, *, user_id: str, jd_snapshot_id: str) -> JDSnapshot | None: ...

    def get_for_run(self, *, user_id: str, run_id: str, selection_index: int) -> StoredJobRecord | None: ...

    def save_analysis(
        self,
        *,
        user_id: str,
        jd_snapshot_id: str,
        analyzer_version: str,
        analysis: JDAnalysisPayload,
    ) -> StoredJDAnalysis: ...

    def get_latest_analysis(
        self,
        *,
        user_id: str,
        job_posting_id: str,
        analyzer_version: str | None = None,
    ) -> StoredJDAnalysis | None: ...

    def mark_availability(self, *, user_id: str, job_posting_id: str, status: AvailabilityStatus, checked_at: datetime | None = None) -> bool: ...

    def find_by_source(self, *, user_id: str, source_name: str, source_job_id: str | None, source_url: str | None) -> str | None: ...


class SQLiteJobPostingRepository:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            apply_schema(
                connection,
                "job_postings",
                2,
                self._migrate,
                {2: self._upgrade_to_v2},
            )
        os.chmod(self.path, 0o600)

    @staticmethod
    def _upgrade_to_v2(connection: sqlite3.Connection) -> None:
        """Add the user's own decision beside the platform's.

        Existing rows default to ``open``: a job saved before this column
        existed was never ruled out, so treating it as still under
        consideration is the truthful reading, not a convenient one.
        """

        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(job_postings)")
        }
        if "pursuit_status" not in columns:
            connection.execute(
                "ALTER TABLE job_postings ADD COLUMN pursuit_status TEXT NOT NULL "
                "DEFAULT 'open' CHECK(pursuit_status IN ('open', 'dismissed'))"
            )
        if "dismissed_at" not in columns:
            connection.execute("ALTER TABLE job_postings ADD COLUMN dismissed_at TEXT")

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
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
                pursuit_status TEXT NOT NULL DEFAULT 'open' CHECK(pursuit_status IN ('open', 'dismissed')),
                dismissed_at TEXT,
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
                jd_snapshot_id TEXT,
                PRIMARY KEY(user_id, run_id, result_ref),
                UNIQUE(user_id, run_id, selection_index),
                FOREIGN KEY(job_posting_id) REFERENCES job_postings(id),
                FOREIGN KEY(jd_snapshot_id) REFERENCES jd_snapshots(id)
            )
            """
        )
        run_link_columns = {row[1] for row in connection.execute("PRAGMA table_info(job_run_links)")}
        if "jd_snapshot_id" not in run_link_columns:
            connection.execute("ALTER TABLE job_run_links ADD COLUMN jd_snapshot_id TEXT REFERENCES jd_snapshots(id)")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS jd_analyses (
                id TEXT PRIMARY KEY,
                jd_snapshot_id TEXT NOT NULL,
                analyzer_version TEXT NOT NULL,
                analysis_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(jd_snapshot_id, analyzer_version),
                FOREIGN KEY(jd_snapshot_id) REFERENCES jd_snapshots(id)
            )
            """
        )
        connection.execute("CREATE INDEX IF NOT EXISTS jd_analyses_snapshot_created_idx ON jd_analyses(jd_snapshot_id, created_at DESC)")
        connection.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS job_posting_fts USING fts5(job_posting_id UNINDEXED, user_id UNINDEXED, title, company_name, city, jd_content, tokenize='unicode61')"
        )

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
        return self._persist_detail(
            user_id=user_id,
            detail=detail,
            run_link=(run_id, result_ref, selection_index),
        )

    def save_captured_detail(
        self,
        *,
        user_id: str,
        detail: JobDetail,
    ) -> StoredJobRecord:
        if not user_id:
            raise ValueError("user_id is required")
        return self._persist_detail(user_id=user_id, detail=detail, run_link=None)

    def _persist_detail(
        self,
        *,
        user_id: str,
        detail: JobDetail,
        run_link: tuple[str, str, int] | None,
    ) -> StoredJobRecord:
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
            if run_link is not None:
                run_id, result_ref, selection_index = run_link
                connection.execute(
                    "INSERT INTO job_run_links(user_id, run_id, result_ref, selection_index, job_posting_id, jd_snapshot_id) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(user_id, run_id, result_ref) DO UPDATE SET selection_index=excluded.selection_index, job_posting_id=excluded.job_posting_id, jd_snapshot_id=excluded.jd_snapshot_id",
                    (user_id, run_id, result_ref, selection_index, posting_id, snapshot.id),
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
            pursuit_status="open",
            last_checked_at=now,
        )

    def set_pursuit_status(
        self, *, user_id: str, job_posting_id: str, status: PursuitStatus
    ) -> bool:
        """Record that the user is, or is no longer, considering this posting.

        Reversible on purpose, and the row is never deleted. A dismissed job
        stays queryable so "why is this not in my list" has an answer, and a
        misclick costs one click rather than a re-capture. Same posture as a
        settled confirmation: no longer shown, still there.
        """

        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE job_postings SET pursuit_status = ?, dismissed_at = ? "
                "WHERE id = ? AND user_id = ? AND pursuit_status != ?",
                (
                    status,
                    datetime.now(timezone.utc).isoformat()
                    if status == "dismissed"
                    else None,
                    job_posting_id,
                    user_id,
                    status,
                ),
            )
            return cursor.rowcount == 1

    def delete_job(self, *, user_id: str, job_posting_id: str) -> bool:
        """Permanently remove one owned posting and its stored JD tree.

        Ignore is deliberately reversible; this is deliberately not.  The
        ownership check and every dependent delete share one transaction so a
        wrong user cannot touch the row and an interrupted deletion cannot
        leave half a JD behind.  Cross-store records are checked by the
        workspace service before this method is called.
        """

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            owned = connection.execute(
                "SELECT 1 FROM job_postings WHERE id = ? AND user_id = ?",
                (job_posting_id, user_id),
            ).fetchone()
            if owned is None:
                return False
            snapshot_ids = tuple(
                row[0]
                for row in connection.execute(
                    "SELECT id FROM jd_snapshots WHERE job_posting_id = ?",
                    (job_posting_id,),
                ).fetchall()
            )
            if snapshot_ids:
                placeholders = ",".join("?" for _ in snapshot_ids)
                connection.execute(
                    f"DELETE FROM jd_analyses WHERE jd_snapshot_id IN ({placeholders})",
                    snapshot_ids,
                )
            connection.execute(
                "DELETE FROM job_run_links WHERE job_posting_id = ?",
                (job_posting_id,),
            )
            connection.execute(
                "DELETE FROM job_posting_fts WHERE job_posting_id = ?",
                (job_posting_id,),
            )
            connection.execute(
                "DELETE FROM jd_snapshots WHERE job_posting_id = ?",
                (job_posting_id,),
            )
            cursor = connection.execute(
                "DELETE FROM job_postings WHERE id = ? AND user_id = ?",
                (job_posting_id, user_id),
            )
            return cursor.rowcount == 1

    def list_jobs(
        self,
        *,
        user_id: str,
        limit: int = 20,
        include_dismissed: bool,
    ) -> tuple[StoredJobSummary, ...]:
        """The shortlist, and every caller says whether ignored jobs belong in it.

        Required rather than defaulted, because the alternative was tried and
        failed here: the filter was added to this one method and four other
        read paths kept returning ignored jobs — the agent's own search, the
        dashboard count, the CLI listing. A default would have hidden all four,
        since each looked correct in isolation.

        Same device as ``pending_for_conversation``'s policy view: the answer
        is occasionally "yes, show them", and the way to keep that from
        happening by accident is to make forgetting a ``TypeError``.
        """

        self._validate_limit(limit)
        clause = "" if include_dismissed else " AND p.pursuit_status = 'open'"
        with self._connect() as connection:
            rows = connection.execute(
                self._SUMMARY_SELECT
                + f" WHERE p.user_id = ?{clause}"
                + " ORDER BY p.last_checked_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return tuple(self._summary_from_row(row) for row in rows)

    def count_jobs(self, *, user_id: str, include_dismissed: bool) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM job_postings WHERE user_id = ?"
                + ("" if include_dismissed else " AND pursuit_status = 'open'"),
                (user_id,),
            ).fetchone()
        return int(row[0])

    def stale_open_job_stats(
        self,
        *,
        user_id: str,
        checked_before: datetime,
        excluded_job_posting_ids: frozenset[str] = frozenset(),
    ) -> tuple[int, datetime | None]:
        """Summarize every stale shortlist row without a display-list cap.

        ``list_jobs`` is deliberately bounded and newest-first for rendering.
        Reusing it for a library-wide condition hides exactly the oldest rows
        once the library grows past that bound. This query is therefore an
        aggregate source, not another presentation path.

        Applied jobs are excluded by identity supplied by the application
        store. An application is already a stronger statement than "still
        deciding whether to apply", and Action Center has its own follow-up
        lifecycle for it.
        """

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, last_checked_at FROM job_postings "
                "WHERE user_id = ? AND pursuit_status = 'open' "
                "AND julianday(last_checked_at) < julianday(?)",
                (user_id, checked_before.isoformat()),
            ).fetchall()
        eligible = [
            datetime.fromisoformat(row[1])
            for row in rows
            if row[0] not in excluded_job_posting_ids
        ]
        return (len(eligible), min(eligible) if eligible else None)

    def search_saved_jobs(self, *, user_id: str, query: str, limit: int = 20, include_dismissed: bool) -> tuple[StoredJobSummary, ...]:
        self._validate_limit(limit)
        normalized_query = query.strip()
        if not normalized_query:
            return self.list_jobs(
                user_id=user_id, limit=limit, include_dismissed=include_dismissed
            )
        like = f"%{normalized_query}%"
        tokens = tuple(dict.fromkeys(re.findall(r"[\w+#.-]+", normalized_query, flags=re.UNICODE)))
        fts_query = " OR ".join(json.dumps(token, ensure_ascii=False) for token in tokens)
        # Applied to both branches. An ignored job must not come back through
        # the agent's own search after the user removed it from the library —
        # that is precisely the job the removal was about.
        ignored_clause = "" if include_dismissed else " AND p.pursuit_status = 'open'"
        if fts_query:
            sql = (
                "WITH fts_hits AS ("
                "SELECT job_posting_id, bm25(job_posting_fts, 0.0, 0.0, 8.0, 6.0, 3.0, 1.0) AS relevance "
                "FROM job_posting_fts WHERE user_id = ? AND job_posting_fts MATCH ?"
                ") "
                + self._SUMMARY_SELECT
                + " LEFT JOIN fts_hits f ON f.job_posting_id = p.id"
                + " WHERE p.user_id = ? AND (p.title LIKE ? OR p.company_name LIKE ? OR COALESCE(p.city, '') LIKE ? OR s.content LIKE ? OR f.job_posting_id IS NOT NULL)"
                + ignored_clause
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
                + ignored_clause
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
            if row is None:
                return None
            record = self._record_from_row(row)
            analysis = self._analysis_for_snapshot(connection, user_id=user_id, jd_snapshot_id=record.snapshot.id)
        return record.model_copy(update={"analysis": analysis})

    def get_snapshot(
        self, *, user_id: str, jd_snapshot_id: str
    ) -> JDSnapshot | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT s.job_posting_id, s.id, s.version, s.content,
                       s.content_hash, s.captured_at, s.provenance_json,
                       s.normalizer_version
                FROM jd_snapshots AS s
                JOIN job_postings AS p ON p.id = s.job_posting_id
                WHERE p.user_id = ? AND s.id = ?
                """,
                (user_id, jd_snapshot_id),
            ).fetchone()
        return self._snapshot_from_row(row[0], row[1:]) if row else None

    def get_for_run(self, *, user_id: str, run_id: str, selection_index: int) -> StoredJobRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                self._RUN_RECORD_SELECT
                + " WHERE p.user_id = ? AND l.user_id = ? AND l.run_id = ? AND l.selection_index = ?",
                (user_id, user_id, run_id, selection_index),
            ).fetchone()
            if row is None:
                return None
            record = self._record_from_row(row)
            analysis = self._analysis_for_snapshot(connection, user_id=user_id, jd_snapshot_id=record.snapshot.id)
        return record.model_copy(update={"analysis": analysis})

    def save_analysis(
        self,
        *,
        user_id: str,
        jd_snapshot_id: str,
        analyzer_version: str,
        analysis: JDAnalysisPayload,
    ) -> StoredJDAnalysis:
        if not user_id or not jd_snapshot_id or not analyzer_version.strip():
            raise ValueError("user_id, jd_snapshot_id, and analyzer_version are required")
        payload = JDAnalysisPayload.model_validate(analysis)
        created_at = datetime.now(timezone.utc)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            owner = connection.execute(
                "SELECT s.job_posting_id FROM jd_snapshots s JOIN job_postings p ON p.id = s.job_posting_id WHERE s.id = ? AND p.user_id = ?",
                (jd_snapshot_id, user_id),
            ).fetchone()
            if owner is None:
                raise ValueError("JD snapshot not found for this user")
            existing = connection.execute(
                "SELECT id, jd_snapshot_id, analyzer_version, analysis_json, created_at FROM jd_analyses WHERE jd_snapshot_id = ? AND analyzer_version = ?",
                (jd_snapshot_id, analyzer_version),
            ).fetchone()
            if existing is not None:
                return self._analysis_from_row(owner[0], existing)
            stored = StoredJDAnalysis(
                id=new_id("jd_analysis"),
                job_posting_id=owner[0],
                jd_snapshot_id=jd_snapshot_id,
                analyzer_version=analyzer_version,
                analysis=payload,
                created_at=created_at,
            )
            connection.execute(
                "INSERT INTO jd_analyses(id, jd_snapshot_id, analyzer_version, analysis_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (stored.id, jd_snapshot_id, analyzer_version, payload.model_dump_json(), created_at.isoformat()),
            )
        os.chmod(self.path, 0o600)
        return stored

    def get_latest_analysis(
        self,
        *,
        user_id: str,
        job_posting_id: str,
        analyzer_version: str | None = None,
    ) -> StoredJDAnalysis | None:
        sql = (
            "SELECT a.id, a.jd_snapshot_id, a.analyzer_version, a.analysis_json, a.created_at "
            "FROM job_postings p JOIN jd_analyses a ON a.jd_snapshot_id = p.latest_snapshot_id "
            "WHERE p.user_id = ? AND p.id = ?"
        )
        params: tuple[object, ...] = (user_id, job_posting_id)
        if analyzer_version is not None:
            sql += " AND a.analyzer_version = ?"
            params = (*params, analyzer_version)
        sql += " ORDER BY a.created_at DESC, a.rowid DESC LIMIT 1"
        with self._connect() as connection:
            row = connection.execute(sql, params).fetchone()
        return self._analysis_from_row(job_posting_id, row) if row else None

    def find_by_source(
        self,
        *,
        user_id: str,
        source_name: str,
        source_job_id: str | None,
        source_url: str | None,
    ) -> str | None:
        """The saved job a page belongs to, resolved the way capture resolves it.

        The extension knows a URL, never the internal id, so anything it reports
        about a posting has to be matched by source identity — and by the *same*
        identity capture writes, or a report would land on nothing while the job
        sits in the library under a different key.
        """

        if source_job_id:
            identity = f"source_job_id:{source_job_id}"
        elif source_url:
            identity = "source_url:" + hashlib.sha256(
                source_url.encode("utf-8")
            ).hexdigest()
        else:
            return None
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id FROM job_postings WHERE user_id = ? AND source_name = ? "
                "AND source_identity = ?",
                (user_id, source_name, identity),
            ).fetchone()
        return row[0] if row else None

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
        snapshot = cls._snapshot_from_row(posting.id, row[18:])
        return StoredJobRecord(
            posting=posting,
            snapshot=snapshot,
            city=row[13],
            salary=row[14],
            availability_status=posting.external_status,
            pursuit_status=row[17],
            last_checked_at=row[15],
            closed_at=row[16],
        )

    @staticmethod
    def _summary_from_row(row: tuple) -> StoredJobSummary:
        return StoredJobSummary(
            job_posting_id=row[0], title=row[1], company_name=row[2], city=row[3], salary=row[4],
            source_name=row[5], source_url=row[6], availability_status=row[7],
            pursuit_status=row[8], captured_at=row[9], last_checked_at=row[10],
        )

    @classmethod
    def _analysis_for_snapshot(
        cls,
        connection: sqlite3.Connection,
        *,
        user_id: str,
        jd_snapshot_id: str,
    ) -> StoredJDAnalysis | None:
        row = connection.execute(
            "SELECT a.id, a.jd_snapshot_id, a.analyzer_version, a.analysis_json, a.created_at "
            "FROM jd_analyses a JOIN jd_snapshots s ON s.id = a.jd_snapshot_id "
            "JOIN job_postings p ON p.id = s.job_posting_id "
            "WHERE p.user_id = ? AND a.jd_snapshot_id = ? "
            "ORDER BY a.created_at DESC, a.rowid DESC LIMIT 1",
            (user_id, jd_snapshot_id),
        ).fetchone()
        if row is None:
            return None
        posting_id = connection.execute(
            "SELECT job_posting_id FROM jd_snapshots WHERE id = ?",
            (jd_snapshot_id,),
        ).fetchone()[0]
        return cls._analysis_from_row(posting_id, row)

    @staticmethod
    def _analysis_from_row(job_posting_id: str, row: tuple) -> StoredJDAnalysis:
        return StoredJDAnalysis(
            id=row[0],
            job_posting_id=job_posting_id,
            jd_snapshot_id=row[1],
            analyzer_version=row[2],
            analysis=JDAnalysisPayload.model_validate_json(row[3]),
            created_at=row[4],
        )

    _SUMMARY_SELECT = """
        SELECT p.id, p.title, p.company_name, p.city, p.salary, p.source_name,
               p.source_url, p.availability_status, p.pursuit_status,
               s.captured_at, p.last_checked_at
        FROM job_postings p
        JOIN jd_snapshots s ON s.id = p.latest_snapshot_id
    """
    _RECORD_SELECT = """
        SELECT p.id, p.user_id, p.source_name, p.source_job_id, p.source_url,
               p.title, p.company_name, p.availability_status, p.persisted_at,
               p.last_seen_at, p.latest_snapshot_id, p.company_title_fingerprint,
               p.content_fingerprint, p.city, p.salary, p.last_checked_at, p.closed_at,
               p.pursuit_status,
               s.id, s.version, s.content, s.content_hash,
               s.captured_at, s.provenance_json, s.normalizer_version
        FROM job_postings p
        JOIN jd_snapshots s ON s.id = p.latest_snapshot_id
    """
    _RUN_RECORD_SELECT = """
        SELECT p.id, p.user_id, p.source_name, p.source_job_id, p.source_url,
               p.title, p.company_name, p.availability_status, p.persisted_at,
               p.last_seen_at, p.latest_snapshot_id, p.company_title_fingerprint,
               p.content_fingerprint, p.city, p.salary, p.last_checked_at, p.closed_at,
               p.pursuit_status,
               s.id, s.version, s.content, s.content_hash,
               s.captured_at, s.provenance_json, s.normalizer_version
        FROM job_run_links l
        JOIN job_postings p ON p.id = l.job_posting_id
        JOIN jd_snapshots s ON s.id = COALESCE(l.jd_snapshot_id, p.latest_snapshot_id)
    """

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection
