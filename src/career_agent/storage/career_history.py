from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from typing import Literal
import unicodedata
from uuid import uuid4

from career_agent.domain.career_history import (
    CareerEvidence,
    CareerEvidenceCorrection,
    CareerEvidenceEvent,
    CareerEvidenceTombstone,
    CareerRecord,
    career_evidence_detail_ref,
    career_evidence_lineage_ref,
    career_evidence_scope_key,
    career_evidence_source_ref,
)
from career_agent.agent.resume_analysis_contracts import ResumeAnalysisResult
from career_agent.storage.intent_versions import intent_content_digest
from career_agent.storage.schema import apply_schema


EvidenceOrigin = Literal["resume_extraction", "user_input", "agent_inference"]
EvidenceStatus = Literal["pending", "confirmed", "rejected"]
RecordType = Literal["education", "work", "internship", "project", "certification"]
_HISTORY_CURSOR = re.compile(
    r"^history_(?P<query>[a-f0-9]{8})_(?P<offset>[a-f0-9]{8})$"
)
_MEMORY_CURSOR = re.compile(
    r"^memory_(?P<query>[a-f0-9]{8})_(?P<offset>[a-f0-9]{8})$"
)
_MAX_HISTORY_OFFSET = 10_000


_EVIDENCE_FIELD_NAMES = (
    "id",
    "user_id",
    "career_record_id",
    "claim",
    "origin",
    "verification_status",
    "source_resume_version_id",
    "source_locator",
    "source_quote",
    "source_ref",
    "detail_ref",
    "scope_key",
    "update_id",
    "content_digest",
    "revision",
    "valid_from",
    "supersedes_id",
    "superseded_at",
    "superseded_by",
    "tombstoned_at",
    "tombstoned_by",
    "tombstone_reason",
    "created_at",
    "updated_at",
)
_EVIDENCE_COLUMNS = ", ".join(_EVIDENCE_FIELD_NAMES)


@dataclass(frozen=True)
class RankedCareerEvidence:
    """An FTS hit plus the SQLite bm25() score that produced its order.

    Mixed short/long queries sum scores from their isolated FTS indexes. More
    negative ``bm25_score`` is a better match. The projector normalizes this
    set; items outside it have relevance 0.
    """

    evidence: CareerEvidence
    bm25_score: float


@dataclass(frozen=True)
class CareerEvidenceQueryTerms:
    """Typed term groups shared by current-evidence MATCH and normalization."""

    user_id: str
    latin: tuple[str, ...]
    cjk: tuple[str, ...]
    idf_sum: float

    @property
    def match_tokens(self) -> tuple[str, ...]:
        return (*self.latin, *self.cjk)


@dataclass(frozen=True)
class CareerHistoryImportResult:
    records: tuple[CareerRecord, ...]
    evidence: tuple[CareerEvidence, ...]


class CareerHistoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        with self._connect() as connection:
            apply_schema(
                connection,
                "career_history",
                8,
                self._migrate,
                upgrades={
                    3: self._upgrade_to_v3,
                    4: self._upgrade_to_v4,
                    5: self._upgrade_to_v5,
                    6: self._upgrade_to_v6,
                    7: self._upgrade_to_v7,
                    8: self._upgrade_to_v8,
                },
            )
        os.chmod(self.path, 0o600)

    def create_record(
        self,
        *,
        user_id: str,
        record_type: RecordType,
        title: str,
        organization: str | None = None,
        start_year: int | None = None,
        start_month: int | None = None,
        end_year: int | None = None,
        end_month: int | None = None,
        is_current: bool = False,
    ) -> CareerRecord:
        now = datetime.now(timezone.utc)
        record = CareerRecord(
            id=f"career_record_{uuid4().hex}",
            user_id=user_id,
            record_type=record_type,
            organization=organization,
            title=title,
            start_year=start_year,
            start_month=start_month,
            end_year=end_year,
            end_month=end_month,
            is_current=is_current,
            created_at=now,
            updated_at=now,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO career_records(
                    id, user_id, record_type, organization, title,
                    start_year, start_month, end_year, end_month, is_current,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.user_id,
                    record.record_type,
                    record.organization,
                    record.title,
                    record.start_year,
                    record.start_month,
                    record.end_year,
                    record.end_month,
                    int(record.is_current),
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                ),
            )
        return record

    def get_record(
        self, *, user_id: str, career_record_id: str
    ) -> CareerRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, user_id, record_type, organization, title,
                       start_year, start_month, end_year, end_month, is_current,
                       created_at, updated_at
                FROM career_records
                WHERE id = ? AND user_id = ?
                """,
                (career_record_id, user_id),
            ).fetchone()
        return self._record(row) if row else None

    def list_records(
        self, *, user_id: str, limit: int | None = None
    ) -> tuple[CareerRecord, ...]:
        if limit is not None and limit < 1:
            raise ValueError("record limit must be positive")
        query = """
            SELECT id, user_id, record_type, organization, title,
                   start_year, start_month, end_year, end_month, is_current,
                   created_at, updated_at
            FROM career_records
            WHERE user_id = ?
            ORDER BY is_current DESC,
                     COALESCE(start_year, 0) DESC,
                     COALESCE(start_month, 0) DESC,
                     created_at DESC
        """
        parameters: list[object] = [user_id]
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(self._record(row) for row in rows)

    def count_records(self, *, user_id: str) -> int:
        with self._connect() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM career_records WHERE user_id = ?",
                    (user_id,),
                ).fetchone()[0]
            )

    def create_evidence(
        self,
        *,
        user_id: str,
        career_record_id: str,
        claim: str,
        origin: EvidenceOrigin,
        source_resume_version_id: str | None = None,
        source_locator: str | None = None,
        source_quote: str | None = None,
    ) -> CareerEvidence:
        now = datetime.now(timezone.utc)
        evidence_id = f"career_evidence_{uuid4().hex}"
        evidence = CareerEvidence(
            id=evidence_id,
            user_id=user_id,
            career_record_id=career_record_id,
            claim=claim,
            origin=origin,
            verification_status="pending",
            source_resume_version_id=source_resume_version_id,
            source_locator=source_locator,
            source_quote=source_quote,
            source_ref=career_evidence_source_ref(
                user_id=user_id,
                evidence_id=evidence_id,
                source_resume_version_id=source_resume_version_id,
                source_locator=source_locator,
            ),
            detail_ref=career_evidence_detail_ref(
                user_id=user_id,
                evidence_id=evidence_id,
            ),
            created_at=now,
            updated_at=now,
        )
        actor_type = {
            "resume_extraction": "system",
            "user_input": "user",
            "agent_inference": "agent",
        }[evidence.origin]
        event = CareerEvidenceEvent(
            id=f"career_evidence_event_{uuid4().hex}",
            user_id=evidence.user_id,
            career_evidence_id=evidence.id,
            event_type="created",
            previous_status=None,
            new_status="pending",
            actor_type=actor_type,
            occurred_at=now,
        )

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            record_owner = connection.execute(
                "SELECT user_id FROM career_records WHERE id = ?",
                (evidence.career_record_id,),
            ).fetchone()
            if record_owner is None or record_owner[0] != evidence.user_id:
                raise ValueError("Career record not found.")
            if evidence.source_resume_version_id is not None and not self._resume_version_belongs_to_user(
                connection,
                user_id=evidence.user_id,
                resume_version_id=evidence.source_resume_version_id,
            ):
                raise ValueError("Source resume version not found.")

            connection.execute(
                """
                INSERT INTO career_evidence(
                    id, user_id, career_record_id, claim, short_terms, origin,
                    verification_status, source_resume_version_id,
                    source_locator, source_quote, source_ref, detail_ref,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence.id,
                    evidence.user_id,
                    evidence.career_record_id,
                    evidence.claim,
                    self._short_terms_index_text(evidence.claim),
                    evidence.origin,
                    evidence.verification_status,
                    evidence.source_resume_version_id,
                    evidence.source_locator,
                    evidence.source_quote,
                    evidence.source_ref,
                    evidence.detail_ref,
                    evidence.created_at.isoformat(),
                    evidence.updated_at.isoformat(),
                ),
            )
            self._insert_event(connection, event)
        return evidence

    def get_evidence(
        self, *, user_id: str, career_evidence_id: str
    ) -> CareerEvidence | None:
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT {_EVIDENCE_COLUMNS}
                FROM career_evidence
                WHERE id = ? AND user_id = ? AND tombstoned_at IS NULL
                """,
                (career_evidence_id, user_id),
            ).fetchone()
        return self._evidence(row) if row else None

    def get_evidence_by_source_ref(
        self, *, user_id: str, source_ref: str
    ) -> CareerEvidence | None:
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT {_EVIDENCE_COLUMNS}
                FROM career_evidence
                WHERE user_id = ? AND source_ref = ?
                  AND verification_status = 'confirmed'
                  AND superseded_by IS NULL
                  AND tombstoned_at IS NULL
                """,
                (user_id, source_ref),
            ).fetchone()
        return self._evidence(row) if row else None

    def get_evidence_by_detail_ref(
        self, *, user_id: str, detail_ref: str
    ) -> CareerEvidence | None:
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT {_EVIDENCE_COLUMNS}
                FROM career_evidence
                WHERE user_id = ? AND detail_ref = ?
                  AND verification_status = 'confirmed'
                  AND superseded_by IS NULL
                  AND tombstoned_at IS NULL
                """,
                (user_id, detail_ref),
            ).fetchone()
        return self._evidence(row) if row else None

    def search_historical_evidence(
        self,
        *,
        user_id: str,
        query: str,
        limit: int = 8,
        cursor: str | None = None,
    ) -> tuple[tuple[CareerEvidence, ...], int, str | None]:
        """Search superseded or rolled-back claims through the bounded FTS index."""

        if not 1 <= limit <= 20:
            raise ValueError("historical evidence limit must be between 1 and 20")
        tokens = tuple(
            dict.fromkeys(
                token.casefold()
                for token in re.findall(r"[\w+#.-]{3,}", query.strip(), flags=re.UNICODE)
            )
        )
        if not tokens:
            raise ValueError("historical evidence query needs a term of 3+ characters")
        normalized_query = " ".join(tokens)
        query_digest = hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()[:8]
        offset = 0
        if cursor is not None:
            match = _HISTORY_CURSOR.fullmatch(cursor)
            if match is None or match.group("query") != query_digest:
                raise ValueError("historical evidence cursor does not match this query")
            offset = int(match.group("offset"), 16)
            if offset > _MAX_HISTORY_OFFSET:
                raise ValueError("historical evidence cursor exceeds the safety limit")
        match_query = " OR ".join(
            json.dumps(token, ensure_ascii=False) for token in tokens
        )
        selected_columns = ", ".join(
            f"evidence.{name}" for name in _EVIDENCE_FIELD_NAMES
        )
        with self._connect() as connection:
            total = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM career_evidence_fts AS search
                    JOIN career_evidence AS evidence
                      ON evidence.id = search.evidence_id
                    WHERE search.user_id = ?
                      AND career_evidence_fts MATCH ?
                      AND evidence.verification_status = 'confirmed'
                      AND evidence.tombstoned_at IS NULL
                      AND evidence.superseded_by IS NOT NULL
                    """,
                    (user_id, match_query),
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"""
                SELECT {selected_columns}
                FROM career_evidence_fts AS search
                JOIN career_evidence AS evidence
                  ON evidence.id = search.evidence_id
                WHERE search.user_id = ?
                  AND career_evidence_fts MATCH ?
                  AND evidence.verification_status = 'confirmed'
                  AND evidence.tombstoned_at IS NULL
                  AND evidence.superseded_by IS NOT NULL
                ORDER BY bm25(career_evidence_fts), evidence.created_at DESC
                LIMIT ? OFFSET ?
                """,
                (user_id, match_query, limit, offset),
            ).fetchall()
        next_offset = offset + len(rows)
        next_cursor = (
            f"history_{query_digest}_{next_offset:08x}"
            if next_offset < total and next_offset <= _MAX_HISTORY_OFFSET
            else None
        )
        return tuple(self._evidence(row) for row in rows), total, next_cursor

    def search_current_evidence(
        self,
        *,
        user_id: str,
        query: str,
        limit: int = 8,
        cursor: str | None = None,
    ) -> tuple[tuple[CareerEvidence, ...], int, str | None]:
        """Search active confirmed claims through a user-scoped FTS population."""

        if not 1 <= limit <= 20:
            raise ValueError("current evidence limit must be between 1 and 20")
        tokens = tuple(
            dict.fromkeys(
                token.casefold()
                for token in re.findall(
                    r"[\w+#.-]{3,}", query.strip(), flags=re.UNICODE
                )
            )
        )
        if not tokens:
            raise ValueError("current evidence query needs a term of 3+ characters")
        normalized_query = " ".join(tokens)
        query_digest = hashlib.sha256(
            normalized_query.encode("utf-8")
        ).hexdigest()[:8]
        offset = 0
        if cursor is not None:
            match = _MEMORY_CURSOR.fullmatch(cursor)
            if match is None or match.group("query") != query_digest:
                raise ValueError("current evidence cursor does not match this query")
            offset = int(match.group("offset"), 16)
            if offset > _MAX_HISTORY_OFFSET:
                raise ValueError("current evidence cursor exceeds the safety limit")
        match_query = " OR ".join(
            json.dumps(token, ensure_ascii=False) for token in tokens
        )
        selected_columns = ", ".join(
            f"evidence.{name}" for name in _EVIDENCE_FIELD_NAMES
        )
        where = """
            current_evidence_rank_fts MATCH ?
            AND evidence.verification_status = 'confirmed'
            AND evidence.tombstoned_at IS NULL
            AND evidence.superseded_by IS NULL
        """
        with self._connect() as connection:
            self._create_current_evidence_rank_indexes(
                connection,
                user_id=user_id,
                include_long=True,
                include_short=False,
            )
            total = int(
                connection.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM current_evidence_rank_fts AS search
                    JOIN career_evidence AS evidence
                      ON evidence.id = search.evidence_id
                    WHERE {where}
                    """,
                    (match_query,),
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"""
                SELECT {selected_columns}
                FROM current_evidence_rank_fts AS search
                JOIN career_evidence AS evidence
                  ON evidence.id = search.evidence_id
                WHERE {where}
                ORDER BY bm25(current_evidence_rank_fts), evidence.created_at DESC
                LIMIT ? OFFSET ?
                """,
                (match_query, limit, offset),
            ).fetchall()
        next_offset = offset + len(rows)
        next_cursor = (
            f"memory_{query_digest}_{next_offset:08x}"
            if next_offset < total and next_offset <= _MAX_HISTORY_OFFSET
            else None
        )
        return tuple(self._evidence(row) for row in rows), total, next_cursor

    def rank_current_evidence(
        self,
        *,
        user_id: str,
        query: str | None = None,
        query_terms: CareerEvidenceQueryTerms | None = None,
        limit: int = 45,
    ) -> tuple[RankedCareerEvidence, ...]:
        """Over-recall active claims from FTS for Tier-1 reranking."""

        if not 1 <= limit <= 100:
            raise ValueError("ranked evidence limit must be between 1 and 100")
        if (query is None) == (query_terms is None):
            raise ValueError("provide exactly one of query or query_terms")
        if query_terms is None:
            query_terms = self.current_evidence_query_terms(
                user_id=user_id,
                query=query or "",
            )
        elif query_terms.user_id != user_id:
            raise ValueError("query terms do not belong to this user")
        tokens = query_terms.match_tokens
        if not tokens:
            return ()
        long_match_query, short_match_query = (
            self._current_evidence_match_queries(query_terms)
        )
        selected_columns = ", ".join(
            f"evidence.{name}" for name in _EVIDENCE_FIELD_NAMES
        )
        match_sources: list[tuple[str, str]] = []
        if long_match_query is not None:
            match_sources.append(("current_evidence_rank_fts", long_match_query))
        if short_match_query is not None:
            match_sources.append(
                ("current_evidence_rank_short_fts", short_match_query)
            )
        evidence_by_id: dict[str, CareerEvidence] = {}
        score_by_id: dict[str, float] = {}
        with self._connect() as connection:
            self._create_current_evidence_rank_indexes(
                connection,
                user_id=user_id,
                include_long=long_match_query is not None,
                include_short=short_match_query is not None,
            )
            for table, match_query in match_sources:
                rows = connection.execute(
                    f"""
                    SELECT {selected_columns}, bm25({table})
                    FROM {table} AS search
                    JOIN career_evidence AS evidence
                      ON evidence.id = search.evidence_id
                    WHERE {table} MATCH ?
                      AND evidence.verification_status = 'confirmed'
                      AND evidence.tombstoned_at IS NULL
                      AND evidence.superseded_by IS NULL
                    ORDER BY bm25({table}), evidence.created_at DESC
                    LIMIT ?
                    """,
                    (match_query, limit),
                ).fetchall()
                for row in rows:
                    score = row[-1]
                    if not isinstance(score, (int, float)):
                        continue
                    evidence = self._evidence(row[:-1])
                    evidence_by_id[evidence.id] = evidence
                    score_by_id[evidence.id] = (
                        score_by_id.get(evidence.id, 0.0) + float(score)
                    )
        ranked = sorted(
            (
                RankedCareerEvidence(
                    evidence=evidence_by_id[evidence_id],
                    bm25_score=score,
                )
                for evidence_id, score in score_by_id.items()
            ),
            key=lambda hit: (
                hit.bm25_score,
                -hit.evidence.created_at.timestamp(),
                hit.evidence.id,
            ),
        )
        return tuple(ranked[:limit])

    def current_evidence_query_terms(
        self,
        *,
        user_id: str,
        query: str,
    ) -> CareerEvidenceQueryTerms:
        """Return typed MATCH terms and the ranked population's IDF sum.

        IDF uses one filtered FTS ``COUNT`` per distinct term, so this phase is
        linear in both query-term count and the indexed population. Keep query
        terms bounded if the per-user evidence population grows substantially.
        """

        normalized = query.strip().casefold()
        latin_tokens = re.findall(r"[a-z0-9][a-z0-9+#.-]*", normalized)
        cjk_tokens = [
            token
            for chunk in re.findall(r"[\u4e00-\u9fff]+", normalized)
            for token in (
                tuple(
                    chunk[index : index + 3]
                    for index in range(len(chunk) - 2)
                )
                if len(chunk) >= 3
                else (chunk,)
            )
        ]
        lexical_terms = CareerEvidenceQueryTerms(
            user_id=user_id,
            latin=tuple(dict.fromkeys(latin_tokens)),
            cjk=tuple(dict.fromkeys(cjk_tokens)),
            idf_sum=0.0,
        )
        if not lexical_terms.match_tokens:
            return lexical_terms
        with self._connect() as connection:
            document_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM career_evidence
                    WHERE user_id = ?
                      AND verification_status = 'confirmed'
                      AND superseded_by IS NULL
                      AND tombstoned_at IS NULL
                    """,
                    (user_id,),
                ).fetchone()[0]
            )

            def matching_documents(token: str) -> int:
                table, match_term = self._current_evidence_term_match(token)
                return int(
                    connection.execute(
                        f"""
                        SELECT COUNT(*)
                        FROM {table} AS search
                        JOIN career_evidence AS evidence
                          ON evidence.id = search.evidence_id
                        WHERE {table} MATCH ?
                          AND evidence.user_id = ?
                          AND evidence.verification_status = 'confirmed'
                          AND evidence.superseded_by IS NULL
                          AND evidence.tombstoned_at IS NULL
                        """,
                        (match_term, user_id),
                    ).fetchone()[0]
                )

            idf_sum = sum(
                self._fts_idf(
                    document_count=document_count,
                    matching_documents=matching_documents(token),
                )
                for token in lexical_terms.match_tokens
            )
        return CareerEvidenceQueryTerms(
            user_id=user_id,
            latin=lexical_terms.latin,
            cjk=lexical_terms.cjk,
            idf_sum=idf_sum,
        )

    @staticmethod
    def _fts_idf(*, document_count: int, matching_documents: int) -> float:
        idf = math.log(
            (document_count - matching_documents + 0.5)
            / (matching_documents + 0.5)
        )
        return max(1e-6, idf)

    @staticmethod
    def _create_current_evidence_rank_indexes(
        connection: sqlite3.Connection,
        *,
        user_id: str,
        include_long: bool,
        include_short: bool,
    ) -> None:
        """Build query-local FTS populations identical to the ranked scope.

        This deliberately rebuilds the user's current-confirmed population on
        every query: a persistent shared FTS table would make SQLite's BM25
        statistics cross-user again. The cost is O(current evidence). A local
        10-trigram probe measured project() at 12.5 ms for 120 rows, 37.3 ms
        for 2,000, and 117.4 ms for 6,000. The first two are acceptable for a
        once-per-turn projection; at several thousand rows, measure and prefer
        safe session-level reuse before increasing the population further.
        """

        population = """
            user_id = ?
            AND verification_status = 'confirmed'
            AND superseded_by IS NULL
            AND tombstoned_at IS NULL
        """
        if include_long:
            connection.execute(
                """
                CREATE VIRTUAL TABLE temp.current_evidence_rank_fts USING fts5(
                    evidence_id UNINDEXED,
                    claim,
                    tokenize='trigram'
                )
                """
            )
            connection.execute(
                f"""
                INSERT INTO current_evidence_rank_fts(evidence_id, claim)
                SELECT id, claim FROM career_evidence
                WHERE {population}
                """,
                (user_id,),
            )
        if include_short:
            connection.execute(
                """
                CREATE VIRTUAL TABLE temp.current_evidence_rank_short_fts
                USING fts5(
                    evidence_id UNINDEXED,
                    short_terms,
                    tokenize='trigram'
                )
                """
            )
            connection.execute(
                f"""
                INSERT INTO current_evidence_rank_short_fts(
                    evidence_id, short_terms
                )
                SELECT id, short_terms FROM career_evidence
                WHERE {population}
                """,
                (user_id,),
            )

    @staticmethod
    def _short_match_token(token: str) -> str:
        if len(token) == 1:
            return f"\ue001\ue000{token}"
        if len(token) == 2:
            return f"\ue000{token}"
        raise ValueError("short MATCH tokens must contain one or two characters")

    @classmethod
    def _current_evidence_term_match(cls, token: str) -> tuple[str, str]:
        if len(token) < 3:
            return (
                "career_evidence_short_fts",
                json.dumps(cls._short_match_token(token), ensure_ascii=False),
            )
        return "career_evidence_fts", json.dumps(token, ensure_ascii=False)

    @classmethod
    def _current_evidence_match_queries(
        cls,
        terms: CareerEvidenceQueryTerms,
    ) -> tuple[str | None, str | None]:
        grouped: dict[str, list[str]] = {
            "career_evidence_fts": [],
            "career_evidence_short_fts": [],
        }
        for token in terms.match_tokens:
            table, match_term = cls._current_evidence_term_match(token)
            grouped[table].append(match_term)
        return (
            " OR ".join(grouped["career_evidence_fts"]) or None,
            " OR ".join(grouped["career_evidence_short_fts"]) or None,
        )

    @classmethod
    def _short_terms_index_text(cls, claim: str) -> str:
        normalized = claim.casefold()
        latin = re.findall(r"[a-z0-9][a-z0-9+#.-]*", normalized)
        cjk_chunks = re.findall(r"[\u4e00-\u9fff]+", normalized)
        short_terms = [token for token in latin if len(token) < 3]
        for chunk in cjk_chunks:
            short_terms.extend(chunk)
            short_terms.extend(
                chunk[index : index + 2]
                for index in range(len(chunk) - 1)
            )
        return " ".join(
            cls._short_match_token(token)
            for token in dict.fromkeys(short_terms)
        )

    def list_evidence(
        self,
        *,
        user_id: str,
        career_record_id: str | None = None,
        verification_status: EvidenceStatus | None = None,
        source_resume_version_id: str | None = None,
        include_historical: bool = False,
        include_tombstoned: bool = False,
        limit: int | None = None,
    ) -> tuple[CareerEvidence, ...]:
        if limit is not None and limit < 1:
            raise ValueError("evidence limit must be positive")
        query = f"""
            SELECT {_EVIDENCE_COLUMNS}
            FROM career_evidence
            WHERE user_id = ?
        """
        parameters: list[object] = [user_id]
        if not include_tombstoned:
            query += " AND tombstoned_at IS NULL"
        if career_record_id is not None:
            query += " AND career_record_id = ?"
            parameters.append(career_record_id)
        if verification_status is not None:
            query += " AND verification_status = ?"
            parameters.append(verification_status)
        if source_resume_version_id is not None:
            query += " AND source_resume_version_id = ?"
            parameters.append(source_resume_version_id)
        if not include_historical:
            query += (
                " AND (verification_status != 'confirmed' OR "
                "superseded_by IS NULL)"
            )
        query += " ORDER BY created_at, id"
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)

        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(self._evidence(row) for row in rows)

    def list_evidence_versions(
        self,
        *,
        user_id: str,
        scope_keys: Sequence[str],
        limit: int,
    ) -> tuple[CareerEvidence, ...]:
        """Read bounded confirmed histories for explicitly selected lineages."""

        selected = tuple(dict.fromkeys(scope_keys))
        if not selected:
            return ()
        if limit < 1:
            raise ValueError("evidence version limit must be positive")
        if len(selected) > 400:
            raise ValueError("at most 400 evidence scopes may be read at once")
        placeholders = ",".join("?" for _ in selected)
        query = f"""
            SELECT {_EVIDENCE_COLUMNS}
            FROM career_evidence
            WHERE user_id = ?
              AND verification_status = 'confirmed'
              AND tombstoned_at IS NULL
              AND scope_key IN ({placeholders})
            ORDER BY scope_key, revision, created_at, id
            LIMIT ?
        """
        with self._connect() as connection:
            rows = connection.execute(
                query,
                (user_id, *selected, limit),
            ).fetchall()
        return tuple(self._evidence(row) for row in rows)

    def count_evidence(
        self,
        *,
        user_id: str,
        verification_status: EvidenceStatus | None = None,
        include_historical: bool = False,
    ) -> int:
        query = "SELECT COUNT(*) FROM career_evidence WHERE user_id = ?"
        parameters: list[object] = [user_id]
        query += " AND tombstoned_at IS NULL"
        if verification_status is not None:
            query += " AND verification_status = ?"
            parameters.append(verification_status)
        if not include_historical:
            query += (
                " AND (verification_status != 'confirmed' OR "
                "superseded_by IS NULL)"
            )
        with self._connect() as connection:
            return int(connection.execute(query, parameters).fetchone()[0])

    def confirm_evidence(
        self,
        *,
        user_id: str,
        career_evidence_id: str,
        reason: str | None = None,
    ) -> CareerEvidence:
        return self._decide_evidence(
            user_id=user_id,
            career_evidence_id=career_evidence_id,
            new_status="confirmed",
            reason=reason,
        )

    def reject_evidence(
        self,
        *,
        user_id: str,
        career_evidence_id: str,
        reason: str | None = None,
    ) -> CareerEvidence:
        return self._decide_evidence(
            user_id=user_id,
            career_evidence_id=career_evidence_id,
            new_status="rejected",
            reason=reason,
        )

    def list_evidence_events(
        self, *, user_id: str, career_evidence_id: str
    ) -> tuple[CareerEvidenceEvent, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event.id, event.user_id, event.career_evidence_id,
                       event.event_type, event.previous_status, event.new_status,
                       event.actor_type, event.reason,
                       event.related_evidence_id, event.occurred_at
                FROM career_evidence_events AS event
                JOIN career_evidence AS evidence
                  ON evidence.id = event.career_evidence_id
                WHERE event.career_evidence_id = ?
                  AND evidence.user_id = ?
                ORDER BY event.rowid
                """,
                (career_evidence_id, user_id),
            ).fetchall()
        return tuple(self._event(row) for row in rows)

    def get_current_evidence(
        self, *, user_id: str, scope_key: str
    ) -> CareerEvidence | None:
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT {_EVIDENCE_COLUMNS}
                FROM career_evidence
                WHERE user_id = ? AND scope_key = ?
                  AND verification_status = 'confirmed'
                  AND superseded_by IS NULL
                  AND tombstoned_at IS NULL
                """,
                (user_id, scope_key),
            ).fetchone()
        return self._evidence(row) if row is not None else None

    def list_evidence_lineage(
        self, *, user_id: str, scope_key: str
    ) -> tuple[CareerEvidence, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT {_EVIDENCE_COLUMNS}
                FROM career_evidence
                WHERE user_id = ? AND scope_key = ?
                  AND verification_status = 'confirmed'
                  AND tombstoned_at IS NULL
                ORDER BY revision, created_at, id
                """,
                (user_id, scope_key),
            ).fetchall()
        return tuple(self._evidence(row) for row in rows)

    def tombstone_evidence(
        self,
        *,
        user_id: str,
        career_evidence_id: str,
        reason: str,
        actor_type: Literal["user", "agent", "system"] = "user",
        expected_content_sha256: str | None = None,
    ) -> CareerEvidenceTombstone:
        """Redact every linked revision while retaining tombstone metadata."""

        tombstone_reason = reason.strip()
        if not tombstone_reason:
            raise ValueError("Tombstone reason is required.")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            target_row = connection.execute(
                f"""
                SELECT {_EVIDENCE_COLUMNS}
                FROM career_evidence
                WHERE id = ? AND user_id = ?
                """,
                (career_evidence_id, user_id),
            ).fetchone()
            if target_row is None:
                raise ValueError("Career evidence not found.")
            target = self._evidence(target_row)
            if not target.is_current or target.scope_key is None:
                raise ValueError("Only current confirmed evidence can be tombstoned.")
            if (
                expected_content_sha256 is not None
                and target.content_digest != expected_content_sha256
            ):
                raise ValueError("Career evidence changed after deletion was proposed.")
            lineage_rows = connection.execute(
                f"""
                SELECT {_EVIDENCE_COLUMNS}
                FROM career_evidence
                WHERE user_id = ? AND scope_key = ?
                  AND verification_status = 'confirmed'
                ORDER BY revision, created_at, id
                """,
                (user_id, target.scope_key),
            ).fetchall()
            lineage = tuple(self._evidence(row) for row in lineage_rows)
            if not lineage:
                raise ValueError("Career evidence lineage is missing.")
            now = datetime.now(timezone.utc)
            for item in lineage:
                connection.execute(
                    """
                    UPDATE career_evidence
                    SET claim = '', short_terms = '',
                        source_locator = CASE
                            WHEN source_resume_version_id IS NULL THEN NULL
                            ELSE ''
                        END,
                        source_quote = CASE
                            WHEN source_resume_version_id IS NULL THEN NULL
                            ELSE ''
                        END,
                        tombstoned_at = ?, tombstoned_by = ?,
                        tombstone_reason = ?,
                        updated_at = ?
                    WHERE id = ? AND user_id = ? AND tombstoned_at IS NULL
                    """,
                    (
                        now.isoformat(),
                        actor_type,
                        tombstone_reason,
                        now.isoformat(),
                        item.id,
                        user_id,
                    ),
                )
                self._insert_event(
                    connection,
                    CareerEvidenceEvent(
                        id=f"career_evidence_event_{uuid4().hex}",
                        user_id=user_id,
                        career_evidence_id=item.id,
                        event_type="tombstoned",
                        previous_status="confirmed",
                        new_status="confirmed",
                        actor_type=actor_type,
                        reason=tombstone_reason,
                        occurred_at=now,
                    ),
                )
        return CareerEvidenceTombstone(
            scope_key=target.scope_key,
            evidence_ids=tuple(item.id for item in lineage),
            actor_type=actor_type,
            reason=tombstone_reason,
            tombstoned_at=now,
        )

    def list_evidence_tombstones(
        self,
        *,
        user_id: str,
        limit: int = 100,
    ) -> tuple[CareerEvidenceTombstone, ...]:
        """Return one audit tombstone per redacted revision chain."""

        if not 1 <= limit <= 500:
            raise ValueError("tombstone audit limit must be between 1 and 500")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT scope_key, id, tombstoned_by, tombstone_reason,
                       tombstoned_at
                FROM career_evidence
                WHERE user_id = ? AND tombstoned_at IS NOT NULL
                ORDER BY tombstoned_at DESC, revision, id
                """,
                (user_id,),
            ).fetchall()
        grouped: dict[str, list[tuple[object, ...]]] = {}
        for row in rows:
            grouped.setdefault(str(row[0]), []).append(row)
        return tuple(
            CareerEvidenceTombstone(
                scope_key=scope_key,
                evidence_ids=tuple(str(item[1]) for item in items),
                actor_type=str(items[0][2]),
                reason=str(items[0][3]),
                tombstoned_at=str(items[0][4]),
            )
            for scope_key, items in tuple(grouped.items())[:limit]
        )

    def list_tombstoned_scopes(
        self,
    ) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
        """Return redacted scopes and opaque markers for idempotent cleanup."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT user_id, scope_key, detail_ref, source_ref
                FROM career_evidence
                WHERE tombstoned_at IS NOT NULL
                ORDER BY user_id, scope_key, revision, id
                """
            ).fetchall()
        grouped: dict[tuple[str, str], list[str]] = {}
        for user_id, scope_key, detail_ref, source_ref in rows:
            key = (str(user_id), str(scope_key))
            markers = grouped.setdefault(key, [])
            markers.extend(
                str(marker)
                for marker in (detail_ref, source_ref)
                if marker is not None
            )
        return tuple(
            (
                user_id,
                scope_key,
                tuple(
                    dict.fromkeys(
                        (
                            career_evidence_lineage_ref(
                                user_id=user_id, scope_key=scope_key
                            ),
                            *markers,
                        )
                    )
                ),
            )
            for (user_id, scope_key), markers in grouped.items()
        )

    def correct_evidence(
        self,
        *,
        user_id: str,
        career_evidence_id: str,
        new_claim: str,
        reason: str,
    ) -> CareerEvidenceCorrection:
        """Append a new linked revision without mutating older history back."""

        claim = new_claim.strip()
        correction_reason = reason.strip()
        if not claim or not correction_reason:
            raise ValueError("A correction requires a new claim and reason.")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                f"""
                SELECT {_EVIDENCE_COLUMNS}
                FROM career_evidence
                WHERE id = ? AND user_id = ?
                """,
                (career_evidence_id, user_id),
            ).fetchone()
            if row is None:
                raise ValueError("Career evidence not found.")
            current = self._evidence(row)
            if not current.is_current or current.scope_key is None:
                raise ValueError("Only current confirmed evidence can be corrected.")
            if self._claim_digest(current.claim) == self._claim_digest(claim):
                raise ValueError("The corrected claim is unchanged.")
            now = datetime.now(timezone.utc)
            replacement_id = f"career_evidence_{uuid4().hex}"
            latest_revision = int(
                connection.execute(
                    """
                    SELECT COALESCE(MAX(revision), 0)
                    FROM career_evidence
                    WHERE user_id = ? AND scope_key = ?
                    """,
                    (user_id, current.scope_key),
                ).fetchone()[0]
            )
            replacement = CareerEvidence(
                id=replacement_id,
                user_id=user_id,
                career_record_id=current.career_record_id,
                claim=claim,
                origin="user_input",
                verification_status="confirmed",
                detail_ref=career_evidence_detail_ref(
                    user_id=user_id, evidence_id=replacement_id
                ),
                scope_key=current.scope_key,
                update_id=f"career_evidence_update_{uuid4().hex}",
                content_digest=intent_content_digest(claim),
                revision=latest_revision + 1,
                valid_from=now,
                supersedes_id=current.id,
                created_at=now,
                updated_at=now,
            )
            updated = connection.execute(
                """
                UPDATE career_evidence
                SET superseded_at = ?, superseded_by = ?, updated_at = ?
                WHERE id = ? AND user_id = ?
                  AND superseded_by IS NULL
                  AND tombstoned_at IS NULL
                """,
                (
                    now.isoformat(), replacement.id, now.isoformat(),
                    current.id, user_id,
                ),
            ).rowcount
            if updated != 1:
                raise ValueError("Career evidence changed before correction committed.")
            self._insert_evidence(connection, replacement)
            superseded = current.model_copy(
                update={
                    "superseded_at": now,
                    "superseded_by": replacement.id,
                    "updated_at": now,
                }
            )
            for evidence_id, event_type, related_id in (
                (current.id, "superseded", replacement.id),
                (replacement.id, "corrected", current.id),
            ):
                self._insert_event(
                    connection,
                    CareerEvidenceEvent(
                        id=f"career_evidence_event_{uuid4().hex}",
                        user_id=user_id,
                        career_evidence_id=evidence_id,
                        event_type=event_type,
                        previous_status="confirmed",
                        new_status="confirmed",
                        actor_type="user",
                        reason=correction_reason,
                        related_evidence_id=related_id,
                        occurred_at=now,
                    ),
                )
        return CareerEvidenceCorrection(previous=superseded, current=replacement)

    def import_confirmed_resume_analysis(
        self,
        *,
        user_id: str,
        analysis_id: str,
        resume_version_id: str,
        result: ResumeAnalysisResult,
    ) -> CareerHistoryImportResult:
        """Atomically imports one user-confirmed analysis and is idempotent by analysis ID."""
        if not user_id.strip() or not analysis_id.strip() or not resume_version_id.strip():
            raise ValueError("user_id, analysis_id, and resume_version_id are required")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT user_id, career_record_ids_json, career_evidence_ids_json
                FROM resume_analysis_career_imports
                WHERE analysis_id = ?
                """,
                (analysis_id,),
            ).fetchone()
            if existing is not None:
                if existing[0] != user_id:
                    raise ValueError("Resume analysis import not found.")
                return CareerHistoryImportResult(
                    records=self._records_by_ids(connection, json.loads(existing[1])),
                    evidence=self._evidence_by_ids(connection, json.loads(existing[2])),
                )
            if not self._resume_version_belongs_to_user(
                connection,
                user_id=user_id,
                resume_version_id=resume_version_id,
            ):
                raise ValueError("Source resume version not found.")

            now = datetime.now(timezone.utc)
            records: list[CareerRecord] = []
            all_evidence: list[CareerEvidence] = []
            for extracted in result.records:
                record = CareerRecord(
                    id=f"career_record_{uuid4().hex}",
                    user_id=user_id,
                    record_type=extracted.record_type,
                    organization=extracted.organization,
                    title=extracted.title,
                    start_year=extracted.start_year,
                    start_month=extracted.start_month,
                    end_year=extracted.end_year,
                    end_month=extracted.end_month,
                    is_current=extracted.is_current,
                    created_at=now,
                    updated_at=now,
                )
                connection.execute(
                    """
                    INSERT INTO career_records(
                        id, user_id, record_type, organization, title,
                        start_year, start_month, end_year, end_month, is_current,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.id,
                        record.user_id,
                        record.record_type,
                        record.organization,
                        record.title,
                        record.start_year,
                        record.start_month,
                        record.end_year,
                        record.end_month,
                        int(record.is_current),
                        record.created_at.isoformat(),
                        record.updated_at.isoformat(),
                    ),
                )
                records.append(record)

                candidates = [
                    (
                        extracted.source_quote,
                        extracted.source_locator,
                        extracted.source_quote,
                    ),
                    *(
                        (item.claim, item.source_locator, item.source_quote)
                        for item in extracted.evidence
                    ),
                ]
                seen: set[tuple[str, str, str]] = set()
                imported_for_record = 0
                for claim, locator, quote in candidates:
                    key = (claim, locator, quote)
                    if key in seen:
                        continue
                    seen.add(key)
                    evidence_id = f"career_evidence_{uuid4().hex}"
                    evidence = CareerEvidence(
                        id=evidence_id,
                        user_id=user_id,
                        career_record_id=record.id,
                        claim=claim,
                        origin="resume_extraction",
                        verification_status="confirmed",
                        source_resume_version_id=resume_version_id,
                        source_locator=locator,
                        source_quote=quote,
                        source_ref=career_evidence_source_ref(
                            user_id=user_id,
                            evidence_id=evidence_id,
                            source_resume_version_id=resume_version_id,
                            source_locator=locator,
                        ),
                        detail_ref=career_evidence_detail_ref(
                            user_id=user_id,
                            evidence_id=evidence_id,
                        ),
                        scope_key=career_evidence_scope_key(evidence_id),
                        update_id=f"career_evidence_update_{uuid4().hex}",
                        content_digest=intent_content_digest(claim),
                        revision=1,
                        valid_from=now,
                        created_at=now,
                        updated_at=now,
                    )
                    connection.execute(
                        """
                        INSERT INTO career_evidence(
                            id, user_id, career_record_id, claim, short_terms, origin,
                            verification_status, source_resume_version_id,
                            source_locator, source_quote, source_ref, detail_ref,
                            scope_key, update_id, content_digest, revision,
                            valid_from, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            evidence.id,
                            evidence.user_id,
                            evidence.career_record_id,
                            evidence.claim,
                            self._short_terms_index_text(evidence.claim),
                            evidence.origin,
                            evidence.verification_status,
                            evidence.source_resume_version_id,
                            evidence.source_locator,
                            evidence.source_quote,
                            evidence.source_ref,
                            evidence.detail_ref,
                            evidence.scope_key,
                            evidence.update_id,
                            evidence.content_digest,
                            evidence.revision,
                            evidence.valid_from.isoformat(),
                            evidence.created_at.isoformat(),
                            evidence.updated_at.isoformat(),
                        ),
                    )
                    self._insert_event(
                        connection,
                        CareerEvidenceEvent(
                            id=f"career_evidence_event_{uuid4().hex}",
                            user_id=user_id,
                            career_evidence_id=evidence.id,
                            event_type="created",
                            previous_status=None,
                            new_status="pending",
                            actor_type="system",
                            occurred_at=now,
                        ),
                    )
                    self._insert_event(
                        connection,
                        CareerEvidenceEvent(
                            id=f"career_evidence_event_{uuid4().hex}",
                            user_id=user_id,
                            career_evidence_id=evidence.id,
                            event_type="confirmed",
                            previous_status="pending",
                            new_status="confirmed",
                            actor_type="user",
                            reason=f"Confirmed resume analysis {analysis_id}",
                            occurred_at=now,
                        ),
                    )
                    all_evidence.append(evidence)
                    imported_for_record += 1

                if imported_for_record == 0:
                    connection.execute(
                        "DELETE FROM career_records WHERE id = ? AND user_id = ?",
                        (record.id, user_id),
                    )
                    records.pop()

            connection.execute(
                """
                INSERT INTO resume_analysis_career_imports(
                    analysis_id, user_id, resume_version_id,
                    career_record_ids_json, career_evidence_ids_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis_id,
                    user_id,
                    resume_version_id,
                    json.dumps([record.id for record in records]),
                    json.dumps([evidence.id for evidence in all_evidence]),
                    now.isoformat(),
                ),
            )
        return CareerHistoryImportResult(
            records=tuple(records),
            evidence=tuple(all_evidence),
        )

    def _decide_evidence(
        self,
        *,
        user_id: str,
        career_evidence_id: str,
        new_status: Literal["confirmed", "rejected"],
        reason: str | None,
    ) -> CareerEvidence:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                f"""
                SELECT {_EVIDENCE_COLUMNS}
                FROM career_evidence
                WHERE id = ? AND user_id = ?
                """,
                (career_evidence_id, user_id),
            ).fetchone()
            if row is None:
                raise ValueError("Career evidence not found.")
            current = self._evidence(row)
            if current.verification_status == new_status:
                return current
            if current.verification_status != "pending":
                raise ValueError(
                    f"Cannot change {current.verification_status} evidence to {new_status}."
                )

            now = datetime.now(timezone.utc)
            version_update = (
                {
                    "scope_key": career_evidence_scope_key(current.id),
                    "update_id": f"career_evidence_update_{uuid4().hex}",
                    "content_digest": intent_content_digest(current.claim),
                    "revision": 1,
                    "valid_from": now,
                }
                if new_status == "confirmed"
                else {}
            )
            updated = current.model_copy(
                update={
                    "verification_status": new_status,
                    "updated_at": now,
                    **version_update,
                }
            )
            event = CareerEvidenceEvent(
                id=f"career_evidence_event_{uuid4().hex}",
                user_id=current.user_id,
                career_evidence_id=current.id,
                event_type=new_status,
                previous_status="pending",
                new_status=new_status,
                actor_type="user",
                reason=reason,
                occurred_at=now,
            )
            connection.execute(
                """
                UPDATE career_evidence
                SET verification_status = ?, scope_key = ?, update_id = ?,
                    content_digest = ?, revision = ?, valid_from = ?, updated_at = ?
                WHERE id = ? AND user_id = ? AND verification_status = 'pending'
                """,
                (
                    updated.verification_status,
                    updated.scope_key,
                    updated.update_id,
                    updated.content_digest,
                    updated.revision,
                    (
                        updated.valid_from.isoformat()
                        if updated.valid_from is not None
                        else None
                    ),
                    updated.updated_at.isoformat(),
                    updated.id,
                    updated.user_id,
                ),
            )
            self._insert_event(connection, event)
        return updated

    @staticmethod
    def _resume_version_belongs_to_user(
        connection: sqlite3.Connection, *, user_id: str, resume_version_id: str
    ) -> bool:
        has_resume_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'resume_versions'"
        ).fetchone()
        if has_resume_table is None:
            return False
        return (
            connection.execute(
                """
                SELECT 1
                FROM resume_versions AS version
                JOIN resumes AS resume ON resume.id = version.resume_id
                WHERE version.id = ? AND resume.user_id = ?
                """,
                (resume_version_id, user_id),
            ).fetchone()
            is not None
        )

    @staticmethod
    def _insert_evidence(
        connection: sqlite3.Connection, evidence: CareerEvidence
    ) -> None:
        connection.execute(
            """
            INSERT INTO career_evidence(
                id, user_id, career_record_id, claim, short_terms, origin,
                verification_status, source_resume_version_id,
                source_locator, source_quote, source_ref, detail_ref,
                scope_key, update_id, content_digest, revision, valid_from, supersedes_id,
                superseded_at, superseded_by, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evidence.id,
                evidence.user_id,
                evidence.career_record_id,
                evidence.claim,
                CareerHistoryStore._short_terms_index_text(evidence.claim),
                evidence.origin,
                evidence.verification_status,
                evidence.source_resume_version_id,
                evidence.source_locator,
                evidence.source_quote,
                evidence.source_ref,
                evidence.detail_ref,
                evidence.scope_key,
                evidence.update_id,
                evidence.content_digest,
                evidence.revision,
                evidence.valid_from.isoformat() if evidence.valid_from else None,
                evidence.supersedes_id,
                evidence.superseded_at.isoformat() if evidence.superseded_at else None,
                evidence.superseded_by,
                evidence.created_at.isoformat(),
                evidence.updated_at.isoformat(),
            ),
        )

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection, event: CareerEvidenceEvent
    ) -> None:
        connection.execute(
            """
            INSERT INTO career_evidence_events(
                id, user_id, career_evidence_id, event_type,
                previous_status, new_status, actor_type, reason,
                related_evidence_id, occurred_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.id,
                event.user_id,
                event.career_evidence_id,
                event.event_type,
                event.previous_status,
                event.new_status,
                event.actor_type,
                event.reason,
                event.related_evidence_id,
                event.occurred_at.isoformat(),
            ),
        )

    @staticmethod
    def _claim_digest(value: str) -> str:
        return intent_content_digest(value)

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS career_records (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                record_type TEXT NOT NULL CHECK (
                    record_type IN ('education', 'work', 'internship', 'project', 'certification')
                ),
                organization TEXT,
                title TEXT NOT NULL,
                start_year INTEGER CHECK (start_year IS NULL OR start_year BETWEEN 1900 AND 2200),
                start_month INTEGER CHECK (start_month IS NULL OR start_month BETWEEN 1 AND 12),
                end_year INTEGER CHECK (end_year IS NULL OR end_year BETWEEN 1900 AND 2200),
                end_month INTEGER CHECK (end_month IS NULL OR end_month BETWEEN 1 AND 12),
                is_current INTEGER NOT NULL CHECK (is_current IN (0, 1)),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                CHECK (start_month IS NULL OR start_year IS NOT NULL),
                CHECK (end_month IS NULL OR end_year IS NOT NULL),
                CHECK (is_current = 0 OR (end_year IS NULL AND end_month IS NULL))
            );

            CREATE TABLE IF NOT EXISTS career_evidence (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                career_record_id TEXT NOT NULL REFERENCES career_records(id),
                claim TEXT NOT NULL,
                short_terms TEXT NOT NULL DEFAULT '',
                origin TEXT NOT NULL CHECK (
                    origin IN ('resume_extraction', 'user_input', 'agent_inference')
                ),
                verification_status TEXT NOT NULL CHECK (
                    verification_status IN ('pending', 'confirmed', 'rejected')
                ),
                source_resume_version_id TEXT,
                source_locator TEXT,
                source_quote TEXT,
                source_ref TEXT,
                detail_ref TEXT,
                scope_key TEXT,
                update_id TEXT,
                content_digest TEXT,
                revision INTEGER,
                valid_from TEXT,
                supersedes_id TEXT REFERENCES career_evidence(id)
                    DEFERRABLE INITIALLY DEFERRED,
                superseded_at TEXT,
                superseded_by TEXT REFERENCES career_evidence(id)
                    DEFERRABLE INITIALLY DEFERRED,
                tombstoned_at TEXT,
                tombstoned_by TEXT CHECK (
                    tombstoned_by IS NULL
                    OR tombstoned_by IN ('user', 'agent', 'system')
                ),
                tombstone_reason TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                CHECK (source_locator IS NULL OR source_resume_version_id IS NOT NULL),
                CHECK (source_quote IS NULL OR source_resume_version_id IS NOT NULL),
                CHECK (
                    (
                        scope_key IS NULL AND update_id IS NULL
                        AND content_digest IS NULL AND revision IS NULL
                        AND valid_from IS NULL
                    )
                    OR (
                        scope_key IS NOT NULL AND update_id IS NOT NULL
                        AND content_digest IS NOT NULL AND revision IS NOT NULL
                        AND valid_from IS NOT NULL
                    )
                ),
                CHECK (
                    (superseded_at IS NULL AND superseded_by IS NULL)
                    OR (superseded_at IS NOT NULL AND superseded_by IS NOT NULL)
                ),
                CHECK (
                    origin != 'resume_extraction'
                    OR (
                        source_resume_version_id IS NOT NULL
                        AND source_locator IS NOT NULL
                        AND source_quote IS NOT NULL
                    )
                )
            );

            CREATE TABLE IF NOT EXISTS career_evidence_events (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                career_evidence_id TEXT NOT NULL REFERENCES career_evidence(id),
                event_type TEXT NOT NULL CHECK (
                    event_type IN (
                        'created', 'confirmed', 'rejected', 'superseded',
                        'corrected', 'rolled_back', 'restored', 'tombstoned'
                    )
                ),
                previous_status TEXT CHECK (
                    previous_status IS NULL
                    OR previous_status IN ('pending', 'confirmed', 'rejected')
                ),
                new_status TEXT NOT NULL CHECK (
                    new_status IN ('pending', 'confirmed', 'rejected')
                ),
                actor_type TEXT NOT NULL CHECK (
                    actor_type IN ('user', 'agent', 'system')
                ),
                reason TEXT,
                related_evidence_id TEXT REFERENCES career_evidence(id),
                occurred_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS resume_analysis_career_imports (
                analysis_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                resume_version_id TEXT NOT NULL,
                career_record_ids_json TEXT NOT NULL,
                career_evidence_ids_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS career_records_user_time_idx
                ON career_records(user_id, start_year DESC, start_month DESC);
            CREATE INDEX IF NOT EXISTS career_evidence_user_record_idx
                ON career_evidence(user_id, career_record_id, verification_status);
            CREATE INDEX IF NOT EXISTS career_evidence_events_evidence_idx
                ON career_evidence_events(career_evidence_id, occurred_at);
            CREATE INDEX IF NOT EXISTS resume_analysis_career_imports_user_idx
                ON resume_analysis_career_imports(user_id, created_at DESC);
            """
        )
        evidence_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(career_evidence)")
        }
        if "source_quote" not in evidence_columns:
            connection.execute("ALTER TABLE career_evidence ADD COLUMN source_quote TEXT")
        connection.execute(
            """
            UPDATE career_evidence
            SET source_quote = claim
            WHERE origin = 'resume_extraction' AND source_quote IS NULL
            """
        )
        CareerHistoryStore._ensure_source_refs(connection)
        # Cumulative baseline adoption: pre-registry databases do not replay
        # numbered upgrades, so the idempotent backfill must run here as well.
        CareerHistoryStore._ensure_evidence_version_schema(connection)
        CareerHistoryStore._ensure_m4b_read_schema(connection)
        CareerHistoryStore._ensure_m3_tombstone_schema(connection)
        CareerHistoryStore._drop_removed_memory_tables(connection)
        CareerHistoryStore._ensure_short_query_fts_schema(connection)

    @staticmethod
    def _upgrade_to_v3(connection: sqlite3.Connection) -> None:
        CareerHistoryStore._ensure_source_refs(connection)

    @staticmethod
    def _upgrade_to_v4(connection: sqlite3.Connection) -> None:
        CareerHistoryStore._ensure_evidence_version_schema(connection)

    @staticmethod
    def _upgrade_to_v5(connection: sqlite3.Connection) -> None:
        CareerHistoryStore._ensure_m4b_read_schema(connection)

    @staticmethod
    def _upgrade_to_v6(connection: sqlite3.Connection) -> None:
        CareerHistoryStore._ensure_m3_tombstone_schema(connection)

    @staticmethod
    def _upgrade_to_v7(connection: sqlite3.Connection) -> None:
        CareerHistoryStore._drop_removed_memory_tables(connection)

    @staticmethod
    def _upgrade_to_v8(connection: sqlite3.Connection) -> None:
        CareerHistoryStore._ensure_short_query_fts_schema(connection)

    @staticmethod
    def _drop_removed_memory_tables(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            DROP INDEX IF EXISTS career_evidence_suppression_idx;
            DROP INDEX IF EXISTS career_evidence_mutations_user_idx;
            DROP TABLE IF EXISTS career_memory_deletion_operations;
            DROP TABLE IF EXISTS career_evidence_suppressions;
            DROP TABLE IF EXISTS career_evidence_mutations;
            """
        )
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(career_evidence)")
        }
        if "suppression_digest" in columns:
            connection.execute(
                "ALTER TABLE career_evidence DROP COLUMN suppression_digest"
            )

    @staticmethod
    def _ensure_m3_tombstone_schema(connection: sqlite3.Connection) -> None:
        evidence_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(career_evidence)")
        }
        additions = {
            "tombstoned_at": "TEXT",
            "tombstoned_by": "TEXT",
            "tombstone_reason": "TEXT",
        }
        for name, definition in additions.items():
            if name not in evidence_columns:
                connection.execute(
                    f"ALTER TABLE career_evidence ADD COLUMN {name} {definition}"
                )

        event_sql_row = connection.execute(
            """
            SELECT sql FROM sqlite_master
            WHERE type = 'table' AND name = 'career_evidence_events'
            """
        ).fetchone()
        event_sql = str(event_sql_row[0]) if event_sql_row else ""
        if "'tombstoned'" not in event_sql:
            connection.executescript(
                """
                ALTER TABLE career_evidence_events
                    RENAME TO career_evidence_events_v5;
                CREATE TABLE career_evidence_events (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    career_evidence_id TEXT NOT NULL REFERENCES career_evidence(id),
                    event_type TEXT NOT NULL CHECK (
                        event_type IN (
                            'created', 'confirmed', 'rejected', 'superseded',
                            'corrected', 'rolled_back', 'restored', 'tombstoned'
                        )
                    ),
                    previous_status TEXT CHECK (
                        previous_status IS NULL
                        OR previous_status IN ('pending', 'confirmed', 'rejected')
                    ),
                    new_status TEXT NOT NULL CHECK (
                        new_status IN ('pending', 'confirmed', 'rejected')
                    ),
                    actor_type TEXT NOT NULL CHECK (
                        actor_type IN ('user', 'agent', 'system')
                    ),
                    reason TEXT,
                    mutation_id TEXT,
                    related_evidence_id TEXT REFERENCES career_evidence(id),
                    occurred_at TEXT NOT NULL
                );
                INSERT INTO career_evidence_events(
                    id, user_id, career_evidence_id, event_type,
                    previous_status, new_status, actor_type, reason,
                    mutation_id, related_evidence_id, occurred_at
                )
                SELECT id, user_id, career_evidence_id, event_type,
                       previous_status, new_status, actor_type, reason,
                       mutation_id, related_evidence_id, occurred_at
                FROM career_evidence_events_v5;
                DROP TABLE career_evidence_events_v5;
                """
            )

        connection.executescript(
            """
            DROP TRIGGER IF EXISTS career_evidence_fts_insert;
            DROP TRIGGER IF EXISTS career_evidence_fts_delete;
            DROP TRIGGER IF EXISTS career_evidence_fts_update;
            CREATE TRIGGER career_evidence_fts_insert
            AFTER INSERT ON career_evidence
            WHEN new.tombstoned_at IS NULL BEGIN
                INSERT INTO career_evidence_fts(evidence_id, user_id, claim)
                VALUES (new.id, new.user_id, new.claim);
            END;
            CREATE TRIGGER career_evidence_fts_delete
            AFTER DELETE ON career_evidence BEGIN
                DELETE FROM career_evidence_fts WHERE evidence_id = old.id;
            END;
            CREATE TRIGGER career_evidence_fts_update
            AFTER UPDATE OF claim, user_id, tombstoned_at ON career_evidence BEGIN
                DELETE FROM career_evidence_fts WHERE evidence_id = old.id;
                INSERT INTO career_evidence_fts(evidence_id, user_id, claim)
                SELECT new.id, new.user_id, new.claim
                WHERE new.tombstoned_at IS NULL;
            END;
            DELETE FROM career_evidence_fts
            WHERE evidence_id IN (
                SELECT id FROM career_evidence WHERE tombstoned_at IS NOT NULL
            );
            CREATE INDEX IF NOT EXISTS career_evidence_events_evidence_idx
            ON career_evidence_events(career_evidence_id, occurred_at);
            DROP INDEX IF EXISTS career_evidence_active_scope_idx;
            CREATE UNIQUE INDEX career_evidence_active_scope_idx
            ON career_evidence(user_id, scope_key)
            WHERE verification_status = 'confirmed'
              AND superseded_by IS NULL
              AND tombstoned_at IS NULL;
            """
        )

    @staticmethod
    def _ensure_m4b_read_schema(connection: sqlite3.Connection) -> None:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(career_evidence)")
        }
        if "detail_ref" not in columns:
            connection.execute("ALTER TABLE career_evidence ADD COLUMN detail_ref TEXT")
        rows = connection.execute(
            "SELECT id, user_id FROM career_evidence WHERE detail_ref IS NULL"
        ).fetchall()
        connection.executemany(
            "UPDATE career_evidence SET detail_ref = ? WHERE id = ?",
            (
                (
                    career_evidence_detail_ref(
                        user_id=str(user_id),
                        evidence_id=str(evidence_id),
                    ),
                    evidence_id,
                )
                for evidence_id, user_id in rows
            ),
        )
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS career_evidence_detail_ref_unique_idx
            ON career_evidence(detail_ref)
            """
        )
        connection.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS career_evidence_fts USING fts5(
                evidence_id UNINDEXED,
                user_id UNINDEXED,
                claim,
                tokenize='trigram'
            )
            """
        )
        connection.executescript(
            """
            CREATE TRIGGER IF NOT EXISTS career_evidence_fts_insert
            AFTER INSERT ON career_evidence BEGIN
                INSERT INTO career_evidence_fts(evidence_id, user_id, claim)
                VALUES (new.id, new.user_id, new.claim);
            END;
            CREATE TRIGGER IF NOT EXISTS career_evidence_fts_delete
            AFTER DELETE ON career_evidence BEGIN
                DELETE FROM career_evidence_fts WHERE evidence_id = old.id;
            END;
            CREATE TRIGGER IF NOT EXISTS career_evidence_fts_update
            AFTER UPDATE OF claim, user_id ON career_evidence BEGIN
                DELETE FROM career_evidence_fts WHERE evidence_id = old.id;
                INSERT INTO career_evidence_fts(evidence_id, user_id, claim)
                VALUES (new.id, new.user_id, new.claim);
            END;
            """
        )
        connection.execute(
            """
            INSERT INTO career_evidence_fts(evidence_id, user_id, claim)
            SELECT evidence.id, evidence.user_id, evidence.claim
            FROM career_evidence AS evidence
            WHERE NOT EXISTS (
                SELECT 1
                FROM career_evidence_fts AS search
                WHERE search.evidence_id = evidence.id
            )
            """
        )

    @staticmethod
    def _ensure_short_query_fts_schema(
        connection: sqlite3.Connection,
    ) -> None:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(career_evidence)")
        }
        if "short_terms" not in columns:
            connection.execute(
                """
                ALTER TABLE career_evidence
                ADD COLUMN short_terms TEXT NOT NULL DEFAULT ''
                """
            )
        rows = connection.execute(
            "SELECT id, claim FROM career_evidence"
        ).fetchall()
        connection.executemany(
            "UPDATE career_evidence SET short_terms = ? WHERE id = ?",
            (
                (
                    CareerHistoryStore._short_terms_index_text(str(claim)),
                    evidence_id,
                )
                for evidence_id, claim in rows
            ),
        )
        connection.executescript(
            """
            DROP TRIGGER IF EXISTS career_evidence_fts_insert;
            DROP TRIGGER IF EXISTS career_evidence_fts_delete;
            DROP TRIGGER IF EXISTS career_evidence_fts_update;
            DROP TABLE IF EXISTS career_evidence_fts;
            DROP TABLE IF EXISTS career_evidence_short_fts;
            CREATE VIRTUAL TABLE career_evidence_fts USING fts5(
                evidence_id UNINDEXED,
                user_id UNINDEXED,
                claim,
                tokenize='trigram'
            );
            CREATE VIRTUAL TABLE career_evidence_short_fts USING fts5(
                evidence_id UNINDEXED,
                user_id UNINDEXED,
                short_terms,
                tokenize='trigram'
            );
            INSERT INTO career_evidence_fts(evidence_id, user_id, claim)
            SELECT id, user_id, claim
            FROM career_evidence
            WHERE tombstoned_at IS NULL;
            INSERT INTO career_evidence_short_fts(
                evidence_id, user_id, short_terms
            )
            SELECT id, user_id, short_terms
            FROM career_evidence
            WHERE tombstoned_at IS NULL;
            CREATE TRIGGER career_evidence_fts_insert
            AFTER INSERT ON career_evidence
            WHEN new.tombstoned_at IS NULL BEGIN
                INSERT INTO career_evidence_fts(evidence_id, user_id, claim)
                VALUES (new.id, new.user_id, new.claim);
                INSERT INTO career_evidence_short_fts(
                    evidence_id, user_id, short_terms
                ) VALUES (new.id, new.user_id, new.short_terms);
            END;
            CREATE TRIGGER career_evidence_fts_delete
            AFTER DELETE ON career_evidence BEGIN
                DELETE FROM career_evidence_fts WHERE evidence_id = old.id;
                DELETE FROM career_evidence_short_fts WHERE evidence_id = old.id;
            END;
            CREATE TRIGGER career_evidence_fts_update
            AFTER UPDATE OF claim, short_terms, user_id, tombstoned_at
            ON career_evidence BEGIN
                DELETE FROM career_evidence_fts WHERE evidence_id = old.id;
                DELETE FROM career_evidence_short_fts WHERE evidence_id = old.id;
                INSERT INTO career_evidence_fts(evidence_id, user_id, claim)
                SELECT new.id, new.user_id, new.claim
                WHERE new.tombstoned_at IS NULL;
                INSERT INTO career_evidence_short_fts(
                    evidence_id, user_id, short_terms
                )
                SELECT new.id, new.user_id, new.short_terms
                WHERE new.tombstoned_at IS NULL;
            END;
            """
        )

    @staticmethod
    def _ensure_evidence_version_schema(connection: sqlite3.Connection) -> None:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(career_evidence)")
        }
        additions = {
            "scope_key": "TEXT",
            "update_id": "TEXT",
            "content_digest": "TEXT",
            "revision": "INTEGER",
            "valid_from": "TEXT",
            "supersedes_id": (
                "TEXT REFERENCES career_evidence(id) DEFERRABLE INITIALLY DEFERRED"
            ),
            "superseded_at": "TEXT",
            "superseded_by": (
                "TEXT REFERENCES career_evidence(id) DEFERRABLE INITIALLY DEFERRED"
            ),
        }
        for name, definition in additions.items():
            if name not in columns:
                connection.execute(
                    f"ALTER TABLE career_evidence ADD COLUMN {name} {definition}"
                )
        CareerHistoryStore._ensure_lineage_event_schema(connection)
        now = datetime.now(timezone.utc).isoformat()
        legacy_rows = connection.execute(
            """
            SELECT id, claim
            FROM career_evidence
            WHERE verification_status = 'confirmed'
              AND scope_key IS NULL
              AND update_id IS NULL
              AND content_digest IS NULL
              AND revision IS NULL
              AND valid_from IS NULL
            """
        ).fetchall()
        for evidence_id, claim in legacy_rows:
            migration_update_id = "career_evidence_update_" + hashlib.sha256(
                f"{evidence_id}\0m2b-v4".encode("utf-8")
            ).hexdigest()[:32]
            connection.execute(
                """
                UPDATE career_evidence
                SET scope_key = ?, update_id = ?, content_digest = ?,
                    revision = 1, valid_from = ?
                WHERE id = ?
                """,
                (
                    career_evidence_scope_key(str(evidence_id)),
                    migration_update_id,
                    intent_content_digest(str(claim)),
                    now,
                    evidence_id,
                ),
            )
        connection.executescript(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS career_evidence_scope_revision_idx
                ON career_evidence(user_id, scope_key, revision)
                WHERE scope_key IS NOT NULL;
            CREATE UNIQUE INDEX IF NOT EXISTS career_evidence_update_id_idx
                ON career_evidence(update_id)
                WHERE update_id IS NOT NULL;
            CREATE UNIQUE INDEX IF NOT EXISTS career_evidence_active_scope_idx
                ON career_evidence(user_id, scope_key)
                WHERE verification_status = 'confirmed'
                  AND superseded_by IS NULL
    ;
            """
        )

    @staticmethod
    def _ensure_lineage_event_schema(connection: sqlite3.Connection) -> None:
        row = connection.execute(
            """
            SELECT sql FROM sqlite_master
            WHERE type = 'table' AND name = 'career_evidence_events'
            """
        ).fetchone()
        if row is not None and "corrected" in str(row[0]):
            return
        connection.execute(
            "ALTER TABLE career_evidence_events RENAME TO career_evidence_events_v3"
        )
        connection.execute(
            """
            CREATE TABLE career_evidence_events (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                career_evidence_id TEXT NOT NULL REFERENCES career_evidence(id),
                event_type TEXT NOT NULL CHECK (
                    event_type IN (
                        'created', 'confirmed', 'rejected', 'superseded',
                        'corrected', 'rolled_back', 'restored'
                    )
                ),
                previous_status TEXT CHECK (
                    previous_status IS NULL
                    OR previous_status IN ('pending', 'confirmed', 'rejected')
                ),
                new_status TEXT NOT NULL CHECK (
                    new_status IN ('pending', 'confirmed', 'rejected')
                ),
                actor_type TEXT NOT NULL CHECK (
                    actor_type IN ('user', 'agent', 'system')
                ),
                reason TEXT,
                mutation_id TEXT,
                related_evidence_id TEXT REFERENCES career_evidence(id),
                occurred_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO career_evidence_events(
                id, user_id, career_evidence_id, event_type,
                previous_status, new_status, actor_type, reason,
                mutation_id, related_evidence_id, occurred_at
            )
            SELECT id, user_id, career_evidence_id, event_type,
                   previous_status, new_status, actor_type, reason,
                   NULL, NULL, occurred_at
            FROM career_evidence_events_v3
            """
        )
        connection.execute("DROP TABLE career_evidence_events_v3")
        connection.execute(
            """
            CREATE INDEX career_evidence_events_evidence_idx
            ON career_evidence_events(career_evidence_id, occurred_at)
            """
        )

    @staticmethod
    def _ensure_source_refs(connection: sqlite3.Connection) -> None:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(career_evidence)")
        }
        if "source_ref" not in columns:
            connection.execute("ALTER TABLE career_evidence ADD COLUMN source_ref TEXT")
        rows = connection.execute(
            """
            SELECT id, user_id, source_resume_version_id, source_locator
            FROM career_evidence
            WHERE source_resume_version_id IS NOT NULL AND source_ref IS NULL
            """
        ).fetchall()
        for evidence_id, user_id, resume_version_id, source_locator in rows:
            connection.execute(
                "UPDATE career_evidence SET source_ref = ? WHERE id = ?",
                (
                    career_evidence_source_ref(
                        user_id=user_id,
                        evidence_id=evidence_id,
                        source_resume_version_id=resume_version_id,
                        source_locator=source_locator,
                    ),
                    evidence_id,
                ),
            )
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS career_evidence_source_ref_unique_idx
            ON career_evidence(source_ref) WHERE source_ref IS NOT NULL
            """
        )

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
        except Exception:
            connection.rollback()
            raise
        else:
            connection.commit()
        finally:
            connection.close()

    @staticmethod
    def _record(row: tuple[object, ...]) -> CareerRecord:
        return CareerRecord(
            id=row[0],
            user_id=row[1],
            record_type=row[2],
            organization=row[3],
            title=row[4],
            start_year=row[5],
            start_month=row[6],
            end_year=row[7],
            end_month=row[8],
            is_current=bool(row[9]),
            created_at=row[10],
            updated_at=row[11],
        )

    @staticmethod
    def _evidence(row: tuple[object, ...]) -> CareerEvidence:
        return CareerEvidence(
            id=row[0],
            user_id=row[1],
            career_record_id=row[2],
            claim=row[3],
            origin=row[4],
            verification_status=row[5],
            source_resume_version_id=row[6],
            source_locator=row[7],
            source_quote=row[8],
            source_ref=row[9],
            detail_ref=row[10],
            scope_key=row[11],
            update_id=row[12],
            content_digest=row[13],
            revision=row[14],
            valid_from=row[15],
            supersedes_id=row[16],
            superseded_at=row[17],
            superseded_by=row[18],
            tombstoned_at=row[19],
            tombstoned_by=row[20],
            tombstone_reason=row[21],
            created_at=row[22],
            updated_at=row[23],
        )

    @staticmethod
    def _event(row: tuple[object, ...]) -> CareerEvidenceEvent:
        return CareerEvidenceEvent(
            id=row[0],
            user_id=row[1],
            career_evidence_id=row[2],
            event_type=row[3],
            previous_status=row[4],
            new_status=row[5],
            actor_type=row[6],
            reason=row[7],
            related_evidence_id=row[8],
            occurred_at=row[9],
        )

    @classmethod
    def _records_by_ids(
        cls, connection: sqlite3.Connection, ids: list[str]
    ) -> tuple[CareerRecord, ...]:
        records = []
        for record_id in ids:
            row = connection.execute(
                """
                SELECT id, user_id, record_type, organization, title,
                       start_year, start_month, end_year, end_month, is_current,
                       created_at, updated_at
                FROM career_records WHERE id = ?
                """,
                (record_id,),
            ).fetchone()
            if row is None:
                raise ValueError("Imported career record is missing.")
            records.append(cls._record(row))
        return tuple(records)

    @classmethod
    def _evidence_by_ids(
        cls, connection: sqlite3.Connection, ids: list[str]
    ) -> tuple[CareerEvidence, ...]:
        evidence_items = []
        for evidence_id in ids:
            row = connection.execute(
                f"""
                SELECT {_EVIDENCE_COLUMNS}
                FROM career_evidence WHERE id = ?
                """,
                (evidence_id,),
            ).fetchone()
            if row is None:
                raise ValueError("Imported career evidence is missing.")
            evidence_items.append(cls._evidence(row))
        return tuple(evidence_items)
