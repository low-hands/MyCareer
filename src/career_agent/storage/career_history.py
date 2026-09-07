from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Literal
from uuid import uuid4

from career_agent.domain.career_history import (
    CareerEvidence,
    CareerEvidenceCorrection,
    CareerEvidenceEvent,
    CareerEvidenceInvariantReport,
    CareerEvidenceInvariantViolation,
    CareerEvidenceMutationSnapshot,
    CareerEvidencePreimage,
    CareerRecord,
    career_evidence_detail_ref,
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
    "mutation_id",
    "rolled_back_at",
    "created_at",
    "updated_at",
)
_EVIDENCE_COLUMNS = ", ".join(_EVIDENCE_FIELD_NAMES)


@dataclass(frozen=True)
class CareerHistoryImportResult:
    records: tuple[CareerRecord, ...]
    evidence: tuple[CareerEvidence, ...]


class CareerEvidenceInvariantError(RuntimeError):
    def __init__(self, report: CareerEvidenceInvariantReport) -> None:
        super().__init__(
            f"Career evidence invariant check failed with "
            f"{len(report.violations)} violation(s)."
        )
        self.report = report


class CareerHistoryStore:
    def __init__(
        self,
        path: Path,
        *,
        validate_invariants: bool = True,
    ) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        with self._connect() as connection:
            apply_schema(
                connection,
                "career_history",
                5,
                self._migrate,
                upgrades={
                    3: self._upgrade_to_v3,
                    4: self._upgrade_to_v4,
                    5: self._upgrade_to_v5,
                },
            )
        os.chmod(self.path, 0o600)
        if validate_invariants:
            report = self.detect_evidence_invariant_violations()
            if not report.valid:
                raise CareerEvidenceInvariantError(report)

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
                    id, user_id, career_record_id, claim, origin,
                    verification_status, source_resume_version_id,
                    source_locator, source_quote, source_ref, detail_ref,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence.id,
                    evidence.user_id,
                    evidence.career_record_id,
                    evidence.claim,
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
                WHERE id = ? AND user_id = ?
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
                      AND (
                          evidence.superseded_by IS NOT NULL
                          OR evidence.rolled_back_at IS NOT NULL
                      )
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
                  AND (
                      evidence.superseded_by IS NOT NULL
                      OR evidence.rolled_back_at IS NOT NULL
                  )
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

    def list_evidence(
        self,
        *,
        user_id: str,
        career_record_id: str | None = None,
        verification_status: EvidenceStatus | None = None,
        source_resume_version_id: str | None = None,
        include_historical: bool = False,
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
                "(superseded_by IS NULL AND rolled_back_at IS NULL))"
            )
        query += " ORDER BY created_at, id"
        if limit is not None:
            query += " LIMIT ?"
            parameters.append(limit)

        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
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
        if verification_status is not None:
            query += " AND verification_status = ?"
            parameters.append(verification_status)
        if not include_historical:
            query += (
                " AND (verification_status != 'confirmed' OR "
                "(superseded_by IS NULL AND rolled_back_at IS NULL))"
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
                       event.actor_type, event.reason, event.mutation_id,
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
                  AND rolled_back_at IS NULL
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
                ORDER BY revision, created_at, id
                """,
                (user_id, scope_key),
            ).fetchall()
        return tuple(self._evidence(row) for row in rows)

    def correct_evidence(
        self,
        *,
        user_id: str,
        career_evidence_id: str,
        new_claim: str,
        reason: str,
    ) -> CareerEvidenceCorrection:
        """Apply one source-bound correction with its durable preimage."""

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
            mutation_id = f"career_evidence_mutation_{uuid4().hex}"
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
            preimage = CareerEvidencePreimage(
                scope_key=current.scope_key,
                active_evidence_id=current.id,
                active_revision=current.revision,
            )
            snapshot = CareerEvidenceMutationSnapshot(
                id=mutation_id,
                user_id=user_id,
                scope_key=current.scope_key,
                mutation_type="correction",
                status="applied",
                preimage=preimage,
                replacement_evidence_id=replacement_id,
                reason=correction_reason,
                created_at=now,
            )
            connection.execute(
                """
                INSERT INTO career_evidence_mutations(
                    id, user_id, scope_key, mutation_type, status,
                    preimage_json, replacement_evidence_id, reason,
                    created_at, rolled_back_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    snapshot.id,
                    snapshot.user_id,
                    snapshot.scope_key,
                    snapshot.mutation_type,
                    snapshot.status,
                    snapshot.preimage.model_dump_json(),
                    snapshot.replacement_evidence_id,
                    snapshot.reason,
                    snapshot.created_at.isoformat(),
                ),
            )
            replacement = CareerEvidence(
                id=replacement_id,
                user_id=user_id,
                career_record_id=current.career_record_id,
                claim=claim,
                origin="user_input",
                verification_status="confirmed",
                detail_ref=career_evidence_detail_ref(
                    user_id=user_id,
                    evidence_id=replacement_id,
                ),
                scope_key=current.scope_key,
                update_id=f"career_evidence_update_{uuid4().hex}",
                content_digest=intent_content_digest(claim),
                revision=latest_revision + 1,
                valid_from=now,
                supersedes_id=current.id,
                mutation_id=mutation_id,
                created_at=now,
                updated_at=now,
            )
            connection.execute(
                """
                UPDATE career_evidence
                SET superseded_at = ?, superseded_by = ?, updated_at = ?
                WHERE id = ? AND user_id = ?
                  AND superseded_by IS NULL AND rolled_back_at IS NULL
                """,
                (
                    now.isoformat(),
                    replacement.id,
                    now.isoformat(),
                    current.id,
                    user_id,
                ),
            )
            self._insert_evidence(connection, replacement)
            superseded = current.model_copy(
                update={
                    "superseded_at": now,
                    "superseded_by": replacement.id,
                    "updated_at": now,
                }
            )
            self._insert_event(
                connection,
                CareerEvidenceEvent(
                    id=f"career_evidence_event_{uuid4().hex}",
                    user_id=user_id,
                    career_evidence_id=current.id,
                    event_type="superseded",
                    previous_status="confirmed",
                    new_status="confirmed",
                    actor_type="user",
                    reason=correction_reason,
                    mutation_id=mutation_id,
                    related_evidence_id=replacement.id,
                    occurred_at=now,
                ),
            )
            self._insert_event(
                connection,
                CareerEvidenceEvent(
                    id=f"career_evidence_event_{uuid4().hex}",
                    user_id=user_id,
                    career_evidence_id=replacement.id,
                    event_type="corrected",
                    previous_status="confirmed",
                    new_status="confirmed",
                    actor_type="user",
                    reason=correction_reason,
                    mutation_id=mutation_id,
                    related_evidence_id=current.id,
                    occurred_at=now,
                ),
            )
        return CareerEvidenceCorrection(
            previous=superseded,
            current=replacement,
            snapshot=snapshot,
        )

    def get_evidence_mutation(
        self, *, user_id: str, mutation_id: str
    ) -> CareerEvidenceMutationSnapshot | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, user_id, scope_key, mutation_type, status,
                       preimage_json, replacement_evidence_id, reason,
                       created_at, rolled_back_at
                FROM career_evidence_mutations
                WHERE id = ? AND user_id = ?
                """,
                (mutation_id, user_id),
            ).fetchone()
        return self._mutation(row) if row is not None else None

    def list_evidence_mutations(
        self, *, user_id: str, scope_key: str | None = None
    ) -> tuple[CareerEvidenceMutationSnapshot, ...]:
        query = """
            SELECT id, user_id, scope_key, mutation_type, status,
                   preimage_json, replacement_evidence_id, reason,
                   created_at, rolled_back_at
            FROM career_evidence_mutations
            WHERE user_id = ?
        """
        parameters: list[str] = [user_id]
        if scope_key is not None:
            query += " AND scope_key = ?"
            parameters.append(scope_key)
        query += " ORDER BY created_at, id"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(self._mutation(row) for row in rows)

    def rollback_evidence_correction(
        self,
        *,
        user_id: str,
        mutation_id: str,
        reason: str,
    ) -> CareerEvidenceMutationSnapshot:
        """Compensate the latest correction from its durable preimage."""

        rollback_reason = reason.strip()
        if not rollback_reason:
            raise ValueError("Rollback reason is required.")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            mutation_row = connection.execute(
                """
                SELECT id, user_id, scope_key, mutation_type, status,
                       preimage_json, replacement_evidence_id, reason,
                       created_at, rolled_back_at
                FROM career_evidence_mutations
                WHERE id = ? AND user_id = ?
                """,
                (mutation_id, user_id),
            ).fetchone()
            if mutation_row is None:
                raise ValueError("Career evidence mutation not found.")
            snapshot = self._mutation(mutation_row)
            if snapshot.status == "rolled_back":
                return snapshot
            replacement_row = connection.execute(
                f"SELECT {_EVIDENCE_COLUMNS} FROM career_evidence WHERE id = ?",
                (snapshot.replacement_evidence_id,),
            ).fetchone()
            previous_row = connection.execute(
                f"SELECT {_EVIDENCE_COLUMNS} FROM career_evidence WHERE id = ?",
                (snapshot.preimage.active_evidence_id,),
            ).fetchone()
            if replacement_row is None or previous_row is None:
                raise ValueError("Correction snapshot references missing evidence.")
            replacement = self._evidence(replacement_row)
            previous = self._evidence(previous_row)
            active_ids = {
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT id FROM career_evidence
                    WHERE user_id = ? AND scope_key = ?
                      AND verification_status = 'confirmed'
                      AND superseded_by IS NULL
                      AND rolled_back_at IS NULL
                    """,
                    (user_id, snapshot.scope_key),
                ).fetchall()
            }
            has_later_successor = (
                replacement.superseded_by is not None
                and replacement.superseded_by in {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT id FROM career_evidence WHERE user_id = ?",
                        (user_id,),
                    ).fetchall()
                }
            )
            if (
                replacement.user_id != user_id
                or previous.user_id != user_id
                or replacement.scope_key != snapshot.scope_key
                or previous.scope_key != snapshot.scope_key
                or has_later_successor
                or replacement.rolled_back_at is not None
                or not active_ids.issubset({previous.id, replacement.id})
            ):
                raise ValueError(
                    "Only the latest internally consistent correction can be rolled back."
                )
            now = datetime.now(timezone.utc)
            connection.execute(
                """
                UPDATE career_evidence
                SET rolled_back_at = ?, superseded_at = NULL,
                    superseded_by = NULL, updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (now.isoformat(), now.isoformat(), replacement.id, user_id),
            )
            connection.execute(
                """
                UPDATE career_evidence
                SET superseded_at = NULL, superseded_by = NULL, updated_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (now.isoformat(), previous.id, user_id),
            )
            connection.execute(
                """
                UPDATE career_evidence_mutations
                SET status = 'rolled_back', rolled_back_at = ?
                WHERE id = ? AND user_id = ? AND status = 'applied'
                """,
                (now.isoformat(), snapshot.id, user_id),
            )
            for evidence, event_type, related_id in (
                (replacement, "rolled_back", previous.id),
                (previous, "restored", replacement.id),
            ):
                self._insert_event(
                    connection,
                    CareerEvidenceEvent(
                        id=f"career_evidence_event_{uuid4().hex}",
                        user_id=user_id,
                        career_evidence_id=evidence.id,
                        event_type=event_type,
                        previous_status="confirmed",
                        new_status="confirmed",
                        actor_type="system",
                        reason=rollback_reason,
                        mutation_id=snapshot.id,
                        related_evidence_id=related_id,
                        occurred_at=now,
                    ),
                )
        return snapshot.model_copy(
            update={"status": "rolled_back", "rolled_back_at": now}
        )

    def detect_evidence_invariant_violations(
        self, *, user_id: str | None = None
    ) -> CareerEvidenceInvariantReport:
        """Check the durable M2b invariants without trusting model validation."""

        with self._connect() as connection:
            where = " WHERE user_id = ?" if user_id is not None else ""
            parameters: tuple[str, ...] = (user_id,) if user_id is not None else ()
            rows = connection.execute(
                f"SELECT {_EVIDENCE_COLUMNS} FROM career_evidence{where}",
                parameters,
            ).fetchall()
            event_rows = connection.execute(
                """
                SELECT event_type, career_evidence_id, mutation_id,
                       related_evidence_id
                FROM career_evidence_events
                """
                + (" WHERE user_id = ?" if user_id is not None else ""),
                parameters,
            ).fetchall()
            mutation_rows = connection.execute(
                """
                SELECT id, user_id, scope_key, mutation_type, status,
                       preimage_json, replacement_evidence_id, reason,
                       created_at, rolled_back_at
                FROM career_evidence_mutations
                """
                + (" WHERE user_id = ?" if user_id is not None else ""),
                parameters,
            ).fetchall()

        items = {
            str(row[0]): dict(zip(_EVIDENCE_FIELD_NAMES, row, strict=True))
            for row in rows
        }
        events = {
            (str(event_type), str(evidence_id), mutation_id, related_id)
            for event_type, evidence_id, mutation_id, related_id in event_rows
        }
        # The store-open guard scans every user. Index event replay keys once
        # so each evidence row does O(1) lookups instead of rescanning the
        # complete event log and turning startup into O(n²).
        event_types_by_evidence: dict[str, set[str]] = {}
        superseded_event_links: set[tuple[str, str]] = set()
        for event_type, evidence_id, _, related_id in events:
            event_types_by_evidence.setdefault(evidence_id, set()).add(event_type)
            if event_type == "superseded" and related_id is not None:
                superseded_event_links.add((evidence_id, str(related_id)))
        mutation_ids = {str(row[0]) for row in mutation_rows}
        violations: list[CareerEvidenceInvariantViolation] = []
        seen_update_ids: dict[str, str] = {}

        def add(
            code: str,
            message: str,
            *,
            scope_key: str | None = None,
            evidence_ids: tuple[str, ...] = (),
            mutation_id: str | None = None,
        ) -> None:
            violations.append(
                CareerEvidenceInvariantViolation(
                    code=code,
                    message=message,
                    scope_key=scope_key,
                    evidence_ids=evidence_ids,
                    mutation_id=mutation_id,
                )
            )

        for evidence_id, item in items.items():
            version_parts = (
                item["scope_key"],
                item["update_id"],
                item["content_digest"],
                item["revision"],
                item["valid_from"],
            )
            versioned = all(part is not None for part in version_parts)
            if any(part is not None for part in version_parts) != versioned or (
                item["verification_status"] == "confirmed"
            ) != versioned:
                add(
                    "version_binding",
                    "Version binding does not match confirmed evidence state.",
                    scope_key=item["scope_key"],
                    evidence_ids=(evidence_id,),
                )
            if versioned and item["content_digest"] != intent_content_digest(
                str(item["claim"])
            ):
                add(
                    "version_binding",
                    "Stored content digest does not match the evidence claim.",
                    scope_key=item["scope_key"],
                    evidence_ids=(evidence_id,),
                )
            if item["update_id"] is not None:
                update_id = str(item["update_id"])
                prior_id = seen_update_ids.get(update_id)
                if prior_id is not None:
                    add(
                        "version_binding",
                        "Evidence update_id is not unique.",
                        scope_key=item["scope_key"],
                        evidence_ids=(prior_id, evidence_id),
                    )
                else:
                    seen_update_ids[update_id] = evidence_id
            for pointer_name in ("supersedes_id", "superseded_by"):
                target_id = item[pointer_name]
                if target_id is not None and str(target_id) not in items:
                    add(
                        "pointer_target_missing",
                        f"{pointer_name} references a missing evidence row.",
                        scope_key=item["scope_key"],
                        evidence_ids=(evidence_id, str(target_id)),
                        mutation_id=item["mutation_id"],
                    )
            predecessor_id = item["supersedes_id"]
            if predecessor_id is not None and item["mutation_id"] not in mutation_ids:
                add(
                    "snapshot_binding",
                    "Corrected evidence has no durable preimage snapshot.",
                    scope_key=item["scope_key"],
                    evidence_ids=(evidence_id,),
                    mutation_id=item["mutation_id"],
                )
            if predecessor_id is not None and str(predecessor_id) in items:
                predecessor = items[str(predecessor_id)]
                if item["rolled_back_at"] is None and (
                    predecessor["superseded_by"] != evidence_id
                ):
                    add(
                        "pointer_not_reciprocal",
                        "Correction back-pointer is not reciprocated by its predecessor.",
                        scope_key=item["scope_key"],
                        evidence_ids=(str(predecessor_id), evidence_id),
                        mutation_id=item["mutation_id"],
                    )
                if (
                    predecessor["user_id"] != item["user_id"]
                    or predecessor["scope_key"] != item["scope_key"]
                    or predecessor["career_record_id"] != item["career_record_id"]
                ):
                    add(
                        "scope_mismatch",
                        "Linked evidence rows do not share owner, record, and scope.",
                        scope_key=item["scope_key"],
                        evidence_ids=(str(predecessor_id), evidence_id),
                        mutation_id=item["mutation_id"],
                    )
                if (
                    predecessor["revision"] is not None
                    and item["revision"] is not None
                    and int(item["revision"]) <= int(predecessor["revision"])
                ):
                    add(
                        "revision_order",
                        "Correction revision is not newer than its predecessor.",
                        scope_key=item["scope_key"],
                        evidence_ids=(str(predecessor_id), evidence_id),
                        mutation_id=item["mutation_id"],
                    )
            successor_id = item["superseded_by"]
            if successor_id is not None and str(successor_id) in items:
                successor = items[str(successor_id)]
                if successor["supersedes_id"] != evidence_id:
                    add(
                        "pointer_not_reciprocal",
                        "Supersession forward-pointer is not reciprocated.",
                        scope_key=item["scope_key"],
                        evidence_ids=(evidence_id, str(successor_id)),
                        mutation_id=successor["mutation_id"],
                    )

        by_scope: dict[tuple[str, str], list[dict[str, object]]] = {}
        for item in items.values():
            if (
                item["verification_status"] == "confirmed"
                and item["scope_key"] is not None
            ):
                by_scope.setdefault(
                    (str(item["user_id"]), str(item["scope_key"])), []
                ).append(item)
        for (_, scope_key), scoped in by_scope.items():
            revisions = sorted(
                int(item["revision"])
                for item in scoped
                if item["revision"] is not None
            )
            if revisions != list(range(1, len(revisions) + 1)):
                add(
                    "revision_order",
                    "Evidence revisions are not a contiguous chronology.",
                    scope_key=scope_key,
                    evidence_ids=tuple(str(item["id"]) for item in scoped),
                )
            active = [
                item
                for item in scoped
                if item["superseded_by"] is None and item["rolled_back_at"] is None
            ]
            if len(active) != 1:
                add(
                    "active_count",
                    "A versioned evidence scope must have exactly one active row.",
                    scope_key=scope_key,
                    evidence_ids=tuple(str(item["id"]) for item in active),
                )

        for evidence_id, item in items.items():
            event_types = event_types_by_evidence.get(evidence_id, set())
            expected = (
                "rejected"
                if item["verification_status"] == "rejected"
                else "corrected"
                if item["supersedes_id"] is not None
                else "confirmed"
                if item["verification_status"] == "confirmed"
                else "created"
            )
            if expected not in event_types:
                add(
                    "event_replay",
                    f"Evidence state has no matching {expected} event.",
                    scope_key=item["scope_key"],
                    evidence_ids=(evidence_id,),
                    mutation_id=item["mutation_id"],
                )
            if item["superseded_by"] is not None and (
                evidence_id,
                str(item["superseded_by"]),
            ) not in superseded_event_links:
                add(
                    "event_replay",
                    "Superseded evidence has no matching lineage event.",
                    scope_key=item["scope_key"],
                    evidence_ids=(evidence_id, str(item["superseded_by"])),
                    mutation_id=item["mutation_id"],
                )
            if item["rolled_back_at"] is not None and "rolled_back" not in event_types:
                add(
                    "event_replay",
                    "Rolled-back evidence has no rollback event.",
                    scope_key=item["scope_key"],
                    evidence_ids=(evidence_id,),
                    mutation_id=item["mutation_id"],
                )

        for mutation_row in mutation_rows:
            try:
                mutation = self._mutation(mutation_row)
            except ValueError as error:
                add("snapshot_binding", f"Invalid mutation snapshot: {error}")
                continue
            previous = items.get(mutation.preimage.active_evidence_id)
            replacement = items.get(mutation.replacement_evidence_id)
            if (
                previous is None
                or replacement is None
                or replacement["scope_key"] != mutation.scope_key
                or replacement["mutation_id"] != mutation.id
                or replacement["supersedes_id"]
                != mutation.preimage.active_evidence_id
                or (
                    previous is not None
                    and previous["revision"] != mutation.preimage.active_revision
                )
            ):
                add(
                    "snapshot_binding",
                    "Mutation snapshot does not bind its preimage and replacement.",
                    scope_key=mutation.scope_key,
                    evidence_ids=(
                        mutation.preimage.active_evidence_id,
                        mutation.replacement_evidence_id,
                    ),
                    mutation_id=mutation.id,
                )
            if mutation.status == "rolled_back":
                restored = (
                    "restored",
                    mutation.preimage.active_evidence_id,
                    mutation.id,
                    mutation.replacement_evidence_id,
                )
                if replacement is None or replacement["rolled_back_at"] is None:
                    add(
                        "snapshot_binding",
                        "Rolled-back mutation still has a visible replacement.",
                        scope_key=mutation.scope_key,
                        evidence_ids=(mutation.replacement_evidence_id,),
                        mutation_id=mutation.id,
                    )
                if restored not in events:
                    add(
                        "event_replay",
                        "Rolled-back mutation has no restoration event.",
                        scope_key=mutation.scope_key,
                        evidence_ids=(mutation.preimage.active_evidence_id,),
                        mutation_id=mutation.id,
                    )
            elif replacement is not None and replacement["rolled_back_at"] is not None:
                add(
                    "snapshot_binding",
                    "Applied mutation points to a rolled-back replacement.",
                    scope_key=mutation.scope_key,
                    evidence_ids=(mutation.replacement_evidence_id,),
                    mutation_id=mutation.id,
                )

        return CareerEvidenceInvariantReport(
            user_id=user_id,
            checked_at=datetime.now(timezone.utc),
            violations=tuple(violations),
        )

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
                            id, user_id, career_record_id, claim, origin,
                            verification_status, source_resume_version_id,
                            source_locator, source_quote, source_ref, detail_ref,
                            scope_key, update_id, content_digest, revision,
                            valid_from, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            evidence.id,
                            evidence.user_id,
                            evidence.career_record_id,
                            evidence.claim,
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
                id, user_id, career_record_id, claim, origin,
                verification_status, source_resume_version_id,
                source_locator, source_quote, source_ref, detail_ref,
                scope_key, update_id, content_digest, revision, valid_from, supersedes_id,
                superseded_at, superseded_by, mutation_id, rolled_back_at,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evidence.id,
                evidence.user_id,
                evidence.career_record_id,
                evidence.claim,
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
                evidence.mutation_id,
                evidence.rolled_back_at.isoformat() if evidence.rolled_back_at else None,
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
                previous_status, new_status, actor_type, reason, mutation_id,
                related_evidence_id, occurred_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                event.mutation_id,
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
                mutation_id TEXT,
                rolled_back_at TEXT,
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
            );

            CREATE TABLE IF NOT EXISTS career_evidence_mutations (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                scope_key TEXT NOT NULL,
                mutation_type TEXT NOT NULL CHECK (mutation_type = 'correction'),
                status TEXT NOT NULL CHECK (status IN ('applied', 'rolled_back')),
                preimage_json TEXT NOT NULL,
                replacement_evidence_id TEXT NOT NULL REFERENCES career_evidence(id)
                    DEFERRABLE INITIALLY DEFERRED,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL,
                rolled_back_at TEXT
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
            CREATE INDEX IF NOT EXISTS career_evidence_mutations_user_idx
                ON career_evidence_mutations(user_id, created_at DESC);
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
            "mutation_id": "TEXT",
            "rolled_back_at": "TEXT",
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
                  AND rolled_back_at IS NULL;
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

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

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
            mutation_id=row[19],
            rolled_back_at=row[20],
            created_at=row[21],
            updated_at=row[22],
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
            mutation_id=row[8],
            related_evidence_id=row[9],
            occurred_at=row[10],
        )

    @staticmethod
    def _mutation(row: tuple[object, ...]) -> CareerEvidenceMutationSnapshot:
        return CareerEvidenceMutationSnapshot(
            id=row[0],
            user_id=row[1],
            scope_key=row[2],
            mutation_type=row[3],
            status=row[4],
            preimage=CareerEvidencePreimage.model_validate_json(row[5]),
            replacement_evidence_id=row[6],
            reason=row[7],
            created_at=row[8],
            rolled_back_at=row[9],
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
