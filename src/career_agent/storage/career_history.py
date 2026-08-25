from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sqlite3
from typing import Literal
from uuid import uuid4

from career_agent.domain.career_history import (
    CareerEvidence,
    CareerEvidenceEvent,
    CareerRecord,
)
from career_agent.agent.resume_analysis_contracts import ResumeAnalysisResult


EvidenceOrigin = Literal["resume_extraction", "user_input", "agent_inference"]
EvidenceStatus = Literal["pending", "confirmed", "rejected"]
RecordType = Literal["education", "work", "internship", "project", "certification"]


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
            self._migrate(connection)
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

    def list_records(self, *, user_id: str) -> tuple[CareerRecord, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, user_id, record_type, organization, title,
                       start_year, start_month, end_year, end_month, is_current,
                       created_at, updated_at
                FROM career_records
                WHERE user_id = ?
                ORDER BY is_current DESC,
                         COALESCE(start_year, 0) DESC,
                         COALESCE(start_month, 0) DESC,
                         created_at DESC
                """,
                (user_id,),
            ).fetchall()
        return tuple(self._record(row) for row in rows)

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
        evidence = CareerEvidence(
            id=f"career_evidence_{uuid4().hex}",
            user_id=user_id,
            career_record_id=career_record_id,
            claim=claim,
            origin=origin,
            verification_status="pending",
            source_resume_version_id=source_resume_version_id,
            source_locator=source_locator,
            source_quote=source_quote,
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
                    source_locator, source_quote, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                """
                SELECT id, user_id, career_record_id, claim, origin,
                       verification_status, source_resume_version_id,
                       source_locator, source_quote, created_at, updated_at
                FROM career_evidence
                WHERE id = ? AND user_id = ?
                """,
                (career_evidence_id, user_id),
            ).fetchone()
        return self._evidence(row) if row else None

    def list_evidence(
        self,
        *,
        user_id: str,
        career_record_id: str | None = None,
        verification_status: EvidenceStatus | None = None,
    ) -> tuple[CareerEvidence, ...]:
        query = """
            SELECT id, user_id, career_record_id, claim, origin,
                   verification_status, source_resume_version_id,
                   source_locator, source_quote, created_at, updated_at
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
        query += " ORDER BY created_at, id"

        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(self._evidence(row) for row in rows)

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
                       event.actor_type, event.reason, event.occurred_at
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
                    evidence = CareerEvidence(
                        id=f"career_evidence_{uuid4().hex}",
                        user_id=user_id,
                        career_record_id=record.id,
                        claim=claim,
                        origin="resume_extraction",
                        verification_status="confirmed",
                        source_resume_version_id=resume_version_id,
                        source_locator=locator,
                        source_quote=quote,
                        created_at=now,
                        updated_at=now,
                    )
                    connection.execute(
                        """
                        INSERT INTO career_evidence(
                            id, user_id, career_record_id, claim, origin,
                            verification_status, source_resume_version_id,
                            source_locator, source_quote, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                """
                SELECT id, user_id, career_record_id, claim, origin,
                       verification_status, source_resume_version_id,
                       source_locator, source_quote, created_at, updated_at
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
            updated = current.model_copy(
                update={"verification_status": new_status, "updated_at": now}
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
                SET verification_status = ?, updated_at = ?
                WHERE id = ? AND user_id = ? AND verification_status = 'pending'
                """,
                (
                    updated.verification_status,
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
    def _insert_event(
        connection: sqlite3.Connection, event: CareerEvidenceEvent
    ) -> None:
        connection.execute(
            """
            INSERT INTO career_evidence_events(
                id, user_id, career_evidence_id, event_type,
                previous_status, new_status, actor_type, reason, occurred_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                event.occurred_at.isoformat(),
            ),
        )

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
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                CHECK (source_locator IS NULL OR source_resume_version_id IS NOT NULL),
                CHECK (source_quote IS NULL OR source_resume_version_id IS NOT NULL),
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
                    event_type IN ('created', 'confirmed', 'rejected')
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

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
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
            created_at=row[9],
            updated_at=row[10],
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
            occurred_at=row[8],
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
                """
                SELECT id, user_id, career_record_id, claim, origin,
                       verification_status, source_resume_version_id,
                       source_locator, source_quote, created_at, updated_at
                FROM career_evidence WHERE id = ?
                """,
                (evidence_id,),
            ).fetchone()
            if row is None:
                raise ValueError("Imported career evidence is missing.")
            evidence_items.append(cls._evidence(row))
        return tuple(evidence_items)
