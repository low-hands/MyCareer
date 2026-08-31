from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from uuid import uuid4

from career_agent.domain.interviews import (
    InterviewDetails,
    InterviewRetroQuestion,
    InterviewRetroReport,
    InterviewRound,
    InterviewRoundEvent,
    InterviewStatus,
)
from career_agent.storage.schema import apply_schema


class SQLiteInterviewStore:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            apply_schema(
                connection,
                "interviews",
                2,
                self._migrate,
                upgrades={2: self._upgrade_v2},
            )
        os.chmod(self.path, 0o600)

    def create(
        self,
        *,
        user_id: str,
        application_id: str,
        details: InterviewDetails,
        source: str,
        email_event_id: str | None,
        source_thread_id: str | None,
        occurred_at: datetime,
    ) -> InterviewRound:
        now = datetime.now(timezone.utc)
        status: InterviewStatus = (
            "cancelled"
            if details.change_type == "cancelled"
            else "scheduled"
            if details.scheduled_start is not None
            else "identified"
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            sequence_number = int(
                connection.execute(
                    """
                    SELECT COALESCE(MAX(sequence_number), 0) + 1
                    FROM interview_rounds
                    WHERE user_id = ? AND application_id = ?
                    """,
                    (user_id, application_id),
                ).fetchone()[0]
            )
            round_ = InterviewRound(
                id=f"interview_round_{uuid4().hex}",
                user_id=user_id,
                application_id=application_id,
                sequence_number=sequence_number,
                employer_label=details.employer_label,
                status=status,
                scheduled_start=details.scheduled_start,
                scheduled_end=details.scheduled_end,
                timezone=details.timezone,
                interview_format=details.interview_format,
                location=details.location,
                meeting_url=details.meeting_url,
                contact_summary=details.contact_summary,
                created_at=now,
                updated_at=now,
            )
            connection.execute(
                """
                INSERT INTO interview_rounds(
                    id, user_id, application_id, sequence_number, employer_label,
                    status, scheduled_start, scheduled_end, timezone,
                    interview_format, location, meeting_url, contact_summary,
                    created_at, updated_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._round_values(round_),
            )
            self._insert_event(
                connection,
                self._new_event(
                    round_=round_,
                    source=source,
                    event_type="created",
                    email_event_id=email_event_id,
                    source_thread_id=source_thread_id,
                    details=details,
                    occurred_at=occurred_at,
                ),
            )
        return round_

    def update(
        self,
        *,
        round_: InterviewRound,
        details: InterviewDetails,
        source: str,
        email_event_id: str | None,
        source_thread_id: str | None,
        occurred_at: datetime,
    ) -> InterviewRound:
        now = datetime.now(timezone.utc)
        status: InterviewStatus = round_.status
        if round_.status == "completed":
            status = "completed"
        elif details.change_type == "cancelled":
            status = "cancelled"
        elif details.scheduled_start is not None:
            status = "scheduled"
        elif round_.status == "cancelled":
            status = "identified"
        updated = round_.model_copy(
            update={
                "employer_label": details.employer_label or round_.employer_label,
                "status": status,
                "scheduled_start": (
                    details.scheduled_start
                    if details.scheduled_start is not None
                    else round_.scheduled_start
                ),
                "scheduled_end": (
                    details.scheduled_end
                    if details.scheduled_start is not None
                    else round_.scheduled_end
                ),
                "timezone": details.timezone or round_.timezone,
                "interview_format": (
                    details.interview_format
                    if details.interview_format != "unknown"
                    else round_.interview_format
                ),
                "location": details.location or round_.location,
                "meeting_url": details.meeting_url or round_.meeting_url,
                "contact_summary": details.contact_summary or round_.contact_summary,
                "updated_at": now,
            }
        )
        event_type = {
            "invited": "details_updated",
            "rescheduled": "rescheduled",
            "details_updated": "details_updated",
            "cancelled": "cancelled",
        }[details.change_type]
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            changed = connection.execute(
                """
                UPDATE interview_rounds SET
                    employer_label = ?, status = ?, scheduled_start = ?,
                    scheduled_end = ?, timezone = ?, interview_format = ?,
                    location = ?, meeting_url = ?, contact_summary = ?, updated_at = ?
                WHERE id = ? AND user_id = ? AND updated_at = ?
                """,
                (
                    updated.employer_label,
                    updated.status,
                    self._iso(updated.scheduled_start),
                    self._iso(updated.scheduled_end),
                    updated.timezone,
                    updated.interview_format,
                    updated.location,
                    updated.meeting_url,
                    updated.contact_summary,
                    updated.updated_at.isoformat(),
                    updated.id,
                    updated.user_id,
                    round_.updated_at.isoformat(),
                ),
            ).rowcount
            if not changed:
                raise RuntimeError("interview round changed concurrently")
            self._insert_event(
                connection,
                self._new_event(
                    round_=updated,
                    source=source,
                    event_type=event_type,
                    email_event_id=email_event_id,
                    source_thread_id=source_thread_id,
                    details=details,
                    occurred_at=occurred_at,
                ),
            )
        return updated

    def get(self, *, user_id: str, interview_round_id: str) -> InterviewRound | None:
        with self._connect() as connection:
            row = connection.execute(
                self._ROUND_SELECT + " WHERE id = ? AND user_id = ?",
                (interview_round_id, user_id),
            ).fetchone()
        return self._round(row) if row else None

    def complete(
        self,
        *,
        round_: InterviewRound,
        occurred_at: datetime,
    ) -> InterviewRound:
        if round_.status == "cancelled":
            raise ValueError("cancelled interview cannot be completed")
        if round_.status == "completed":
            return round_
        now = datetime.now(timezone.utc)
        updated = round_.model_copy(
            update={
                "status": "completed",
                "completed_at": occurred_at,
                "updated_at": now,
            }
        )
        details = InterviewDetails(
            change_type="details_updated",
            employer_label=updated.employer_label,
            scheduled_start=updated.scheduled_start,
            scheduled_end=updated.scheduled_end,
            timezone=updated.timezone,
            interview_format=updated.interview_format,
            location=updated.location,
            meeting_url=updated.meeting_url,
            contact_summary=updated.contact_summary,
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            changed = connection.execute(
                """
                UPDATE interview_rounds
                SET status = 'completed', completed_at = ?, updated_at = ?
                WHERE id = ? AND user_id = ? AND updated_at = ?
                """,
                (
                    occurred_at.isoformat(),
                    now.isoformat(),
                    round_.id,
                    round_.user_id,
                    round_.updated_at.isoformat(),
                ),
            ).rowcount
            if not changed:
                raise RuntimeError("interview round changed concurrently")
            self._insert_event(
                connection,
                self._new_event(
                    round_=updated,
                    source="user_reported",
                    event_type="completed",
                    email_event_id=None,
                    source_thread_id=None,
                    details=details,
                    occurred_at=occurred_at,
                ),
            )
        return updated

    def list(
        self,
        *,
        user_id: str,
        application_id: str | None = None,
        statuses: tuple[InterviewStatus, ...] = (),
        limit: int = 50,
    ) -> tuple[InterviewRound, ...]:
        query = self._ROUND_SELECT + " WHERE user_id = ?"
        params: list[object] = [user_id]
        if application_id is not None:
            query += " AND application_id = ?"
            params.append(application_id)
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            query += f" AND status IN ({placeholders})"
            params.extend(statuses)
        query += " ORDER BY COALESCE(scheduled_start, created_at), sequence_number LIMIT ?"
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return tuple(self._round(row) for row in rows)

    def list_events(
        self, *, user_id: str, interview_round_id: str
    ) -> tuple[InterviewRoundEvent, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, user_id, application_id, interview_round_id, source,
                       event_type, email_event_id, source_thread_id, details_json,
                       occurred_at
                FROM interview_round_events
                WHERE user_id = ? AND interview_round_id = ?
                ORDER BY occurred_at, rowid
                """,
                (user_id, interview_round_id),
            ).fetchall()
        return tuple(self._event(row) for row in rows)

    def record_retro(
        self,
        *,
        round_: InterviewRound,
        source_notes: str,
        summary: str,
        questions: tuple[InterviewRetroQuestion, ...] = (),
        strengths: tuple[str, ...] = (),
        difficulties: tuple[str, ...] = (),
        interviewer_signals: tuple[str, ...] = (),
        next_focus: tuple[str, ...] = (),
        action_items: tuple[str, ...] = (),
        limitations: tuple[str, ...] = (),
        self_assessment: str = "uncertain",
        created_at: datetime | None = None,
    ) -> InterviewRetroReport:
        timestamp = created_at or datetime.now(timezone.utc)
        canonical = json.dumps(
            {
                "source_notes": source_notes,
                "summary": summary,
                "questions": [item.model_dump(mode="json") for item in questions],
                "strengths": strengths,
                "difficulties": difficulties,
                "interviewer_signals": interviewer_signals,
                "next_focus": next_focus,
                "action_items": action_items,
                "limitations": limitations,
                "self_assessment": self_assessment,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        report = InterviewRetroReport(
            id=f"interview_retro_{uuid4().hex}",
            user_id=round_.user_id,
            application_id=round_.application_id,
            interview_round_id=round_.id,
            source_notes=source_notes,
            summary=summary,
            questions=questions,
            strengths=strengths,
            difficulties=difficulties,
            interviewer_signals=interviewer_signals,
            next_focus=next_focus,
            action_items=action_items,
            limitations=limitations,
            self_assessment=self_assessment,
            content_sha256=digest,
            created_at=timestamp,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO interview_retro_reports(
                    id, user_id, application_id, interview_round_id,
                    source_notes, summary, questions_json, strengths_json,
                    difficulties_json, interviewer_signals_json, next_focus_json,
                    action_items_json, limitations_json, self_assessment,
                    content_sha256, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.id,
                    report.user_id,
                    report.application_id,
                    report.interview_round_id,
                    report.source_notes,
                    report.summary,
                    json.dumps(
                        [item.model_dump(mode="json") for item in report.questions],
                        ensure_ascii=False,
                    ),
                    json.dumps(report.strengths, ensure_ascii=False),
                    json.dumps(report.difficulties, ensure_ascii=False),
                    json.dumps(report.interviewer_signals, ensure_ascii=False),
                    json.dumps(report.next_focus, ensure_ascii=False),
                    json.dumps(report.action_items, ensure_ascii=False),
                    json.dumps(report.limitations, ensure_ascii=False),
                    report.self_assessment,
                    report.content_sha256,
                    report.created_at.isoformat(),
                ),
            )
            row = connection.execute(
                self._RETRO_SELECT
                + " WHERE user_id = ? AND interview_round_id = ? AND content_sha256 = ?",
                (round_.user_id, round_.id, digest),
            ).fetchone()
        if row is None:
            raise RuntimeError("interview retro report was not persisted")
        return self._retro(row)

    def list_retros(
        self,
        *,
        user_id: str,
        interview_round_id: str,
        limit: int = 50,
    ) -> tuple[InterviewRetroReport, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                self._RETRO_SELECT
                + " WHERE user_id = ? AND interview_round_id = ? "
                "ORDER BY created_at, rowid LIMIT ?",
                (user_id, interview_round_id, limit),
            ).fetchall()
        return tuple(self._retro(row) for row in rows)

    def get_retro(
        self, *, user_id: str, retro_report_id: str
    ) -> InterviewRetroReport | None:
        """Read one immutable retrospective for historical presentation."""
        with self._connect() as connection:
            row = connection.execute(
                self._RETRO_SELECT + " WHERE id = ? AND user_id = ?",
                (retro_report_id, user_id),
            ).fetchone()
        return self._retro(row) if row else None

    def find_by_source_thread(
        self, *, user_id: str, application_id: str, source_thread_id: str
    ) -> InterviewRound | None:
        with self._connect() as connection:
            row = connection.execute(
                self._ROUND_SELECT
                + """
                  JOIN interview_round_events e ON e.interview_round_id = interview_rounds.id
                  WHERE interview_rounds.user_id = ?
                    AND interview_rounds.application_id = ?
                    AND e.source_thread_id = ?
                  ORDER BY e.occurred_at DESC LIMIT 1
                  """,
                (user_id, application_id, source_thread_id),
            ).fetchone()
        return self._round(row) if row else None

    def find_by_identity(
        self,
        *,
        user_id: str,
        application_id: str,
        scheduled_start: datetime | None,
        meeting_url: str | None,
    ) -> tuple[InterviewRound, ...]:
        clauses = []
        params: list[object] = [user_id, application_id]
        if scheduled_start is not None:
            clauses.append("scheduled_start = ?")
            params.append(scheduled_start.isoformat())
        if meeting_url is not None:
            clauses.append("meeting_url = ?")
            params.append(meeting_url)
        if not clauses:
            return ()
        query = (
            self._ROUND_SELECT
            + " WHERE user_id = ? AND application_id = ? AND ("
            + " OR ".join(clauses)
            + ")"
        )
        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return tuple(self._round(row) for row in rows)

    def find_for_email_event(
        self, *, user_id: str, email_event_id: str
    ) -> InterviewRound | None:
        with self._connect() as connection:
            row = connection.execute(
                self._ROUND_SELECT
                + """
                  JOIN interview_round_events e ON e.interview_round_id = interview_rounds.id
                  WHERE interview_rounds.user_id = ? AND e.email_event_id = ?
                  LIMIT 1
                  """,
                (user_id, email_event_id),
            ).fetchone()
        return self._round(row) if row else None

    _ROUND_SELECT = (
        "SELECT interview_rounds.id, interview_rounds.user_id, "
        "interview_rounds.application_id, interview_rounds.sequence_number, "
        "interview_rounds.employer_label, interview_rounds.status, "
        "interview_rounds.scheduled_start, interview_rounds.scheduled_end, "
        "interview_rounds.timezone, interview_rounds.interview_format, "
        "interview_rounds.location, interview_rounds.meeting_url, "
        "interview_rounds.contact_summary, interview_rounds.created_at, "
        "interview_rounds.updated_at, interview_rounds.completed_at "
        "FROM interview_rounds"
    )
    _RETRO_SELECT = (
        "SELECT id, user_id, application_id, interview_round_id, source_notes, "
        "summary, questions_json, strengths_json, difficulties_json, "
        "interviewer_signals_json, next_focus_json, action_items_json, "
        "limitations_json, self_assessment, content_sha256, created_at "
        "FROM interview_retro_reports"
    )

    def _migrate(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS interview_rounds (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                application_id TEXT NOT NULL,
                sequence_number INTEGER NOT NULL,
                employer_label TEXT,
                status TEXT NOT NULL,
                scheduled_start TEXT,
                scheduled_end TEXT,
                timezone TEXT,
                interview_format TEXT NOT NULL,
                location TEXT,
                meeting_url TEXT,
                contact_summary TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT,
                UNIQUE(user_id, application_id, sequence_number)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS interview_round_events (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                application_id TEXT NOT NULL,
                interview_round_id TEXT NOT NULL REFERENCES interview_rounds(id),
                source TEXT NOT NULL,
                event_type TEXT NOT NULL,
                email_event_id TEXT UNIQUE,
                source_thread_id TEXT,
                details_json TEXT NOT NULL,
                occurred_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS interview_rounds_user_schedule_idx
            ON interview_rounds(user_id, scheduled_start, status)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS interview_round_events_thread_idx
            ON interview_round_events(user_id, application_id, source_thread_id)
            """
        )
        self._create_retro_schema(connection)

    @staticmethod
    def _upgrade_v2(connection: sqlite3.Connection) -> None:
        SQLiteInterviewStore._create_retro_schema(connection)

    @staticmethod
    def _create_retro_schema(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS interview_retro_reports (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                application_id TEXT NOT NULL,
                interview_round_id TEXT NOT NULL REFERENCES interview_rounds(id),
                source_notes TEXT NOT NULL,
                summary TEXT NOT NULL,
                questions_json TEXT NOT NULL,
                strengths_json TEXT NOT NULL,
                difficulties_json TEXT NOT NULL,
                interviewer_signals_json TEXT NOT NULL,
                next_focus_json TEXT NOT NULL,
                action_items_json TEXT NOT NULL,
                limitations_json TEXT NOT NULL,
                self_assessment TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(user_id, interview_round_id, content_sha256)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS interview_retro_reports_round_idx
            ON interview_retro_reports(user_id, interview_round_id, created_at)
            """
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @classmethod
    def _round_values(cls, round_: InterviewRound) -> tuple[object, ...]:
        return (
            round_.id, round_.user_id, round_.application_id, round_.sequence_number,
            round_.employer_label, round_.status, cls._iso(round_.scheduled_start),
            cls._iso(round_.scheduled_end), round_.timezone, round_.interview_format,
            round_.location, round_.meeting_url, round_.contact_summary,
            round_.created_at.isoformat(), round_.updated_at.isoformat(),
            cls._iso(round_.completed_at),
        )

    @staticmethod
    def _new_event(
        *,
        round_: InterviewRound,
        source: str,
        event_type: str,
        email_event_id: str | None,
        source_thread_id: str | None,
        details: InterviewDetails,
        occurred_at: datetime,
    ) -> InterviewRoundEvent:
        return InterviewRoundEvent(
            id=f"interview_round_event_{uuid4().hex}",
            user_id=round_.user_id,
            application_id=round_.application_id,
            interview_round_id=round_.id,
            source=source,
            event_type=event_type,
            email_event_id=email_event_id,
            source_thread_id=source_thread_id,
            details=details,
            occurred_at=occurred_at,
        )

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection, event: InterviewRoundEvent
    ) -> None:
        connection.execute(
            """
            INSERT INTO interview_round_events(
                id, user_id, application_id, interview_round_id, source,
                event_type, email_event_id, source_thread_id, details_json,
                occurred_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.id, event.user_id, event.application_id,
                event.interview_round_id, event.source, event.event_type,
                event.email_event_id, event.source_thread_id,
                json.dumps(event.details.model_dump(mode="json"), ensure_ascii=False, sort_keys=True),
                event.occurred_at.isoformat(),
            ),
        )

    @staticmethod
    def _round(row: tuple[object, ...]) -> InterviewRound:
        return InterviewRound(
            id=row[0], user_id=row[1], application_id=row[2], sequence_number=row[3],
            employer_label=row[4], status=row[5], scheduled_start=row[6],
            scheduled_end=row[7], timezone=row[8], interview_format=row[9],
            location=row[10], meeting_url=row[11], contact_summary=row[12],
            created_at=row[13], updated_at=row[14], completed_at=row[15],
        )

    @staticmethod
    def _event(row: tuple[object, ...]) -> InterviewRoundEvent:
        return InterviewRoundEvent(
            id=row[0], user_id=row[1], application_id=row[2],
            interview_round_id=row[3], source=row[4], event_type=row[5],
            email_event_id=row[6], source_thread_id=row[7],
            details=json.loads(row[8]), occurred_at=row[9],
        )

    @staticmethod
    def _retro(row: tuple[object, ...]) -> InterviewRetroReport:
        return InterviewRetroReport(
            id=row[0],
            user_id=row[1],
            application_id=row[2],
            interview_round_id=row[3],
            source_notes=row[4],
            summary=row[5],
            questions=tuple(
                InterviewRetroQuestion.model_validate(item)
                for item in json.loads(row[6])
            ),
            strengths=tuple(json.loads(row[7])),
            difficulties=tuple(json.loads(row[8])),
            interviewer_signals=tuple(json.loads(row[9])),
            next_focus=tuple(json.loads(row[10])),
            action_items=tuple(json.loads(row[11])),
            limitations=tuple(json.loads(row[12])),
            self_assessment=row[13],
            content_sha256=row[14],
            created_at=row[15],
        )

    @staticmethod
    def _iso(value: datetime | None) -> str | None:
        return value.isoformat() if value is not None else None
