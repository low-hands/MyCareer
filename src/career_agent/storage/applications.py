from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
from uuid import uuid4

from career_agent.domain.applications import (
    Application,
    ApplicationEvent,
    ApplicationStatus,
)
from career_agent.storage.schema import apply_schema


class SQLiteApplicationStore:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            apply_schema(
                connection,
                "applications",
                2,
                self._migrate,
                upgrades={2: self._allow_unknown_resume},
            )
        os.chmod(self.path, 0o600)

    def create(
        self,
        *,
        user_id: str,
        job_posting_id: str,
        jd_snapshot_id: str,
        resume_version_id: str | None,
        submitted_at: datetime,
        note: str | None = None,
    ) -> Application:
        now = datetime.now(timezone.utc)
        application = Application(
            id=f"application_{uuid4().hex}",
            user_id=user_id,
            job_posting_id=job_posting_id,
            jd_snapshot_id=jd_snapshot_id,
            resume_version_id=resume_version_id,
            status="submitted",
            submitted_at=submitted_at,
            created_at=now,
            updated_at=now,
        )
        event = ApplicationEvent(
            id=f"application_event_{uuid4().hex}",
            application_id=application.id,
            user_id=user_id,
            source="user_reported",
            event_type="created",
            previous_status=None,
            new_status="submitted",
            note=note,
            occurred_at=now,
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO applications(
                    id, user_id, job_posting_id, jd_snapshot_id,
                    resume_version_id, status, submitted_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    application.id,
                    application.user_id,
                    application.job_posting_id,
                    application.jd_snapshot_id,
                    application.resume_version_id,
                    application.status,
                    application.submitted_at.isoformat(),
                    application.created_at.isoformat(),
                    application.updated_at.isoformat(),
                ),
            )
            self._insert_event(connection, event)
        return application

    def get(self, *, user_id: str, application_id: str) -> Application | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, user_id, job_posting_id, jd_snapshot_id,
                       resume_version_id, status, submitted_at, created_at, updated_at
                FROM applications WHERE id = ? AND user_id = ?
                """,
                (application_id, user_id),
            ).fetchone()
        return self._application(row) if row else None

    def find_for_job(
        self, *, user_id: str, job_posting_id: str
    ) -> Application | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, user_id, job_posting_id, jd_snapshot_id,
                       resume_version_id, status, submitted_at, created_at, updated_at
                FROM applications
                WHERE user_id = ? AND job_posting_id = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (user_id, job_posting_id),
            ).fetchone()
        return self._application(row) if row else None

    def list(
        self,
        *,
        user_id: str,
        statuses: tuple[ApplicationStatus, ...] = (),
        limit: int = 20,
    ) -> tuple[Application, ...]:
        query = (
            "SELECT id, user_id, job_posting_id, jd_snapshot_id, resume_version_id, "
            "status, submitted_at, created_at, updated_at "
            "FROM applications WHERE user_id = ?"
        )
        params: list[object] = [user_id]
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            query += f" AND status IN ({placeholders})"
            params.extend(statuses)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return tuple(self._application(row) for row in rows)

    def list_job_posting_ids(self, *, user_id: str) -> frozenset[str]:
        """All jobs that have crossed from consideration into application.

        This deliberately has no UI pagination limit. It feeds derived
        conditions such as the saved-job review reminder, where omitting an
        older application would recreate a task the user has already done.
        """

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT job_posting_id FROM applications WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        return frozenset(row[0] for row in rows)

    def list_resume_version_ids(self, *, user_id: str) -> frozenset[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT resume_version_id FROM applications WHERE user_id = ? AND resume_version_id IS NOT NULL",
                (user_id,),
            ).fetchall()
        return frozenset(row[0] for row in rows)

    def list_events(
        self, *, user_id: str, application_id: str
    ) -> tuple[ApplicationEvent, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, application_id, user_id, source, event_type,
                       previous_status, new_status, note, occurred_at
                FROM application_events
                WHERE application_id = ? AND user_id = ?
                ORDER BY occurred_at, rowid
                """,
                (application_id, user_id),
            ).fetchall()
        return tuple(self._event(row) for row in rows)

    def update(
        self,
        *,
        user_id: str,
        application_id: str,
        expected_status: ApplicationStatus,
        new_status: ApplicationStatus,
        submitted_at: datetime,
        note: str | None,
        source: str = "user_reported",
    ) -> Application | None:
        now = datetime.now(timezone.utc)
        event_type = "note_added" if expected_status == new_status else "status_changed"
        event = ApplicationEvent(
            id=f"application_event_{uuid4().hex}",
            application_id=application_id,
            user_id=user_id,
            source=source,
            event_type=event_type,
            previous_status=expected_status,
            new_status=new_status,
            note=note,
            occurred_at=now,
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                """
                UPDATE applications
                SET status = ?, submitted_at = ?, updated_at = ?
                WHERE id = ? AND user_id = ? AND status = ?
                """,
                (
                    new_status,
                    submitted_at.isoformat(),
                    now.isoformat(),
                    application_id,
                    user_id,
                    expected_status,
                ),
            ).rowcount
            if not updated:
                return None
            self._insert_event(connection, event)
        return self.get(user_id=user_id, application_id=application_id)

    def update_resume_version(self, *, user_id: str, application_id: str, resume_version_id: str | None, note: str | None = None) -> Application | None:
        now = datetime.now(timezone.utc)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT resume_version_id FROM applications WHERE id = ? AND user_id = ?", (application_id, user_id)).fetchone()
            if row is None:
                return None
            connection.execute("UPDATE applications SET resume_version_id = ?, updated_at = ? WHERE id = ? AND user_id = ?", (resume_version_id, now.isoformat(), application_id, user_id))
            current_status = connection.execute("SELECT status FROM applications WHERE id = ? AND user_id = ?", (application_id, user_id)).fetchone()[0]
            event = ApplicationEvent(id=f"application_event_{uuid4().hex}", application_id=application_id, user_id=user_id, source="user_reported", event_type="resume_version_changed", previous_status=current_status, new_status=current_status, note=note, occurred_at=now)
            self._insert_event(connection, event)
        return self.get(user_id=user_id, application_id=application_id)

    def clear_user(self, *, user_id: str) -> int:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            ids = [row[0] for row in connection.execute("SELECT id FROM applications WHERE user_id = ?", (user_id,)).fetchall()]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                connection.execute(f"DELETE FROM application_events WHERE application_id IN ({placeholders}) AND user_id = ?", (*ids, user_id))
            connection.execute("DELETE FROM applications WHERE user_id = ?", (user_id,))
        return len(ids)

    def _migrate(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS applications (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                job_posting_id TEXT NOT NULL,
                jd_snapshot_id TEXT NOT NULL,
                resume_version_id TEXT,
                status TEXT NOT NULL,
                submitted_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, job_posting_id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS application_events (
                id TEXT PRIMARY KEY,
                application_id TEXT NOT NULL REFERENCES applications(id),
                user_id TEXT NOT NULL,
                source TEXT NOT NULL,
                event_type TEXT NOT NULL,
                previous_status TEXT,
                new_status TEXT NOT NULL,
                note TEXT,
                occurred_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS applications_user_updated_idx
            ON applications(user_id, updated_at DESC)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS application_events_application_idx
            ON application_events(application_id, occurred_at)
            """
        )

    @staticmethod
    def _allow_unknown_resume(connection: sqlite3.Connection) -> None:
        """Applications reported from referrals/interviews need not name a CV."""
        connection.execute("DROP INDEX IF EXISTS applications_user_updated_idx")
        connection.execute("DROP INDEX IF EXISTS application_events_application_idx")
        connection.execute("ALTER TABLE application_events RENAME TO application_events_v1")
        connection.execute("ALTER TABLE applications RENAME TO applications_v1")
        connection.execute(
            """
            CREATE TABLE applications (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                job_posting_id TEXT NOT NULL,
                jd_snapshot_id TEXT NOT NULL,
                resume_version_id TEXT,
                status TEXT NOT NULL,
                submitted_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, job_posting_id)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO applications
            SELECT * FROM applications_v1
            """
        )
        connection.execute(
            """
            CREATE TABLE application_events (
                id TEXT PRIMARY KEY,
                application_id TEXT NOT NULL REFERENCES applications(id),
                user_id TEXT NOT NULL,
                source TEXT NOT NULL,
                event_type TEXT NOT NULL,
                previous_status TEXT,
                new_status TEXT NOT NULL,
                note TEXT,
                occurred_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO application_events
            SELECT * FROM application_events_v1
            """
        )
        connection.execute("DROP TABLE application_events_v1")
        connection.execute("DROP TABLE applications_v1")
        connection.execute(
            "CREATE INDEX applications_user_updated_idx "
            "ON applications(user_id, updated_at DESC)"
        )
        connection.execute(
            "CREATE INDEX application_events_application_idx "
            "ON application_events(application_id, occurred_at)"
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection, event: ApplicationEvent
    ) -> None:
        connection.execute(
            """
            INSERT INTO application_events(
                id, application_id, user_id, source, event_type,
                previous_status, new_status, note, occurred_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.id,
                event.application_id,
                event.user_id,
                event.source,
                event.event_type,
                event.previous_status,
                event.new_status,
                event.note,
                event.occurred_at.isoformat(),
            ),
        )

    @staticmethod
    def _application(row: tuple[object, ...]) -> Application:
        return Application(
            id=row[0],
            user_id=row[1],
            job_posting_id=row[2],
            jd_snapshot_id=row[3],
            resume_version_id=row[4],
            status=row[5],
            submitted_at=row[6],
            created_at=row[7],
            updated_at=row[8],
        )

    @staticmethod
    def _event(row: tuple[object, ...]) -> ApplicationEvent:
        return ApplicationEvent(
            id=row[0],
            application_id=row[1],
            user_id=row[2],
            source=row[3],
            event_type=row[4],
            previous_status=row[5],
            new_status=row[6],
            note=row[7],
            occurred_at=row[8],
        )
