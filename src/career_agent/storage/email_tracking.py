from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
from typing import Literal
from uuid import uuid4

from career_agent.domain.email_tracking import (
    EmailAccount,
    EmailAssessment,
    EmailEvent,
    EmailEventStatus,
    EmailMessage,
    EmailProvider,
    EmailSyncCursor,
    RemoteEmailMetadata,
)
from career_agent.storage.schema import apply_schema


class SQLiteEmailTrackingStore:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            apply_schema(connection, "email_tracking", 1, self._migrate)
        os.chmod(self.path, 0o600)

    def add_account(
        self,
        *,
        user_id: str,
        provider: EmailProvider,
        email_address: str,
        credential_ref: str,
    ) -> EmailAccount:
        now = datetime.now(timezone.utc)
        account = EmailAccount(
            id=f"email_account_{uuid4().hex}",
            user_id=user_id,
            provider=provider,
            email_address=email_address.casefold(),
            credential_ref=credential_ref,
            status="active",
            created_at=now,
            updated_at=now,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO email_accounts(
                    id, user_id, provider, email_address, credential_ref,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, provider, email_address) DO UPDATE SET
                    credential_ref = excluded.credential_ref,
                    status = 'active',
                    updated_at = excluded.updated_at
                """,
                (
                    account.id,
                    account.user_id,
                    account.provider,
                    account.email_address,
                    account.credential_ref,
                    account.status,
                    account.created_at.isoformat(),
                    account.updated_at.isoformat(),
                ),
            )
            row = connection.execute(
                """
                SELECT id, user_id, provider, email_address, credential_ref,
                       status, created_at, updated_at
                FROM email_accounts
                WHERE user_id = ? AND provider = ? AND email_address = ?
                """,
                (user_id, provider, email_address.casefold()),
            ).fetchone()
        return self._account(row)

    def get_account(self, *, user_id: str, account_id: str) -> EmailAccount | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, user_id, provider, email_address, credential_ref,
                       status, created_at, updated_at
                FROM email_accounts WHERE id = ? AND user_id = ?
                """,
                (account_id, user_id),
            ).fetchone()
        return self._account(row) if row else None

    def list_accounts(self, *, user_id: str) -> tuple[EmailAccount, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, user_id, provider, email_address, credential_ref,
                       status, created_at, updated_at
                FROM email_accounts WHERE user_id = ?
                ORDER BY created_at
                """,
                (user_id,),
            ).fetchall()
        return tuple(self._account(row) for row in rows)

    def disable_account(self, *, user_id: str, account_id: str) -> EmailAccount | None:
        with self._connect() as connection:
            changed = connection.execute(
                "UPDATE email_accounts SET status = 'disabled', updated_at = ? "
                "WHERE id = ? AND user_id = ?",
                (datetime.now(timezone.utc).isoformat(), account_id, user_id),
            ).rowcount
        return self.get_account(user_id=user_id, account_id=account_id) if changed else None

    def active_credential_ref_count(self, credential_ref: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM email_accounts "
                "WHERE credential_ref = ? AND status != 'disabled'",
                (credential_ref,),
            ).fetchone()
        return int(row[0])

    def get_cursor(self, *, account_id: str) -> EmailSyncCursor | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT account_id, cursor_type, value, uid_validity, updated_at
                FROM email_sync_cursors WHERE account_id = ?
                """,
                (account_id,),
            ).fetchone()
        return self._cursor(row) if row else None

    def save_cursor(self, cursor: EmailSyncCursor) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO email_sync_cursors(
                    account_id, cursor_type, value, uid_validity, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(account_id) DO UPDATE SET
                    cursor_type = excluded.cursor_type,
                    value = excluded.value,
                    uid_validity = excluded.uid_validity,
                    updated_at = excluded.updated_at
                """,
                (
                    cursor.account_id,
                    cursor.cursor_type,
                    cursor.value,
                    cursor.uid_validity,
                    cursor.updated_at.isoformat(),
                ),
            )

    def save_message(
        self,
        *,
        user_id: str,
        account: EmailAccount,
        metadata: RemoteEmailMetadata,
        candidate: bool,
        content_sha256: str | None = None,
        encrypted_content_ref: str | None = None,
    ) -> tuple[EmailMessage, bool]:
        now = datetime.now(timezone.utc)
        message = EmailMessage(
            id=f"email_message_{uuid4().hex}",
            user_id=user_id,
            account_id=account.id,
            provider=account.provider,
            external_message_id=metadata.external_message_id,
            external_thread_id=metadata.external_thread_id,
            sender=metadata.sender,
            subject=metadata.subject,
            received_at=metadata.received_at,
            content_sha256=content_sha256,
            encrypted_content_ref=encrypted_content_ref,
            application_id=None,
            classification=None,
            candidate=candidate,
            created_at=now,
        )
        with self._connect() as connection:
            inserted = connection.execute(
                """
                INSERT INTO email_messages(
                    id, user_id, account_id, provider, external_message_id,
                    external_thread_id, sender, subject, received_at,
                    content_sha256, encrypted_content_ref, application_id,
                    classification, candidate, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(account_id, external_message_id) DO NOTHING
                """,
                (
                    message.id,
                    message.user_id,
                    message.account_id,
                    message.provider,
                    message.external_message_id,
                    message.external_thread_id,
                    message.sender,
                    message.subject,
                    message.received_at.isoformat(),
                    message.content_sha256,
                    message.encrypted_content_ref,
                    message.application_id,
                    message.classification,
                    int(message.candidate),
                    message.created_at.isoformat(),
                ),
            ).rowcount
            row = connection.execute(
                """
                SELECT id, user_id, account_id, provider, external_message_id,
                       external_thread_id, sender, subject, received_at,
                       content_sha256, encrypted_content_ref, application_id,
                       classification, candidate, created_at
                FROM email_messages
                WHERE account_id = ? AND external_message_id = ?
                """,
                (account.id, metadata.external_message_id),
            ).fetchone()
        return self._message(row), bool(inserted)

    def create_event(
        self,
        *,
        user_id: str,
        email_message_id: str,
        assessment: EmailAssessment,
        status: EmailEventStatus,
        occurred_at: datetime,
        classifier: str,
    ) -> tuple[EmailEvent, bool]:
        now = datetime.now(timezone.utc)
        event = EmailEvent(
            id=f"email_event_{uuid4().hex}",
            user_id=user_id,
            email_message_id=email_message_id,
            application_id=assessment.application_id,
            event_type=assessment.event_type,
            status=status,
            confidence=assessment.confidence,
            classifier=classifier,
            summary=assessment.summary,
            interview_details=assessment.interview_details,
            occurred_at=occurred_at,
            created_at=now,
            resolved_at=now if status != "pending_confirmation" else None,
        )
        with self._connect() as connection:
            inserted = connection.execute(
                """
                INSERT INTO email_events(
                    id, user_id, email_message_id, application_id, event_type,
                    status, confidence, classifier, summary, interview_details_json,
                    occurred_at, created_at, resolved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(email_message_id) DO NOTHING
                """,
                (
                    event.id,
                    event.user_id,
                    event.email_message_id,
                    event.application_id,
                    event.event_type,
                    event.status,
                    event.confidence,
                    event.classifier,
                    event.summary,
                    (
                        json.dumps(
                            event.interview_details.model_dump(mode="json"),
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        if event.interview_details is not None
                        else None
                    ),
                    event.occurred_at.isoformat(),
                    event.created_at.isoformat(),
                    event.resolved_at.isoformat() if event.resolved_at else None,
                ),
            ).rowcount
            row = connection.execute(
                self._EVENT_SELECT + " WHERE email_message_id = ?",
                (email_message_id,),
            ).fetchone()
        return self._event(row), bool(inserted)

    def set_message_assessment(
        self,
        *,
        user_id: str,
        message_id: str,
        content_sha256: str,
        application_id: str | None,
        classification: str,
    ) -> EmailMessage:
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE email_messages
                SET content_sha256 = ?, application_id = ?, classification = ?
                WHERE id = ? AND user_id = ?
                """,
                (content_sha256, application_id, classification, message_id, user_id),
            ).rowcount
            row = connection.execute(
                """
                SELECT id, user_id, account_id, provider, external_message_id,
                       external_thread_id, sender, subject, received_at,
                       content_sha256, encrypted_content_ref, application_id,
                       classification, candidate, created_at
                FROM email_messages WHERE id = ? AND user_id = ?
                """,
                (message_id, user_id),
            ).fetchone()
        if not updated or row is None:
            raise ValueError("email message not found")
        return self._message(row)

    def get_event(self, *, user_id: str, event_id: str) -> EmailEvent | None:
        with self._connect() as connection:
            row = connection.execute(
                self._EVENT_SELECT + " WHERE id = ? AND user_id = ?",
                (event_id, user_id),
            ).fetchone()
        return self._event(row) if row else None

    def get_event_for_message(
        self, *, user_id: str, email_message_id: str
    ) -> EmailEvent | None:
        with self._connect() as connection:
            row = connection.execute(
                self._EVENT_SELECT + " WHERE email_message_id = ? AND user_id = ?",
                (email_message_id, user_id),
            ).fetchone()
        return self._event(row) if row else None

    def get_message(
        self, *, user_id: str, email_message_id: str
    ) -> EmailMessage | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, user_id, account_id, provider, external_message_id,
                       external_thread_id, sender, subject, received_at,
                       content_sha256, encrypted_content_ref, application_id,
                       classification, candidate, created_at
                FROM email_messages WHERE id = ? AND user_id = ?
                """,
                (email_message_id, user_id),
            ).fetchone()
        return self._message(row) if row else None

    def list_events(
        self,
        *,
        user_id: str,
        status: EmailEventStatus | None = None,
        limit: int = 20,
    ) -> tuple[EmailEvent, ...]:
        query = self._EVENT_SELECT + " WHERE user_id = ?"
        params: list[object] = [user_id]
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY occurred_at DESC, rowid DESC LIMIT ?"
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return tuple(self._event(row) for row in rows)

    def resolve_event(
        self,
        *,
        user_id: str,
        event_id: str,
        status: Literal["applied", "dismissed"],
        application_id: str | None,
    ) -> EmailEvent | None:
        now = datetime.now(timezone.utc)
        with self._connect() as connection:
            updated = connection.execute(
                """
                UPDATE email_events
                SET status = ?, application_id = ?, resolved_at = ?
                WHERE id = ? AND user_id = ? AND status = 'pending_confirmation'
                """,
                (status, application_id, now.isoformat(), event_id, user_id),
            ).rowcount
        if not updated:
            return None
        return self.get_event(user_id=user_id, event_id=event_id)

    _EVENT_SELECT = (
        "SELECT id, user_id, email_message_id, application_id, event_type, "
        "status, confidence, classifier, summary, interview_details_json, "
        "occurred_at, created_at, resolved_at "
        "FROM email_events"
    )

    def _migrate(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS email_accounts (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                provider TEXT NOT NULL CHECK(provider IN ('gmail', 'qq')),
                email_address TEXT NOT NULL,
                credential_ref TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, provider, email_address)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS email_sync_cursors (
                account_id TEXT PRIMARY KEY REFERENCES email_accounts(id),
                cursor_type TEXT NOT NULL,
                value TEXT NOT NULL,
                uid_validity TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS email_messages (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                account_id TEXT NOT NULL REFERENCES email_accounts(id),
                provider TEXT NOT NULL,
                external_message_id TEXT NOT NULL,
                external_thread_id TEXT,
                sender TEXT NOT NULL,
                subject TEXT NOT NULL,
                received_at TEXT NOT NULL,
                content_sha256 TEXT,
                encrypted_content_ref TEXT,
                application_id TEXT,
                classification TEXT,
                candidate INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(account_id, external_message_id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS email_events (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                email_message_id TEXT NOT NULL UNIQUE REFERENCES email_messages(id),
                application_id TEXT,
                event_type TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                classifier TEXT NOT NULL,
                summary TEXT NOT NULL,
                interview_details_json TEXT,
                occurred_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                resolved_at TEXT
            )
            """
        )
        event_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(email_events)")
        }
        if "interview_details_json" not in event_columns:
            connection.execute(
                "ALTER TABLE email_events ADD COLUMN interview_details_json TEXT"
            )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS email_events_user_status_idx
            ON email_events(user_id, status, occurred_at DESC)
            """
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @staticmethod
    def _account(row: tuple[object, ...]) -> EmailAccount:
        return EmailAccount(
            id=row[0], user_id=row[1], provider=row[2], email_address=row[3],
            credential_ref=row[4], status=row[5], created_at=row[6], updated_at=row[7]
        )

    @staticmethod
    def _cursor(row: tuple[object, ...]) -> EmailSyncCursor:
        return EmailSyncCursor(
            account_id=row[0], cursor_type=row[1], value=row[2],
            uid_validity=row[3], updated_at=row[4]
        )

    @staticmethod
    def _message(row: tuple[object, ...]) -> EmailMessage:
        return EmailMessage(
            id=row[0], user_id=row[1], account_id=row[2], provider=row[3],
            external_message_id=row[4], external_thread_id=row[5], sender=row[6],
            subject=row[7], received_at=row[8], content_sha256=row[9],
            encrypted_content_ref=row[10], application_id=row[11],
            classification=row[12], candidate=bool(row[13]), created_at=row[14]
        )

    @staticmethod
    def _event(row: tuple[object, ...]) -> EmailEvent:
        return EmailEvent(
            id=row[0], user_id=row[1], email_message_id=row[2], application_id=row[3],
            event_type=row[4], status=row[5], confidence=row[6], classifier=row[7],
            summary=row[8],
            interview_details=(json.loads(row[9]) if row[9] else None),
            occurred_at=row[10], created_at=row[11], resolved_at=row[12]
        )
