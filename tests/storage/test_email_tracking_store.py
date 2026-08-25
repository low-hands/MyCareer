from datetime import datetime, timezone
import sqlite3

from career_agent.domain.email_tracking import EmailAssessment, RemoteEmailMetadata
from career_agent.domain.interviews import InterviewDetails
from career_agent.storage.email_tracking import SQLiteEmailTrackingStore


def test_account_message_and_event_are_user_scoped_and_idempotent(tmp_path) -> None:
    store = SQLiteEmailTrackingStore(tmp_path / "email.sqlite3")
    account = store.add_account(
        user_id="u1", provider="qq", email_address="123@qq.com",
        credential_ref="env:QQ_AUTH_CODE",
    )
    metadata = RemoteEmailMetadata(
        external_message_id="42", sender="Acme", subject="录用通知",
        received_at=datetime(2026, 8, 20, tzinfo=timezone.utc),
    )
    message, inserted = store.save_message(
        user_id="u1", account=account, metadata=metadata, candidate=True
    )
    same, inserted_again = store.save_message(
        user_id="u1", account=account, metadata=metadata, candidate=True
    )
    assessment = EmailAssessment(
        event_type="offer", application_id="app-1", confidence=0.97,
        summary="识别到录用通知。",
    )
    event, event_inserted = store.create_event(
        user_id="u1", email_message_id=message.id, assessment=assessment,
        status="pending_confirmation", occurred_at=metadata.received_at,
        classifier="test_v1",
    )
    same_event, event_inserted_again = store.create_event(
        user_id="u1", email_message_id=message.id, assessment=assessment,
        status="pending_confirmation", occurred_at=metadata.received_at,
        classifier="test_v1",
    )

    assert inserted and not inserted_again and same.id == message.id
    assert event_inserted and not event_inserted_again and same_event.id == event.id
    assert store.get_event(user_id="u2", event_id=event.id) is None


def test_interview_details_round_trip_and_previous_schema_migrates(tmp_path) -> None:
    path = tmp_path / "email.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE email_events (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                email_message_id TEXT NOT NULL UNIQUE,
                application_id TEXT,
                event_type TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                classifier TEXT NOT NULL,
                summary TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                resolved_at TEXT
            )
            """
        )
    store = SQLiteEmailTrackingStore(path)
    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(email_events)")}
    assert "interview_details_json" in columns

    account = store.add_account(
        user_id="u1", provider="gmail", email_address="user@gmail.com",
        credential_ref="env:GMAIL_SECRET",
    )
    metadata = RemoteEmailMetadata(
        external_message_id="m1", sender="Acme", subject="面试邀请",
        received_at=datetime(2026, 8, 26, tzinfo=timezone.utc),
    )
    message, _ = store.save_message(
        user_id="u1", account=account, metadata=metadata, candidate=True
    )
    assessment = EmailAssessment(
        event_type="interview_invitation",
        application_id="app-1",
        confidence=0.95,
        summary="识别到面试邀请。",
        interview_details=InterviewDetails(
            employer_label="二面",
            scheduled_start=datetime(2026, 8, 27, tzinfo=timezone.utc),
            timezone="Asia/Shanghai",
        ),
    )
    event, _ = store.create_event(
        user_id="u1", email_message_id=message.id, assessment=assessment,
        status="pending_confirmation", occurred_at=metadata.received_at,
        classifier="test_v1",
    )

    loaded = store.get_event(user_id="u1", event_id=event.id)
    assert loaded.interview_details.employer_label == "二面"
