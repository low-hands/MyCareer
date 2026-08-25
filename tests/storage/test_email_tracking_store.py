from datetime import datetime, timezone

from career_agent.domain.email_tracking import EmailAssessment, RemoteEmailMetadata
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
