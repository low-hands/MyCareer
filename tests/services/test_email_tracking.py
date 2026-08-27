import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from career_agent.domain.email_tracking import (
    EmailAssessment,
    EmailSyncBatch,
    RemoteEmailContent,
    RemoteEmailMetadata,
)
from career_agent.domain.interviews import InterviewDetails
from career_agent.services.email_tracking import EmailTrackingService
from career_agent.services.interviews import InterviewService
from career_agent.storage.email_tracking import SQLiteEmailTrackingStore
from career_agent.storage.interviews import SQLiteInterviewStore


class Connector:
    provider = "gmail"

    def __init__(self, metadata):
        self.metadata = metadata
        self.content_calls = []

    def sync_metadata(self, *, cursor, since):
        return EmailSyncBatch(messages=(self.metadata,), next_cursor_value="h2")

    def get_content(self, *, external_message_id):
        self.content_calls.append(external_message_id)
        return RemoteEmailContent(
            external_message_id=external_message_id,
            text="Acme 邀请您参加 AI Engineer 面试，请选择面试时间。",
        )


class Resolver:
    def __init__(self, connector):
        self.connector = connector

    def resolve(self, **kwargs):
        return self.connector


class Applications:
    def __init__(self):
        self.applied = []
        self.application = SimpleNamespace(
            id="app-1",
            status="submitted",
            submitted_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        )
        self.job = SimpleNamespace(
            posting=SimpleNamespace(company_name="Acme", title="AI Engineer")
        )

    def list_applications(self, **kwargs):
        return (SimpleNamespace(application=self.application, job=self.job),)

    def apply_email_event(self, **kwargs):
        self.applied.append(kwargs)
        self.application.status = "interviewing"
        return self.application

    def get_application(self, *, user_id, application_id):
        if user_id != "u1" or application_id != self.application.id:
            raise ValueError("application unavailable")
        return SimpleNamespace(application=self.application)


def test_heuristic_worker_never_auto_applies_a_keyword_match(tmp_path: Path) -> None:
    store = SQLiteEmailTrackingStore(tmp_path / "email.sqlite3")
    account = store.add_account(
        user_id="u1",
        provider="gmail",
        email_address="user@gmail.com",
        credential_ref="env:GMAIL_SECRET",
    )
    metadata = RemoteEmailMetadata(
        external_message_id="m1",
        external_thread_id="t1",
        sender="Acme Recruiting <jobs@acme.example>",
        subject="Acme AI Engineer 面试邀请",
        received_at=datetime(2026, 8, 20, tzinfo=timezone.utc),
    )
    connector = Connector(metadata)
    applications = Applications()
    interviews = InterviewService(
        SQLiteInterviewStore(tmp_path / "applications.sqlite3"), applications
    )
    service = EmailTrackingService(
        store, applications, Resolver(connector), interview_service=interviews
    )

    result = service.sync(user_id="u1", account_id=account.id)

    assert result.messages_seen == 1
    assert result.candidate_messages == 1
    assert connector.content_calls == ["m1"]
    assert result.events_created[0].event_type == "interview_invitation"
    # A company-name substring hit reaches 0.95 under the heuristic worker, but
    # that score cannot authorize a durable write on its own.
    assert result.events_created[0].status == "pending_confirmation"
    assert applications.applied == []
    assert interviews.list_interviews(user_id="u1") == ()
    assert store.get_cursor(account_id=account.id).value == "h2"
    assert b"Acme \xe9\x82\x80\xe8\xaf\xb7" not in (tmp_path / "email.sqlite3").read_bytes()


def test_non_candidate_does_not_fetch_body(tmp_path: Path) -> None:
    store = SQLiteEmailTrackingStore(tmp_path / "email.sqlite3")
    account = store.add_account(
        user_id="u1", provider="gmail", email_address="user@gmail.com",
        credential_ref="env:GMAIL_SECRET",
    )
    metadata = RemoteEmailMetadata(
        external_message_id="m2",
        sender="Newsletter <news@example.com>",
        subject="Weekly product digest",
        received_at=datetime(2026, 8, 20, tzinfo=timezone.utc),
    )
    connector = Connector(metadata)
    service = EmailTrackingService(store, Applications(), Resolver(connector))

    result = service.sync(user_id="u1", account_id=account.id)

    assert result.candidate_messages == 0
    assert connector.content_calls == []
    assert result.events_created == ()


class PendingWorker:
    classifier = "test_pending_v1"
    authorizes_auto_apply = True

    def assess(self, **kwargs):
        return EmailAssessment(
            event_type="interview_invitation",
            application_id="app-1",
            confidence=0.7,
            summary="识别到面试邀请，但置信度不足。",
        )


class CalibratedWorker:
    classifier = "test_calibrated_v1"
    authorizes_auto_apply = True

    def assess(self, **kwargs):
        return EmailAssessment(
            event_type="interview_invitation",
            application_id="app-1",
            confidence=0.97,
            summary="已唯一匹配投递记录。",
            interview_details=InterviewDetails(change_type="invited"),
        )


def test_calibrated_worker_may_auto_apply_above_the_threshold(tmp_path: Path) -> None:
    store = SQLiteEmailTrackingStore(tmp_path / "email.sqlite3")
    account = store.add_account(
        user_id="u1", provider="gmail", email_address="user@gmail.com",
        credential_ref="env:GMAIL_SECRET",
    )
    metadata = RemoteEmailMetadata(
        external_message_id="m4", sender="Acme Recruiting",
        subject="面试邀请", received_at=datetime(2026, 8, 20, tzinfo=timezone.utc),
    )
    applications = Applications()
    interviews = InterviewService(
        SQLiteInterviewStore(tmp_path / "applications.sqlite3"), applications
    )
    service = EmailTrackingService(
        store, applications, Resolver(Connector(metadata)), CalibratedWorker(),
        interview_service=interviews,
    )

    event = service.sync(user_id="u1", account_id=account.id).events_created[0]

    assert event.status == "applied"
    assert applications.applied[0]["application_id"] == "app-1"
    interview = interviews.list_interviews(user_id="u1")[0]
    assert interview.sequence_number == 1
    assert interview.employer_label is None
    assert interview.status == "identified"


def test_pending_event_requires_explicit_resolution(tmp_path: Path) -> None:
    store = SQLiteEmailTrackingStore(tmp_path / "email.sqlite3")
    account = store.add_account(
        user_id="u1", provider="gmail", email_address="user@gmail.com",
        credential_ref="env:GMAIL_SECRET",
    )
    metadata = RemoteEmailMetadata(
        external_message_id="m3", sender="Acme Recruiting",
        subject="面试邀请", received_at=datetime(2026, 8, 20, tzinfo=timezone.utc),
    )
    applications = Applications()
    service = EmailTrackingService(
        store, applications, Resolver(Connector(metadata)), PendingWorker()
    )

    event = service.sync(user_id="u1", account_id=account.id).events_created[0]
    assert event.status == "pending_confirmation"
    assert applications.applied == []

    resolved = service.resolve_event(
        user_id="u1", event_id=event.id, approve=True
    )
    assert resolved.status == "applied"
    assert len(applications.applied) == 1


class CredentialLeakingConnector(Connector):
    """A mailbox whose body carries credentials alongside the real signal."""

    BODY = (
        "Acme 邀请您参加 AI Engineer 面试。\n"
        "会议链接 https://meet.google.com/abc-defg-hij\n"
        "登录 https://portal.acme.example/sso?token=eyJhbGciOi.SECRET\n"
        "您的验证码是 483920"
    )

    def get_content(self, *, external_message_id):
        self.content_calls.append(external_message_id)
        return RemoteEmailContent(
            external_message_id=external_message_id, text=self.BODY
        )


class BodyCapturingWorker:
    classifier = "test_capture_v1"
    authorizes_auto_apply = False

    def __init__(self) -> None:
        self.seen_bodies: list[str] = []
        self.seen_subjects: list[str] = []

    def assess(self, *, metadata, content, applications):
        self.seen_bodies.append(content.text)
        self.seen_subjects.append(metadata.subject)
        return EmailAssessment(
            event_type="interview_invitation", application_id="app-1",
            confidence=0.9, summary="面试邀请。",
        )


def test_email_body_is_scrubbed_before_it_reaches_the_worker(tmp_path: Path) -> None:
    store = SQLiteEmailTrackingStore(tmp_path / "email.sqlite3")
    account = store.add_account(
        user_id="u1", provider="gmail", email_address="user@gmail.com",
        credential_ref="env:GMAIL_SECRET",
    )
    metadata = RemoteEmailMetadata(
        external_message_id="m5", sender="Acme Recruiting",
        subject="面试邀请", received_at=datetime(2026, 8, 20, tzinfo=timezone.utc),
    )
    worker = BodyCapturingWorker()
    service = EmailTrackingService(
        store, Applications(), Resolver(CredentialLeakingConnector(metadata)), worker
    )

    service.sync(user_id="u1", account_id=account.id)

    body = worker.seen_bodies[0]
    # The model runs on a third-party API, so the credentials must be gone
    # before the call, not merely absent from what we persist.
    assert "eyJhbGciOi.SECRET" not in body
    assert "483920" not in body
    # The meeting link is the point of an invitation and has to survive.
    assert "https://meet.google.com/abc-defg-hij" in body
    assert "sso?token=<redacted>" in body


def test_a_code_in_the_subject_is_scrubbed_on_both_the_model_and_store_paths(
    tmp_path: Path,
) -> None:
    """The envelope reaches the model and the database just as the body does."""
    store = SQLiteEmailTrackingStore(tmp_path / "email.sqlite3")
    account = store.add_account(
        user_id="u1", provider="gmail", email_address="user@gmail.com",
        credential_ref="env:GMAIL_SECRET",
    )
    metadata = RemoteEmailMetadata(
        external_message_id="m6", sender="Acme Recruiting",
        subject="Acme 面试邀请 验证码 483920",
        received_at=datetime(2026, 8, 20, tzinfo=timezone.utc),
    )
    worker = BodyCapturingWorker()
    service = EmailTrackingService(
        store, Applications(), Resolver(CredentialLeakingConnector(metadata)), worker
    )

    service.sync(user_id="u1", account_id=account.id)

    assert "483920" not in worker.seen_subjects[0]
    # The company and the intent still have to be readable for classification.
    assert "Acme" in worker.seen_subjects[0]
    with sqlite3.connect(tmp_path / "email.sqlite3") as connection:
        subjects = [
            row[0]
            for row in connection.execute("SELECT subject FROM email_messages")
        ]
    assert subjects and all("483920" not in subject for subject in subjects)
