from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from career_agent.connectors.calendar import CalendarWriteResult
from career_agent.domain.interviews import InterviewRound
from career_agent.services.calendar import (
    CalendarProposalConflictError,
    CalendarService,
    CalendarSyncNotAvailableError,
)
from career_agent.services.interviews import InterviewDetail
from career_agent.storage.calendar import SQLiteCalendarStore


NOW = datetime(2026, 8, 26, tzinfo=timezone.utc)


class Interviews:
    def __init__(self):
        self.interview = InterviewRound(
            id="interview-1", user_id="u1", application_id="application-1",
            sequence_number=1, employer_label=None, status="scheduled",
            scheduled_start=NOW + timedelta(days=2),
            scheduled_end=NOW + timedelta(days=2, hours=1),
            timezone="Asia/Shanghai", interview_format="video",
            meeting_url="https://meet.example/interview",
            created_at=NOW, updated_at=NOW,
        )

    def get_interview(self, **kwargs):
        if kwargs["user_id"] != "u1" or kwargs["interview_round_id"] != self.interview.id:
            raise AssertionError("unexpected interview lookup")
        return InterviewDetail(interview=self.interview, events=())


class Applications:
    def get_application(self, **kwargs):
        assert kwargs == {"user_id": "u1", "application_id": "application-1"}
        posting = SimpleNamespace(company_name="Acme", title="AI Engineer")
        return SimpleNamespace(job=SimpleNamespace(posting=posting))


class Connector:
    def __init__(self):
        self.calls = []

    def apply(self, **kwargs):
        self.calls.append(kwargs)
        return CalendarWriteResult(
            external_event_id=kwargs["external_event_id"],
            etag=f'etag-{len(self.calls)}',
            html_link="https://calendar.example/event",
        )


class Resolver:
    def __init__(self, connector):
        self.connector = connector

    def resolve(self, account):
        assert account.user_id == "u1"
        return self.connector


def build_service(tmp_path):
    store = SQLiteCalendarStore(tmp_path / "calendar.sqlite3")
    account = store.add_account(
        user_id="u1", email_address="user@example.com", calendar_id="primary",
        credential_ref="env:GOOGLE_CALENDAR_CREDENTIAL", now=NOW,
    )
    interviews = Interviews()
    connector = Connector()
    service = CalendarService(
        store, interviews, Applications(), Resolver(connector)
    )
    return service, store, account, interviews, connector


def test_calendar_requires_proposal_then_reuses_event_for_updates_and_cancel(tmp_path) -> None:
    service, store, account, interviews, connector = build_service(tmp_path)

    create = service.prepare_interview_sync(
        user_id="u1", interview_round_id="interview-1", now=NOW,
    )
    assert create.operation == "create"
    assert connector.calls == []

    created = service.execute_proposal(
        user_id="u1", proposal_id=create.id, now=NOW + timedelta(minutes=1),
    )
    external_event_id = created.link.external_event_id
    assert created.link.status == "active"
    assert connector.calls[0]["operation"] == "create"

    with pytest.raises(CalendarSyncNotAvailableError, match="already synchronized"):
        service.prepare_interview_sync(
            user_id="u1", interview_round_id="interview-1",
            calendar_account_id=account.id, now=NOW + timedelta(minutes=2),
        )

    interviews.interview = interviews.interview.model_copy(
        update={
            "scheduled_start": NOW + timedelta(days=3),
            "scheduled_end": NOW + timedelta(days=3, hours=1),
            "updated_at": NOW + timedelta(hours=1),
        }
    )
    update = service.prepare_interview_sync(
        user_id="u1", interview_round_id="interview-1", now=NOW + timedelta(hours=1),
    )
    assert update.operation == "update"
    assert update.external_event_id == external_event_id
    service.execute_proposal(
        user_id="u1", proposal_id=update.id, now=NOW + timedelta(hours=1, minutes=1),
    )
    assert connector.calls[1]["operation"] == "update"

    interviews.interview = interviews.interview.model_copy(
        update={"status": "cancelled", "updated_at": NOW + timedelta(hours=2)}
    )
    cancel = service.prepare_interview_sync(
        user_id="u1", interview_round_id="interview-1", now=NOW + timedelta(hours=2),
    )
    assert cancel.operation == "cancel"
    assert cancel.external_event_id == external_event_id
    cancelled = service.execute_proposal(
        user_id="u1", proposal_id=cancel.id,
        now=NOW + timedelta(hours=2, minutes=1),
    )
    assert cancelled.link.status == "cancelled"
    assert connector.calls[2]["operation"] == "cancel"
    assert [event.event_type for event in store.list_events(
        user_id="u1", proposal_id=cancel.id
    )] == ["proposed", "executed"]


def test_changed_interview_invalidates_fixed_payload_approval(tmp_path) -> None:
    service, store, _, interviews, connector = build_service(tmp_path)
    proposal = service.prepare_interview_sync(
        user_id="u1", interview_round_id="interview-1", now=NOW,
    )
    interviews.interview = interviews.interview.model_copy(
        update={
            "scheduled_start": NOW + timedelta(days=4),
            "scheduled_end": NOW + timedelta(days=4, hours=1),
            "updated_at": NOW + timedelta(minutes=2),
        }
    )

    with pytest.raises(CalendarProposalConflictError, match="changed after approval"):
        service.execute_proposal(
            user_id="u1", proposal_id=proposal.id, now=NOW + timedelta(minutes=2),
        )

    assert connector.calls == []
    assert store.get_proposal(user_id="u1", proposal_id=proposal.id).status == "superseded"


def test_expired_proposal_never_calls_external_calendar(tmp_path) -> None:
    service, store, _, _, connector = build_service(tmp_path)
    proposal = service.prepare_interview_sync(
        user_id="u1", interview_round_id="interview-1", now=NOW,
    )

    with pytest.raises(CalendarProposalConflictError, match="expired"):
        service.execute_proposal(
            user_id="u1", proposal_id=proposal.id, now=NOW + timedelta(minutes=16),
        )

    assert connector.calls == []
    assert store.get_proposal(user_id="u1", proposal_id=proposal.id).status == "expired"
