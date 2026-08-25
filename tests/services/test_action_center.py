from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from career_agent.domain.email_tracking import EmailEvent
from career_agent.domain.interviews import InterviewRound
from career_agent.services.action_center import (
    ActionCenterService,
    InvalidActionTransitionError,
)
from career_agent.storage.action_center import SQLiteActionItemStore


NOW = datetime(2026, 8, 26, 0, 0, tzinfo=timezone.utc)


class Applications:
    def __init__(self):
        self.items = (
            self._item("app-upcoming", NOW - timedelta(days=10), "Acme"),
            self._item("app-stale", NOW - timedelta(days=8), "Beta"),
        )

    @staticmethod
    def _item(application_id, updated_at, company):
        application = SimpleNamespace(
            id=application_id,
            status="submitted",
            updated_at=updated_at,
        )
        job = SimpleNamespace(
            posting=SimpleNamespace(company_name=company, title="AI Engineer")
        )
        return SimpleNamespace(application=application, job=job)

    def list_applications(self, **kwargs):
        return self.items


class Emails:
    def __init__(self):
        self.pending = True

    def list_events(self, **kwargs):
        pending = EmailEvent(
            id="email-pending", user_id="u1", email_message_id="message-1",
            application_id="app-stale", event_type="unclear",
            status="pending_confirmation", confidence=0.5, classifier="test",
            summary="需要确认该邮件对应的投递。", occurred_at=NOW - timedelta(hours=2),
            created_at=NOW - timedelta(hours=2),
        )
        material = EmailEvent(
            id="email-material", user_id="u1", email_message_id="message-2",
            application_id="app-stale", event_type="material_request",
            status="applied", confidence=0.95, classifier="test",
            summary="招聘方请求补充材料。", occurred_at=NOW - timedelta(days=1),
            created_at=NOW - timedelta(days=1), resolved_at=NOW - timedelta(days=1),
        )
        return ((pending,) if self.pending else ()) + (material,)


class Interviews:
    def list_interviews(self, **kwargs):
        upcoming_start = NOW + timedelta(days=1)
        completed_at = NOW - timedelta(hours=3)
        return (
            InterviewRound(
                id="interview-upcoming", user_id="u1",
                application_id="app-upcoming", sequence_number=1,
                status="scheduled", scheduled_start=upcoming_start,
                scheduled_end=upcoming_start + timedelta(hours=1),
                timezone="Asia/Shanghai", interview_format="video",
                created_at=NOW, updated_at=NOW,
            ),
            InterviewRound(
                id="interview-completed", user_id="u1",
                application_id="app-stale", sequence_number=1,
                status="completed", interview_format="video",
                created_at=NOW - timedelta(days=2), updated_at=completed_at,
                completed_at=completed_at,
            ),
        )


def build_service(tmp_path):
    emails = Emails()
    store = SQLiteActionItemStore(tmp_path / "actions.sqlite3")
    service = ActionCenterService(
        store, Applications(), emails, Interviews()
    )
    return service, store, emails


def test_refresh_generates_idempotent_source_grounded_actions(tmp_path) -> None:
    service, _, _ = build_service(tmp_path)

    first = service.refresh(user_id="u1", now=NOW)
    repeated = service.refresh(user_id="u1", now=NOW + timedelta(minutes=1))

    assert {item.action_type for item in first} == {
        "application_follow_up",
        "email_event_confirmation",
        "interview_preparation",
        "material_submission",
        "interview_reminder",
        "interview_retro",
    }
    assert {item.id for item in repeated} == {item.id for item in first}
    follow_ups = [item for item in first if item.action_type == "application_follow_up"]
    assert len(follow_ups) == 1
    assert follow_ups[0].application_id == "app-stale"


def test_daily_brief_groups_due_items_without_persisting_report_text(tmp_path) -> None:
    service, _, _ = build_service(tmp_path)

    brief = service.daily_brief(user_id="u1", now=NOW)

    assert brief.timezone == "Asia/Shanghai"
    assert {item.action_type for item in brief.overdue} == {
        "interview_preparation"
    }
    assert {item.action_type for item in brief.due_today} == {
        "application_follow_up",
        "email_event_confirmation",
        "interview_retro",
    }
    assert {item.action_type for item in brief.upcoming} == {"interview_reminder"}
    assert {item.action_type for item in brief.no_due_date} == {"material_submission"}


def test_complete_and_snooze_survive_refresh_and_expired_snooze_reopens(tmp_path) -> None:
    service, store, _ = build_service(tmp_path)
    items = service.refresh(user_id="u1", now=NOW)
    confirmation = next(
        item for item in items if item.action_type == "email_event_confirmation"
    )
    reminder = next(item for item in items if item.action_type == "interview_reminder")

    service.complete_action(user_id="u1", action_item_id=confirmation.id)
    service.snooze_action(
        user_id="u1", action_item_id=reminder.id,
        snoozed_until=NOW + timedelta(hours=3), now=NOW,
    )
    active = service.refresh(user_id="u1", now=NOW + timedelta(hours=1))
    assert confirmation.id not in {item.id for item in active}
    assert reminder.id not in {item.id for item in active}

    reopened = service.refresh(user_id="u1", now=NOW + timedelta(hours=4))
    assert reminder.id in {item.id for item in reopened}
    assert [event.event_type for event in store.list_events(
        user_id="u1", action_item_id=reminder.id
    )] == ["created", "snoozed", "reopened"]


def test_source_resolution_auto_completes_obsolete_generated_action(tmp_path) -> None:
    service, store, emails = build_service(tmp_path)
    confirmation = next(
        item for item in service.refresh(user_id="u1", now=NOW)
        if item.action_type == "email_event_confirmation"
    )

    emails.pending = False
    service.refresh(user_id="u1", now=NOW + timedelta(hours=1))

    resolved = store.get(user_id="u1", action_item_id=confirmation.id)
    assert resolved.status == "completed"
    assert resolved.resolved_at is not None


def test_snooze_rejects_timestamp_without_timezone(tmp_path) -> None:
    service, _, _ = build_service(tmp_path)
    item = service.refresh(user_id="u1", now=NOW)[0]

    with pytest.raises(InvalidActionTransitionError, match="requires a timezone"):
        service.snooze_action(
            user_id="u1",
            action_item_id=item.id,
            snoozed_until=datetime(2026, 8, 27),
            now=NOW,
        )
