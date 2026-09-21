from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from career_agent.domain.email_tracking import EmailEvent
from career_agent.domain.interviews import InterviewRound
from career_agent.agent.resume_tailoring_contracts import ResumeTailoringResult
from career_agent.services.action_center import (
    ActionCenterService,
    InvalidActionTransitionError,
)
from career_agent.storage.action_center import SQLiteActionItemStore
from career_agent.storage.resume_tailoring import SQLiteResumeTailoringDraftStore


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


def test_unresolved_tailoring_gaps_become_actionable_and_idempotent(tmp_path) -> None:
    draft_store = SQLiteResumeTailoringDraftStore(tmp_path / "resumes.sqlite3")
    draft = draft_store.create(
        user_id="u1",
        match_id="match-1",
        tailoring_goal=None,
        worker_version="test",
        result=ResumeTailoringResult(
            strategy_summary="突出直接证据。",
            unresolved_gaps=("缺少 Python 熟练度证据", "缺少可到岗时间"),
        ),
    )
    service = ActionCenterService(
        SQLiteActionItemStore(tmp_path / "actions.sqlite3"),
        Applications(),
        Emails(),
        Interviews(),
        resume_tailoring_drafts=draft_store,
    )

    first = service.refresh(user_id="u1", now=NOW)
    repeated = service.refresh(user_id="u1", now=NOW + timedelta(minutes=1))

    gap = next(item for item in first if item.action_type == "resume_gap_resolution")
    assert gap.source_type == "resume_tailoring_draft"
    assert gap.source_id == draft.id
    assert "Python 熟练度" in gap.summary
    assert "可到岗时间" in gap.summary
    assert next(
        item for item in repeated if item.action_type == "resume_gap_resolution"
    ).id == gap.id


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


def test_recording_source_work_can_complete_matching_retro_action(tmp_path) -> None:
    service, store, _ = build_service(tmp_path)
    retro = next(
        item
        for item in service.refresh(user_id="u1", now=NOW)
        if item.action_type == "interview_retro"
    )

    completed = service.complete_source_action(
        user_id="u1",
        action_type="interview_retro",
        source_id="interview-completed",
    )

    assert completed is not None
    assert completed.id == retro.id
    assert completed.status == "completed"
    assert store.get(user_id="u1", action_item_id=retro.id).status == "completed"


def test_source_resolution_marks_generated_action_obsolete_not_completed(
    tmp_path,
) -> None:
    service, store, emails = build_service(tmp_path)
    confirmation = next(
        item for item in service.refresh(user_id="u1", now=NOW)
        if item.action_type == "email_event_confirmation"
    )

    emails.pending = False
    service.refresh(user_id="u1", now=NOW + timedelta(hours=1))

    # The pending event disappeared on its own; the user never acted on it, so
    # the item must not be counted as completed work.
    resolved = store.get(user_id="u1", action_item_id=confirmation.id)
    assert resolved.status == "obsolete"
    assert resolved.resolved_at is not None
    assert [event.event_type for event in store.list_events(
        user_id="u1", action_item_id=confirmation.id
    )][-1] == "obsoleted"


def test_obsolete_action_reopens_when_its_condition_returns(tmp_path) -> None:
    service, store, emails = build_service(tmp_path)
    confirmation = next(
        item for item in service.refresh(user_id="u1", now=NOW)
        if item.action_type == "email_event_confirmation"
    )
    emails.pending = False
    service.refresh(user_id="u1", now=NOW + timedelta(hours=1))

    emails.pending = True
    service.refresh(user_id="u1", now=NOW + timedelta(hours=2))

    # Nobody decided this was done, so the returning condition owes the user
    # attention again rather than staying silently resolved.
    reopened = store.get(user_id="u1", action_item_id=confirmation.id)
    assert reopened.status == "open"
    assert reopened.resolved_at is None
    assert [event.event_type for event in store.list_events(
        user_id="u1", action_item_id=confirmation.id
    )][-1] == "reopened"


def test_user_completed_action_stays_resolved_when_its_condition_returns(
    tmp_path,
) -> None:
    service, store, emails = build_service(tmp_path)
    confirmation = next(
        item for item in service.refresh(user_id="u1", now=NOW)
        if item.action_type == "email_event_confirmation"
    )
    service.complete_action(user_id="u1", action_item_id=confirmation.id)

    emails.pending = False
    service.refresh(user_id="u1", now=NOW + timedelta(hours=1))
    emails.pending = True
    service.refresh(user_id="u1", now=NOW + timedelta(hours=2))

    # A user decision is not a system observation: refresh must not undo it.
    assert store.get(user_id="u1", action_item_id=confirmation.id).status == "completed"


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


class OneApplication:
    """A single application whose status and last-progress time the test owns."""

    def __init__(self, *, status: str, updated_at: datetime) -> None:
        self.application = SimpleNamespace(
            id="app-1", status=status, updated_at=updated_at
        )
        self.job = SimpleNamespace(
            posting=SimpleNamespace(company_name="Acme", title="AI Engineer")
        )

    def list_applications(self, **kwargs):
        return (self,)


class NoInterviews:
    def list_interviews(self, **kwargs):
        return ()


class NoEmails:
    def list_events(self, **kwargs):
        return ()


def build_follow_up_service(tmp_path, *, status: str, quiet_days: float):
    store = SQLiteActionItemStore(tmp_path / "actions.sqlite3")
    applications = OneApplication(
        status=status, updated_at=NOW - timedelta(days=quiet_days)
    )
    service = ActionCenterService(
        store, applications, NoEmails(), NoInterviews()
    )
    return service


def follow_ups(service, *, now):
    return [
        item
        for item in service.refresh(user_id="u1", now=now)
        if item.action_type == "application_follow_up"
    ]


@pytest.mark.parametrize(
    "status, quiet_days, expected",
    [
        # Silence shorter than the status's own interval is not yet a follow-up.
        ("submitted", 6, 0),
        ("submitted", 7, 1),
        # A reply puts the application in an active conversation, where waiting
        # a full week before following up is already too late.
        ("acknowledged", 2, 0),
        ("acknowledged", 3, 1),
        ("interviewing", 4, 0),
        ("interviewing", 5, 1),
    ],
)
def test_each_status_waits_its_own_interval(
    tmp_path, status, quiet_days, expected
) -> None:
    service = build_follow_up_service(tmp_path, status=status, quiet_days=quiet_days)

    assert len(follow_ups(service, now=NOW)) == expected


def test_a_silent_application_stops_nagging_after_the_cap(tmp_path) -> None:
    """The whole point of the cap: dead applications leave the daily brief.

    Without it every quiet application generates one more reminder per cycle
    forever, each under a different stable key, until the brief is nothing but
    applications that already went nowhere.
    """
    service = build_follow_up_service(tmp_path, status="submitted", quiet_days=7)

    first = follow_ups(service, now=NOW)
    second = follow_ups(service, now=NOW + timedelta(days=7))
    third = follow_ups(service, now=NOW + timedelta(days=14))
    much_later = follow_ups(service, now=NOW + timedelta(days=365))

    assert len(first) == 1
    assert len(second) == 1
    assert first[0].id != second[0].id
    assert "最后一次" in second[0].summary
    assert third == []
    assert much_later == []


def test_the_last_reminder_says_it_is_the_last(tmp_path) -> None:
    """A reminder that simply stops appearing reads like a lost application."""
    service = build_follow_up_service(tmp_path, status="submitted", quiet_days=7)

    first = follow_ups(service, now=NOW)

    assert "最后一次" not in first[0].summary


def test_new_progress_restarts_the_cadence(tmp_path) -> None:
    """The cap counts silence, not the application's whole lifetime."""
    store = SQLiteActionItemStore(tmp_path / "actions.sqlite3")
    applications = OneApplication(
        status="submitted", updated_at=NOW - timedelta(days=30)
    )
    service = ActionCenterService(store, applications, NoEmails(), NoInterviews())
    assert follow_ups(service, now=NOW) == []

    applications.application = SimpleNamespace(
        id="app-1", status="acknowledged", updated_at=NOW
    )

    assert len(follow_ups(service, now=NOW + timedelta(days=3))) == 1


def test_a_terminal_application_is_never_followed_up(tmp_path) -> None:
    for status in ("offer", "rejected", "withdrawn"):
        service = build_follow_up_service(
            tmp_path / status, status=status, quiet_days=90
        )
        assert follow_ups(service, now=NOW) == []
