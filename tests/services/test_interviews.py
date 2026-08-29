from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from career_agent.domain.interviews import InterviewDetails, InterviewRetroQuestion
from career_agent.services.interviews import (
    AmbiguousInterviewMatchError,
    InterviewApplicationConflictError,
    InterviewService,
)
from career_agent.storage.interviews import SQLiteInterviewStore


NOW = datetime(2026, 8, 26, 2, 0, tzinfo=timezone.utc)


class Applications:
    def __init__(self):
        self.application = SimpleNamespace(id="app-1", status="interviewing")

    def get_application(self, *, user_id, application_id):
        if user_id != "u1" or application_id != "app-1":
            raise ValueError("application unavailable")
        return SimpleNamespace(application=self.application)


def details(
    *,
    change_type="invited",
    start=NOW,
    meeting_url="https://meet.example/one",
    employer_label=None,
):
    return InterviewDetails(
        change_type=change_type,
        employer_label=employer_label,
        scheduled_start=start,
        scheduled_end=(start + timedelta(hours=1) if start else None),
        timezone="Asia/Shanghai" if start else None,
        interview_format="video",
        meeting_url=meeting_url,
    )


def test_same_thread_updates_one_round_and_new_thread_allocates_next_sequence(tmp_path) -> None:
    store = SQLiteInterviewStore(tmp_path / "applications.sqlite3")
    service = InterviewService(store, Applications())

    first = service.record_email_event(
        user_id="u1", application_id="app-1", email_event_id="email-1",
        source_thread_id="thread-1", details=details(), occurred_at=NOW,
    )
    updated = service.record_email_event(
        user_id="u1", application_id="app-1", email_event_id="email-2",
        source_thread_id="thread-1",
        details=details(change_type="details_updated", employer_label="技术面"),
        occurred_at=NOW + timedelta(minutes=5),
    )
    second = service.record_email_event(
        user_id="u1", application_id="app-1", email_event_id="email-3",
        source_thread_id="thread-2",
        details=details(
            start=NOW + timedelta(days=2),
            meeting_url="https://meet.example/two",
        ),
        occurred_at=NOW + timedelta(days=1),
    )

    assert first.created is True
    assert updated.created is False
    assert updated.interview.id == first.interview.id
    assert updated.interview.sequence_number == 1
    assert updated.interview.employer_label == "技术面"
    assert second.created is True
    assert second.interview.sequence_number == 2
    assert len(service.get_interview(user_id="u1", interview_round_id=first.interview.id).events) == 2


def test_reschedule_updates_same_round_and_preserves_history(tmp_path) -> None:
    store = SQLiteInterviewStore(tmp_path / "applications.sqlite3")
    service = InterviewService(store, Applications())
    created = service.record_email_event(
        user_id="u1", application_id="app-1", email_event_id="email-1",
        source_thread_id="thread-1", details=details(), occurred_at=NOW,
    ).interview
    new_start = NOW + timedelta(days=1)

    rescheduled = service.record_email_event(
        user_id="u1", application_id="app-1", email_event_id="email-2",
        source_thread_id="thread-1",
        details=details(change_type="rescheduled", start=new_start),
        occurred_at=NOW + timedelta(hours=1),
    ).interview

    assert rescheduled.id == created.id
    assert rescheduled.scheduled_start == new_start
    events = service.get_interview(
        user_id="u1", interview_round_id=created.id
    ).events
    assert [event.event_type for event in events] == ["created", "rescheduled"]
    assert events[0].details.scheduled_start == NOW


def test_ambiguous_update_waits_for_explicit_round_selection(tmp_path) -> None:
    store = SQLiteInterviewStore(tmp_path / "applications.sqlite3")
    service = InterviewService(store, Applications())
    for index in (1, 2):
        service.record_email_event(
            user_id="u1", application_id="app-1", email_event_id=f"email-{index}",
            source_thread_id=f"thread-{index}",
            details=details(
                start=NOW + timedelta(days=index),
                meeting_url=f"https://meet.example/{index}",
            ),
            occurred_at=NOW,
        )

    with pytest.raises(AmbiguousInterviewMatchError):
        service.record_email_event(
            user_id="u1", application_id="app-1", email_event_id="email-3",
            source_thread_id=None,
            details=InterviewDetails(change_type="cancelled"),
            occurred_at=NOW + timedelta(days=3),
        )


def test_email_event_is_idempotent_and_completion_requires_user_action(tmp_path) -> None:
    store = SQLiteInterviewStore(tmp_path / "applications.sqlite3")
    service = InterviewService(store, Applications())
    first = service.record_email_event(
        user_id="u1", application_id="app-1", email_event_id="email-1",
        source_thread_id="thread-1", details=details(), occurred_at=NOW,
    )
    repeated = service.record_email_event(
        user_id="u1", application_id="app-1", email_event_id="email-1",
        source_thread_id="thread-1", details=details(), occurred_at=NOW,
    )

    completed = service.complete_interview(
        user_id="u1", interview_round_id=first.interview.id,
        completed_at=NOW + timedelta(hours=1),
    )

    assert repeated.created is False
    assert repeated.interview.id == first.interview.id
    assert completed.status == "completed"
    assert completed.completed_at == NOW + timedelta(hours=1)


def test_real_interview_retro_requires_completion_and_is_versioned(tmp_path) -> None:
    store = SQLiteInterviewStore(tmp_path / "applications.sqlite3")
    service = InterviewService(store, Applications())
    interview = service.create_manual(
        user_id="u1",
        application_id="app-1",
        details=details(),
    )

    with pytest.raises(InterviewApplicationConflictError, match="completed"):
        service.record_retro(
            user_id="u1",
            interview_round_id=interview.id,
            source_notes="问了检索评测。",
            summary="需要补充评测方法。",
        )

    service.complete_interview(
        user_id="u1",
        interview_round_id=interview.id,
        completed_at=NOW + timedelta(hours=1),
    )
    first = service.record_retro(
        user_id="u1",
        interview_round_id=interview.id,
        source_notes="问了检索评测；数据集构造没答完整。",
        summary="评测框架基本清楚，数据集构造需要补强。",
        questions=(
            InterviewRetroQuestion(
                question="如何评估 RAG 检索效果？",
                answer_summary="回答了 Recall，但没讲数据集构造。",
                self_assessment="mixed",
            ),
        ),
        strengths=("知道核心检索指标",),
        difficulties=("没有说明评测集构造",),
        next_focus=("补充离线评测数据集设计",),
        action_items=("整理一版 RAG 评测回答",),
        limitations=("没有面试官书面反馈",),
        self_assessment="mixed",
    )
    duplicate = service.record_retro(
        user_id="u1",
        interview_round_id=interview.id,
        source_notes="问了检索评测；数据集构造没答完整。",
        summary="评测框架基本清楚，数据集构造需要补强。",
        questions=first.questions,
        strengths=first.strengths,
        difficulties=first.difficulties,
        next_focus=first.next_focus,
        action_items=first.action_items,
        limitations=first.limitations,
        self_assessment="mixed",
    )
    revised = service.record_retro(
        user_id="u1",
        interview_round_id=interview.id,
        source_notes="补充：面试官还追问了线上稳定性。",
        summary="需要同时准备评测与稳定性。",
        next_focus=("线上降级与监控",),
    )

    detail = service.get_interview(
        user_id="u1", interview_round_id=interview.id
    )
    assert duplicate.id == first.id
    assert revised.id != first.id
    assert [item.id for item in detail.retros] == [first.id, revised.id]
    assert detail.retros[0].questions[0].self_assessment == "mixed"


def test_new_invitation_after_completed_round_gets_new_internal_sequence(tmp_path) -> None:
    store = SQLiteInterviewStore(tmp_path / "applications.sqlite3")
    service = InterviewService(store, Applications())
    first = service.record_email_event(
        user_id="u1", application_id="app-1", email_event_id="email-1",
        source_thread_id="shared-thread", details=details(), occurred_at=NOW,
    ).interview
    service.complete_interview(
        user_id="u1", interview_round_id=first.id,
        completed_at=NOW + timedelta(hours=1),
    )

    next_interview = service.record_email_event(
        user_id="u1", application_id="app-1", email_event_id="email-2",
        source_thread_id="shared-thread",
        details=details(
            start=NOW + timedelta(days=3),
            meeting_url="https://meet.example/next",
        ),
        occurred_at=NOW + timedelta(days=2),
    ).interview

    assert next_interview.id != first.id
    assert next_interview.sequence_number == 2


def test_explicit_new_employer_label_can_split_appointments_in_same_thread(tmp_path) -> None:
    store = SQLiteInterviewStore(tmp_path / "applications.sqlite3")
    service = InterviewService(store, Applications())
    first = service.record_email_event(
        user_id="u1", application_id="app-1", email_event_id="email-1",
        source_thread_id="shared-thread",
        details=details(employer_label="一面"), occurred_at=NOW,
    ).interview

    second = service.record_email_event(
        user_id="u1", application_id="app-1", email_event_id="email-2",
        source_thread_id="shared-thread",
        details=details(
            start=NOW + timedelta(days=2),
            meeting_url="https://meet.example/second",
            employer_label="二面",
        ),
        occurred_at=NOW + timedelta(days=1),
    ).interview

    assert first.sequence_number == 1
    assert second.sequence_number == 2
    assert second.employer_label == "二面"
