from __future__ import annotations

from datetime import timedelta

import pytest

from career_agent.storage.job_captures import SQLiteJobCaptureStore


@pytest.fixture
def store(tmp_path) -> SQLiteJobCaptureStore:
    return SQLiteJobCaptureStore(tmp_path / "jobs.sqlite3")


def _intent(store, *, conversation_id="c1", user_id="u1", **overrides):
    return store.create_intent(
        user_id=user_id,
        conversation_id=conversation_id,
        platform="boss",
        keyword="AI 产品经理",
        city="上海",
        **overrides,
    )


def test_intent_is_durable_and_bound_to_its_owner(tmp_path) -> None:
    path = tmp_path / "jobs.sqlite3"
    intent = _intent(SQLiteJobCaptureStore(path))

    assert intent.id.startswith("capint_")
    assert intent.expires_at - intent.created_at == timedelta(hours=2)
    # A fresh handle on the same file sees the intent: nothing lives in memory.
    reopened = SQLiteJobCaptureStore(path)
    assert reopened.get_live_intent(user_id="u1", intent_id=intent.id) == intent
    # Another user's id for the same intent reads as "no intent", not an error.
    assert reopened.get_live_intent(user_id="u2", intent_id=intent.id) is None
    assert reopened.get_live_intent(user_id="u1", intent_id="capint_unknown") is None


def test_expired_intent_reads_as_absent(store) -> None:
    intent = _intent(store, ttl=timedelta(minutes=5))

    still_live = store.get_live_intent(
        user_id="u1",
        intent_id=intent.id,
        now=intent.created_at + timedelta(minutes=4),
    )
    expired = store.get_live_intent(
        user_id="u1",
        intent_id=intent.id,
        now=intent.created_at + timedelta(minutes=5),
    )

    assert still_live == intent
    assert expired is None
    with pytest.raises(ValueError):
        _intent(store, ttl=timedelta(0))


def test_consumed_intent_only_replays_the_exact_snapshot(store) -> None:
    intent = _intent(store)

    first = store.record_capture(
        intent=intent,
        job_posting_id="job-1",
        jd_snapshot_id="snap-1",
        title="AI 产品经理",
        company_name="示例科技",
    )
    duplicate = store.record_capture(
        intent=intent,
        job_posting_id="job-1",
        jd_snapshot_id="snap-1",
        title="AI 产品经理",
        company_name="示例科技",
    )
    second = store.record_capture(
        intent=intent,
        job_posting_id="job-1",
        jd_snapshot_id="snap-2",
        title="AI 产品经理",
        company_name="示例科技",
    )
    other_job = store.record_capture(
        intent=intent,
        job_posting_id="job-2",
        jd_snapshot_id="snap-3",
        title="AI 产品总监",
        company_name="示例科技",
    )

    assert first is not None and first.created is True
    assert duplicate is not None and duplicate.created is False
    assert duplicate.event == first.event
    assert second is None
    assert other_job is None
    consumed = store.get_intent(user_id="u1", intent_id=intent.id)
    assert consumed is not None and consumed.consumed_event_id == first.event.id
    assert store.get_live_intent(user_id="u1", intent_id=intent.id) is None
    assert [event.job_posting_id for event in store.list_pending_events(user_id="u1")] == [
        "job-1",
    ]


def test_events_stay_in_their_own_conversation(store) -> None:
    first_search = _intent(store, conversation_id="c1")
    second_search = _intent(store, conversation_id="c2")

    store.record_capture(
        intent=first_search,
        job_posting_id="job-1",
        jd_snapshot_id="snap-1",
        title="A",
        company_name="X",
    )
    store.record_capture(
        intent=second_search,
        job_posting_id="job-2",
        jd_snapshot_id="snap-2",
        title="B",
        company_name="Y",
    )

    for_first = store.list_pending_events(user_id="u1", conversation_id="c1")
    for_second = store.list_pending_events(user_id="u1", conversation_id="c2")
    assert [event.job_posting_id for event in for_first] == ["job-1"]
    assert [event.job_posting_id for event in for_second] == ["job-2"]
    assert for_first[0].conversation_id == "c1"
    assert for_second[0].conversation_id == "c2"
    assert store.list_pending_events(user_id="u2") == ()


def test_pending_event_survives_until_its_owner_acknowledges_it(tmp_path) -> None:
    path = tmp_path / "jobs.sqlite3"
    store = SQLiteJobCaptureStore(path)
    intent = _intent(store)
    event = store.record_capture(
        intent=intent,
        job_posting_id="job-1",
        jd_snapshot_id="snap-1",
        title="A",
        company_name="X",
    ).event

    # The page that opened the search is gone; the next one reads the event.
    reopened = SQLiteJobCaptureStore(path)
    assert reopened.list_pending_events(user_id="u1") == (event,)
    assert reopened.acknowledge_event(user_id="u2", event_id=event.id) is False
    assert reopened.list_pending_events(user_id="u1") == (event,)
    assert reopened.acknowledge_event(user_id="u1", event_id=event.id) is True
    assert reopened.acknowledge_event(user_id="u1", event_id=event.id) is False
    assert reopened.list_pending_events(user_id="u1") == ()
