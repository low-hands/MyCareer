"""The capture-intent chain: BOSS opened by the agent → job saved by the
extension → a durable ``job_captured`` event the page turns into a new turn."""

from __future__ import annotations

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from career_agent.api.app import create_app
from career_agent.storage.api_keys import CAPTURE_WRITE, CHAT_WRITE, WORKSPACE_READ
from career_agent.storage.job_captures import SQLiteJobCaptureStore
from career_agent.storage.jobs import SQLiteJobPostingRepository

CAPTURE_HEADERS = {"X-Career-Agent-Capture": "v1"}


class Runtime:
    """Never reached: the capture chain ends at the durable event; the page
    starts the follow-up turn through ``/v1/chat/stream`` on its own."""

    def run_turn(self, **arguments):
        raise AssertionError("a capture must not run a turn by itself")


def _job(source_job_id: str = "boss-123", **overrides) -> dict:
    return {
        "source_url": f"https://www.zhipin.com/job_detail/{source_job_id}.html",
        "title": "AI 产品经理",
        "company_name": "示例科技",
        "description": "负责 AI 产品规划和交付。",
        **overrides,
    }


@pytest.fixture
def stores(tmp_path):
    path = tmp_path / "jobs.sqlite3"
    return SQLiteJobPostingRepository(path), SQLiteJobCaptureStore(path)


@pytest.fixture
def app(api_keys, stores):
    repository, capture_store = stores
    return create_app(
        api_key_store_factory=lambda: api_keys,
        runtime_factory=Runtime,
        capture_repository_factory=lambda: repository,
        job_capture_store_factory=lambda: capture_store,
    )


def test_save_with_live_intent_records_one_event_for_the_original_conversation(
    app, stores, auth
) -> None:
    repository, capture_store = stores
    intent = capture_store.create_intent(
        user_id="u1", conversation_id="c1", platform="boss", keyword="AI", city=None
    )

    with TestClient(app) as client:
        first = client.post(
            "/v1/browser-captures/jobs",
            headers={**auth, **CAPTURE_HEADERS},
            json=_job(capture_intent_id=intent.id),
        )
        # A second click on the same "save" button.
        duplicate = client.post(
            "/v1/browser-captures/jobs",
            headers={**auth, **CAPTURE_HEADERS},
            json=_job(capture_intent_id=intent.id),
        )
        pending = client.get("/v1/job-captures/events", headers=auth)
        for_conversation = client.get(
            "/v1/job-captures/events", headers=auth, params={"conversation_id": "c1"}
        )
        other_conversation = client.get(
            "/v1/job-captures/events", headers=auth, params={"conversation_id": "c2"}
        )

    assert first.status_code == 200
    body = first.json()
    assert body["conversation_id"] == "c1"
    assert body["capture_event_created"] is True
    assert body["capture_event_id"].startswith("jobcap_")
    assert duplicate.status_code == 200
    assert duplicate.json()["capture_event_created"] is False
    assert duplicate.json()["capture_event_id"] == body["capture_event_id"]

    events = pending.json()["events"]
    assert len(events) == 1
    assert events[0]["id"] == body["capture_event_id"]
    assert events[0]["conversation_id"] == "c1"
    assert events[0]["job_posting_id"] == body["job_posting_id"]
    assert events[0]["jd_snapshot_id"] == body["jd_snapshot_id"]
    assert for_conversation.json() == pending.json()
    assert other_conversation.json() == {"events": []}
    # The saved job is the one the event points at.
    assert repository.get_job(user_id="u1", job_posting_id=events[0]["job_posting_id"]) is not None


def test_save_without_intent_only_enters_the_library(app, stores, auth) -> None:
    repository, _ = stores

    with TestClient(app) as client:
        response = client.post(
            "/v1/browser-captures/jobs",
            headers={**auth, **CAPTURE_HEADERS},
            json=_job(),
        )
        pending = client.get("/v1/job-captures/events", headers=auth)

    assert response.status_code == 200
    assert response.json()["conversation_id"] is None
    assert response.json()["capture_event_id"] is None
    assert pending.json() == {"events": []}
    assert len(repository.list_jobs(user_id="u1", include_dismissed=True)) == 1


def test_expired_or_foreign_intent_still_saves_but_starts_no_conversation(
    app, stores, auth, issue_key
) -> None:
    repository, capture_store = stores
    expired = capture_store.create_intent(
        user_id="u1",
        conversation_id="c1",
        platform="boss",
        keyword="AI",
        city=None,
        ttl=timedelta(microseconds=1),
    )
    someone_elses = capture_store.create_intent(
        user_id="u2", conversation_id="c-other", platform="boss", keyword="AI", city=None
    )
    other_user = issue_key("u2")

    with TestClient(app) as client:
        stale = client.post(
            "/v1/browser-captures/jobs",
            headers={**auth, **CAPTURE_HEADERS},
            json=_job("boss-1", capture_intent_id=expired.id),
        )
        # u1 presents u2's intent id: the save is u1's, the event never exists.
        foreign = client.post(
            "/v1/browser-captures/jobs",
            headers={**auth, **CAPTURE_HEADERS},
            json=_job("boss-2", capture_intent_id=someone_elses.id),
        )
        malformed = client.post(
            "/v1/browser-captures/jobs",
            headers={**auth, **CAPTURE_HEADERS},
            json=_job("boss-3", capture_intent_id="c-other"),
        )
        u1_pending = client.get("/v1/job-captures/events", headers=auth)
        u2_pending = client.get("/v1/job-captures/events", headers=other_user)

    assert stale.status_code == 200
    assert stale.json()["conversation_id"] is None
    assert foreign.status_code == 200
    assert foreign.json()["conversation_id"] is None
    assert malformed.status_code == 422
    assert u1_pending.json() == {"events": []}
    assert u2_pending.json() == {"events": []}
    assert len(repository.list_jobs(user_id="u1", include_dismissed=True)) == 2
    assert repository.list_jobs(user_id="u2", include_dismissed=True) == ()


def test_two_conversations_searching_at_once_do_not_cross(app, stores, auth) -> None:
    _, capture_store = stores
    first = capture_store.create_intent(
        user_id="u1", conversation_id="c1", platform="boss", keyword="AI", city=None
    )
    second = capture_store.create_intent(
        user_id="u1", conversation_id="c2", platform="boss", keyword="数据", city=None
    )

    with TestClient(app) as client:
        client.post(
            "/v1/browser-captures/jobs",
            headers={**auth, **CAPTURE_HEADERS},
            json=_job("boss-1", capture_intent_id=first.id),
        )
        client.post(
            "/v1/browser-captures/jobs",
            headers={**auth, **CAPTURE_HEADERS},
            json=_job("boss-2", title="数据分析师", capture_intent_id=second.id),
        )
        events = client.get("/v1/job-captures/events", headers=auth).json()["events"]

    assert [(event["conversation_id"], event["title"]) for event in events] == [
        ("c1", "AI 产品经理"),
        ("c2", "数据分析师"),
    ]


def test_event_waits_for_the_page_and_is_acknowledged_by_its_owner_only(
    api_keys, stores, auth, issue_key
) -> None:
    repository, capture_store = stores
    intent = capture_store.create_intent(
        user_id="u1", conversation_id="c1", platform="boss", keyword="AI", city=None
    )
    other_user = issue_key("u2")

    # Page closed: only the extension talks to the backend.
    with TestClient(
        create_app(
            api_key_store_factory=lambda: api_keys,
            runtime_factory=Runtime,
            capture_repository_factory=lambda: repository,
            job_capture_store_factory=lambda: capture_store,
        )
    ) as client:
        saved = client.post(
            "/v1/browser-captures/jobs",
            headers={**auth, **CAPTURE_HEADERS},
            json=_job(capture_intent_id=intent.id),
        ).json()

    # Page reopened later, against a fresh process on the same stores.
    with TestClient(
        create_app(
            api_key_store_factory=lambda: api_keys,
            runtime_factory=Runtime,
            capture_repository_factory=lambda: repository,
            job_capture_store_factory=lambda: capture_store,
        )
    ) as client:
        pending = client.get("/v1/job-captures/events", headers=auth).json()["events"]
        foreign_ack = client.post(
            f"/v1/job-captures/events/{saved['capture_event_id']}/ack",
            headers=other_user,
        )
        still_pending = client.get("/v1/job-captures/events", headers=auth).json()["events"]
        ack = client.post(
            f"/v1/job-captures/events/{saved['capture_event_id']}/ack", headers=auth
        )
        again = client.post(
            f"/v1/job-captures/events/{saved['capture_event_id']}/ack", headers=auth
        )
        drained = client.get("/v1/job-captures/events", headers=auth).json()["events"]

    assert [event["id"] for event in pending] == [saved["capture_event_id"]]
    assert foreign_ack.json() == {
        "event_id": saved["capture_event_id"],
        "acknowledged": False,
    }
    assert still_pending == pending
    assert ack.json()["acknowledged"] is True
    assert again.json()["acknowledged"] is False
    assert drained == []


def test_capture_event_routes_need_the_page_scopes_not_the_extension_scope(
    app, issue_key
) -> None:
    capture_only = issue_key("u1", CAPTURE_WRITE)
    read_only = issue_key("u1", WORKSPACE_READ)
    chat_only = issue_key("u1", CHAT_WRITE)

    with TestClient(app) as client:
        listed_by_extension = client.get("/v1/job-captures/events", headers=capture_only)
        listed_by_page = client.get("/v1/job-captures/events", headers=read_only)
        acked_by_reader = client.post(
            "/v1/job-captures/events/jobcap_x/ack", headers=read_only
        )
        acked_by_chat = client.post(
            "/v1/job-captures/events/jobcap_x/ack", headers=chat_only
        )
        anonymous = client.get("/v1/job-captures/events")

    assert listed_by_extension.status_code == 403
    assert listed_by_page.status_code == 200
    assert acked_by_reader.status_code == 403
    assert acked_by_chat.status_code == 200
    assert acked_by_chat.json()["acknowledged"] is False
    assert anonymous.status_code == 401
