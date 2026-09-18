"""Backend-owned capture continuation, receipt replay and owner-only delivery."""

from __future__ import annotations

from datetime import timedelta
from threading import Event, Thread
from time import monotonic, sleep

import pytest
from fastapi.testclient import TestClient

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import AgentDecision
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.agent.session_manager import SessionManager
from career_agent.api.app import create_app
from career_agent.storage.api_keys import (
    CAPTURE_WRITE, CHAT_WRITE, WORKSPACE_READ, WORKSPACE_WRITE,
)
from career_agent.storage.context import CareerContextStore
from career_agent.storage.job_captures import SQLiteJobCaptureStore
from career_agent.storage.jobs import SQLiteJobPostingRepository
from career_agent.storage.turn_receipts import SQLiteTurnReceiptStore

CAPTURE_HEADERS = {"X-Career-Agent-Capture": "v1"}


class ConversationReader:
    def __init__(self, context: CareerContextStore) -> None:
        self.context = context

    def delete_conversation(self, *, user_id: str, conversation_id: str) -> bool:
        return self.context.delete_conversation(
            user_id=user_id, conversation_id=conversation_id
        )


class DecisionMaker:
    def __init__(self) -> None:
        self.calls = 0

    def decide(self, context, tool_specs) -> AgentDecision:
        self.calls += 1
        return AgentDecision(action="final", message="已记下这个岗位。")


class BlockingDecisionMaker(DecisionMaker):
    def __init__(self) -> None:
        super().__init__()
        self.started = Event()
        self.finish = Event()

    def decide(self, context, tool_specs) -> AgentDecision:
        self.started.set()
        assert self.finish.wait(10)
        return super().decide(context, tool_specs)


def _runtime(context_store, stores, decisions) -> MainAgentRuntime:
    repository, captures = stores
    return MainAgentRuntime(
        context_manager=ContextManager(context_store),
        decision_maker=decisions,
        tools=MainAgentToolRegistry(
            job_repository=repository, job_capture_store=captures
        ),
        turn_receipt_store=SQLiteTurnReceiptStore(context_store.path),
    )


def _capture(client, stores, auth, *, conversation_id="c1"):
    _, captures = stores
    intent = captures.create_intent(
        user_id="u1", conversation_id=conversation_id,
        platform="boss", keyword="AI", city=None,
    )
    response = client.post(
        "/v1/browser-captures/jobs",
        headers={**auth, **CAPTURE_HEADERS},
        json=_job(capture_intent_id=intent.id),
    )
    assert response.status_code == 200
    return response.json()


def _continuation(saved):
    return {
        "conversation_id": saved["conversation_id"],
        "message": "已保存这个岗位，先记下来。",
        "input_resources": [{"kind": "jd_snapshot", "id": saved["jd_snapshot_id"]}],
    }


def _wait_for_capture(app, captures, event_id):
    deadline = monotonic() + 15
    while monotonic() < deadline:
        event = captures.get_event(user_id="u1", event_id=event_id)
        assert event is not None
        assert event.continuation_status not in {"failed", "discarded"}
        if event.continuation_status == "completed" and app.state.run_gate.active_count == 0:
            return event
        sleep(0.01)
    pytest.fail("backend continuation did not complete and release its gate")


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
def context_store(tmp_path):
    store = CareerContextStore(tmp_path / "context.sqlite3")
    sessions = SessionManager(store)
    for conversation_id in ("c1", "c2"):
        sessions.get_or_create(user_id="u1", session_id=conversation_id)
    return store


@pytest.fixture
def decisions():
    return DecisionMaker()


@pytest.fixture
def app(api_keys, stores, context_store, decisions):
    repository, capture_store = stores
    return create_app(
        api_key_store_factory=lambda: api_keys,
        runtime_factory=lambda: _runtime(context_store, stores, decisions),
        capture_repository_factory=lambda: repository,
        job_capture_store_factory=lambda: capture_store,
        owner_settings_store_factory=lambda: context_store,
        workspace_reader_factory=lambda: ConversationReader(context_store),
    )


def test_save_with_live_intent_records_one_event_for_the_original_conversation(
    app, stores, auth, decisions, context_store
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
        completed = _wait_for_capture(app, capture_store, first.json()["capture_event_id"])
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
    assert events[0]["continuation_status"] == "completed"
    assert events[0]["continuation_turn_id"] == completed.continuation_turn_id
    assert decisions.calls == 1
    assert len(context_store.list_messages("u1", "c1", limit=10)) == 2
    # The saved job is the one the event points at.
    assert repository.get_job(user_id="u1", job_posting_id=events[0]["job_posting_id"]) is not None


def test_save_without_intent_only_enters_the_library(app, stores, auth, decisions) -> None:
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
    assert decisions.calls == 0


def test_expired_or_foreign_intent_still_saves_but_starts_no_conversation(
    app, stores, auth, issue_key, decisions
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
    assert decisions.calls == 0


def test_two_conversations_searching_at_once_do_not_cross(
    app, stores, auth, decisions, context_store
) -> None:
    _, capture_store = stores
    first = capture_store.create_intent(
        user_id="u1", conversation_id="c1", platform="boss", keyword="AI", city=None
    )
    second = capture_store.create_intent(
        user_id="u1", conversation_id="c2", platform="boss", keyword="数据", city=None
    )

    with TestClient(app) as client:
        first_saved = client.post(
            "/v1/browser-captures/jobs",
            headers={**auth, **CAPTURE_HEADERS},
            json=_job("boss-1", capture_intent_id=first.id),
        )
        second_saved = client.post(
            "/v1/browser-captures/jobs",
            headers={**auth, **CAPTURE_HEADERS},
            json=_job("boss-2", title="数据分析师", capture_intent_id=second.id),
        )
        for saved in (first_saved, second_saved):
            _wait_for_capture(app, capture_store, saved.json()["capture_event_id"])
        events = client.get("/v1/job-captures/events", headers=auth).json()["events"]

    assert [(event["conversation_id"], event["title"]) for event in events] == [
        ("c1", "AI 产品经理"),
        ("c2", "数据分析师"),
    ]
    assert decisions.calls == 2
    for saved in (first_saved, second_saved):
        body = saved.json()
        messages = context_store.list_messages("u1", body["conversation_id"], limit=10)
        assert len(messages) == 2
        assert all(message.resource_refs[0].resource_id == body["jd_snapshot_id"] for message in messages)


def test_completed_event_waits_for_owner_ack_after_the_page_reopens(
    api_keys, stores, auth, issue_key, context_store, decisions
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
            runtime_factory=lambda: _runtime(context_store, stores, decisions),
            capture_repository_factory=lambda: repository,
            job_capture_store_factory=lambda: capture_store,
            owner_settings_store_factory=lambda: context_store,
        )
    ) as client:
        saved = client.post(
            "/v1/browser-captures/jobs",
            headers={**auth, **CAPTURE_HEADERS},
            json=_job(capture_intent_id=intent.id),
        ).json()
        _wait_for_capture(client.app, capture_store, saved["capture_event_id"])

    # Page reopened later, against a fresh process on the same stores.
    with TestClient(
        create_app(
            api_key_store_factory=lambda: api_keys,
            runtime_factory=lambda: _runtime(context_store, stores, decisions),
            capture_repository_factory=lambda: repository,
            job_capture_store_factory=lambda: capture_store,
            owner_settings_store_factory=lambda: context_store,
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
    assert pending[0]["continuation_status"] == "completed"
    assert decisions.calls == 1
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


@pytest.mark.parametrize("state", ["deleted", "closed", "missing"])
def test_unavailable_conversation_only_saves_to_library(
    app, stores, auth, context_store, issue_key, state, decisions
) -> None:
    repository, captures = stores
    conversation_id = "missing" if state == "missing" else "c1"
    intent = captures.create_intent(
        user_id="u1", conversation_id=conversation_id,
        platform="boss", keyword="AI", city=None,
    )
    with TestClient(app) as client:
        if state == "deleted":
            assert client.delete(
                "/v1/conversations/c1", headers=issue_key("u1", WORKSPACE_WRITE)
            ).status_code == 200
        elif state == "closed":
            context_store.close_session("u1", "c1")
        saved = client.post(
            "/v1/browser-captures/jobs",
            headers={**auth, **CAPTURE_HEADERS},
            json=_job(capture_intent_id=intent.id),
        )
        assert saved.status_code == 200
        assert saved.json()["conversation_id"] is None
        assert saved.json()["capture_event_id"] is None
        assert saved.json()["capture_event_created"] is False
        assert client.get(
            "/v1/job-captures/events", headers=auth
        ).json() == {"events": []}
    assert len(repository.list_jobs(user_id="u1", include_dismissed=True)) == 1
    assert decisions.calls == 0


@pytest.mark.parametrize("poll_before_retry", [True, False])
def test_deleted_conversation_cannot_be_recreated_by_a_cached_capture(
    app, stores, auth, context_store, issue_key, poll_before_retry
) -> None:
    with TestClient(app) as client:
        saved = _capture(client, stores, auth)
        _wait_for_capture(app, stores[1], saved["capture_event_id"])
        assert client.delete(
            "/v1/conversations/c1", headers=issue_key("u1", WORKSPACE_WRITE)
        ).status_code == 200
        if poll_before_retry:
            assert client.get(
                "/v1/job-captures/events", headers=auth
            ).json() == {"events": []}
        refused = client.post(
            "/v1/chat/stream",
            headers={**auth, "Idempotency-Key": saved["capture_event_id"]},
            json=_continuation(saved),
        )
        assert refused.status_code == 409
        assert refused.json()["detail"]["code"] == "JOB_CAPTURE_CONTINUATION_BACKEND_OWNED"
        assert app.state.run_gate.active_count == 0
        assert client.get(
            "/v1/job-captures/events", headers=auth
        ).json() == {"events": []}
    assert context_store.get_session("u1", "c1") is None
    assert context_store.list_messages("u1", "c1", limit=10) == ()


def test_pruning_deleted_capture_keeps_other_conversations_pending(
    app, stores, auth, issue_key
) -> None:
    with TestClient(app) as client:
        first = _capture(client, stores, auth)
        second = _capture(client, stores, auth, conversation_id="c2")
        for saved in (first, second):
            _wait_for_capture(app, stores[1], saved["capture_event_id"])
        assert client.delete(
            "/v1/conversations/c1", headers=issue_key("u1", WORKSPACE_WRITE)
        ).status_code == 200
        events = client.get("/v1/job-captures/events", headers=auth).json()["events"]
    assert [event["id"] for event in events] == [second["capture_event_id"]]
    captures = stores[1]
    assert captures.get_event(
        user_id="u1", event_id=first["capture_event_id"]
    ).acknowledged_at is not None


@pytest.mark.parametrize("key", ["own", "foreign", "unknown"])
def test_chat_refuses_capture_keys_immediately_while_the_backend_turn_runs(
    app, stores, auth, issue_key, key, context_store
) -> None:
    decisions = BlockingDecisionMaker()
    with TestClient(app) as client:
        app.state.runtime = _runtime(context_store, stores, decisions)
        saved = _capture(client, stores, auth)
        headers = {**auth, "Idempotency-Key": saved["capture_event_id"]}
        if key == "foreign":
            headers = {**issue_key("u2"), "Idempotency-Key": saved["capture_event_id"]}
        elif key == "unknown":
            headers["Idempotency-Key"] = "jobcap_" + "0" * 32
        try:
            assert decisions.started.wait(10)
            started = monotonic()
            refused = client.post(
                "/v1/chat/stream", headers=headers, json=_continuation(saved)
            )
            assert monotonic() - started < 1
            assert refused.status_code == 409
            assert refused.json()["detail"]["code"] == "JOB_CAPTURE_CONTINUATION_BACKEND_OWNED"
            assert app.state.run_gate.active_count == 1
        finally:
            decisions.finish.set()
        _wait_for_capture(app, stores[1], saved["capture_event_id"])
        assert app.state.run_gate.active_count == 0
    assert decisions.calls == 1
    assert len(context_store.list_messages("u1", "c1", limit=10)) == 2
    assert context_store.get_session("u2", "c1") is None


def test_deletion_waits_for_a_running_capture_turn(
    app, stores, auth, context_store, issue_key
) -> None:
    decisions = BlockingDecisionMaker()
    with TestClient(app) as client:
        app.state.runtime = _runtime(context_store, stores, decisions)
        saved = _capture(client, stores, auth)
        write_headers = issue_key("u1", WORKSPACE_WRITE)
        try:
            assert decisions.started.wait(10)
            refused = client.delete("/v1/conversations/c1", headers=write_headers)
            assert refused.status_code == 409
            assert refused.json()["detail"]["code"] == "CONVERSATION_TURN_IN_PROGRESS"
            assert context_store.get_session("u1", "c1") is not None
        finally:
            decisions.finish.set()
        _wait_for_capture(app, stores[1], saved["capture_event_id"])
        assert client.delete(
            "/v1/conversations/c1", headers=write_headers
        ).status_code == 200
    assert context_store.get_session("u1", "c1") is None


def test_shutdown_waits_for_a_running_capture_turn_to_settle(
    app, stores, auth, context_store
) -> None:
    decisions = BlockingDecisionMaker()
    client = TestClient(app)
    client.__enter__()
    app.state.runtime = _runtime(context_store, stores, decisions)
    saved = _capture(client, stores, auth)
    assert decisions.started.wait(10)
    closing = Thread(target=client.__exit__, args=(None, None, None))
    closing.start()
    sleep(0.3)
    # Shutdown is draining the gate the capture worker still holds.
    assert closing.is_alive()
    decisions.finish.set()
    closing.join(15)
    assert not closing.is_alive()
    assert app.state.shutdown_drained is True
    event = stores[1].get_event(user_id="u1", event_id=saved["capture_event_id"])
    assert event is not None and event.continuation_status == "completed"
