from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
import time

import pytest
from fastapi.testclient import TestClient

from career_agent.api.app import (
    ChatStreamRequest,
    ConversationBusyError,
    ConversationRunGate,
    GateAwareStreamingResponse,
    _runtime_args_from_env,
    _sse_stream,
    create_app,
)
from career_agent.agent.openai_compatible_client import AgentConfigurationError
from career_agent.harness.streaming import (
    ContentDeltaEvent,
    InteractionResponse,
    TurnCompletedEvent,
    TurnFailedEvent,
    TurnStartedEvent,
)
from career_agent.storage.api_keys import CAPTURE_WRITE, WORKSPACE_READ
from career_agent.storage.jobs import SQLiteJobPostingRepository


class Runtime:
    def __init__(self, *, fail: bool = False, delay: float = 0.0) -> None:
        self.fail = fail
        self.delay = delay
        self.calls = []
        self.closed = False

    def run_turn(
        self,
        *,
        user_id,
        conversation_id,
        user_message,
        request_id=None,
        interaction_response=None,
        event_sink=None,
    ):
        self.calls.append((user_id, conversation_id, user_message))
        self.request_id = request_id
        self.interaction_response = interaction_response
        assert event_sink is not None
        event_sink(TurnStartedEvent(turn_id="turn-1"))
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            event_sink(
                TurnFailedEvent(
                    turn_id="turn-1",
                    code="TURN_EXECUTION_FAILED",
                    message="安全错误消息。",
                )
            )
            raise RuntimeError("PRIVATE INTERNAL ERROR")
        event_sink(ContentDeltaEvent(delta="你好"))
        event_sink(ContentDeltaEvent(delta="，世界"))
        event_sink(TurnCompletedEvent(turn_id="turn-1"))
        return object()

    def close(self) -> None:
        self.closed = True


def test_api_runtime_bootstrap_no_longer_requires_boss_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CAREER_AGENT_BOSS_DATA_DIR", raising=False)

    args = _runtime_args_from_env()

    assert args.command == "chat"
    # The option is gone, not merely unset: nothing in the API path can ask for
    # a BOSS data directory any more.
    assert not hasattr(args, "boss_data_dir")


def _events(body: str) -> list[tuple[str, dict]]:
    parsed = []
    for block in body.split("\n\n"):
        if not block or block.startswith(":"):
            continue
        lines = block.splitlines()
        event_type = next(line[7:] for line in lines if line.startswith("event: "))
        data = next(line[6:] for line in lines if line.startswith("data: "))
        parsed.append((event_type, json.loads(data)))
    return parsed


def test_chat_endpoint_serializes_typed_events_as_sse_and_closes_runtime(api_keys, auth) -> None:
    runtime = Runtime()
    app = create_app(api_key_store_factory=lambda: api_keys, runtime_factory=lambda: runtime)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/stream",
            headers=auth,
            json={
                "conversation_id": "c1",
                "message": "你好",
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "no-cache, no-transform"
        assert response.headers["x-accel-buffering"] == "no"
        events = _events(response.text)

    assert runtime.closed is True
    assert runtime.calls == [("u1", "c1", "你好")]
    assert [event_type for event_type, _ in events] == [
        "turn_started",
        "content_delta",
        "content_delta",
        "turn_completed",
    ]
    assert "".join(
        payload["delta"]
        for event_type, payload in events
        if event_type == "content_delta"
    ) == "你好，世界"


def test_chat_endpoint_transports_bound_interaction_response(api_keys, auth) -> None:
    runtime = Runtime()
    app = create_app(api_key_store_factory=lambda: api_keys, runtime_factory=lambda: runtime)
    response_value = InteractionResponse(
        interaction_id="interaction_0123456789abcdef0123",
        scope="resume_analysis_confirmation",
        action="confirm",
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/stream",
            headers=auth,
            json={
                "conversation_id": "c1",
                "message": "确认并导入",
                "interaction_response": response_value.model_dump(),
            },
        )

    assert response.status_code == 200
    assert runtime.interaction_response == response_value


def test_chat_endpoint_transports_the_idempotency_key_outside_model_input(
    api_keys, auth
) -> None:
    runtime = Runtime()
    app = create_app(api_key_store_factory=lambda: api_keys, runtime_factory=lambda: runtime)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/stream",
            headers={**auth, "Idempotency-Key": "request-123"},
            json={"conversation_id": "c1", "message": "记录投递"},
        )

    assert response.status_code == 200
    assert runtime.request_id == "request-123"


def test_sse_does_not_expose_raw_runtime_exception(api_keys, auth) -> None:
    runtime = Runtime(fail=True)
    app = create_app(api_key_store_factory=lambda: api_keys, runtime_factory=lambda: runtime)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/stream",
            headers=auth,
            json={
                "conversation_id": "c1",
                "message": "触发失败",
            },
        )

    events = _events(response.text)
    assert events[-1][0] == "turn_failed"
    assert events[-1][1]["message"] == "安全错误消息。"
    assert "PRIVATE INTERNAL ERROR" not in response.text


def test_sse_sends_heartbeat_while_runtime_is_quiet() -> None:
    async def collect() -> str:
        request = ChatStreamRequest(conversation_id="c1", message="慢任务")
        return "".join(
            [
                chunk
                async for chunk in _sse_stream(
                    Runtime(delay=0.03),
                    request,
                    user_id="u1",
                    heartbeat_seconds=0.005,
                )
            ]
        )

    body = asyncio.run(collect())

    assert ": keep-alive\n\n" in body
    assert "event: turn_completed" in body


def test_disconnecting_observer_releases_gate_only_after_turn_finishes() -> None:
    async def exercise() -> None:
        gate = ConversationRunGate()
        await gate.acquire("u1", "c1")
        request = ChatStreamRequest(conversation_id="c1", message="慢任务")

        async def release_gate() -> None:
            await gate.release("u1", "c1")

        stream = _sse_stream(
            Runtime(delay=0.05),
            request,
            user_id="u1",
            heartbeat_seconds=1,
            on_turn_finished=release_gate,
        )
        first = await anext(stream)
        assert "event: turn_started" in first
        await stream.aclose()

        with pytest.raises(ConversationBusyError):
            await gate.acquire("u1", "c1")
        await asyncio.sleep(0.08)
        await gate.acquire("u1", "c1")

    asyncio.run(exercise())


def test_a_response_closed_before_body_iteration_releases_its_gate() -> None:
    class NeverStartsBody(GateAwareStreamingResponse):
        async def stream_response(self, send) -> None:
            return

    async def exercise() -> None:
        gate = ConversationRunGate()
        await gate.acquire("u1", "c1")
        started = False

        async def body():
            nonlocal started
            started = True
            yield b"unused"

        async def release() -> None:
            await gate.release("u1", "c1")

        response = NeverStartsBody(
            body(),
            stream_started=lambda: started,
            release_unstarted=release,
        )
        await response(
            {"type": "http", "asgi": {"spec_version": "2.4"}},
            lambda: None,
            lambda message: None,
        )

        await gate.acquire("u1", "c1")

    asyncio.run(exercise())


def test_conversation_gate_rejects_only_the_same_active_conversation() -> None:
    async def exercise() -> None:
        gate = ConversationRunGate()
        await gate.acquire("u1", "c1")
        await gate.acquire("u1", "c2")
        with pytest.raises(ConversationBusyError):
            await gate.acquire("u1", "c1")
        await gate.release("u1", "c1")
        await gate.acquire("u1", "c1")

    asyncio.run(exercise())


def test_request_contract_rejects_unknown_fields(api_keys, auth) -> None:
    runtime = Runtime()
    app = create_app(api_key_store_factory=lambda: api_keys, runtime_factory=lambda: runtime)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/stream",
            headers=auth,
            json={
                "conversation_id": "c1",
                "message": "你好",
                "internal_job_id": "must-not-pass",
            },
        )

    assert response.status_code == 422
    assert runtime.calls == []


def test_configuration_failure_keeps_api_alive_and_reports_exact_missing_keys(api_keys, auth) -> None:
    def unavailable_runtime():
        raise AgentConfigurationError(
            "AGENT_CONFIGURATION_MISSING",
            (
                "RESUME_ANALYSIS_AGENT_BASE_URL, RESUME_ANALYSIS_AGENT_API_KEY, "
                "and RESUME_ANALYSIS_AGENT_MODEL are required."
            ),
        )

    app = create_app(api_key_store_factory=lambda: api_keys, runtime_factory=unavailable_runtime)
    with TestClient(app) as client:
        health = client.get("/health")
        ready = client.get("/ready")
        stream = client.post(
            "/v1/chat/stream",
            headers=auth,
            json={
                "conversation_id": "c1",
                "message": "你好",
            },
        )

    assert health.status_code == 200
    assert ready.status_code == 503
    assert stream.status_code == 503
    assert ready.json()["detail"]["code"] == "AGENT_CONFIGURATION_MISSING"
    assert "RESUME_ANALYSIS_AGENT_API_KEY" in ready.json()["detail"]["message"]


def test_ready_reports_runtime_is_available(api_keys) -> None:
    runtime = Runtime()
    app = create_app(api_key_store_factory=lambda: api_keys, runtime_factory=lambda: runtime)

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_browser_capture_saves_only_after_explicit_endpoint_call(tmp_path, api_keys, auth) -> None:
    runtime = Runtime()
    repository = SQLiteJobPostingRepository(tmp_path / "jobs.sqlite3")
    app = create_app(
        api_key_store_factory=lambda: api_keys,
        runtime_factory=lambda: runtime,
        capture_repository_factory=lambda: repository,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/browser-captures/jobs",
            headers={**auth, "X-Career-Agent-Capture": "v1"},
            json={
                "source_url": (
                    "https://www.zhipin.com/job_detail/boss-123.html"
                    "?securityId=must-not-be-stored#detail"
                ),
                "title": "AI 产品经理",
                "company_name": "示例科技",
                "description": "负责 AI 产品规划和交付。",
                "city": "上海",
                "salary": "25-35K",
                "experience": "3-5年",
                "education": "本科",
            },
        )

    assert response.status_code == 200
    saved = repository.get_job(
        user_id="u1",
        job_posting_id=response.json()["job_posting_id"],
    )
    assert saved is not None
    assert saved.posting.source_job_id == "boss-123"
    assert saved.posting.source_url == "https://www.zhipin.com/job_detail/boss-123.html"
    assert saved.snapshot.content == "负责 AI 产品规划和交付。"


def test_browser_capture_requires_a_capture_scoped_credential(
    tmp_path, api_keys, issue_key
) -> None:
    runtime = Runtime()
    repository = SQLiteJobPostingRepository(tmp_path / "jobs.sqlite3")
    app = create_app(
        api_key_store_factory=lambda: api_keys,
        runtime_factory=lambda: runtime,
        capture_repository_factory=lambda: repository,
    )
    payload = {
        "source_url": "https://example.com/job_detail/123.html",
        "title": "AI 产品经理",
        "company_name": "示例科技",
        "description": "负责 AI 产品规划和交付。",
    }
    capture_only = issue_key("u1", CAPTURE_WRITE)
    read_only = issue_key("u1", WORKSPACE_READ)

    with TestClient(app) as client:
        # The static header was never access control — anyone could send it —
        # and it used to be the only gate on this route. It is now a payload
        # version marker, so sending it without a credential proves nothing.
        no_credential = client.post(
            "/v1/browser-captures/jobs",
            headers={"X-Career-Agent-Capture": "v1"},
            json=payload,
        )
        # A real key for the right user, but the wrong capability. This is the
        # reason the extension gets its own scope: whatever it holds cannot be
        # turned into a read of the workspace, and vice versa.
        wrong_scope = client.post(
            "/v1/browser-captures/jobs",
            headers={**read_only, "X-Career-Agent-Capture": "v1"},
            json=payload,
        )
        missing_header = client.post(
            "/v1/browser-captures/jobs", headers=capture_only, json=payload
        )
        cross_site = client.post(
            "/v1/browser-captures/jobs",
            headers={**capture_only, "X-Career-Agent-Capture": "v1"},
            json=payload,
        )
        workspace_via_capture_key = client.get("/v1/jobs", headers=capture_only)

    assert no_credential.status_code == 401
    assert wrong_scope.status_code == 403
    # Unsupported version, not forbidden: the caller is allowed here and simply
    # sent a shape this build does not read.
    assert missing_header.status_code == 422
    assert cross_site.status_code == 422
    assert workspace_via_capture_key.status_code == 403
    assert repository.list_jobs(user_id="u1", include_dismissed=True) == ()


def _brief_app(tmp_path, api_keys, *, applications=(), interviews=(), events=()):
    from career_agent.api.app import create_app
    from career_agent.services.action_center import ActionCenterService
    from career_agent.storage.action_center import SQLiteActionItemStore

    class Applications:
        def list_applications(self, *, user_id, **kwargs):
            # Scoping lives in the real services; the fake has to honour it or
            # the isolation test would pass for the wrong reason.
            return applications if user_id == "u1" else ()

    class Interviews:
        def list_interviews(self, **kwargs):
            return interviews

    class Emails:
        def list_events(self, **kwargs):
            return events

    service = ActionCenterService(
        SQLiteActionItemStore(tmp_path / "actions.sqlite3"),
        Applications(),
        Emails(),
        Interviews(),
    )
    return create_app(
        api_key_store_factory=lambda: api_keys,
        runtime_factory=Runtime,
        capture_repository_factory=lambda: None,
        action_center_factory=lambda: service,
    )


def _stale_application(application_id: str, *, days: int):
    from types import SimpleNamespace

    return SimpleNamespace(
        application=SimpleNamespace(
            id=application_id,
            status="submitted",
            updated_at=datetime.now(timezone.utc) - timedelta(days=days),
        ),
        job=SimpleNamespace(
            posting=SimpleNamespace(company_name="Acme", title="AI Engineer")
        ),
    )


def test_daily_brief_groups_actions_for_the_dashboard(tmp_path, auth, api_keys) -> None:
    app = _brief_app(tmp_path, api_keys, applications=(_stale_application("app-1", days=9),))

    with TestClient(app) as client:
        response = client.get("/v1/daily-brief", headers=auth)

    payload = response.json()
    assert response.status_code == 200
    assert payload["timezone"] == "Asia/Shanghai"
    # A follow-up becomes due the moment it is generated, so it lands in
    # today's bucket rather than in the overdue one.
    assert [item["title"] for item in payload["due_today"]] == ["跟进投递：Acme"]
    assert payload["overdue"] == []


def test_the_brief_never_returns_derivation_plumbing(tmp_path, auth, api_keys) -> None:
    """The client must not be able to key its own state on internal fields."""
    app = _brief_app(tmp_path, api_keys, applications=(_stale_application("app-1", days=9),))

    with TestClient(app) as client:
        payload = client.get("/v1/daily-brief", headers=auth).json()

    item = payload["due_today"][0]
    assert set(item) == {
        "id",
        "action_type",
        "source_type",
        "application_id",
        "title",
        "summary",
        "due_at",
        "status",
        "snoozed_until",
    }
    assert "stable_key" not in item
    assert "user_id" not in item


def test_the_brief_is_scoped_to_the_asserted_user(tmp_path, auth, api_keys, issue_key) -> None:
    app = _brief_app(tmp_path, api_keys, applications=(_stale_application("app-1", days=9),))

    with TestClient(app) as client:
        mine = client.get("/v1/daily-brief", headers=auth).json()
        theirs = client.get("/v1/daily-brief", headers=issue_key("u2")).json()

    assert mine["due_today"]
    assert theirs["due_today"] == []


def test_the_brief_requires_a_credential_not_a_named_user(tmp_path, api_keys) -> None:
    """What "requires a user" now means: hold a key, not type a name.

    This used to assert 422 for a missing or empty ``user_id`` query parameter —
    a validation check on a field any caller could fill with anyone's id. The
    field is gone, so the property worth pinning is that an unauthenticated
    request is refused, and that a malformed credential is not accidentally
    accepted by a lenient header parse.
    """
    app = _brief_app(tmp_path, api_keys)

    with TestClient(app) as client:
        assert client.get("/v1/daily-brief").status_code == 401
        for header in ("", "Bearer", "Bearer ", "Basic abc", "not-a-scheme x"):
            assert client.get(
                "/v1/daily-brief", headers={"Authorization": header}
            ).status_code == 401
        assert client.get(
            "/v1/daily-brief", headers={"Authorization": "Bearer wrong-secret"}
        ).status_code == 401


def test_reading_the_brief_needs_no_model_configuration(
    tmp_path, monkeypatch,
    api_keys, auth,
) -> None:
    """A dashboard must survive a machine where the worker keys are missing."""
    for name in ("MAIN_AGENT_BASE_URL", "MAIN_AGENT_API_KEY", "MAIN_AGENT_MODEL"):
        monkeypatch.delenv(name, raising=False)
    app = _brief_app(tmp_path, api_keys, applications=(_stale_application("app-1", days=9),))

    with TestClient(app) as client:
        assert client.get("/v1/daily-brief", headers=auth).status_code == 200


def test_a_rejected_overlapping_turn_is_counted_not_only_refused(
    tmp_path, api_keys, auth
) -> None:
    """Whether the process-local gate is enough is a question about contention.

    A lease and an optimistic version number suit opposite contention levels,
    and this deployment has never measured which one it has. The gate already
    knows — it returns 409 — so the fact is recorded rather than discarded, and
    the replacement can be chosen from data instead of from a guess.
    """
    class CountingRuntime(Runtime):
        def __init__(self) -> None:
            super().__init__()
            self.rejected = []

        def record_rejected_turn(self, *, user_id, conversation_id):
            self.rejected.append((user_id, conversation_id))

    runtime = CountingRuntime()
    app = create_app(
        api_key_store_factory=lambda: api_keys, runtime_factory=lambda: runtime
    )

    with TestClient(app) as client:
        app.state.run_gate._active.add(("u1", "c1"))
        response = client.post(
            "/v1/chat/stream",
            headers=auth,
            json={"conversation_id": "c1", "message": "再来一次"},
        )

    assert response.status_code == 409
    # The user comes from the credential, so a rejection is attributed to whoever
    # actually holds the key rather than to a name the request supplied.
    assert runtime.rejected == [("u1", "c1")]
