from __future__ import annotations

import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from career_agent.api.app import (
    ChatStreamRequest,
    ConversationBusyError,
    ConversationRunGate,
    _runtime_args_from_env,
    _sse_stream,
    create_app,
)
from career_agent.agent.openai_compatible_client import AgentConfigurationError
from career_agent.harness.streaming import (
    ContentDeltaEvent,
    TurnCompletedEvent,
    TurnFailedEvent,
    TurnStartedEvent,
)
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
        event_sink=None,
    ):
        self.calls.append((user_id, conversation_id, user_message))
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


def test_chat_endpoint_serializes_typed_events_as_sse_and_closes_runtime() -> None:
    runtime = Runtime()
    app = create_app(runtime_factory=lambda: runtime)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/stream",
            json={
                "user_id": "u1",
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


def test_sse_does_not_expose_raw_runtime_exception() -> None:
    runtime = Runtime(fail=True)
    app = create_app(runtime_factory=lambda: runtime)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/stream",
            json={
                "user_id": "u1",
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
        request = ChatStreamRequest(
            user_id="u1",
            conversation_id="c1",
            message="慢任务",
        )
        return "".join(
            [
                chunk
                async for chunk in _sse_stream(
                    Runtime(delay=0.03),
                    request,
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
        request = ChatStreamRequest(
            user_id="u1",
            conversation_id="c1",
            message="慢任务",
        )

        async def release_gate() -> None:
            await gate.release("u1", "c1")

        stream = _sse_stream(
            Runtime(delay=0.05),
            request,
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


def test_request_contract_rejects_unknown_fields() -> None:
    runtime = Runtime()
    app = create_app(runtime_factory=lambda: runtime)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/stream",
            json={
                "user_id": "u1",
                "conversation_id": "c1",
                "message": "你好",
                "internal_job_id": "must-not-pass",
            },
        )

    assert response.status_code == 422
    assert runtime.calls == []


def test_configuration_failure_keeps_api_alive_and_reports_exact_missing_keys() -> None:
    def unavailable_runtime():
        raise AgentConfigurationError(
            "AGENT_CONFIGURATION_MISSING",
            (
                "RESUME_ANALYSIS_AGENT_BASE_URL, RESUME_ANALYSIS_AGENT_API_KEY, "
                "and RESUME_ANALYSIS_AGENT_MODEL are required."
            ),
        )

    app = create_app(runtime_factory=unavailable_runtime)
    with TestClient(app) as client:
        health = client.get("/health")
        ready = client.get("/ready")
        stream = client.post(
            "/v1/chat/stream",
            json={
                "user_id": "u1",
                "conversation_id": "c1",
                "message": "你好",
            },
        )

    assert health.status_code == 200
    assert ready.status_code == 503
    assert stream.status_code == 503
    assert ready.json()["detail"]["code"] == "AGENT_CONFIGURATION_MISSING"
    assert "RESUME_ANALYSIS_AGENT_API_KEY" in ready.json()["detail"]["message"]


def test_ready_reports_runtime_is_available() -> None:
    runtime = Runtime()
    app = create_app(runtime_factory=lambda: runtime)

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_browser_capture_saves_only_after_explicit_endpoint_call(tmp_path) -> None:
    runtime = Runtime()
    repository = SQLiteJobPostingRepository(tmp_path / "jobs.sqlite3")
    app = create_app(
        runtime_factory=lambda: runtime,
        capture_repository_factory=lambda: repository,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/browser-captures/jobs",
            headers={"X-Career-Agent-Capture": "v1"},
            json={
                "user_id": "u1",
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


def test_browser_capture_rejects_cross_site_urls_and_missing_capture_header(tmp_path) -> None:
    runtime = Runtime()
    repository = SQLiteJobPostingRepository(tmp_path / "jobs.sqlite3")
    app = create_app(
        runtime_factory=lambda: runtime,
        capture_repository_factory=lambda: repository,
    )
    payload = {
        "user_id": "u1",
        "source_url": "https://example.com/job_detail/123.html",
        "title": "AI 产品经理",
        "company_name": "示例科技",
        "description": "负责 AI 产品规划和交付。",
    }

    with TestClient(app) as client:
        missing_header = client.post("/v1/browser-captures/jobs", json=payload)
        cross_site = client.post(
            "/v1/browser-captures/jobs",
            headers={"X-Career-Agent-Capture": "v1"},
            json=payload,
        )

    assert missing_header.status_code == 403
    assert cross_site.status_code == 422
    assert repository.list_jobs(user_id="u1") == ()
