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
    _sse_stream,
    create_app,
)
from career_agent.harness.streaming import (
    ContentDeltaEvent,
    TurnCompletedEvent,
    TurnFailedEvent,
    TurnStartedEvent,
)


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
