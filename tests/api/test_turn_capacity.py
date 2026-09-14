from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from career_agent.api.app import (
    ConversationBusyError,
    ConversationRunGate,
    DEFAULT_MAX_CONCURRENT_TURNS,
    MAX_CONCURRENT_TURNS_ENV,
    TurnCapacityError,
    create_app,
    max_concurrent_turns_from_env,
)


def test_gate_caps_total_active_turns_and_frees_a_slot_on_release() -> None:
    async def exercise() -> None:
        gate = ConversationRunGate(max_active=2)
        await gate.acquire("u1", "c1")
        await gate.acquire("u1", "c2")
        with pytest.raises(TurnCapacityError) as refused:
            await gate.acquire("u1", "c3")
        assert refused.value.max_active == 2
        # The per-conversation refusal still wins over the global one: the
        # client gets the more specific reason.
        with pytest.raises(ConversationBusyError):
            await gate.acquire("u1", "c1")
        await gate.release("u1", "c1")
        await gate.acquire("u1", "c3")
        assert gate.active_count == 2

    asyncio.run(exercise())


def test_gate_without_a_cap_keeps_the_old_behaviour() -> None:
    async def exercise() -> None:
        gate = ConversationRunGate(max_active=None)
        for index in range(20):
            await gate.acquire("u1", f"c{index}")

    asyncio.run(exercise())


def test_gate_rejects_a_cap_below_one() -> None:
    with pytest.raises(ValueError):
        ConversationRunGate(max_active=0)


def test_cap_defaults_and_reads_the_environment() -> None:
    assert max_concurrent_turns_from_env({}) == DEFAULT_MAX_CONCURRENT_TURNS
    assert max_concurrent_turns_from_env({MAX_CONCURRENT_TURNS_ENV: " "}) == DEFAULT_MAX_CONCURRENT_TURNS
    assert max_concurrent_turns_from_env({MAX_CONCURRENT_TURNS_ENV: "4"}) == 4
    for bad in ("0", "-1", "two", "1.5"):
        with pytest.raises(ValueError, match=MAX_CONCURRENT_TURNS_ENV):
            max_concurrent_turns_from_env({MAX_CONCURRENT_TURNS_ENV: bad})


def test_chat_endpoint_returns_503_with_retry_after_when_full(api_keys, auth) -> None:
    class IdleRuntime:
        def __init__(self) -> None:
            self.rejected = []
            self.calls = []

        def run_turn(self, **kwargs):
            self.calls.append(kwargs)
            raise AssertionError("a refused turn must never reach the runtime")

        def record_rejected_turn(self, *, user_id, conversation_id):
            self.rejected.append((user_id, conversation_id))

        def close(self) -> None:
            pass

    runtime = IdleRuntime()
    app = create_app(
        api_key_store_factory=lambda: api_keys,
        runtime_factory=lambda: runtime,
        max_concurrent_turns=2,
    )

    with TestClient(app) as client:
        assert app.state.run_gate.max_active == 2
        app.state.run_gate._active.update({("u1", "c1"), ("u1", "c2")})
        response = client.post(
            "/v1/chat/stream",
            headers=auth,
            json={"conversation_id": "c3", "message": "再来一个"},
        )

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "10"
    detail = response.json()["detail"]
    assert detail["code"] == "TURN_CAPACITY_EXHAUSTED"
    assert detail["max_concurrent_turns"] == 2
    assert "请稍后再试" in detail["message"]
    assert runtime.calls == []
    # Distinct from the per-conversation 409, which is what record_rejected_turn
    # measures; a full server is not the same signal as a doubled conversation.
    assert runtime.rejected == []


def test_app_reads_the_cap_from_the_environment_at_startup(api_keys, monkeypatch) -> None:
    class Runtime:
        def close(self) -> None:
            pass

    monkeypatch.setenv(MAX_CONCURRENT_TURNS_ENV, "5")
    app = create_app(api_key_store_factory=lambda: api_keys, runtime_factory=Runtime)
    with TestClient(app):
        assert app.state.run_gate.max_active == 5

    monkeypatch.delenv(MAX_CONCURRENT_TURNS_ENV)
    app = create_app(api_key_store_factory=lambda: api_keys, runtime_factory=Runtime)
    with TestClient(app):
        assert app.state.run_gate.max_active == DEFAULT_MAX_CONCURRENT_TURNS


def test_an_invalid_cap_fails_startup_instead_of_silently_running_uncapped(api_keys, monkeypatch) -> None:
    monkeypatch.setenv(MAX_CONCURRENT_TURNS_ENV, "0")
    app = create_app(api_key_store_factory=lambda: api_keys, runtime_factory=lambda: None)
    with pytest.raises(ValueError, match=MAX_CONCURRENT_TURNS_ENV):
        with TestClient(app):
            pass


def test_create_app_rejects_a_cap_below_one(api_keys) -> None:
    with pytest.raises(ValueError):
        create_app(api_key_store_factory=lambda: api_keys, max_concurrent_turns=0)
