"""A reader who comes back to a running turn sees it unfold, not a blank page.

The turn keeps running after its page leaves; its events so far are kept so a
new reader replays them and then follows the rest, the way a terminal agent
shows a session that was running while you looked elsewhere.
"""

from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from career_agent.api.app import ChatStreamRequest, _sse_stream, create_app
from career_agent.api.live_turns import LiveTurnRegistry
from career_agent.harness.streaming import ContentDeltaEvent, TurnCompletedEvent
from tests.api.test_chat_sse import Runtime


def test_a_follower_who_arrives_late_gets_every_event_then_the_end() -> None:
    async def exercise() -> tuple[list[str], list[str], bool]:
        registry = LiveTurnRegistry()
        live = registry.start("u1", "c1", "帮我研究公司")

        async def finished() -> None:
            await registry.end("u1", "c1", live)

        stream = _sse_stream(
            Runtime(delay=0.05),
            ChatStreamRequest(conversation_id="c1", message="帮我研究公司"),
            user_id="u1",
            heartbeat_seconds=1,
            on_turn_finished=finished,
            live=live,
        )
        first = await anext(stream)
        # The page leaves after the first event; the turn runs on.
        await stream.aclose()
        assert "turn_started" in first

        found = registry.get("u1", "c1")
        assert found is live and found.user_message == "帮我研究公司"
        followed = [
            event.type
            async for event in found.follow(heartbeat_seconds=1)
            if event is not None
        ]
        deltas = [
            event.delta
            for event in found._events
            if isinstance(event, ContentDeltaEvent)
        ]
        return followed, deltas, registry.get("u1", "c1") is None

    followed, deltas, gone = asyncio.run(exercise())

    assert followed[0] == "turn_started"
    assert followed[-1] == "turn_completed"
    assert deltas == ["你好", "，世界"]
    # Settled turns are not followable: the transcript has the reply now.
    assert gone


def test_a_quiet_turn_yields_heartbeats_while_the_follower_waits() -> None:
    async def exercise() -> list[object]:
        registry = LiveTurnRegistry()
        live = registry.start("u1", "c1", "慢任务")
        seen: list[object] = []

        async def follow() -> None:
            async for event in live.follow(heartbeat_seconds=0.01):
                seen.append(event)

        follower = asyncio.create_task(follow())
        await asyncio.sleep(0.035)
        await live.append(TurnCompletedEvent(turn_id="t1"))
        await registry.end("u1", "c1", live)
        await follower
        return seen

    seen = asyncio.run(exercise())

    assert None in seen
    assert seen[-1] == TurnCompletedEvent(turn_id="t1")


def test_ending_an_older_turn_does_not_hide_a_newer_one() -> None:
    async def exercise() -> bool:
        registry = LiveTurnRegistry()
        older = registry.start("u1", "c1", "第一轮")
        newer = registry.start("u1", "c1", "第二轮")
        await registry.end("u1", "c1", older)
        return registry.get("u1", "c1") is newer

    assert asyncio.run(exercise())


def test_following_a_conversation_with_no_running_turn_is_not_found(api_keys, auth) -> None:
    app = create_app(api_key_store_factory=lambda: api_keys, runtime_factory=Runtime)

    with TestClient(app) as client:
        response = client.get("/v1/conversations/c1/live", headers=auth)

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "NO_LIVE_TURN"
