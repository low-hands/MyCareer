from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from career_agent.harness.streaming import (
    ClientActionEvent,
    ContentDeltaEvent,
    InteractionOption,
    InteractionRequiredEvent,
    TurnCompletedEvent,
    astream_turn_events,
    interaction_id,
    iter_content_deltas,
)


def test_fake_stream_chunks_preserve_the_completed_message() -> None:
    message = "第一段内容比较长，需要分块交付。\n\n第二段保留 Markdown 边界。"

    chunks = tuple(iter_content_deltas(message, target_chars=10))

    assert "".join(chunks) == message
    assert len(chunks) > 1


def test_interaction_contract_rejects_ambiguous_option_identity() -> None:
    with pytest.raises(ValidationError):
        InteractionOption(label="错误选项", selection_index=1, value="one")


def test_interaction_id_is_stable_and_opaque() -> None:
    first = interaction_id("conversation-1", "internal-run-1", "selection_required")

    assert first == interaction_id(
        "conversation-1", "internal-run-1", "selection_required"
    )
    assert "internal-run-1" not in first


def test_async_adapter_bridges_sync_callback_and_can_pace_content() -> None:
    class Runtime:
        def run_turn(
            self,
            *,
            user_id,
            conversation_id,
            user_message,
            event_sink=None,
        ):
            assert (user_id, conversation_id, user_message) == ("u1", "c1", "hello")
            assert event_sink is not None
            event_sink(ContentDeltaEvent(delta="hello"))
            event_sink(TurnCompletedEvent(turn_id="turn-1"))

    async def collect():
        return [
            event
            async for event in astream_turn_events(
                Runtime(),
                user_id="u1",
                conversation_id="c1",
                user_message="hello",
            )
        ]

    events = asyncio.run(collect())

    assert [event.type for event in events] == ["content_delta", "turn_completed"]


def test_selection_interaction_requires_options() -> None:
    with pytest.raises(ValidationError):
        InteractionRequiredEvent(
            interaction_id=interaction_id("c1", "selection"),
            kind="single_selection",
            prompt="请选择。",
        )


def test_client_action_requires_https_url() -> None:
    with pytest.raises(ValidationError):
        ClientActionEvent(
            action="open_url",
            url="javascript:alert(1)",
            label="打开",
        )
