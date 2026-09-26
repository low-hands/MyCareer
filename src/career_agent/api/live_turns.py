"""The events of turns still running in this process, for a page to follow.

A chat stream is one HTTP response; a reader who switches conversations drops
it while the turn runs on in its detached producer. To show that turn again
the way it is unfolding, not just its stored result, every event the turn has
emitted so far is kept here until it settles, and a new reader replays them
and then follows the rest.

It lives in memory because the turn does: the API runs as a single worker, and
a turn does not survive the process (its receipt is settled as failed on the
next start), so there is nothing to follow after a restart.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from career_agent.harness.streaming import PublicStreamEvent


class LiveTurn:
    """One running turn: the user's message and every event emitted so far."""

    def __init__(self, user_message: str) -> None:
        self.user_message = user_message
        self._events: list[PublicStreamEvent] = []
        self._finished = False
        self._changed = asyncio.Condition()

    async def append(self, event: PublicStreamEvent) -> None:
        async with self._changed:
            self._events.append(event)
            self._changed.notify_all()

    async def finish(self) -> None:
        async with self._changed:
            self._finished = True
            self._changed.notify_all()

    async def follow(
        self, *, heartbeat_seconds: float
    ) -> AsyncIterator[PublicStreamEvent | None]:
        """Every event from the first, then each new one until the turn settles.

        Yields ``None`` when ``heartbeat_seconds`` pass without an event, so
        the caller can keep its connection alive.
        """

        sent = 0
        while True:
            async with self._changed:
                try:
                    await asyncio.wait_for(
                        self._changed.wait_for(
                            lambda: sent < len(self._events) or self._finished
                        ),
                        timeout=heartbeat_seconds,
                    )
                except TimeoutError:
                    batch = None
                else:
                    batch = self._events[sent:]
                    sent += len(batch)
                done = self._finished and sent >= len(self._events)
            if batch is None:
                yield None
                continue
            for event in batch:
                yield event
            if done:
                return


class LiveTurnRegistry:
    def __init__(self) -> None:
        self._turns: dict[tuple[str, str], LiveTurn] = {}

    def start(self, user_id: str, conversation_id: str, user_message: str) -> LiveTurn:
        turn = LiveTurn(user_message)
        self._turns[(user_id, conversation_id)] = turn
        return turn

    def get(self, user_id: str, conversation_id: str) -> LiveTurn | None:
        return self._turns.get((user_id, conversation_id))

    async def end(self, user_id: str, conversation_id: str, turn: LiveTurn) -> None:
        """Settle ``turn``: followers get the rest and stop, new readers find none."""

        if self._turns.get((user_id, conversation_id)) is turn:
            del self._turns[(user_id, conversation_id)]
        await turn.finish()
