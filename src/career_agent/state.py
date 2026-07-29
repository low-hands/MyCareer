"""Conversation state contracts independent of an orchestration framework."""

from __future__ import annotations

import asyncio
from typing import Protocol

from langchain_core.messages import AnyMessage


class ConversationStore(Protocol):
    async def load(self, thread_id: str) -> list[AnyMessage]: ...

    async def save(self, thread_id: str, messages: list[AnyMessage]) -> None: ...


class InMemoryConversationStore:
    """Single-process conversation store used by the initial runtime."""

    def __init__(self) -> None:
        self._messages: dict[str, list[AnyMessage]] = {}
        self._lock = asyncio.Lock()

    async def load(self, thread_id: str) -> list[AnyMessage]:
        async with self._lock:
            return list(self._messages.get(thread_id, []))

    async def save(self, thread_id: str, messages: list[AnyMessage]) -> None:
        async with self._lock:
            self._messages[thread_id] = list(messages)
