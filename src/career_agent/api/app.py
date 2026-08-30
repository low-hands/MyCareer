from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.cli import build_main_agent_runtime, build_parser
from career_agent.harness.streaming import PublicStreamEvent, astream_turn_events


class ChatStreamRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    user_id: str = Field(min_length=1, max_length=200)
    conversation_id: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=100_000)


class ConversationBusyError(Exception):
    pass


class ConversationRunGate:
    """Reject overlapping turns for one conversation without serializing users."""

    def __init__(self) -> None:
        self._guard = asyncio.Lock()
        self._active: set[tuple[str, str]] = set()

    async def acquire(self, user_id: str, conversation_id: str) -> None:
        key = (user_id, conversation_id)
        async with self._guard:
            if key in self._active:
                raise ConversationBusyError
            self._active.add(key)

    async def release(self, user_id: str, conversation_id: str) -> None:
        async with self._guard:
            self._active.discard((user_id, conversation_id))


def _runtime_args_from_env() -> argparse.Namespace:
    boss_data_dir = os.environ.get("CAREER_AGENT_BOSS_DATA_DIR", "").strip()
    if not boss_data_dir:
        raise RuntimeError(
            "CAREER_AGENT_BOSS_DATA_DIR is required to start the API runtime"
        )
    return build_parser().parse_args(
        [
            "chat",
            "--user-id",
            "api-runtime",
            "--session-id",
            "api-runtime",
            "--message",
            "runtime-bootstrap",
            "--boss-data-dir",
            boss_data_dir,
        ]
    )


def build_api_runtime() -> MainAgentRuntime:
    return build_main_agent_runtime(_runtime_args_from_env())


def _encode_sse(event: PublicStreamEvent) -> str:
    return (
        f"event: {event.type}\n"
        f"data: {event.model_dump_json(exclude_none=True)}\n\n"
    )


async def _sse_stream(
    runtime: MainAgentRuntime,
    request: ChatStreamRequest,
    *,
    heartbeat_seconds: float,
    on_turn_finished: Callable[[], Awaitable[None]] | None = None,
) -> AsyncIterator[str]:
    queue: asyncio.Queue[PublicStreamEvent | object] = asyncio.Queue()
    sentinel = object()

    async def produce() -> None:
        try:
            async for event in astream_turn_events(
                runtime,
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                user_message=request.message,
            ):
                await queue.put(event)
        except Exception:
            # MainAgentRuntime emits a safe turn_failed event before raising.
            # The HTTP adapter must not serialize the raw exception after the
            # response headers have already been sent.
            pass
        finally:
            try:
                if on_turn_finished is not None:
                    await on_turn_finished()
            finally:
                await queue.put(sentinel)

    producer = asyncio.create_task(produce())
    try:
        while True:
            try:
                item = await asyncio.wait_for(
                    queue.get(),
                    timeout=heartbeat_seconds,
                )
            except TimeoutError:
                yield ": keep-alive\n\n"
                continue
            if item is sentinel:
                break
            yield _encode_sse(item)
    finally:
        if producer.done():
            producer.result()
        else:
            # Closing the HTTP response must not cancel a business turn that
            # may already be committing. Keep draining its events in this
            # detached producer; it releases the conversation gate only after
            # the synchronous runtime has genuinely returned.
            producer.add_done_callback(
                lambda completed: None
                if completed.cancelled()
                else completed.exception()
            )


def create_app(
    *,
    runtime_factory: Callable[[], MainAgentRuntime] | None = None,
    heartbeat_seconds: float = 15.0,
) -> FastAPI:
    if heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be positive")
    factory = runtime_factory or build_api_runtime

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = factory()
        app.state.runtime = runtime
        app.state.run_gate = ConversationRunGate()
        try:
            yield
        finally:
            close = getattr(runtime, "close", None)
            if close is not None:
                close()

    application = FastAPI(
        title="Career Agent API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.post("/v1/chat/stream")
    async def chat_stream(request: ChatStreamRequest) -> StreamingResponse:
        gate: ConversationRunGate = application.state.run_gate
        try:
            await gate.acquire(request.user_id, request.conversation_id)
        except ConversationBusyError as error:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "CONVERSATION_TURN_IN_PROGRESS",
                    "message": "This conversation already has a running turn.",
                },
            ) from error

        runtime: MainAgentRuntime = application.state.runtime

        async def release_gate() -> None:
            await gate.release(request.user_id, request.conversation_id)

        async def generate() -> AsyncIterator[str]:
            async for chunk in _sse_stream(
                runtime,
                request,
                heartbeat_seconds=heartbeat_seconds,
                on_turn_finished=release_gate,
            ):
                yield chunk

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return application


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run(
        "career_agent.api.app:app",
        host="127.0.0.1",
        port=int(os.environ.get("CAREER_AGENT_API_PORT", "8000")),
        reload=False,
    )
