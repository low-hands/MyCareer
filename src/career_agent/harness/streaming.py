from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from hashlib import sha256
from typing import Annotated, Literal, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StreamContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class TurnStartedEvent(StreamContract):
    type: Literal["turn_started"] = "turn_started"
    turn_id: str = Field(min_length=1)


class ProgressEvent(StreamContract):
    type: Literal["progress"] = "progress"
    stage: Literal[
        "loading_context",
        "deciding",
        "running_capability",
        "presenting",
        "saving",
    ]
    message: str = Field(min_length=1, max_length=240)


class CapabilityStartedEvent(StreamContract):
    type: Literal["capability_started"] = "capability_started"
    capability: Literal[
        "job_search",
        "job_research",
        "resume",
        "application_tracking",
        "interview",
        "calendar",
        "action_center",
        "career_task",
    ]
    message: str = Field(min_length=1, max_length=240)


class CapabilityCompletedEvent(StreamContract):
    type: Literal["capability_completed"] = "capability_completed"
    capability: Literal[
        "job_search",
        "job_research",
        "resume",
        "application_tracking",
        "interview",
        "calendar",
        "action_center",
        "career_task",
    ]
    state: str = Field(pattern=r"^[a-z0-9_]+$", max_length=80)
    message: str = Field(min_length=1, max_length=240)


class InteractionOption(StreamContract):
    label: str = Field(min_length=1, max_length=240)
    description: str | None = Field(default=None, max_length=500)
    selection_index: int | None = Field(default=None, ge=1)
    value: str | None = Field(default=None, min_length=1, max_length=80)

    @model_validator(mode="after")
    def _one_public_value(self) -> "InteractionOption":
        if (self.selection_index is None) == (self.value is None):
            raise ValueError("an interaction option needs one public value")
        return self


class InteractionRequiredEvent(StreamContract):
    type: Literal["interaction_required"] = "interaction_required"
    interaction_id: str = Field(pattern=r"^interaction_[a-f0-9]{20}$")
    kind: Literal[
        "single_selection",
        "multiple_selection",
        "confirmation",
        "free_text",
        "approval",
        "file_upload",
    ]
    prompt: str = Field(min_length=1, max_length=5000)
    options: tuple[InteractionOption, ...] = Field(default=(), max_length=50)
    allow_free_text: bool = False

    @model_validator(mode="after")
    def _validate_options(self) -> "InteractionRequiredEvent":
        requires_options = self.kind in {
            "single_selection",
            "multiple_selection",
            "confirmation",
            "approval",
        }
        if requires_options and not self.options:
            raise ValueError(f"{self.kind} requires options")
        if self.kind in {"free_text", "file_upload"} and self.options:
            raise ValueError(f"{self.kind} cannot carry options")
        return self


class ContentDeltaEvent(StreamContract):
    # Whitespace is content here: stripping a blank Markdown line would make
    # the streamed text differ from the committed assistant message.
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=False,
    )

    type: Literal["content_delta"] = "content_delta"
    delta: str = Field(min_length=1)


class ArtifactReadyEvent(StreamContract):
    type: Literal["artifact_ready"] = "artifact_ready"
    artifact_id: str = Field(min_length=1)
    filename: str = Field(min_length=1, max_length=180)
    media_type: str = Field(min_length=1, max_length=100)
    byte_size: int = Field(ge=1)


class TurnSuspendedEvent(StreamContract):
    type: Literal["turn_suspended"] = "turn_suspended"
    turn_id: str = Field(min_length=1)
    reason: Literal["interaction_required"] = "interaction_required"
    interaction_id: str = Field(pattern=r"^interaction_[a-f0-9]{20}$")


class TurnCompletedEvent(StreamContract):
    type: Literal["turn_completed"] = "turn_completed"
    turn_id: str = Field(min_length=1)


class TurnFailedEvent(StreamContract):
    type: Literal["turn_failed"] = "turn_failed"
    turn_id: str = Field(min_length=1)
    code: str = Field(pattern=r"^[A-Z0-9_]+$", max_length=80)
    message: str = Field(min_length=1, max_length=500)


PublicStreamEvent: TypeAlias = Annotated[
    TurnStartedEvent
    | ProgressEvent
    | CapabilityStartedEvent
    | CapabilityCompletedEvent
    | InteractionRequiredEvent
    | ContentDeltaEvent
    | ArtifactReadyEvent
    | TurnSuspendedEvent
    | TurnCompletedEvent
    | TurnFailedEvent,
    Field(discriminator="type"),
]
StreamEventSink: TypeAlias = Callable[[PublicStreamEvent], None]


class StreamableTurnRuntime(Protocol):
    def run_turn(
        self,
        *,
        user_id: str,
        conversation_id: str,
        user_message: str,
        event_sink: StreamEventSink | None = None,
    ) -> object: ...


def interaction_id(*durable_parts: object) -> str:
    """Derive an opaque stable handle from state that was already committed."""

    canonical = "\x1f".join(str(part) for part in durable_parts)
    return f"interaction_{sha256(canonical.encode('utf-8')).hexdigest()[:20]}"


def iter_content_deltas(text: str, *, target_chars: int = 48) -> Iterator[str]:
    """Split completed prose on semantic boundaries for deterministic fake streaming."""

    if target_chars < 1:
        raise ValueError("target_chars must be at least one")
    buffer = ""
    for line in text.splitlines(keepends=True):
        buffer += line
        if len(buffer) >= target_chars or not line.strip():
            if buffer:
                yield buffer
                buffer = ""
    if buffer:
        yield buffer


async def astream_turn_events(
    runtime: StreamableTurnRuntime,
    *,
    user_id: str,
    conversation_id: str,
    user_message: str,
    content_delay_seconds: float = 0.0,
) -> AsyncIterator[PublicStreamEvent]:
    """Bridge the synchronous runtime to an async SSE/WebSocket consumer.

    The core never sleeps. A presentation adapter may opt into a small delay
    for fake-streamed completed prose without slowing tests or non-stream calls.
    """

    if content_delay_seconds < 0:
        raise ValueError("content_delay_seconds cannot be negative")
    queue: asyncio.Queue[PublicStreamEvent | object] = asyncio.Queue()
    sentinel = object()
    loop = asyncio.get_running_loop()

    def post(item: PublicStreamEvent | object) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, item)
        except RuntimeError:
            # The client event loop may already be closed. The runtime treats
            # its sink as best-effort and continues the committed business turn.
            return

    def sink(event: PublicStreamEvent) -> None:
        post(event)

    def execute() -> None:
        try:
            runtime.run_turn(
                user_id=user_id,
                conversation_id=conversation_id,
                user_message=user_message,
                event_sink=sink,
            )
        finally:
            post(sentinel)

    task = asyncio.create_task(asyncio.to_thread(execute))
    try:
        while True:
            event = await queue.get()
            if event is sentinel:
                break
            if content_delay_seconds and isinstance(event, ContentDeltaEvent):
                await asyncio.sleep(content_delay_seconds)
            yield event
        await task
    finally:
        # Cancelling a thread-backed operation cannot safely stop a tool that
        # may already be committing. Drop this observer and let the business
        # turn reach its own terminal state.
        if task.done():
            task.result()
        else:
            task.add_done_callback(
                lambda completed: None
                if completed.cancelled()
                else completed.exception()
            )
