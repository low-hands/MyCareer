from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
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


class InteractionResponse(StreamContract):
    """A UI response whose authority is bound to one durable interaction."""

    interaction_id: str = Field(pattern=r"^interaction_[a-f0-9]{20}$")
    scope: Literal["resume_analysis_confirmation", "capability_confirmation"]
    action: Literal["confirm", "cancel"]


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
    scope: Literal["resume_analysis_confirmation", "capability_confirmation"] | None = None

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
    # Transport pacing is internal metadata. It must not alter the public SSE
    # contract, but the async adapter needs to distinguish provider-paced
    # tokens from completed prose split into synthetic chunks.
    delivery: Literal["provider", "synthetic"] = Field(
        default="provider",
        exclude=True,
    )


class ArtifactReadyEvent(StreamContract):
    type: Literal["artifact_ready"] = "artifact_ready"
    artifact_id: str = Field(min_length=1)
    filename: str = Field(min_length=1, max_length=180)
    media_type: str = Field(min_length=1, max_length=100)
    byte_size: int = Field(ge=1)


class ReportReadyEvent(StreamContract):
    """A turn produced a stored report the reply only summarizes.

    Carries the reference rather than the report: the body is fetched from the
    read API, which is the same path a reloaded transcript takes, so the live
    card and the restored one cannot drift. The id is safe here because this
    stream goes to the person whose report it is; the decision model gets a
    turn-local index instead.
    """

    type: Literal["report_ready"] = "report_ready"
    kind: Literal[
        "job_research_report",
        "mock_interview_report",
        "interview_preparation",
        "interview_retro_report",
        "resume_job_match",
        "resume_tailoring_draft",
    ]
    resource_id: str = Field(min_length=1, max_length=200)
    status_at_delivery: Literal["current", "outdated", "superseded"] | None = None
    anchored_by_other_job: bool | None = None


class ClientActionEvent(StreamContract):
    type: Literal["client_action"] = "client_action"
    action: Literal["open_url"]
    url: str = Field(pattern=r"^https://", max_length=2000)
    label: str = Field(min_length=1, max_length=240)


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
    | ReportReadyEvent
    | ClientActionEvent
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
        request_id: str | None = None,
        interaction_response: InteractionResponse | None = None,
        event_sink: StreamEventSink | None = None,
    ) -> object: ...


def interaction_id(*durable_parts: object) -> str:
    """Derive an opaque stable handle from state that was already committed."""

    canonical = "\x1f".join(str(part) for part in durable_parts)
    return f"interaction_{sha256(canonical.encode('utf-8')).hexdigest()[:20]}"


InteractionScope: TypeAlias = Literal[
    "resume_analysis_confirmation", "capability_confirmation"
]

_CAPABILITY_CONFIRMATION_OPTIONS = (
    InteractionOption(value="confirm", label="确认执行"),
    InteractionOption(value="cancel", label="不要执行"),
)
_RESUME_ANALYSIS_CONFIRMATION_OPTIONS = (
    InteractionOption(value="confirm", label="确认并导入"),
    InteractionOption(value="cancel", label="取消导入"),
)
_SCOPED_OPTIONS: Mapping[InteractionScope, tuple[InteractionOption, ...]] = {
    "capability_confirmation": _CAPABILITY_CONFIRMATION_OPTIONS,
    "resume_analysis_confirmation": _RESUME_ANALYSIS_CONFIRMATION_OPTIONS,
}


def scoped_interaction_message(
    scope: InteractionScope, action: Literal["confirm", "cancel"]
) -> str:
    """The user message a scoped answer carries: the label of the button pressed.

    The web client sends the option's label as the turn's user message, so a
    client without buttons (the CLI) uses the same text and the transcript
    reads identically whichever client answered.
    """

    return next(
        option.label for option in _SCOPED_OPTIONS[scope] if option.value == action
    )


def capability_confirmation_event(
    *, conversation_id: str, confirmation_id: str, prompt: str
) -> InteractionRequiredEvent:
    """The gate an owner rule opened, rebuildable from durable state alone.

    Keyed on the sealed confirmation rather than on anything about the turn, so
    a reload — or a different process — offers the same interaction id the
    pending seal answers to. An id derived from turn-local state would make the
    owner's "yes" unroutable the moment the page was refreshed.
    """

    return InteractionRequiredEvent(
        interaction_id=interaction_id(
            conversation_id, "capability_confirmation", confirmation_id
        ),
        scope="capability_confirmation",
        kind="approval",
        prompt=prompt,
        options=_CAPABILITY_CONFIRMATION_OPTIONS,
    )


def resume_analysis_confirmation_event(
    *, conversation_id: str, analysis_id: str
) -> InteractionRequiredEvent:
    """Rebuild the same pending gate for live delivery and transcript reload."""

    return InteractionRequiredEvent(
        interaction_id=interaction_id(
            conversation_id,
            "resume_analysis_confirmation",
            analysis_id,
        ),
        scope="resume_analysis_confirmation",
        kind="confirmation",
        prompt="请核对上面的候选事实。确认后才会写入职业事实库。",
        options=_RESUME_ANALYSIS_CONFIRMATION_OPTIONS,
    )


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
    request_id: str | None = None,
    interaction_response: InteractionResponse | None = None,
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
            turn_arguments = dict(
                user_id=user_id,
                conversation_id=conversation_id,
                user_message=user_message,
                event_sink=sink,
            )
            if request_id is not None:
                turn_arguments["request_id"] = request_id
            if interaction_response is not None:
                turn_arguments["interaction_response"] = interaction_response
            runtime.run_turn(**turn_arguments)
        finally:
            post(sentinel)

    task = asyncio.create_task(asyncio.to_thread(execute))
    try:
        while True:
            event = await queue.get()
            if event is sentinel:
                break
            if (
                content_delay_seconds
                and isinstance(event, ContentDeltaEvent)
                and event.delivery == "synthetic"
            ):
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
