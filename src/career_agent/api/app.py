from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.openai_compatible_client import AgentConfigurationError
from career_agent.cli import build_main_agent_runtime, build_parser
from career_agent.domain.job_discovery import JobDetail, Provenance
from career_agent.api.reads import build_action_center_service, build_read_router
from career_agent.harness.streaming import PublicStreamEvent, astream_turn_events
from career_agent.services.action_center import ActionCenterService
from career_agent.storage.jobs import JobPostingRepository, SQLiteJobPostingRepository


class ChatStreamRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    user_id: str = Field(min_length=1, max_length=200)
    conversation_id: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=100_000)


class BrowserJobCaptureRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    user_id: str = Field(min_length=1, max_length=200)
    source_url: str = Field(min_length=1, max_length=2_000)
    title: str = Field(min_length=1, max_length=500)
    company_name: str = Field(min_length=1, max_length=500)
    description: str = Field(min_length=1, max_length=100_000)
    city: str | None = Field(default=None, max_length=200)
    salary: str | None = Field(default=None, max_length=200)
    experience: str | None = Field(default=None, max_length=200)
    education: str | None = Field(default=None, max_length=200)


class BrowserJobCaptureResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    job_posting_id: str
    jd_snapshot_id: str
    snapshot_version: int
    title: str
    company_name: str


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
    return build_parser().parse_args(
        [
            "chat",
            "--user-id",
            "api-runtime",
            "--session-id",
            "api-runtime",
            "--message",
            "runtime-bootstrap",
        ]
    )


def build_api_runtime() -> MainAgentRuntime:
    return build_main_agent_runtime(_runtime_args_from_env())


def build_capture_repository() -> JobPostingRepository:
    args = _runtime_args_from_env()
    return SQLiteJobPostingRepository(Path(args.job_store).expanduser())


def _canonical_boss_job_url(raw_url: str) -> tuple[str, str | None]:
    try:
        parsed = urlsplit(raw_url)
    except ValueError as error:
        raise ValueError("Invalid BOSS job URL") from error
    hostname = (parsed.hostname or "").casefold()
    if parsed.scheme != "https" or not (
        hostname == "zhipin.com" or hostname.endswith(".zhipin.com")
    ):
        raise ValueError("Only HTTPS BOSS job URLs are accepted")
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    canonical = urlunsplit(("https", hostname, path, "", ""))
    match = re.search(r"/job_detail/([^/?#]+?)(?:\.html)?$", path, re.IGNORECASE)
    source_job_id = match.group(1) if match else None
    return canonical, source_job_id


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
    capture_repository_factory: Callable[[], JobPostingRepository] | None = None,
    action_center_factory: Callable[[], ActionCenterService] | None = None,
    heartbeat_seconds: float = 15.0,
) -> FastAPI:
    if heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be positive")
    factory = runtime_factory or build_api_runtime
    capture_factory = capture_repository_factory or build_capture_repository
    # Read endpoints are built eagerly and separately from the agent runtime:
    # they need no model configuration, so a dashboard stays usable on a machine
    # where the worker keys are missing and /ready is reporting a failure.
    read_factory = action_center_factory or (
        lambda: build_action_center_service(_runtime_args_from_env())
    )
    # Built on first use, not at import: constructing it opens the local
    # databases, and creating an app must not touch the real store paths.
    application_router = build_read_router(read_factory)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime: MainAgentRuntime | None = None
        startup_error: dict[str, str] | None = None
        try:
            runtime = factory()
        except AgentConfigurationError as error:
            startup_error = {
                "code": error.code,
                "message": (
                    f"服务尚未配置完成：{error} "
                    "请在 .env 补齐这些配置后重启 FastAPI。"
                ),
            }
        app.state.runtime = runtime
        app.state.startup_error = startup_error
        app.state.run_gate = ConversationRunGate()
        app.state.capture_repository = None
        app.state.action_center = None
        try:
            yield
        finally:
            close = getattr(runtime, "close", None) if runtime is not None else None
            if close is not None:
                close()

    application = FastAPI(
        title="Career Agent API",
        version="0.1.0",
        lifespan=lifespan,
    )


    application.include_router(application_router)

    @application.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/ready")
    async def ready() -> dict[str, str]:
        startup_error: dict[str, str] | None = application.state.startup_error
        if startup_error is not None:
            raise HTTPException(status_code=503, detail=startup_error)
        return {"status": "ready"}

    @application.post("/v1/chat/stream")
    async def chat_stream(request: ChatStreamRequest) -> StreamingResponse:
        runtime: MainAgentRuntime | None = application.state.runtime
        if runtime is None:
            raise HTTPException(
                status_code=503,
                detail=application.state.startup_error,
            )

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

    @application.post(
        "/v1/browser-captures/jobs",
        response_model=BrowserJobCaptureResponse,
    )
    async def capture_job(
        request: BrowserJobCaptureRequest,
        capture_version: str | None = Header(
            default=None,
            alias="X-Career-Agent-Capture",
        ),
    ) -> BrowserJobCaptureResponse:
        if capture_version != "v1":
            raise HTTPException(status_code=403, detail="Browser capture header is required")
        try:
            source_url, source_job_id = _canonical_boss_job_url(request.source_url)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        repository: JobPostingRepository | None = application.state.capture_repository
        if repository is None:
            repository = capture_factory()
            application.state.capture_repository = repository
        captured_at = datetime.now(timezone.utc)
        provenance = Provenance(
            source_name="boss",
            source_job_id=source_job_id,
            source_url=source_url,
            captured_at=captured_at,
            operation="browser_explicit_save",
            adapter_version="career-agent-browser-capture-v1",
        )
        detail = JobDetail(
            source_name="boss",
            source_job_id=source_job_id,
            source_url=source_url,
            title=request.title,
            company_name=request.company_name,
            description=request.description,
            city=request.city,
            salary=request.salary,
            experience=request.experience,
            education=request.education,
            captured_at=captured_at,
            provenance=provenance,
        )
        try:
            saved = repository.save_captured_detail(
                user_id=request.user_id,
                detail=detail,
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return BrowserJobCaptureResponse(
            job_posting_id=saved.posting.id,
            jd_snapshot_id=saved.snapshot.id,
            snapshot_version=saved.snapshot.version,
            title=saved.posting.title,
            company_name=saved.posting.company_name,
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
