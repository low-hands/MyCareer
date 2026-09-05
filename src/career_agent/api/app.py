from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Literal

from career_agent.agent.context_manager import ContextManager
from career_agent.storage.context import CareerContextStore
import re
from urllib.parse import urlsplit, urlunsplit

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_contracts import OwnerSettingsContext
from career_agent.agent.openai_compatible_client import AgentConfigurationError
from career_agent.cli import build_main_agent_runtime, build_parser
from career_agent.domain.job_discovery import JobDetail, Provenance
from career_agent.api.reads import (
    WorkspaceReader,
    build_action_center_service,
    build_read_router,
    build_workspace_reader,
)
from career_agent.harness.streaming import (
    InteractionResponse,
    PublicStreamEvent,
    astream_turn_events,
)
from career_agent.security.authentication import require_scope
from career_agent.services.action_center import ActionCenterService
from career_agent.storage.api_keys import (
    ApiKeyPrincipal,
    ApiKeyStore,
    CAPTURE_WRITE,
    CHAT_WRITE,
    SETTINGS_WRITE,
    WORKSPACE_READ,
    SQLiteApiKeyStore,
)
from career_agent.storage.context import OwnerSettingsConflictError, OwnerSettingsEvent
from career_agent.storage.jobs import JobPostingRepository, SQLiteJobPostingRepository


class ChatStreamRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    # No ``user_id``. Identity comes from the API key, so a request cannot name
    # a user it does not hold a credential for — a stronger property than
    # checking that a stated user matches the token, because there is nothing to
    # state. Same discipline as the projection boundary refusing internal ids.
    conversation_id: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=100_000)
    interaction_response: InteractionResponse | None = None


class BrowserJobCaptureRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

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


class OwnerSettingsPatchRequest(BaseModel):
    """A conditional partial update; omitted fields retain their current value."""

    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(ge=0)
    boss_search: Literal["explicit_request_only", "allowed"] | None = None
    application_confirmation: Literal["always_ask", "on_user_report"] | None = None


class OwnerSettingsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    owner_settings: OwnerSettingsContext


class OwnerSettingsHistoryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    events: tuple[OwnerSettingsEvent, ...]


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


class GateAwareStreamingResponse(StreamingResponse):
    """Release a gate if the ASGI server never starts the body iterator.

    Once iteration starts, the detached business-turn producer owns release and
    keeps the gate until the turn truly finishes, even after client disconnect.
    Before iteration starts there is no producer, so this response owns the
    otherwise-leaked admission.
    """

    def __init__(self, *args, stream_started, release_unstarted, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._stream_started = stream_started
        self._release_unstarted = release_unstarted

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            if not self._stream_started():
                await self._release_unstarted()


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


def build_owner_settings_store() -> CareerContextStore:
    args = _runtime_args_from_env()
    return CareerContextStore(Path(args.context_store).expanduser())


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
    user_id: str,
    request_id: str | None = None,
    heartbeat_seconds: float,
    synthetic_content_delay_seconds: float = 0.025,
    on_turn_finished: Callable[[], Awaitable[None]] | None = None,
) -> AsyncIterator[str]:
    queue: asyncio.Queue[PublicStreamEvent | object] = asyncio.Queue()
    sentinel = object()

    async def produce() -> None:
        try:
            async for event in astream_turn_events(
                runtime,
                user_id=user_id,
                conversation_id=request.conversation_id,
                user_message=request.message,
                request_id=request_id,
                interaction_response=request.interaction_response,
                content_delay_seconds=synthetic_content_delay_seconds,
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


def build_api_key_store() -> ApiKeyStore:
    return SQLiteApiKeyStore(
        Path(os.environ.get("CAREER_AGENT_DATA_DIR", "data")) / "api_keys.sqlite3"
    )


def create_app(
    *,
    runtime_factory: Callable[[], MainAgentRuntime] | None = None,
    api_key_store_factory: Callable[[], ApiKeyStore] | None = None,
    capture_repository_factory: Callable[[], JobPostingRepository] | None = None,
    action_center_factory: Callable[[], ActionCenterService] | None = None,
    workspace_reader_factory: Callable[[], WorkspaceReader] | None = None,
    owner_settings_store_factory: Callable[[], CareerContextStore] | None = None,
    heartbeat_seconds: float = 15.0,
    synthetic_content_delay_seconds: float = 0.025,
) -> FastAPI:
    if heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be positive")
    if synthetic_content_delay_seconds < 0:
        raise ValueError("synthetic_content_delay_seconds cannot be negative")
    factory = runtime_factory or build_api_runtime
    capture_factory = capture_repository_factory or build_capture_repository
    # Read endpoints are built eagerly and separately from the agent runtime:
    # they need no model configuration, so a dashboard stays usable on a machine
    # where the worker keys are missing and /ready is reporting a failure.
    read_factory = action_center_factory or (
        lambda: build_action_center_service(_runtime_args_from_env())
    )
    workspace_factory = workspace_reader_factory or (
        lambda: build_workspace_reader(_runtime_args_from_env())
    )
    # Built on first use, not at import: constructing it opens the local
    # databases, and creating an app must not touch the real store paths.
    application_router = build_read_router(read_factory, workspace_factory)
    key_store_factory = api_key_store_factory or build_api_key_store
    settings_factory = owner_settings_store_factory or build_owner_settings_store

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
        # Opened before anything is served. Unlike the model runtime, a missing
        # credential store is not a degraded mode the dashboard can survive: it
        # is the difference between an authenticated API and an open one, so
        # ``authenticate`` refuses every request rather than assuming a default.
        app.state.api_key_store = key_store_factory()
        app.state.capture_repository = None
        app.state.owner_settings_store = None
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
    async def chat_stream(
        request: ChatStreamRequest,
        principal: ApiKeyPrincipal = Depends(require_scope(CHAT_WRITE)),
        idempotency_key: str | None = Header(
            default=None,
            alias="Idempotency-Key",
            min_length=1,
            max_length=200,
        ),
    ) -> StreamingResponse:
        runtime: MainAgentRuntime | None = application.state.runtime
        if runtime is None:
            raise HTTPException(
                status_code=503,
                detail=application.state.startup_error,
            )

        gate: ConversationRunGate = application.state.run_gate
        try:
            await gate.acquire(principal.user_id, request.conversation_id)
        except ConversationBusyError as error:
            # Counted, not just refused. Whether a process-local gate is enough
            # is a question about contention, and nothing else measures it.
            if runtime is not None:
                runtime.record_rejected_turn(
                    user_id=principal.user_id,
                    conversation_id=request.conversation_id,
                )
            # Counted, not just refused. Whether a process-local gate is enough
            # is a question about contention, and nothing else measures it.
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "CONVERSATION_TURN_IN_PROGRESS",
                    "message": "This conversation already has a running turn.",
                },
            ) from error

        async def release_gate() -> None:
            await gate.release(principal.user_id, request.conversation_id)

        stream_started = False

        async def generate() -> AsyncIterator[str]:
            nonlocal stream_started
            stream_started = True
            async for chunk in _sse_stream(
                runtime,
                request,
                user_id=principal.user_id,
                request_id=idempotency_key,
                heartbeat_seconds=heartbeat_seconds,
                synthetic_content_delay_seconds=synthetic_content_delay_seconds,
                on_turn_finished=release_gate,
            ):
                yield chunk

        return GateAwareStreamingResponse(
            generate(),
            stream_started=lambda: stream_started,
            release_unstarted=release_gate,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    def settings_store() -> CareerContextStore:
        store: CareerContextStore | None = application.state.owner_settings_store
        if store is None:
            store = settings_factory()
            application.state.owner_settings_store = store
        return store

    @application.get(
        "/v1/settings",
        response_model=OwnerSettingsResponse,
    )
    async def read_owner_settings(
        principal: ApiKeyPrincipal = Depends(require_scope(WORKSPACE_READ)),
    ) -> OwnerSettingsResponse:
        current = settings_store().get_owner_settings(principal.user_id)
        return OwnerSettingsResponse(
            owner_settings=current or OwnerSettingsContext()
        )

    @application.put(
        "/v1/settings",
        response_model=OwnerSettingsResponse,
    )
    async def update_owner_settings(
        request: OwnerSettingsPatchRequest,
        principal: ApiKeyPrincipal = Depends(require_scope(SETTINGS_WRITE)),
    ) -> OwnerSettingsResponse:
        if request.boss_search is None and request.application_confirmation is None:
            raise HTTPException(status_code=422, detail="At least one setting is required")
        store = settings_store()
        current = store.get_owner_settings(principal.user_id) or OwnerSettingsContext()
        desired = current.model_copy(
            update={
                "preferences": current.preferences.model_copy(
                    update={
                        "boss_search": request.boss_search
                        or current.preferences.boss_search
                    }
                ),
                "behavior_policy": current.behavior_policy.model_copy(
                    update={
                        "application_confirmation": (
                            request.application_confirmation
                            or current.behavior_policy.application_confirmation
                        )
                    }
                ),
            }
        )
        try:
            updated = store.update_owner_settings(
                user_id=principal.user_id,
                desired=desired,
                expected_revision=request.expected_revision,
                actor_type="api_key",
                actor_id=principal.key_id,
            )
        except OwnerSettingsConflictError as error:
            latest = store.get_owner_settings(principal.user_id) or OwnerSettingsContext()
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "OWNER_SETTINGS_REVISION_CONFLICT",
                    "message": str(error),
                    "current_revision": latest.revision,
                },
            ) from error
        return OwnerSettingsResponse(owner_settings=updated)

    @application.get(
        "/v1/settings/history",
        response_model=OwnerSettingsHistoryResponse,
    )
    async def read_owner_settings_history(
        principal: ApiKeyPrincipal = Depends(require_scope(WORKSPACE_READ)),
    ) -> OwnerSettingsHistoryResponse:
        return OwnerSettingsHistoryResponse(
            events=settings_store().list_owner_settings_events(
                user_id=principal.user_id
            )
        )

    @application.post(
        "/v1/browser-captures/jobs",
        response_model=BrowserJobCaptureResponse,
    )
    async def capture_job(
        request: BrowserJobCaptureRequest,
        principal: ApiKeyPrincipal = Depends(require_scope(CAPTURE_WRITE)),
        capture_version: str | None = Header(
            default=None,
            alias="X-Career-Agent-Capture",
        ),
    ) -> BrowserJobCaptureResponse:
        # The header stays, demoted to what it always was: a payload-shape
        # version marker. It never was access control — anyone could send it —
        # and leaving it as the only gate while every other route gained one
        # would keep the weakest door open.
        if capture_version != "v1":
            raise HTTPException(
                status_code=422, detail="Unsupported browser capture version"
            )
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
                user_id=principal.user_id,
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
