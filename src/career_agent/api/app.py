from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
import math
import os
from pathlib import Path
from typing import Literal

from career_agent.storage.context import CareerContextStore
import re
from urllib.parse import urlsplit, urlunsplit

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator

from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_contracts import (
    ConfirmBefore,
    OwnerSettingsContext,
    canonical_confirm_before,
)
from career_agent.agent.openai_compatible_client import AgentConfigurationError
from career_agent.cli import build_main_agent_runtime, build_parser
from career_agent.domain.job_discovery import JobDetail, Provenance
from career_agent.api.reads import (
    WorkspaceReader,
    build_action_center_service,
    build_read_router,
    build_workspace_reader,
)
from career_agent.api.integrations import build_integration_router
from career_agent.api.single_worker import (
    SingleWorkerLock,
    lock_path_for,
    refuse_multi_worker_configuration,
)
from career_agent.harness.streaming import (
    InteractionResponse,
    PublicStreamEvent,
    TurnInputResource,
    TurnCompletedEvent,
    TurnFailedEvent,
    TurnSuspendedEvent,
    astream_turn_events,
)
from career_agent.security.authentication import require_scope
from career_agent.services.action_center import ActionCenterService
from career_agent.services.integrations import IntegrationConnectionService
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
from career_agent.storage.connector_secrets import KeyringConnectorSecretStore
from career_agent.storage.oauth_flows import SQLiteOAuthFlowStore
from career_agent.storage.email_tracking import SQLiteEmailTrackingStore
from career_agent.storage.calendar import SQLiteCalendarStore
from career_agent.storage.job_captures import (
    JobCaptureStore,
    JobCapturedEvent,
    SQLiteJobCaptureStore,
)
from career_agent.storage.jobs import (
    JobPostingRepository,
    SQLiteJobPostingRepository,
    StoredJobRecord,
)


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
    input_resources: tuple[TurnInputResource, ...] = Field(default=(), max_length=8)
    """Structured references to durable user assets this message is about.

    A resume is attached by ``resume_version`` id, never by pasting the file
    or its id into ``message``. The runtime resolves each id against the
    authenticated user; one that does not resolve fails the turn.
    """

    @field_validator("input_resources")
    @classmethod
    def _distinct_input_resources(
        cls, value: tuple[TurnInputResource, ...]
    ) -> tuple[TurnInputResource, ...]:
        seen: set[tuple[str, str]] = set()
        for resource in value:
            key = (resource.kind, resource.id)
            if key in seen:
                raise ValueError("input_resources must not repeat a resource")
            seen.add(key)
        return value


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
    capture_intent_id: str | None = Field(
        default=None, pattern=r"^capint_[a-f0-9]{32}$"
    )
    """The search this save came from, when the agent opened it.

    Optional on purpose: a job saved from the user's own browsing has no
    intent and goes into the library alone. A stale, foreign, or unknown
    intent is treated the same way rather than refused, so the save itself
    never fails over correlation.
    """


def capture_follow_up_message(event: JobCapturedEvent) -> str:
    """The user message the backend continuation turn runs with.

    Product decision (092, 方案 4): a capture continuation only records the save
    in its conversation, so the reply ends with the JD card and later "这个岗位"
    resolves to this exact snapshot. It does not analyse the JD. Analysis is the
    standalone task the user starts from the job library ("让 Agent 分析"), so
    an unattended turn never spends a model analysis the user did not ask for,
    and a batch of saves does not queue a batch of analyses. The text therefore
    states that analysis is available rather than requesting it.
    """
    return (
        f"我已经从 BOSS 保存了岗位「{_page_label(event.title)} · "
        f"{_page_label(event.company_name)}」，先记下来就好。"
        "暂不需要分析；之后我可以在岗位库点「让 Agent 分析」，再让你仅基于这份 JD 做岗位分析。"
    )


_PAGE_LABEL_LIMIT = 40
# Quotes and brackets that could close the 「…」 span and let page text read as
# the user's own words, plus every control and line-separator character.
_PAGE_LABEL_STRIP = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029「」『』\"'`“”‘’<>\[\]{}]")


def _page_label(text: str) -> str:
    """A BOSS page title or company name, made safe to quote in a user message.

    Both come from the page DOM (up to 500 chars each), so they are untrusted.
    The follow-up turn speaks as the user; a crafted title must stay a short,
    single-line name inside its brackets, never an instruction.
    """
    flat = " ".join(_PAGE_LABEL_STRIP.sub(" ", text).split())
    if len(flat) > _PAGE_LABEL_LIMIT:
        flat = flat[: _PAGE_LABEL_LIMIT - 1].rstrip() + "…"
    return flat or "未命名"


class BrowserJobCaptureResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    job_posting_id: str
    jd_snapshot_id: str
    snapshot_version: int
    title: str
    company_name: str
    conversation_id: str | None = None
    """The conversation the agent will continue in, when a live intent matched."""
    capture_event_id: str | None = None
    capture_event_created: bool = False
    continuation_status: Literal[
        "saved_only", "pending", "completed", "discarded", "failed", "expired"
    ] = "saved_only"
    continuation_reason: Literal[
        "no_intent", "invalid_intent", "expired_intent", "consumed_intent",
        "conversation_unavailable",
    ] | None = "no_intent"


class JobCapturedEventView(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    conversation_id: str
    job_posting_id: str
    jd_snapshot_id: str
    title: str
    company_name: str
    created_at: datetime
    continuation_status: Literal["pending", "completed", "discarded", "failed", "expired"]
    """``expired``: saved, but not continued within 24 hours; the page says so."""
    continuation_turn_id: str | None = None

    @classmethod
    def from_event(cls, event: JobCapturedEvent) -> "JobCapturedEventView":
        return cls(
            id=event.id,
            conversation_id=event.conversation_id,
            job_posting_id=event.job_posting_id,
            jd_snapshot_id=event.jd_snapshot_id,
            title=event.title,
            company_name=event.company_name,
            created_at=event.created_at,
            continuation_status=event.continuation_status,
            continuation_turn_id=event.continuation_turn_id,
        )


class JobCapturedEventsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    events: tuple[JobCapturedEventView, ...]


class JobCapturedEventAckResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str
    acknowledged: bool
    """False when the event was unknown, another user's, or already acknowledged."""


class JobCaptureRetryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str
    retried: bool
    """False unless the event was this user's and its continuation had failed."""


class OwnerSettingsPatchRequest(BaseModel):
    """A conditional partial update; omitted fields retain their current value."""

    model_config = ConfigDict(extra="forbid")

    expected_revision: int = Field(ge=0)
    boss_search: Literal["explicit_request_only", "allowed"] | None = None
    application_confirmation: Literal["always_ask", "on_user_report"] | None = None
    confirm_before: ConfirmBefore | None = Field(
        default=None,
        description=(
            "Full replacement list of WRITE capability names to approve one by "
            "one before they run. An empty list clears the rule."
        ),
    )

    @field_validator("confirm_before")
    @classmethod
    def normalise_confirm_before(cls, value: ConfirmBefore | None) -> ConfirmBefore | None:
        return None if value is None else canonical_confirm_before(value)


class OwnerSettingsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    owner_settings: OwnerSettingsContext


class OwnerSettingsHistoryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    events: tuple[OwnerSettingsEvent, ...]


class ConversationBusyError(Exception):
    pass


class TurnCapacityError(Exception):
    def __init__(self, max_active: int) -> None:
        super().__init__(f"{max_active} turns are already running")
        self.max_active = max_active


DEFAULT_MAX_CONCURRENT_TURNS = 3
MAX_CONCURRENT_TURNS_ENV = "CAREER_AGENT_MAX_CONCURRENT_TURNS"
DEFAULT_SHUTDOWN_DRAIN_SECONDS = 30.0
SHUTDOWN_DRAIN_SECONDS_ENV = "CAREER_AGENT_SHUTDOWN_DRAIN_SECONDS"


def max_concurrent_turns_from_env(
    environ: Mapping[str, str] | None = None,
) -> int:
    raw = (environ if environ is not None else os.environ).get(MAX_CONCURRENT_TURNS_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_MAX_CONCURRENT_TURNS
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise ValueError(
            f"{MAX_CONCURRENT_TURNS_ENV}={raw!r} 无效：需要不小于 1 的整数。"
        )
    return value


def shutdown_drain_seconds_from_env(
    environ: Mapping[str, str] | None = None,
) -> float:
    raw = (environ if environ is not None else os.environ).get(SHUTDOWN_DRAIN_SECONDS_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_SHUTDOWN_DRAIN_SECONDS
    try:
        value = float(raw)
    except ValueError:
        value = -1.0
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"{SHUTDOWN_DRAIN_SECONDS_ENV}={raw!r} 无效：需要不小于 0 的有限秒数。"
        )
    return value


class ConversationRunGate:
    """Reject overlapping turns for one conversation without serializing users.

    Also caps how many turns run at once in this process. Every turn holds a
    worker thread for the whole model call, so with no cap a burst of
    conversations can hold every thread until the model times out and the
    machine feels hung. There is no queue: a refused turn is refused now, and
    the client asks again.
    """

    def __init__(self, *, max_active: int | None = DEFAULT_MAX_CONCURRENT_TURNS) -> None:
        if max_active is not None and max_active < 1:
            raise ValueError("max_active must be at least 1")
        self._guard = asyncio.Lock()
        self._active: set[tuple[str, str]] = set()
        self._max_active = max_active
        self._idle = asyncio.Event()
        self._idle.set()

    @property
    def max_active(self) -> int | None:
        return self._max_active

    @property
    def active_count(self) -> int:
        return len(self._active)

    def is_active(self, user_id: str, conversation_id: str) -> bool:
        return (user_id, conversation_id) in self._active

    async def acquire(self, user_id: str, conversation_id: str) -> None:
        key = (user_id, conversation_id)
        async with self._guard:
            if key in self._active:
                raise ConversationBusyError
            if self._max_active is not None and len(self._active) >= self._max_active:
                raise TurnCapacityError(self._max_active)
            self._active.add(key)
            self._idle.clear()

    async def release(self, user_id: str, conversation_id: str) -> None:
        async with self._guard:
            self._active.discard((user_id, conversation_id))
            if not self._active:
                self._idle.set()

    async def drain(self, timeout: float) -> bool:
        """Wait until no turn is admitted; ``False`` if some still are after ``timeout``.

        A turn stays admitted until its detached producer sees the synchronous
        runtime return, client connected or not, so an idle gate means no turn
        thread is executing in this process.
        """

        if self._idle.is_set():
            return True
        try:
            await asyncio.wait_for(self._idle.wait(), timeout=timeout)
        except TimeoutError:
            return False
        return True


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
    # This runs before ``OpenAICompatibleAgentConfig.from_env`` gets a chance
    # to load the workspace dotenv file. Load it here so API-only settings such
    # as the main-model timeout work from the same deployment file as the model
    # endpoint and key; an explicitly exported value still wins.
    load_dotenv()
    args = build_parser().parse_args(
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
    raw_timeout = os.environ.get("MAIN_AGENT_TIMEOUT_SECONDS")
    if raw_timeout is None:
        return args
    try:
        timeout_seconds = float(raw_timeout)
    except ValueError as error:
        raise AgentConfigurationError(
            "AGENT_CONFIGURATION_INVALID",
            "MAIN_AGENT_TIMEOUT_SECONDS must be a finite positive number.",
        ) from error
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise AgentConfigurationError(
            "AGENT_CONFIGURATION_INVALID",
            "MAIN_AGENT_TIMEOUT_SECONDS must be a finite positive number.",
        )
    args.main_agent_timeout_seconds = timeout_seconds
    return args


def build_api_runtime() -> MainAgentRuntime:
    return build_main_agent_runtime(_runtime_args_from_env())


def build_capture_repository() -> JobPostingRepository:
    args = _runtime_args_from_env()
    return SQLiteJobPostingRepository(Path(args.job_store).expanduser())


def build_job_capture_store() -> JobCaptureStore:
    args = _runtime_args_from_env()
    return SQLiteJobCaptureStore(Path(args.job_store).expanduser())


def build_owner_settings_store() -> CareerContextStore:
    args = _runtime_args_from_env()
    return CareerContextStore(Path(args.context_store).expanduser())


def build_integration_service() -> IntegrationConnectionService:
    args = _runtime_args_from_env()
    return IntegrationConnectionService.from_env(
        email_store=SQLiteEmailTrackingStore(Path(args.email_store).expanduser()),
        calendar_store=SQLiteCalendarStore(Path(args.calendar_store).expanduser()),
        flow_store=SQLiteOAuthFlowStore(Path(args.context_store).expanduser()),
        secret_store=KeyringConnectorSecretStore(),
    )


class JobClosureRequest(BaseModel):
    """A saved posting the user just found closed. Only the page's own URL:
    identity comes from the credential, and the rest from what is stored."""

    model_config = ConfigDict(extra="forbid")

    source_url: str = Field(min_length=1, max_length=2000)


class JobClosureResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    matched: bool
    job_posting_id: str | None = None


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
                input_resources=request.input_resources,
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


def build_single_worker_lock(args: argparse.Namespace | None = None) -> SingleWorkerLock:
    """Lock the directory holding the conversation store, not the API-key store.

    ``ConversationRunGate`` guards turn commits, and those land in the stores
    the runtime opens from these parsed arguments (``~/.career-agent`` by
    default). ``CAREER_AGENT_DATA_DIR`` only holds ``api_keys.sqlite3``; two
    processes with different key directories would still share every business
    database, so the lock must live where the business databases live.
    """

    context_store = Path((args or _runtime_args_from_env()).context_store).expanduser()
    return SingleWorkerLock(lock_path_for(context_store.parent))


def create_app(
    *,
    runtime_factory: Callable[[], MainAgentRuntime] | None = None,
    api_key_store_factory: Callable[[], ApiKeyStore] | None = None,
    capture_repository_factory: Callable[[], JobPostingRepository] | None = None,
    job_capture_store_factory: Callable[[], JobCaptureStore] | None = None,
    action_center_factory: Callable[[], ActionCenterService] | None = None,
    workspace_reader_factory: Callable[[], WorkspaceReader] | None = None,
    integration_service_factory: Callable[[], IntegrationConnectionService] | None = None,
    owner_settings_store_factory: Callable[[], CareerContextStore] | None = None,
    single_worker_lock_factory: Callable[[], SingleWorkerLock | None] | None = None,
    max_concurrent_turns: int | None = None,
    shutdown_drain_seconds: float | None = None,
    heartbeat_seconds: float = 15.0,
    synthetic_content_delay_seconds: float = 0.025,
) -> FastAPI:
    if heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be positive")
    if synthetic_content_delay_seconds < 0:
        raise ValueError("synthetic_content_delay_seconds cannot be negative")
    if max_concurrent_turns is not None and max_concurrent_turns < 1:
        raise ValueError("max_concurrent_turns must be at least 1")
    if shutdown_drain_seconds is not None and (
        not math.isfinite(shutdown_drain_seconds) or shutdown_drain_seconds < 0
    ):
        raise ValueError("shutdown_drain_seconds must be a finite, non-negative number")
    factory = runtime_factory or build_api_runtime
    capture_wakeup = asyncio.Event()
    capture_workers: set[asyncio.Task[None]] = set()
    capture_factory = capture_repository_factory or build_capture_repository
    capture_store_factory = job_capture_store_factory or build_job_capture_store
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
    def ensure_conversation_idle(user_id: str, conversation_id: str) -> None:
        gate: ConversationRunGate = application.state.run_gate
        if gate.is_active(user_id, conversation_id):
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "CONVERSATION_TURN_IN_PROGRESS",
                    "message": "这个对话仍在处理中，请等待本轮结束后再删除。",
                },
            )

    application_router = build_read_router(
        read_factory,
        workspace_factory,
        before_conversation_delete=ensure_conversation_idle,
    )
    integration_router = build_integration_router(
        integration_service_factory or build_integration_service
    )
    key_store_factory = api_key_store_factory or build_api_key_store
    settings_factory = owner_settings_store_factory or build_owner_settings_store
    lock_factory = single_worker_lock_factory or build_single_worker_lock

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal capture_wakeup
        capture_wakeup = asyncio.Event()
        # ``ConversationRunGate`` below is per process, so the process must be
        # the only one on these databases. Taken before any store is opened
        # and held until shutdown; a second worker fails here, not at commit.
        refuse_multi_worker_configuration()
        turn_cap = (
            max_concurrent_turns
            if max_concurrent_turns is not None
            else max_concurrent_turns_from_env()
        )
        drain_seconds = (
            shutdown_drain_seconds
            if shutdown_drain_seconds is not None
            else shutdown_drain_seconds_from_env()
        )
        lock = lock_factory()
        if lock is not None:
            lock.acquire()
        app.state.single_worker_lock = lock
        runtime: MainAgentRuntime | None = None
        gate: ConversationRunGate | None = None
        try:
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
            gate = ConversationRunGate(max_active=turn_cap)
            app.state.run_gate = gate
            # Opened before anything is served. Unlike the model runtime, a missing
            # credential store is not a degraded mode the dashboard can survive: it
            # is the difference between an authenticated API and an open one, so
            # ``authenticate`` refuses every request rather than assuming a default.
            app.state.api_key_store = key_store_factory()
            app.state.capture_repository = None
            app.state.job_capture_store = None
            app.state.owner_settings_store = None
            app.state.action_center = None
            dispatcher = asyncio.create_task(dispatch_captures())
            try:
                yield
            finally:
                # Only the loop that starts new continuations stops here. A
                # running one holds its conversation's gate until it settles,
                # so the gate drain below waits for it like any other turn.
                dispatcher.cancel()
                await asyncio.gather(dispatcher, return_exceptions=True)
        finally:
            # A turn whose client has gone keeps running in a detached producer
            # thread, and the lock promises the next process that nothing here
            # is still executing. So: wait for those turns, and if some outlive
            # the wait, leave the lock (and the stores under them) to the
            # kernel, which releases only once the process is truly gone.
            drained = gate is None or await gate.drain(drain_seconds)
            app.state.shutdown_drained = drained
            if drained:
                try:
                    close = getattr(runtime, "close", None) if runtime is not None else None
                    if close is not None:
                        close()
                finally:
                    if lock is not None:
                        lock.release()

    application = FastAPI(
        title="Career Agent API",
        version="0.1.0",
        lifespan=lifespan,
    )


    application.include_router(application_router)
    application.include_router(integration_router)

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

        if idempotency_key and idempotency_key.startswith("jobcap_"):
            # ``jobcap_`` ids are the backend dispatcher's request ids. A client
            # key in that namespace would share the dispatcher turn's idempotency
            # record, so it is refused up front: no lookup, no wait, no gate.
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "JOB_CAPTURE_CONTINUATION_BACKEND_OWNED",
                    "message": "岗位续接由后端执行，请读取采集事件状态。",
                },
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
        except TurnCapacityError as error:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "TURN_CAPACITY_EXHAUSTED",
                    "message": (
                        f"当前任务较多（已有 {error.max_active} 轮在处理），请稍后再试。"
                    ),
                    "max_concurrent_turns": error.max_active,
                },
                headers={"Retry-After": "10"},
            ) from error

        async def release_gate() -> None:
            await gate.release(principal.user_id, request.conversation_id)
            capture_wakeup.set()

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

    def conversation_accepts_capture(user_id: str, conversation_id: str) -> bool:
        session = settings_store().get_session(user_id, conversation_id)
        return session is not None and session.status == "active"

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
        if (
            request.boss_search is None
            and request.application_confirmation is None
            and request.confirm_before is None
        ):
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
                        ),
                        "confirm_before": (
                            current.behavior_policy.confirm_before
                            if request.confirm_before is None
                            else request.confirm_before
                        ),
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
        "/v1/browser-captures/job-closures",
        response_model=JobClosureResponse,
    )
    async def report_job_closure(
        request: JobClosureRequest,
        principal: ApiKeyPrincipal = Depends(require_scope(CAPTURE_WRITE)),
        capture_version: str | None = Header(
            default=None, alias="X-Career-Agent-Capture"
        ),
    ) -> JobClosureResponse:
        """Record that a saved posting was found closed on the site.

        This is the only way ``availability_status`` ever leaves ``active``.
        Nothing here polls: the extension reports a page the user opened
        themselves, which is why the field can be trusted and why it is often
        simply unknown. Re-fetching every saved posting on a schedule is what
        got the earlier API search blocked, and it would also make the field a
        claim about the site rather than an observation of it.

        Reported through ``capture:write`` — the same credential, in the same
        weak storage, saying the same class of thing about the same page. A
        separate scope would suggest this carries more authority than a
        capture, and it carries less: one status field on a job the user has
        already saved.
        """

        if capture_version != "v1":
            raise HTTPException(
                status_code=422, detail="Unsupported browser capture version"
            )
        try:
            source_url, source_job_id = _canonical_boss_job_url(request.source_url)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        repository = capture_factory()
        job_posting_id = repository.find_by_source(
            user_id=principal.user_id,
            source_name="boss",
            source_job_id=source_job_id,
            source_url=source_url,
        )
        if job_posting_id is None:
            # Not an error: a closed page for a posting that was never saved is
            # simply not about anything this workspace holds. Reporting it as a
            # failure would put a red banner on the user's screen for a page
            # they had no stake in.
            return JobClosureResponse(matched=False)
        repository.mark_availability(
            user_id=principal.user_id,
            job_posting_id=job_posting_id,
            status="closed",
        )
        return JobClosureResponse(matched=True, job_posting_id=job_posting_id)

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
        response = BrowserJobCaptureResponse(
            job_posting_id=saved.posting.id,
            jd_snapshot_id=saved.snapshot.id,
            snapshot_version=saved.snapshot.version,
            title=saved.posting.title,
            company_name=saved.posting.company_name,
        )
        resolved = resolve_capture_intent(
            principal.user_id, request.capture_intent_id, response, saved
        )
        record_capture(
            user_id=principal.user_id,
            conversation_id=resolved.conversation_id,
            capture_event_id=resolved.capture_event_id,
            phase="intent",
            status=resolved.continuation_reason or "matched",
        )
        return resolved

    def resolve_capture_intent(
        user_id: str,
        intent_id: str | None,
        response: BrowserJobCaptureResponse,
        saved: StoredJobRecord,
    ) -> BrowserJobCaptureResponse:
        # Ownership is checked inside the store: an intent another user
        # created reads as no intent at all, and the save stays a library save.
        store = _job_capture_store()
        # The app can be running in an embedded browser while the extension is
        # running in Chrome. In that topology there is no shared extension
        # bridge, so recover only the newest very recent BOSS intent. An
        # ordinary save with no recent search remains library-only.
        intent = (
            store.get_intent(user_id=user_id, intent_id=intent_id)
            if intent_id is not None
            else store.get_recent_live_intent(user_id=user_id, platform="boss")
        )
        if intent is None:
            return response.model_copy(update={
                "continuation_reason": (
                    "invalid_intent" if intent_id is not None else "no_intent"
                )
            })
        if not conversation_accepts_capture(user_id, intent.conversation_id):
            return response.model_copy(
                update={"continuation_reason": "conversation_unavailable"}
            )
        recording = store.record_capture(
            intent=intent,
            job_posting_id=saved.posting.id,
            jd_snapshot_id=saved.snapshot.id,
            title=saved.posting.title,
            company_name=saved.posting.company_name,
        )
        if recording is None:
            return response.model_copy(update={
                "continuation_reason": (
                    "consumed_intent" if intent.consumed_at else "expired_intent"
                ),
            })
        capture_wakeup.set()
        return response.model_copy(
            update={
                "conversation_id": recording.event.conversation_id,
                "capture_event_id": recording.event.id,
                "capture_event_created": recording.created,
                "continuation_status": recording.event.continuation_status,
                "continuation_reason": None,
            }
        )

    def _job_capture_store() -> JobCaptureStore:
        store: JobCaptureStore | None = application.state.job_capture_store
        if store is None:
            store = capture_store_factory()
            application.state.job_capture_store = store
        return store

    def record_capture(
        *,
        user_id: str,
        conversation_id: str | None,
        capture_event_id: str | None,
        phase: Literal["intent", "settled"],
        status: str,
        turn_id: str | None = None,
        error_code: str | None = None,
    ) -> None:
        # Best-effort: the save and the settlement are already durable, and a
        # trace that cannot be written must not turn either into a 500.
        runtime: MainAgentRuntime | None = application.state.runtime
        try:
            if runtime is not None:
                runtime.record_capture_continuation(
                    user_id=user_id, conversation_id=conversation_id,
                    capture_event_id=capture_event_id, phase=phase, status=status,
                    turn_id=turn_id, error_code=error_code,
                )
        except Exception:
            return

    def settle_capture(
        event: JobCapturedEvent,
        status: Literal["completed", "discarded", "failed", "expired"],
        *,
        turn_id: str | None = None,
        error_code: str | None = None,
    ) -> None:
        _job_capture_store().settle_continuation(
            user_id=event.user_id, event_id=event.id, status=status, turn_id=turn_id,
        )
        record_capture(
            user_id=event.user_id, conversation_id=event.conversation_id,
            capture_event_id=event.id, phase="settled", status=status,
            turn_id=turn_id, error_code=error_code,
        )

    async def run_capture(event: JobCapturedEvent) -> None:
        gate: ConversationRunGate = application.state.run_gate
        turn_id: str | None = None
        failure_code: str | None = None
        try:
            async for output in astream_turn_events(
                application.state.runtime,
                user_id=event.user_id,
                conversation_id=event.conversation_id,
                user_message=capture_follow_up_message(event),
                request_id=event.id,
                input_resources=(TurnInputResource(kind="jd_snapshot", id=event.jd_snapshot_id),),
            ):
                if isinstance(output, (TurnCompletedEvent, TurnSuspendedEvent)):
                    turn_id = output.turn_id
                elif isinstance(output, TurnFailedEvent):
                    failure_code = output.code
            if turn_id:
                settle_capture(event, "completed", turn_id=turn_id)
            else:
                settle_capture(
                    event, "failed", error_code=failure_code or "CAPTURE_TURN_NOT_COMMITTED",
                )
        except Exception as error:
            logging.getLogger(__name__).warning(
                "Capture continuation failed: %s", type(error).__name__
            )
            settle_capture(event, "failed", error_code=type(error).__name__)
        finally:
            await gate.release(event.user_id, event.conversation_id)
            capture_wakeup.set()

    def capture_ready(event: JobCapturedEvent) -> bool | None:
        """``None``: the conversation is gone. ``False``: it is waiting on the user."""
        if not conversation_accepts_capture(event.user_id, event.conversation_id):
            return None
        return application.state.runtime.accepts_background_turn(
            user_id=event.user_id, conversation_id=event.conversation_id,
        )

    async def dispatch_pending_captures() -> None:
        store = _job_capture_store()
        gate: ConversationRunGate = application.state.run_gate
        for event in store.expire_continuations():
            record_capture(
                user_id=event.user_id, conversation_id=event.conversation_id,
                capture_event_id=event.id, phase="settled", status="expired",
            )
        for event in store.list_pending_continuations():
            # Checked before taking the gate, so a conversation that stays
            # busy or waiting on the user is not briefly locked every pass,
            # which would refuse the user's own message with a 409.
            ready = capture_ready(event)
            if ready is None:
                settle_capture(event, "discarded")
                continue
            if not ready or gate.is_active(event.user_id, event.conversation_id):
                continue
            try:
                await gate.acquire(event.user_id, event.conversation_id)
            except (ConversationBusyError, TurnCapacityError):
                continue
            handed_off = False
            try:
                # Re-checked under the gate: the state may have changed between.
                ready = capture_ready(event)
                if ready is None:
                    settle_capture(event, "discarded")
                elif ready:
                    worker = asyncio.create_task(run_capture(event))
                    capture_workers.add(worker)
                    worker.add_done_callback(capture_workers.discard)
                    handed_off = True
            finally:
                if not handed_off:
                    await gate.release(event.user_id, event.conversation_id)

    async def dispatch_captures() -> None:
        while True:
            capture_wakeup.clear()
            if application.state.runtime is not None:
                try:
                    await dispatch_pending_captures()
                except Exception as error:
                    logging.getLogger(__name__).warning(
                        "Capture dispatch failed: %s", type(error).__name__
                    )
            try:
                await asyncio.wait_for(capture_wakeup.wait(), timeout=5)
            except TimeoutError:
                pass

    @application.get(
        "/v1/job-captures/events",
        response_model=JobCapturedEventsResponse,
    )
    async def pending_job_captures(
        principal: ApiKeyPrincipal = Depends(require_scope(WORKSPACE_READ)),
        conversation_id: str | None = None,
    ) -> JobCapturedEventsResponse:
        """Durable capture deliveries awaiting presentation in the web client."""
        events = _job_capture_store().list_pending_events(
            user_id=principal.user_id,
            conversation_id=conversation_id or None,
        )
        pending: list[JobCapturedEventView] = []
        for event in events:
            if conversation_accepts_capture(principal.user_id, event.conversation_id):
                pending.append(JobCapturedEventView.from_event(event))
            else:
                _job_capture_store().acknowledge_event(
                    user_id=principal.user_id, event_id=event.id
                )
        return JobCapturedEventsResponse(events=tuple(pending))

    @application.post(
        "/v1/job-captures/events/{event_id}/retry",
        response_model=JobCaptureRetryResponse,
    )
    async def retry_job_capture(
        event_id: str,
        principal: ApiKeyPrincipal = Depends(require_scope(CHAT_WRITE)),
    ) -> JobCaptureRetryResponse:
        retried = _job_capture_store().retry_continuation(
            user_id=principal.user_id, event_id=event_id,
        )
        if retried:
            capture_wakeup.set()
        return JobCaptureRetryResponse(event_id=event_id, retried=retried)

    @application.post(
        "/v1/job-captures/events/{event_id}/ack",
        response_model=JobCapturedEventAckResponse,
    )
    async def acknowledge_job_capture(
        event_id: str,
        principal: ApiKeyPrincipal = Depends(require_scope(CHAT_WRITE)),
    ) -> JobCapturedEventAckResponse:
        acknowledged = _job_capture_store().acknowledge_event(
            user_id=principal.user_id,
            event_id=event_id,
        )
        return JobCapturedEventAckResponse(event_id=event_id, acknowledged=acknowledged)

    return application


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run(
        "career_agent.api.app:app",
        host="127.0.0.1",
        port=int(os.environ.get("CAREER_AGENT_API_PORT", "8000")),
        reload=False,
        workers=1,
    )
