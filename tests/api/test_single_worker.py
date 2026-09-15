"""A data directory admits one API process.

The conversation gate is per process, so these pin the only thing that keeps
two processes from sharing it: the second acquirer fails, loudly, at startup.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import subprocess
import sys
import textwrap
import threading
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from career_agent.api.app import (
    DEFAULT_SHUTDOWN_DRAIN_SECONDS,
    ChatStreamRequest,
    ConversationRunGate,
    _sse_stream,
    build_single_worker_lock,
    create_app,
    shutdown_drain_seconds_from_env,
)
from career_agent.api.single_worker import (
    LOCK_FILE_NAME,
    SingleWorkerError,
    SingleWorkerLock,
    lock_path_for,
    refuse_multi_worker_configuration,
)
from career_agent.harness.streaming import TurnCompletedEvent, TurnStartedEvent


def test_second_acquirer_of_the_same_lock_file_is_refused(tmp_path: Path) -> None:
    first = SingleWorkerLock(lock_path_for(tmp_path))
    second = SingleWorkerLock(lock_path_for(tmp_path))

    first.acquire()
    try:
        with pytest.raises(SingleWorkerError) as refused:
            second.acquire()
        assert "单 worker" in str(refused.value)
        assert str(tmp_path / LOCK_FILE_NAME) in str(refused.value)
        assert not second.held
    finally:
        first.release()

    second.acquire()
    second.release()


def test_lock_is_released_when_the_holder_releases_or_exits(tmp_path: Path) -> None:
    lock = SingleWorkerLock(lock_path_for(tmp_path))
    with lock:
        assert lock.held
        assert (tmp_path / LOCK_FILE_NAME).read_text().strip().isdigit()
    assert not lock.held
    SingleWorkerLock(lock_path_for(tmp_path)).acquire()


def test_another_process_holding_the_lock_blocks_startup(tmp_path: Path) -> None:
    lock = SingleWorkerLock(lock_path_for(tmp_path))
    lock.acquire()
    script = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path
        from career_agent.api.single_worker import SingleWorkerError, SingleWorkerLock, lock_path_for
        try:
            SingleWorkerLock(lock_path_for(Path({str(tmp_path)!r}))).acquire()
        except SingleWorkerError as error:
            print(error)
            sys.exit(3)
        sys.exit(0)
        """
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    finally:
        lock.release()
    assert result.returncode == 3, result.stderr
    assert "另一个进程" in result.stdout or "进程" in result.stdout


def test_app_startup_takes_the_lock_and_a_second_app_cannot_start(tmp_path: Path, api_keys) -> None:
    def build(lock_dir: Path):
        return create_app(
            api_key_store_factory=lambda: api_keys,
            runtime_factory=lambda: None,
            single_worker_lock_factory=lambda: SingleWorkerLock(lock_path_for(lock_dir)),
        )

    with TestClient(build(tmp_path)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        with pytest.raises(SingleWorkerError):
            with TestClient(build(tmp_path)):
                pass

    with TestClient(build(tmp_path)) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_default_lock_sits_beside_the_conversation_store_not_the_key_store(
    tmp_path: Path, monkeypatch
) -> None:
    """The gate protects turn commits, so the lock follows the business databases."""

    monkeypatch.setenv("CAREER_AGENT_DATA_DIR", str(tmp_path / "keys-elsewhere"))
    home = tmp_path / "career-agent-home"
    args = argparse.Namespace(context_store=str(home / "context.sqlite3"))

    lock = build_single_worker_lock(args)

    assert lock.path == home / LOCK_FILE_NAME
    assert lock.path.parent != tmp_path / "keys-elsewhere"


def test_default_lock_expands_the_home_directory(monkeypatch) -> None:
    args = argparse.Namespace(context_store="~/.career-agent/context.sqlite3")
    assert build_single_worker_lock(args).path == Path("~/.career-agent").expanduser() / LOCK_FILE_NAME


def test_app_without_an_explicit_factory_uses_the_default_lock(
    tmp_path: Path, api_keys, isolated_single_worker_lock: Path
) -> None:
    app = create_app(api_key_store_factory=lambda: api_keys, runtime_factory=lambda: None)
    with TestClient(app):
        assert (isolated_single_worker_lock / LOCK_FILE_NAME).exists()
        with pytest.raises(SingleWorkerError):
            SingleWorkerLock(lock_path_for(isolated_single_worker_lock)).acquire()
    SingleWorkerLock(lock_path_for(isolated_single_worker_lock)).acquire()


def test_the_runtime_is_built_only_once_the_lock_is_held(tmp_path: Path, api_keys) -> None:
    """Startup recovery of orphaned RUNNING receipts lives in the runtime builder.

    It is only sound because, by the time the builder runs, this process owns
    the lock and so nothing else can still be executing a turn.
    """

    lock_path = lock_path_for(tmp_path)
    held_when_built: list[bool] = []

    def build_runtime():
        rival = SingleWorkerLock(lock_path)
        try:
            rival.acquire()
        except SingleWorkerError:
            held_when_built.append(True)
        else:  # pragma: no cover - the failure this test exists to catch
            rival.release()
            held_when_built.append(False)
        return None

    app = create_app(
        api_key_store_factory=lambda: api_keys,
        runtime_factory=build_runtime,
        single_worker_lock_factory=lambda: SingleWorkerLock(lock_path),
    )
    with TestClient(app):
        pass

    assert held_when_built == [True]


class DetachableRuntime:
    """A turn that keeps running after its SSE observer has gone."""

    def __init__(self, *, hold_seconds: float) -> None:
        self.hold_seconds = hold_seconds
        self.started = threading.Event()
        self.finished_at: float | None = None
        self.closed_at: float | None = None

    def run_turn(
        self,
        *,
        user_id,
        conversation_id,
        user_message,
        request_id=None,
        interaction_response=None,
        event_sink=None,
    ):
        assert event_sink is not None
        event_sink(TurnStartedEvent(turn_id="turn-1"))
        self.started.set()
        time.sleep(self.hold_seconds)
        event_sink(TurnCompletedEvent(turn_id="turn-1"))
        self.finished_at = time.monotonic()
        return object()

    def close(self) -> None:
        self.closed_at = time.monotonic()


async def _serve_then_shut_down_around_an_abandoned_turn(
    app: FastAPI, runtime: DetachableRuntime
) -> float:
    """Run the lifespan; inside it, start a turn whose observer hangs up.

    ``TestClient`` buffers a whole SSE body before returning, so a client that
    disconnects mid-turn cannot be expressed through it. This drives the same
    admission, stream and detached producer the route uses, then shuts down.
    Returns when the lifespan has finished.
    """

    async with app.router.lifespan_context(app):
        gate: ConversationRunGate = app.state.run_gate
        await gate.acquire("u1", "c1")

        async def release_gate() -> None:
            await gate.release("u1", "c1")

        stream = _sse_stream(
            runtime,
            ChatStreamRequest(conversation_id="c1", message="慢任务"),
            user_id="u1",
            heartbeat_seconds=1,
            on_turn_finished=release_gate,
        )
        assert "event: turn_started" in await anext(stream)
        await stream.aclose()
        assert runtime.started.is_set()
        assert runtime.finished_at is None
    return time.monotonic()


def test_shutdown_waits_for_a_detached_turn_before_releasing_the_lock(
    tmp_path: Path, api_keys
) -> None:
    """The lock tells the next process nothing here still runs a turn.

    A client that hangs up does not stop its turn, so a graceful restart must
    not hand the lock over while that thread is still executing: the new
    process would settle its still-live receipt as orphaned.
    """

    runtime = DetachableRuntime(hold_seconds=0.3)
    app = create_app(
        api_key_store_factory=lambda: api_keys,
        runtime_factory=lambda: runtime,
        single_worker_lock_factory=lambda: SingleWorkerLock(lock_path_for(tmp_path)),
        shutdown_drain_seconds=5,
    )

    shut_down_at = asyncio.run(_serve_then_shut_down_around_an_abandoned_turn(app, runtime))

    assert runtime.finished_at is not None
    assert runtime.closed_at is not None
    assert runtime.finished_at <= runtime.closed_at <= shut_down_at
    assert app.state.shutdown_drained is True
    assert not app.state.single_worker_lock.held
    SingleWorkerLock(lock_path_for(tmp_path)).acquire()


def test_a_turn_outliving_the_drain_keeps_the_lock_for_the_kernel_to_release(
    tmp_path: Path, api_keys
) -> None:
    """Past the drain budget the process still must not release the lock itself.

    Its thread is still executing; only process exit, when the kernel drops
    the lock, can truthfully say otherwise. Closing the stores under that
    thread is left to exit as well.
    """

    runtime = DetachableRuntime(hold_seconds=0.4)
    app = create_app(
        api_key_store_factory=lambda: api_keys,
        runtime_factory=lambda: runtime,
        single_worker_lock_factory=lambda: SingleWorkerLock(lock_path_for(tmp_path)),
        shutdown_drain_seconds=0.05,
    )

    shut_down_at = asyncio.run(_serve_then_shut_down_around_an_abandoned_turn(app, runtime))

    assert app.state.shutdown_drained is False
    assert runtime.closed_at is None
    assert app.state.single_worker_lock.held
    with pytest.raises(SingleWorkerError):
        SingleWorkerLock(lock_path_for(tmp_path)).acquire()
    # asyncio.run() waited for the worker thread on the way out, as the
    # interpreter would; the turn ended after shutdown, and the lock was kept.
    assert runtime.finished_at is not None
    assert runtime.finished_at >= shut_down_at
    app.state.single_worker_lock.release()


def test_shutdown_drain_budget_comes_from_the_environment() -> None:
    assert shutdown_drain_seconds_from_env({}) == DEFAULT_SHUTDOWN_DRAIN_SECONDS
    assert shutdown_drain_seconds_from_env({"CAREER_AGENT_SHUTDOWN_DRAIN_SECONDS": "2.5"}) == 2.5
    assert shutdown_drain_seconds_from_env({"CAREER_AGENT_SHUTDOWN_DRAIN_SECONDS": "0"}) == 0
    for bad in ("-1", "soon", "inf", "nan"):
        with pytest.raises(ValueError):
            shutdown_drain_seconds_from_env({"CAREER_AGENT_SHUTDOWN_DRAIN_SECONDS": bad})
    for bad_arg in (-0.1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            create_app(shutdown_drain_seconds=bad_arg)


def test_a_zero_drain_budget_still_releases_the_lock_when_nothing_is_running(
    tmp_path: Path, api_keys
) -> None:
    """``wait_for(timeout=0)`` times out even on an already-set event; an idle
    gate must count as drained regardless of the budget."""

    runtime = DetachableRuntime(hold_seconds=0)
    app = create_app(
        api_key_store_factory=lambda: api_keys,
        runtime_factory=lambda: runtime,
        single_worker_lock_factory=lambda: SingleWorkerLock(lock_path_for(tmp_path)),
        shutdown_drain_seconds=0,
    )

    with TestClient(app):
        pass

    assert app.state.shutdown_drained is True
    assert runtime.closed_at is not None
    assert not app.state.single_worker_lock.held
    SingleWorkerLock(lock_path_for(tmp_path)).acquire()


def test_lock_release_runs_even_when_the_runtime_fails_to_build(tmp_path: Path, api_keys) -> None:
    class Boom(Exception):
        pass

    def explode():
        raise Boom

    app = create_app(
        api_key_store_factory=lambda: api_keys,
        runtime_factory=explode,
        single_worker_lock_factory=lambda: SingleWorkerLock(lock_path_for(tmp_path)),
    )
    with pytest.raises(Boom):
        with TestClient(app):
            pass
    SingleWorkerLock(lock_path_for(tmp_path)).acquire()


@pytest.mark.parametrize("name", ["WEB_CONCURRENCY", "UVICORN_WORKERS", "GUNICORN_WORKERS"])
def test_multi_worker_environment_is_refused_with_a_readable_message(name: str) -> None:
    with pytest.raises(SingleWorkerError) as refused:
        refuse_multi_worker_configuration({name: "2"})
    assert name in str(refused.value)
    assert "单 worker" in str(refused.value)


@pytest.mark.parametrize("value", ["1", "", " ", "not-a-number"])
def test_single_or_unparseable_worker_environment_is_allowed(value: str) -> None:
    refuse_multi_worker_configuration({"WEB_CONCURRENCY": value})


def test_multi_worker_environment_stops_app_startup(tmp_path: Path, api_keys, monkeypatch) -> None:
    monkeypatch.setenv("WEB_CONCURRENCY", "4")
    app = create_app(
        api_key_store_factory=lambda: api_keys,
        runtime_factory=lambda: None,
        single_worker_lock_factory=lambda: SingleWorkerLock(lock_path_for(tmp_path)),
    )
    with pytest.raises(SingleWorkerError):
        with TestClient(app):
            pass
    assert not (tmp_path / LOCK_FILE_NAME).exists()
