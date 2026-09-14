"""A data directory admits one API process.

The conversation gate is per process, so these pin the only thing that keeps
two processes from sharing it: the second acquirer fails, loudly, at startup.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import textwrap

from fastapi.testclient import TestClient
import pytest

from career_agent.api.app import build_single_worker_lock, create_app
from career_agent.api.single_worker import (
    LOCK_FILE_NAME,
    SingleWorkerError,
    SingleWorkerLock,
    lock_path_for,
    refuse_multi_worker_configuration,
)


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
