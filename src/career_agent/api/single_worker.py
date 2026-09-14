"""One API process per data directory.

``ConversationRunGate`` keeps two turns of one conversation from overlapping,
but it lives in process memory: a second uvicorn worker, or a second API
instance pointed at the same databases, would carry its own empty gate and the
two could commit the same conversation concurrently. The local product is a
single SQLite deployment, so instead of a cross-process lease (TTL, fencing,
heartbeat and crash recovery all at once) the process takes an exclusive lock on
a file in the data directory for as long as it runs. A second process cannot
take the lock and fails at startup with a message that says so.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

LOCK_FILE_NAME = "api-server.lock"
_WORKER_ENV_VARS = ("WEB_CONCURRENCY", "UVICORN_WORKERS", "GUNICORN_WORKERS")


class SingleWorkerError(RuntimeError):
    """The API refused to start because another worker would share its databases."""


class SingleWorkerLock:
    """Exclusive, process-lifetime lock on one file in the data directory.

    The lock is advisory and held by the open file descriptor, so the kernel
    drops it when the process exits for any reason; nothing has to clean up
    after a crash. The file's contents are informational only (the holder's
    pid), never the source of truth for who holds the lock.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._handle = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def held(self) -> bool:
        return self._handle is not None

    def acquire(self) -> None:
        if self._handle is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self._path, "a+", encoding="utf-8")
        try:
            _lock_exclusive_nonblocking(handle)
        except OSError:
            holder = _read_holder(handle)
            handle.close()
            who = f"进程 {holder}" if holder else "另一个进程"
            raise SingleWorkerError(
                f"{who} 已持有 {self._path}，说明同一数据目录上已有一个 API 进程在运行。"
                "当前版本只支持单 worker：请使用 `uvicorn --workers 1`（默认值），"
                "不要以 `--workers 2+` 启动，也不要让多个 API 实例共用同一数据库。"
                "如果确认没有其他进程在运行，请检查是否有残留的 API 进程未退出。"
            ) from None
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        self._handle = handle

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            handle.seek(0)
            handle.truncate()
            handle.flush()
            _unlock(handle)
        finally:
            handle.close()

    def __enter__(self) -> SingleWorkerLock:
        self.acquire()
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()


def refuse_multi_worker_configuration(environ=os.environ) -> None:
    """Fail early with a readable message when the server was told to fork workers.

    ``uvicorn`` reads ``WEB_CONCURRENCY`` when ``--workers`` is not given, and
    gunicorn deployments commonly export the same variable. Catching it here
    turns "the second worker died acquiring a lock" into "you asked for two
    workers". It is a courtesy, not the guard: the lock still decides.
    """

    for name in _WORKER_ENV_VARS:
        raw = environ.get(name)
        if raw is None or not raw.strip():
            continue
        try:
            requested = int(raw)
        except ValueError:
            continue
        if requested > 1:
            raise SingleWorkerError(
                f"{name}={raw} 要求启动 {requested} 个 worker，但当前版本只支持单 worker："
                "多个 worker 会共用同一份 SQLite 数据库，会话级互斥只在单进程内生效。"
                "请把它设为 1 或删除，并用 `uvicorn --workers 1` 启动。"
            )


def lock_path_for(data_dir: Path) -> Path:
    return data_dir / LOCK_FILE_NAME


def _lock_exclusive_nonblocking(handle) -> None:
    if sys.platform == "win32":
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(handle) -> None:
    if sys.platform == "win32":
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_holder(handle) -> str:
    try:
        handle.seek(0)
        return handle.read().strip()
    except OSError:
        return ""
