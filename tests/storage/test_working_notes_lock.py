import errno
from types import SimpleNamespace

import pytest

from career_agent.storage import working_notes
from career_agent.storage.working_notes import WorkingNotesStore


def _windows_lock(monkeypatch, locking, sleeps):
    monkeypatch.setattr(working_notes, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(working_notes, "msvcrt", SimpleNamespace(
        locking=locking, LK_LOCK=1, LK_UNLCK=0,
    ), raising=False)
    monkeypatch.setattr(working_notes, "time", SimpleNamespace(sleep=sleeps.append))


@pytest.mark.parametrize("lock_errno", [errno.EACCES, errno.EAGAIN, errno.EDEADLK])
def test_windows_lock_retries_contention_then_saves_and_unlocks(tmp_path, monkeypatch, lock_errno):
    notes = WorkingNotesStore(tmp_path / "notes")
    calls, sleeps = [], []

    def locking(fd, mode, size):
        calls.append((mode, size))
        if mode == 1 and len(calls) <= 2:
            raise OSError(lock_errno, "another writer holds the lock")

    _windows_lock(monkeypatch, locking, sleeps)
    result = notes.replace(user_id="u1", markdown="新笔记", expected_revision="empty")
    assert result.markdown == "新笔记"
    assert notes.read(user_id="u1").markdown == "新笔记"
    assert calls == [(1, 1), (1, 1), (1, 1), (0, 1)]
    assert sleeps == [0.1, 0.1]


def test_windows_lock_does_not_retry_unrelated_io_errors(tmp_path, monkeypatch):
    notes = WorkingNotesStore(tmp_path / "notes")
    calls, sleeps = [], []

    def locking(fd, mode, size):
        calls.append(mode)
        raise OSError(errno.EBADF, "invalid descriptor")

    _windows_lock(monkeypatch, locking, sleeps)
    with pytest.raises(OSError) as error:
        notes.replace(user_id="u1", markdown="不应写入", expected_revision="empty")
    assert error.value.errno == errno.EBADF
    assert calls == [1]
    assert sleeps == []
    assert notes.read(user_id="u1").revision == "empty"


def test_windows_lock_is_released_when_the_write_fails(tmp_path, monkeypatch):
    notes = WorkingNotesStore(tmp_path / "notes")
    calls = []
    _windows_lock(monkeypatch, lambda fd, mode, size: calls.append(mode), [])
    with pytest.raises(RuntimeError, match="write failed"):
        with notes._locked(notes._path("u1")):
            raise RuntimeError("write failed")
    assert calls == [1, 0]
