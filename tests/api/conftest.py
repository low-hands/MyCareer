"""Credentials for the API tests, and the reason they cannot be skipped.

Every endpoint derives its user from an API key, so a test that does not present
one is testing the 401 path. Rather than let each test file grow its own way of
faking that, this issues real keys from a real store: the tests then exercise the
same verification the deployment does, and a change that breaks scope checking
fails here rather than in production.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from career_agent.api import app as app_module
from career_agent.api.single_worker import SingleWorkerLock, lock_path_for
from career_agent.storage.api_keys import (
    CAPTURE_WRITE,
    CHAT_WRITE,
    SETTINGS_WRITE,
    SQLiteApiKeyStore,
    WORKSPACE_READ,
)


@pytest.fixture(autouse=True)
def isolated_single_worker_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep the default single-worker lock out of the real ``~/.career-agent``.

    ``create_app`` without an explicit factory would otherwise lock the
    developer's own data directory, and two tests could not both hold it.
    """

    lock_dir = tmp_path / "career-agent-home"
    monkeypatch.setattr(
        app_module,
        "build_single_worker_lock",
        lambda args=None: SingleWorkerLock(lock_path_for(lock_dir)),
    )
    monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
    return lock_dir


@pytest.fixture
def api_keys(tmp_path: Path) -> SQLiteApiKeyStore:
    return SQLiteApiKeyStore(tmp_path / "api_keys.sqlite3")


@pytest.fixture
def issue_key(api_keys: SQLiteApiKeyStore):
    """Mint a key and return the header a client would send with it."""

    def issue(user_id: str = "u1", *scopes: str) -> dict[str, str]:
        granted = frozenset(scopes) or frozenset(
            {WORKSPACE_READ, CHAT_WRITE, CAPTURE_WRITE, SETTINGS_WRITE}
        )
        key = api_keys.issue(user_id=user_id, name="test", scopes=granted)
        return {"Authorization": f"Bearer {key.secret}"}

    return issue


@pytest.fixture
def auth(issue_key) -> dict[str, str]:
    """A full-scope key for "u1", the user most of these tests act as."""

    return issue_key("u1")
