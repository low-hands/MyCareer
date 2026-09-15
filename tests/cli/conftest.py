from pathlib import Path

import pytest

from career_agent import cli
from career_agent.api.single_worker import SingleWorkerLock, lock_path_for


@pytest.fixture(autouse=True)
def isolated_workspace_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write commands run with the default ``--context-store`` would otherwise
    lock the developer's own ``~/.career-agent``."""

    lock_dir = tmp_path / "career-agent-home"
    monkeypatch.setattr(
        cli,
        "build_workspace_lock",
        lambda args: SingleWorkerLock(lock_path_for(lock_dir)),
    )
    return lock_dir
