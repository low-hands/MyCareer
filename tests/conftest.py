"""Repository-wide test configuration."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_live_embedding_service(monkeypatch):
    """Keep deterministic tests off the embedding service configured in .env.

    Runtime code calls ``load_dotenv()``, which never overrides a variable that
    is already set. Present-but-empty values therefore read as "not configured"
    for the whole test, while the developer's .env stays untouched. A test that
    needs embedding settings sets them itself.
    """
    for name in (
        "CAREER_EMBEDDING_BASE_URL",
        "CAREER_EMBEDDING_API_KEY",
        "CAREER_EMBEDDING_MODEL",
    ):
        monkeypatch.setenv(name, "")
