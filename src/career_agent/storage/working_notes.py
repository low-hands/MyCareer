from __future__ import annotations

import hashlib
import os
from pathlib import Path


WORKING_NOTES_MAX_CHARS = 2000


class WorkingNotesStore:
    """Per-user markdown scratchpads whose content is intentionally non-authoritative."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)

    def read(self, *, user_id: str) -> str:
        path = self._path(user_id)
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8")
        if len(text) > WORKING_NOTES_MAX_CHARS:
            raise ValueError("working notes exceed the 2000-character safety limit")
        return text

    def replace(self, *, user_id: str, markdown: str) -> str:
        if len(markdown) > WORKING_NOTES_MAX_CHARS:
            raise ValueError("working notes exceed the 2000-character safety limit")
        path = self._path(user_id)
        if not markdown:
            if path.exists():
                path.unlink()
            return ""
        temporary = path.with_suffix(".tmp")
        temporary.write_text(markdown, encoding="utf-8")
        os.chmod(temporary, 0o600)
        temporary.replace(path)
        return markdown

    def clear(self, *, user_id: str) -> bool:
        """Delete the whole unbound scratchpad after any memory tombstone."""

        path = self._path(user_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def _path(self, user_id: str) -> Path:
        if not user_id.strip():
            raise ValueError("user_id is required")
        digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.md"
