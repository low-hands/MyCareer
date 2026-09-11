from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


WORKING_NOTES_MAX_CHARS = 2000
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkingNotesSnapshot:
    markdown: str
    revision: str
    clipped: bool


@dataclass(frozen=True)
class WorkingNotesConflict:
    current: WorkingNotesSnapshot


class WorkingNotesStore:
    """Per-user markdown scratchpads whose content is intentionally non-authoritative."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)

    def read(self, *, user_id: str) -> WorkingNotesSnapshot:
        return self._read_path(self._path(user_id))

    def replace(
        self,
        *,
        user_id: str,
        markdown: str,
        expected_revision: str,
    ) -> WorkingNotesSnapshot | WorkingNotesConflict:
        if len(markdown) > WORKING_NOTES_MAX_CHARS:
            raise ValueError("working notes exceed the 2000-character safety limit")
        path = self._path(user_id)
        with self._locked(path):
            current = self._read_path(path)
            if current.revision != expected_revision:
                return WorkingNotesConflict(current=current)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(markdown, encoding="utf-8")
            os.chmod(temporary, 0o600)
            temporary.replace(path)
            return self._snapshot(markdown)

    def clear(self, *, user_id: str) -> bool:
        """Delete the whole unbound scratchpad after any memory tombstone."""

        path = self._path(user_id)
        with self._locked(path):
            if not path.exists():
                return False
            path.unlink()
            return True

    def _path(self, user_id: str) -> Path:
        if not user_id.strip():
            raise ValueError("user_id is required")
        digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.md"

    @contextmanager
    def _locked(self, path: Path) -> Iterator[None]:
        lock_path = path.with_suffix(".lock")
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        with os.fdopen(lock_fd, "a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            yield

    def _read_path(self, path: Path) -> WorkingNotesSnapshot:
        try:
            markdown = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return WorkingNotesSnapshot(markdown="", revision="empty", clipped=False)
        except (OSError, UnicodeDecodeError) as error:
            logger.warning(
                "Unable to read working notes; projecting an empty scratchpad",
                extra={"path": str(path), "error_type": type(error).__name__},
            )
            return WorkingNotesSnapshot(markdown="", revision="empty", clipped=False)
        return self._snapshot(markdown)

    @staticmethod
    def _snapshot(markdown: str) -> WorkingNotesSnapshot:
        revision = (
            "empty"
            if not markdown
            else hashlib.sha256(markdown.encode("utf-8")).hexdigest()[:12]
        )
        clipped = len(markdown) > WORKING_NOTES_MAX_CHARS
        return WorkingNotesSnapshot(
            markdown=markdown[:WORKING_NOTES_MAX_CHARS],
            revision=revision,
            clipped=clipped,
        )
