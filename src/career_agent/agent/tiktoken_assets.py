"""Package-local cl100k_base vocab for tiktoken, filled at assemble/install.

tiktoken caches by ``sha1(blob_url)``, not a human-readable name. A file
called ``cl100k_base.tiktoken`` in this directory never hits, and tiktoken
then falls through to ``requests.get`` with no timeout.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
import uuid
from pathlib import Path

CL100K_BLOB_URL = (
    "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
)
CL100K_BLOB_SHA256 = (
    "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
)
DOWNLOAD_TIMEOUT_SECONDS = 30.0


def tiktoken_cache_key(blobpath: str) -> str:
    return hashlib.sha1(blobpath.encode()).hexdigest()


def bundled_tiktoken_cache_dir() -> Path:
    return Path(__file__).resolve().parent / "_static" / "tiktoken_cache"


def bundled_cl100k_vocab_path() -> Path:
    return bundled_tiktoken_cache_dir() / tiktoken_cache_key(CL100K_BLOB_URL)


def populate_bundled_tiktoken_cache(
    *,
    timeout: float = DOWNLOAD_TIMEOUT_SECONDS,
) -> Path:
    """Write cl100k_base into the package cache under its URL-sha1 name."""

    dest = bundled_cl100k_vocab_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if _valid_vocab(dest):
        return dest
    source = _existing_vocab_copy_source(dest)
    if source is not None:
        shutil.copy2(source, dest)
        if _valid_vocab(dest):
            return dest
        dest.unlink(missing_ok=True)
    _download_vocab(dest, timeout=timeout)
    return dest


def _valid_vocab(path: Path) -> bool:
    if not path.is_file():
        return False
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest == CL100K_BLOB_SHA256


def _existing_vocab_copy_source(dest: Path) -> Path | None:
    key = tiktoken_cache_key(CL100K_BLOB_URL)
    candidates: list[Path] = []
    for env_name in ("TIKTOKEN_CACHE_DIR", "DATA_GYM_CACHE_DIR"):
        raw = os.environ.get(env_name)
        if raw:
            candidates.append(Path(raw) / key)
    candidates.append(Path(tempfile.gettempdir()) / "data-gym-cache" / key)
    for candidate in candidates:
        if candidate.resolve() == dest.resolve():
            continue
        if _valid_vocab(candidate):
            return candidate
    return None


def _download_vocab(dest: Path, *, timeout: float) -> None:
    try:
        with urllib.request.urlopen(
            CL100K_BLOB_URL, timeout=timeout
        ) as response:
            data = response.read()
    except Exception as error:
        raise RuntimeError(
            "failed to download cl100k_base vocab into the package cache; "
            "tiktoken's own fetch has no timeout and would hang offline. "
            f"Cause: {error}"
        ) from error
    digest = hashlib.sha256(data).hexdigest()
    if digest != CL100K_BLOB_SHA256:
        raise ValueError(
            f"Hash mismatch for {CL100K_BLOB_URL} "
            f"(expected {CL100K_BLOB_SHA256}, got {digest})."
        )
    tmp_path = dest.with_name(dest.name + "." + uuid.uuid4().hex + ".tmp")
    tmp_path.write_bytes(data)
    tmp_path.replace(dest)


if __name__ == "__main__":
    print(populate_bundled_tiktoken_cache())
