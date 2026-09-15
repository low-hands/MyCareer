"""Whole-workspace backup and restore for the local SQLite stores.

A workspace is a handful of SQLite files plus the working-notes directory.
Losing one of them (disk failure, a mistaken ``rm``, a botched machine
migration) is the realistic failure for a single-user local deployment, so
the unit of backup is *everything at once*: a directory holding one
consistent copy of every store, a manifest with a checksum per file, and a
restore that refuses to touch the live workspace unless the whole set
verifies.

SQLite copies go through the online backup API rather than ``shutil.copy``:
a plain copy of a database in WAL mode can miss committed pages that still
live in ``-wal``, and the backup API produces a single self-contained file.

Each copy is consistent with itself, not with the others: the databases are
copied one after another, so a turn committing in between leaves the set
from two moments (an application recorded in one file, the conversation that
recorded it absent from another). The set is one point in time only when
nothing writes during the copy, which the CLI guarantees by holding the API's
single-worker lock. A copy taken without it is marked ``consistent_snapshot:
false`` in the manifest and reported as such by ``verify``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Literal, Optional


MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1
DATABASES_DIR = "databases"
DIRECTORIES_DIR = "directories"
# Scratch and lock files a store leaves beside its real content. Never worth
# carrying across: a stale ``-wal`` next to a restored database would be
# replayed into it on open and corrupt the copy that just verified.
_SKIPPED_SUFFIXES = (".tmp", ".lock", "-wal", "-shm", "-journal")


class BackupError(Exception):
    pass


@dataclass(frozen=True)
class BackupPlan:
    """What a workspace consists of. Paths that do not exist yet are skipped
    at backup time and left alone at restore time."""

    databases: tuple[Path, ...]
    directories: tuple[Path, ...]

    def __post_init__(self) -> None:
        names = [path.name for path in self.databases] + [
            path.name for path in self.directories
        ]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise BackupError(
                "workspace paths must have distinct file names: "
                + ", ".join(duplicates)
            )


@dataclass(frozen=True)
class BackupEntry:
    name: str
    kind: Literal["sqlite", "file"]
    source: str
    size: int
    sha256: str

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "source": self.source,
            "size": self.size,
            "sha256": self.sha256,
        }

    @classmethod
    def from_json(cls, payload: object) -> "BackupEntry":
        if not isinstance(payload, dict):
            raise BackupError("manifest entry is not an object")
        try:
            kind = payload["kind"]
            if kind not in ("sqlite", "file"):
                raise BackupError(f"manifest entry has unknown kind {kind!r}")
            return cls(
                name=str(payload["name"]),
                kind=kind,
                source=str(payload["source"]),
                size=int(payload["size"]),
                sha256=str(payload["sha256"]),
            )
        except KeyError as error:
            raise BackupError(f"manifest entry is missing {error.args[0]!r}") from error


@dataclass(frozen=True)
class BackupManifest:
    created_at: datetime
    entries: tuple[BackupEntry, ...]
    consistent_snapshot: Optional[bool] = True
    """Whether every entry comes from the same moment. False when the copy was
    taken while the API could still write, so files may disagree with each
    other even though each passes its own integrity check. None when the
    manifest predates the flag: those backups could be taken with the API
    running, so nothing is known about them either way."""
    version: int = MANIFEST_VERSION

    def to_json(self) -> dict[str, object]:
        return {
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "consistent_snapshot": self.consistent_snapshot,
            "entries": [entry.to_json() for entry in self.entries],
        }

    @classmethod
    def from_json(cls, payload: object) -> "BackupManifest":
        if not isinstance(payload, dict):
            raise BackupError("manifest is not an object")
        version = payload.get("version")
        if version != MANIFEST_VERSION:
            raise BackupError(
                f"manifest version {version!r} is not supported (expected {MANIFEST_VERSION})"
            )
        entries = payload.get("entries")
        if not isinstance(entries, list):
            raise BackupError("manifest entries must be a list")
        try:
            created_at = datetime.fromisoformat(str(payload["created_at"]))
        except (KeyError, ValueError) as error:
            raise BackupError("manifest created_at is missing or invalid") from error
        consistent = payload.get("consistent_snapshot")
        if consistent is not None and not isinstance(consistent, bool):
            raise BackupError("manifest consistent_snapshot must be a boolean")
        return cls(
            created_at=created_at,
            entries=tuple(BackupEntry.from_json(entry) for entry in entries),
            consistent_snapshot=consistent,
        )


@dataclass(frozen=True)
class VerificationReport:
    directory: Path
    manifest: BackupManifest | None
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def warnings(self) -> tuple[str, ...]:
        """Findings that do not fail verification but a restorer should know.
        A backup that is not one point in time is still every byte it was
        written as, so it verifies; it just may not agree with itself."""
        if self.manifest is None:
            return ()
        if self.manifest.consistent_snapshot is None:
            return (UNKNOWN_SNAPSHOT_WARNING,)
        if not self.manifest.consistent_snapshot:
            return (INCONSISTENT_SNAPSHOT_WARNING,)
        return ()


INCONSISTENT_SNAPSHOT_WARNING = (
    "backup was taken while the API could still write; its databases may not "
    "be from the same moment"
)
UNKNOWN_SNAPSHOT_WARNING = (
    "backup manifest predates the consistent_snapshot flag; whether its "
    "databases are from the same moment is unknown"
)


@dataclass(frozen=True)
class RestoreReport:
    restored: tuple[str, ...]
    skipped_missing_target: tuple[str, ...]
    safety_copy: Path | None
    warnings: tuple[str, ...] = ()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_skipped(path: Path) -> bool:
    return path.name.endswith(_SKIPPED_SUFFIXES)


def _sqlite_check(path: Path) -> str | None:
    """Return None when SQLite considers the file sound, else its complaint."""

    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as error:
        return str(error)
    try:
        rows = connection.execute("PRAGMA integrity_check").fetchall()
    except sqlite3.Error as error:
        return str(error)
    finally:
        connection.close()
    if rows == [("ok",)]:
        return None
    return "; ".join(str(row[0]) for row in rows) or "integrity_check returned nothing"


def _copy_sqlite(source: Path, destination: Path) -> None:
    try:
        src = sqlite3.connect(source)
    except sqlite3.Error as error:
        raise BackupError(f"cannot open {source}: {error}") from error
    try:
        dst = sqlite3.connect(destination)
        try:
            src.backup(dst)
            # The copy inherits the source's WAL flag. A backup is a single
            # self-contained file, so switch it to rollback journaling: no
            # ``-wal``/``-shm`` appear beside it when it is later checked.
            dst.execute("PRAGMA journal_mode=DELETE")
        except sqlite3.Error as error:
            raise BackupError(f"{source} is not a readable SQLite database: {error}") from error
        finally:
            dst.close()
    finally:
        src.close()


def _make_private_dirs(path: Path, *, stop_at: Path) -> None:
    missing: list[Path] = []
    current = path
    while current != stop_at and not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir()
        os.chmod(directory, 0o700)


def _directory_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink() and not _is_skipped(path):
            files.append(path)
    return files


def create_backup(
    plan: BackupPlan, destination: Path, *, consistent_snapshot: bool = True
) -> BackupManifest:
    """Write one copy of every existing workspace path into ``destination``,
    which must not exist yet or must be an empty directory.

    The caller says with ``consistent_snapshot`` whether it has made sure
    nothing writes to the workspace meanwhile (the CLI holds the API lock);
    this function only copies and records the answer in the manifest.
    """

    destination = destination.expanduser()
    if destination.exists():
        if not destination.is_dir():
            raise BackupError(f"{destination} exists and is not a directory")
        if any(destination.iterdir()):
            raise BackupError(f"{destination} is not empty")
    destination.mkdir(parents=True, exist_ok=True)
    os.chmod(destination, 0o700)

    entries: list[BackupEntry] = []
    for database in plan.databases:
        database = database.expanduser()
        if not database.exists():
            continue
        target = destination / DATABASES_DIR / database.name
        _make_private_dirs(target.parent, stop_at=destination)
        _copy_sqlite(database, target)
        os.chmod(target, 0o600)
        complaint = _sqlite_check(target)
        if complaint is not None:
            raise BackupError(f"backup copy of {database} failed integrity_check: {complaint}")
        entries.append(
            BackupEntry(
                name=f"{DATABASES_DIR}/{database.name}",
                kind="sqlite",
                source=str(database),
                size=target.stat().st_size,
                sha256=_sha256(target),
            )
        )

    for directory in plan.directories:
        directory = directory.expanduser()
        if not directory.is_dir():
            continue
        for file in _directory_files(directory):
            relative = file.relative_to(directory)
            target = destination / DIRECTORIES_DIR / directory.name / relative
            _make_private_dirs(target.parent, stop_at=destination)
            shutil.copyfile(file, target)
            os.chmod(target, 0o600)
            entries.append(
                BackupEntry(
                    name=f"{DIRECTORIES_DIR}/{directory.name}/{relative.as_posix()}",
                    kind="file",
                    source=str(file),
                    size=target.stat().st_size,
                    sha256=_sha256(target),
                )
            )

    manifest = BackupManifest(
        created_at=datetime.now(timezone.utc).replace(microsecond=0),
        entries=tuple(entries),
        consistent_snapshot=consistent_snapshot,
    )
    manifest_path = destination / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(manifest.to_json(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.chmod(manifest_path, 0o600)
    return manifest


def read_manifest(directory: Path) -> BackupManifest:
    manifest_path = directory.expanduser() / MANIFEST_NAME
    if not manifest_path.is_file():
        raise BackupError(f"{manifest_path} not found")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise BackupError(f"{manifest_path} is not readable JSON: {error}") from error
    return BackupManifest.from_json(payload)


def verify_backup(directory: Path) -> VerificationReport:
    """Check every manifest entry is present, byte-identical to when it was
    written, and (for databases) still passes SQLite's own integrity check."""

    directory = directory.expanduser()
    try:
        manifest = read_manifest(directory)
    except BackupError as error:
        return VerificationReport(directory=directory, manifest=None, problems=(str(error),))

    problems: list[str] = []
    for entry in manifest.entries:
        path = directory / entry.name
        if not path.is_file():
            problems.append(f"{entry.name}: missing")
            continue
        size = path.stat().st_size
        if size != entry.size:
            problems.append(f"{entry.name}: size {size} != {entry.size}")
            continue
        if _sha256(path) != entry.sha256:
            problems.append(f"{entry.name}: sha256 mismatch")
            continue
        if entry.kind == "sqlite":
            complaint = _sqlite_check(path)
            if complaint is not None:
                problems.append(f"{entry.name}: integrity_check failed: {complaint}")
    return VerificationReport(directory=directory, manifest=manifest, problems=tuple(problems))


def _replace_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(target.parent, 0o700)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        os.chmod(temporary, 0o600)
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def restore_backup(
    directory: Path,
    plan: BackupPlan,
    *,
    safety_copy_dir: Path | None,
) -> RestoreReport:
    """Replace the workspace described by ``plan`` with the backup in
    ``directory``.

    Entries are matched to the current plan by file name, so a backup taken
    under another home directory restores into this one. The backup is
    verified first and nothing is touched if it fails. Before overwriting, the
    current workspace is itself backed up into ``safety_copy_dir`` (unless
    None), so a restore of the wrong snapshot is one more restore away from
    undone.
    """

    directory = directory.expanduser()
    report = verify_backup(directory)
    if report.manifest is None or not report.ok:
        raise BackupError(
            "backup failed verification; nothing restored: " + "; ".join(report.problems)
        )
    manifest = report.manifest

    databases = {path.name: path.expanduser() for path in plan.databases}
    directories = {path.name: path.expanduser() for path in plan.directories}

    planned: list[tuple[BackupEntry, Path]] = []
    skipped: list[str] = []
    restored_dirs: set[Path] = set()
    for entry in manifest.entries:
        parts = entry.name.split("/")
        if entry.kind == "sqlite" and len(parts) == 2 and parts[0] == DATABASES_DIR:
            target = databases.get(parts[1])
        elif entry.kind == "file" and len(parts) >= 3 and parts[0] == DIRECTORIES_DIR:
            root = directories.get(parts[1])
            target = None if root is None else root.joinpath(*parts[2:])
            if root is not None:
                restored_dirs.add(root)
        else:
            raise BackupError(f"manifest entry {entry.name!r} has an unexpected layout")
        if target is None:
            skipped.append(entry.name)
            continue
        planned.append((entry, target))

    safety_copy: Path | None = None
    if safety_copy_dir is not None:
        create_backup(plan, safety_copy_dir)
        safety_copy = safety_copy_dir.expanduser()

    # Files inside a restored directory that the snapshot does not know about
    # would otherwise survive and make the result a merge rather than a
    # restore. The safety copy above already holds them.
    wanted: set[Path] = {target for _, target in planned}
    for root in restored_dirs:
        if root.is_dir():
            for file in _directory_files(root):
                if file not in wanted:
                    file.unlink()

    restored: list[str] = []
    for entry, target in planned:
        _replace_file(directory / entry.name, target)
        if entry.kind == "sqlite":
            for suffix in ("-wal", "-shm", "-journal"):
                target.with_name(target.name + suffix).unlink(missing_ok=True)
            complaint = _sqlite_check(target)
            if complaint is not None:
                raise BackupError(
                    f"restored {target} failed integrity_check: {complaint}"
                )
        restored.append(entry.name)

    return RestoreReport(
        restored=tuple(restored),
        skipped_missing_target=tuple(skipped),
        safety_copy=safety_copy,
        warnings=report.warnings,
    )
