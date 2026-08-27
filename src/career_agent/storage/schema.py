"""Per-component schema versioning for shared SQLite files.

Several store classes write tables into one file: seven share `resumes.sqlite3`
and two share `applications.sqlite3`. SQLite's `PRAGMA user_version` is a single
integer per *file*, so it cannot describe them. Today only `ResumeStore` sets it,
which means the number silently claims to describe a file that six other stores
also own.

This registry gives each component its own row, and refuses to open a file whose
component version is newer than the running code expects. Without that check, an
older build opening a newer file finds its `CREATE TABLE IF NOT EXISTS` calls
satisfied, reads columns that have since changed meaning, and writes rows the
newer build considers malformed. Failing on open turns silent corruption into a
startup error.

`apply_schema` enforces the ordering that makes the check worth having:

    read the recorded version
    → refuse anything newer than this build
    → adopt a new/unregistered component, or run pending upgrades oldest first
    → run the idempotent current-shape baseline
    → record the version they reached

The check has to come *first*. Recording after a migration that already ran would
let a newer file raise an ordinary `OperationalError` on a changed column before
the version check was ever consulted, so the failure would look like a corrupt
database rather than an out-of-date build.

The baseline always runs. Every one is a cumulative set of
`CREATE TABLE IF NOT EXISTS` statements, so re-running it is free and repairs a
file left half-built by an interrupted first start — a version recorded as current
must not stop that repair. For an unregistered component (`found == 0`) that
baseline adopts the database directly at the current version; historical,
non-idempotent upgrades must not be replayed over the current shape. For an
already registered older component, upgrades run first and the baseline repairs
the resulting current shape afterward.

A version raised without a matching migration is caught by
`tests/storage/test_schema_version.py`, not at runtime: these numbers predate the
registry, so the code cannot derive them from the migrations it holds.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import sqlite3


class SchemaVersionError(RuntimeError):
    """A store met a schema newer than it knows how to read."""

    def __init__(self, component: str, found: int, supported: int) -> None:
        super().__init__(
            f"{component} schema is at version {found} but this build supports "
            f"{supported}. Upgrade the application instead of writing to this file."
        )
        self.component = component
        self.found = found
        self.supported = supported


def check_schema_version(
    connection: sqlite3.Connection, component: str, supported: int
) -> int:
    """Read the recorded version, refusing anything newer than this build.

    Call before any migration statement runs. Returns the version found, so a
    caller can tell a fresh file (0) from one it has to upgrade in place.
    """
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_versions (
            component TEXT PRIMARY KEY,
            version INTEGER NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    row = connection.execute(
        "SELECT version FROM schema_versions WHERE component = ?", (component,)
    ).fetchone()
    found = int(row[0]) if row else 0
    if found > supported:
        raise SchemaVersionError(component, found, supported)
    return found


def record_schema_version(
    connection: sqlite3.Connection, component: str, version: int
) -> None:
    """Record the version a migration just reached.

    Call only after the migration succeeded. Recording first would mark a
    half-applied schema as complete, and the next start would skip the repair.
    """
    connection.execute(
        """
        INSERT INTO schema_versions(component, version, updated_at)
        VALUES (?, ?, datetime('now'))
        ON CONFLICT(component) DO UPDATE
            SET version = excluded.version, updated_at = excluded.updated_at
        """,
        (component, version),
    )


def apply_schema(
    connection: sqlite3.Connection,
    component: str,
    version: int,
    baseline: Callable[[sqlite3.Connection], None],
    upgrades: Mapping[int, Callable[[sqlite3.Connection], None]] | None = None,
) -> int:
    """Check, migrate, then record — in that order.

    ``baseline`` builds or repairs the current shape and always runs first; it must
    be idempotent. ``upgrades`` holds the one-way steps — backfills, drops,
    rewrites — keyed by the version that introduced them, and each runs after the
    baseline has put the tables it needs in place. They run only for a component
    already registered at an older version. A fresh or pre-registry component is
    adopted by its cumulative baseline without replaying historical operations.

    Returns the version found before this call, so a caller can tell a fresh file
    (0) from one it upgraded.
    """
    if upgrades and max(upgrades) > version:
        raise ValueError(
            f"{component} declares an upgrade to version {max(upgrades)} but reports "
            f"version {version}; the recorded version would understate the schema."
        )
    found = check_schema_version(connection, component, version)
    # Baseline before upgrades. A version that adds a table *and* backfills it
    # keeps both halves in the same release: the baseline creates the table, the
    # upgrade fills it. Running the upgrade first would hit a table that does not
    # exist yet. The reverse hazard does not exist — the baseline is only
    # `CREATE TABLE IF NOT EXISTS`, so it cannot touch data an upgrade still needs.
    baseline(connection)
    if 0 < found < version:
        pending_versions = tuple(range(found + 1, version + 1))
        missing = tuple(
            step_version
            for step_version in pending_versions
            if step_version not in (upgrades or {})
        )
        if missing:
            raise ValueError(
                f"{component} cannot advance from version {found} to {version}: "
                f"missing upgrades for versions {missing}."
            )
        for step_version in pending_versions:
            upgrades[step_version](connection)
    if found != version:
        record_schema_version(connection, component, version)
    return found
