from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
import hashlib
import json
import sqlite3
import unicodedata
from uuid import uuid4

from career_agent.domain.intent_memory import IntentMemoryVersion

# The same schema is instantiated in context.sqlite3 for person intent and in
# resumes.sqlite3 for role intent. M6b aggregation must read both stores.


def apply_intent_version_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS career_intent_versions (
            update_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            scope_key TEXT NOT NULL,
            value TEXT NOT NULL,
            content_digest TEXT NOT NULL,
            revision INTEGER NOT NULL CHECK (revision >= 1),
            valid_from TEXT NOT NULL,
            superseded_at TEXT,
            superseded_by TEXT REFERENCES career_intent_versions(update_id)
                DEFERRABLE INITIALLY DEFERRED,
            source TEXT NOT NULL,
            UNIQUE(user_id, scope_key, revision),
            CHECK (
                (superseded_at IS NULL AND superseded_by IS NULL)
                OR (superseded_at IS NOT NULL AND superseded_by IS NOT NULL)
            )
        );

        CREATE UNIQUE INDEX IF NOT EXISTS career_intent_versions_active_idx
            ON career_intent_versions(user_id, scope_key)
            WHERE superseded_at IS NULL;
        CREATE INDEX IF NOT EXISTS career_intent_versions_history_idx
            ON career_intent_versions(user_id, scope_key, revision DESC);
        """
    )


def append_intent_version(
    connection: sqlite3.Connection,
    *,
    user_id: str,
    scope_key: str,
    value: str,
    source: str,
    valid_from: datetime | None = None,
) -> IntentMemoryVersion:
    normalized = value.strip()
    if not normalized:
        raise ValueError("intent version value is required")
    digest = intent_content_digest(normalized)
    current_row = connection.execute(
        _SELECT
        + """
        WHERE user_id = ? AND scope_key = ? AND superseded_at IS NULL
        """,
        (user_id, scope_key),
    ).fetchone()
    if current_row is not None:
        current = _version(current_row)
        if current.content_digest == digest:
            return current
        revision = current.revision + 1
    else:
        latest = connection.execute(
            """
            SELECT COALESCE(MAX(revision), 0)
            FROM career_intent_versions
            WHERE user_id = ? AND scope_key = ?
            """,
            (user_id, scope_key),
        ).fetchone()
        revision = int(latest[0]) + 1

    now = valid_from or datetime.now(timezone.utc)
    version = IntentMemoryVersion(
        update_id=f"intent_update_{uuid4().hex}",
        user_id=user_id,
        scope_key=scope_key,
        value=normalized,
        content_digest=digest,
        revision=revision,
        valid_from=now,
        source=source,
    )
    if current_row is not None:
        connection.execute(
            """
            UPDATE career_intent_versions
            SET superseded_at = ?, superseded_by = ?
            WHERE update_id = ?
            """,
            (now.isoformat(), version.update_id, current.update_id),
        )
    connection.execute(
        """
        INSERT INTO career_intent_versions(
            update_id, user_id, scope_key, value, content_digest, revision,
            valid_from, superseded_at, superseded_by, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?)
        """,
        (
            version.update_id,
            version.user_id,
            version.scope_key,
            version.value,
            version.content_digest,
            version.revision,
            version.valid_from.isoformat(),
            version.source,
        ),
    )
    return version


def list_intent_versions(
    connection: sqlite3.Connection,
    *,
    user_id: str,
    scope_key: str | None = None,
    scope_keys: Sequence[str] | None = None,
    active_only: bool = False,
    limit: int | None = None,
) -> tuple[IntentMemoryVersion, ...]:
    if scope_key is not None and scope_keys is not None:
        raise ValueError("use either scope_key or scope_keys, not both")
    if limit is not None and limit < 1:
        raise ValueError("intent version limit must be positive")
    where = "WHERE user_id = ?"
    parameters: list[object] = [user_id]
    if scope_key is not None:
        where += " AND scope_key = ?"
        parameters.append(scope_key)
    elif scope_keys is not None:
        selected = tuple(dict.fromkeys(scope_keys))
        if not selected:
            return ()
        if len(selected) > 400:
            raise ValueError("at most 400 intent scopes may be read at once")
        where += " AND scope_key IN (" + ",".join("?" for _ in selected) + ")"
        parameters.extend(selected)
    if active_only:
        where += " AND superseded_at IS NULL"
    query = _SELECT + where + " ORDER BY scope_key, revision"
    if limit is not None:
        query += " LIMIT ?"
        parameters.append(limit)
    rows = connection.execute(query, parameters).fetchall()
    return tuple(_version(row) for row in rows)


def intent_content_digest(value: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", value).split())
    canonical = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


_SELECT = """
    SELECT update_id, user_id, scope_key, value, content_digest, revision,
           valid_from, superseded_at, superseded_by, source
    FROM career_intent_versions
"""


def _version(row: tuple[object, ...]) -> IntentMemoryVersion:
    return IntentMemoryVersion(
        update_id=row[0],
        user_id=row[1],
        scope_key=row[2],
        value=row[3],
        content_digest=row[4],
        revision=row[5],
        valid_from=row[6],
        superseded_at=row[7],
        superseded_by=row[8],
        source=row[9],
    )
