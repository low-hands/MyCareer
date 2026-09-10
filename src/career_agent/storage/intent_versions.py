from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
import hashlib
import json
import re
import sqlite3
from typing import TYPE_CHECKING
import unicodedata
from uuid import uuid4

from career_agent.domain.intent_memory import (
    IntentAdmissionStatus,
    IntentCaptureAction,
    IntentLayer,
    IntentMemoryVersion,
    IntentTimescale,
)

if TYPE_CHECKING:
    from career_agent.services.intent_capture import (
        IntentCaptureCandidate,
        IntentCaptureDecision,
    )

# The same schema is instantiated in context.sqlite3 for person intent and in
# resumes.sqlite3 for role intent. M6b aggregation must read both stores.


def apply_intent_version_schema(connection: sqlite3.Connection) -> None:
    """Adopt or upgrade the shared intent ledger to its scoped temporal shape."""

    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' "
        "AND name = 'career_intent_versions'"
    ).fetchone()
    if exists is None:
        _create_intent_version_table(connection)
    else:
        columns = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info(career_intent_versions)"
            )
        }
        if not {
            "pref_scope",
            "timescale",
            "layer",
            "last_corroborated_at",
            "base_confidence",
            "admission_status",
            "capture_action",
        } <= columns:
            upgrade_intent_version_schema(connection)
    upgrade_intent_semantic_stance_schema(connection)
    _create_intent_version_indexes(connection)


def upgrade_intent_semantic_stance_schema(
    connection: sqlite3.Connection,
) -> None:
    columns = {
        str(row[1])
        for row in connection.execute(
            "PRAGMA table_info(career_intent_versions)"
        )
    }
    if "semantic_stance" not in columns:
        connection.execute(
            "ALTER TABLE career_intent_versions ADD COLUMN semantic_stance TEXT"
        )


def upgrade_intent_version_schema(connection: sqlite3.Connection) -> None:
    """Rebuild the old flat table so both UNIQUE constraints include scope."""

    columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(career_intent_versions)")
    }
    if "pref_scope" in columns:
        _create_intent_version_indexes(connection)
        return
    connection.executescript(
        """
        DROP INDEX IF EXISTS career_intent_versions_active_idx;
        DROP INDEX IF EXISTS career_intent_versions_history_idx;
        ALTER TABLE career_intent_versions
            RENAME TO career_intent_versions_flat_v1;
        """
    )
    _create_intent_version_table(connection)
    connection.execute(
        """
        INSERT INTO career_intent_versions(
            update_id, user_id, scope_key, pref_scope, value, content_digest,
            revision, valid_from, timescale, layer, last_corroborated_at,
            base_confidence, admission_status, capture_action,
            superseded_at, superseded_by, source
        )
        SELECT update_id, user_id, scope_key, 'global', value, content_digest,
               revision, valid_from, 'permanent', 'stable', valid_from,
               1.0, 'active', 'retain',
               superseded_at, superseded_by, source
        FROM career_intent_versions_flat_v1
        """
    )
    connection.execute("DROP TABLE career_intent_versions_flat_v1")
    _create_intent_version_indexes(connection)


def _create_intent_version_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS career_intent_versions (
            update_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            scope_key TEXT NOT NULL,
            pref_scope TEXT NOT NULL DEFAULT 'global'
                CHECK(length(pref_scope) BETWEEN 1 AND 120),
            value TEXT NOT NULL,
            content_digest TEXT NOT NULL,
            revision INTEGER NOT NULL CHECK (revision >= 1),
            valid_from TEXT NOT NULL,
            timescale TEXT NOT NULL DEFAULT 'permanent'
                CHECK(timescale IN ('permanent', 'situational')),
            layer TEXT NOT NULL DEFAULT 'stable'
                CHECK(layer IN ('stable', 'contextual', 'transient')),
            last_corroborated_at TEXT NOT NULL,
            base_confidence REAL NOT NULL DEFAULT 1.0
                CHECK(base_confidence BETWEEN 0.0 AND 1.0),
            admission_status TEXT NOT NULL DEFAULT 'active'
                CHECK(admission_status IN ('active', 'quarantined')),
            capture_action TEXT NOT NULL DEFAULT 'add'
                CHECK(capture_action IN (
                    'retain', 'add', 'narrow-to-scope', 'revise',
                    'quarantine', 'ask'
                )),
            semantic_stance TEXT
                CHECK (
                    semantic_stance IS NULL
                    OR (
                        length(semantic_stance) BETWEEN 1 AND 80
                        AND semantic_stance GLOB '[a-z]*'
                    )
                ),
            superseded_at TEXT,
            superseded_by TEXT REFERENCES career_intent_versions(update_id)
                DEFERRABLE INITIALLY DEFERRED,
            source TEXT NOT NULL,
            UNIQUE(user_id, scope_key, pref_scope, revision),
            CHECK (
                (superseded_at IS NULL AND superseded_by IS NULL)
                OR (superseded_at IS NOT NULL AND superseded_by IS NOT NULL)
            )
        )
        """
    )


def _create_intent_version_indexes(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS career_intent_versions_active_idx
            ON career_intent_versions(user_id, scope_key, pref_scope)
            WHERE superseded_at IS NULL AND admission_status = 'active';
        CREATE UNIQUE INDEX IF NOT EXISTS career_intent_versions_quarantine_idx
            ON career_intent_versions(user_id, scope_key, pref_scope)
            WHERE superseded_at IS NULL AND admission_status = 'quarantined';
        CREATE INDEX IF NOT EXISTS career_intent_versions_history_idx
            ON career_intent_versions(
                user_id, scope_key, pref_scope, revision DESC
            );
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
    pref_scope: str = "global",
    timescale: IntentTimescale = "permanent",
    layer: IntentLayer = "stable",
    last_corroborated_at: datetime | None = None,
    base_confidence: float = 1.0,
    admission_status: IntentAdmissionStatus = "active",
    capture_action: IntentCaptureAction | None = None,
    semantic_stance: str | None = None,
) -> IntentMemoryVersion:
    """Append a revision, or corroborate in place when asked.

    Writing the same digest again is a no-op unless ``last_corroborated_at``
    is supplied. Migration backfills must not pass that field; live writes
    and CAPTURE retain/revise paths must, so decay only moves on evidence.
    """
    normalized = value.strip()
    if not normalized:
        raise ValueError("intent version value is required")
    digest = intent_content_digest(normalized)
    _validate_pref_scope(pref_scope)
    status = admission_status
    current_row = connection.execute(
        _SELECT
        + """
        WHERE user_id = ? AND scope_key = ? AND pref_scope = ?
          AND admission_status = ? AND superseded_at IS NULL
        """,
        (user_id, scope_key, pref_scope, status),
    ).fetchone()
    if current_row is not None:
        current = _version(current_row)
        if current.content_digest == digest:
            # Same digest is a no-op unless a caller explicitly corroborates.
            # Inferring from valid_from or now() would reset the CAPTURE decay
            # clock on every store open / migration backfill.
            if last_corroborated_at is None:
                return current
            corroborated_at = last_corroborated_at
            if corroborated_at > current.last_corroborated_at:
                connection.execute(
                    """
                    UPDATE career_intent_versions
                    SET last_corroborated_at = ?,
                        base_confidence = MAX(base_confidence, ?)
                    WHERE update_id = ?
                    """,
                    (
                        corroborated_at.isoformat(),
                        base_confidence,
                        current.update_id,
                    ),
                )
                return _version(
                    connection.execute(
                        _SELECT + " WHERE update_id = ?", (current.update_id,)
                    ).fetchone()
                )
            return current
    latest = connection.execute(
        """
        SELECT COALESCE(MAX(revision), 0)
        FROM career_intent_versions
        WHERE user_id = ? AND scope_key = ? AND pref_scope = ?
        """,
        (user_id, scope_key, pref_scope),
    ).fetchone()
    revision = int(latest[0]) + 1

    now = valid_from or datetime.now(timezone.utc)
    corroborated_at = last_corroborated_at or now
    action = capture_action or (
        "quarantine"
        if admission_status == "quarantined"
        else "revise"
        if current_row is not None
        else "narrow-to-scope"
        if pref_scope != "global"
        else "add"
    )
    version = IntentMemoryVersion(
        update_id=f"intent_update_{uuid4().hex}",
        user_id=user_id,
        scope_key=scope_key,
        value=normalized,
        content_digest=digest,
        revision=revision,
        valid_from=now,
        pref_scope=pref_scope,
        timescale=timescale,
        layer=layer,
        last_corroborated_at=corroborated_at,
        base_confidence=base_confidence,
        admission_status=admission_status,
        capture_action=action,
        semantic_stance=semantic_stance,
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
            update_id, user_id, scope_key, pref_scope, value, content_digest,
            revision, valid_from, timescale, layer, last_corroborated_at,
            base_confidence, admission_status, capture_action,
            semantic_stance, superseded_at, superseded_by, source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?)
        """,
        (
            version.update_id,
            version.user_id,
            version.scope_key,
            version.pref_scope,
            version.value,
            version.content_digest,
            version.revision,
            version.valid_from.isoformat(),
            version.timescale,
            version.layer,
            version.last_corroborated_at.isoformat(),
            version.base_confidence,
            version.admission_status,
            version.capture_action,
            version.semantic_stance,
            version.source,
        ),
    )
    return version


def capture_intent_version(
    connection: sqlite3.Connection,
    *,
    candidate: IntentCaptureCandidate,
) -> tuple[IntentCaptureDecision, IntentMemoryVersion | None]:
    """Apply the deterministic write gate and persist only its authorized action."""

    from career_agent.services.intent_capture import select_intent_capture_action

    track = list_intent_versions(
        connection,
        user_id=candidate.user_id,
        scope_key=candidate.scope_key,
        pref_scope=candidate.pref_scope,
    )
    active = next(
        (
            item
            for item in reversed(track)
            if item.superseded_at is None and item.admission_status == "active"
        ),
        None,
    )
    quarantined = next(
        (
            item
            for item in reversed(track)
            if item.superseded_at is None
            and item.admission_status == "quarantined"
        ),
        None,
    )
    decision = select_intent_capture_action(candidate, active=active)
    if decision.reason == "abstained":
        return decision, None
    if decision.action == "retain":
        if active is None:
            return decision, None
        version = append_intent_version(
            connection,
            user_id=candidate.user_id,
            scope_key=candidate.scope_key,
            pref_scope=candidate.pref_scope,
            value=active.value,
            source=candidate.source,
            valid_from=candidate.observed_at,
            timescale=active.timescale,
            layer=active.layer,
            last_corroborated_at=candidate.observed_at,
            base_confidence=max(active.base_confidence, candidate.confidence),
            capture_action="retain",
            semantic_stance=active.semantic_stance,
        )
        return decision, version
    status: IntentAdmissionStatus = (
        "quarantined" if decision.action == "quarantine" else "active"
    )
    version = append_intent_version(
        connection,
        user_id=candidate.user_id,
        scope_key=candidate.scope_key,
        pref_scope=candidate.pref_scope,
        value=candidate.value,
        source=candidate.source,
        valid_from=candidate.observed_at,
        timescale=candidate.timescale,
        layer=candidate.layer,
        last_corroborated_at=candidate.observed_at,
        base_confidence=candidate.confidence,
        admission_status=status,
        capture_action=decision.action,
        semantic_stance=candidate.semantic_stance,
    )
    # An admitted value closes the quarantine row holding the same value, so a
    # held candidate leaves quarantine by being confirmed rather than only by
    # expiring at the confidence floor.
    if (
        version.admission_status == "active"
        and quarantined is not None
        and quarantined.superseded_at is None
        and quarantined.content_digest == version.content_digest
    ):
        connection.execute(
            """
            UPDATE career_intent_versions
            SET superseded_at = ?, superseded_by = ?
            WHERE update_id = ? AND superseded_at IS NULL
            """,
            (
                candidate.observed_at.isoformat(),
                version.update_id,
                quarantined.update_id,
            ),
        )
    return decision, version


def list_intent_versions(
    connection: sqlite3.Connection,
    *,
    user_id: str,
    scope_key: str | None = None,
    scope_keys: Sequence[str] | None = None,
    pref_scope: str | None = None,
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
    if pref_scope is not None:
        _validate_pref_scope(pref_scope)
        where += " AND pref_scope = ?"
        parameters.append(pref_scope)
    if active_only:
        where += " AND superseded_at IS NULL AND admission_status = 'active'"
    query = _SELECT + where + " ORDER BY scope_key, pref_scope, revision"
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


def intent_entry_id(scope_key: str, pref_scope: str = "global") -> str:
    """Return the P1 identity of one independently versioned preference track."""

    _validate_pref_scope(pref_scope)
    return scope_key if pref_scope == "global" else f"{scope_key}#{pref_scope}"


def _validate_pref_scope(value: str) -> None:
    if re.fullmatch(r"(?:global|[a-z][a-z0-9_.:-]{0,119})", value) is None:
        raise ValueError(
            "pref_scope must be 'global' or a lowercase named domain"
        )


_SELECT = """
    SELECT update_id, user_id, scope_key, value, content_digest, revision,
           valid_from, superseded_at, superseded_by, source, pref_scope,
           timescale, layer, last_corroborated_at, base_confidence,
           admission_status, capture_action, semantic_stance
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
        pref_scope=row[10],
        timescale=row[11],
        layer=row[12],
        last_corroborated_at=row[13],
        base_confidence=row[14],
        admission_status=row[15],
        capture_action=row[16],
        semantic_stance=row[17],
    )
