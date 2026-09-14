from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import TYPE_CHECKING, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from career_agent.agent.delivered_body_contracts import (
    BodyDependency,
    DeliveredBodySource,
)
from career_agent.storage.turn_receipts import (
    redact_conversation_receipts_on,
    redact_turn_receipts_on,
)

from career_agent.agent.main_agent_contracts import (
    MAX_CONVERSATION_SPAN_MESSAGES,
    MAX_CONVERSATION_SPAN_RESOURCE_REFS,
    CareerProfileContext,
    ConversationMessageContext,
    ConversationSpanMessage,
    ConversationSpanView,
    ConversationTaskState,
    OwnerSettingsContext,
)
from career_agent.agent.conversation_memory_contracts import (
    ConversationSummaryContent,
    DistilledFreeTextPreferenceCandidate,
    SUMMARY_SOURCE_MAX_CHARS,
    StoredConversationSummary,
    SummaryMessage,
)
from career_agent.agent.session_contracts import AgentSession
from career_agent.domain.episodes import CareerEpisodeDraft
from career_agent.domain.intent_memory import IntentMemoryVersion
from career_agent.storage.episodes import (
    SQLiteCareerEpisodeStore,
    apply_episode_schema,
)
from career_agent.storage.intent_versions import (
    append_intent_version,
    apply_intent_version_schema,
    capture_intent_version,
    intent_entry_id,
    list_intent_versions,
    upgrade_intent_semantic_stance_schema,
    upgrade_intent_valid_until_schema,
    upgrade_intent_version_schema,
)
from career_agent.services.free_text_preferences import (
    extract_free_text_preference,
    normalize_preference_stance,
    preference_fts_query,
    preference_scope_key,
    preference_storage_assignment,
    preference_topic_key,
)
from career_agent.services.intent_capture import IntentCaptureCandidate
from career_agent.storage.schema import apply_schema

if TYPE_CHECKING:
    from career_agent.services.intent_capture import (
        IntentCaptureDecision,
    )

_OPAQUE_LINEAGE_MARKER = re.compile(
    r"^(?:detail|evidence|lineage)_[a-f0-9]{24}$"
)
_NOT_SUPPRESSED_SQL = """
                  AND NOT EXISTS (
                        SELECT 1 FROM memory_deletion_message_suppressions AS hidden
                        WHERE hidden.user_id = conversation_messages.user_id
                          AND hidden.conversation_id = conversation_messages.conversation_id
                          AND hidden.sequence = conversation_messages.sequence
                      )
"""


def _constraint_digest(text: str) -> str:
    """Identify a constraint by its exact text.

    A retirement has to outlive a rewrite that copies constraints forward
    verbatim, so the identity is the text itself rather than a position or a
    row id.
    """

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _usable_lineage_markers(markers: Sequence[str]) -> tuple[str, ...]:
    """Accept only opaque exact refs; free-text claims are never scan keys."""

    usable: list[str] = []
    for marker in markers:
        text = marker.strip()
        if not text:
            continue
        if _OPAQUE_LINEAGE_MARKER.fullmatch(text):
            usable.append(text)
    return tuple(dict.fromkeys(usable))


class ArchivedConversationConstraint(BaseModel):
    """One row of the constraint ledger behind a conversation summary."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    seq: int
    text: str
    status: Literal["active", "omitted", "retired"]
    first_seen_at: datetime
    status_changed_at: datetime


class StoredConversationOverview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation_id: str
    status: str
    created_at: datetime
    last_active_at: datetime
    title: str
    last_message_preview: str
    message_count: int


class StoredConversationMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence: int
    message: ConversationMessageContext


class DeliveredBodyDraft(BaseModel):
    """A body a turn showed in full while its row kept only a receipt.

    Written beside the assistant row it belongs to, so a reloaded transcript
    can open what the stream once displayed. The reader's copy only: the
    decision model keeps seeing the bounded row, which is why this is not a
    ``resource_ref`` on the message.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str = Field(min_length=1)
    """The tool-result state that produced the body."""

    title: str = Field(min_length=1)
    retention: Literal["snapshot", "source"] = "snapshot"
    body: str = ""
    source: DeliveredBodySource | None = None
    dependencies: tuple[BodyDependency, ...] = ()

    @model_validator(mode="after")
    def retention_matches_content(self) -> "DeliveredBodyDraft":
        if self.retention == "snapshot":
            if not self.body or self.source is not None:
                raise ValueError("snapshots require a body and no source")
        elif self.source is None or self.body:
            raise ValueError("source cards require a handle and no body")
        return self


class StoredDeliveredBodyReference(BaseModel):
    """Where a kept body sits in the transcript, without the body itself."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    body_id: str
    sequence: int
    kind: str
    title: str


class StoredDeliveredBody(DeliveredBodyDraft):
    body_id: str
    conversation_id: str
    sequence: int
    created_at: datetime


class OwnerSettingsConflictError(RuntimeError):
    """The caller edited a stale settings revision."""


class OwnerSettingsEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: int
    user_id: str
    revision: int
    policy_revision: int
    actor_type: Literal["api_key", "cli", "system", "confirmed_agent_proposal"]
    actor_id: str
    changed_fields: tuple[str, ...]
    before: OwnerSettingsContext
    after: OwnerSettingsContext
    changed_at: datetime


class CareerProfileStore(Protocol):
    """The narrow slice of context storage that may change stated job intent.

    Tools receive this instead of the whole context store so that recording an
    intent cannot reach conversation history, task state, or summaries.
    """

    def get_profile(self, user_id: str) -> CareerProfileContext | None: ...

    def upsert_profile(
        self,
        profile: CareerProfileContext,
        *,
        source: str = "career_profile_upsert",
        pref_scope: str = "global",
        timescale: str = "permanent",
        layer: str = "stable",
    ) -> None: ...

    def list_profile_intent_versions(
        self,
        *,
        user_id: str,
        scope_keys: Sequence[str] | None = None,
        pref_scope: str | None = None,
        active_only: bool = False,
        limit: int | None = None,
    ) -> tuple[IntentMemoryVersion, ...]: ...

    def list_free_text_preferences(
        self,
        *,
        user_id: str,
        statuses: Sequence[str] = ("active", "quarantined"),
    ) -> tuple[IntentMemoryVersion, ...]: ...


class CareerContextStore:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            apply_schema(
                connection,
                "agent_context",
                17,
                self._migrate,
                {
                    2: self._upgrade_to_v2,
                    3: self._upgrade_to_v3,
                    4: self._upgrade_to_v4,
                    5: self._upgrade_to_v5,
                    6: self._upgrade_to_v6,
                    7: self._upgrade_to_v7,
                    8: self._upgrade_to_v8,
                    9: upgrade_intent_version_schema,
                    10: self._upgrade_to_v10,
                    11: self._upgrade_to_v11,
                    12: upgrade_intent_semantic_stance_schema,
                    13: self._upgrade_to_v13,
                    14: self._upgrade_to_v14,
                    15: self._upgrade_to_v15,
                    16: self._upgrade_to_v16,
                    17: self._upgrade_to_v17,
                },
            )
            apply_episode_schema(connection)
            self._adopt_legacy_preferences(connection)
        os.chmod(self.path, 0o600)

    @staticmethod
    def _upgrade_to_v16(connection: sqlite3.Connection) -> None:
        columns = {row[1] for row in connection.execute(
            "PRAGMA table_info(conversation_delivered_bodies)"
        )}
        for name, definition in (
            ("retention", "TEXT NOT NULL DEFAULT 'snapshot'"),
            ("source_json", "TEXT"),
            ("dependencies_json", "TEXT"),
        ):
            if name not in columns:
                connection.execute(
                    f"ALTER TABLE conversation_delivered_bodies ADD COLUMN {name} {definition}"
                )
        for user_id, conversation_id in connection.execute(
            "SELECT DISTINCT user_id, conversation_id FROM conversation_delivered_bodies "
            "WHERE kind NOT IN ('daily_brief_ready', 'saved_jobs_compared')"
        ).fetchall():
            redact_conversation_receipts_on(connection, user_id, conversation_id)
        connection.execute(
            """
            DELETE FROM conversation_delivered_bodies
            WHERE kind NOT IN ('daily_brief_ready', 'saved_jobs_compared')
              OR EXISTS (
                  SELECT 1 FROM memory_deletion_message_suppressions AS hidden
                  WHERE hidden.user_id = conversation_delivered_bodies.user_id
                    AND hidden.conversation_id = conversation_delivered_bodies.conversation_id
                    AND hidden.sequence = conversation_delivered_bodies.sequence
              )
            """
        )
        for user_id, conversation_id in connection.execute(
            "SELECT DISTINCT user_id, conversation_id FROM memory_deletion_message_suppressions"
        ).fetchall():
            redact_conversation_receipts_on(connection, user_id, conversation_id)

    @staticmethod
    def _upgrade_to_v17(connection: sqlite3.Connection) -> None:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(conversation_messages)")
        }
        if "turn_id" not in columns:
            connection.execute("ALTER TABLE conversation_messages ADD COLUMN turn_id TEXT")

    @staticmethod
    def _redact_receipts_for_rows(
        connection: sqlite3.Connection,
        user_id: str,
        rows: Iterable[tuple[str, int]],
    ) -> None:
        """Clear the receipts of the turns that wrote these transcript rows.

        A row that predates turn recording cannot name its turn, so the whole
        conversation up to now is cleared instead: over-redacting bounded
        history beats letting removed content replay.
        """

        by_conversation: dict[str, set[str]] = {}
        unattributed: set[str] = set()
        for conversation_id, sequence in rows:
            row = connection.execute(
                "SELECT turn_id FROM conversation_messages "
                "WHERE user_id = ? AND conversation_id = ? AND sequence = ?",
                (user_id, conversation_id, sequence),
            ).fetchone()
            if row is None or row[0] is None:
                unattributed.add(conversation_id)
            else:
                by_conversation.setdefault(conversation_id, set()).add(str(row[0]))
        for conversation_id in sorted(unattributed):
            redact_conversation_receipts_on(connection, user_id, conversation_id)
        for conversation_id, turn_ids in sorted(by_conversation.items()):
            if conversation_id not in unattributed:
                redact_turn_receipts_on(
                    connection, user_id, conversation_id, sorted(turn_ids)
                )

    @staticmethod
    def _upgrade_to_v14(connection: sqlite3.Connection) -> None:
        CareerContextStore._ensure_free_text_preference_fts(connection)

    @staticmethod
    def _upgrade_to_v15(connection: sqlite3.Connection) -> None:
        CareerContextStore._ensure_memory_review_schema(connection)

    @staticmethod
    def _ensure_memory_review_schema(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_review_exports (
                export_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                items_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS memory_review_exports_user_idx
            ON memory_review_exports(user_id, created_at DESC)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS intent_memory_tombstones (
                user_id TEXT NOT NULL,
                scope_key TEXT NOT NULL,
                pref_scope TEXT NOT NULL,
                update_id TEXT NOT NULL,
                content_digest TEXT NOT NULL,
                reason TEXT NOT NULL,
                deleted_at TEXT NOT NULL,
                PRIMARY KEY(user_id, scope_key, pref_scope)
            )
            """
        )

    @staticmethod
    def _upgrade_to_v13(connection: sqlite3.Connection) -> None:
        """Add read-time expiry and canonicalize the first dual-track scope."""

        upgrade_intent_valid_until_schema(connection)
        old_scope = "person_intent/self/employer_scale_preference"
        new_scope = "person_intent/self/company_scale"
        connection.execute(
            """
            UPDATE career_intent_versions AS old
            SET scope_key = ?
            WHERE scope_key = ?
              AND NOT EXISTS (
                    SELECT 1 FROM career_intent_versions AS current
                    WHERE current.user_id = old.user_id
                      AND current.scope_key = ?
                  )
            """,
            (new_scope, old_scope, new_scope),
        )
        for table, columns in (
            (
                "conversation_message_memory_bindings",
                "user_id, conversation_id, sequence, scope_key, created_at",
            ),
            (
                "memory_deletion_message_suppressions",
                "user_id, conversation_id, sequence, scope_key, suppressed_at",
            ),
            (
                "memory_deleted_scopes",
                "user_id, scope_key, deleted_at",
            ),
            (
                "career_episode_memory_bindings",
                "episode_id, user_id, scope_key, created_at",
            ),
        ):
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone()
            if exists is None:
                continue
            selected = columns.replace("scope_key", "?")
            connection.execute(
                f"INSERT OR IGNORE INTO {table}({columns}) "
                f"SELECT {selected} FROM {table} WHERE scope_key = ?",
                (new_scope, old_scope),
            )
            connection.execute(
                f"DELETE FROM {table} WHERE scope_key = ?",
                (old_scope,),
            )

    @staticmethod
    def _upgrade_to_v2(connection: sqlite3.Connection) -> None:
        """One turn can store two reports, so the row keeps a list.

        This is the *only* place that knows the singular shape ever existed.
        ``ConversationMessageContext`` deliberately has no compatibility shim:
        a permanent reader for both shapes would contradict this migration and
        leave every predicate here, and every future reader, asking two
        questions — which is how the live stream and the reloaded transcript
        drifted apart in the first place. The migration runs before any read
        (``apply_schema`` in ``__init__``), so a stored row is always converted
        by the time the contract sees it.

        Pinned by ``tests/storage/test_context_resource_refs_migration.py``,
        which builds a real v1 file and reads it back.
        """
        connection.execute(
            """
            UPDATE conversation_messages
            SET payload = json_set(
                json_remove(payload, '$.resource_ref'),
                '$.resource_refs',
                json_array(json_extract(payload, '$.resource_ref'))
            )
            WHERE json_extract(payload, '$.resource_ref') IS NOT NULL
            """
        )

    @staticmethod
    def _upgrade_to_v3(connection: sqlite3.Connection) -> None:
        """Split legacy preference JSON into soft preferences and hard policy."""

        present = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='agent_preferences_context'"
        ).fetchone()
        if present is None:
            return
        rows = connection.execute(
            "SELECT user_id, payload, updated_at FROM agent_preferences_context"
        ).fetchall()
        for user_id, payload, updated_at in rows:
            settings = OwnerSettingsContext.model_validate_json(payload)
            connection.execute(
                "INSERT OR IGNORE INTO owner_settings_context"
                "(user_id, payload, updated_at) VALUES (?, ?, ?)",
                (user_id, settings.model_dump_json(), updated_at),
            )
        connection.execute("DROP TABLE agent_preferences_context")

    @staticmethod
    def _upgrade_to_v4(connection: sqlite3.Connection) -> None:
        """Give every durable session one stable randomized spotlight nonce."""
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(sessions)")
        }
        if "spotlight_nonce" not in columns:
            connection.execute(
                "ALTER TABLE sessions ADD COLUMN spotlight_nonce TEXT"
            )
        connection.execute(
            "UPDATE sessions SET spotlight_nonce = lower(hex(randomblob(16))) "
            "WHERE spotlight_nonce IS NULL"
        )

    @staticmethod
    def _upgrade_to_v5(connection: sqlite3.Connection) -> None:
        CareerContextStore._backfill_profile_intent_versions(connection)

    @staticmethod
    def _upgrade_to_v6(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_deletion_cutoffs (
                user_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                through_sequence INTEGER NOT NULL CHECK(through_sequence >= 0),
                updated_at TEXT NOT NULL,
                PRIMARY KEY(user_id, conversation_id)
            )
            """
        )
        CareerContextStore._ensure_memory_deletion_schema(connection)

    @staticmethod
    def _upgrade_to_v7(connection: sqlite3.Connection) -> None:
        CareerContextStore._ensure_memory_deletion_schema(connection)

    @staticmethod
    def _upgrade_to_v8(connection: sqlite3.Connection) -> None:
        # v6's user-wide cutoff was intentionally ignored after field-level
        # provenance shipped in v7. Remove the now-unread compatibility table.
        connection.execute("DROP TABLE IF EXISTS memory_deletion_cutoffs")

    @staticmethod
    def _upgrade_to_v10(connection: sqlite3.Connection) -> None:
        CareerContextStore._drop_removed_scope_queue(connection)

    @staticmethod
    def _upgrade_to_v11(connection: sqlite3.Connection) -> None:
        CareerContextStore._ensure_constraint_archive_schema(connection)
        CareerContextStore._backfill_constraint_archive(connection)

    @staticmethod
    def _ensure_constraint_archive_schema(connection: sqlite3.Connection) -> None:
        """Give conversation constraints an entry *and* an exit.

        The summary is a bounded projection; this table is the ledger behind
        it. Three facts need somewhere durable to live that a lossy rewrite
        cannot reach:

        ``retired`` is why the table exists. The summary worker receives the
        previous summary and copies constraints forward verbatim, and the very
        messages announcing a retirement are in the batch being summarized, so
        dropping a constraint from the summary alone invites the next rewrite
        to re-extract it. Matching on the exact text digest blocks that
        verbatim path; it cannot block a paraphrase, and nothing here pretends
        otherwise.

        ``omitted`` turns the visible cap from a discard into a page. A
        constraint pushed out by the cap keeps its ``seq``, so retiring one
        constraint readmits the oldest omitted one on the next compaction
        instead of stranding it.

        ``seq`` is first-seen order, which is the only ordering the cap needs:
        a constraint that already survived a rewrite is never traded for a
        newer one.
        """

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_constraint_archive (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                constraint_sha256 TEXT NOT NULL,
                constraint_text TEXT NOT NULL,
                status TEXT NOT NULL
                    CHECK(status IN ('active', 'omitted', 'retired')),
                first_seen_at TEXT NOT NULL,
                status_changed_at TEXT NOT NULL,
                UNIQUE(user_id, conversation_id, constraint_sha256)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS conversation_constraint_archive_idx
            ON conversation_constraint_archive(
                user_id, conversation_id, status, seq
            )
            """
        )

    @staticmethod
    def _backfill_constraint_archive(connection: sqlite3.Connection) -> None:
        """Seed the ledger from summaries written before it existed.

        Only the constraints still visible in a stored summary can be
        recovered; anything an earlier cap already discarded was never
        persisted anywhere and is not reconstructable here.
        """

        rows = connection.execute(
            "SELECT user_id, conversation_id, content_json, updated_at "
            "FROM conversation_summaries"
        ).fetchall()
        for user_id, conversation_id, content_json, updated_at in rows:
            content = ConversationSummaryContent.model_validate_json(content_json)
            for text in content.active_constraints:
                connection.execute(
                    """
                    INSERT INTO conversation_constraint_archive(
                        user_id, conversation_id, constraint_sha256,
                        constraint_text, status, first_seen_at,
                        status_changed_at
                    ) VALUES (?, ?, ?, ?, 'active', ?, ?)
                    ON CONFLICT(user_id, conversation_id, constraint_sha256)
                        DO NOTHING
                    """,
                    (
                        user_id,
                        conversation_id,
                        _constraint_digest(text),
                        text,
                        updated_at,
                        updated_at,
                    ),
                )

    @staticmethod
    def _drop_removed_scope_queue(connection: sqlite3.Connection) -> None:
        connection.execute("DROP TABLE IF EXISTS scope_resolution_events")
        connection.execute("DROP TABLE IF EXISTS scope_resolution_queue")
        connection.execute(
            "DELETE FROM schema_versions WHERE component = 'memory_scope'"
        )

    @staticmethod
    def _ensure_memory_deletion_schema(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_message_memory_bindings (
                user_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                sequence INTEGER NOT NULL CHECK(sequence >= 1),
                scope_key TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(user_id, conversation_id, sequence, scope_key)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS conversation_message_memory_scope_idx
            ON conversation_message_memory_bindings(user_id, scope_key)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_deletion_message_suppressions (
                user_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                sequence INTEGER NOT NULL CHECK(sequence >= 1),
                scope_key TEXT NOT NULL,
                suppressed_at TEXT NOT NULL,
                PRIMARY KEY(user_id, conversation_id, sequence, scope_key)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_deleted_scopes (
                user_id TEXT NOT NULL,
                scope_key TEXT NOT NULL,
                deleted_at TEXT NOT NULL,
                PRIMARY KEY(user_id, scope_key)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS memory_deletion_message_lookup_idx
            ON memory_deletion_message_suppressions(
                user_id, conversation_id, sequence
            )
            """
        )

    @staticmethod
    def _ensure_free_text_preference_fts(
        connection: sqlite3.Connection,
    ) -> None:
        connection.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS free_text_preferences_fts
            USING fts5(
                update_id UNINDEXED,
                user_id UNINDEXED,
                topic_key,
                statement,
                tokenize='trigram'
            )
            """
        )
        connection.execute("DELETE FROM free_text_preferences_fts")
        rows = connection.execute(
            """
            SELECT update_id, user_id, scope_key, value
            FROM career_intent_versions
            WHERE pref_scope = 'freeform'
               OR pref_scope LIKE 'freeform.%'
            """
        ).fetchall()
        connection.executemany(
            """
            INSERT INTO free_text_preferences_fts(
                update_id, user_id, topic_key, statement
            ) VALUES (?, ?, ?, ?)
            """,
            (
                (
                    str(update_id),
                    str(user_id),
                    preference_topic_key(str(scope_key)),
                    str(value),
                )
                for update_id, user_id, scope_key, value in rows
            ),
        )

    @staticmethod
    def _adopt_legacy_preferences(connection: sqlite3.Connection) -> None:
        """Handle a pre-registry database, for which apply_schema skips upgrades."""

        CareerContextStore._upgrade_to_v3(connection)
        connection.execute(
            """
            UPDATE conversation_messages
            SET payload = json_remove(payload, '$.resource_ref')
            WHERE json_type(payload, '$.resource_ref') = 'null'
            """
        )

    @staticmethod
    def _backfill_profile_intent_versions(connection: sqlite3.Connection) -> None:
        backfilled_at = datetime.now(timezone.utc)
        rows = connection.execute(
            "SELECT user_id, payload FROM career_profile_context"
        ).fetchall()
        for user_id, payload in rows:
            profile = CareerProfileContext.model_validate_json(payload)
            city_scope = "person_intent/self/default_city"
            if profile.default_city is not None and not list_intent_versions(
                connection,
                user_id=user_id,
                scope_key=city_scope,
                limit=1,
            ):
                append_intent_version(
                    connection,
                    user_id=user_id,
                    scope_key=city_scope,
                    value=profile.default_city,
                    source="migration:career_profile_context",
                    valid_from=backfilled_at,
                )
            for constraint in profile.hard_constraints:
                scope_key = f"person_intent/self/{constraint.relation}"
                if list_intent_versions(
                    connection,
                    user_id=user_id,
                    scope_key=scope_key,
                    limit=1,
                ):
                    continue
                append_intent_version(
                    connection,
                    user_id=user_id,
                    scope_key=scope_key,
                    value=constraint.value,
                    source="migration:career_profile_context",
                    valid_from=backfilled_at,
                )

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
        connection.execute("CREATE TABLE IF NOT EXISTS sessions (session_id TEXT NOT NULL, user_id TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL, last_active_at TEXT NOT NULL, spotlight_nonce TEXT NOT NULL, PRIMARY KEY(user_id, session_id))")
        CareerContextStore._upgrade_to_v4(connection)
        connection.execute("CREATE INDEX IF NOT EXISTS sessions_user_idx ON sessions(user_id, last_active_at DESC)")
        connection.execute("CREATE TABLE IF NOT EXISTS career_profile_context (user_id TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at TEXT NOT NULL)")
        apply_intent_version_schema(connection)
        # Intentional cumulative-baseline exception: pre-registry databases do
        # not replay numbered upgrades. Existing tracks are skipped, and
        # append_intent_version no-ops on the same digest unless a caller
        # explicitly corroborates, so opening the store never resets decay.
        CareerContextStore._backfill_profile_intent_versions(connection)
        connection.execute("CREATE TABLE IF NOT EXISTS owner_settings_context (user_id TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at TEXT NOT NULL)")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS owner_settings_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
                policy_revision INTEGER NOT NULL,
                actor_type TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                changed_fields_json TEXT NOT NULL,
                before_json TEXT NOT NULL,
                after_json TEXT NOT NULL,
                changed_at TEXT NOT NULL,
                UNIQUE(user_id, revision)
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS owner_settings_events_user_idx "
            "ON owner_settings_events(user_id, revision DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS owner_settings_events_actor_idx "
            "ON owner_settings_events(user_id, actor_type, actor_id)"
        )
        connection.execute("CREATE TABLE IF NOT EXISTS conversation_task_state (user_id TEXT NOT NULL, conversation_id TEXT NOT NULL, payload TEXT NOT NULL, updated_at TEXT NOT NULL, PRIMARY KEY(user_id, conversation_id))")
        connection.execute("CREATE TABLE IF NOT EXISTS conversation_messages (user_id TEXT NOT NULL, conversation_id TEXT NOT NULL, sequence INTEGER NOT NULL, payload TEXT NOT NULL, turn_id TEXT, PRIMARY KEY(user_id, conversation_id, sequence))")
        connection.execute("CREATE INDEX IF NOT EXISTS conversation_messages_recent_idx ON conversation_messages(user_id, conversation_id, sequence DESC)")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_delivered_bodies (
                user_id TEXT NOT NULL,
                body_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                kind TEXT NOT NULL,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                retention TEXT NOT NULL DEFAULT 'snapshot',
                source_json TEXT,
                dependencies_json TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY(user_id, body_id)
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS conversation_delivered_bodies_message_idx "
            "ON conversation_delivered_bodies(user_id, conversation_id, sequence)"
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS delivered_body_deleted_dependencies (
                user_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                resource_id TEXT NOT NULL,
                PRIMARY KEY(user_id, kind, resource_id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                user_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                content_json TEXT NOT NULL,
                through_sequence INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(user_id, conversation_id)
            )
            """
        )
        CareerContextStore._ensure_memory_deletion_schema(connection)
        CareerContextStore._ensure_memory_review_schema(connection)
        CareerContextStore._ensure_free_text_preference_fts(connection)
        CareerContextStore._drop_removed_scope_queue(connection)
        CareerContextStore._ensure_constraint_archive_schema(connection)

    def get_session(self, user_id: str, session_id: str) -> AgentSession | None:
        with self._connect() as connection:
            row = connection.execute("SELECT session_id, user_id, status, created_at, last_active_at, spotlight_nonce FROM sessions WHERE session_id = ? AND user_id = ?", (session_id, user_id)).fetchone()
        return AgentSession(session_id=row[0], user_id=row[1], status=row[2], created_at=row[3], last_active_at=row[4], spotlight_nonce=row[5]) if row else None

    def upsert_session(self, session: AgentSession) -> AgentSession:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO sessions(session_id, user_id, status, created_at, last_active_at, spotlight_nonce) VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(user_id, session_id) DO UPDATE SET status=excluded.status, last_active_at=excluded.last_active_at",
                (session.session_id, session.user_id, session.status, session.created_at.isoformat(), session.last_active_at.isoformat(), session.spotlight_nonce),
            )
        os.chmod(self.path, 0o600)
        return session

    def close_session(self, user_id: str, session_id: str) -> AgentSession | None:
        session = self.get_session(user_id, session_id)
        if session is None:
            return None
        return self.upsert_session(session.model_copy(update={"status": "closed"}))

    def delete_conversation(self, *, user_id: str, conversation_id: str) -> bool:
        """Delete one user's chat memory, but retain security/audit records.

        Capability confirmations, action executions, and telemetry are not UI
        conversation content. They remain durable so a hidden conversation
        cannot erase evidence of an external write or make it replayable.
        """

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            present = connection.execute(
                "SELECT 1 FROM sessions WHERE user_id = ? AND session_id = ?",
                (user_id, conversation_id),
            ).fetchone()
            if present is None:
                return False
            redact_conversation_receipts_on(connection, user_id, conversation_id)
            for table in (
                "conversation_summaries",
                "conversation_messages",
                "conversation_delivered_bodies",
                "conversation_task_state",
                "conversation_message_memory_bindings",
                "memory_deletion_message_suppressions",
                "conversation_constraint_archive",
            ):
                connection.execute(
                    f"DELETE FROM {table} WHERE user_id = ? AND conversation_id = ?",
                    (user_id, conversation_id),
                )
            connection.execute(
                "DELETE FROM sessions WHERE user_id = ? AND session_id = ?",
                (user_id, conversation_id),
            )
        return True

    def list_conversations(
        self, *, user_id: str, limit: int = 50
    ) -> tuple[StoredConversationOverview, ...]:
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT s.session_id, s.status, s.created_at, s.last_active_at,
                       (SELECT payload FROM conversation_messages AS first
                        WHERE first.user_id = s.user_id
                          AND first.conversation_id = s.session_id
                          AND NOT EXISTS (
                                SELECT 1
                                FROM memory_deletion_message_suppressions AS hidden
                                WHERE hidden.user_id = first.user_id
                                  AND hidden.conversation_id = first.conversation_id
                                  AND hidden.sequence = first.sequence
                              )
                        ORDER BY first.sequence ASC LIMIT 1),
                       (SELECT payload FROM conversation_messages AS last
                        WHERE last.user_id = s.user_id
                          AND last.conversation_id = s.session_id
                          AND NOT EXISTS (
                                SELECT 1
                                FROM memory_deletion_message_suppressions AS hidden
                                WHERE hidden.user_id = last.user_id
                                  AND hidden.conversation_id = last.conversation_id
                                  AND hidden.sequence = last.sequence
                              )
                        ORDER BY last.sequence DESC LIMIT 1),
                       (SELECT COUNT(*) FROM conversation_messages AS messages
                        WHERE messages.user_id = s.user_id
                          AND messages.conversation_id = s.session_id
                          AND NOT EXISTS (
                                SELECT 1
                                FROM memory_deletion_message_suppressions AS hidden
                                WHERE hidden.user_id = messages.user_id
                                  AND hidden.conversation_id = messages.conversation_id
                                  AND hidden.sequence = messages.sequence
                              ))
                FROM sessions AS s
                WHERE s.user_id = ?
                  AND EXISTS (
                      SELECT 1 FROM conversation_messages AS present
                      WHERE present.user_id = s.user_id
                        AND present.conversation_id = s.session_id
                        AND NOT EXISTS (
                              SELECT 1
                              FROM memory_deletion_message_suppressions AS hidden
                              WHERE hidden.user_id = present.user_id
                                AND hidden.conversation_id = present.conversation_id
                                AND hidden.sequence = present.sequence
                            )
                  )
                ORDER BY s.last_active_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        conversations = []
        for row in rows:
            first = (
                ConversationMessageContext.model_validate_json(row[4])
                if row[4]
                else None
            )
            last = (
                ConversationMessageContext.model_validate_json(row[5])
                if row[5]
                else None
            )
            title = first.content.strip() if first else "新对话"
            preview = last.content.strip() if last else "尚未发送消息"
            conversations.append(
                StoredConversationOverview(
                    conversation_id=row[0],
                    status=row[1],
                    created_at=row[2],
                    last_active_at=row[3],
                    title=title[:80],
                    last_message_preview=preview[:160],
                    message_count=int(row[6]),
                )
            )
        return tuple(conversations)

    def get_profile(self, user_id: str) -> CareerProfileContext | None:
        return self._get_single("career_profile_context", user_id, CareerProfileContext)

    def upsert_profile(
        self,
        profile: CareerProfileContext,
        *,
        source: str = "career_profile_upsert",
        pref_scope: str = "global",
        timescale: str = "permanent",
        layer: str = "stable",
    ) -> None:
        if pref_scope != "global":
            raise ValueError(
                "named-scope intent must use capture_profile_intent, not the flat profile"
            )
        now = datetime.now(timezone.utc)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload FROM career_profile_context WHERE user_id = ?",
                (profile.user_id,),
            ).fetchone()
            before = (
                CareerProfileContext.model_validate_json(row[0])
                if row is not None
                else None
            )
            if (
                before is not None
                and before.default_city is not None
                and profile.default_city is None
            ):
                raise ValueError(
                    "clearing confirmed job intent requires the M3 forget primitive"
                )
            previous_constraints = (
                {item.relation for item in before.hard_constraints}
                if before is not None
                else set()
            )
            desired_constraints = {
                item.relation for item in profile.hard_constraints
            }
            if previous_constraints - desired_constraints:
                raise ValueError(
                    "removing a hard constraint requires the M3 forget primitive"
                )
            connection.execute(
                """
                INSERT INTO career_profile_context(user_id, payload, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE
                SET payload = excluded.payload, updated_at = excluded.updated_at
                """,
                (profile.user_id, profile.model_dump_json(), now.isoformat()),
            )
            if profile.default_city is not None:
                append_intent_version(
                    connection,
                    user_id=profile.user_id,
                    scope_key="person_intent/self/default_city",
                    value=profile.default_city,
                    source=source,
                    valid_from=now,
                    last_corroborated_at=now,
                    pref_scope=pref_scope,
                    timescale=timescale,
                    layer=layer,
                )
            for constraint in profile.hard_constraints:
                append_intent_version(
                    connection,
                    user_id=profile.user_id,
                    scope_key=f"person_intent/self/{constraint.relation}",
                    value=constraint.value,
                    source=source,
                    valid_from=now,
                    last_corroborated_at=now,
                    pref_scope=pref_scope,
                    timescale=timescale,
                    layer=layer,
                )

    def list_profile_intent_versions(
        self,
        *,
        user_id: str,
        scope_key: str | None = None,
        scope_keys: Sequence[str] | None = None,
        pref_scope: str | None = None,
        active_only: bool = False,
        limit: int | None = None,
    ) -> tuple[IntentMemoryVersion, ...]:
        with self._connect() as connection:
            deleted_by_scope = {
                str(row[0]): datetime.fromisoformat(str(row[1]))
                for row in connection.execute(
                    """
                    SELECT scope_key, deleted_at
                    FROM memory_deleted_scopes
                    WHERE user_id = ?
                    """,
                    (user_id,),
                ).fetchall()
            }
            tombstoned_tracks = {
                (str(row[0]), str(row[1])): datetime.fromisoformat(str(row[2]))
                for row in connection.execute(
                    """
                    SELECT scope_key, pref_scope, deleted_at
                    FROM intent_memory_tombstones
                    WHERE user_id = ?
                    """,
                    (user_id,),
                ).fetchall()
            }
            versions = list_intent_versions(
                connection,
                user_id=user_id,
                scope_key=scope_key,
                scope_keys=scope_keys,
                pref_scope=pref_scope,
                active_only=active_only,
                limit=None if deleted_by_scope or tombstoned_tracks else limit,
            )
        visible = tuple(
            item
            for item in versions
            if (
                item.scope_key not in deleted_by_scope
                or item.valid_from > deleted_by_scope[item.scope_key]
            )
            and (
                (item.scope_key, item.pref_scope) not in tombstoned_tracks
                or item.valid_from
                > tombstoned_tracks[(item.scope_key, item.pref_scope)]
            )
        )
        return visible[:limit] if limit is not None else visible

    def capture_profile_intent(
        self,
        candidate: IntentCaptureCandidate,
    ) -> tuple[IntentCaptureDecision, IntentMemoryVersion | None]:
        if not candidate.scope_key.startswith("person_intent/self/"):
            raise ValueError("profile intent requires a person_intent/self scope")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            decision, version = capture_intent_version(
                connection, candidate=candidate
            )
            # Free-text rows are ranked by FTS wherever they were written from.
            # This path skipped the index, which stayed invisible while only
            # quarantined candidates were ranked and every one of those came in
            # through the message path.
            if version is not None and version.pref_scope.startswith("freeform"):
                self._index_free_text_preference_on(connection, version)
            return decision, version

    def capture_free_text_preference_from_message(
        self,
        *,
        user_id: str,
        conversation_id: str,
        message: str,
    ) -> IntentMemoryVersion | None:
        """Persist an explicit supported mutation before context projection.

        Extraction deliberately abstains unless the message itself contains a
        supported first-person preference or an explicit deletion request.
        Replaying ``load_for_turn`` is idempotent by content digest.
        """

        mutation = extract_free_text_preference(message)
        if mutation is None:
            return None
        scope_key = self._free_text_preference_scope(mutation.topic_key)
        if mutation.action == "delete":
            self.purge_derived_memory(user_id=user_id, scope_key=scope_key)
            return None
        assert mutation.statement is not None
        assert mutation.ownership is not None
        task = self.get_task(user_id, conversation_id)
        observed_at = datetime.now(timezone.utc)
        assignment = preference_storage_assignment(
            mutation.ownership,
            observed_at=observed_at,
            conversation_id=conversation_id,
            job_posting_id=(
                task.active_job_posting_id if task is not None else None
            ),
            statement=mutation.statement,
        )
        candidate = IntentCaptureCandidate(
            user_id=user_id,
            scope_key=scope_key,
            value=mutation.statement,
            source=f"conversation_user_statement:{conversation_id}"[:200],
            pref_scope=assignment.pref_scope,
            timescale=assignment.timescale,
            layer=assignment.layer,
            valid_until=assignment.valid_until,
            ambiguous=True,
            scope_ambiguous=assignment.scope_ambiguous,
            semantic_stance=mutation.stance,
            observed_at=observed_at,
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            return self._capture_free_text_candidate_on(
                connection,
                candidate=candidate,
                corroborate_active=True,
            )

    def _capture_free_text_candidate_on(
        self,
        connection: sqlite3.Connection,
        *,
        candidate: IntentCaptureCandidate,
        corroborate_active: bool = False,
    ) -> IntentMemoryVersion | None:
        candidate_stance = normalize_preference_stance(
            candidate.semantic_stance
        )
        if candidate_stance is None:
            raise ValueError(
                "free-text preference stance must have controlled polarity"
            )
        candidate = candidate.model_copy(
            update={"semantic_stance": candidate_stance}
        )
        if candidate.pref_scope == "freeform.person_default":
            legacy_track = list_intent_versions(
                connection,
                user_id=candidate.user_id,
                scope_key=candidate.scope_key,
                pref_scope="freeform",
            )
            if any(
                item.superseded_at is None for item in legacy_track
            ):
                candidate = candidate.model_copy(
                    update={"pref_scope": "freeform"}
                )
        deleted_at = self._memory_scope_deleted_at_on(
            connection,
            user_id=candidate.user_id,
            scope_key=candidate.scope_key,
        )
        active = next(
            (
                item
                for item in reversed(
                    list_intent_versions(
                        connection,
                        user_id=candidate.user_id,
                        scope_key=candidate.scope_key,
                        pref_scope=candidate.pref_scope,
                    )
                )
                if item.superseded_at is None
                and item.admission_status == "active"
            ),
            None,
        )
        active_stance = (
            normalize_preference_stance(active.semantic_stance)
            if active is not None
            else None
        )
        if active is not None and active_stance is None:
            legacy_mutation = extract_free_text_preference(active.value)
            active_stance = (
                legacy_mutation.stance
                if legacy_mutation is not None
                and legacy_mutation.action == "quarantine"
                else None
            )
        if (
            active is not None
            and deleted_at is None
            and active_stance is not None
            and active_stance == candidate.semantic_stance
        ):
            if not corroborate_active:
                # A model-produced distillation can propose a contradiction,
                # but it cannot refresh an active preference's user-evidence
                # clock. Same-stance inference is already covered by active.
                return active
            corroborated = append_intent_version(
                connection,
                user_id=candidate.user_id,
                scope_key=active.scope_key,
                value=active.value,
                source=candidate.source,
                pref_scope=active.pref_scope,
                timescale=active.timescale,
                layer=active.layer,
                valid_until=candidate.valid_until or active.valid_until,
                last_corroborated_at=candidate.observed_at,
                base_confidence=max(active.base_confidence, candidate.confidence),
                admission_status="active",
                capture_action="retain",
                semantic_stance=active_stance,
            )
            # A same-stance restatement resolves any older opposite candidate
            # that was still waiting in quarantine.
            connection.execute(
                """
                UPDATE career_intent_versions
                SET superseded_at = ?, superseded_by = ?
                WHERE user_id = ? AND scope_key = ? AND pref_scope = ?
                  AND admission_status = 'quarantined'
                  AND superseded_at IS NULL
                """,
                (
                    candidate.observed_at.isoformat(),
                    corroborated.update_id,
                    candidate.user_id,
                    candidate.scope_key,
                    candidate.pref_scope,
                ),
            )
            # Corroboration mints a new update_id for the same statement, and the
            # index is keyed by update_id: without this the refreshed preference
            # would drop out of every ranking.
            self._index_free_text_preference_on(connection, corroborated)
            return corroborated
        if not corroborate_active:
            quarantined = next(
                (
                    item
                    for item in reversed(
                        list_intent_versions(
                            connection,
                            user_id=candidate.user_id,
                            scope_key=candidate.scope_key,
                            pref_scope=candidate.pref_scope,
                        )
                    )
                    if item.superseded_at is None
                    and item.admission_status == "quarantined"
                ),
                None,
            )
            if (
                quarantined is not None
                and normalize_preference_stance(
                    quarantined.semantic_stance
                )
                == candidate.semantic_stance
            ):
                # Keep an explicit/direct candidate when summary distillation
                # later paraphrases the same stance.
                return quarantined
        _, version = capture_intent_version(connection, candidate=candidate)
        if version is not None:
            self._index_free_text_preference_on(connection, version)
        return version

    def list_free_text_preferences(
        self,
        *,
        user_id: str,
        statuses: Sequence[str] = ("active", "quarantined"),
    ) -> tuple[IntentMemoryVersion, ...]:
        with self._connect() as connection:
            versions = list_intent_versions(
                connection,
                user_id=user_id,
            )
            deleted_by_scope = {
                str(row[0]): datetime.fromisoformat(str(row[1]))
                for row in connection.execute(
                    """
                    SELECT scope_key, deleted_at
                    FROM memory_deleted_scopes
                    WHERE user_id = ?
                    """,
                    (user_id,),
                ).fetchall()
            }
            tombstoned_tracks = {
                (str(row[0]), str(row[1])): datetime.fromisoformat(str(row[2]))
                for row in connection.execute(
                    """
                    SELECT scope_key, pref_scope, deleted_at
                    FROM intent_memory_tombstones
                    WHERE user_id = ?
                    """,
                    (user_id,),
                ).fetchall()
            }
        selected = set(statuses)
        if not selected <= {"active", "quarantined"}:
            raise ValueError("invalid free-text preference status")
        return tuple(
            item
            for item in versions
            if item.pref_scope.startswith("freeform")
            and item.superseded_at is None
            and item.admission_status in selected
            and (
                item.scope_key not in deleted_by_scope
                or item.valid_from > deleted_by_scope[item.scope_key]
            )
            and (
                (item.scope_key, item.pref_scope) not in tombstoned_tracks
                or item.valid_from
                > tombstoned_tracks[(item.scope_key, item.pref_scope)]
            )
        )

    def get_active_free_text_preference(
        self,
        *,
        user_id: str,
        update_id: str,
    ) -> IntentMemoryVersion | None:
        return next(
            (
                item
                for item in self.list_free_text_preferences(
                    user_id=user_id,
                    statuses=("active",),
                )
                if item.update_id == update_id
            ),
            None,
        )

    def get_current_free_text_preference_track(
        self,
        *,
        user_id: str,
        scope_key: str,
        pref_scope: str,
    ) -> IntentMemoryVersion | None:
        """Return the active head of one preference track, not one revision."""

        return next(
            iter(
                self.list_profile_intent_versions(
                    user_id=user_id,
                    scope_key=scope_key,
                    pref_scope=pref_scope,
                    active_only=True,
                    limit=1,
                )
            ),
            None,
        )

    def create_memory_review_export(
        self,
        *,
        user_id: str,
        items: Sequence[dict[str, object]],
    ) -> str:
        export_id = f"memory_export_{uuid4().hex}"
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO memory_review_exports(
                    export_id, user_id, items_json, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    export_id,
                    user_id,
                    json.dumps(items, ensure_ascii=False, separators=(",", ":")),
                    now,
                ),
            )
            connection.execute(
                """
                DELETE FROM memory_review_exports
                WHERE user_id = ? AND export_id NOT IN (
                    SELECT export_id FROM memory_review_exports
                    WHERE user_id = ?
                    ORDER BY created_at DESC, export_id DESC
                    LIMIT 20
                )
                """,
                (user_id, user_id),
            )
        return export_id

    def get_memory_review_export(
        self,
        *,
        user_id: str,
        export_id: str,
    ) -> tuple[dict[str, object], ...] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT items_json FROM memory_review_exports
                WHERE export_id = ? AND user_id = ?
                """,
                (export_id, user_id),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(str(row[0]))
        if not isinstance(payload, list) or not all(
            isinstance(item, dict) for item in payload
        ):
            raise ValueError("memory review export snapshot is invalid")
        return tuple(payload)

    @staticmethod
    def _redact_memory_review_exports_on(
        connection: sqlite3.Connection,
        *,
        user_id: str,
        update_ids: Sequence[str],
    ) -> int:
        """Remove exact tombstoned items from every retained review snapshot."""

        selected = frozenset(update_ids)
        if not selected:
            return 0
        removed = 0
        rows = connection.execute(
            """
            SELECT export_id, items_json
            FROM memory_review_exports
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchall()
        for export_id, raw_items in rows:
            items = json.loads(str(raw_items))
            if not isinstance(items, list):
                raise ValueError("memory review export snapshot is invalid")
            retained = [
                item
                for item in items
                if not (
                    isinstance(item, dict)
                    and str(item.get("update_id")) in selected
                )
            ]
            removed += len(items) - len(retained)
            if len(retained) != len(items):
                connection.execute(
                    """
                    UPDATE memory_review_exports
                    SET items_json = ?
                    WHERE export_id = ? AND user_id = ?
                    """,
                    (
                        json.dumps(
                            retained,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        export_id,
                        user_id,
                    ),
                )
        return removed

    def tombstone_free_text_preference(
        self,
        *,
        user_id: str,
        scope_key: str,
        pref_scope: str,
        update_id: str,
        expected_content_sha256: str,
        reason: str,
    ) -> bool:
        current = self.get_active_free_text_preference(
            user_id=user_id,
            update_id=update_id,
        )
        if (
            current is None
            or current.scope_key != scope_key
            or current.pref_scope != pref_scope
            or current.content_digest != expected_content_sha256
        ):
            return False
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT update_id, content_digest
                FROM career_intent_versions
                WHERE user_id = ? AND scope_key = ? AND pref_scope = ?
                  AND admission_status = 'active' AND superseded_at IS NULL
                """,
                (user_id, scope_key, pref_scope),
            ).fetchone()
            if row != (update_id, expected_content_sha256):
                return False
            lineage_update_ids = tuple(
                str(item[0])
                for item in connection.execute(
                    """
                    SELECT update_id
                    FROM career_intent_versions
                    WHERE user_id = ? AND scope_key = ? AND pref_scope = ?
                    ORDER BY revision, update_id
                    """,
                    (user_id, scope_key, pref_scope),
                ).fetchall()
            )
            connection.execute(
                """
                INSERT INTO intent_memory_tombstones(
                    user_id, scope_key, pref_scope, update_id,
                    content_digest, reason, deleted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, scope_key, pref_scope) DO UPDATE SET
                    update_id=excluded.update_id,
                    content_digest=excluded.content_digest,
                    reason=excluded.reason,
                    deleted_at=excluded.deleted_at
                """,
                (
                    user_id,
                    scope_key,
                    pref_scope,
                    update_id,
                    expected_content_sha256,
                    reason,
                    now,
                ),
            )
            connection.execute(
                "DELETE FROM free_text_preferences_fts WHERE update_id = ?",
                (update_id,),
            )
            self._redact_memory_review_exports_on(
                connection,
                user_id=user_id,
                update_ids=lineage_update_ids,
            )
        return True

    def search_free_text_preference_rankings(
        self,
        *,
        user_id: str,
        query: str,
        limit: int = 32,
        statuses: Sequence[str] = ("quarantined",),
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Return independent topic and statement FTS rankings for RRF.

        ``statuses`` defaults to quarantine because relevance first served the
        gate that lets a held candidate surface. Ranking confirmed preferences
        needs the same two rankings, so the caller asks again with
        ``("active",)``. Both statuses fit in one call, but they then share
        ``limit``, and the more numerous status takes every row.
        """

        if limit < 1 or limit > 100:
            raise ValueError("preference search limit must be between 1 and 100")
        selected = tuple(statuses)
        if not selected or not set(selected) <= {"active", "quarantined"}:
            raise ValueError("invalid free-text preference status")
        match_query = preference_fts_query(query)
        if match_query is None:
            return (), ()
        status_placeholders = ", ".join("?" for _ in selected)
        now = datetime.now(timezone.utc).isoformat()
        rankings = []
        with self._connect() as connection:
            for column, topic_weight, statement_weight in (
                ("topic_key", 8.0, 0.0),
                ("statement", 0.0, 8.0),
            ):
                rows = connection.execute(
                    f"""
                    SELECT search.update_id
                    FROM free_text_preferences_fts AS search
                    JOIN career_intent_versions AS intent
                      ON intent.update_id = search.update_id
                    LEFT JOIN memory_deleted_scopes AS deleted
                      ON deleted.user_id = intent.user_id
                     AND deleted.scope_key = intent.scope_key
                    WHERE free_text_preferences_fts MATCH ?
                      AND search.user_id = ?
                      AND intent.admission_status IN ({status_placeholders})
                      AND intent.superseded_at IS NULL
                      AND (
                            intent.valid_until IS NULL
                            OR intent.valid_until > ?
                          )
                      AND (
                            deleted.deleted_at IS NULL
                            OR intent.valid_from > deleted.deleted_at
                          )
                    ORDER BY bm25(
                        free_text_preferences_fts,
                        0.0, 0.0, {topic_weight}, {statement_weight}
                    ), intent.valid_from DESC
                    LIMIT ?
                    """,
                    (
                        f"{column} : ({match_query})",
                        user_id,
                        *selected,
                        now,
                        limit,
                    ),
                ).fetchall()
                rankings.append(tuple(str(row[0]) for row in rows))
        return rankings[0], rankings[1]

    @staticmethod
    def _index_free_text_preference_on(
        connection: sqlite3.Connection,
        version: IntentMemoryVersion,
    ) -> None:
        connection.execute(
            "DELETE FROM free_text_preferences_fts WHERE update_id = ?",
            (version.update_id,),
        )
        connection.execute(
            """
            INSERT INTO free_text_preferences_fts(
                update_id, user_id, topic_key, statement
            ) VALUES (?, ?, ?, ?)
            """,
            (
                version.update_id,
                version.user_id,
                preference_topic_key(version.scope_key),
                version.value,
            ),
        )

    def confirm_free_text_preference(
        self,
        *,
        user_id: str,
        update_id: str,
        conversation_id: str | None = None,
        job_posting_id: str | None = None,
        scope_choice: str | None = None,
        scope_domain: str | None = None,
    ) -> IntentMemoryVersion | None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            pending = next(
                (
                    item
                    for item in list_intent_versions(
                        connection,
                        user_id=user_id,
                    )
                    if item.update_id == update_id
                    and item.pref_scope.startswith("freeform")
                    and item.superseded_at is None
                    and item.admission_status == "quarantined"
                ),
                None,
            )
            if pending is None:
                return None
            deleted_at = self._memory_scope_deleted_at_on(
                connection,
                user_id=user_id,
                scope_key=pending.scope_key,
            )
            if deleted_at is not None and pending.valid_from <= deleted_at:
                return None
            observed_at = datetime.now(timezone.utc)
            assignment = (
                preference_storage_assignment(
                    scope_choice,
                    observed_at=observed_at,
                    conversation_id=conversation_id or "current",
                    job_posting_id=job_posting_id,
                    role_domain=scope_domain,
                    statement=pending.value,
                )
                if scope_choice is not None
                else None
            )
            _, active = capture_intent_version(
                connection,
                candidate=IntentCaptureCandidate(
                    user_id=user_id,
                    scope_key=pending.scope_key,
                    value=pending.value,
                    source="user_input:confirmed_free_text_preference",
                    pref_scope=(
                        assignment.pref_scope
                        if assignment is not None
                        else pending.pref_scope
                    ),
                    timescale=(
                        assignment.timescale
                        if assignment is not None
                        else pending.timescale
                    ),
                    layer=(
                        assignment.layer
                        if assignment is not None
                        else pending.layer
                    ),
                    valid_until=(
                        assignment.valid_until
                        if assignment is not None
                        else pending.valid_until
                    ),
                    confidence=1.0,
                    semantic_stance=pending.semantic_stance,
                    observed_at=observed_at,
                ),
            )
            if active is not None:
                self._index_free_text_preference_on(connection, active)
            if (
                active is not None
                and pending.superseded_at is None
                and pending.update_id != active.update_id
            ):
                connection.execute(
                    """
                    UPDATE career_intent_versions
                    SET superseded_at = ?, superseded_by = ?
                    WHERE update_id = ? AND superseded_at IS NULL
                    """,
                    (
                        observed_at.isoformat(),
                        active.update_id,
                        pending.update_id,
                    ),
                )
            if active is not None and deleted_at is not None:
                # Reopening a tombstoned scope must not resurrect older rows
                # from sibling ownership tracks. Close every pre-deletion
                # current row before removing the read-time tombstone.
                connection.execute(
                    """
                    UPDATE career_intent_versions
                    SET superseded_at = ?, superseded_by = ?
                    WHERE user_id = ? AND scope_key = ?
                      AND update_id != ?
                      AND superseded_at IS NULL
                      AND valid_from <= ?
                    """,
                    (
                        observed_at.isoformat(),
                        active.update_id,
                        user_id,
                        pending.scope_key,
                        active.update_id,
                        deleted_at.isoformat(),
                    ),
                )
                connection.execute(
                    "DELETE FROM memory_deleted_scopes WHERE user_id = ? AND scope_key = ?",
                    (user_id, pending.scope_key),
                )
            return active

    def confirm_free_text_preference_amendment(
        self,
        *,
        user_id: str,
        base_update_id: str,
        expected_content_sha256: str,
        statement: str,
    ) -> IntentMemoryVersion | None:
        """Append an edited review line only after its proposal was confirmed."""

        observed_at = datetime.now(timezone.utc)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = next(
                (
                    item
                    for item in list_intent_versions(connection, user_id=user_id)
                    if item.update_id == base_update_id
                    and item.pref_scope.startswith("freeform")
                    and item.admission_status == "active"
                    and item.superseded_at is None
                ),
                None,
            )
            if current is None or current.content_digest != expected_content_sha256:
                return None
            mutation = extract_free_text_preference(statement)
            stance = (
                mutation.stance
                if mutation is not None and mutation.action == "quarantine"
                else current.semantic_stance
            )
            amended = append_intent_version(
                connection,
                user_id=user_id,
                scope_key=current.scope_key,
                value=statement,
                source="user_input:confirmed_memory_review",
                valid_from=observed_at,
                valid_until=current.valid_until,
                pref_scope=current.pref_scope,
                timescale=current.timescale,
                layer=current.layer,
                last_corroborated_at=observed_at,
                base_confidence=1.0,
                admission_status="active",
                capture_action="revise",
                semantic_stance=stance,
            )
            self._index_free_text_preference_on(connection, amended)
            connection.execute(
                """
                DELETE FROM intent_memory_tombstones
                WHERE user_id = ? AND scope_key = ? AND pref_scope = ?
                """,
                (user_id, amended.scope_key, amended.pref_scope),
            )
            return amended

    def free_text_preference_deleted_at(
        self,
        *,
        user_id: str,
        scope_key: str,
    ) -> datetime | None:
        with self._connect() as connection:
            return self._memory_scope_deleted_at_on(
                connection,
                user_id=user_id,
                scope_key=scope_key,
            )

    @staticmethod
    def _memory_scope_deleted_at_on(
        connection: sqlite3.Connection,
        *,
        user_id: str,
        scope_key: str,
    ) -> datetime | None:
        row = connection.execute(
            "SELECT deleted_at FROM memory_deleted_scopes WHERE user_id = ? AND scope_key = ?",
            (user_id, scope_key),
        ).fetchone()
        return datetime.fromisoformat(str(row[0])) if row is not None else None

    def free_text_preference_tombstone_matches(
        self,
        *,
        user_id: str,
        scope_key: str,
        pref_scope: str,
        update_id: str,
        content_digest: str,
    ) -> bool:
        """Recognize a committed delete so failed cleanup can be retried."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM intent_memory_tombstones
                WHERE user_id = ? AND scope_key = ? AND pref_scope = ?
                  AND update_id = ? AND content_digest = ?
                """,
                (
                    user_id,
                    scope_key,
                    pref_scope,
                    update_id,
                    content_digest,
                ),
            ).fetchone()
        return row is not None

    @staticmethod
    def _free_text_preference_scope(topic_key: str) -> str:
        return preference_scope_key(topic_key)

    def get_owner_settings(self, user_id: str) -> OwnerSettingsContext | None:
        return self._get_single("owner_settings_context", user_id, OwnerSettingsContext)

    def update_owner_settings(
        self,
        *,
        user_id: str,
        desired: OwnerSettingsContext,
        expected_revision: int,
        actor_type: Literal["api_key", "cli", "system", "confirmed_agent_proposal"],
        actor_id: str,
    ) -> OwnerSettingsContext:
        """Compare-and-swap one owner document and append the same transaction's audit."""

        now = datetime.now(timezone.utc)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if actor_type == "confirmed_agent_proposal":
                replay = connection.execute(
                    "SELECT after_json FROM owner_settings_events WHERE user_id=? "
                    "AND actor_type=? AND actor_id=? ORDER BY revision DESC LIMIT 1",
                    (user_id, actor_type, actor_id),
                ).fetchone()
                if replay is not None:
                    return OwnerSettingsContext.model_validate_json(replay[0])
            row = connection.execute(
                "SELECT payload FROM owner_settings_context WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            before = (
                OwnerSettingsContext.model_validate_json(row[0])
                if row
                else OwnerSettingsContext()
            )
            if before.revision != expected_revision:
                raise OwnerSettingsConflictError(
                    f"settings revision is {before.revision}, not {expected_revision}"
                )
            soft_changed = desired.preferences != before.preferences
            policy_changed = desired.behavior_policy.model_copy(
                update={"revision": before.behavior_policy.revision}
            ) != before.behavior_policy
            changed_fields = []
            if soft_changed:
                changed_fields.append("preferences")
            if policy_changed:
                changed_fields.append("behavior_policy")
            if not changed_fields:
                return before
            after = desired.model_copy(
                update={
                    "revision": before.revision + 1,
                    "behavior_policy": desired.behavior_policy.model_copy(
                        update={
                            "revision": before.behavior_policy.revision
                            + int(policy_changed)
                        }
                    ),
                }
            )
            connection.execute(
                "INSERT INTO owner_settings_context(user_id, payload, updated_at) "
                "VALUES (?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET "
                "payload=excluded.payload, updated_at=excluded.updated_at",
                (user_id, after.model_dump_json(), now.isoformat()),
            )
            connection.execute(
                "INSERT INTO owner_settings_events(user_id, revision, policy_revision, "
                "actor_type, actor_id, changed_fields_json, before_json, after_json, changed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    user_id,
                    after.revision,
                    after.behavior_policy.revision,
                    actor_type,
                    actor_id,
                    json.dumps(changed_fields),
                    before.model_dump_json(),
                    after.model_dump_json(),
                    now.isoformat(),
                ),
            )
        os.chmod(self.path, 0o600)
        return after

    def list_owner_settings_events(
        self, *, user_id: str, limit: int = 100
    ) -> tuple[OwnerSettingsEvent, ...]:
        if limit < 1 or limit > 500:
            raise ValueError("limit must be between 1 and 500")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT event_id, user_id, revision, policy_revision, actor_type, "
                "actor_id, changed_fields_json, before_json, after_json, changed_at "
                "FROM owner_settings_events WHERE user_id = ? "
                "ORDER BY revision DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return tuple(
            OwnerSettingsEvent(
                event_id=row[0], user_id=row[1], revision=row[2],
                policy_revision=row[3], actor_type=row[4], actor_id=row[5],
                changed_fields=tuple(json.loads(row[6])),
                before=OwnerSettingsContext.model_validate_json(row[7]),
                after=OwnerSettingsContext.model_validate_json(row[8]),
                changed_at=datetime.fromisoformat(row[9]),
            )
            for row in rows
        )

    # Compatibility for call sites/tests predating the semantic split.
    def upsert_preferences(self, user_id: str, preferences: OwnerSettingsContext) -> None:
        current = self.get_owner_settings(user_id) or OwnerSettingsContext()
        self.update_owner_settings(
            user_id=user_id,
            desired=preferences,
            expected_revision=current.revision,
            actor_type="system",
            actor_id="legacy-upsert",
        )

    def get_task(self, user_id: str, conversation_id: str) -> ConversationTaskState | None:
        with self._connect() as connection:
            row = connection.execute("SELECT payload FROM conversation_task_state WHERE user_id = ? AND conversation_id = ?", (user_id, conversation_id)).fetchone()
        return ConversationTaskState.model_validate_json(row[0]) if row else None

    def upsert_task(
        self,
        *,
        user_id: str,
        conversation_id: str,
        task: ConversationTaskState,
        episode_drafts: tuple[CareerEpisodeDraft, ...] = (),
    ) -> None:
        """Persist routing ownership and any episode at the same task seam."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO conversation_task_state(user_id, conversation_id, payload, updated_at) VALUES (?, ?, ?, ?) ON CONFLICT(user_id, conversation_id) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at",
                (user_id, conversation_id, task.model_dump_json(), now),
            )
            for draft in episode_drafts:
                SQLiteCareerEpisodeStore.upsert_on(connection, draft)
        os.chmod(self.path, 0o600)

    def list_message_records(
        self,
        user_id: str,
        conversation_id: str,
        *,
        limit: int,
        after_sequence: int = 0,
    ) -> tuple[StoredConversationMessage, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT sequence, payload
                FROM conversation_messages
                WHERE user_id = ? AND conversation_id = ?
                  AND sequence > ?
                  {_NOT_SUPPRESSED_SQL}
                ORDER BY sequence DESC
                LIMIT ?
                """,
                (
                    user_id,
                    conversation_id,
                    after_sequence,
                    limit,
                ),
            ).fetchall()
        return tuple(
            StoredConversationMessage(
                sequence=row[0],
                message=ConversationMessageContext.model_validate_json(row[1]),
            )
            for row in reversed(rows)
        )

    def list_delivered_body_references(
        self,
        user_id: str,
        conversation_id: str,
        *,
        from_sequence: int,
    ) -> tuple[StoredDeliveredBodyReference, ...]:
        """The kept bodies on rows at or after ``from_sequence``, oldest first.

        References only: a conversation can hold many bodies of thousands of
        characters each, and restoring the transcript should not pull them all.
        Bodies on suppressed rows are left out the way the rows themselves are.
        """
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT bodies.body_id, bodies.sequence, bodies.kind, bodies.title
                FROM conversation_delivered_bodies AS bodies
                WHERE bodies.user_id = ? AND bodies.conversation_id = ?
                  AND bodies.sequence >= ?
                  AND EXISTS (
                      SELECT 1 FROM conversation_messages AS message
                      WHERE message.user_id = bodies.user_id
                        AND message.conversation_id = bodies.conversation_id
                        AND message.sequence = bodies.sequence
                  )
                  AND NOT EXISTS (
                        SELECT 1 FROM memory_deletion_message_suppressions AS hidden
                        WHERE hidden.user_id = bodies.user_id
                          AND hidden.conversation_id = bodies.conversation_id
                          AND hidden.sequence = bodies.sequence
                      )
                ORDER BY bodies.sequence ASC, bodies.rowid ASC
                """,
                (user_id, conversation_id, from_sequence),
            ).fetchall()
        return tuple(
            StoredDeliveredBodyReference(
                body_id=row[0], sequence=row[1], kind=row[2], title=row[3]
            )
            for row in rows
        )

    def get_delivered_body(
        self, user_id: str, body_id: str
    ) -> StoredDeliveredBody | None:
        """One kept body, or nothing once its row has been suppressed."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT bodies.body_id, bodies.conversation_id, bodies.sequence,
                       bodies.kind, bodies.title, bodies.body, bodies.created_at,
                       bodies.retention, bodies.source_json, bodies.dependencies_json
                FROM conversation_delivered_bodies AS bodies
                WHERE bodies.user_id = ? AND bodies.body_id = ?
                  AND EXISTS (
                      SELECT 1 FROM conversation_messages AS message
                      WHERE message.user_id = bodies.user_id
                        AND message.conversation_id = bodies.conversation_id
                        AND message.sequence = bodies.sequence
                  )
                  AND NOT EXISTS (
                        SELECT 1 FROM memory_deletion_message_suppressions AS hidden
                        WHERE hidden.user_id = bodies.user_id
                          AND hidden.conversation_id = bodies.conversation_id
                          AND hidden.sequence = bodies.sequence
                      )
                """,
                (user_id, body_id),
            ).fetchone()
        if row is None:
            return None
        return StoredDeliveredBody(
            body_id=row[0],
            conversation_id=row[1],
            sequence=row[2],
            kind=row[3],
            title=row[4],
            body=row[5],
            created_at=datetime.fromisoformat(row[6]),
            retention=row[7],
            source=TypeAdapter(DeliveredBodySource).validate_json(row[8]) if row[8] else None,
            dependencies=TypeAdapter(tuple[BodyDependency, ...]).validate_json(row[9] or "[]"),
        )

    def purge_delivered_body_dependency(
        self, *, user_id: str, dependency: BodyDependency
    ) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT OR IGNORE INTO delivered_body_deleted_dependencies VALUES (?, ?, ?)",
                (user_id, dependency.kind, dependency.resource_id),
            )
            rows = connection.execute(
                """
                SELECT body_id, conversation_id, sequence
                FROM conversation_delivered_bodies AS bodies
                WHERE user_id = ?
                  AND (dependencies_json IS NULL OR EXISTS (
                      SELECT 1 FROM json_each(bodies.dependencies_json) AS dependency
                      WHERE json_extract(dependency.value, '$.kind') = ?
                        AND json_extract(dependency.value, '$.resource_id') = ?
                  ))
                """,
                (user_id, dependency.kind, dependency.resource_id),
            ).fetchall()
            connection.executemany(
                "DELETE FROM conversation_delivered_bodies WHERE user_id = ? AND body_id = ?",
                ((user_id, row[0]) for row in rows),
            )
            self._redact_receipts_for_rows(
                connection, user_id, {(str(row[1]), int(row[2])) for row in rows}
            )

    def list_messages(self, user_id: str, conversation_id: str, *, limit: int, after_sequence: int = 0) -> tuple[ConversationMessageContext, ...]:
        return tuple(
            record.message
            for record in self.list_message_records(
                user_id,
                conversation_id,
                limit=limit,
                after_sequence=after_sequence,
            )
        )

    def read_conversation_span(
        self,
        *,
        user_id: str,
        conversation_id: str,
        from_sequence: int,
        through_sequence: int,
        query: str | None = None,
    ) -> ConversationSpanView:
        """Read only rows inside the requested inclusive span, oldest first.

        Without a query, a span outside this owned conversation returns no rows
        and a large span returns its earliest rows plus the honest uncapped
        count. With a query, exact owned rows are ranked by term matches before
        the same output ceiling is applied.

        Resource references on returned rows cross with the text. This lets a
        paged-in historical turn recover the same durable handles it originally
        carried instead of returning prose that points to an unreachable report.
        """
        if from_sequence < 1 or through_sequence < from_sequence:
            raise ValueError("invalid conversation span")
        with self._connect() as connection:
            total = int(
                connection.execute(
                    f"""
                    SELECT COUNT(*) FROM conversation_messages
                    WHERE user_id = ? AND conversation_id = ?
                      AND sequence BETWEEN ? AND ?
                      {_NOT_SUPPRESSED_SQL}
                    """,
                    (user_id, conversation_id, from_sequence, through_sequence),
                ).fetchone()[0]
            )
            if query is None:
                rows = connection.execute(
                    f"""
                    SELECT sequence, payload FROM conversation_messages
                    WHERE user_id = ? AND conversation_id = ?
                      AND sequence BETWEEN ? AND ?
                      {_NOT_SUPPRESSED_SQL}
                    ORDER BY sequence
                    LIMIT ?
                    """,
                    (
                        user_id,
                        conversation_id,
                        from_sequence,
                        through_sequence,
                        MAX_CONVERSATION_SPAN_MESSAGES,
                    ),
                ).fetchall()
            else:
                rows = connection.execute(
                    f"""
                    SELECT sequence, payload FROM conversation_messages
                    WHERE user_id = ? AND conversation_id = ?
                      AND sequence BETWEEN ? AND ?
                      {_NOT_SUPPRESSED_SQL}
                    ORDER BY sequence
                    """,
                    (
                        user_id,
                        conversation_id,
                        from_sequence,
                        through_sequence,
                    ),
                ).fetchall()
        if query is not None:
            normalized_query = query.casefold().strip()
            terms = tuple(
                dict.fromkeys(
                    (
                        normalized_query,
                        *re.findall(r"[\w\u3400-\u9fff]+", normalized_query),
                    )
                )
            )
            scored = []
            for row in rows:
                message = ConversationMessageContext.model_validate_json(row[1])
                searchable = message.content.casefold()
                score = sum(searchable.count(term) for term in terms if term)
                if score:
                    scored.append((score, row[0], row[1]))
            total = len(scored)
            # Prefer the strongest and newest matches, then restore chronology
            # in the returned window so adjacent user/assistant facts read as
            # a conversation rather than a search ranking.
            rows = [
                (sequence, payload)
                for _, sequence, payload in sorted(
                    scored, key=lambda item: (item[0], item[1]), reverse=True
                )[:MAX_CONVERSATION_SPAN_MESSAGES]
            ]
            rows.sort(key=lambda row: row[0])
        messages = []
        resource_refs = []
        seen_resource_ids: set[str] = set()
        for row in rows:
            message = ConversationMessageContext.model_validate_json(row[1])
            for reference in message.resource_refs:
                if reference.resource_id in seen_resource_ids:
                    continue
                seen_resource_ids.add(reference.resource_id)
                if len(resource_refs) < MAX_CONVERSATION_SPAN_RESOURCE_REFS:
                    resource_refs.append(reference)
            messages.append(
                ConversationSpanMessage(
                    sequence=row[0],
                    role=message.role,
                    content=message.content[:SUMMARY_SOURCE_MAX_CHARS],
                    content_clipped=(
                        len(message.content) > SUMMARY_SOURCE_MAX_CHARS
                    ),
                    created_at=message.created_at,
                )
            )
        return ConversationSpanView(
            from_sequence=from_sequence,
            through_sequence=through_sequence,
            returned=len(messages),
            total=total,
            resource_ref_total=len(seen_resource_ids),
            resource_refs=tuple(resource_refs),
            messages=tuple(messages),
        )

    def list_archived_resource_messages(
        self,
        *,
        user_id: str,
        conversation_id: str,
        through_sequence: int,
        limit: int,
    ) -> tuple[ConversationMessageContext, ...]:
        """Resource-backed messages the recent window has already scrolled past.

        The catalogue of past reports is these rows, not a second copy of them.
        A stored catalogue would be durable state that has to be kept in step
        with the messages it describes; this cannot drift, because it is the
        messages.

        Filtered in SQL on the stored payload rather than in Python so a long
        conversation does not have to be read back to find the few turns that
        delivered something. Newest first, capped, then reversed so the caller
        gets chronological order.
        """
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT payload FROM conversation_messages
                WHERE user_id = ? AND conversation_id = ? AND sequence <= ?
                  {_NOT_SUPPRESSED_SQL}
                  AND json_array_length(
                        COALESCE(json_extract(payload, '$.resource_refs'), json_array())
                      ) > 0
                ORDER BY sequence DESC LIMIT ?
                """,
                (
                    user_id,
                    conversation_id,
                    through_sequence,
                    limit,
                ),
            ).fetchall()
        return tuple(
            ConversationMessageContext.model_validate_json(row[0])
            for row in reversed(rows)
        )

    def count_archived_resources(
        self, *, user_id: str, conversation_id: str, through_sequence: int
    ) -> int:
        """How many resources the catalogue would hold if it were not capped.

        Counted rather than inferred from the capped list, which can only ever
        say "at least twelve". The model needs the difference: a list of twelve
        with nothing else said reads as the complete set, so a report that
        scrolled past the cap looks like it should be in there somewhere.

        Counts references, not messages — one turn can deliver two reports.
        """
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT COALESCE(SUM(json_array_length(
                    COALESCE(json_extract(payload, '$.resource_refs'), json_array())
                )), 0)
                FROM conversation_messages
                WHERE user_id = ? AND conversation_id = ? AND sequence <= ?
                  {_NOT_SUPPRESSED_SQL}
                """,
                (
                    user_id,
                    conversation_id,
                    through_sequence,
                ),
            ).fetchone()
        return int(row[0])

    def get_conversation_summary(
        self, *, user_id: str, conversation_id: str
    ) -> StoredConversationSummary | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT content_json, through_sequence, updated_at
                FROM conversation_summaries
                WHERE user_id = ? AND conversation_id = ?
                """,
                (user_id, conversation_id),
            ).fetchone()
        if row is None:
            return None
        return StoredConversationSummary(
            user_id=user_id,
            conversation_id=conversation_id,
            content=ConversationSummaryContent.model_validate_json(row[0]),
            through_sequence=row[1],
            updated_at=row[2],
        )

    def purge_derived_memory(
        self,
        *,
        user_id: str,
        scope_key: str,
        lineage_markers: Sequence[str] = (),
        update_ids: Sequence[str] = (),
    ) -> dict[str, int]:
        """Invalidate derivations that observed one tombstoned lineage.

        New rows carry an explicit scope binding. ``lineage_markers`` is a
        SQL-side migration fallback for messages written before those
        bindings existed; only opaque detail/source/lineage refs are accepted.
        Free-text claims are never used as scan keys.

        Original messages remain reconstructable: affected summaries are
        dropped so the next compaction rebuilds from unsuppressed rows.
        Episodes lose this scope association and are deleted only when no other
        live memory association remains.
        """

        now = datetime.now(timezone.utc).isoformat()
        markers = _usable_lineage_markers(lineage_markers)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            binding_keys = (scope_key,)
            if "#" not in scope_key:
                binding_keys = tuple(
                    dict.fromkeys(
                        (
                            scope_key,
                            *(
                                str(row[0])
                                for row in connection.execute(
                                    """
                                    SELECT DISTINCT scope_key
                                    FROM conversation_message_memory_bindings
                                    WHERE user_id = ?
                                      AND (
                                            scope_key = ?
                                            OR substr(
                                                scope_key, 1, length(?) + 1
                                            ) = ? || '#'
                                          )
                                    """,
                                    (
                                        user_id,
                                        scope_key,
                                        scope_key,
                                        scope_key,
                                    ),
                                ).fetchall()
                            ),
                            *(
                                str(row[0])
                                for row in connection.execute(
                                    """
                                    SELECT DISTINCT scope_key
                                    FROM career_episode_memory_bindings
                                    WHERE user_id = ?
                                      AND (
                                            scope_key = ?
                                            OR substr(
                                                scope_key, 1, length(?) + 1
                                            ) = ? || '#'
                                          )
                                    """,
                                    (
                                        user_id,
                                        scope_key,
                                        scope_key,
                                        scope_key,
                                    ),
                                ).fetchall()
                            ),
                        )
                    )
                )
            connection.execute(
                """
                INSERT INTO memory_deleted_scopes(user_id, scope_key, deleted_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id, scope_key) DO NOTHING
                """,
                (user_id, scope_key, now),
            )
            connection.execute(
                """
                DELETE FROM free_text_preferences_fts
                WHERE update_id IN (
                    SELECT update_id
                    FROM career_intent_versions
                    WHERE user_id = ? AND scope_key = ?
                )
                """,
                (user_id, scope_key),
            )
            bound_rows = {
                (str(row[0]), int(row[1]))
                for row in connection.execute(
                    f"""
                    SELECT conversation_id, sequence
                    FROM conversation_message_memory_bindings
                    WHERE user_id = ?
                      AND scope_key IN (
                          {",".join("?" for _ in binding_keys)}
                      )
                    """,
                    (user_id, *binding_keys),
                ).fetchall()
            }
            if markers:
                marker_predicate = " OR ".join(
                    "instr(COALESCE(json_extract(messages.payload, '$.content'), ''), ?) > 0"
                    for _ in markers
                )
                for conversation_id, sequence in connection.execute(
                    f"""
                    SELECT conversation_id, sequence
                    FROM conversation_messages AS messages
                    WHERE user_id = ?
                      AND NOT EXISTS (
                            SELECT 1
                            FROM conversation_message_memory_bindings AS binding
                            WHERE binding.user_id = messages.user_id
                              AND binding.conversation_id = messages.conversation_id
                              AND binding.sequence = messages.sequence
                          )
                      AND ({marker_predicate})
                    """,
                    (user_id, *markers),
                ).fetchall():
                    bound_rows.add((str(conversation_id), int(sequence)))
            connection.executemany(
                """
                INSERT OR IGNORE INTO memory_deletion_message_suppressions(
                    user_id, conversation_id, sequence, scope_key, suppressed_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (user_id, conversation_id, sequence, scope_key, now)
                    for conversation_id, sequence in sorted(bound_rows)
                ),
            )
            affected_conversations = tuple(
                sorted({conversation_id for conversation_id, _ in bound_rows})
            )
            connection.execute(
                """
                DELETE FROM conversation_delivered_bodies
                WHERE user_id = ? AND EXISTS (
                    SELECT 1 FROM memory_deletion_message_suppressions AS hidden
                    WHERE hidden.user_id = conversation_delivered_bodies.user_id
                      AND hidden.conversation_id = conversation_delivered_bodies.conversation_id
                      AND hidden.sequence = conversation_delivered_bodies.sequence
                )
                """,
                (user_id,),
            )
            self._redact_receipts_for_rows(connection, user_id, sorted(bound_rows))
            summary_count = 0
            if affected_conversations:
                placeholders = ",".join("?" for _ in affected_conversations)
                cursor = connection.execute(
                    f"""
                    DELETE FROM conversation_summaries
                    WHERE user_id = ?
                      AND conversation_id IN ({placeholders})
                      AND EXISTS (
                            SELECT 1
                            FROM memory_deletion_message_suppressions AS hidden
                            WHERE hidden.user_id = conversation_summaries.user_id
                              AND hidden.conversation_id = conversation_summaries.conversation_id
                              AND hidden.scope_key = ?
                              AND hidden.sequence <= conversation_summaries.through_sequence
                          )
                    """,
                    (user_id, *affected_conversations, scope_key),
                )
                summary_count = int(cursor.rowcount)
                # Constraints are summary-derived text, so a tombstone has to
                # reach them too. Which archive rows came from the deleted
                # scope is not recorded, so the whole live set for a dropped
                # summary goes and the rebuild re-derives it from unsuppressed
                # messages. Retirements are kept: they are the record of a
                # user decision, not a derivation, and dropping them would let
                # the rebuild resurrect a retired constraint.
                connection.execute(
                    f"""
                    DELETE FROM conversation_constraint_archive
                    WHERE user_id = ?
                      AND conversation_id IN ({placeholders})
                      AND status IN ('active', 'omitted')
                      AND NOT EXISTS (
                            SELECT 1 FROM conversation_summaries AS kept
                            WHERE kept.user_id
                                  = conversation_constraint_archive.user_id
                              AND kept.conversation_id
                                  = conversation_constraint_archive.conversation_id
                          )
                    """,
                    (user_id, *affected_conversations),
                )
            episode_count = sum(
                SQLiteCareerEpisodeStore.delete_for_scope_on(
                    connection,
                    user_id=user_id,
                    scope_key=binding_key,
                )
                for binding_key in binding_keys
            )
            review_item_count = self._redact_memory_review_exports_on(
                connection,
                user_id=user_id,
                update_ids=update_ids,
            )
        return {
            "conversation_fragments": len(bound_rows),
            "conversation_summaries": summary_count,
            "career_episodes": episode_count,
            "affected_conversations": len(affected_conversations),
            "memory_review_items": review_item_count,
        }

    def list_messages_after(
        self,
        *,
        user_id: str,
        conversation_id: str,
        after_sequence: int,
        limit: int,
    ) -> tuple[SummaryMessage, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT sequence, payload
                FROM conversation_messages
                WHERE user_id = ? AND conversation_id = ?
                  AND sequence > ?
                  {_NOT_SUPPRESSED_SQL}
                ORDER BY sequence
                LIMIT ?
                """,
                (
                    user_id,
                    conversation_id,
                    after_sequence,
                    limit,
                ),
            ).fetchall()
        messages = []
        for row in rows:
            message = ConversationMessageContext.model_validate_json(row[1])
            messages.append(
                SummaryMessage(
                    sequence=row[0],
                    role=message.role,
                    # Stored conversation text has a much larger abuse limit
                    # than a summary request. Keep those two policies
                    # independent so one long message can never make every
                    # future load of this conversation fail validation.
                    content=message.content[:SUMMARY_SOURCE_MAX_CHARS],
                )
            )
        return tuple(messages)

    def list_conversation_constraints(
        self,
        *,
        user_id: str,
        conversation_id: str,
        statuses: Sequence[str] = ("active", "omitted", "retired"),
    ) -> tuple[ArchivedConversationConstraint, ...]:
        """Read the constraint ledger in first-seen order."""

        if not statuses:
            return ()
        placeholders = ", ".join("?" for _ in statuses)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT seq, constraint_text, status, first_seen_at,
                       status_changed_at
                FROM conversation_constraint_archive
                WHERE user_id = ? AND conversation_id = ?
                  AND status IN ({placeholders})
                ORDER BY seq
                """,
                (user_id, conversation_id, *statuses),
            ).fetchall()
        return tuple(
            ArchivedConversationConstraint(
                seq=row[0],
                text=row[1],
                status=row[2],
                first_seen_at=row[3],
                status_changed_at=row[4],
            )
            for row in rows
        )

    def retire_conversation_constraint(
        self,
        *,
        user_id: str,
        conversation_id: str,
        constraint_text: str,
    ) -> bool:
        """Retire one constraint by exact text.

        Returns ``False`` when the text is not a live constraint of this
        conversation, so a stale readback cannot retire something the user
        never saw. Retirement is idempotent only in the sense that a second
        attempt reports ``False`` rather than silently succeeding.
        """

        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            changed = connection.execute(
                """
                UPDATE conversation_constraint_archive
                SET status = 'retired', status_changed_at = ?
                WHERE user_id = ? AND conversation_id = ?
                  AND constraint_sha256 = ?
                  AND status IN ('active', 'omitted')
                """,
                (
                    now,
                    user_id,
                    conversation_id,
                    _constraint_digest(constraint_text),
                ),
            ).rowcount
            if not changed:
                return False
            # The summary still shows the retired constraint until the next
            # compaction rewrites it, and compaction only runs once a full
            # batch accumulates. Drop it from the stored projection now so the
            # retirement is visible on the very next turn.
            row = connection.execute(
                """
                SELECT content_json FROM conversation_summaries
                WHERE user_id = ? AND conversation_id = ?
                """,
                (user_id, conversation_id),
            ).fetchone()
            if row is None:
                return True
            content = ConversationSummaryContent.model_validate_json(row[0])
            if constraint_text not in content.active_constraints:
                return True
            content = content.model_copy(
                update={
                    "active_constraints": tuple(
                        text
                        for text in content.active_constraints
                        if text != constraint_text
                    )
                }
            )
            connection.execute(
                """
                UPDATE conversation_summaries
                SET content_json = ?, updated_at = ?
                WHERE user_id = ? AND conversation_id = ?
                """,
                (content.model_dump_json(), now, user_id, conversation_id),
            )
        return True

    def compact_conversation_summary(
        self,
        *,
        user_id: str,
        conversation_id: str,
        expected_previous_through_sequence: int,
        content: ConversationSummaryContent,
        through_sequence: int,
        omitted_constraints: Sequence[str] = (),
        preference_candidates: Sequence[
            DistilledFreeTextPreferenceCandidate
        ] = (),
    ) -> bool:
        """Store one summary and reconcile the constraint ledger with it.

        ``content.active_constraints`` and ``omitted_constraints`` are the two
        halves of the caller's cap decision. Both land in the same transaction
        as the summary, so the visible projection and the ledger behind it
        cannot disagree.
        """

        if through_sequence <= expected_previous_through_sequence:
            raise ValueError("conversation summary must advance its covered sequence")
        observed_at = datetime.now(timezone.utc)
        now = observed_at.isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT through_sequence FROM conversation_summaries
                WHERE user_id = ? AND conversation_id = ?
                """,
                (user_id, conversation_id),
            ).fetchone()
            current_through = row[0] if row else 0
            if current_through != expected_previous_through_sequence:
                return False
            retired = {
                str(retired_row[0])
                for retired_row in connection.execute(
                    """
                    SELECT constraint_sha256
                    FROM conversation_constraint_archive
                    WHERE user_id = ? AND conversation_id = ?
                      AND status = 'retired'
                    """,
                    (user_id, conversation_id),
                ).fetchall()
            }
            visible: list[str] = []
            for status, texts in (
                ("active", tuple(content.active_constraints)),
                ("omitted", tuple(omitted_constraints)),
            ):
                for text in texts:
                    digest = _constraint_digest(text)
                    # The caller read the ledger outside this transaction, so a
                    # retirement may have landed in between. Retirement wins;
                    # a stale read never resurrects one.
                    if digest in retired:
                        continue
                    if status == "active":
                        visible.append(text)
                    connection.execute(
                        """
                        INSERT INTO conversation_constraint_archive(
                            user_id, conversation_id, constraint_sha256,
                            constraint_text, status, first_seen_at,
                            status_changed_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(user_id, conversation_id, constraint_sha256)
                            DO UPDATE SET
                                status=excluded.status,
                                status_changed_at=excluded.status_changed_at
                            WHERE status != excluded.status
                        """,
                        (
                            user_id,
                            conversation_id,
                            digest,
                            text,
                            status,
                            now,
                            now,
                        ),
                    )
            if len(visible) != len(content.active_constraints):
                content = content.model_copy(
                    update={"active_constraints": tuple(visible)}
                )
            connection.execute(
                """
                INSERT INTO conversation_summaries(
                    user_id, conversation_id, content_json,
                    through_sequence, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, conversation_id) DO UPDATE SET
                    content_json=excluded.content_json,
                    through_sequence=excluded.through_sequence,
                    updated_at=excluded.updated_at
                """,
                (
                    user_id,
                    conversation_id,
                    content.model_dump_json(),
                    through_sequence,
                    now,
                ),
            )
            task_row = connection.execute(
                """
                SELECT payload FROM conversation_task_state
                WHERE user_id = ? AND conversation_id = ?
                """,
                (user_id, conversation_id),
            ).fetchone()
            task = (
                ConversationTaskState.model_validate_json(task_row[0])
                if task_row is not None
                else None
            )
            for distilled in preference_candidates:
                scope_key = self._free_text_preference_scope(
                    distilled.topic_key
                )
                ownership = (
                    "person_default"
                    if distilled.ownership == "ask"
                    else distilled.ownership
                )
                assignment = preference_storage_assignment(
                    ownership,
                    observed_at=observed_at,
                    conversation_id=conversation_id,
                    job_posting_id=(
                        task.active_job_posting_id
                        if task is not None
                        else None
                    ),
                    role_domain=distilled.scope_domain,
                    valid_for_days=distilled.valid_for_days,
                    statement=distilled.statement,
                )
                version = self._capture_free_text_candidate_on(
                    connection,
                    candidate=IntentCaptureCandidate(
                        user_id=user_id,
                        scope_key=scope_key,
                        value=distilled.statement,
                        source=(
                            "agent_inference:conversation_distillation:"
                            f"{conversation_id}:{distilled.source_sequence}"
                        )[:200],
                        pref_scope=assignment.pref_scope,
                        timescale=assignment.timescale,
                        layer=assignment.layer,
                        valid_until=assignment.valid_until,
                        confidence=distilled.confidence,
                        ambiguous=True,
                        scope_ambiguous=(
                            distilled.ownership == "ask"
                            or assignment.scope_ambiguous
                        ),
                        semantic_stance=distilled.stance,
                        observed_at=observed_at,
                    ),
                )
                if version is None:
                    continue
                # Bind the exact source message now. Otherwise a later
                # tombstone could suppress the projected reminder yet leave
                # compaction free to distill the original sentence again.
                connection.execute(
                    """
                    INSERT OR IGNORE INTO conversation_message_memory_bindings(
                        user_id, conversation_id, sequence, scope_key, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        conversation_id,
                        distilled.source_sequence,
                        intent_entry_id(scope_key, version.pref_scope),
                        now,
                    ),
                )
            # Covered rows are permanent reconstruction inputs. If a memory
            # tombstone invalidates this summary, the retained unsuppressed
            # messages let the next turn regenerate it without data loss.
        return True

    def commit_turn(
        self,
        *,
        user_id: str,
        conversation_id: str,
        task: ConversationTaskState,
        user_message: ConversationMessageContext,
        assistant_message: ConversationMessageContext,
        assistant_bodies: tuple[DeliveredBodyDraft, ...] = (),
        episode_drafts: tuple[CareerEpisodeDraft, ...] = (),
        memory_scope_keys: tuple[str, ...] = (),
        turn_id: str | None = None,
    ) -> None:
        """Append one turn. The transcript is never pruned here.

        What the model reads is bounded by the context projection; what the file
        keeps is not. Deleting rows to bound the window would trade an
        irreversible loss for nothing, since the read is already limited to the
        same number of messages, and it would take the resource references the
        archived-resource lookup scans for along with it.

        ``assistant_bodies`` land in the same transaction, keyed to the
        assistant row's sequence: a body without its row, or a row whose body
        was lost to a crash between two writes, would both read as a transcript
        that never showed it.

        ``turn_id`` names the runtime turn that wrote the rows, which is also
        the key of its receipt. Removing content from these rows later clears
        that one receipt; without it the whole conversation's receipts go.
        """

        messages = (user_message, assistant_message)
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO conversation_task_state(user_id, conversation_id, payload, updated_at) VALUES (?, ?, ?, ?) ON CONFLICT(user_id, conversation_id) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at",
                (user_id, conversation_id, task.model_dump_json(), now),
            )
            latest_message_sequence = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) FROM conversation_messages WHERE user_id = ? AND conversation_id = ?",
                (user_id, conversation_id),
            ).fetchone()[0]
            summary_sequence_row = connection.execute(
                "SELECT through_sequence FROM conversation_summaries WHERE user_id = ? AND conversation_id = ?",
                (user_id, conversation_id),
            ).fetchone()
            latest_summary_sequence = summary_sequence_row[0] if summary_sequence_row else 0
            next_sequence = max(latest_message_sequence, latest_summary_sequence) + 1
            connection.executemany(
                "INSERT INTO conversation_messages(user_id, conversation_id, sequence, payload, turn_id) VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        user_id,
                        conversation_id,
                        next_sequence + offset,
                        message.model_dump_json(),
                        turn_id,
                    )
                    for offset, message in enumerate(messages)
                ],
            )
            assistant_sequence = next_sequence + len(messages) - 1
            scope_keys = tuple(dict.fromkeys(memory_scope_keys))
            connection.executemany(
                """
                INSERT INTO conversation_message_memory_bindings(
                    user_id, conversation_id, sequence, scope_key, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (
                        user_id,
                        conversation_id,
                        next_sequence + offset,
                        scope_key,
                        now,
                    )
                    for offset in range(len(messages))
                    for scope_key in scope_keys
                ),
            )
            deleted_scope_keys = {
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT scope_key FROM memory_deleted_scopes
                    WHERE user_id = ?
                    """,
                    (user_id,),
                ).fetchall()
            }
            connection.executemany(
                """
                INSERT OR IGNORE INTO memory_deletion_message_suppressions(
                    user_id, conversation_id, sequence, scope_key, suppressed_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    (
                        user_id,
                        conversation_id,
                        next_sequence + offset,
                        scope_key,
                        now,
                    )
                    for offset in range(len(messages))
                    for scope_key in scope_keys
                    if scope_key in deleted_scope_keys
                ),
            )
            suppressed = bool(set(scope_keys) & deleted_scope_keys)
            deleted_dependencies = {
                (row[0], row[1])
                for row in connection.execute(
                    "SELECT kind, resource_id FROM delivered_body_deleted_dependencies WHERE user_id = ?",
                    (user_id,),
                )
            }
            withheld = suppressed
            for draft in assistant_bodies:
                if suppressed or any(
                    (dependency.kind, dependency.resource_id) in deleted_dependencies
                    for dependency in draft.dependencies
                ):
                    withheld = True
                    continue
                connection.execute(
                    """
                    INSERT INTO conversation_delivered_bodies(
                        user_id, body_id, conversation_id, sequence,
                        kind, title, body, created_at, retention, source_json,
                        dependencies_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id, f"body_{uuid4().hex}", conversation_id,
                        assistant_sequence, draft.kind, draft.title, draft.body, now,
                        draft.retention,
                        draft.source.model_dump_json() if draft.source else None,
                        json.dumps([item.model_dump(mode="json") for item in draft.dependencies]),
                    ),
                )
            if withheld:
                if turn_id is None:
                    redact_conversation_receipts_on(connection, user_id, conversation_id)
                else:
                    redact_turn_receipts_on(connection, user_id, conversation_id, (turn_id,))
            for draft in episode_drafts:
                SQLiteCareerEpisodeStore.upsert_on(
                    connection,
                    draft,
                    memory_scope_keys=scope_keys,
                )
        os.chmod(self.path, 0o600)

    def _get_single(self, table: str, user_id: str, model):
        with self._connect() as connection:
            row = connection.execute(f"SELECT payload FROM {table} WHERE user_id = ?", (user_id,)).fetchone()
        return model.model_validate_json(row[0]) if row else None

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30.0)
