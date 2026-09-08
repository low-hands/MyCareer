from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

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
    list_intent_versions,
)
from career_agent.storage.schema import apply_schema

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
    ) -> None: ...

    def list_profile_intent_versions(
        self,
        *,
        user_id: str,
        scope_keys: Sequence[str] | None = None,
        active_only: bool = False,
        limit: int | None = None,
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
                8,
                self._migrate,
                {
                    2: self._upgrade_to_v2,
                    3: self._upgrade_to_v3,
                    4: self._upgrade_to_v4,
                    5: self._upgrade_to_v5,
                    6: self._upgrade_to_v6,
                    7: self._upgrade_to_v7,
                    8: self._upgrade_to_v8,
                },
            )
            apply_episode_schema(connection)
            self._adopt_legacy_preferences(connection)
        os.chmod(self.path, 0o600)

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
            if profile.default_city is not None:
                append_intent_version(
                    connection,
                    user_id=user_id,
                    scope_key="person_intent/self/default_city",
                    value=profile.default_city,
                    source="migration:career_profile_context",
                    valid_from=backfilled_at,
                )
            for constraint in profile.hard_constraints:
                append_intent_version(
                    connection,
                    user_id=user_id,
                    scope_key=f"person_intent/self/{constraint.relation}",
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
        # not replay numbered upgrades. This backfill is safe on every open
        # because append_intent_version no-ops on the active value's digest.
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
        connection.execute("CREATE TABLE IF NOT EXISTS conversation_messages (user_id TEXT NOT NULL, conversation_id TEXT NOT NULL, sequence INTEGER NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(user_id, conversation_id, sequence))")
        connection.execute("CREATE INDEX IF NOT EXISTS conversation_messages_recent_idx ON conversation_messages(user_id, conversation_id, sequence DESC)")
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
            for table in (
                "conversation_summaries",
                "conversation_messages",
                "conversation_task_state",
                "conversation_message_memory_bindings",
                "memory_deletion_message_suppressions",
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
    ) -> None:
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
                )
            for constraint in profile.hard_constraints:
                append_intent_version(
                    connection,
                    user_id=profile.user_id,
                    scope_key=f"person_intent/self/{constraint.relation}",
                    value=constraint.value,
                    source=source,
                    valid_from=now,
                )

    def list_profile_intent_versions(
        self,
        *,
        user_id: str,
        scope_key: str | None = None,
        scope_keys: Sequence[str] | None = None,
        active_only: bool = False,
        limit: int | None = None,
    ) -> tuple[IntentMemoryVersion, ...]:
        with self._connect() as connection:
            return list_intent_versions(
                connection,
                user_id=user_id,
                scope_key=scope_key,
                scope_keys=scope_keys,
                active_only=active_only,
                limit=limit,
            )

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
    def get_preferences(self, user_id: str) -> OwnerSettingsContext | None:
        return self.get_owner_settings(user_id)

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
    ) -> dict[str, int]:
        """Suppress only derived rows that observed one tombstoned lineage.

        New rows carry an explicit scope binding. ``lineage_markers`` is a
        SQL-side migration fallback for messages written before those
        bindings existed; only opaque detail/source/lineage refs are accepted.
        Free-text claims are never used as scan keys.
        """

        now = datetime.now(timezone.utc).isoformat()
        markers = _usable_lineage_markers(lineage_markers)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO memory_deleted_scopes(user_id, scope_key, deleted_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id, scope_key) DO NOTHING
                """,
                (user_id, scope_key, now),
            )
            bound_rows = {
                (str(row[0]), int(row[1]))
                for row in connection.execute(
                    """
                    SELECT conversation_id, sequence
                    FROM conversation_message_memory_bindings
                    WHERE user_id = ? AND scope_key = ?
                    """,
                    (user_id, scope_key),
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
                              AND hidden.sequence <= conversation_summaries.through_sequence
                          )
                    """,
                    (user_id, *affected_conversations),
                )
                summary_count = int(cursor.rowcount)
            episode_count = SQLiteCareerEpisodeStore.delete_for_scope_on(
                connection,
                user_id=user_id,
                scope_key=scope_key,
                lineage_markers=markers,
            )
        return {
            "conversation_fragments": len(bound_rows),
            "conversation_summaries": summary_count,
            "career_episodes": episode_count,
            "affected_conversations": len(affected_conversations),
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

    def compact_conversation_summary(
        self,
        *,
        user_id: str,
        conversation_id: str,
        expected_previous_through_sequence: int,
        content: ConversationSummaryContent,
        through_sequence: int,
    ) -> bool:
        if through_sequence <= expected_previous_through_sequence:
            raise ValueError("conversation summary must advance its covered sequence")
        now = datetime.now(timezone.utc).isoformat()
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
            # Covered rows are kept. Read paths already skip them, so they cost
            # file size and nothing else, and a summary is a model output: if it
            # drops or distorts something, the original is the only way to tell.
            # Reclaiming the space is a separate, explicit request.
        return True

    def count_compacted_messages(
        self, *, user_id: str, conversation_id: str | None = None
    ) -> tuple[int, int]:
        """How many covered rows are still stored, and how many bytes they hold.

        Covered means a summary already claims the row and the prune would
        delete it. The two have to apply the same filter: counting rows the
        command keeps would leave the notice standing after every prune, telling
        the operator there is disk to reclaim and then reclaiming none of it.

        Delivering turns are what the filter excludes. Their rows are kept
        deliberately — they are the only handle the agent has on a report once
        the recent window scrolls past — so they are not reclaimable and must
        not be counted as such.
        """
        clause = "AND m.conversation_id = ?" if conversation_id else ""
        parameters: tuple[str, ...] = (
            (user_id, conversation_id) if conversation_id else (user_id,)
        )
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT COUNT(*), COALESCE(SUM(LENGTH(m.payload)), 0)
                FROM conversation_messages AS m
                JOIN conversation_summaries AS s
                  ON s.user_id = m.user_id
                 AND s.conversation_id = m.conversation_id
                WHERE m.user_id = ? {clause} AND m.sequence <= s.through_sequence
                  AND json_array_length(
                        COALESCE(json_extract(m.payload, '$.resource_refs'), json_array())
                      ) = 0
                """,
                parameters,
            ).fetchone()
        return int(row[0]), int(row[1])

    def prune_compacted_messages(
        self, *, user_id: str, conversation_id: str | None = None
    ) -> int:
        """Delete only rows a stored summary already covers.

        The bound is read inside the same transaction as the delete. Passing a
        sequence in from the caller would let a turn committed in between be
        deleted while no summary had claimed it yet.
        """
        clause = "AND m.conversation_id = ?" if conversation_id else ""
        parameters: tuple[str, ...] = (
            (user_id, conversation_id) if conversation_id else (user_id,)
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                f"""
                DELETE FROM conversation_messages
                WHERE rowid IN (
                    SELECT m.rowid
                    FROM conversation_messages AS m
                    JOIN conversation_summaries AS s
                      ON s.user_id = m.user_id
                     AND s.conversation_id = m.conversation_id
                    WHERE m.user_id = ? {clause}
                      AND m.sequence <= s.through_sequence
                      -- A delivered report's row is the only handle the agent
                      -- has on it once the recent window scrolls past. Its
                      -- content is a bounded line by construction, so keeping
                      -- it costs almost nothing, while dropping it would make
                      -- the report unreachable to the agent while the UI still
                      -- shows the card.
                      AND json_array_length(
                            COALESCE(
                                json_extract(m.payload, '$.resource_refs'),
                                json_array()
                            )
                          ) = 0
                )
                """,
                parameters,
            )
            deleted = cursor.rowcount
        os.chmod(self.path, 0o600)
        return int(deleted)

    def commit_turn(
        self,
        *,
        user_id: str,
        conversation_id: str,
        task: ConversationTaskState,
        user_message: ConversationMessageContext,
        assistant_message: ConversationMessageContext,
        episode_drafts: tuple[CareerEpisodeDraft, ...] = (),
        memory_scope_keys: tuple[str, ...] = (),
    ) -> None:
        """Append one turn. The transcript is never pruned here.

        What the model reads is bounded by the context projection; what the file
        keeps is not. Deleting rows to bound the window would trade an
        irreversible loss for nothing, since the read is already limited to the
        same number of messages, and it would take the resource references the
        archived-resource lookup scans for along with it.
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
                "INSERT INTO conversation_messages(user_id, conversation_id, sequence, payload) VALUES (?, ?, ?, ?)",
                [
                    (
                        user_id,
                        conversation_id,
                        next_sequence + offset,
                        message.model_dump_json(),
                    )
                    for offset, message in enumerate(messages)
                ],
            )
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

    def _upsert_single(self, table: str, user_id: str, payload: str) -> None:
        with self._connect() as connection:
            connection.execute(
                f"INSERT INTO {table}(user_id, payload, updated_at) VALUES (?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at",
                (user_id, payload, datetime.now(timezone.utc).isoformat()),
            )
        os.chmod(self.path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30.0)
