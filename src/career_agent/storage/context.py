from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3

from career_agent.agent.main_agent_contracts import AgentPreferencesContext, CareerProfileContext, ConversationMessageContext, ConversationTaskState
from career_agent.agent.conversation_memory_contracts import (
    ConversationSummaryContent,
    StoredConversationSummary,
    SummaryMessage,
)
from career_agent.agent.session_contracts import AgentSession


class CareerContextStore:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("CREATE TABLE IF NOT EXISTS sessions (session_id TEXT NOT NULL, user_id TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL, last_active_at TEXT NOT NULL, PRIMARY KEY(user_id, session_id))")
            connection.execute("CREATE INDEX IF NOT EXISTS sessions_user_idx ON sessions(user_id, last_active_at DESC)")
            connection.execute("CREATE TABLE IF NOT EXISTS career_profile_context (user_id TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS agent_preferences_context (user_id TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at TEXT NOT NULL)")
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
        os.chmod(self.path, 0o600)

    def get_session(self, user_id: str, session_id: str) -> AgentSession | None:
        with self._connect() as connection:
            row = connection.execute("SELECT session_id, user_id, status, created_at, last_active_at FROM sessions WHERE session_id = ? AND user_id = ?", (session_id, user_id)).fetchone()
        return AgentSession(session_id=row[0], user_id=row[1], status=row[2], created_at=row[3], last_active_at=row[4]) if row else None

    def get_session_owner(self, session_id: str) -> str | None:
        return None

    def upsert_session(self, session: AgentSession) -> AgentSession:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO sessions(session_id, user_id, status, created_at, last_active_at) VALUES (?, ?, ?, ?, ?) ON CONFLICT(user_id, session_id) DO UPDATE SET status=excluded.status, last_active_at=excluded.last_active_at",
                (session.session_id, session.user_id, session.status, session.created_at.isoformat(), session.last_active_at.isoformat()),
            )
        os.chmod(self.path, 0o600)
        return session

    def close_session(self, user_id: str, session_id: str) -> AgentSession | None:
        session = self.get_session(user_id, session_id)
        if session is None:
            return None
        return self.upsert_session(session.model_copy(update={"status": "closed"}))

    def get_profile(self, user_id: str) -> CareerProfileContext | None:
        return self._get_single("career_profile_context", user_id, CareerProfileContext)

    def upsert_profile(self, profile: CareerProfileContext) -> None:
        self._upsert_single("career_profile_context", profile.user_id, profile.model_dump_json())

    def get_preferences(self, user_id: str) -> AgentPreferencesContext | None:
        return self._get_single("agent_preferences_context", user_id, AgentPreferencesContext)

    def upsert_preferences(self, user_id: str, preferences: AgentPreferencesContext) -> None:
        self._upsert_single("agent_preferences_context", user_id, preferences.model_dump_json())

    def get_task(self, user_id: str, conversation_id: str) -> ConversationTaskState | None:
        with self._connect() as connection:
            row = connection.execute("SELECT payload FROM conversation_task_state WHERE user_id = ? AND conversation_id = ?", (user_id, conversation_id)).fetchone()
        return ConversationTaskState.model_validate_json(row[0]) if row else None

    def list_messages(self, user_id: str, conversation_id: str, *, limit: int, after_sequence: int = 0) -> tuple[ConversationMessageContext, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM conversation_messages WHERE user_id = ? AND conversation_id = ? AND sequence > ? ORDER BY sequence DESC LIMIT ?",
                (user_id, conversation_id, after_sequence, limit),
            ).fetchall()
        return tuple(ConversationMessageContext.model_validate_json(row[0]) for row in reversed(rows))

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
                """
                SELECT sequence, payload
                FROM conversation_messages
                WHERE user_id = ? AND conversation_id = ? AND sequence > ?
                ORDER BY sequence
                LIMIT ?
                """,
                (user_id, conversation_id, after_sequence, limit),
            ).fetchall()
        messages = []
        for row in rows:
            message = ConversationMessageContext.model_validate_json(row[1])
            messages.append(
                SummaryMessage(
                    sequence=row[0],
                    role=message.role,
                    content=message.content,
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
        return True

    def commit_turn(
        self,
        *,
        user_id: str,
        conversation_id: str,
        task: ConversationTaskState,
        user_message: ConversationMessageContext,
        assistant_message: ConversationMessageContext,
        message_limit: int | None,
    ) -> None:
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
            connection.execute(
                "INSERT INTO conversation_messages(user_id, conversation_id, sequence, payload) VALUES (?, ?, ?, ?), (?, ?, ?, ?)",
                (user_id, conversation_id, next_sequence, user_message.model_dump_json(), user_id, conversation_id, next_sequence + 1, assistant_message.model_dump_json()),
            )
            if message_limit is not None:
                connection.execute(
                    "DELETE FROM conversation_messages WHERE user_id = ? AND conversation_id = ? AND sequence NOT IN (SELECT sequence FROM conversation_messages WHERE user_id = ? AND conversation_id = ? ORDER BY sequence DESC LIMIT ?)",
                    (user_id, conversation_id, user_id, conversation_id, message_limit),
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
