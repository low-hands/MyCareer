from __future__ import annotations

from datetime import datetime, timezone
import secrets

from career_agent.agent.session_contracts import AgentSession
from career_agent.storage.context import CareerContextStore


class SessionManager:
    """Owns session identity, ownership, lifecycle, and activity timestamps."""

    def __init__(self, store: CareerContextStore) -> None:
        self._store = store

    def get_or_create(self, *, user_id: str, session_id: str) -> AgentSession:
        if not user_id.strip() or not session_id.strip():
            raise ValueError("user_id and session_id are required")
        existing = self._store.get_session(user_id, session_id)
        if existing is None:
            now = datetime.now(timezone.utc)
            return self._store.upsert_session(
                AgentSession(
                    user_id=user_id,
                    session_id=session_id,
                    created_at=now,
                    last_active_at=now,
                    spotlight_nonce=secrets.token_hex(16),
                )
            )
        if existing.status == "closed":
            raise ValueError("Session is closed")
        return self._store.upsert_session(existing.touch())

    def touch(self, *, user_id: str, session_id: str) -> AgentSession:
        session = self.get_or_create(user_id=user_id, session_id=session_id)
        return session

    def close(self, *, user_id: str, session_id: str) -> AgentSession:
        session = self._store.close_session(user_id, session_id)
        if session is None:
            raise ValueError("Session does not exist")
        return session
