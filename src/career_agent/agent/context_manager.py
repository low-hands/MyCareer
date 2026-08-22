from __future__ import annotations

from datetime import datetime, timezone

from career_agent.agent.main_agent_contracts import AgentPreferencesContext, CareerProfileContext, ConversationMessageContext, ConversationTaskState, MainAgentContext
from career_agent.agent.session_manager import SessionManager
from career_agent.storage.context import CareerContextStore


class ContextManager:
    def __init__(self, store: CareerContextStore, *, session_manager: SessionManager | None = None, recent_message_limit: int = 8, max_message_chars: int = 4000) -> None:
        self._store = store
        self._sessions = session_manager or SessionManager(store)
        self._recent_message_limit = recent_message_limit
        self._max_message_chars = max_message_chars

    def load_for_turn(self, *, user_id: str, conversation_id: str, user_message: str) -> MainAgentContext:
        self._sessions.get_or_create(user_id=user_id, session_id=conversation_id)
        profile = self._store.get_profile(user_id) or CareerProfileContext(user_id=user_id)
        preferences = self._store.get_preferences(user_id) or AgentPreferencesContext()
        task = self._store.get_task(user_id, conversation_id) or ConversationTaskState()
        return MainAgentContext(
            conversation_id=conversation_id,
            profile=profile,
            preferences=preferences,
            task=task,
            recent_messages=self._store.list_messages(user_id, conversation_id, limit=self._recent_message_limit),
            user_message=self._truncate(user_message),
        )

    def commit_turn(self, *, context: MainAgentContext, task: ConversationTaskState, assistant_message: str) -> None:
        now = datetime.now(timezone.utc)
        self._store.commit_turn(
            user_id=context.profile.user_id,
            conversation_id=context.conversation_id,
            task=task,
            user_message=ConversationMessageContext(role="user", content=self._truncate(context.user_message), created_at=now),
            assistant_message=ConversationMessageContext(role="assistant", content=self._truncate(assistant_message), created_at=now),
            message_limit=self._recent_message_limit,
        )
        self._sessions.touch(user_id=context.profile.user_id, session_id=context.conversation_id)

    def upsert_profile(self, profile: CareerProfileContext) -> None:
        self._store.upsert_profile(profile)

    def upsert_preferences(self, *, user_id: str, preferences: AgentPreferencesContext) -> None:
        self._store.upsert_preferences(user_id, preferences)

    def _truncate(self, content: str) -> str:
        return content[:self._max_message_chars]
