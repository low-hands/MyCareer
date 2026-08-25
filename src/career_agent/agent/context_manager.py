from __future__ import annotations

from datetime import datetime, timezone

from career_agent.agent.conversation_memory_contracts import ConversationSummaryWorker
from career_agent.agent.main_agent_contracts import AgentPreferencesContext, CareerProfileContext, ConversationMessageContext, ConversationTaskState, MainAgentContext
from career_agent.agent.openai_compatible_client import AgentWorkerError
from career_agent.agent.session_manager import SessionManager
from career_agent.storage.context import CareerContextStore


class ContextManager:
    def __init__(self, store: CareerContextStore, *, session_manager: SessionManager | None = None, summary_worker: ConversationSummaryWorker | None = None, recent_message_limit: int = 8, summary_batch_size: int = 4, max_message_chars: int = 4000, max_recent_context_chars: int = 16000) -> None:
        if recent_message_limit < 2 or summary_batch_size < 2:
            raise ValueError("conversation memory limits must be at least two")
        if max_message_chars < 1 or max_recent_context_chars < max_message_chars:
            raise ValueError("conversation character budgets are invalid")
        self._store = store
        self._sessions = session_manager or SessionManager(store)
        self._summary_worker = summary_worker
        self._recent_message_limit = recent_message_limit
        self._summary_batch_size = summary_batch_size
        self._max_message_chars = max_message_chars
        self._max_recent_context_chars = max_recent_context_chars

    def load_for_turn(self, *, user_id: str, conversation_id: str, user_message: str) -> MainAgentContext:
        self._sessions.get_or_create(user_id=user_id, session_id=conversation_id)
        self._maybe_summarize(user_id=user_id, conversation_id=conversation_id)
        profile = self._store.get_profile(user_id) or CareerProfileContext(user_id=user_id)
        preferences = self._store.get_preferences(user_id) or AgentPreferencesContext()
        task = self._store.get_task(user_id, conversation_id) or ConversationTaskState()
        summary = self._store.get_conversation_summary(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        raw_limit = (
            self._recent_message_limit + self._summary_batch_size - 1
            if self._summary_worker is not None
            else self._recent_message_limit
        )
        recent_messages = self._bound_recent_messages(
            self._store.list_messages(
                user_id,
                conversation_id,
                limit=raw_limit,
                after_sequence=summary.through_sequence if summary else 0,
            )
        )
        return MainAgentContext(
            conversation_id=conversation_id,
            profile=profile,
            preferences=preferences,
            task=task,
            recent_messages=recent_messages,
            conversation_summary=summary.content if summary else None,
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
            message_limit=(
                None if self._summary_worker is not None else self._recent_message_limit
            ),
        )
        self._maybe_summarize(
            user_id=context.profile.user_id,
            conversation_id=context.conversation_id,
        )
        self._sessions.touch(user_id=context.profile.user_id, session_id=context.conversation_id)

    def upsert_profile(self, profile: CareerProfileContext) -> None:
        self._store.upsert_profile(profile)

    def upsert_preferences(self, *, user_id: str, preferences: AgentPreferencesContext) -> None:
        self._store.upsert_preferences(user_id, preferences)

    def _truncate(self, content: str) -> str:
        return content[:self._max_message_chars]

    def _maybe_summarize(self, *, user_id: str, conversation_id: str) -> None:
        if self._summary_worker is None:
            return
        previous = self._store.get_conversation_summary(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        previous_through = previous.through_sequence if previous else 0
        threshold = self._recent_message_limit + self._summary_batch_size
        messages = self._store.list_messages_after(
            user_id=user_id,
            conversation_id=conversation_id,
            after_sequence=previous_through,
            limit=threshold,
        )
        if len(messages) < threshold:
            return
        to_summarize = messages[: self._summary_batch_size]
        try:
            content = self._summary_worker.summarize(
                previous=previous.content if previous else None,
                messages=to_summarize,
            )
        except AgentWorkerError:
            return
        self._store.compact_conversation_summary(
            user_id=user_id,
            conversation_id=conversation_id,
            expected_previous_through_sequence=previous_through,
            content=content,
            through_sequence=to_summarize[-1].sequence,
        )

    def _bound_recent_messages(
        self, messages: tuple[ConversationMessageContext, ...]
    ) -> tuple[ConversationMessageContext, ...]:
        selected = []
        used = 0
        for message in reversed(messages):
            remaining = self._max_recent_context_chars - used
            if remaining <= 0:
                break
            content = message.content[:remaining]
            selected.append(message.model_copy(update={"content": content}))
            used += len(content)
        return tuple(reversed(selected))
