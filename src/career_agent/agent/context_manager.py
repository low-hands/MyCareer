from __future__ import annotations

from datetime import datetime, timezone

from career_agent.agent.conversation_memory_contracts import ConversationSummaryWorker
from career_agent.agent.main_agent_contracts import AgentPreferencesContext, CareerProfileContext, ConversationMessageContext, ConversationResourceReference, ConversationTaskState, MainAgentContext
from career_agent.agent.openai_compatible_client import AgentWorkerError
from career_agent.agent.session_manager import SessionManager
from career_agent.storage.context import CareerContextStore


class ContextManager:
    def __init__(self, store: CareerContextStore, *, session_manager: SessionManager | None = None, summary_worker: ConversationSummaryWorker | None = None, recent_message_limit: int = 8, summary_batch_size: int = 4, max_message_chars: int = 32000, max_recent_context_chars: int = 32000, max_recent_message_chars: int | None = None, compacted_message_warning_threshold: int = 200, archived_resource_limit: int = 12) -> None:
        if recent_message_limit < 2 or summary_batch_size < 2:
            raise ValueError("conversation memory limits must be at least two")
        if max_message_chars < 1 or max_recent_context_chars < 2:
            raise ValueError("conversation character budgets are invalid")
        per_message_context_chars = (
            max_recent_message_chars
            if max_recent_message_chars is not None
            else max_recent_context_chars // 2
        )
        if not 1 <= per_message_context_chars <= max_recent_context_chars:
            raise ValueError("recent per-message budget is invalid")
        if not 0 <= archived_resource_limit <= 12:
            # Capped at the contract's own bound: each entry costs a kind, a
            # timestamp, and one condensed line, and the catalogue is carried
            # every turn for the whole life of the conversation.
            raise ValueError("archived resource limit is invalid")
        if compacted_message_warning_threshold < 1:
            raise ValueError("compacted message warning threshold must be positive")
        self._store = store
        self._sessions = session_manager or SessionManager(store)
        self._summary_worker = summary_worker
        self._recent_message_limit = recent_message_limit
        self._summary_batch_size = summary_batch_size
        self._max_message_chars = max_message_chars
        self._max_recent_context_chars = max_recent_context_chars
        self._max_recent_message_chars = per_message_context_chars
        self._compacted_message_warning_threshold = compacted_message_warning_threshold
        self._archived_resource_limit = archived_resource_limit

    def compacted_message_notice(
        self, *, user_id: str, conversation_id: str | None = None
    ) -> str | None:
        """Tell the operator when summarised originals are worth reclaiming.

        Deliberately not part of the agent's context. The whole point of keeping
        these rows is that a human decides when they stop being worth their disk,
        and a model that could read this notice could also decide to act on it.
        """
        count, byte_size = self._store.count_compacted_messages(
            user_id=user_id, conversation_id=conversation_id
        )
        if count < self._compacted_message_warning_threshold:
            return None
        return (
            f"已摘要的原始消息累计 {count} 条（约 {byte_size // 1024} KiB）仍然保留。"
            "运行 career-agent context prune 会同时从可见历史对话中永久删除这些消息；"
            "只有明确不再需要回看时才应执行。"
        )

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
            # Read from the messages the window has scrolled past, so a report
            # delivered weeks ago stays nameable. Skipped entirely before the
            # first summary exists, when nothing has scrolled past yet and the
            # window already holds every reference.
            archived_resources=(
                self._store.list_archived_resource_messages(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    through_sequence=summary.through_sequence,
                    limit=self._archived_resource_limit,
                )
                if summary is not None and self._archived_resource_limit
                else ()
            ),
            conversation_summary=summary.content if summary else None,
            user_message=self._truncate(user_message),
        )

    def get_task(
        self, *, user_id: str, conversation_id: str
    ) -> ConversationTaskState:
        """Read only the routing state before choosing a context owner."""
        return self._store.get_task(user_id, conversation_id) or ConversationTaskState()

    def load_for_workflow_turn(
        self,
        *,
        user_id: str,
        conversation_id: str,
        task: ConversationTaskState,
    ) -> MainAgentContext:
        """Build a routing envelope without loading Main Agent memory."""
        self._sessions.get_or_create(user_id=user_id, session_id=conversation_id)
        profile = self._store.get_profile(user_id) or CareerProfileContext(user_id=user_id)
        preferences = self._store.get_preferences(user_id) or AgentPreferencesContext()
        return MainAgentContext(
            conversation_id=conversation_id,
            profile=profile,
            preferences=preferences,
            task=task,
            recent_messages=(),
            conversation_summary=None,
            user_message="[workflow-owned input withheld]",
        )

    def commit_turn(self, *, context: MainAgentContext, task: ConversationTaskState, assistant_message: str, assistant_resource_ref: ConversationResourceReference | None = None) -> None:
        now = datetime.now(timezone.utc)
        self._store.commit_turn(
            user_id=context.profile.user_id,
            conversation_id=context.conversation_id,
            task=task,
            user_message=ConversationMessageContext(role="user", content=self._truncate(context.user_message), created_at=now),
            assistant_message=ConversationMessageContext(role="assistant", content=self._truncate(assistant_message), created_at=now, resource_ref=assistant_resource_ref),
            message_limit=(
                None if self._summary_worker is not None else self._recent_message_limit
            ),
        )
        self._maybe_summarize(
            user_id=context.profile.user_id,
            conversation_id=context.conversation_id,
        )
        self._sessions.touch(user_id=context.profile.user_id, session_id=context.conversation_id)

    def commit_workflow_entry(
        self, *, context: MainAgentContext, task: ConversationTaskState
    ) -> ConversationTaskState:
        """Hold the request that started a workflow instead of writing it.

        The reply this turn produced is the run's opening move, withheld on the
        same grounds as every later turn, so writing the request now would leave
        the conversation showing a question nobody answered for as long as the
        run lasts. Holding it lets ``commit_workflow_exit`` write the request and
        its reply as one turn, so the stored conversation is never mid-exchange.

        Returns the task state that was persisted, which carries the held
        request and must be the one the caller reports.
        """
        held = task.hold_entry_message(self._truncate(context.user_message))
        self.commit_workflow_turn(context=context, task=held)
        return held

    def commit_workflow_exit(
        self,
        *,
        context: MainAgentContext,
        task: ConversationTaskState,
        assistant_message: str,
        assistant_resource_ref: ConversationResourceReference | None = None,
    ) -> None:
        """Write the whole run as the request that began it and the reply.

        The run's own turns never reach the conversation, so without this it
        would jump from that request to whatever the user says next and Main
        Agent would have no way to know a run had happened. The request comes
        from the held copy rather than from this turn's input, which was the
        workflow's; both land in one write, so the conversation goes from not
        mentioning the run to describing it whole.
        """
        # Read the held request from the task as it stood before this turn:
        # releasing the slot clears the field, and that release is exactly what
        # brought us here.
        entry = context.task.workflow_entry_message
        if entry is None:
            # Nothing claimed a request, so there is no exchange to close. This
            # is reachable when a run is adopted mid-flight rather than started
            # here, and dropping the reply beats inventing a request for it.
            self.commit_workflow_turn(context=context, task=task)
            return
        # Clearing here rather than relying on the caller's transition: an exit
        # that keeps the slot to record why the run died would otherwise leave
        # this request behind, and the next run would answer it instead of its
        # own. Written and cleared in one call, so the two cannot drift.
        self.commit_turn(
            context=context.model_copy(update={"user_message": entry}),
            task=task.model_copy(update={"workflow_entry_message": None}),
            assistant_message=assistant_message,
            assistant_resource_ref=assistant_resource_ref,
        )

    def commit_workflow_turn(
        self, *, context: MainAgentContext, task: ConversationTaskState
    ) -> None:
        """Persist workflow ownership without copying child-agent transcripts."""
        self._store.upsert_task(
            user_id=context.profile.user_id,
            conversation_id=context.conversation_id,
            task=task,
        )
        self._sessions.touch(
            user_id=context.profile.user_id,
            session_id=context.conversation_id,
        )

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
            content = message.content[
                : min(remaining, self._max_recent_message_chars)
            ]
            selected.append(message.model_copy(update={"content": content}))
            used += len(content)
        return tuple(reversed(selected))
