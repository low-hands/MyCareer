from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from career_agent.agent.conversation_memory_contracts import ConversationSummaryWorker
from career_agent.agent.main_agent_contracts import OwnerSettingsContext, CareerProfileContext, ConversationMessageContext, ConversationResourceReference, ConversationTaskState, MainAgentContext
from career_agent.agent.openai_compatible_client import AgentWorkerError
from career_agent.agent.session_manager import SessionManager
from career_agent.harness.observability import (
    conversation_trace_key,
    record_active_trace,
)
from career_agent.storage.context import CareerContextStore, StoredConversationMessage


class ContextManager:
    def __init__(self, store: CareerContextStore, *, session_manager: SessionManager | None = None, summary_worker: ConversationSummaryWorker | None = None, recent_message_limit: int = 8, summary_batch_size: int = 4, max_message_chars: int = 32000, max_recent_context_chars: int = 32000, max_recent_message_chars: int | None = None, compact_occupancy_threshold: float = 0.75, compacted_message_warning_threshold: int = 200, archived_resource_limit: int = 12) -> None:
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
            # Capped at the contract's own bound, which counts *messages*, not
            # catalogue entries: a message can carry several references since a
            # turn can store two reports, so twelve messages project to at most
            # twelve times that many entries. Each entry costs a kind, a
            # timestamp and one condensed line, and the catalogue is carried
            # every turn for the whole life of the conversation — so the real
            # ceiling is looser than a per-entry reading of this number
            # suggests. Left as a message bound because that is what the store
            # can filter on; tighten it to entries if a conversation is ever
            # observed storing enough multi-report turns for it to matter.
            raise ValueError("archived resource limit is invalid")
        if compacted_message_warning_threshold < 1:
            raise ValueError("compacted message warning threshold must be positive")
        if not 0.7 <= compact_occupancy_threshold <= 0.9:
            raise ValueError("compact occupancy threshold must be between 0.7 and 0.9")
        self._store = store
        self._sessions = session_manager or SessionManager(store)
        self._summary_worker = summary_worker
        self._recent_message_limit = recent_message_limit
        self._summary_batch_size = summary_batch_size
        self._max_message_chars = max_message_chars
        self._max_recent_context_chars = max_recent_context_chars
        self._max_recent_message_chars = per_message_context_chars
        self._compact_occupancy_threshold = compact_occupancy_threshold
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
        self._maybe_summarize(
            user_id=user_id,
            conversation_id=conversation_id,
            trigger="occupancy",
        )
        profile = self._store.get_profile(user_id) or CareerProfileContext(user_id=user_id)
        preferences = self._store.get_owner_settings(user_id) or OwnerSettingsContext()
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
        recent_records = self._bound_recent_messages(
            self._store.list_message_records(
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
            recent_messages=tuple(record.message for record in recent_records),
            through_sequence=summary.through_sequence if summary else 0,
            recent_from_sequence=(
                recent_records[0].sequence if recent_records else None
            ),
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
            # Sent alongside the capped list so the model can tell a complete
            # catalogue from a window onto a longer one. Without it twelve
            # entries read as everything there is, and a report past the cap
            # looks like it must be one of them.
            archived_resource_total=(
                self._store.count_archived_resources(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    through_sequence=summary.through_sequence,
                )
                if summary is not None and self._archived_resource_limit
                else 0
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
        preferences = self._store.get_owner_settings(user_id) or OwnerSettingsContext()
        return MainAgentContext(
            conversation_id=conversation_id,
            profile=profile,
            preferences=preferences,
            task=task,
            recent_messages=(),
            conversation_summary=None,
            user_message="[workflow-owned input withheld]",
        )

    def commit_turn(self, *, context: MainAgentContext, task: ConversationTaskState, assistant_message: str, assistant_resource_refs: tuple[ConversationResourceReference, ...] = (), compaction_trigger: Literal["occupancy", "seam"] = "occupancy") -> None:
        now = datetime.now(timezone.utc)
        self._store.commit_turn(
            user_id=context.profile.user_id,
            conversation_id=context.conversation_id,
            task=task,
            user_message=ConversationMessageContext(role="user", content=self._truncate(context.user_message), created_at=now),
            assistant_message=ConversationMessageContext(role="assistant", content=self._truncate(assistant_message), created_at=now, resource_refs=assistant_resource_refs),
        )
        self._maybe_summarize(
            user_id=context.profile.user_id,
            conversation_id=context.conversation_id,
            trigger=compaction_trigger,
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
        assistant_resource_refs: tuple[ConversationResourceReference, ...] = (),
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
            assistant_resource_refs=assistant_resource_refs,
            compaction_trigger="seam",
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

    def preferences(self, *, user_id: str) -> OwnerSettingsContext:
        """The owner's rules, defaulted rather than absent.

        A user who has never set one is not a user with no rules — the defaults
        are the rules, and returning ``None`` would push that decision onto
        every caller.
        """

        return self._store.get_owner_settings(user_id) or OwnerSettingsContext()

    def upsert_preferences(self, *, user_id: str, preferences: OwnerSettingsContext) -> None:
        self._store.upsert_preferences(user_id, preferences)

    def update_owner_settings(
        self,
        *,
        user_id: str,
        desired: OwnerSettingsContext,
        expected_revision: int,
        actor_type: str,
        actor_id: str,
    ) -> OwnerSettingsContext:
        return self._store.update_owner_settings(
            user_id=user_id,
            desired=desired,
            expected_revision=expected_revision,
            actor_type=actor_type,
            actor_id=actor_id,
        )

    def _truncate(self, content: str) -> str:
        return content[:self._max_message_chars]

    def _maybe_summarize(
        self,
        *,
        user_id: str,
        conversation_id: str,
        trigger: Literal["occupancy", "seam"],
    ) -> None:
        if self._summary_worker is None:
            return
        compacted = self._compact_one_batch(
            user_id=user_id,
            conversation_id=conversation_id,
            trigger=trigger,
            require_occupancy=trigger == "occupancy",
        )
        if trigger != "seam" or not compacted:
            return
        while self._compact_one_batch(
            user_id=user_id,
            conversation_id=conversation_id,
            trigger="seam",
            require_occupancy=True,
        ):
            pass

    def _compact_one_batch(
        self,
        *,
        user_id: str,
        conversation_id: str,
        trigger: Literal["occupancy", "seam"],
        require_occupancy: bool,
    ) -> bool:
        previous = self._store.get_conversation_summary(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        previous_through = previous.through_sequence if previous else 0
        occupancy, projection_overflow = self._recent_pressure(
            user_id=user_id,
            conversation_id=conversation_id,
            after_sequence=previous_through,
        )
        if (
            require_occupancy
            and not projection_overflow
            and occupancy < self._compact_occupancy_threshold
        ):
            return False
        messages = self._store.list_messages_after(
            user_id=user_id,
            conversation_id=conversation_id,
            after_sequence=previous_through,
            limit=self._summary_batch_size,
        )
        if len(messages) < self._summary_batch_size:
            return False
        to_summarize = messages[: self._summary_batch_size]
        try:
            content = self._summary_worker.summarize(
                previous=previous.content if previous else None,
                messages=to_summarize,
            )
        except AgentWorkerError:
            return False
        compacted = self._store.compact_conversation_summary(
            user_id=user_id,
            conversation_id=conversation_id,
            expected_previous_through_sequence=previous_through,
            content=content,
            through_sequence=to_summarize[-1].sequence,
        )
        if not compacted:
            return False
        record_active_trace(
            "context_compacted",
            "conversation_summary",
            outcome="succeeded",
            details={
                "conversation_id": conversation_id,
                "conversation_key": conversation_trace_key(
                    user_id, conversation_id
                ),
                "trigger": (
                    "projection_overflow"
                    if trigger == "occupancy" and projection_overflow
                    else trigger
                ),
                "through_sequence": to_summarize[-1].sequence,
                "occupancy": occupancy,
                "projection_overflow": projection_overflow,
                "batch_size": len(to_summarize),
            },
        )
        return True

    def _recent_pressure(
        self,
        *,
        user_id: str,
        conversation_id: str,
        after_sequence: int,
    ) -> tuple[float, bool]:
        raw_limit = self._recent_message_limit + self._summary_batch_size - 1
        candidates = self._store.list_message_records(
            user_id,
            conversation_id,
            # One extra row proves that the oldest unsummarised message would
            # disappear from the projection before a watermark can name it.
            limit=raw_limit + 1,
            after_sequence=after_sequence,
        )
        projection_overflow = len(candidates) > raw_limit
        recent = self._bound_recent_messages(
            candidates[-raw_limit:]
        )
        used = sum(len(record.message.content) for record in recent)
        return used / self._max_recent_context_chars, projection_overflow

    def _bound_recent_messages(
        self, messages: tuple[StoredConversationMessage, ...]
    ) -> tuple[StoredConversationMessage, ...]:
        selected = []
        used = 0
        for record in reversed(messages):
            remaining = self._max_recent_context_chars - used
            if remaining <= 0:
                break
            content = record.message.content[
                : min(remaining, self._max_recent_message_chars)
            ]
            selected.append(
                record.model_copy(
                    update={
                        "message": record.message.model_copy(
                            update={"content": content}
                        )
                    }
                )
            )
            used += len(content)
        return tuple(reversed(selected))
