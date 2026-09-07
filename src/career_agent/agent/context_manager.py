from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Literal, Protocol

from career_agent.agent.conversation_memory_contracts import (
    SUMMARY_TEXT_MAX_CHARS as _SUMMARY_TEXT_BUDGET,
    ConversationSummaryContent,
    ConversationSummaryWorker,
)
from career_agent.agent.main_agent_contracts import (
    CareerProfileContext,
    ConversationMessageContext,
    ConversationResourceReference,
    ConversationTaskState,
    MainAgentContext,
    CurrentTargetContext,
    OwnerSettingsContext,
)
from career_agent.agent.openai_compatible_client import AgentWorkerError
from career_agent.agent.session_manager import SessionManager
from career_agent.domain.episodes import CareerEpisodeDraft
from career_agent.domain.resume import TargetRole
from career_agent.harness.observability import (
    conversation_trace_key,
    record_active_trace,
)
from career_agent.services.episode_consolidation import mock_interview_exit_draft
from career_agent.storage.context import CareerContextStore, StoredConversationMessage


class TargetRoleSource(Protocol):
    def list_target_roles(self, *, user_id: str) -> tuple[TargetRole, ...]: ...


class ContextManager:
    _MAX_STATIC_INPUT_FRACTION = 0.5
    _RECENT_DEDUP_MIN_CHARS = 512
    _RECENT_DUPLICATE_MARKER = (
        "[duplicate content omitted; identical to a newer visible message "
        "in this recent window]"
    )

    def __init__(self, store: CareerContextStore, *, session_manager: SessionManager | None = None, summary_worker: ConversationSummaryWorker | None = None, recent_message_limit: int = 8, summary_batch_size: int = 4, max_message_chars: int = 32000, max_recent_context_chars: int = 32000, max_recent_message_chars: int | None = None, compact_occupancy_threshold: float = 0.75, compacted_message_warning_threshold: int = 200, archived_resource_limit: int = 12, target_role_source: TargetRoleSource | None = None) -> None:
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
        self._target_role_source = target_role_source
        self._request_token_estimator: (
            Callable[[MainAgentContext], tuple[int, int]] | None
        ) = None

    def configure_request_token_estimator(
        self,
        estimator: Callable[[MainAgentContext], tuple[int, int]],
        *,
        static_input_tokens: int | None = None,
        max_input_tokens: int | None = None,
    ) -> None:
        """Measure compaction against the complete model request.

        Runtime wiring owns the system prompt and installed tool universe, so
        it supplies this callback after constructing both.  Keeping that
        dependency out of the storage constructor also leaves lightweight
        context-only callers usable in tests and maintenance commands.
        """
        if (static_input_tokens is None) != (max_input_tokens is None):
            raise ValueError(
                "static input tokens and maximum input tokens must be provided together"
            )
        if static_input_tokens is not None and max_input_tokens is not None:
            if static_input_tokens < 0 or max_input_tokens < 1:
                raise ValueError("static request token estimate is invalid")
            if (
                static_input_tokens / max_input_tokens
                > self._MAX_STATIC_INPUT_FRACTION
            ):
                raise ValueError(
                    "static model request consumes more than 50% of "
                    "max_input_tokens; increase the input budget or reduce "
                    "the installed tool universe"
                )
        self._request_token_estimator = estimator

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

    def load_for_turn(
        self, *, user_id: str, conversation_id: str, user_message: str
    ) -> MainAgentContext:
        self._sessions.get_or_create(user_id=user_id, session_id=conversation_id)
        self._maybe_summarize(
            user_id=user_id,
            conversation_id=conversation_id,
            trigger="occupancy",
            user_message=user_message,
        )
        return self._build_context(
            user_id=user_id,
            conversation_id=conversation_id,
            user_message=user_message,
        )

    def _build_context(
        self, *, user_id: str, conversation_id: str, user_message: str
    ) -> MainAgentContext:
        session = self._store.get_session(user_id, conversation_id)
        profile = self._profile_context(user_id)
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
            self._deduplicate_recent_messages(
                self._store.list_message_records(
                    user_id,
                    conversation_id,
                    limit=raw_limit,
                    after_sequence=summary.through_sequence if summary else 0,
                )
            )
        )
        return MainAgentContext(
            conversation_id=conversation_id,
            spotlight_nonce=session.spotlight_nonce if session else None,
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
        profile = self._stored_profile_context(user_id)
        preferences = self._store.get_owner_settings(user_id) or OwnerSettingsContext()
        return MainAgentContext(
            conversation_id=conversation_id,
            spotlight_nonce=self._sessions.get_or_create(
                user_id=user_id, session_id=conversation_id
            ).spotlight_nonce,
            profile=profile,
            preferences=preferences,
            task=task,
            recent_messages=(),
            conversation_summary=None,
            user_message="[workflow-owned input withheld]",
        )

    def _stored_profile_context(self, user_id: str) -> CareerProfileContext:
        return self._store.get_profile(user_id) or CareerProfileContext(
            user_id=user_id
        )

    def _profile_context(self, user_id: str) -> CareerProfileContext:
        profile = self._stored_profile_context(user_id)
        if self._target_role_source is None:
            return profile
        current_targets = tuple(
            CurrentTargetContext(
                title=role.title,
                priority=role.priority,
                status=role.status,
                city=role.city,
                salary_expectation=role.salary_expectation,
                experience=role.experience,
                education=role.education,
            )
            for role in self._target_role_source.list_target_roles(user_id=user_id)
        )
        return profile.model_copy(update={"current_targets": current_targets})

    def commit_turn(self, *, context: MainAgentContext, task: ConversationTaskState, assistant_message: str, assistant_resource_refs: tuple[ConversationResourceReference, ...] = (), compaction_trigger: Literal["occupancy", "seam"] = "occupancy", episode_drafts: tuple[CareerEpisodeDraft, ...] = ()) -> None:
        now = datetime.now(timezone.utc)
        self._store.commit_turn(
            user_id=context.profile.user_id,
            conversation_id=context.conversation_id,
            task=task,
            user_message=ConversationMessageContext(role="user", content=self._truncate(context.user_message), created_at=now),
            assistant_message=ConversationMessageContext(role="assistant", content=self._truncate(assistant_message), created_at=now, resource_refs=assistant_resource_refs),
            episode_drafts=episode_drafts,
        )
        self._maybe_summarize(
            user_id=context.profile.user_id,
            conversation_id=context.conversation_id,
            trigger=compaction_trigger,
            user_message=context.user_message,
        )
        self._sessions.touch(user_id=context.profile.user_id, session_id=context.conversation_id)

    def commit_workflow_entry(
        self,
        *,
        context: MainAgentContext,
        task: ConversationTaskState,
        episode_drafts: tuple[CareerEpisodeDraft, ...] = (),
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
        self.commit_workflow_turn(
            context=context,
            task=held,
            episode_drafts=episode_drafts,
        )
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
        episode_drafts: tuple[CareerEpisodeDraft, ...] = ()
        if (
            context.task.active_workflow == "mock_interview"
            and context.task.run_id is not None
        ):
            episode_drafts = (
                mock_interview_exit_draft(
                    user_id=context.profile.user_id,
                    conversation_id=context.conversation_id,
                    source_run_id=context.task.run_id,
                    assistant_message=assistant_message,
                    resource_refs=assistant_resource_refs,
                ),
            )
        if entry is None:
            # Nothing claimed a request, so there is no exchange to close. This
            # is reachable when a run is adopted mid-flight rather than started
            # here, and dropping the reply beats inventing a request for it.
            self.commit_workflow_turn(
                context=context,
                task=task,
                episode_drafts=episode_drafts,
            )
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
            episode_drafts=episode_drafts,
        )

    def commit_workflow_turn(
        self,
        *,
        context: MainAgentContext,
        task: ConversationTaskState,
        episode_drafts: tuple[CareerEpisodeDraft, ...] = (),
    ) -> None:
        """Persist workflow ownership without copying child-agent transcripts."""
        self._store.upsert_task(
            user_id=context.profile.user_id,
            conversation_id=context.conversation_id,
            task=task,
            episode_drafts=episode_drafts,
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
        user_message: str,
    ) -> None:
        if self._summary_worker is None:
            return
        self._compact_one_batch(
            user_id=user_id,
            conversation_id=conversation_id,
            trigger=trigger,
            require_occupancy=trigger == "occupancy",
            user_message=user_message,
        )

    def _compact_one_batch(
        self,
        *,
        user_id: str,
        conversation_id: str,
        trigger: Literal["occupancy", "seam"],
        require_occupancy: bool,
        user_message: str,
    ) -> bool:
        previous = self._store.get_conversation_summary(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        previous_through = previous.through_sequence if previous else 0
        (
            occupancy,
            projection_overflow,
            request_tokens,
            max_input_tokens,
        ) = self._recent_pressure(
            user_id=user_id,
            conversation_id=conversation_id,
            after_sequence=previous_through,
            user_message=user_message,
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
        restored_constraints = 0
        dropped_constraints = 0
        omitted_active_constraint_count = (
            previous.content.omitted_active_constraint_count
            if previous is not None
            else 0
        )
        if previous is not None:
            prior_constraints = previous.content.active_constraints
            candidate_constraints = tuple(
                dict.fromkeys((*prior_constraints, *content.active_constraints))
            )
            merged_constraints = candidate_constraints[:15]
            # The contract budgets every summary field together, so fifteen
            # constraints can exceed the whole allowance on their own and leave
            # nothing for the fields below. Oldest first, so a constraint that
            # already survived a rewrite is never traded for a newer one.
            constraint_chars = 0
            bounded: list[str] = []
            for constraint in merged_constraints:
                if constraint_chars + len(constraint) > _SUMMARY_TEXT_BUDGET:
                    break
                bounded.append(constraint)
                constraint_chars += len(constraint)
            merged_constraints = tuple(bounded)
            dropped_constraints = len(candidate_constraints) - len(
                merged_constraints
            )
            restored_constraints = sum(
                constraint not in content.active_constraints
                for constraint in merged_constraints
            )
            if merged_constraints != content.active_constraints:
                # Constraint deletion is not delegated to a lossy rewrite.
                # Explicit state transitions should retire constraints in a
                # future typed operation; until then, old constraints win.
                fields = {
                    "user_goals": list(content.user_goals),
                    "confirmed_decisions": list(content.confirmed_decisions),
                    "unresolved_questions": list(content.unresolved_questions),
                }
                available = _SUMMARY_TEXT_BUDGET - constraint_chars
                while (
                    any(fields.values())
                    and sum(
                        len(item)
                        for values in fields.values()
                        for item in values
                    )
                    > available
                ):
                    field = max(
                        (name for name, values in fields.items() if values),
                        key=lambda name: len(fields[name][-1]),
                    )
                    fields[field].pop()
                content = ConversationSummaryContent(
                    **fields,
                    active_constraints=merged_constraints,
                )
        omitted_active_constraint_count += dropped_constraints
        content = content.model_copy(
            update={
                "omitted_active_constraint_count": (
                    omitted_active_constraint_count
                )
            }
        )
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
                "input_occupancy_numerator": request_tokens,
                "input_occupancy_denominator": max_input_tokens,
                "restored_constraints": restored_constraints,
                "dropped_constraints": dropped_constraints,
                "omitted_active_constraint_count": (
                    omitted_active_constraint_count
                ),
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
        user_message: str,
    ) -> tuple[float, bool, int | None, int | None]:
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
            self._deduplicate_recent_messages(candidates[-raw_limit:])
        )
        if self._request_token_estimator is not None:
            pressure_context = self._build_context(
                user_id=user_id,
                conversation_id=conversation_id,
                user_message=user_message,
            )
            request_tokens, max_input_tokens = self._request_token_estimator(
                pressure_context
            )
            if request_tokens < 0 or max_input_tokens < 1:
                raise ValueError("request token estimator returned an invalid budget")
            return (
                request_tokens / max_input_tokens,
                projection_overflow,
                request_tokens,
                max_input_tokens,
            )
        # Maintenance callers may not own a model or a tool registry. Preserve
        # their legacy recent-window signal, but production runtime always
        # installs the complete-request estimator above.
        used = sum(len(record.message.content) for record in recent)
        return (
            used / self._max_recent_context_chars,
            projection_overflow,
            None,
            None,
        )

    def _bound_recent_messages(
        self, messages: tuple[StoredConversationMessage, ...]
    ) -> tuple[StoredConversationMessage, ...]:
        selected = []
        used = 0
        # Reserve a fair share for at least four recent messages when four are
        # available, while letting one or two messages use the old half-window
        # ceiling. This prevents two giant turns from evicting the other six
        # without needlessly clipping a genuinely short window.
        uncrowded_chars = sum(
            min(len(record.message.content), self._max_recent_message_chars)
            for record in messages
        )
        if uncrowded_chars <= self._max_recent_context_chars:
            per_message_chars = self._max_recent_message_chars
        else:
            fair_message_chars = max(
                1,
                self._max_recent_context_chars // min(len(messages) or 1, 4),
            )
            per_message_chars = min(
                self._max_recent_message_chars, fair_message_chars
            )
        for record in reversed(messages):
            remaining = self._max_recent_context_chars - used
            if remaining <= 0:
                break
            content = record.message.content[
                : min(remaining, per_message_chars)
            ]
            content_clipped = len(content) < len(record.message.content)
            selected.append(
                record.model_copy(
                    update={
                        "message": record.message.model_copy(
                            update={
                                "content": content,
                                "content_clipped": content_clipped,
                            }
                        )
                    }
                )
            )
            used += len(content)
        return tuple(reversed(selected))

    def _deduplicate_recent_messages(
        self, messages: tuple[StoredConversationMessage, ...]
    ) -> tuple[StoredConversationMessage, ...]:
        """Clear older exact large-body duplicates without touching storage."""
        seen: set[tuple[str, str]] = set()
        projected = []
        for record in reversed(messages):
            message = record.message
            identity = (message.role, message.content)
            if (
                len(message.content) >= self._RECENT_DEDUP_MIN_CHARS
                and identity in seen
            ):
                record = record.model_copy(
                    update={
                        "message": message.model_copy(
                            update={"content": self._RECENT_DUPLICATE_MARKER}
                        )
                    }
                )
            elif len(message.content) >= self._RECENT_DEDUP_MIN_CHARS:
                seen.add(identity)
            projected.append(record)
        return tuple(reversed(projected))
