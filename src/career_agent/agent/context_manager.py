from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Literal, Protocol

from career_agent.agent.conversation_memory_contracts import (
    ACTIVE_CONSTRAINT_MAX_ITEMS as _ACTIVE_CONSTRAINT_MAX_ITEMS,
    SUMMARY_TEXT_MAX_CHARS as _SUMMARY_TEXT_BUDGET,
    ConversationSummaryContent,
    ConversationSummaryWorker,
    DistilledFreeTextPreferenceCandidate,
    SummaryMessage,
)
from career_agent.agent.career_context import reciprocal_rank_fusion
from career_agent.agent.main_agent_contracts import (
    CareerProfileBudgets,
    CareerProfileContext,
    ConversationMessageContext,
    ConversationResourceReference,
    ConversationTaskState,
    EpisodeProjectionContext,
    MainAgentContext,
    CurrentTargetContext,
    FreeTextPreferenceContext,
    MemoryTelemetryBinding,
    OwnerSettingsContext,
    WorkingNotesContext,
)
from career_agent.agent.openai_compatible_client import AgentWorkerError
from career_agent.agent.token_budget import clip_to_tokens, message_token_count
from career_agent.agent.session_manager import SessionManager
from career_agent.domain.episodes import CareerEpisodeDraft
from career_agent.domain.resume import TargetRole
from career_agent.harness.observability import (
    conversation_trace_key,
    record_active_trace,
)
from career_agent.services.episode_consolidation import mock_interview_exit_draft
from career_agent.storage.context import CareerContextStore, StoredConversationMessage
from career_agent.storage.episodes import SQLiteCareerEpisodeStore
from career_agent.storage.intent_versions import intent_entry_id
from career_agent.storage.working_notes import WorkingNotesSnapshot
from career_agent.services.free_text_preferences import (
    preference_scope_domain,
    preference_topic_is_relevant,
    preference_topic_key,
)
from career_agent.services.preference_resolution import (
    PreferenceResolutionContext,
    preference_scope_name,
    resolve_effective_preferences,
)


WORKING_NOTES_STALE_DAYS = 14


class TargetRoleSource(Protocol):
    def list_target_roles(
        self, *, user_id: str, limit: int | None = None
    ) -> tuple[TargetRole, ...]: ...

    def count_target_roles(self, *, user_id: str) -> int: ...

    def list_target_role_intent_versions(
        self,
        *,
        user_id: str,
        scope_keys: tuple[str, ...] | None = None,
        pref_scope: str | None = None,
        active_only: bool = False,
        limit: int | None = None,
    ) -> tuple[object, ...]: ...


class WorkingNotesSource(Protocol):
    def read(self, *, user_id: str) -> WorkingNotesSnapshot: ...


@dataclass(frozen=True)
class _Pressure:
    """How full the next request is expected to be, and how that was known."""

    occupancy: float | None
    """``None`` when nothing was measured, which never triggers compaction."""

    projection_overflow: bool
    request_tokens: int | None
    max_input_tokens: int | None
    context: MainAgentContext | None
    """The context built to measure, so a load that does not compact can use it
    instead of building again. Only the estimated path builds one."""

    source: Literal["estimated", "carried", "legacy", "seam", "overflow_only"]


class ContextManager:
    _MAX_STATIC_INPUT_FRACTION = 0.5
    # Shares of what the static request leaves, so raising the input budget
    # raises them without retuning. The other 40% is for projections and this
    # turn's observations, the headroom the 0.75 compaction threshold implies.
    _USER_MESSAGE_FRACTION = 0.20
    _RECENT_WINDOW_FRACTION = 0.40
    _RECENT_MESSAGE_FRACTION = 0.15
    # Consecutive summarizer failures after which a conversation stops calling
    # it, and how long it then waits before letting one attempt through. Each
    # failed call can cost the client's retries times its timeout, on the load
    # and again on the commit. The interval is a setting, not a measurement.
    _COMPACTION_FAILURE_LIMIT = 3
    _COMPACTION_RETRY_AFTER = timedelta(minutes=10)
    _TARGET_ROLE_SAFETY_LIMIT = 100
    _TELEMETRY_VERSION_LIMIT = 512
    _RECENT_DEDUP_MIN_CHARS = 512
    _RECENT_DUPLICATE_MARKER = (
        "[duplicate content omitted; identical to a newer visible message "
        "in this recent window]"
    )

    def __init__(self, store: CareerContextStore, *, session_manager: SessionManager | None = None, summary_worker: ConversationSummaryWorker | None = None, recent_message_limit: int = 8, summary_batch_size: int = 4, max_message_chars: int = 32000, max_recent_context_chars: int = 32000, max_recent_message_chars: int | None = None, max_user_message_tokens: int | None = None, max_recent_context_tokens: int | None = None, max_recent_message_tokens: int | None = None, compact_occupancy_threshold: float = 0.75, archived_resource_limit: int = 12, target_role_source: TargetRoleSource | None = None, career_profile_budgets: CareerProfileBudgets | None = None, episode_store: SQLiteCareerEpisodeStore | None = None, working_notes_store: WorkingNotesSource | None = None, clock: Callable[[], datetime] | None = None) -> None:
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
        if any(
            limit is not None and limit < 1
            for limit in (
                max_user_message_tokens,
                max_recent_context_tokens,
                max_recent_message_tokens,
            )
        ):
            raise ValueError("conversation token budgets are invalid")
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
        self._archived_resource_limit = archived_resource_limit
        self._target_role_source = target_role_source
        self._episode_store = episode_store
        self._working_notes_store = working_notes_store
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._career_profile_budgets = (
            career_profile_budgets or CareerProfileBudgets()
        )
        self._request_token_estimator: (
            Callable[[MainAgentContext], tuple[int, int]] | None
        ) = None
        # Explicit token caps win over the ones derived from the request
        # budget. With neither, messages are bounded by characters alone.
        self._max_user_message_tokens = max_user_message_tokens
        self._max_recent_context_tokens = max_recent_context_tokens
        self._max_recent_message_tokens = max_recent_message_tokens
        self._user_message_tokens = max_user_message_tokens
        self._recent_context_tokens = max_recent_context_tokens
        self._recent_message_tokens = max_recent_message_tokens
        # The complete-request estimate each conversation's latest load took,
        # left for that turn's commit. The commit removes what it reads, so this
        # holds at most one entry per conversation with a turn in flight.
        self._carried_request_tokens: dict[tuple[str, str], tuple[int, int]] = {}
        # Consecutive summarizer failures per conversation, and when the latest
        # one suspended compaction. Removed by the next successful summary.
        self._compaction_failures: dict[
            tuple[str, str], tuple[int, datetime | None]
        ] = {}

    def now(self) -> datetime:
        """The one clock for stamping proposals and expiring them.

        The runtime passes this to ``reduce_task_state`` so a proposal's stamp
        and the age check that later expires it read the same time source.
        """
        return self._clock()

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

        The static measurement also sets the token caps on this turn's message
        and the recent window, as shares of what the static request leaves.
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
            dynamic_tokens = max_input_tokens - static_input_tokens

            def derived(explicit: int | None, fraction: float) -> int:
                if explicit is not None:
                    return explicit
                return max(1, int(dynamic_tokens * fraction))

            self._user_message_tokens = derived(
                self._max_user_message_tokens, self._USER_MESSAGE_FRACTION
            )
            self._recent_context_tokens = derived(
                self._max_recent_context_tokens, self._RECENT_WINDOW_FRACTION
            )
            self._recent_message_tokens = derived(
                self._max_recent_message_tokens, self._RECENT_MESSAGE_FRACTION
            )
        self._request_token_estimator = estimator

    def load_for_turn(
        self, *, user_id: str, conversation_id: str, user_message: str
    ) -> MainAgentContext:
        self._sessions.get_or_create(user_id=user_id, session_id=conversation_id)
        self._store.capture_free_text_preference_from_message(
            user_id=user_id,
            conversation_id=conversation_id,
            message=user_message,
        )
        pressure = self._maybe_summarize(
            user_id=user_id,
            conversation_id=conversation_id,
            trigger="occupancy",
            measure=lambda after_sequence: self._recent_pressure(
                user_id=user_id,
                conversation_id=conversation_id,
                after_sequence=after_sequence,
                user_message=user_message,
            ),
        )
        expired = self._expire_stale_proposals(
            user_id=user_id, conversation_id=conversation_id
        )
        # Measuring pressure builds the whole context, and when the measurement
        # decides not to compact, nothing downstream has changed it: same
        # summary, same messages, same projections. Rebuilding it here would
        # repeat a full construction, two FTS rankings and a notes read for an
        # identical result. Compaction or an expired proposal invalidates it, so
        # those paths fall through to a fresh build.
        if (
            pressure is not None
            and pressure.context is not None
            and pressure.request_tokens is not None
            and pressure.max_input_tokens is not None
            and not expired
        ):
            self._carry_request_estimate(
                user_id=user_id,
                conversation_id=conversation_id,
                context=pressure.context,
                measured=(pressure.request_tokens, pressure.max_input_tokens),
            )
            return pressure.context
        context = self._build_context(
            user_id=user_id,
            conversation_id=conversation_id,
            user_message=user_message,
        )
        self._carry_request_estimate(
            user_id=user_id,
            conversation_id=conversation_id,
            context=context,
            measured=None,
        )
        return context

    def _carry_request_estimate(
        self,
        *,
        user_id: str,
        conversation_id: str,
        context: MainAgentContext,
        measured: tuple[int, int] | None,
    ) -> None:
        """Leave this load's complete-request estimate for the turn's commit.

        The commit decides whether to compact from this number plus its reply,
        instead of building the whole context again to measure it. Both use the
        same estimator, so the load and the commit judge occupancy on one scale.

        ``measured`` is ``None`` when compaction or an expired proposal changed
        storage after the pressure measurement. That measurement then describes
        a context this turn no longer has, and a pre-compaction figure would
        compact a second batch at commit, so the returned context is estimated
        instead. That costs an estimate, not a build, and only on a turn that
        already paid for a summary.
        """
        if self._summary_worker is None or self._request_token_estimator is None:
            return
        request_tokens, max_input_tokens = (
            measured if measured is not None else self._estimate_request(context)
        )
        self._carried_request_tokens[(user_id, conversation_id)] = (
            request_tokens,
            max_input_tokens,
        )
        # Numbers only, under the conversation's pseudonymous key. The first of
        # these in a run pairs with the run's first model_succeeded: that
        # request carries no observations yet, so its provider input_units
        # counts the same request this estimated. The pair is what calibrating
        # the estimator needs, and before this it was recorded only on turns
        # that compacted.
        record_active_trace(
            "context_estimated",
            "request_estimate",
            outcome="succeeded",
            details={
                "conversation_key": conversation_trace_key(
                    user_id, conversation_id
                ),
                # Named as on context_compacted, which carries the same two
                # numbers. Trace redaction masks keys containing "token" unless
                # they are on its short allowlist of cache metrics.
                "input_occupancy_numerator": request_tokens,
                "input_occupancy_denominator": max_input_tokens,
            },
        )

    def _expire_stale_proposals(
        self, *, user_id: str, conversation_id: str
    ) -> bool:
        """Clear pending proposals the user has not confirmed within the TTL.

        Done at load rather than left to the confirm gates alone, so an expired
        proposal is gone from the state the turn runs on, not merely refused
        once the model has already reached for it.

        Returns whether anything expired, so a caller holding a context built
        before this ran knows it is stale.
        """
        task = self._store.get_task(user_id, conversation_id)
        if task is None:
            return False
        kept, expired = task.expire_stale_proposals(self.now())
        if not expired:
            return False
        self._store.upsert_task(
            user_id=user_id,
            conversation_id=conversation_id,
            task=kept,
        )
        record_active_trace(
            "memory_proposal_expired",
            "task_state",
            outcome="succeeded",
            details={
                "conversation_key": conversation_trace_key(
                    user_id, conversation_id
                ),
                "slots": list(expired),
            },
        )
        return True

    def _build_context(
        self, *, user_id: str, conversation_id: str, user_message: str
    ) -> MainAgentContext:
        # Retrieval reads the prompt copy too. An FTS query gets one OR term
        # per trigram of the message, and MATCH time grows faster than the term
        # count: a 32k-character paste costs seconds per search, and a few
        # thousand common trigrams dilute every ranking. The head of a long
        # paste already carries its key terms.
        prompt_message, user_message_clipped, user_message_source = (
            self._bound_user_message(user_message)
        )
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
        working_notes = None
        if self._working_notes_store is not None:
            snapshot = self._working_notes_store.read(user_id=user_id)
            stale_days = None
            if snapshot.markdown and snapshot.updated_at is not None:
                age_days = max(
                    0,
                    int(
                        (self._clock() - snapshot.updated_at).total_seconds()
                        // 86_400
                    ),
                )
                if age_days > WORKING_NOTES_STALE_DAYS:
                    stale_days = age_days
            working_notes = WorkingNotesContext(
                markdown=snapshot.markdown,
                revision=snapshot.revision,
                clipped=snapshot.clipped,
                stale_days=stale_days,
            )
            if snapshot.clipped:
                record_active_trace(
                    "working_notes_oversize",
                    "working_notes",
                    outcome="succeeded",
                    details={"chars": len(snapshot.markdown)},
                )
        (
            free_text_preferences,
            active_preference_total,
            quarantined_preference_total,
        ) = self._free_text_preference_context(
            user_id=user_id,
            conversation_id=conversation_id,
            user_message=prompt_message,
            task=task,
        )
        return MainAgentContext(
            conversation_id=conversation_id,
            spotlight_nonce=session.spotlight_nonce if session else None,
            profile=profile,
            career_profile_budgets=self._career_profile_budgets,
            preferences=preferences,
            free_text_preferences=free_text_preferences,
            free_text_preferences_active_total=active_preference_total,
            free_text_preferences_quarantined_total=(
                quarantined_preference_total
            ),
            working_notes=working_notes,
            career_episodes=self._episode_context(
                user_id=user_id,
                conversation_id=conversation_id,
                user_message=prompt_message,
            ),
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
            user_message=prompt_message,
            user_message_source=user_message_source,
            user_message_clipped=user_message_clipped,
        )

    def _episode_context(
        self,
        *,
        user_id: str,
        conversation_id: str,
        user_message: str,
    ) -> tuple[EpisodeProjectionContext, ...]:
        if self._episode_store is None:
            return ()
        episodes = self._episode_store.project_relevant(
            user_id=user_id,
            query=user_message,
            limit=5,
            exclude_conversation_id=conversation_id,
        )
        return tuple(
            EpisodeProjectionContext(
                detail_ref=f"episode:{episode.id}",
                kind=episode.kind,
                occurred_at=episode.occurred_at,
                title=episode.title,
                synopsis=episode.summary,
            )
            for episode in episodes
        )

    def mark_episodes_projected(
        self, *, user_id: str, context: MainAgentContext
    ) -> None:
        """Stamp exposure once, for the episodes the model is about to be shown.

        The store's projection is a pure read, so this is where an episode's
        access is recorded. Called from the runtime immediately before the
        decision call rather than from ``_build_context``: a turn builds its
        context several times over (pressure measurement, the load itself, a
        reload after a memory write) and only one of those is a moment the model
        actually saw the episodes.
        """
        if self._episode_store is None:
            return
        episode_ids = tuple(
            item.detail_ref.removeprefix("episode:")
            for item in context.career_episodes
        )
        if not episode_ids:
            return
        self._episode_store.mark_accessed(
            user_id=user_id,
            episode_ids=episode_ids,
        )

    def get_task(
        self, *, user_id: str, conversation_id: str
    ) -> ConversationTaskState:
        """Read only the routing state before choosing a context owner."""
        return self._store.get_task(user_id, conversation_id) or ConversationTaskState()

    def disarm_bare_confirmation(
        self,
        *,
        user_id: str,
        conversation_id: str,
        task: ConversationTaskState,
    ) -> ConversationTaskState:
        """Consume one proposal's adjacent-turn bare-confirmation privilege."""

        if task.bare_confirmation_target is None:
            return task
        disarmed = task.model_copy(
            update={"bare_confirmation_target": None}
        )
        self._store.upsert_task(
            user_id=user_id,
            conversation_id=conversation_id,
            task=disarmed,
        )
        return disarmed

    def load_for_workflow_turn(
        self,
        *,
        user_id: str,
        conversation_id: str,
        task: ConversationTaskState,
    ) -> MainAgentContext:
        """Build a routing envelope without loading Main Agent memory."""
        # This load measures nothing. An estimate left by an earlier load whose
        # turn never committed must not reach this turn's commit.
        self._carried_request_tokens.pop((user_id, conversation_id), None)
        self._sessions.get_or_create(user_id=user_id, session_id=conversation_id)
        profile = self._stored_profile_context(user_id)
        preferences = self._store.get_owner_settings(user_id) or OwnerSettingsContext()
        (
            free_text_preferences,
            active_preference_total,
            quarantined_preference_total,
        ) = self._free_text_preference_context(
            user_id=user_id,
            conversation_id=conversation_id,
            user_message="",
            task=task,
        )
        return MainAgentContext(
            conversation_id=conversation_id,
            spotlight_nonce=self._sessions.get_or_create(
                user_id=user_id, session_id=conversation_id
            ).spotlight_nonce,
            profile=profile,
            career_profile_budgets=self._career_profile_budgets,
            preferences=preferences,
            free_text_preferences=free_text_preferences,
            free_text_preferences_active_total=active_preference_total,
            free_text_preferences_quarantined_total=(
                quarantined_preference_total
            ),
            task=task,
            recent_messages=(),
            conversation_summary=None,
            user_message="[workflow-owned input withheld]",
        )

    def _free_text_preference_context(
        self,
        *,
        user_id: str,
        conversation_id: str,
        user_message: str,
        task: ConversationTaskState,
    ) -> tuple[tuple[FreeTextPreferenceContext, ...], int, int]:
        """Project the preferences that survive relevance, plus what was cut.

        Returns the projected items and the totals they were selected from, so
        the projection can tell the model it is looking at a window rather than
        the whole set. Without the totals the cap is silent, and the first thing
        it drops is a quarantined candidate that was waiting to be confirmed.
        """
        versions = self._store.list_free_text_preferences(user_id=user_id)
        current_by_scope: dict[tuple[str, str], list[object]] = {}
        for item in versions:
            current_by_scope.setdefault(
                (item.scope_key, item.pref_scope), []
            ).append(item)
        projected = []
        active_candidates = []
        quarantined_candidates = []
        for (scope_key, _), track in sorted(current_by_scope.items()):
            deleted_at = self._store.free_text_preference_deleted_at(
                user_id=user_id,
                scope_key=scope_key,
            )
            quarantined = next(
                (
                    item
                    for item in track
                    if item.admission_status == "quarantined"
                    and (deleted_at is None or item.valid_from > deleted_at)
                ),
                None,
            )
            active = (
                next(
                    (
                        item
                        for item in track
                        if item.admission_status == "active"
                    ),
                    None,
                )
                if deleted_at is None
                else None
            )
            # A conflicting candidate suspends only its own ownership track.
            # Narrower active tracks are resolved together below.
            if quarantined is not None:
                quarantined_candidates.append(quarantined)
            elif active is not None:
                active_candidates.append(active)

        target_role_id = None
        role_domains: tuple[str, ...] = ()
        get_resume_version = getattr(
            self._target_role_source, "get_version", None
        )
        if (
            task.active_resume_version_id is not None
            and callable(get_resume_version)
        ):
            source = get_resume_version(
                user_id=user_id,
                resume_version_id=task.active_resume_version_id,
            )
            if source is not None:
                resume, _ = source
                target_role_id = resume.target_role_id
                get_target_role = getattr(
                    self._target_role_source, "get_target_role", None
                )
                role = (
                    get_target_role(
                        user_id=user_id,
                        target_role_id=target_role_id,
                    )
                    if callable(get_target_role)
                    else None
                )
                domain = preference_scope_domain(
                    role.title if role is not None else None
                )
                role_domains = (domain,) if domain is not None else ()
        selected = (
            *resolve_effective_preferences(
                active_candidates,
                context=PreferenceResolutionContext(
                    target_role_id=target_role_id,
                    role_domains=role_domains,
                    job_posting_id=task.active_job_posting_id,
                    conversation_id=conversation_id,
                ),
            ),
            *quarantined_candidates,
        )
        # One search per status, each with its own limit. A shared limit lets a
        # user with many confirmed preferences fill every row with active hits,
        # and a candidate that never enters the rankings cannot pass the gate.
        quarantine_relevance = reciprocal_rank_fusion(
            *self._store.search_free_text_preference_rankings(
                user_id=user_id,
                query=user_message,
                limit=32,
                statuses=("quarantined",),
            )
        )
        active_relevance = reciprocal_rank_fusion(
            *self._store.search_free_text_preference_rankings(
                user_id=user_id,
                query=user_message,
                limit=32,
                statuses=("active",),
            )
        )
        retrieved_quarantine_ids = {
            update_id
            for update_id, _ in sorted(
                quarantine_relevance.items(),
                key=lambda item: (-item[1], item[0]),
            )[:8]
        }
        for item in selected:
            topic_key = preference_topic_key(item.scope_key)
            if (
                item.admission_status == "quarantined"
                and not preference_topic_is_relevant(
                    topic_key,
                    user_message,
                    item.value,
                )
                and item.update_id not in retrieved_quarantine_ids
            ):
                continue
            scope_name = preference_scope_name(item.pref_scope)
            ownership = (
                "person_stable"
                if item.layer == "stable"
                else (
                    "person_situational"
                    if scope_name == "person_situational"
                    else "situational"
                )
                if item.layer == "transient"
                else "role"
                if scope_name.startswith("role.")
                else "person_default"
            )
            projected.append(
                FreeTextPreferenceContext(
                    scope_key=item.scope_key,
                    topic_key=topic_key,
                    statement=item.value,
                    status=item.admission_status,
                    ownership=ownership,
                    pref_scope=item.pref_scope,
                    layer=item.layer,
                    timescale=item.timescale,
                    valid_until=item.valid_until,
                    needs_scope_clarification=(
                        item.capture_action == "ask"
                    ),
                    observed_at=item.valid_from,
                    confirmed_at=(
                        item.last_corroborated_at
                        if item.admission_status == "active"
                        else None
                    ),
                    update_id=item.update_id,
                )
            )
        # Confirmed preferences are ranked by relevance to this message before
        # the cap bites, so what survives is what the turn is about rather than
        # wherever the resolver happened to leave it. Ties keep resolver order,
        # which is stable, and an unranked statement sorts after every hit.
        active_projected = sorted(
            (item for item in projected if item.status == "active"),
            key=lambda item: -active_relevance.get(item.update_id, 0.0),
        )
        quarantined_projected = [
            item for item in projected if item.status == "quarantined"
        ]
        # Quarantined candidates get a reserved share of the eight slots, capped
        # at the three the projection renders, so a long list of confirmed
        # preferences cannot starve the candidate the turn is meant to raise. An
        # unused quarantined slot goes to active; the reverse does not hold,
        # because a fourth candidate would not be rendered anyway.
        quarantined_slots = min(len(quarantined_projected), 3)
        return (
            (
                *active_projected[: 8 - quarantined_slots],
                *quarantined_projected[:quarantined_slots],
            ),
            len(active_projected),
            len(quarantined_projected),
        )

    def _stored_profile_context(self, user_id: str) -> CareerProfileContext:
        return self._store.get_profile(user_id) or CareerProfileContext(
            user_id=user_id
        )

    def _profile_context(self, user_id: str) -> CareerProfileContext:
        profile = self._profile_with_confirmation_times(
            self._stored_profile_context(user_id)
        )
        if self._target_role_source is None:
            return self._with_intent_telemetry_bindings(
                profile,
                current_targets=(),
            )
        try:
            target_roles = self._target_role_source.list_target_roles(
                user_id=user_id,
                limit=self._TARGET_ROLE_SAFETY_LIMIT,
            )
        except TypeError:
            # Structural test doubles written before the safety-limit argument
            # still get bounded at the projection boundary.
            target_roles = self._target_role_source.list_target_roles(
                user_id=user_id
            )[: self._TARGET_ROLE_SAFETY_LIMIT]
        count_roles = getattr(self._target_role_source, "count_target_roles", None)
        current_targets_total = (
            int(count_roles(user_id=user_id))
            if callable(count_roles)
            else len(target_roles)
        )
        if current_targets_total != len(target_roles):
            raise ValueError(
                "current target safety limit would make profile projection incomplete"
            )
        current_targets = self._targets_with_confirmation_times(
            profile.user_id,
            tuple(
                CurrentTargetContext(
                target_role_id=role.id,
                title=role.title,
                priority=role.priority,
                status=role.status,
                city=role.city,
                salary_expectation=role.salary_expectation,
                experience=role.experience,
                education=role.education,
            )
            for role in target_roles
            ),
        )
        projected = profile.model_copy(
            update={
                "current_targets": current_targets,
                "current_targets_total": current_targets_total,
            }
        )
        return self._with_intent_telemetry_bindings(
            projected,
            current_targets=current_targets,
        )

    def _profile_with_confirmation_times(
        self,
        profile: CareerProfileContext,
    ) -> CareerProfileContext:
        scopes = {
            *(
                ("person_intent/self/default_city",)
                if profile.default_city is not None
                else ()
            ),
            *(
                f"person_intent/self/{constraint.relation}"
                for constraint in profile.hard_constraints
            ),
        }
        if not scopes:
            return profile
        versions = self._store.list_profile_intent_versions(
            user_id=profile.user_id,
            scope_keys=tuple(sorted(scopes)),
            pref_scope="global",
            active_only=True,
        )
        if not versions:
            return profile
        confirmed = {
            item.scope_key: item.last_corroborated_at for item in versions
        }
        return profile.model_copy(
            update={
                "default_city_confirmed_at": confirmed.get(
                    "person_intent/self/default_city"
                ),
                "hard_constraints": tuple(
                    constraint.model_copy(
                        update={
                            "confirmed_at": confirmed.get(
                                f"person_intent/self/{constraint.relation}"
                            )
                        }
                    )
                    for constraint in profile.hard_constraints
                ),
            }
        )

    def _targets_with_confirmation_times(
        self,
        user_id: str,
        targets: tuple[CurrentTargetContext, ...],
    ) -> tuple[CurrentTargetContext, ...]:
        version_reader = getattr(
            self._target_role_source,
            "list_target_role_intent_versions",
            None,
        )
        if not callable(version_reader):
            return targets
        try:
            versions = version_reader(
                user_id=user_id,
                pref_scope="global",
                active_only=True,
            )
        except TypeError:
            versions = tuple(
                item
                for item in version_reader(
                    user_id=user_id,
                    active_only=True,
                )
                if getattr(item, "pref_scope", "global") == "global"
            )
        confirmed = {
            item.scope_key: item.last_corroborated_at
            for item in versions
        }
        result = []
        for target in targets:
            changes = {}
            for relation in (
                "city",
                "salary_expectation",
                "experience",
                "education",
            ):
                if getattr(target, relation) is None:
                    continue
                confirmed_at = confirmed.get(
                    f"target_role_intent/{target.target_role_id}/{relation}"
                )
                if confirmed_at is not None:
                    changes[f"{relation}_confirmed_at"] = confirmed_at
            result.append(
                target.model_copy(update=changes) if changes else target
            )
        return tuple(result)

    def _with_intent_telemetry_bindings(
        self,
        profile: CareerProfileContext,
        *,
        current_targets: tuple[CurrentTargetContext, ...],
    ) -> CareerProfileContext:
        profile_scopes = (
            {"person_intent/self/default_city"}
            if profile.default_city
            else set()
        )
        profile_scopes.update(
            f"person_intent/self/{constraint.relation}"
            for constraint in profile.hard_constraints
        )
        target_scopes = {
            f"target_role_intent/{target.target_role_id}/{relation}"
            for target in current_targets
            if target.target_role_id is not None
            for relation in ("city", "salary_expectation", "experience", "education")
            if getattr(target, relation) is not None
        }
        limit = self._TELEMETRY_VERSION_LIMIT + 1
        profile_versions = self._store.list_profile_intent_versions(
            user_id=profile.user_id,
            scope_keys=tuple(sorted(profile_scopes)),
            limit=limit,
        )
        target_versions: tuple[object, ...] = ()
        version_reader = getattr(
            self._target_role_source,
            "list_target_role_intent_versions",
            None,
        )
        target_reader_available = callable(version_reader)
        if target_scopes and target_reader_available:
            target_versions = tuple(
                version_reader(
                    user_id=profile.user_id,
                    scope_keys=tuple(sorted(target_scopes)),
                    limit=limit,
                )
            )
        versions = tuple(
            version
            for version in (tuple(profile_versions) + target_versions)
            if getattr(version, "admission_status", "active") == "active"
            and getattr(version, "pref_scope", "global") == "global"
        )
        clipped = len(versions) > self._TELEMETRY_VERSION_LIMIT
        selected = versions[: self._TELEMETRY_VERSION_LIMIT]
        bindings = tuple(
            MemoryTelemetryBinding(
                entry_id=intent_entry_id(
                    version.scope_key,
                    getattr(version, "pref_scope", "global"),
                ),
                update_id=version.update_id,
                content_digest=version.content_digest,
                value=version.value,
                revision=version.revision,
                lifecycle_status=(
                    "current"
                    if version.superseded_at is None
                    else "superseded"
                ),
            )
            for version in selected
            if all(
                hasattr(version, name)
                for name in (
                    "scope_key",
                    "update_id",
                    "content_digest",
                    "value",
                    "revision",
                    "superseded_at",
                )
            )
            and len(version.value) <= 32_000
        )
        active_scopes = {
            binding.entry_id
            for binding in bindings
            if binding.lifecycle_status == "current"
        }
        expected_scopes = profile_scopes | target_scopes
        complete = (
            not clipped
            and len(bindings) == len(selected)
            and expected_scopes <= active_scopes
            and (not target_scopes or target_reader_available)
        )
        return profile.model_copy(
            update={
                "telemetry_bindings": bindings,
                "telemetry_inventory_complete": complete,
            }
        )

    def commit_turn(self, *, context: MainAgentContext, task: ConversationTaskState, assistant_message: str, assistant_resource_refs: tuple[ConversationResourceReference, ...] = (), compaction_trigger: Literal["occupancy", "seam"] = "occupancy", episode_drafts: tuple[CareerEpisodeDraft, ...] = (), memory_scope_keys: tuple[str, ...] = ()) -> None:
        now = datetime.now(timezone.utc)
        # This is deliberately exposure-level provenance. Every career scope
        # shown to the model binds both stored messages in the turn, even when
        # the reply did not visibly use it. A tombstone may therefore suppress
        # incidental text from that turn; relying on model-reported usage would
        # create a false-negative path for deleted claims.
        #
        # The caller reports what was exposed, because no single context value
        # can: a turn that amends or tombstones memory reloads its context
        # mid-flight, which would erase the record of what the model had
        # already been shown before the write.
        memory_scope_keys = tuple(dict.fromkeys(memory_scope_keys))
        self._store.commit_turn(
            user_id=context.profile.user_id,
            conversation_id=context.conversation_id,
            task=task,
            user_message=ConversationMessageContext(role="user", content=self._truncate(context.stored_user_message()), created_at=now),
            assistant_message=ConversationMessageContext(role="assistant", content=self._truncate(assistant_message), created_at=now, resource_refs=assistant_resource_refs),
            episode_drafts=episode_drafts,
            memory_scope_keys=memory_scope_keys,
        )
        carried = self._carried_request_tokens.pop(
            (context.profile.user_id, context.conversation_id), None
        )
        stored_assistant_message = self._truncate(assistant_message)
        self._maybe_summarize(
            user_id=context.profile.user_id,
            conversation_id=context.conversation_id,
            trigger=compaction_trigger,
            measure=lambda after_sequence: self._commit_pressure(
                user_id=context.profile.user_id,
                conversation_id=context.conversation_id,
                after_sequence=after_sequence,
                trigger=compaction_trigger,
                carried=carried,
                assistant_message=stored_assistant_message,
            ),
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
        held = task.hold_entry_message(
            self._truncate(context.stored_user_message())
        )
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
            context=context.model_copy(
                update={
                    "user_message": entry,
                    # The held request is already the stored copy. A source
                    # left over from this turn's input would be written instead.
                    "user_message_source": None,
                    "user_message_clipped": False,
                }
            ),
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

    def _bound_user_message(
        self, user_message: str
    ) -> tuple[str, bool, str | None]:
        """The prompt copy of this turn's message, whether it was cut, and the
        original when it was.

        Without a token cap this is the character cut alone and reports
        nothing, as it did before the cap existed.
        """
        prompt_message = self._truncate(user_message)
        if self._user_message_tokens is None:
            return prompt_message, False, None
        prompt_message, _ = clip_to_tokens(
            prompt_message, self._user_message_tokens
        )
        clipped = len(prompt_message) < len(user_message)
        return prompt_message, clipped, user_message if clipped else None

    def _maybe_summarize(
        self,
        *,
        user_id: str,
        conversation_id: str,
        trigger: Literal["occupancy", "seam"],
        measure: Callable[[int], _Pressure],
    ) -> _Pressure | None:
        """Compact one batch if the turn is under pressure.

        ``measure`` receives the summary boundary and reports the pressure. It
        runs only when a summary worker exists, so a manager that cannot compact
        never measures.

        Returns the measurement when storage is still as it found it — that is,
        when no compaction happened — so ``load_for_turn`` can use the context
        it built instead of building the same thing again. ``None`` means the
        summary boundary moved or nothing was measured.
        """
        if self._summary_worker is None:
            return None
        return self._compact_one_batch(
            user_id=user_id,
            conversation_id=conversation_id,
            trigger=trigger,
            require_occupancy=trigger == "occupancy",
            measure=measure,
        )

    def _compact_one_batch(
        self,
        *,
        user_id: str,
        conversation_id: str,
        trigger: Literal["occupancy", "seam"],
        require_occupancy: bool,
        measure: Callable[[int], _Pressure],
    ) -> _Pressure | None:
        """Compact at most one batch.

        Returns the measurement, but only when this call left storage as it
        found it. Compacting invalidates any context it carries, so those paths
        return ``None`` and the caller rebuilds.
        """
        previous = self._store.get_conversation_summary(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        previous_through = previous.through_sequence if previous else 0
        pressure = measure(previous_through)
        if (
            require_occupancy
            and not pressure.projection_overflow
            # Unmeasured occupancy is never a reason to compact; the next load
            # measures. A seam does not require occupancy and skips this.
            and (
                pressure.occupancy is None
                or pressure.occupancy < self._compact_occupancy_threshold
            )
        ):
            return pressure
        messages = self._store.list_messages_after(
            user_id=user_id,
            conversation_id=conversation_id,
            after_sequence=previous_through,
            limit=self._summary_batch_size,
        )
        if len(messages) < self._summary_batch_size:
            return pressure
        to_summarize = messages[: self._summary_batch_size]
        if self._compaction_suspended(
            user_id=user_id, conversation_id=conversation_id
        ):
            return pressure
        try:
            content = self._summary_worker.summarize(
                previous=previous.content if previous else None,
                messages=to_summarize,
            )
        except AgentWorkerError as error:
            self._record_compaction_failure(
                user_id=user_id, conversation_id=conversation_id, error=error
            )
            return pressure
        self._compaction_failures.pop((user_id, conversation_id), None)
        preference_candidates_proposed = len(content.long_term_memory_candidates)
        preference_candidates = self._validated_preference_candidates(
            content.long_term_memory_candidates,
            messages=to_summarize,
        )
        dropped_user_goals = 0
        dropped_confirmed_decisions = 0
        dropped_unresolved_questions = 0
        omitted_user_goal_count = (
            previous.content.omitted_user_goal_count
            if previous is not None
            else 0
        )
        omitted_confirmed_decision_count = (
            previous.content.omitted_confirmed_decision_count
            if previous is not None
            else 0
        )
        omitted_unresolved_question_count = (
            previous.content.omitted_unresolved_question_count
            if previous is not None
            else 0
        )
        # Constraint deletion is not delegated to a lossy rewrite. The ledger
        # behind the summary owns both directions: ``retired`` rows are
        # excluded here so a rewrite that copies constraints forward verbatim
        # cannot resurrect one, and ``omitted`` rows re-enter as candidates so
        # retiring a constraint readmits the oldest archived one instead of
        # stranding it.
        ledger = self._store.list_conversation_constraints(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        retired_constraints = {
            row.text for row in ledger if row.status == "retired"
        }
        prior_constraints = (
            previous.content.active_constraints if previous is not None else ()
        )
        candidate_constraints = tuple(
            text
            for text in dict.fromkeys(
                (
                    # Ledger order is first-seen order, which is the ordering
                    # the cap needs. Prior visible constraints follow it to
                    # cover a summary written before the ledger existed.
                    *(
                        row.text
                        for row in ledger
                        if row.status in ("active", "omitted")
                    ),
                    *prior_constraints,
                    *content.active_constraints,
                )
            )
            if text not in retired_constraints
        )
        # The contract budgets every summary field together, so fifteen
        # constraints can exceed the whole allowance on their own and leave
        # nothing for the fields below. Oldest first, so a constraint that
        # already survived a rewrite is never traded for a newer one.
        constraint_chars = 0
        visible: list[str] = []
        for constraint in candidate_constraints:
            if len(visible) >= _ACTIVE_CONSTRAINT_MAX_ITEMS:
                break
            if constraint_chars + len(constraint) > _SUMMARY_TEXT_BUDGET:
                # Skip this one rather than stop: an oversized constraint used
                # to discard every later constraint regardless of its size.
                continue
            visible.append(constraint)
            constraint_chars += len(constraint)
        merged_constraints = tuple(visible)
        visible_lookup = set(merged_constraints)
        omitted_constraints = tuple(
            constraint
            for constraint in candidate_constraints
            if constraint not in visible_lookup
        )
        dropped_constraints = len(omitted_constraints)
        restored_constraints = sum(
            constraint not in content.active_constraints
            for constraint in merged_constraints
        )
        readmitted_constraints = sum(
            row.status == "omitted" and row.text in visible_lookup
            for row in ledger
        )
        # Field pops run on every batch, including the first summary and
        # batches whose constraint merge was a no-op.
        fields = {
            "user_goals": list(content.user_goals),
            "confirmed_decisions": list(content.confirmed_decisions),
            "unresolved_questions": list(content.unresolved_questions),
        }
        dropped_fields = {
            "user_goals": 0,
            "confirmed_decisions": 0,
            "unresolved_questions": 0,
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
            dropped_fields[field] += 1
        dropped_user_goals = dropped_fields["user_goals"]
        dropped_confirmed_decisions = dropped_fields["confirmed_decisions"]
        dropped_unresolved_questions = dropped_fields["unresolved_questions"]
        # Absolute, not cumulative: archived constraints are retrievable, so
        # the count answers "how many are held back right now". The three
        # counters below still accumulate because those entries are destroyed.
        omitted_active_constraint_count = len(omitted_constraints)
        omitted_user_goal_count += dropped_user_goals
        omitted_confirmed_decision_count += dropped_confirmed_decisions
        omitted_unresolved_question_count += dropped_unresolved_questions
        content = ConversationSummaryContent(
            user_goals=tuple(fields["user_goals"]),
            confirmed_decisions=tuple(fields["confirmed_decisions"]),
            unresolved_questions=tuple(fields["unresolved_questions"]),
            active_constraints=merged_constraints,
            omitted_active_constraint_count=omitted_active_constraint_count,
            omitted_user_goal_count=omitted_user_goal_count,
            omitted_confirmed_decision_count=omitted_confirmed_decision_count,
            omitted_unresolved_question_count=omitted_unresolved_question_count,
        )
        compacted = self._store.compact_conversation_summary(
            user_id=user_id,
            conversation_id=conversation_id,
            expected_previous_through_sequence=previous_through,
            content=content,
            through_sequence=to_summarize[-1].sequence,
            omitted_constraints=omitted_constraints,
            preference_candidates=preference_candidates,
        )
        if not compacted:
            # A lost race, not a no-op: whoever won it moved the boundary, so the
            # measured context no longer describes storage.
            return None
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
                    if trigger == "occupancy" and pressure.projection_overflow
                    else trigger
                ),
                "through_sequence": to_summarize[-1].sequence,
                "occupancy": pressure.occupancy,
                "occupancy_source": pressure.source,
                "projection_overflow": pressure.projection_overflow,
                "input_occupancy_numerator": pressure.request_tokens,
                "input_occupancy_denominator": pressure.max_input_tokens,
                "restored_constraints": restored_constraints,
                "dropped_constraints": dropped_constraints,
                "readmitted_constraints": readmitted_constraints,
                "dropped_user_goals": dropped_user_goals,
                "dropped_confirmed_decisions": dropped_confirmed_decisions,
                "dropped_unresolved_questions": dropped_unresolved_questions,
                "omitted_active_constraint_count": (
                    omitted_active_constraint_count
                ),
                "omitted_user_goal_count": omitted_user_goal_count,
                "omitted_confirmed_decision_count": (
                    omitted_confirmed_decision_count
                ),
                "omitted_unresolved_question_count": (
                    omitted_unresolved_question_count
                ),
                "batch_size": len(to_summarize),
                "preference_candidates_proposed": (
                    preference_candidates_proposed
                ),
                "preference_candidates_admitted": len(
                    preference_candidates
                ),
            },
        )
        return None

    def _compaction_suspended(
        self, *, user_id: str, conversation_id: str
    ) -> bool:
        """Whether this conversation's summarizer is failing and not yet due a retry.

        After the retry interval one attempt goes through: a breaker that
        waited for a success would never make the call that could produce one.
        If that attempt fails, the interval starts again from then.
        """
        failures, suspended_at = self._compaction_failures.get(
            (user_id, conversation_id), (0, None)
        )
        if failures < self._COMPACTION_FAILURE_LIMIT or suspended_at is None:
            return False
        return self._clock() - suspended_at < self._COMPACTION_RETRY_AFTER

    def _record_compaction_failure(
        self,
        *,
        user_id: str,
        conversation_id: str,
        error: AgentWorkerError,
    ) -> None:
        key = (user_id, conversation_id)
        failures = self._compaction_failures.get(key, (0, None))[0] + 1
        suspended = failures >= self._COMPACTION_FAILURE_LIMIT
        self._compaction_failures[key] = (
            failures,
            self._clock() if suspended else None,
        )
        # The summary worker records no trace of its own, so without this an
        # outage, and the suspension it causes, would leave nothing behind.
        record_active_trace(
            "context_compaction_failed",
            "conversation_summary",
            outcome="failed",
            error_code=getattr(error, "code", type(error).__name__),
            recoverable=getattr(error, "retryable", None),
            details={
                "conversation_key": conversation_trace_key(
                    user_id, conversation_id
                ),
                "consecutive_failures": failures,
                "compaction_suspended": suspended,
            },
        )

    @staticmethod
    def _validated_preference_candidates(
        candidates: tuple[DistilledFreeTextPreferenceCandidate, ...],
        *,
        messages: tuple[SummaryMessage, ...],
    ) -> tuple[DistilledFreeTextPreferenceCandidate, ...]:
        """Keep only candidates grounded in one exact user message in this batch."""

        source_by_sequence = {message.sequence: message for message in messages}
        selected: dict[str, DistilledFreeTextPreferenceCandidate] = {}
        for candidate in candidates:
            source = source_by_sequence.get(candidate.source_sequence)
            if (
                source is None
                or source.role != "user"
                or candidate.source_quote not in source.content
            ):
                continue
            selected.setdefault(candidate.topic_key, candidate)
        return tuple(selected.values())

    def _recent_pressure(
        self,
        *,
        user_id: str,
        conversation_id: str,
        after_sequence: int,
        user_message: str,
    ) -> _Pressure:
        """Pressure at load: the complete request, built and estimated."""
        if self._request_token_estimator is None:
            return self._legacy_pressure(
                user_id=user_id,
                conversation_id=conversation_id,
                after_sequence=after_sequence,
            )
        pressure_context = self._build_context(
            user_id=user_id,
            conversation_id=conversation_id,
            user_message=user_message,
        )
        request_tokens, max_input_tokens = self._estimate_request(
            pressure_context
        )
        return _Pressure(
            occupancy=request_tokens / max_input_tokens,
            projection_overflow=self._projection_overflows(
                user_id=user_id,
                conversation_id=conversation_id,
                after_sequence=after_sequence,
            ),
            request_tokens=request_tokens,
            max_input_tokens=max_input_tokens,
            context=pressure_context,
            source="estimated",
        )

    def _commit_pressure(
        self,
        *,
        user_id: str,
        conversation_id: str,
        after_sequence: int,
        trigger: Literal["occupancy", "seam"],
        carried: tuple[int, int] | None,
        assistant_message: str,
    ) -> _Pressure:
        """Pressure at commit, without building the context again."""
        if trigger == "seam":
            # A seam compacts whatever the occupancy, so it measures nothing.
            return self._overflow_only_pressure(
                user_id=user_id,
                conversation_id=conversation_id,
                after_sequence=after_sequence,
                source="seam",
            )
        if self._request_token_estimator is None:
            return self._legacy_pressure(
                user_id=user_id,
                conversation_id=conversation_id,
                after_sequence=after_sequence,
            )
        if carried is None:
            # No load measured this turn: a workflow-owned turn, or an
            # interrupted commit that never loaded. Only overflow is decided
            # here; occupancy waits for the next load.
            return self._overflow_only_pressure(
                user_id=user_id,
                conversation_id=conversation_id,
                after_sequence=after_sequence,
                source="overflow_only",
            )
        request_tokens, max_input_tokens = carried
        # The load's estimate already holds this turn's message as the current
        # message, under a larger cap than the window gives it next turn, so the
        # reply is the only thing the next request adds.
        reply_tokens = message_token_count(assistant_message)
        if self._recent_message_tokens is not None:
            reply_tokens = min(reply_tokens, self._recent_message_tokens)
        predicted = request_tokens + reply_tokens
        return _Pressure(
            occupancy=predicted / max_input_tokens,
            projection_overflow=self._projection_overflows(
                user_id=user_id,
                conversation_id=conversation_id,
                after_sequence=after_sequence,
            ),
            request_tokens=predicted,
            max_input_tokens=max_input_tokens,
            context=None,
            source="carried",
        )

    def _overflow_only_pressure(
        self,
        *,
        user_id: str,
        conversation_id: str,
        after_sequence: int,
        source: Literal["seam", "overflow_only"],
    ) -> _Pressure:
        return _Pressure(
            occupancy=None,
            projection_overflow=self._projection_overflows(
                user_id=user_id,
                conversation_id=conversation_id,
                after_sequence=after_sequence,
            ),
            request_tokens=None,
            max_input_tokens=None,
            context=None,
            source=source,
        )

    def _legacy_pressure(
        self,
        *,
        user_id: str,
        conversation_id: str,
        after_sequence: int,
    ) -> _Pressure:
        # Maintenance callers may not own a model or a tool registry. Preserve
        # their legacy recent-window signal, but production runtime always
        # installs the complete-request estimator.
        raw_limit = self._recent_message_limit + self._summary_batch_size - 1
        candidates = self._store.list_message_records(
            user_id,
            conversation_id,
            limit=raw_limit + 1,
            after_sequence=after_sequence,
        )
        recent = self._bound_recent_messages(
            self._deduplicate_recent_messages(candidates[-raw_limit:])
        )
        used = sum(len(record.message.content) for record in recent)
        return _Pressure(
            occupancy=used / self._max_recent_context_chars,
            projection_overflow=len(candidates) > raw_limit,
            request_tokens=None,
            max_input_tokens=None,
            context=None,
            source="legacy",
        )

    def _projection_overflows(
        self, *, user_id: str, conversation_id: str, after_sequence: int
    ) -> bool:
        raw_limit = self._recent_message_limit + self._summary_batch_size - 1
        return (
            len(
                self._store.list_message_records(
                    user_id,
                    conversation_id,
                    # One extra row proves that the oldest unsummarised message
                    # would disappear from the projection before a watermark
                    # can name it.
                    limit=raw_limit + 1,
                    after_sequence=after_sequence,
                )
            )
            > raw_limit
        )

    def _estimate_request(self, context: MainAgentContext) -> tuple[int, int]:
        request_tokens, max_input_tokens = self._request_token_estimator(context)
        if request_tokens < 0 or max_input_tokens < 1:
            raise ValueError("request token estimator returned an invalid budget")
        return request_tokens, max_input_tokens

    def _bound_recent_messages(
        self, messages: tuple[StoredConversationMessage, ...]
    ) -> tuple[StoredConversationMessage, ...]:
        # Measured in tokens once the request budget has set the caps, and in
        # characters otherwise, which is the legacy behaviour unchanged.
        token_bounded = (
            self._recent_context_tokens is not None
            and self._recent_message_tokens is not None
        )
        if token_bounded:
            window_cap = self._recent_context_tokens
            message_cap = self._recent_message_tokens
            measure = message_token_count
        else:
            window_cap = self._max_recent_context_chars
            message_cap = self._max_recent_message_chars
            measure = len
        selected = []
        used = 0
        # Reserve a fair share for at least four recent messages when four are
        # available, while letting one or two messages use the old half-window
        # ceiling. This prevents two giant turns from evicting the other six
        # without needlessly clipping a genuinely short window.
        uncrowded = sum(
            min(measure(record.message.content), message_cap)
            for record in messages
        )
        if uncrowded <= window_cap:
            per_message = message_cap
        else:
            fair_message = max(1, window_cap // min(len(messages) or 1, 4))
            per_message = min(message_cap, fair_message)
        for record in reversed(messages):
            remaining = window_cap - used
            if remaining <= 0:
                break
            allowance = min(remaining, per_message)
            if token_bounded:
                content, content_clipped = clip_to_tokens(
                    record.message.content, allowance
                )
                if content_clipped and not content:
                    # Too little room for even the first character. A message
                    # that projects as nothing is not a message; end the window
                    # here and let recent_from_sequence name the first one
                    # actually shown. A stored message that was empty to begin
                    # with is not clipped and stays.
                    break
                # A clipped message is charged its whole allowance, so the
                # tokens of a split character it dropped are not handed on to
                # the next older message as a scrap of room.
                used += (
                    allowance
                    if content_clipped
                    else message_token_count(content)
                )
            else:
                content = record.message.content[:allowance]
                content_clipped = len(content) < len(record.message.content)
                used += len(content)
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
