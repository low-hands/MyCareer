from __future__ import annotations

from collections.abc import Mapping
import hashlib
import hmac
from datetime import datetime, timedelta, timezone
import json
import re
from typing import Annotated, Any, Literal, NamedTuple, Protocol, get_args

from pydantic import AliasChoices, Field, field_validator, model_validator

from career_agent.agent.support.summary_text import DELIVERY_SUMMARY_LIMIT, clamp
from career_agent.agent.presentation.delivery_policy import is_failed, is_waiting
from career_agent.agent.presentation.body_contracts import (
    BodyDependency,
    DeliveredBodySource,
)
from career_agent.agent.contracts.memory import (
    SUMMARY_ITEM_MAX_CHARS,
    SUMMARY_SOURCE_MAX_CHARS,
    ConversationSummaryContent,
)
from career_agent.agent.providers.token_budget import serialized_token_count
from career_agent.agent.capabilities.effects import approval_policy, owner_rule_capabilities
from career_agent.agent.contracts.questionnaire import PendingQuestionnaire, UserQuestion
from career_agent.domain.applications import ApplicationStatus
from career_agent.domain.action_center import ActionSourceType, ActionStatus, ActionType
from career_agent.domain.email_tracking import EmailEventStatus
from career_agent.domain.interviews import (
    InterviewDetails,
    InterviewRetroQuestion,
    InterviewSelfAssessment,
    InterviewStatus,
)
from career_agent.domain.job_discovery import ContractModel
from career_agent.domain.job_research.models import company_key
from career_agent.domain.mock_interviews import MockInterviewType

from career_agent.agent.contracts.profile import *
from career_agent.agent.contracts.candidates import *
from career_agent.agent.contracts.task_state import *
from career_agent.agent.contracts.resources import *
from career_agent.agent.contracts.resources import _memory_overflow_notice
from career_agent.agent.contracts.observations import *
from career_agent.agent.contracts.observations import (
    _HANDLE_PREFIXES, _HANDLE_SUFFIX_LENGTH, _PREFERENCE_CHAR_CAP, _bounded_markdown,
)

def compressed_history_available(
    through_sequence: int | None, recent_from_sequence: int | None,
) -> bool:
    """Summary coverage or an omitted recent prefix permits readback.

    A missing runtime watermark retains legacy direct-registry behavior.
    """
    return not (through_sequence == 0 and recent_from_sequence in (None, 1))


class MainAgentContext(ContractModel):
    conversation_id: str
    capability_selection: Any = Field(default=None, exclude=True)
    received_at: datetime | None = Field(default=None, exclude=True)
    """When this turn's user message arrived; harness-only, never projected.

    A workflow's opening request is written only when the run ends, and the
    transcript places it by time, so it needs the moment it was sent rather
    than the moment the turn that started the run finished."""
    spotlight_nonce: str | None = Field(default=None, min_length=32, max_length=32)
    """Harness-only session delimiter; deliberately absent from model_context JSON."""
    profile: CareerProfileContext
    preferences: AgentPreferencesContext = AgentPreferencesContext()
    task: ConversationTaskState = ConversationTaskState()
    career_memory: CareerMemoryContext = CareerMemoryContext()
    free_text_preferences: tuple[FreeTextPreferenceContext, ...] = Field(
        default=(),
        max_length=8,
    )
    free_text_preferences_active_total: int = Field(default=0, ge=0, exclude=True)
    """How many confirmed preferences the projection selected from.

    Carried for the same reason as ``archived_resource_total``: the eight-slot
    cap cannot say it was applied, so a full list reads as everything the user
    has ever confirmed. Rendered as a count of what is missing, not a flag.
    """

    free_text_preferences_quarantined_total: int = Field(
        default=0, ge=0, exclude=True
    )
    """How many quarantined candidates the projection selected from.

    Separate from the active total because the two are rendered in different
    sections under different authority, and because a candidate held back is
    the more consequential omission: it is waiting to be confirmed, and a silent
    cut leaves it waiting indefinitely.
    """

    working_notes: WorkingNotesContext | None = None
    career_episodes: tuple[EpisodeProjectionContext, ...] = Field(
        default=(),
        max_length=5,
    )
    career_profile_budgets: CareerProfileBudgets = Field(
        default_factory=CareerProfileBudgets,
        exclude=True,
    )
    recent_messages: tuple[ConversationMessageContext, ...] = ()
    through_sequence: int = Field(default=0, ge=0)
    """Last durable message covered by ``conversation_summary``; zero if absent."""

    recent_from_sequence: int | None = Field(default=None, ge=1)
    """Sequence of the first raw message projected into the recent window."""

    @property
    def has_compressed_history(self) -> bool:
        return compressed_history_available(self.through_sequence, self.recent_from_sequence)

    archived_resource_total: int = Field(default=0, ge=0)
    """How many resources the catalogue would list uncapped.

    Zero when there is no catalogue. Sent because a capped list cannot say it
    was capped: twelve entries and nothing else read as the complete set, and a
    report that scrolled past the cap then looks like it must be one of the
    twelve. Measured behaviour is that the model reaches for the nearest
    plausible entry when it believes the thing it wants is on screen, so the
    fix is to stop the projection from implying that.

    A count rather than a flag, on the same reasoning that made ``next_action``
    prose: "there are more" leaves the model to guess how many it cannot see,
    while 12 of 27 says how far short the list falls.
    """

    archived_resources: tuple[ConversationMessageContext, ...] = Field(
        default=(), max_length=12
    )
    """Messages delivered before the recent window, as a reachable catalogue.

    Bounded by messages rather than references: one message can carry several,
    because one turn can store several reports.

    Without these a report becomes unreachable to the agent the moment its
    turn is summarised away: ``resource_refs`` live only on the original
    message, and the summary carries none. The UI kept working — it reads
    the whole transcript — so the failure was asymmetric and silent, with
    the user looking at a card the model could no longer open.

    Only the reference and bounded producer-owned metadata cross. The report itself stays
    in its entity, which is the whole point of storing a pointer.
    """

    tool_observations: tuple[DecisionObservation, ...] = Field(
        default=(),
        max_length=MAX_DECISION_OBSERVATIONS,
    )
    turn_proactive_capabilities: tuple[str, ...] = ()
    """Per-turn W successors survive observation trimming and graph checkpointing.

    This runtime field is never included in ``model_context()``; only the
    selected ``available_now`` names are shown to the model.
    """
    turn_continuation_capability: str | None = None
    conversation_summary: ConversationSummaryContent | None = None
    attached_resumes: tuple[AttachedResumeContext, ...] = Field(
        default=(), max_length=8
    )
    """Exact resume versions the current user message is about, already verified."""

    attached_jobs: tuple[SavedJobCandidateContextItem, ...] = Field(
        default=(), max_length=8,
    )

    user_message: str = Field(min_length=1)
    user_interaction_id: str | None = Field(
        default=None, pattern=r"^interaction_[a-f0-9]{20}$", exclude=True
    )
    user_message_source: str | None = Field(default=None, exclude=True)
    """The message as the user sent it, kept only when ``user_message`` was clipped."""

    user_message_clipped: bool = Field(default=False, exclude=True)
    """Whether ``user_message`` is shorter than what the user sent.

    Rendered on the prompt copy the same way a clipped recent-window message
    is, so a cut-off request never reads as the whole of it.
    """

    def stored_user_message(self) -> str:
        """The message to persist or reload from, never the prompt's clipped copy."""
        return (
            self.user_message_source
            if self.user_message_source is not None
            else self.user_message
        )

    def user_input_resource_refs(self) -> tuple[ConversationResourceReference, ...]:
        """The references the stored user message carries for its attachments.

        Only the id and a display snapshot: the excerpt is turn-local and the
        file stays in the resume library.
        """
        return tuple(
            ConversationResourceReference(
                kind="resume_version",
                resource_id=item.resume_version_id,
                title=clamp(
                    f"{item.resume_name} v{item.version_number}", limit=80
                ),
                description=clamp(
                    f"{item.document_format} · {item.byte_size} bytes · "
                    f"上传于 {item.uploaded_at.isoformat()}",
                    limit=200,
                ),
            )
            for item in self.attached_resumes
        ) + tuple(
            ConversationResourceReference(
                kind="saved_job",
                resource_id=item.jd_snapshot_id,
                job_posting_id=item.job_posting_id,
                title=clamp(f"{item.company_name}｜{item.title}", limit=80),
                description=f"JD 第 {item.jd_version} 版",
            )
            for item in self.attached_jobs
            if item.jd_snapshot_id is not None
        )

    @model_validator(mode="before")
    @classmethod
    def populate_free_text_preference_totals(cls, value: object) -> object:
        """Default each total to the count projected, as the uncapped case.

        A caller that never hit the cap should not have to say so twice, and a
        context assembled without the totals would otherwise claim a truncation
        it does not have. ``ContextManager`` passes them explicitly, which is the
        only place they can exceed what is projected.
        """
        if not isinstance(value, dict):
            return value
        projected = value.get("free_text_preferences", ())
        try:
            statuses = [
                item.status
                if hasattr(item, "status")
                else item.get("status")
                for item in projected
            ]
        except (AttributeError, TypeError):
            return value
        filled = dict(value)
        if "free_text_preferences_active_total" not in filled:
            filled["free_text_preferences_active_total"] = sum(
                status == "active" for status in statuses
            )
        if "free_text_preferences_quarantined_total" not in filled:
            filled["free_text_preferences_quarantined_total"] = sum(
                status == "quarantined" for status in statuses
            )
        return filled

    @model_validator(mode="after")
    def observation_bodies_are_only_on_the_newest_item(self) -> "MainAgentContext":
        if self.recent_from_sequence is not None and (
            self.recent_from_sequence <= self.through_sequence
        ):
            raise ValueError("recent messages must begin after the summary boundary")
        stale = self.tool_observations[:-MAX_DECISION_OBSERVATION_BODIES]
        if any(item.body is not None for item in stale):
            raise ValueError(
                "only the newest decision observation may retain a body"
            )
        if decision_observation_chars(self.tool_observations) > (
            MAX_DECISION_OBSERVATION_CHARS
        ):
            raise ValueError("decision observations exceed the character budget")
        if self.free_text_preferences_active_total < sum(
            item.status == "active" for item in self.free_text_preferences
        ):
            raise ValueError(
                "free_text_preferences_active_total cannot be smaller than "
                "projected confirmed preferences"
            )
        if self.free_text_preferences_quarantined_total < sum(
            item.status == "quarantined" for item in self.free_text_preferences
        ):
            raise ValueError(
                "free_text_preferences_quarantined_total cannot be smaller "
                "than projected quarantined preferences"
            )
        return self

    def referenced_resources(self) -> tuple[ConversationResourceReference, ...]:
        """Every resource the model can name this turn.

        Archived catalogue, then the recent window, then what this turn produced
        and no stored line names yet. The order is no longer load-bearing — a
        handle is derived from the resource, not from where it sits — but it is
        kept because it reads the way the conversation happened.

        A resource already named is skipped rather than listed twice: reading an
        old report back produces a reference to something the catalogue already
        holds, and its handle would be identical anyway.
        """
        references: list[ConversationResourceReference] = []
        seen: set[str] = set()
        for reference in (
            *(
                reference
                for message in (*self.archived_resources, *self.recent_messages)
                for reference in message.resource_refs
            ),
            *self.user_input_resource_refs(),
            *(
                observation.resource_ref
                for observation in self.tool_observations
                if observation.resource_ref is not None
            ),
            *(
                reference
                for observation in self.tool_observations
                for reference in observation.resource_refs
            ),
        ):
            if reference.resource_id in seen:
                continue
            seen.add(reference.resource_id)
            references.append(reference)
        return tuple(references)

    def reference_handle(self, reference: ConversationResourceReference) -> str:
        """The name the model may pass back for one resource.

        ``<kind prefix>_<derived suffix>``, following the shape every published
        tool API uses for the same job — ``file_abc123``, ``toolu_01A...``, an
        MCP URI. Two of that shape's three properties are adopted and one is
        not:

        * **Unguessable.** The whole reason for the migration. An ordinal is
          guessable by construction, and a model that has never been given one
          will still write ``1``: recorded behaviour, not a worry. It then
          resolves — to whatever sits first — because the number is real and its
          kind matches. Nothing in a positional scheme can tell "the number I
          was given" from "the number I counted to".
        * **Prefixed by kind.** So a mistake reads as "that is a report handle
          and you used it where a preparation was wanted" instead of "not
          found", and so the model need not pair a handle with a separate
          ``kind`` field to know what it holds.
        * **Not the primary key.** Published APIs expose the id itself; we do
          not, because internal identifiers stay out of the projection. The
          handle is derived one-way from the resource id instead, which also
          means no stored row changes and nothing has to be migrated.

        Derived rather than random so the same report keeps the same handle
        across turns and restarts, with no new column to store it in. The
        conversation id is the salt, which makes handles differ between
        conversations; that is scope hygiene, not a security boundary — the
        property being bought is that a handle cannot be reached by counting,
        not that an adversary holding the transcript could not recompute one.
        """
        digest = hmac.new(
            self.conversation_id.encode("utf-8"),
            reference.resource_id.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"{_HANDLE_PREFIXES[reference.kind]}_{digest[:_HANDLE_SUFFIX_LENGTH]}"

    def reference_handles(self) -> dict[str, str]:
        """Handle → resource id, for everything nameable this turn.

        Built once and read by both the projection and the resolver, so what the
        model is shown and what comes back cannot be two different derivations.

        A collision is resolved by lengthening, not by falling back to order:
        the point of the scheme is that a handle carries no positional meaning.
        Two six-hex-digit suffixes colliding inside one conversation's handful
        of resources is remote, but it would silently hand back the wrong report
        — the exact failure being migrated away from — so it is detected here
        rather than assumed away.
        """
        handles: dict[str, str] = {}
        for reference in self.referenced_resources():
            handle = self.reference_handle(reference)
            length = _HANDLE_SUFFIX_LENGTH
            while handle in handles and handles[handle] != reference.resource_id:
                length += 2
                handle = self._lengthened_handle(reference, length)
            handles[handle] = reference.resource_id
        return handles

    def _lengthened_handle(
        self, reference: ConversationResourceReference, length: int
    ) -> str:
        digest = hmac.new(
            self.conversation_id.encode("utf-8"),
            reference.resource_id.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"{_HANDLE_PREFIXES[reference.kind]}_{digest[:length]}"

    def resolve_reference(self, *, reference: str, kind: str) -> str:
        """Turn a handle the model wrote back into the internal resource id.

        The kind is still checked rather than trusted. The prefix tells the model
        what it is holding, but it is a hint in text the model itself produced;
        the server has to verify against what it actually handed out.
        """
        resource_id = self.reference_handles().get(reference)
        if resource_id is None:
            raise ValueError(f"unknown resource reference '{reference}'")
        held = next(
            item
            for item in self.referenced_resources()
            if item.resource_id == resource_id
        )
        if held.kind != kind:
            raise ValueError(
                f"resource reference '{reference}' is a {held.kind}, not a {kind}"
            )
        return resource_id

    def model_context(self) -> dict[str, Any]:
        model_messages = []
        # One derivation, read here and by the resolver. Nothing depends on the
        # order of this walk any more: a handle names its resource, so the
        # catalogue and the window cannot disagree about what a name means.
        handles = {
            resource_id: handle
            for handle, resource_id in self.reference_handles().items()
        }
        archived = [
            {
                "kind": reference.kind,
                "reference": handles[reference.resource_id],
                "delivered_at": message.created_at.isoformat(),
                **({"title": reference.title} if reference.title else {}),
                **(
                    {"description": reference.description}
                    if reference.description
                    else {}
                ),
            }
            for message in self.archived_resources
            for reference in message.resource_refs
        ]
        unlisted_count = max(0, self.archived_resource_total - len(archived))
        for index, message in enumerate(self.recent_messages):
            projected = {
                "role": message.role,
                "content": message.content,
                "created_at": message.created_at.isoformat(),
            }
            if self.recent_from_sequence is not None:
                projected["sequence"] = self.recent_from_sequence + index
            if message.resource_refs:
                resources = [
                    {
                        "kind": reference.kind,
                        "reference": handles[reference.resource_id],
                        **({"title": reference.title} if reference.title else {}),
                        **(
                            {"description": reference.description}
                            if reference.description
                            else {}
                        ),
                    }
                    for reference in message.resource_refs
                ]
                # Plural because one turn can store two reports. The key stays
                # ``resources`` in both cases so the model reads one shape.
                projected["resources"] = resources
            model_messages.append(projected)
        if self.profile.current_targets_total != len(self.profile.current_targets):
            raise ValueError(
                "career profile projection requires the complete current-target set"
            )
        career_memory = self.career_memory.tier_one_projection(
            token_budget=self.career_profile_budgets.records_input_units,
        )
        career_memory.update(_memory_overflow_notice(career_memory))
        active_free_text_preferences = [
            item
            for item in self.free_text_preferences
            if item.status == "active" and item.confirmed_at is not None
        ]
        quarantined_free_text_preferences = [
            item
            for item in self.free_text_preferences
            if item.status == "quarantined"
        ][:3]
        preference_lines = ["## 已确认的自由文本偏好（可用于推荐）"]
        preference_lines.extend(
            f"- {item.statement}（确认于 {item.confirmed_at.isoformat()}）"
            for item in active_free_text_preferences
            if item.confirmed_at is not None
        )
        if not active_free_text_preferences:
            preference_lines.append("- 无")
        # Only when something was actually held back. A line that is always
        # present would be a constant, and a constant tells the model nothing
        # about this turn.
        hidden_active = max(
            0,
            self.free_text_preferences_active_total
            - len(active_free_text_preferences),
        )
        if hidden_active:
            preference_lines.append(
                f"- （另有 {hidden_active} 条已确认偏好未列出）"
            )
        preference_lines.append("## 待确认偏好（隔离态，不得用于筛选、排序或推荐）")
        preference_lines.extend(
            f"{index}. {item.statement}"
            for index, item in enumerate(
                quarantined_free_text_preferences,
                start=1,
            )
        )
        if not quarantined_free_text_preferences:
            preference_lines.append("- 无")
        hidden_quarantined = max(
            0,
            self.free_text_preferences_quarantined_total
            - len(quarantined_free_text_preferences),
        )
        if hidden_quarantined:
            preference_lines.append(
                f"- （另有 {hidden_quarantined} 条待确认偏好未列出）"
            )
        preference_markdown = _bounded_markdown(
            preference_lines,
            budget=_PREFERENCE_CHAR_CAP,
            line_limit=220,
        )
        episode_budget = PREFERENCE_EPISODE_CHAR_BUDGET - len(
            preference_markdown
        )
        episode_lines = [
            "## 相关的过往求职事件（渐进披露目录）",
            "这里只是摘要；需要细节时调用 search_career_episodes，并传入 detail_ref。",
        ]
        episode_lines.extend(
            f"- [{item.kind}] {item.title}：{item.synopsis} "
            f"[detail_ref={item.detail_ref}]"
            for item in self.career_episodes
        )
        bounded_episode_markdown = (
            _bounded_markdown(
                episode_lines,
                budget=episode_budget,
                line_limit=220,
            )
            if self.career_episodes
            else ""
        )
        # A catalogue with no entry is two headings and, now that bounding says
        # what it dropped, possibly a count. Checking for an actual entry rather
        # than a line total keeps that count from passing as content.
        episode_markdown = (
            bounded_episode_markdown
            if "detail_ref=" in bounded_episode_markdown
            else ""
        )
        return {
            "career_profile": career_profile_memory_files(self.profile),
            "career_memory": career_memory,
            "free_text_preferences": preference_markdown,
            **(
                {
                    "working_notes": {
                        "revision": self.working_notes.revision,
                        "markdown": self.working_notes.markdown,
                        **({"clipped": True} if self.working_notes.clipped else {}),
                        **(
                            {"stale_days": self.working_notes.stale_days}
                            if self.working_notes.stale_days is not None
                            else {}
                        ),
                    }
                }
                if self.working_notes is not None
                else {}
            ),
            **(
                {"career_episodes": episode_markdown}
                if episode_markdown
                else {}
            ),
            "preferences": {
                "boss_search": self.preferences.boss_search,
            },
            **(
                {
                    "behavior_policy": {
                        **(
                            {
                                "application_confirmation": (
                                    self.preferences.application_confirmation
                                )
                            }
                            if self.preferences.application_confirmation
                            != "on_user_report"
                            else {}
                        ),
                        **(
                            {
                                "confirm_before": list(
                                    self.preferences.behavior_policy.confirm_before
                                )
                            }
                            if self.preferences.behavior_policy.confirm_before
                            else {}
                        ),
                    }
                }
                if self.preferences.application_confirmation != "on_user_report"
                or self.preferences.behavior_policy.confirm_before
                else {}
            ),
            "task": {
                **self.task.active_resource_flags(),
                "active_calendar_proposal_expires_at": (
                    self.task.active_calendar_proposal_expires_at.isoformat()
                    if self.task.active_calendar_proposal_expires_at is not None
                    else None
                ),
                "active_workflow": self.task.active_workflow,
                "active_saved_job": (
                    {
                        "title": focus.title,
                        "company_name": focus.company_name,
                        "jd_version": focus.jd_version,
                        "resource_status": (
                            "readable" if focus.readable else "unavailable"
                        ),
                    }
                    if (focus := self.task.focused_saved_job()) is not None
                    else None
                ),
                **(
                    self.capability_selection.tool_projection
                    if self.capability_selection is not None else {}
                ),
                "phase": self.task.phase,
                "email_sync_phase": self.task.email_sync_phase,
                "manual_search_query": self.task.manual_search_query,
                "candidates": [
                    {
                        "selection_index": index,
                        "title": candidate.title,
                        "company_name": candidate.company_name,
                        "city": candidate.city,
                        "salary": candidate.salary,
                    }
                    for index, candidate in enumerate(self.task.candidates, start=1)
                ],
                "resume_job_match_status": self.task.resume_job_match_status,
                "job_analysis_status": self.task.job_analysis_status,
                "resume_tailoring_status": self.task.resume_tailoring_status,
                "active_application_status": self.task.active_application_status,
                "application_candidates": [
                    {
                        "selection_index": index,
                        "title": candidate.title,
                        "company_name": candidate.company_name,
                        "status": candidate.status,
                    }
                    for index, candidate in enumerate(
                        self.task.application_candidates, start=1
                    )
                ],
                "interview_candidates": [
                    {
                        "selection_index": index,
                        "sequence_number": candidate.sequence_number,
                        "employer_label": candidate.employer_label,
                        "status": candidate.status,
                        "scheduled_start": (
                            candidate.scheduled_start.isoformat()
                            if candidate.scheduled_start is not None
                            else None
                        ),
                    }
                    for index, candidate in enumerate(
                        self.task.interview_candidates, start=1
                    )
                ],
                "interview_preparation_ready": (
                    self.task.active_interview_preparation_id is not None
                ),
                "job_research_status": self.task.job_research_status,
                "action_candidates": [
                    {
                        "selection_index": index,
                        "action_type": candidate.action_type,
                        "source_type": candidate.source_type,
                        "title": candidate.title,
                        "status": candidate.status,
                        "due_at": (
                            candidate.due_at.isoformat()
                            if candidate.due_at is not None
                            else None
                        ),
                    }
                    for index, candidate in enumerate(
                        self.task.action_candidates, start=1
                    )
                ],
                "calendar_accounts": [
                    {
                        "selection_index": index,
                        "provider": candidate.provider,
                        "email_address": candidate.email_address,
                        "calendar_id": candidate.calendar_id,
                    }
                    for index, candidate in enumerate(
                        self.task.calendar_account_candidates, start=1
                    )
                ],
                "saved_jobs": [
                    {
                        "selection_index": index,
                        "title": candidate.title,
                        "company_name": candidate.company_name,
                        "city": candidate.city,
                        "salary": candidate.salary,
                    }
                    for index, candidate in enumerate(
                        self.task.saved_job_candidates, start=1
                    )
                ],
                "target_roles": [
                    {
                        "selection_index": index,
                        "title": candidate.title,
                        "priority": candidate.priority,
                        "status": candidate.status,
                    }
                    for index, candidate in enumerate(
                        self.task.target_role_candidates, start=1
                    )
                ],
                "resumes": [
                    {
                        "selection_index": index,
                        "name": candidate.name,
                        "status": candidate.status,
                    }
                    for index, candidate in enumerate(
                        self.task.resume_candidates, start=1
                    )
                ],
                "resume_versions": [
                    {
                        "selection_index": index,
                        **(
                            {"resume_name": candidate.resume_name}
                            if candidate.resume_name is not None
                            else {}
                        ),
                        "version_number": candidate.version_number,
                        "source_type": candidate.source_type,
                        "document_format": candidate.document_format,
                        "byte_size": candidate.byte_size,
                    }
                    for index, candidate in enumerate(
                        self.task.resume_version_candidates, start=1
                    )
                ],
                "email_events": [
                    {
                        "selection_index": index,
                        "event_type": candidate.event_type,
                        "status": candidate.status,
                        "summary": candidate.summary,
                    }
                    for index, candidate in enumerate(
                        self.task.email_event_candidates, start=1
                    )
                ],
            },
            # Internal resource IDs stay in durable messages for the UI and
            # projection layer. The decision model receives only turn-local
            # indexes, matching every other selectable object contract.
            "archived_reports": {
                "items": tuple(archived),
                "unlisted": (
                    f"另有 {unlisted_count} 份更早的调研未列出，无法按引用取回。"
                    if unlisted_count
                    else None
                ),
            },
            "recent_messages": tuple(model_messages),
            **({
                "omitted_history": {
                    "from_sequence": 1,
                    "through_sequence": self.recent_from_sequence - 1,
                    "summary_through_sequence": self.through_sequence,
                    "readback_tool": "read_conversation_span",
                }
            } if self.recent_from_sequence is not None
                 and self.recent_from_sequence > 1 else {}),
            **(
                {
                    "through_sequence": self.through_sequence,
                    "recent_from_sequence": self.recent_from_sequence,
                }
                if self.has_compressed_history
                else {}
            ),
            "tool_observations": decision_observation_projection(
                self.tool_observations,
                {
                    resource_id: handle
                    for handle, resource_id in self.reference_handles().items()
                }
            ),
            "conversation_summary": (
                {
                    **self.conversation_summary.model_dump(mode="json"),
                    "coverage": {
                        "from_sequence": 1,
                        "through_sequence": self.through_sequence,
                        "retains": "key facts with source sequences, user goals, confirmed decisions, unresolved questions, and active constraints",
                        "omits": "verbatim messages and facts not selected for these bounded summary categories",
                        "readback_tool": "read_conversation_span",
                    },
                }
                if self.conversation_summary
                else None
            ),
            **(
                {
                    "attached_resumes": tuple(
                        {
                            "reference": handles[item.resume_version_id],
                            **item.model_dump(mode="json"),
                        }
                        for item in self.attached_resumes
                    )
                }
                if self.attached_resumes
                else {}
            ),
            "user_message": self.user_message,
        }

