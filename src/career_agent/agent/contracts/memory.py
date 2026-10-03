from __future__ import annotations

from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from career_agent.services.free_text_preferences import (
    PreferenceStance,
    normalize_preference_stance,
)

SUMMARY_SOURCE_MAX_CHARS = 4000
SUMMARY_ITEM_MAX_CHARS = 500
SUMMARY_TEXT_MAX_CHARS = 6000
"""Every summary field shares this allowance, so one field can starve the rest."""

ACTIVE_CONSTRAINT_MAX_ITEMS = 15
"""Visible constraint cap. Overflow is archived, not discarded."""

HARNESS_SUMMARY_COUNTER_FIELDS = frozenset(
    {
        "omitted_active_constraint_count",
        "omitted_user_goal_count",
        "omitted_confirmed_decision_count",
        "omitted_unresolved_question_count",
    }
)


class ConversationMemoryContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class SummaryMessage(ConversationMemoryContract):
    sequence: int = Field(ge=1)
    role: Literal["user", "assistant"]
    content: str = Field(max_length=SUMMARY_SOURCE_MAX_CHARS)


class DistilledFreeTextPreferenceCandidate(ConversationMemoryContract):
    """A model-proposed long-term candidate, never an authorized preference."""

    topic_key: str = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")
    statement: str = Field(min_length=1, max_length=500)
    stance: PreferenceStance
    source_sequence: int = Field(ge=1)
    source_quote: str = Field(min_length=1, max_length=500)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    ownership: Literal[
        "person_stable",
        "person_default",
        "person_situational",
        "role",
        "situational",
        "ask",
    ] = "person_default"
    scope_domain: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_.:-]{0,79}$",
    )
    valid_for_days: int = Field(default=30, ge=1, le=90)

    @field_validator("stance", mode="before")
    @classmethod
    def normalize_stance(cls, value: object) -> object:
        normalized = normalize_preference_stance(value)
        if normalized is None:
            raise ValueError(
                "stance must express positive or negative polarity"
            )
        return normalized


class ConversationSummaryContent(ConversationMemoryContract):
    user_goals: tuple[str, ...] = Field(default=(), max_length=10)
    confirmed_decisions: tuple[str, ...] = Field(default=(), max_length=20)
    unresolved_questions: tuple[str, ...] = Field(default=(), max_length=10)
    active_constraints: tuple[str, ...] = Field(
        default=(), max_length=ACTIVE_CONSTRAINT_MAX_ITEMS
    )
    omitted_active_constraint_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Constraints held in the archive rather than shown here. Unlike "
            "the three counters below, these entries are still retrievable: "
            "the count is the current archive size, not a running total of "
            "destroyed entries."
        ),
    )
    omitted_user_goal_count: int = Field(
        default=0,
        ge=0,
        description=(
            "User-goal entries omitted by harness summary-budget enforcement."
        ),
    )
    omitted_confirmed_decision_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Confirmed-decision entries omitted by harness summary-budget "
            "enforcement."
        ),
    )
    omitted_unresolved_question_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Unresolved-question entries omitted by harness summary-budget "
            "enforcement."
        ),
    )
    long_term_memory_candidates: tuple[
        DistilledFreeTextPreferenceCandidate, ...
    ] = Field(default=(), max_length=8, exclude=True)
    """Ephemeral summary-worker output; admitted atomically to quarantine."""

    @field_validator("long_term_memory_candidates", mode="before")
    @classmethod
    def discard_invalid_memory_candidates(cls, value: object) -> object:
        """Keep optional candidate failures from vetoing the required summary."""

        if value is None:
            return ()
        if not isinstance(value, (list, tuple)):
            return ()
        valid: list[DistilledFreeTextPreferenceCandidate] = []
        for candidate in value:
            try:
                valid.append(
                    DistilledFreeTextPreferenceCandidate.model_validate(candidate)
                )
            except (TypeError, ValueError):
                continue
            if len(valid) == 8:
                break
        return tuple(valid)

    @model_validator(mode="after")
    def bound_summary_text(self) -> "ConversationSummaryContent":
        values = (
            *self.user_goals,
            *self.confirmed_decisions,
            *self.unresolved_questions,
            *self.active_constraints,
        )
        if any(len(value) > SUMMARY_ITEM_MAX_CHARS for value in values):
            raise ValueError(
                "conversation summary items must be at most "
                f"{SUMMARY_ITEM_MAX_CHARS} characters"
            )
        if sum(len(value) for value in values) > SUMMARY_TEXT_MAX_CHARS:
            raise ValueError("conversation summary exceeds its context budget")
        return self


class StoredConversationSummary(ConversationMemoryContract):
    user_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    content: ConversationSummaryContent
    through_sequence: int = Field(ge=1)
    updated_at: datetime


class ConversationSummaryWorker(Protocol):
    def summarize(
        self,
        *,
        previous: ConversationSummaryContent | None,
        messages: tuple[SummaryMessage, ...],
    ) -> ConversationSummaryContent: ...
