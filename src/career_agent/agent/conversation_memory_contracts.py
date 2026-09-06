from __future__ import annotations

from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

SUMMARY_SOURCE_MAX_CHARS = 4000
SUMMARY_ITEM_MAX_CHARS = 500
SUMMARY_TEXT_MAX_CHARS = 6000
"""Every summary field shares this allowance, so one field can starve the rest."""


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


class ConversationSummaryContent(ConversationMemoryContract):
    user_goals: tuple[str, ...] = Field(default=(), max_length=10)
    confirmed_decisions: tuple[str, ...] = Field(default=(), max_length=20)
    unresolved_questions: tuple[str, ...] = Field(default=(), max_length=10)
    active_constraints: tuple[str, ...] = Field(default=(), max_length=15)
    omitted_active_constraint_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Constraint entries omitted by harness summary-budget enforcement."
        ),
    )

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
