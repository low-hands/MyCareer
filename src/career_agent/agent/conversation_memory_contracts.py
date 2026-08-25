from __future__ import annotations

from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConversationMemoryContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class SummaryMessage(ConversationMemoryContract):
    sequence: int = Field(ge=1)
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ConversationSummaryContent(ConversationMemoryContract):
    user_goals: tuple[str, ...] = Field(default=(), max_length=10)
    confirmed_decisions: tuple[str, ...] = Field(default=(), max_length=20)
    unresolved_questions: tuple[str, ...] = Field(default=(), max_length=10)
    active_constraints: tuple[str, ...] = Field(default=(), max_length=15)

    @model_validator(mode="after")
    def bound_summary_text(self) -> "ConversationSummaryContent":
        values = (
            *self.user_goals,
            *self.confirmed_decisions,
            *self.unresolved_questions,
            *self.active_constraints,
        )
        if any(len(value) > 500 for value in values):
            raise ValueError("conversation summary items must be at most 500 characters")
        if sum(len(value) for value in values) > 6000:
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
