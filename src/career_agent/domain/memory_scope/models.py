from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MemoryScopeContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


ScopeFamily = Literal[
    "person_intent",
    "target_role_intent",
    "career_evidence",
]
ScopeResolutionStatus = Literal[
    "unresolved",
    "clarification_requested",
    "resolved",
    "dismissed",
]
ScopeSourceKind = Literal[
    "job_intent",
    "career_evidence",
    "episode_consolidation",
]


class ScopeProposal(MemoryScopeContract):
    """A write candidate before any durable version key is derived."""

    user_id: str = Field(min_length=1)
    family: ScopeFamily
    subject_id: str = Field(min_length=1)
    relation: str = Field(min_length=1, max_length=100)
    proposed_value: str = Field(min_length=1, max_length=2000)
    source_kind: ScopeSourceKind
    source_id: str = Field(min_length=1)
    # The turn that proposed the write. Optional because a future consolidation
    # pass has no conversation, but the observation of a write is scoped to one
    # conversation like every other turn event, and an unresolved proposal has
    # to be able to name the conversation its clarification belongs to.
    conversation_id: str | None = Field(default=None, min_length=1)


class CanonicalScope(MemoryScopeContract):
    """A resolver-issued identity; callers may not manufacture ``scope_key``."""

    family: ScopeFamily
    subject_id: str = Field(min_length=1)
    relation: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    scope_key: str = Field(pattern=r"^[a-z_]+/[A-Za-z0-9_.:-]+/[a-z][a-z0-9_]*$")

    @model_validator(mode="after")
    def key_matches_parts(self) -> "CanonicalScope":
        expected = f"{self.family}/{self.subject_id}/{self.relation}"
        if self.scope_key != expected:
            raise ValueError("scope_key must be derived from the canonical parts")
        return self


class ScopeResolution(MemoryScopeContract):
    proposal: ScopeProposal
    canonical_scope: CanonicalScope | None = None
    reason: str | None = Field(default=None, min_length=1, max_length=500)
    candidate_scope_keys: tuple[str, ...] = ()

    @model_validator(mode="after")
    def outcome_is_unambiguous(self) -> "ScopeResolution":
        if (self.canonical_scope is None) == (self.reason is None):
            raise ValueError(
                "resolved outcomes need a canonical scope and unresolved outcomes need a reason"
            )
        return self

    @property
    def resolved(self) -> bool:
        return self.canonical_scope is not None


class ScopeResolutionQueueItem(MemoryScopeContract):
    id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    conversation_id: str | None = Field(default=None, min_length=1)
    family: ScopeFamily
    subject_id: str = Field(min_length=1)
    relation: str = Field(min_length=1)
    proposed_value: str = Field(min_length=1)
    source_kind: ScopeSourceKind
    source_id: str = Field(min_length=1)
    status: ScopeResolutionStatus
    reason: str = Field(min_length=1)
    candidate_scope_keys: tuple[str, ...] = ()
    resolved_scope_key: str | None = Field(default=None, min_length=1)
    content_digest: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    idempotency_key: str = Field(pattern=r"^[a-f0-9]{64}$")
    clarification_attempts: int = Field(default=0, ge=0)
    created_at: datetime
    updated_at: datetime
    clarification_requested_at: datetime | None = None
    resolved_at: datetime | None = None

    @model_validator(mode="after")
    def terminal_fields_match_status(self) -> "ScopeResolutionQueueItem":
        if self.status == "resolved":
            if self.resolved_scope_key is None or self.resolved_at is None:
                raise ValueError("resolved queue items require a scope key and timestamp")
        elif self.resolved_scope_key is not None:
            raise ValueError("only resolved queue items may carry a resolved scope key")
        if self.status == "dismissed" and self.resolved_at is None:
            raise ValueError("dismissed queue items require a terminal timestamp")
        if (
            self.status == "clarification_requested"
            and self.clarification_requested_at is None
        ):
            raise ValueError(
                "clarification_requested items require a request timestamp"
            )
        return self


class ScopeResolutionQueueEvent(MemoryScopeContract):
    id: str = Field(min_length=1)
    queue_item_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    event_type: Literal[
        "enqueued",
        "clarification_requested",
        "resolved",
        "dismissed",
    ]
    previous_status: ScopeResolutionStatus | None = None
    new_status: ScopeResolutionStatus
    reason: str | None = Field(default=None, min_length=1)
    occurred_at: datetime

    @model_validator(mode="after")
    def transition_is_valid(self) -> "ScopeResolutionQueueEvent":
        expected = {
            "enqueued": (None, "unresolved"),
            "clarification_requested": (
                ("unresolved", "clarification_requested"),
                "clarification_requested",
            ),
            "resolved": (("unresolved", "clarification_requested"), "resolved"),
            "dismissed": (
                ("unresolved", "clarification_requested"),
                "dismissed",
            ),
        }
        allowed_previous, expected_new = expected[self.event_type]
        if self.new_status != expected_new:
            raise ValueError("queue event has the wrong new status")
        if isinstance(allowed_previous, tuple):
            if self.previous_status not in allowed_previous:
                raise ValueError("queue event has an invalid previous status")
        elif self.previous_status != allowed_previous:
            raise ValueError("queue event has an invalid previous status")
        return self
