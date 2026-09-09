from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MemoryScopeContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


# Only families whose relations come from a closed registry belong here. Free-text
# career evidence is addressed by ``career_evidence_scope_key`` instead, which
# anchors a correction lineage on the evidence id, so a claim never becomes a key.
ScopeFamily = Literal[
    "person_intent",
    "target_role_intent",
]


class ScopeProposal(MemoryScopeContract):
    """A write candidate before any durable version key is derived."""

    user_id: str = Field(min_length=1)
    family: ScopeFamily
    subject_id: str = Field(min_length=1)
    relation: str = Field(min_length=1, max_length=100)
    proposed_value: str = Field(min_length=1, max_length=2000)
    # The turn that proposed the write, absent only when the calling tool was
    # given none. It is carried because an unresolved proposal has to be able
    # to name the conversation its clarification belongs to.
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
