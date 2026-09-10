from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from career_agent.domain.intent_memory import (
    IntentCaptureAction,
    IntentLayer,
    IntentMemoryVersion,
    IntentTimescale,
)
from career_agent.storage.intent_versions import intent_content_digest


class IntentCaptureCandidate(BaseModel):
    """A proposer output; deterministic policy still decides whether to write it."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    user_id: str = Field(min_length=1)
    scope_key: str = Field(
        pattern=r"^[a-z_]+/[A-Za-z0-9_.:-]+/[a-z][a-z0-9_]*$"
    )
    value: str = Field(min_length=1, max_length=2000)
    source: str = Field(min_length=1, max_length=200)
    pref_scope: str = Field(
        default="global",
        pattern=r"^(?:global|[a-z][a-z0-9_.:-]*)$",
        max_length=120,
    )
    timescale: IntentTimescale = "permanent"
    layer: IntentLayer = "stable"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    contains_preference_signal: bool = True
    ambiguous: bool = False
    suspicious: bool = False
    semantic_stance: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_.:-]{0,79}$",
    )
    observed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def content_digest(self) -> str:
        return intent_content_digest(self.value)


class IntentCaptureDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: IntentCaptureAction
    reason: Literal[
        "abstained",
        "corroborated",
        "new_preference",
        "scoped_preference",
        "ordinary_revision",
        "ambiguous_or_suspicious",
    ]


def select_intent_capture_action(
    candidate: IntentCaptureCandidate,
    *,
    active: IntentMemoryVersion | None,
) -> IntentCaptureDecision:
    """Choose one CAPTURE write action without delegating the action to a model.

    Quarantine admission depends on a proposer that can flag its own output as
    ambiguous or suspicious. Every current caller writes intent the user has
    already confirmed, so that branch stays unreachable until an inferring
    proposer exists; it is not a substitute for gating confirmed input.
    """

    if not candidate.contains_preference_signal:
        return IntentCaptureDecision(action="retain", reason="abstained")
    if candidate.ambiguous or candidate.suspicious:
        return IntentCaptureDecision(
            action="quarantine",
            reason="ambiguous_or_suspicious",
        )
    if active is not None and active.content_digest == candidate.content_digest:
        return IntentCaptureDecision(action="retain", reason="corroborated")
    if active is None:
        return IntentCaptureDecision(
            action=(
                "narrow-to-scope"
                if candidate.pref_scope != "global"
                else "add"
            ),
            reason=(
                "scoped_preference"
                if candidate.pref_scope != "global"
                else "new_preference"
            ),
        )
    return IntentCaptureDecision(action="revise", reason="ordinary_revision")
