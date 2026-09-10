from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


IntentLayer = Literal["stable", "contextual", "transient"]
IntentTimescale = Literal["permanent", "situational"]
IntentAdmissionStatus = Literal["active", "quarantined"]
IntentCaptureAction = Literal[
    "retain",
    "add",
    "narrow-to-scope",
    "revise",
    "quarantine",
    "ask",
]


class IntentMemoryVersion(BaseModel):
    """One immutable value in a canonical mutable-intent history."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    update_id: str = Field(pattern=r"^intent_update_[a-f0-9]{32}$")
    user_id: str = Field(min_length=1)
    scope_key: str = Field(
        pattern=r"^[a-z_]+/[A-Za-z0-9_.:-]+/[a-z][a-z0-9_]*$"
    )
    value: str = Field(min_length=1, max_length=2000)
    content_digest: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    revision: int = Field(ge=1)
    valid_from: datetime
    valid_until: datetime | None = None
    pref_scope: str = Field(
        default="global",
        min_length=1,
        max_length=120,
        pattern=r"^(?:global|[a-z][a-z0-9_.:-]*)$",
    )
    timescale: IntentTimescale = "permanent"
    layer: IntentLayer = "stable"
    last_corroborated_at: datetime
    base_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    admission_status: IntentAdmissionStatus = "active"
    capture_action: IntentCaptureAction = "add"
    semantic_stance: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_.:-]{0,79}$",
    )
    superseded_at: datetime | None = None
    superseded_by: str | None = Field(
        default=None,
        pattern=r"^intent_update_[a-f0-9]{32}$",
    )
    source: str = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def supersession_fields_move_together(self) -> "IntentMemoryVersion":
        if (self.superseded_at is None) != (self.superseded_by is None):
            raise ValueError(
                "superseded_at and superseded_by must either both be set or both be null"
            )
        if self.timescale == "situational":
            if self.valid_until is None:
                raise ValueError("situational intent requires valid_until")
            if self.valid_until <= self.valid_from:
                raise ValueError("valid_until must be after valid_from")
        elif self.valid_until is not None:
            raise ValueError("permanent intent cannot carry valid_until")
        return self
