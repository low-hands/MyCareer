from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


EPISODE_SUMMARY_MAX_CHARS = 400
EpisodeKind = Literal[
    "mock_interview",
    "job_research",
    "application",
    "interview_round",
]


class EpisodeContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class EpisodeResourceRef(EpisodeContract):
    """Pointer to a durable domain resource; never a copy of its body."""

    kind: str = Field(min_length=1)
    resource_id: str = Field(min_length=1)
    title: str | None = Field(default=None, min_length=1, max_length=80)


class CareerEpisodeDraft(EpisodeContract):
    """Deterministic write request. Code decides identity; text is retrieval-only."""

    user_id: str = Field(min_length=1)
    kind: EpisodeKind
    source_run_id: str = Field(min_length=1)
    occurred_at: datetime
    title: str = Field(min_length=1, max_length=80)
    summary: str = Field(min_length=1, max_length=EPISODE_SUMMARY_MAX_CHARS)
    conversation_id: str | None = Field(default=None, min_length=1)
    resource_refs: tuple[EpisodeResourceRef, ...] = ()


class CareerEpisode(CareerEpisodeDraft):
    id: str = Field(min_length=1)
    salience: float = 1.0
    last_accessed_at: datetime | None = None
    access_count: int = Field(default=0, ge=0)
