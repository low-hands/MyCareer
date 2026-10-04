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
from career_agent.agent.capabilities.catalog import (
    DOMAIN_TOOL_PROFILES,
    TOOL_PROFILE_NAMES,
    ToolProfile,
)
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

from career_agent.agent.contracts.candidates import *

class OpenJobSearchToolArguments(ContractModel):
    platform: Literal["boss"] = "boss"
    keyword: str = Field(min_length=1, max_length=100)
    city: str | None = Field(default=None, min_length=1, max_length=40)
    job_type: Literal["internship", "full_time", "part_time"] | None = Field(
        default=None,
        description="BOSS 求职类型筛选：实习、全职或兼职。不要把类型词重复放进 keyword。",
    )


class ReadConversationSpanToolArguments(ContractModel):
    from_sequence: int = Field(ge=1)
    through_sequence: int = Field(ge=1)
    query: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description=(
            "Focused terms from the user's request, such as 目标公司 or Rust. "
            "Use this for long omitted ranges; omit it only for an exact "
            "sequence span the user named."
        ),
    )

    @model_validator(mode="after")
    def require_forward_span(self) -> "ReadConversationSpanToolArguments":
        if self.from_sequence > self.through_sequence:
            raise ValueError("from_sequence cannot exceed through_sequence")
        return self


class ResolveClaimSourceToolArguments(ContractModel):
    source_ref: str = Field(pattern=r"^evidence_[a-f0-9]{24}$")


class GetCareerMemoryDetailToolArguments(ContractModel):
    detail_ref: str = Field(pattern=r"^detail_[a-f0-9]{24}$")


class SearchCareerMemoryToolArguments(ContractModel):
    query: str = Field(
        min_length=3,
        max_length=200,
        description=(
            "Focused terms expected inside active career claims omitted from "
            "the bounded Tier-1 window."
        ),
    )
    limit: int = Field(default=8, ge=1, le=20)
    cursor: str | None = Field(
        default=None,
        pattern=r"^memory_[a-f0-9]{8}_[a-f0-9]{8}$",
        description="Opaque next-page cursor returned by an earlier identical query.",
    )


class UpdateWorkingNotesToolArguments(ContractModel):
    expected_revision: str = Field(
        pattern=r"^(?:empty|[a-f0-9]{12})$",
        description="Revision from the current working_notes context projection.",
    )
    markdown: str = Field(
        max_length=2000,
        description=(
            "Complete replacement markdown. It may guide questions and response "
            "style, but never filtering, ranking, applications, or other actions."
        ),
    )


class SearchCareerEpisodesToolArguments(ContractModel):
    """Bounded L1 recall across completed workflows and past conversations."""

    detail_ref: str | None = Field(
        default=None,
        pattern=r"^episode:career_episode_[a-f0-9]{32}$",
        description=(
            "Exact projected episode detail_ref. When present, dereference that "
            "episode instead of running a broad query."
        ),
    )
    query: str = Field(
        default="",
        max_length=200,
        description=(
            "Focused terms from the remembered event. Leave empty only when "
            "listing recent episodes within the supplied filters."
        ),
    )
    start_datetime: datetime | None = Field(
        default=None,
        description="Inclusive lower bound for when the episode occurred.",
    )
    end_datetime: datetime | None = Field(
        default=None,
        description="Inclusive upper bound for when the episode occurred.",
    )
    kinds: tuple[
        Literal[
            "mock_interview",
            "job_research",
            "application",
            "interview_round",
            "resume_analysis",
            "intent_confirmation",
            "resume_tailoring",
        ],
        ...,
    ] = Field(
        default=(),
        max_length=7,
        description="Optional episode-type filters.",
    )
    top_k: int = Field(default=8, ge=1, le=20)

    @model_validator(mode="after")
    def time_window_is_forward(self) -> "SearchCareerEpisodesToolArguments":
        for name, value in (
            ("start_datetime", self.start_datetime),
            ("end_datetime", self.end_datetime),
        ):
            if value is not None and value.utcoffset() is None:
                raise ValueError(f"{name} must include a timezone offset")
        if (
            self.start_datetime is not None
            and self.end_datetime is not None
            and self.start_datetime > self.end_datetime
        ):
            raise ValueError("start_datetime cannot exceed end_datetime")
        return self


class SearchCareerHistoryToolArguments(ContractModel):
    query: str = Field(
        min_length=3,
        max_length=200,
        description=(
            "Focused terms expected inside the earlier claim. Do not pass a "
            "generic request such as 'what did I say before'."
        ),
    )
    limit: int = Field(default=8, ge=1, le=20)
    cursor: str | None = Field(
        default=None,
        pattern=r"^history_[a-f0-9]{8}_[a-f0-9]{8}$",
        description="Opaque next-page cursor returned by an earlier identical query.",
    )


class ProposeMemoryTombstoneToolArguments(ContractModel):
    detail_ref: str = Field(pattern=r"^detail_[a-f0-9]{24}$")
    reason: str = Field(min_length=1, max_length=2000)


class ConfirmMemoryTombstoneToolArguments(ContractModel):
    pass


class ProposeConstraintRetirementToolArguments(ContractModel):
    constraint: str = Field(min_length=1, max_length=SUMMARY_ITEM_MAX_CHARS)
    reason: str = Field(min_length=1, max_length=2000)


class ConfirmConstraintRetirementToolArguments(ContractModel):
    pass


class FetchArchivedConstraintsToolArguments(ContractModel):
    pass


class ProposeMemoryAmendmentToolArguments(ContractModel):
    detail_ref: str = Field(pattern=r"^detail_[a-f0-9]{24}$")
    new_claim: str = Field(min_length=1, max_length=32_000)
    reason: str = Field(min_length=1, max_length=2000)


class ConfirmMemoryAmendmentToolArguments(ContractModel):
    pass


class ProposeCareerFactToolArguments(ContractModel):
    record_selection_index: SelectionIndex
    claim: str = Field(min_length=1, max_length=2000)
    reason: str = Field(min_length=1, max_length=2000)
    user_quote: str | None = Field(
        default=None, min_length=4, max_length=500,
        description=(
            "Exact verbatim excerpt from the user's own message or questionnaire "
            "answer supporting this claim. Omit when the claim is an inference."
        ),
    )


class ConfirmCareerFactToolArguments(ContractModel):
    pass

