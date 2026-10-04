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

class GetResumeJobMatchToolArguments(ContractModel):
    match_id: str | None = Field(default=None, min_length=1)


class DraftResumeTailoringToolArguments(ContractModel):
    match_id: str | None = Field(default=None, min_length=1)
    tailoring_goal: str | None = Field(default=None, min_length=1, max_length=2000)


class GetResumeTailoringDraftToolArguments(ContractModel):
    draft_id: str | None = Field(default=None, min_length=1)


class ReviewResumeTailoringToolArguments(ContractModel):
    draft_id: str | None = Field(default=None, min_length=1)
    accepted_change_indices: tuple[SelectionIndex, ...] = Field(
        default=(), max_length=30
    )
    rejected_change_indices: tuple[SelectionIndex, ...] = Field(
        default=(), max_length=30
    )
    feedback: str | None = Field(default=None, min_length=1, max_length=2000)

    @model_validator(mode="after")
    def validate_decisions(self) -> ReviewResumeTailoringToolArguments:
        accepted = self.accepted_change_indices
        rejected = self.rejected_change_indices
        if not accepted and not rejected:
            raise ValueError("at least one accepted or rejected change index is required")
        if any(index < 1 for index in (*accepted, *rejected)):
            raise ValueError("change indices must be positive")
        if len(set(accepted)) != len(accepted) or len(set(rejected)) != len(rejected):
            raise ValueError("change indices must be unique")
        if set(accepted).intersection(rejected):
            raise ValueError("a change cannot be both accepted and rejected")
        return self


class ReviseResumeTailoringToolArguments(ContractModel):
    draft_id: str | None = Field(default=None, min_length=1)
    feedback: str = Field(min_length=1, max_length=2000)


class FinalizeResumeTailoringToolArguments(ContractModel):
    draft_id: str | None = Field(default=None, min_length=1)


class ExportResumeArtifactToolArguments(ContractModel):
    resume_version_id: str | None = Field(default=None, min_length=1)

