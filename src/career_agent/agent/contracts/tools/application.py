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

from career_agent.agent.contracts.profile import *
from career_agent.agent.contracts.candidates import *

class CreateApplicationToolArguments(ContractModel):
    job_posting_id: str | None = Field(default=None, min_length=1)
    resume_version_id: str | None = Field(default=None, min_length=1)
    job_selection_index: SelectionIndex | None = None
    resume_version_selection_index: SelectionIndex | None = None
    submitted_at: datetime | None = None
    note: str | None = Field(default=None, min_length=1, max_length=2000)


class UpdateOwnerSettingsToolArguments(ContractModel):
    """An agent proposal; the runtime always seals it for owner review."""

    boss_search: Literal["explicit_request_only", "allowed"] | None = None
    application_confirmation: Literal["always_ask", "on_user_report"] | None = None
    confirm_before: ConfirmBefore | None = Field(
        default=None,
        description=(
            "Full replacement list of WRITE capability names the owner wants to "
            "approve one by one before they run. Pass an empty list to clear."
        ),
    )

    @model_validator(mode="after")
    def changes_something(self) -> "UpdateOwnerSettingsToolArguments":
        if (
            self.boss_search is None
            and self.application_confirmation is None
            and self.confirm_before is None
        ):
            raise ValueError("an owner-settings proposal must change at least one setting")
        return self

    @field_validator("confirm_before")
    @classmethod
    def normalise_confirm_before(cls, value: ConfirmBefore | None) -> ConfirmBefore | None:
        return None if value is None else canonical_confirm_before(value)


class UpdateApplicationStatusToolArguments(ContractModel):
    application_id: str | None = Field(default=None, min_length=1)
    selection_index: SelectionIndex | None = None
    status: ApplicationStatus
    note: str | None = Field(default=None, min_length=1, max_length=2000)

    @model_validator(mode="after")
    def validate_application_selector(self) -> "UpdateApplicationStatusToolArguments":
        if self.application_id is not None and self.selection_index is not None:
            raise ValueError("use either application_id or selection_index")
        return self


class ListApplicationsToolArguments(ContractModel):
    statuses: tuple[ApplicationStatus, ...] = Field(default=(), max_length=8)
    limit: int = Field(default=20, ge=1, le=50)

    @model_validator(mode="after")
    def validate_statuses(self) -> "ListApplicationsToolArguments":
        if len(set(self.statuses)) != len(self.statuses):
            raise ValueError("application statuses must be unique")
        return self


class GetApplicationToolArguments(ContractModel):
    application_id: str | None = Field(default=None, min_length=1)
    selection_index: SelectionIndex | None = None

    @model_validator(mode="after")
    def validate_application_selector(self) -> "GetApplicationToolArguments":
        if self.application_id is not None and self.selection_index is not None:
            raise ValueError("use either application_id or selection_index")
        return self


class SyncApplicationEmailsToolArguments(ContractModel):
    account_id: str | None = Field(default=None, min_length=1)


class ListEmailEventsToolArguments(ContractModel):
    status: EmailEventStatus | None = None
    limit: int = Field(default=20, ge=1, le=50)


class ResolveEmailEventToolArguments(ContractModel):
    event_id: str | None = Field(default=None, min_length=1)
    selection_index: SelectionIndex | None = None
    approve: bool
    application_id: str | None = Field(default=None, min_length=1)
    interview_round_id: str | None = Field(default=None, min_length=1)

