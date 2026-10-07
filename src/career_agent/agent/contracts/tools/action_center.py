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

from career_agent.agent.contracts.candidates import *

class GetDailyBriefToolArguments(ContractModel):
    timezone: str = Field(default="Asia/Shanghai", min_length=1, max_length=100)


class ListActionItemsToolArguments(ContractModel):
    statuses: tuple[ActionStatus, ...] = Field(
        default=("open", "snoozed"), min_length=1, max_length=4
    )
    limit: int = Field(default=50, ge=1, le=100)
    timezone: str = Field(default="Asia/Shanghai", min_length=1, max_length=100)


class ResolveActionItemToolArguments(ContractModel):
    action_item_id: str | None = Field(default=None, min_length=1)
    selection_index: SelectionIndex | None = None

    @model_validator(mode="after")
    def validate_selector(self) -> "ResolveActionItemToolArguments":
        if self.action_item_id is not None and self.selection_index is not None:
            raise ValueError("use either action_item_id or selection_index")
        return self


class SnoozeActionItemToolArguments(ResolveActionItemToolArguments):
    snoozed_until: datetime


class ListCalendarAccountsToolArguments(ContractModel):
    pass


class ListCalendarLinksToolArguments(ContractModel):
    pass


class PrepareInterviewCalendarSyncToolArguments(ContractModel):
    interview_round_id: str | None = Field(default=None, min_length=1)
    interview_selection_index: SelectionIndex | None = None
    calendar_account_id: str | None = Field(default=None, min_length=1)
    calendar_account_selection_index: SelectionIndex | None = None

    @model_validator(mode="after")
    def validate_selectors(self) -> "PrepareInterviewCalendarSyncToolArguments":
        if self.interview_round_id and self.interview_selection_index:
            raise ValueError("use one interview selector")
        if self.calendar_account_id and self.calendar_account_selection_index:
            raise ValueError("use one calendar account selector")
        return self


class GetCalendarProposalToolArguments(ContractModel):
    proposal_id: str | None = Field(default=None, min_length=1)


class ExecuteCalendarProposalToolArguments(GetCalendarProposalToolArguments):
    pass

