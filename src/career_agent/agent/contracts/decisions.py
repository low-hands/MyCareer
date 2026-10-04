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

from career_agent.agent.contracts.context import MainAgentContext

class ToolCall(ContractModel):
    name: str
    arguments: dict[str, Any] = {}


class AgentDecision(ContractModel):
    action: Literal["ask_user", "questionnaire", "tool_call", "final"]
    message: str | None = None
    tool_call: ToolCall | None = None
    questions: tuple[UserQuestion, ...] = Field(default=(), max_length=8)
    selection_source: Literal["latest_tool_result"] | None = None

    @model_validator(mode="after")
    def _questionnaire_shape(self) -> "AgentDecision":
        if self.action == "questionnaire":
            if not 2 <= len(self.questions) <= 8:
                raise ValueError("questionnaire needs 2-8 questions")
            if tuple(item.question_id for item in self.questions) != tuple(
                f"q{index}" for index in range(1, len(self.questions) + 1)
            ):
                raise ValueError("questionnaire ids must be ordered q1..qN")
        elif self.questions:
            raise ValueError("questions require questionnaire action")
        if self.selection_source is not None and self.action != "ask_user":
            raise ValueError("selection_source requires ask_user action")
        return self


class DecisionMaker(Protocol):
    def decide(self, context: MainAgentContext, tool_specs: tuple[dict[str, Any], ...]) -> AgentDecision: ...


def _reject_internal_identifiers(name: str, arguments: dict[str, Any]) -> None:
    forbidden = sorted(key for key in arguments if key == "user_id" or key.endswith("_id"))
    if forbidden:
        raise ValueError(
            f"{name} cannot accept internal identifiers: {', '.join(forbidden)}"
        )

