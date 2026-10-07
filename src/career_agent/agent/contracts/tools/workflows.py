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


class StartMockInterviewWorkflowInput(ContractModel):
    user_id: str = Field(min_length=1)
    application_id: str | None = Field(default=None, min_length=1)
    interview_round_id: str | None = Field(default=None, min_length=1)
    interview_type: MockInterviewType
    target_role: str | None = Field(default=None, min_length=1, max_length=300)
    max_primary_questions: int = Field(ge=1, le=20)
    max_follow_ups_per_question: int = Field(ge=0, le=5)
    conversation_id: str | None = Field(default=None, min_length=1)
    # Free practice only. ``required`` means nobody has said which resume to
    # use yet, so the run must not start; the tool asks instead of guessing.
    resume_choice: Literal["chosen", "none", "required"] = "chosen"
    resume_version_id: str | None = Field(default=None, min_length=1)
    # ``check`` means the user named a company but no job: the tool offers any
    # saved jobs at that company before starting company-only practice.
    job_choice: Literal["chosen", "none", "check"] = "none"
    job_posting_id: str | None = Field(default=None, min_length=1)
    jd_snapshot_id: str | None = Field(default=None, min_length=1)
    target_company: str | None = Field(default=None, min_length=1, max_length=200)

    @model_validator(mode="after")
    def validate_resume_choice(self) -> "StartMockInterviewWorkflowInput":
        if (self.resume_choice == "chosen") != (self.resume_version_id is not None) and (
            self.application_id is None
        ):
            raise ValueError("a chosen resume needs its version, and only then")
        return self

