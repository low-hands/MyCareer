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

from career_agent.agent.contracts.profile import *
from career_agent.agent.contracts.candidates import *
from career_agent.agent.contracts.observations import ResourceHandle

class FindSavedJobsToolArguments(ContractModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=20)


class GetSavedJobToolArguments(ContractModel):
    job_posting_id: str | None = Field(default=None, min_length=1)
    selection_index: SelectionIndex | None = None


class AnalyzeJobToolArguments(ContractModel):
    job_posting_id: str | None = Field(default=None, min_length=1)
    selection_index: SelectionIndex | None = None


class CorrectJobRequirementTierToolArguments(ContractModel):
    analysis_id: str = Field(min_length=1)
    requirement_id: str = Field(pattern=r"^job_requirement_[a-f0-9]{20}$")
    tier: Literal["S", "A", "B", "C"]
    reason: str = Field(min_length=1, max_length=1000)
    confirmation: Literal["confirm", "correct"] = "correct"


class ResearchJobToolArguments(ContractModel):
    job_posting_id: str | None = Field(default=None, min_length=1)
    selection_index: SelectionIndex | None = None
    focus: str | None = Field(default=None, min_length=1, max_length=1000)
    user_provided_context: str | None = Field(
        default=None,
        min_length=1,
        max_length=4000,
        description=(
            "Relevant business or product clues the user explicitly chose to use, "
            "for example something they heard in an interview. This is unverified "
            "user-reported context, not a public fact."
        ),
    )
    max_sources: int = Field(default=8, ge=2, le=15)


class RetryJobResearchToolArguments(ContractModel):
    run_id: str | None = Field(default=None, min_length=1)


IMPLICIT_REQUEST_KEY = "implicit_request"
"""Set by projection, never by the model: the arguments model forbids it."""


class GetJobResearchToolArguments(ContractModel):
    """Selectors for reading back one job-research report.

    ``reference`` identifies an exact delivered report. ``selection_index``
    selects a saved job and reads its company's latest available report, which
    may differ from a requested historical version. Omitting both uses the
    active report or job. Internal ids are stripped from model-facing schemas
    and supplied by argument projection.
    """

    report_id: str | None = Field(default=None, min_length=1)
    job_posting_id: str | None = Field(default=None, min_length=1)
    selection_index: SelectionIndex | None = None
    reference: ResourceHandle | None = None

    @model_validator(mode="after")
    def validate_selector(self) -> "GetJobResearchToolArguments":
        if self.reference is not None and self.selection_index is not None:
            raise ValueError("use either reference or selection_index")
        return self


class ListTargetRolesToolArguments(ContractModel):
    pass


class ListResumesToolArguments(ContractModel):
    target_role_id: str | None = Field(default=None, min_length=1)
    target_role_selection_index: SelectionIndex | None = None
    query: str | None = Field(
        default=None,
        min_length=1,
        max_length=120,
        description="用户提到的简历名称、方向或标签；优先用它缩小候选范围。",
    )


class GetResumeMetadataToolArguments(ContractModel):
    resume_id: str | None = Field(default=None, min_length=1)
    selection_index: SelectionIndex | None = None


MainAgentSkill = Literal["resume-critique"]
"""Skills the main agent itself may load. Specialist skills (job research,
mock interview, tailoring) belong to their workers and are not listed."""


class LoadSkillToolArguments(ContractModel):
    skill: MainAgentSkill = Field(
        description=(
            "resume-critique: how to critique a resume as a document when the "
            "user asks to review, critique or improve it without naming a job."
        )
    )


class MatchResumeToJobToolArguments(ContractModel):
    resume_version_id: str | None = Field(default=None, min_length=1)
    job_posting_id: str | None = Field(default=None, min_length=1)
    resume_version_selection_index: SelectionIndex | None = None
    job_selection_index: SelectionIndex | None = None


class ProposeJobIntentToolArguments(ContractModel):
    target_role_selection_index: SelectionIndex | None = None
    pref_scope: str = Field(
        default="global",
        pattern=r"^(?:global|[a-z][a-z0-9_.:-]*)$",
        max_length=120,
        description=(
            "Use global unless the user explicitly limits this preference to "
            "a named situation or domain."
        ),
    )
    timescale: Literal["permanent", "situational"] = "permanent"
    city: str | None = Field(default=None, min_length=1, max_length=40)
    salary_expectation: str | None = Field(default=None, min_length=1, max_length=100)
    experience: str | None = Field(default=None, min_length=1, max_length=100)
    education: str | None = Field(default=None, min_length=1, max_length=100)
    hard_constraints: tuple[HardConstraintContext, ...] = Field(
        default=(),
        max_length=8,
    )


class ConfirmJobIntentToolArguments(ContractModel):
    pass


class ProposeFreeTextPreferenceConfirmationToolArguments(ContractModel):
    selection_index: SelectionIndex


class ConfirmFreeTextPreferenceToolArguments(ContractModel):
    scope_choice: Literal[
        "person_stable",
        "person_default",
        "person_situational",
        "role",
        "situational",
    ] | None = None
    scope_domain: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_.:-]{0,79}$",
    )

    @model_validator(mode="after")
    def role_choice_names_its_domain(
        self,
    ) -> "ConfirmFreeTextPreferenceToolArguments":
        if self.scope_choice == "role" and self.scope_domain is None:
            raise ValueError("role scope requires scope_domain")
        if self.scope_choice != "role" and self.scope_domain is not None:
            raise ValueError("scope_domain is only valid for role scope")
        return self


class CompareSavedJobsToolArguments(ContractModel):
    job_selection_indices: tuple[SelectionIndex, ...] = Field(
        min_length=2, max_length=10
    )

