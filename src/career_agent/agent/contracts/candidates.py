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


SelectionIndex = Annotated[
    int,
    Field(
        ge=1,
        description=(
            "A 1-based selection_index explicitly shown beside the intended "
            "object in this tool's corresponding candidate list. Never invent "
            "an index or borrow numbering from another list or from the order "
            "of tool observations."
        ),
    ),
]
"""A 1-based pointer into a list the model was shown this turn.

The lower bound belongs to the type, not to each declaration. Written out
per-field it was correct twenty-nine times and missing once — on
``CompareSavedJobsToolArguments.job_selection_indices``, whose element type
carried no bound at all, so index 0 resolved through ``candidates[-1]`` to the
last saved job and a comparison ran against the wrong pair without erroring.
"""


class CandidateContextItem(ContractModel):
    result_ref: str
    title: str
    company_name: str
    city: str | None = None
    salary: str | None = None


class ApplicationCandidateContextItem(ContractModel):
    application_id: str
    title: str
    company_name: str
    status: ApplicationStatus


class InterviewCandidateContextItem(ContractModel):
    interview_round_id: str
    application_id: str
    sequence_number: int
    employer_label: str | None = None
    status: InterviewStatus
    scheduled_start: datetime | None = None






class ActionCandidateContextItem(ContractModel):
    action_item_id: str
    action_type: ActionType
    source_type: ActionSourceType
    source_id: str
    title: str
    status: ActionStatus
    due_at: datetime | None = None


class CalendarAccountCandidateContextItem(ContractModel):
    calendar_account_id: str
    provider: Literal["google"]
    email_address: str
    calendar_id: str


class SavedJobCandidateContextItem(ContractModel):
    job_posting_id: str
    title: str
    company_name: str
    city: str | None = None
    salary: str | None = None
    jd_snapshot_id: str | None = None
    jd_version: int | None = Field(default=None, ge=1)


class ActiveSavedJobContextItem(ContractModel):
    """Which saved job, and which JD version of it, the conversation is on.

    ``active_job_posting_id`` says which posting; this pins the snapshot that
    turn actually read so "this job" keeps resolving to the same text after the
    posting is re-captured as v2. The ids never reach the model — the
    projection shows the title, the version and whether the snapshot can
    still be read.
    """

    job_posting_id: str = Field(min_length=1)
    jd_snapshot_id: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=200)
    company_name: str = Field(min_length=1, max_length=200)
    jd_version: int = Field(ge=1)
    readable: bool = True


class TargetRoleCandidateContextItem(ContractModel):
    target_role_id: str
    title: str
    priority: int
    status: str
    # The intent recorded against this track. It rides along here rather than
    # being flattened into career_profile so the model sees which numbers belong
    # to which role instead of one blended set.
    city: str | None = None
    salary_expectation: str | None = None
    experience: str | None = None
    education: str | None = None


class ResumeCandidateContextItem(ContractModel):
    resume_id: str
    target_role_id: str
    name: str
    status: str
    latest_version_id: str | None = None


class ResumeVersionCandidateContextItem(ContractModel):
    resume_version_id: str
    version_number: int
    source_type: str
    document_format: str
    byte_size: int
    # Set when the list spans several resumes, as the mock-interview resume
    # choice does; a single resume's version list leaves it empty.
    resume_name: str | None = None


class EmailEventCandidateContextItem(ContractModel):
    email_event_id: str
    event_type: str
    status: EmailEventStatus
    summary: str

