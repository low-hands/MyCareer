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
from career_agent.agent.contracts.observations import ResourceHandle

class ListInterviewsToolArguments(ContractModel):
    application_id: str | None = Field(default=None, min_length=1)
    statuses: tuple[InterviewStatus, ...] = Field(default=(), max_length=4)
    limit: int = Field(default=20, ge=1, le=50)


class GetInterviewToolArguments(ContractModel):
    interview_round_id: str | None = Field(default=None, min_length=1)
    selection_index: SelectionIndex | None = None

    @model_validator(mode="after")
    def validate_selector(self) -> "GetInterviewToolArguments":
        if self.interview_round_id is not None and self.selection_index is not None:
            raise ValueError("use either interview_round_id or selection_index")
        return self


class CreateInterviewToolArguments(ContractModel):
    application_id: str | None = Field(default=None, min_length=1)
    application_selection_index: SelectionIndex | None = None
    job_posting_id: str | None = Field(default=None, min_length=1)
    job_selection_index: SelectionIndex | None = None
    details: InterviewDetails


class UpdateInterviewToolArguments(ContractModel):
    interview_round_id: str | None = Field(default=None, min_length=1)
    selection_index: SelectionIndex | None = None
    details: InterviewDetails

    @model_validator(mode="after")
    def validate_selector(self) -> "UpdateInterviewToolArguments":
        if self.interview_round_id is not None and self.selection_index is not None:
            raise ValueError("use either interview_round_id or selection_index")
        return self


class CompleteInterviewToolArguments(ContractModel):
    interview_round_id: str | None = Field(default=None, min_length=1)
    selection_index: SelectionIndex | None = None
    completed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_selector(self) -> "CompleteInterviewToolArguments":
        if self.interview_round_id is not None and self.selection_index is not None:
            raise ValueError("use either interview_round_id or selection_index")
        return self


class RecordInterviewRetroToolArguments(ContractModel):
    interview_round_id: str | None = Field(default=None, min_length=1)
    selection_index: SelectionIndex | None = None
    source_notes: str = Field(min_length=1, max_length=20_000)
    summary: str = Field(min_length=1, max_length=5000)
    questions: tuple[InterviewRetroQuestion, ...] = Field(default=(), max_length=30)
    strengths: tuple[str, ...] = Field(default=(), max_length=20)
    difficulties: tuple[str, ...] = Field(default=(), max_length=20)
    interviewer_signals: tuple[str, ...] = Field(default=(), max_length=20)
    next_focus: tuple[str, ...] = Field(default=(), max_length=20)
    action_items: tuple[str, ...] = Field(default=(), max_length=20)
    limitations: tuple[str, ...] = Field(default=(), max_length=20)
    self_assessment: InterviewSelfAssessment = "uncertain"

    @model_validator(mode="after")
    def validate_selector(self) -> "RecordInterviewRetroToolArguments":
        if self.interview_round_id is not None and self.selection_index is not None:
            raise ValueError("use either interview_round_id or selection_index")
        return self


class PrepareInterviewToolArguments(ContractModel):
    interview_round_id: str | None = Field(default=None, min_length=1)
    selection_index: SelectionIndex | None = None
    action_selection_index: SelectionIndex | None = None

    @model_validator(mode="after")
    def validate_selector(self) -> "PrepareInterviewToolArguments":
        selectors = (
            self.interview_round_id,
            self.selection_index,
            self.action_selection_index,
        )
        if sum(value is not None for value in selectors) > 1:
            raise ValueError("use one interview selector")
        return self


class GetInterviewPreparationToolArguments(ContractModel):
    """Selectors for reading back one interview preparation.

    Defaults to the active preparation. ``reference`` reaches one an earlier
    turn in the window produced, which the active pointer no longer names once
    the focus has moved on.
    """

    preparation_id: str | None = Field(default=None, min_length=1)
    reference: ResourceHandle | None = None
    interview_selection_index: SelectionIndex | None = None
    """The interview whose preparation to read, from ``list_interviews``.

    The entity-keyed route the other two readbacks already had: job research
    takes a saved job, a mock interview result takes an application. Without one
    here, a preparation fell out of reach for good once it stopped being active
    and its message left the recent window — while the report itself was still
    on disk and still visible in the transcript.
    """


class StartMockInterviewToolArguments(ContractModel):
    """The resume, job and company fields are free-practice only: an
    application run always uses its submitted resume and immutable JD.

    ``company_name`` is the employer as the user named it when no saved job is
    chosen; ``without_job`` records that the user declined the saved jobs
    offered for that company and wants company-only practice.
    """

    practice_scope: Literal["application", "free"] | None = None
    application_selection_index: SelectionIndex | None = None
    interview_selection_index: SelectionIndex | None = None
    interview_type: MockInterviewType | None = None
    target_role: str | None = Field(default=None, min_length=1, max_length=300)
    max_primary_questions: int = Field(default=10, ge=1, le=20)
    max_follow_ups_per_question: int = Field(default=2, ge=0, le=5)
    resume_version_selection_index: SelectionIndex | None = None
    without_resume: bool = False
    job_selection_index: SelectionIndex | None = None
    company_name: str | None = Field(default=None, min_length=1, max_length=100)
    without_job: bool = False

    @model_validator(mode="after")
    def validate_selector(self) -> "StartMockInterviewToolArguments":
        if self.without_job and self.job_selection_index is not None:
            raise ValueError("choose a job or without_job, not both")
        if (
            self.application_selection_index is not None
            and self.interview_selection_index is not None
        ):
            raise ValueError("use either an application or interview selector")
        if self.without_resume and self.resume_version_selection_index is not None:
            raise ValueError("choose a resume or without_resume, not both")
        return self


class RestartMockInterviewToolArguments(ContractModel):
    """No arguments: the replacement copies the stuck run's own settings.

    Letting the model restate the application or the question caps would let a
    restart quietly practise against a different resume than the run it
    replaces.
    """




class GetMockInterviewResultToolArguments(ContractModel):
    """Selectors for reading back one finished mock interview.

    Defaults to the latest finished run for the active application, which is
    what "how did I do" means in practice. ``reference`` instead names the
    exact run an earlier turn reported, which matters here more than elsewhere:
    one application can be practised against repeatedly, so "the latest" and
    "the one you just told me about" stop agreeing after a second run. A
    question number narrows the read to one exchange, because returning every
    answer in full would crowd out the rest of the conversation.
    """

    application_selection_index: SelectionIndex | None = None
    reference: ResourceHandle | None = None
    question_number: int | None = Field(default=None, ge=1, le=20)

    @model_validator(mode="after")
    def validate_selector(self) -> "GetMockInterviewResultToolArguments":
        if (
            self.reference is not None
            and self.application_selection_index is not None
        ):
            raise ValueError(
                "use either reference or application_selection_index"
            )
        return self

