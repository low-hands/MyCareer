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
from career_agent.agent.contracts.tools.job import *
from career_agent.agent.contracts.decisions import _reject_internal_identifiers

def project_saved_job_arguments(context: MainAgentContext, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    _reject_internal_identifiers(name, arguments)
    if name == "find_saved_jobs":
        model_arguments = FindSavedJobsToolArguments.model_validate(arguments)
    elif name == "get_saved_job":
        model_arguments = GetSavedJobToolArguments.model_validate(arguments)
    elif name == "compare_saved_jobs":
        model_arguments = CompareSavedJobsToolArguments.model_validate(arguments)
    elif name == "analyze_job":
        model_arguments = AnalyzeJobToolArguments.model_validate(arguments)
    else:
        raise ValueError(f"Unknown saved-job tool: {name}")
    payload = model_arguments.model_dump()
    if name == "compare_saved_jobs":
        indices = payload.pop("job_selection_indices", ())
        candidates = context.task.saved_job_candidates
        # Both bounds here as well as in the schema. These guards checked only
        # the upper one, so the lower rested entirely on ``ge=1`` per field —
        # correct twenty-nine times and absent on this collection's element
        # type, where index 0 read ``candidates[-1]`` and compared a job
        # against itself without erroring.
        job_posting_ids = []
        for index in indices:
            if not 1 <= index <= len(candidates):
                raise ValueError("saved-job selection index is out of range")
            job_posting_ids.append(candidates[index - 1].job_posting_id)
        payload["job_posting_ids"] = tuple(job_posting_ids)
        payload["preferred_city"] = context.profile.default_city
    if name in {"get_saved_job", "analyze_job"}:
        selection_index = payload.pop("selection_index", None)
        job_posting_id = context.task.active_job_posting_id
        if selection_index is not None:
            if not 1 <= selection_index <= len(context.task.saved_job_candidates):
                raise ValueError("saved-job selection index is out of range")
            job_posting_id = context.task.saved_job_candidates[
                selection_index - 1
            ].job_posting_id
        if job_posting_id is None:
            raise ValueError(f"{name} requires a selected or active saved job")
        payload["job_posting_id"] = job_posting_id
        payload["jd_snapshot_id"] = _pinned_jd_snapshot_id(
            context, job_posting_id=job_posting_id, explicit_selection=selection_index is not None
        )
    return {"user_id": context.profile.user_id, **payload}


def _pinned_jd_snapshot_id(
    context: MainAgentContext, *, job_posting_id: str, explicit_selection: bool
) -> str | None:
    # "This job" with no selection is the pinned snapshot, not whatever the
    # posting's latest capture is; an explicit selection reads the latest.
    focus = context.task.focused_saved_job()
    if explicit_selection or focus is None or focus.job_posting_id != job_posting_id:
        return None
    return focus.jd_snapshot_id

