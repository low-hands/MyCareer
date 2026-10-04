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
from career_agent.agent.contracts.tools.resume import *
from career_agent.agent.contracts.tools.application import *
from career_agent.agent.contracts.decisions import _reject_internal_identifiers
from career_agent.agent.contracts.projections.common import _pinned_jd_snapshot_id

def project_resume_arguments(context: MainAgentContext, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    _reject_internal_identifiers(name, arguments)
    if name == "list_target_roles":
        model_arguments = ListTargetRolesToolArguments.model_validate(arguments)
    elif name == "list_resumes":
        model_arguments = ListResumesToolArguments.model_validate(arguments)
    elif name == "get_resume_metadata":
        model_arguments = GetResumeMetadataToolArguments.model_validate(arguments)
    elif name == "match_resume_to_job":
        model_arguments = MatchResumeToJobToolArguments.model_validate(arguments)
    elif name == "get_resume_job_match":
        model_arguments = GetResumeJobMatchToolArguments.model_validate(arguments)
    elif name == "draft_resume_tailoring":
        model_arguments = DraftResumeTailoringToolArguments.model_validate(arguments)
    elif name == "get_resume_tailoring_draft":
        model_arguments = GetResumeTailoringDraftToolArguments.model_validate(arguments)
    elif name == "review_resume_tailoring":
        model_arguments = ReviewResumeTailoringToolArguments.model_validate(arguments)
    elif name == "revise_resume_tailoring":
        model_arguments = ReviseResumeTailoringToolArguments.model_validate(arguments)
    elif name == "finalize_resume_tailoring":
        model_arguments = FinalizeResumeTailoringToolArguments.model_validate(arguments)
    elif name == "export_resume_artifact":
        model_arguments = ExportResumeArtifactToolArguments.model_validate(arguments)
    elif name == "create_application":
        model_arguments = CreateApplicationToolArguments.model_validate(arguments)
    elif name == "update_application_status":
        model_arguments = UpdateApplicationStatusToolArguments.model_validate(arguments)
    elif name == "list_applications":
        model_arguments = ListApplicationsToolArguments.model_validate(arguments)
    elif name == "get_application":
        model_arguments = GetApplicationToolArguments.model_validate(arguments)
    else:
        raise ValueError(f"Unknown resume tool: {name}")
    payload = model_arguments.model_dump()
    if name == "list_resumes":
        selection_index = payload.pop("target_role_selection_index", None)
        if selection_index is not None:
            if not 1 <= selection_index <= len(context.task.target_role_candidates):
                raise ValueError("target-role selection index is out of range")
            payload["target_role_id"] = context.task.target_role_candidates[
                selection_index - 1
            ].target_role_id
    if name == "get_resume_metadata":
        selection_index = payload.pop("selection_index", None)
        if selection_index is not None:
            if not 1 <= selection_index <= len(context.task.resume_candidates):
                raise ValueError("resume selection index is out of range")
            payload["resume_id"] = context.task.resume_candidates[
                selection_index - 1
            ].resume_id
        if payload.get("resume_id") is None:
            raise ValueError("get_resume_metadata requires a selected resume")
    if name == "match_resume_to_job":
        resume_selection_index = payload.pop(
            "resume_version_selection_index", None
        )
        job_selection_index = payload.pop("job_selection_index", None)
        resume_version_id = context.task.active_resume_version_id
        if resume_selection_index is not None:
            if not 1 <= resume_selection_index <= len(context.task.resume_version_candidates):
                raise ValueError("resume-version selection index is out of range")
            resume_version_id = context.task.resume_version_candidates[
                resume_selection_index - 1
            ].resume_version_id
        job_posting_id = context.task.active_job_posting_id
        if job_selection_index is not None:
            if not 1 <= job_selection_index <= len(context.task.saved_job_candidates):
                raise ValueError("saved-job selection index is out of range")
            job_posting_id = context.task.saved_job_candidates[
                job_selection_index - 1
            ].job_posting_id
        if resume_version_id is None or job_posting_id is None:
            raise ValueError(
                "match_resume_to_job requires selected or active resume and saved job"
            )
        payload["resume_version_id"] = resume_version_id
        payload["job_posting_id"] = job_posting_id
        payload["jd_snapshot_id"] = _pinned_jd_snapshot_id(
            context,
            job_posting_id=job_posting_id,
            explicit_selection=job_selection_index is not None,
        )
    if name == "get_resume_job_match":
        match_id = payload.get("match_id") or context.task.active_resume_job_match_id
        if match_id is None:
            raise ValueError("get_resume_job_match requires an active resume-job match")
        payload["match_id"] = match_id
    if name == "draft_resume_tailoring":
        match_id = payload.get("match_id") or context.task.active_resume_job_match_id
        if match_id is None:
            raise ValueError("draft_resume_tailoring requires an active resume-job match")
        payload["match_id"] = match_id
    if name in {
        "get_resume_tailoring_draft",
        "review_resume_tailoring",
        "revise_resume_tailoring",
        "finalize_resume_tailoring",
    }:
        draft_id = payload.get("draft_id") or context.task.active_resume_tailoring_draft_id
        if draft_id is None:
            raise ValueError(f"{name} requires an active tailoring draft")
        payload["draft_id"] = draft_id
    if name == "export_resume_artifact":
        resume_version_id = (
            payload.get("resume_version_id") or context.task.active_resume_version_id
        )
        if resume_version_id is None:
            raise ValueError("export_resume_artifact requires an active resume version")
        payload["resume_version_id"] = resume_version_id
    if name == "create_application":
        job_selection_index = payload.pop("job_selection_index", None)
        resume_selection_index = payload.pop(
            "resume_version_selection_index", None
        )
        job_posting_id = context.task.active_job_posting_id
        if job_selection_index is not None:
            if not 1 <= job_selection_index <= len(context.task.saved_job_candidates):
                raise ValueError("saved-job selection index is out of range")
            job_posting_id = context.task.saved_job_candidates[
                job_selection_index - 1
            ].job_posting_id
        resume_version_id = context.task.active_resume_version_id
        if resume_selection_index is not None:
            if not 1 <= resume_selection_index <= len(context.task.resume_version_candidates):
                raise ValueError("resume-version selection index is out of range")
            resume_version_id = context.task.resume_version_candidates[
                resume_selection_index - 1
            ].resume_version_id
        if job_posting_id is None:
            raise ValueError("create_application requires a selected or active job")
        payload["job_posting_id"] = job_posting_id
        payload["resume_version_id"] = resume_version_id
    if name in {"update_application_status", "get_application"}:
        application_id = payload.get("application_id")
        selection_index = payload.pop("selection_index", None)
        if application_id is None and selection_index is not None:
            if not 1 <= selection_index <= len(context.task.application_candidates):
                raise ValueError("application selection index is out of range")
            application_id = context.task.application_candidates[
                selection_index - 1
            ].application_id
        application_id = application_id or context.task.active_application_id
        if application_id is None:
            raise ValueError(f"{name} requires an active application")
        payload["application_id"] = application_id
    return {"user_id": context.profile.user_id, **payload}

