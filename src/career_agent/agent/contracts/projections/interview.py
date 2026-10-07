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

from career_agent.agent.contracts.candidates import SavedJobCandidateContextItem
from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.contracts.tools.interview import *
from career_agent.agent.contracts.tools.workflows import StartMockInterviewWorkflowInput
from career_agent.agent.contracts.decisions import _reject_internal_identifiers

def project_interview_arguments(
    context: MainAgentContext, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    _reject_internal_identifiers(name, arguments)
    if name == "list_interviews":
        model_arguments = ListInterviewsToolArguments.model_validate(arguments)
    elif name == "get_interview":
        model_arguments = GetInterviewToolArguments.model_validate(arguments)
    elif name == "create_interview":
        model_arguments = CreateInterviewToolArguments.model_validate(arguments)
    elif name == "update_interview":
        model_arguments = UpdateInterviewToolArguments.model_validate(arguments)
    elif name == "complete_interview":
        model_arguments = CompleteInterviewToolArguments.model_validate(arguments)
    elif name == "record_interview_retro":
        model_arguments = RecordInterviewRetroToolArguments.model_validate(arguments)
    else:
        raise ValueError(f"Unknown interview tool: {name}")
    payload = model_arguments.model_dump()
    if name == "create_interview":
        application_id = payload.get("application_id")
        selection_index = payload.pop("application_selection_index", None)
        if application_id is None and selection_index is not None:
            if not 1 <= selection_index <= len(context.task.application_candidates):
                raise ValueError("application selection index is out of range")
            application_id = context.task.application_candidates[
                selection_index - 1
            ].application_id
        application_id = application_id or context.task.active_application_id
        payload["application_id"] = application_id
        job_selection_index = payload.pop("job_selection_index", None)
        job_posting_id = context.task.active_job_posting_id
        if job_selection_index is not None:
            if not 1 <= job_selection_index <= len(context.task.saved_job_candidates):
                raise ValueError("saved-job selection index is out of range")
            job_posting_id = context.task.saved_job_candidates[
                job_selection_index - 1
            ].job_posting_id
        if application_id is None and job_posting_id is None:
            raise ValueError(
                "create_interview requires an active application or saved job"
            )
        payload["job_posting_id"] = job_posting_id
    if name in {
        "get_interview",
        "update_interview",
        "complete_interview",
        "record_interview_retro",
    }:
        interview_round_id = payload.get("interview_round_id")
        selection_index = payload.pop("selection_index", None)
        if interview_round_id is None and selection_index is not None:
            if not 1 <= selection_index <= len(context.task.interview_candidates):
                raise ValueError("interview selection index is out of range")
            interview_round_id = context.task.interview_candidates[
                selection_index - 1
            ].interview_round_id
        interview_round_id = interview_round_id or context.task.active_interview_round_id
        if interview_round_id is None:
            raise ValueError(f"{name} requires an active interview")
        payload["interview_round_id"] = interview_round_id
    return {"user_id": context.profile.user_id, **payload}


def project_interview_preparation_arguments(
    context: MainAgentContext, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    _reject_internal_identifiers(name, arguments)
    if name == "prepare_interview":
        model_arguments = PrepareInterviewToolArguments.model_validate(arguments)
        payload = model_arguments.model_dump()
        interview_round_id = payload.get("interview_round_id")
        selection_index = payload.pop("selection_index", None)
        action_selection_index = payload.pop("action_selection_index", None)
        if interview_round_id is None and selection_index is not None:
            if not 1 <= selection_index <= len(context.task.interview_candidates):
                raise ValueError("interview selection index is out of range")
            interview_round_id = context.task.interview_candidates[
                selection_index - 1
            ].interview_round_id
        if interview_round_id is None and action_selection_index is not None:
            if not 1 <= action_selection_index <= len(context.task.action_candidates):
                raise ValueError("action selection index is out of range")
            action = context.task.action_candidates[action_selection_index - 1]
            if (
                action.action_type != "interview_preparation"
                or action.source_type != "interview_round"
            ):
                raise ValueError("selected action is not interview preparation")
            interview_round_id = action.source_id
        interview_round_id = interview_round_id or context.task.active_interview_round_id
        if interview_round_id is None:
            raise ValueError("prepare_interview requires an active interview")
        payload["interview_round_id"] = interview_round_id
    elif name == "get_interview_preparation":
        model_arguments = GetInterviewPreparationToolArguments.model_validate(arguments)
        payload = model_arguments.model_dump()
        reference = payload.pop("reference", None)
        interview_index = payload.pop("interview_selection_index", None)
        preparation_id = None
        if reference is not None:
            preparation_id = context.resolve_reference(
                reference=reference,
                kind="interview_preparation",
            )
        elif interview_index is not None:
            if not 1 <= interview_index <= len(context.task.interview_candidates):
                raise ValueError("interview selection index is out of range")
            # Resolved to the interview, not to a preparation: which preparation
            # belongs to it is the store's answer, and the newest is the one a
            # candidate means by "the prep for that interview".
            payload["interview_round_id"] = context.task.interview_candidates[
                interview_index - 1
            ].interview_round_id
        else:
            preparation_id = (
                payload.get("preparation_id")
                or context.task.active_interview_preparation_id
            )
        if preparation_id is None and "interview_round_id" not in payload:
            raise ValueError("get_interview_preparation requires an active preparation")
        payload["preparation_id"] = preparation_id
    else:
        raise ValueError(f"Unknown interview preparation tool: {name}")
    return {"user_id": context.profile.user_id, **payload}


def project_mock_interview_arguments(
    context: MainAgentContext, arguments: dict[str, Any]
) -> dict[str, Any]:
    _reject_internal_identifiers("start_mock_interview", arguments)
    model_arguments = StartMockInterviewToolArguments.model_validate(arguments)
    application_id = (
        None
        if model_arguments.practice_scope == "free"
        else context.task.active_application_id
    )
    interview_round_id: str | None = None

    if model_arguments.practice_scope == "free" and (
        model_arguments.application_selection_index is not None
        or model_arguments.interview_selection_index is not None
    ):
        raise ValueError("free practice cannot select an application or interview")
    if model_arguments.practice_scope != "free" and (
        model_arguments.without_resume
        or model_arguments.resume_version_selection_index is not None
        or model_arguments.job_selection_index is not None
        or model_arguments.company_name is not None
        or model_arguments.without_job
    ):
        raise ValueError(
            "an application run uses its submitted resume and JD; resume, job and "
            "company choice is free practice only"
        )
    if model_arguments.application_selection_index is not None:
        index = model_arguments.application_selection_index
        if not 1 <= index <= len(context.task.application_candidates):
            raise ValueError("application selection index is out of range")
        application_id = context.task.application_candidates[index - 1].application_id
    elif model_arguments.interview_selection_index is not None:
        index = model_arguments.interview_selection_index
        if not 1 <= index <= len(context.task.interview_candidates):
            raise ValueError("interview selection index is out of range")
        candidate = context.task.interview_candidates[index - 1]
        application_id = candidate.application_id
        interview_round_id = candidate.interview_round_id

    if application_id is None and model_arguments.interview_type is None:
        raise ValueError("start_mock_interview requires interview_type for free practice")
    target_role = model_arguments.target_role
    if application_id is None and target_role is None and context.profile.current_targets:
        target_role = context.profile.current_targets[0].title
    if interview_round_id is None and context.task.active_interview_round_id is not None:
        active = next(
            (
                candidate
                for candidate in context.task.interview_candidates
                if candidate.interview_round_id
                == context.task.active_interview_round_id
                and candidate.application_id == application_id
            ),
            None,
        )
        if active is not None:
            interview_round_id = active.interview_round_id

    resume_choice: Literal["chosen", "none", "required"] = "chosen"
    resume_version_id: str | None = None
    job: SavedJobCandidateContextItem | None = None
    job_choice: Literal["chosen", "none", "check"] = "none"
    if application_id is None:
        resume_choice, resume_version_id = _free_practice_resume(context, model_arguments)
        job = _free_practice_job(context, model_arguments)
        job_choice = (
            "chosen"
            if job is not None
            else "check"
            if model_arguments.company_name is not None and not model_arguments.without_job
            else "none"
        )
        if job is not None and model_arguments.target_role is None:
            # The job's own title is the role; the profile's default target
            # role filled in above would contradict it.
            target_role = None

    return StartMockInterviewWorkflowInput(
        user_id=context.profile.user_id,
        application_id=application_id,
        interview_round_id=interview_round_id,
        interview_type=model_arguments.interview_type or "mixed",
        target_role=target_role,
        conversation_id=context.conversation_id,
        max_primary_questions=model_arguments.max_primary_questions,
        max_follow_ups_per_question=model_arguments.max_follow_ups_per_question,
        resume_choice=resume_choice,
        resume_version_id=resume_version_id,
        job_choice=job_choice,
        job_posting_id=job.job_posting_id if job is not None else None,
        jd_snapshot_id=job.jd_snapshot_id if job is not None else None,
        target_company=(
            job.company_name if job is not None else model_arguments.company_name
        ),
    ).model_dump()


def _free_practice_job(
    context: MainAgentContext, arguments: StartMockInterviewToolArguments
) -> SavedJobCandidateContextItem | None:
    """The saved job free practice is for: a pick from the offered list, or a
    single job attached to this message. Never inferred from the conversation.
    """
    index = arguments.job_selection_index
    if index is not None:
        if not 1 <= index <= len(context.task.saved_job_candidates):
            raise ValueError("saved-job selection index is out of range")
        return context.task.saved_job_candidates[index - 1]
    if len(context.attached_jobs) == 1:
        return context.attached_jobs[0]
    return None


def _free_practice_resume(
    context: MainAgentContext, arguments: StartMockInterviewToolArguments
) -> tuple[Literal["chosen", "none", "required"], str | None]:
    """Which resume free practice runs on, or ``required`` when nobody said.

    Only the user decides this: an explicit "no resume", a pick from the
    offered list, or a single resume attached to this very message. Anything
    else, including a resume the conversation touched earlier, asks again. The
    old fallback took the most recently updated resume, which silently ran a
    practice on a test fixture.
    """
    if arguments.without_resume:
        return "none", None
    index = arguments.resume_version_selection_index
    if index is not None:
        if not 1 <= index <= len(context.task.resume_version_candidates):
            raise ValueError("resume-version selection index is out of range")
        return "chosen", context.task.resume_version_candidates[index - 1].resume_version_id
    if len(context.attached_resumes) == 1:
        return "chosen", context.attached_resumes[0].resume_version_id
    return "required", None


def project_restart_mock_interview_arguments(
    context: MainAgentContext, arguments: dict[str, Any]
) -> dict[str, Any]:
    _reject_internal_identifiers("restart_mock_interview", arguments)
    RestartMockInterviewToolArguments.model_validate(arguments)
    # The stuck run is found by the harness-owned conversation scope, not named
    # by the model: users may have several unfinished runs in other chats, and
    # exposing a session id would let the model cross those boundaries.
    return {
        "user_id": context.profile.user_id,
        "conversation_id": context.conversation_id,
        **(
            {"session_id": context.task.run_id}
            if context.task.active_workflow == "mock_interview"
            and context.task.run_id is not None
            else {}
        ),
    }




def project_mock_interview_result_arguments(
    context: MainAgentContext, arguments: dict[str, Any]
) -> dict[str, Any]:
    _reject_internal_identifiers("get_mock_interview_result", arguments)
    model_arguments = GetMockInterviewResultToolArguments.model_validate(arguments)
    if model_arguments.reference is not None:
        # A report id names one exact run, so the application is not needed and
        # must not be sent: passing both would let the handler fall back to the
        # newest run for that application and quietly answer about a different
        # interview than the turn the user pointed at.
        return {
            "user_id": context.profile.user_id,
            "report_id": context.resolve_reference(
                reference=model_arguments.reference,
                kind="mock_interview_report",
            ),
            "question_number": model_arguments.question_number,
        }
    application_id = context.task.active_application_id
    if model_arguments.application_selection_index is not None:
        index = model_arguments.application_selection_index
        if not 1 <= index <= len(context.task.application_candidates):
            raise ValueError("application selection index is out of range")
        application_id = context.task.application_candidates[index - 1].application_id
    if application_id is None:
        raise ValueError("get_mock_interview_result requires an active application")
    return {
        "user_id": context.profile.user_id,
        "application_id": application_id,
        "question_number": model_arguments.question_number,
    }

