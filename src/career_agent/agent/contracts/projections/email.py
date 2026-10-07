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

from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.contracts.tools.application import *
from career_agent.agent.contracts.decisions import _reject_internal_identifiers

def project_email_arguments(
    context: MainAgentContext, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    _reject_internal_identifiers(name, arguments)
    if name == "sync_application_emails":
        model_arguments = SyncApplicationEmailsToolArguments.model_validate(arguments)
    elif name == "list_email_events":
        model_arguments = ListEmailEventsToolArguments.model_validate(arguments)
    elif name == "resolve_email_event":
        model_arguments = ResolveEmailEventToolArguments.model_validate(arguments)
    else:
        raise ValueError(f"Unknown email tool: {name}")
    payload = model_arguments.model_dump()
    if name == "resolve_email_event":
        selection_index = payload.pop("selection_index", None)
        if selection_index is not None:
            if not 1 <= selection_index <= len(context.task.email_event_candidates):
                raise ValueError("email-event selection index is out of range")
            payload["event_id"] = context.task.email_event_candidates[
                selection_index - 1
            ].email_event_id
        if payload.get("event_id") is None:
            raise ValueError("resolve_email_event requires a selected email event")
    if name == "resolve_email_event" and payload.get("application_id") is None:
        payload["application_id"] = context.task.active_application_id
    if name == "resolve_email_event" and payload.get("interview_round_id") is None:
        payload["interview_round_id"] = context.task.active_interview_round_id
    return {"user_id": context.profile.user_id, **payload}

