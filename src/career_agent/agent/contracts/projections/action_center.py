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
from career_agent.agent.contracts.tools.action_center import *
from career_agent.agent.contracts.decisions import _reject_internal_identifiers

def project_action_center_arguments(
    context: MainAgentContext, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    _reject_internal_identifiers(name, arguments)
    if name == "get_daily_brief":
        model_arguments = GetDailyBriefToolArguments.model_validate(arguments)
    elif name == "list_action_items":
        model_arguments = ListActionItemsToolArguments.model_validate(arguments)
    elif name in {"complete_action_item", "dismiss_action_item"}:
        model_arguments = ResolveActionItemToolArguments.model_validate(arguments)
    elif name == "snooze_action_item":
        model_arguments = SnoozeActionItemToolArguments.model_validate(arguments)
    else:
        raise ValueError(f"Unknown action-center tool: {name}")
    payload = model_arguments.model_dump()
    if name in {"complete_action_item", "dismiss_action_item", "snooze_action_item"}:
        action_item_id = payload.get("action_item_id")
        selection_index = payload.pop("selection_index", None)
        if action_item_id is None and selection_index is not None:
            if not 1 <= selection_index <= len(context.task.action_candidates):
                raise ValueError("action selection index is out of range")
            action_item_id = context.task.action_candidates[
                selection_index - 1
            ].action_item_id
        action_item_id = action_item_id or context.task.active_action_item_id
        if action_item_id is None:
            raise ValueError(f"{name} requires an active action item")
        payload["action_item_id"] = action_item_id
    return {"user_id": context.profile.user_id, **payload}

