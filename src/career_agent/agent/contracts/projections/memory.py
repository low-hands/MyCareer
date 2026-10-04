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
from career_agent.agent.contracts.interactions import pending_confirmation_proposal
from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.contracts.tools.core_memory import *
from career_agent.agent.contracts.tools.job import *
from career_agent.agent.contracts.decisions import _reject_internal_identifiers

def project_job_intent_arguments(
    context: MainAgentContext,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    _reject_internal_identifiers(name, arguments)
    if name == "propose_job_intent":
        model_arguments = ProposeJobIntentToolArguments.model_validate(arguments)
        payload = model_arguments.model_dump(exclude_none=True)
        selection_index = payload.pop("target_role_selection_index", None)
        if selection_index is not None:
            candidates = context.task.target_role_candidates
            if not 1 <= selection_index <= len(candidates):
                raise ValueError("target-role selection index is out of range")
            payload["target_role_id"] = candidates[selection_index - 1].target_role_id
        update = JobIntentUpdate.model_validate(payload)
        return {
            "user_id": context.profile.user_id,
            "update": update,
            "current": context.profile,
        }
    ConfirmJobIntentToolArguments.model_validate(arguments)
    pending = pending_confirmation_proposal(context.task, name)
    return {
        "user_id": context.profile.user_id,
        "conversation_id": context.conversation_id,
        "update": pending,
        "current": context.profile,
    }


def project_free_text_preference_arguments(
    context: MainAgentContext,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    _reject_internal_identifiers(name, arguments)
    if name == "propose_free_text_preference_confirmation":
        model_arguments = (
            ProposeFreeTextPreferenceConfirmationToolArguments.model_validate(
                arguments
            )
        )
        candidates = tuple(
            item
            for item in context.free_text_preferences
            if item.status == "quarantined"
        )
        index = model_arguments.selection_index
        if not 1 <= index <= len(candidates):
            raise ValueError("free-text preference selection index is out of range")
        candidate = candidates[index - 1]
        return {
            "user_id": context.profile.user_id,
            "proposal": FreeTextPreferenceConfirmationProposal(
                update_id=candidate.update_id,
                topic_key=candidate.topic_key,
                statement=candidate.statement,
                ownership=candidate.ownership,
                pref_scope=candidate.pref_scope,
                needs_scope_clarification=candidate.needs_scope_clarification,
            ),
        }
    confirmation = ConfirmFreeTextPreferenceToolArguments.model_validate(
        arguments
    )
    pending = pending_confirmation_proposal(context.task, name)
    if (
        pending.needs_scope_clarification
        and confirmation.scope_choice is None
    ):
        raise ValueError(
            "this preference needs person_default or a named role scope"
        )
    return {
        "user_id": context.profile.user_id,
        "update_id": pending.update_id,
        "proposal": pending,
        "conversation_id": context.conversation_id,
        "job_posting_id": context.task.active_job_posting_id,
        "scope_choice": confirmation.scope_choice,
        "scope_domain": confirmation.scope_domain,
    }


def project_memory_tombstone_arguments(
    context: MainAgentContext,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    _reject_internal_identifiers(name, arguments)
    if name == "propose_memory_tombstone":
        model_arguments = ProposeMemoryTombstoneToolArguments.model_validate(
            arguments
        )
        return {
            "user_id": context.profile.user_id,
            "proposal": MemoryTombstoneProposal(
                target_kind="career_evidence",
                detail_ref=model_arguments.detail_ref,
                reason=model_arguments.reason,
            ),
        }
    ConfirmMemoryTombstoneToolArguments.model_validate(arguments)
    pending = pending_confirmation_proposal(context.task, name)
    return {
        "user_id": context.profile.user_id,
        "conversation_id": context.conversation_id,
        "proposal": pending.model_dump(mode="json"),
    }


def project_memory_amendment_arguments(
    context: MainAgentContext,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    _reject_internal_identifiers(name, arguments)
    if name == "propose_memory_amendment":
        model_arguments = ProposeMemoryAmendmentToolArguments.model_validate(
            arguments
        )
        return {
            "user_id": context.profile.user_id,
            "proposal": MemoryAmendmentProposal(
                target_kind="career_evidence",
                detail_ref=model_arguments.detail_ref,
                new_claim=model_arguments.new_claim,
                reason=model_arguments.reason,
            ),
        }
    ConfirmMemoryAmendmentToolArguments.model_validate(arguments)
    pending = pending_confirmation_proposal(context.task, name)
    return {
        "user_id": context.profile.user_id,
        "conversation_id": context.conversation_id,
        "proposal": pending,
    }


def project_working_notes_arguments(
    context: MainAgentContext,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    _reject_internal_identifiers(name, arguments)
    model_arguments = UpdateWorkingNotesToolArguments.model_validate(arguments)
    return {
        "user_id": context.profile.user_id,
        "markdown": model_arguments.markdown,
        "expected_revision": model_arguments.expected_revision,
    }


def project_career_fact_arguments(
    context: MainAgentContext,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    _reject_internal_identifiers(name, arguments)
    if name == "propose_career_fact":
        proposed = ProposeCareerFactToolArguments.model_validate(arguments)
        projected = context.career_memory.tier_one_projection(
            token_budget=context.career_profile_budgets.records_input_units
        )
        records = projected.get("records", [])
        if not 1 <= proposed.record_selection_index <= len(records):
            raise ValueError("career record selection is not projected")
        record = context.career_memory.records[
            proposed.record_selection_index - 1
        ]
        if record.record_id is None:
            raise ValueError("career record selection has no durable identity")
        source_interaction_id = None
        if proposed.user_quote is not None:
            # A model-supplied reason is not provenance. Match the exact quote
            # against stored user speech, preferring a questionnaire answer
            # over a later message that merely repeats part of that answer.
            recent_user = tuple(
                message for message in reversed(context.recent_messages)
                if message.role == "user"
            )
            current = (context.stored_user_message(), context.user_interaction_id)
            sources = (
                ((current,) if context.user_interaction_id else ())
                + tuple((item.content, item.user_interaction_id) for item in recent_user
                        if item.user_interaction_id)
                + (() if context.user_interaction_id else (current,))
                + tuple((item.content, None) for item in recent_user
                        if not item.user_interaction_id)
            )
            matched_source = next(
                ((content, interaction_id) for content, interaction_id in sources
                 if proposed.user_quote in content), None,
            )
            if matched_source is None:
                raise ValueError("user_quote is absent from user messages")
            source_interaction_id = matched_source[1]
        return {
            "user_id": context.profile.user_id,
            "conversation_id": context.conversation_id,
            "career_record_id": record.record_id,
            "claim": proposed.claim,
            "reason": proposed.reason,
            "origin": "user_input" if proposed.user_quote is not None else "agent_inference",
            "source_user_quote": proposed.user_quote,
            "source_user_interaction_id": source_interaction_id,
        }
    ConfirmCareerFactToolArguments.model_validate(arguments)
    pending = pending_confirmation_proposal(context.task, name)
    return {
        "user_id": context.profile.user_id,
        "conversation_id": context.conversation_id,
        "proposal": pending,
    }


def project_constraint_retirement_arguments(
    context: MainAgentContext,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    _reject_internal_identifiers(name, arguments)
    if name == "fetch_archived_constraints":
        FetchArchivedConstraintsToolArguments.model_validate(arguments)
        return {
            "user_id": context.profile.user_id,
            "conversation_id": context.conversation_id,
        }
    if name == "propose_constraint_retirement":
        model_arguments = (
            ProposeConstraintRetirementToolArguments.model_validate(arguments)
        )
        return {
            "user_id": context.profile.user_id,
            "conversation_id": context.conversation_id,
            "proposal": ConstraintRetirementProposal(
                target_kind="conversation_constraint",
                constraint=model_arguments.constraint,
                reason=model_arguments.reason,
            ),
        }
    ConfirmConstraintRetirementToolArguments.model_validate(arguments)
    pending = pending_confirmation_proposal(context.task, name)
    return {
        "user_id": context.profile.user_id,
        "conversation_id": context.conversation_id,
        "proposal": pending.model_dump(mode="json"),
    }

