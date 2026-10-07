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

from career_agent.agent.contracts.profile import *

PendingProposalSlot = Literal[
    "pending_job_intent_update",
    "pending_free_text_preference",
    "pending_memory_amendment",
    "pending_memory_tombstone",
    "pending_career_fact",
    "pending_constraint_retirement",
]
PENDING_PROPOSAL_SLOTS: tuple[PendingProposalSlot, ...] = get_args(
    PendingProposalSlot
)


class PendingConfirmationBase(ContractModel):
    """One proposal shown to the user and awaiting confirmation."""

    proposed_at: datetime | None = None
    bare_confirmation: bool = False


class PendingQuestionnaireInteraction(ContractModel):
    kind: Literal["questionnaire"] = "questionnaire"
    questionnaire: PendingQuestionnaire


class PendingJobIntentConfirmation(PendingConfirmationBase):
    kind: Literal["job_intent"] = "job_intent"
    proposal: JobIntentUpdate


class PendingFreeTextPreferenceConfirmation(PendingConfirmationBase):
    kind: Literal["free_text_preference"] = "free_text_preference"
    proposal: FreeTextPreferenceConfirmationProposal


class PendingMemoryAmendmentConfirmation(PendingConfirmationBase):
    kind: Literal["memory_amendment"] = "memory_amendment"
    proposal: MemoryAmendmentProposal


class PendingMemoryTombstoneConfirmation(PendingConfirmationBase):
    kind: Literal["memory_tombstone"] = "memory_tombstone"
    proposal: MemoryTombstoneProposal


class PendingCareerFactConfirmation(PendingConfirmationBase):
    kind: Literal["career_fact"] = "career_fact"
    proposal: CareerFactProposal


class PendingConstraintRetirementConfirmation(PendingConfirmationBase):
    kind: Literal["constraint_retirement"] = "constraint_retirement"
    proposal: ConstraintRetirementProposal


PendingConfirmation = (
    PendingJobIntentConfirmation
    | PendingFreeTextPreferenceConfirmation
    | PendingMemoryAmendmentConfirmation
    | PendingMemoryTombstoneConfirmation
    | PendingCareerFactConfirmation
    | PendingConstraintRetirementConfirmation
)
PendingInteraction = Annotated[
    PendingQuestionnaireInteraction | PendingConfirmation,
    Field(discriminator="kind"),
]

_PENDING_KIND_BY_SLOT: dict[PendingProposalSlot, str] = {
    "pending_job_intent_update": "job_intent",
    "pending_free_text_preference": "free_text_preference",
    "pending_memory_amendment": "memory_amendment",
    "pending_memory_tombstone": "memory_tombstone",
    "pending_career_fact": "career_fact",
    "pending_constraint_retirement": "constraint_retirement",
}
_PENDING_SLOT_BY_KIND: dict[str, PendingProposalSlot] = {
    kind: slot for slot, kind in _PENDING_KIND_BY_SLOT.items()
}
_PENDING_CONFIRMATION_TYPES = {
    "job_intent": PendingJobIntentConfirmation,
    "free_text_preference": PendingFreeTextPreferenceConfirmation,
    "memory_amendment": PendingMemoryAmendmentConfirmation,
    "memory_tombstone": PendingMemoryTombstoneConfirmation,
    "career_fact": PendingCareerFactConfirmation,
    "constraint_retirement": PendingConstraintRetirementConfirmation,
}


class ConfirmationSpec(NamedTuple):
    slot: PendingProposalSlot
    missing_message: str
    proposed_state: str
    requires_seal: bool = False


CONFIRMATION_SPECS: dict[str, ConfirmationSpec] = {
    "confirm_job_intent": ConfirmationSpec(
        "pending_job_intent_update",
        "confirm_job_intent requires a proposed update the user has seen",
        "job_intent_proposed",
    ),
    "confirm_free_text_preference": ConfirmationSpec(
        "pending_free_text_preference",
        "confirm_free_text_preference requires a proposal the user has seen",
        "free_text_preference_confirmation_proposed",
    ),
    "confirm_memory_amendment": ConfirmationSpec(
        "pending_memory_amendment",
        "confirm_memory_amendment requires a proposed correction the user has seen",
        "memory_amendment_proposed",
    ),
    "confirm_memory_tombstone": ConfirmationSpec(
        "pending_memory_tombstone",
        "confirm_memory_tombstone requires a proposed deletion the user has seen",
        "memory_tombstone_proposed",
        True,
    ),
    "confirm_career_fact": ConfirmationSpec(
        "pending_career_fact",
        "confirm_career_fact requires a proposed fact the user has seen",
        "career_fact_proposed",
    ),
    "confirm_constraint_retirement": ConfirmationSpec(
        "pending_constraint_retirement",
        "confirm_constraint_retirement requires a proposed retirement the user has seen",
        "constraint_retirement_proposed",
        True,
    ),
}
# A confirmation authorizes what the user was shown. After a week "confirm that
# one" can no longer be assumed to mean a readback the user still remembers.
PENDING_PROPOSAL_TTL = timedelta(days=7)
_BARE_CONFIRMATION_SLOTS: dict[str, PendingProposalSlot] = {
    "career_fact": "pending_career_fact",
    "job_intent": "pending_job_intent_update",
    "free_text_preference": "pending_free_text_preference",
}


def expired_proposal_message(tool_name: str) -> str:
    return (
        f"{tool_name} refused: the proposal was shown more than "
        f"{PENDING_PROPOSAL_TTL.days} days ago and has expired; show the "
        "proposal again and wait for explicit agreement"
    )


def pending_confirmation_proposal(
    task: "ConversationTaskState", tool_name: str
) -> Any:
    """Resolve the single shown, live proposal authorized for a confirm tool."""
    spec = CONFIRMATION_SPECS[tool_name]
    slot = spec.slot
    proposal = getattr(task, slot)
    if proposal is None:
        raise ValueError(spec.missing_message)
    if not task.pending_proposal_is_live(slot, datetime.now(timezone.utc)):
        raise ValueError(expired_proposal_message(tool_name))
    return proposal


def confirmation_arguments_snapshot(
    task: "ConversationTaskState",
    tool_name: str,
    *,
    user_id: str,
    conversation_id: str,
) -> dict[str, Any]:
    """Capture the selected proposal as JSON before any confirmation route."""
    proposal = pending_confirmation_proposal(task, tool_name)
    payload_key = "update" if tool_name == "confirm_job_intent" else "proposal"
    return {
        "user_id": user_id,
        "conversation_id": conversation_id,
        payload_key: proposal.model_dump(mode="json"),
    }

