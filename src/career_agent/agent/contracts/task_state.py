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
    CAPABILITIES,
    DOMAIN_TOOL_PROFILES,
    TOOL_PROFILE_NAMES,
    ToolProfile,
)
from career_agent.agent.contracts.questionnaire import PendingQuestionnaire, UserQuestion
from career_agent.domain.action_center import ActionSourceType, ActionStatus, ActionType
from career_agent.domain.applications import ApplicationStatus
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
from career_agent.agent.contracts.interactions import *
from career_agent.agent.contracts.interactions import (
    _PENDING_CONFIRMATION_TYPES,
    _PENDING_KIND_BY_SLOT,
    _PENDING_SLOT_BY_KIND,
)
from career_agent.agent.contracts.resources import ConversationResourceReference
from career_agent.agent.contracts.domain_context import DomainTaskContext


class WorkflowStateBase(ContractModel):
    run_id: str = Field(min_length=1)
    phase: str | None = None
    entry_message: str | None = None
    entry_resource_refs: tuple[ConversationResourceReference, ...] = ()
    entry_at: datetime | None = None


class JobDiscoveryWorkflowState(WorkflowStateBase):
    kind: Literal["job_discovery"] = "job_discovery"
    selected_result_ref: str | None = None
    manual_search_query: str | None = None
    candidates: tuple[CandidateContextItem, ...] = ()


class MockInterviewWorkflowState(WorkflowStateBase):
    kind: Literal["mock_interview"] = "mock_interview"


WorkflowState = Annotated[
    JobDiscoveryWorkflowState | MockInterviewWorkflowState,
    Field(discriminator="kind"),
]

ACTIVE_RESOURCE_ID_FIELDS = (
    "active_application_id",
    "active_interview_round_id",
    "active_interview_preparation_id",
    "active_action_item_id",
    "active_calendar_proposal_id",
    "active_job_posting_id",
    "active_jd_snapshot_id",
    "active_job_analysis_id",
    "active_job_analysis_jd_snapshot_id",
    "active_job_research_run_id",
    "active_job_research_report_id",
    "active_resume_job_match_id",
    "active_resume_tailoring_draft_id",
    "active_resume_version_id",
    "active_resume_artifact_id",
)


class RouteToCapabilityToolArguments(ContractModel):
    domain: ToolProfile = Field(
        description=(
            "The capability domain the user's current request belongs to. "
            "Choose core to leave a domain once its work is finished."
        ),
    )


class ConversationTaskState(ContractModel):
    """Durable per-conversation task state.

    ``active_workflow`` names the one multi-turn workflow that currently holds a
    suspended run, and ``run_id``/``phase``/``selected_result_ref``/
    ``manual_search_query`` are scoped to that workflow alone. Single-turn
    capabilities must not touch the slot: they finish inside one turn and have
    no run to resume, so claiming it would silently discard a workflow the user
    is still in the middle of. Use ``enter_workflow``/``leave_workflow`` rather
    than updating the fields piecemeal.

    ``tool_profile`` is a separate axis: which fixed group of tools the next
    decision is made against. It answers "what domain is the user working in",
    not "is a run suspended", so a resume-tailoring turn changes it while
    ``active_workflow`` stays ``none``. It is switched only through
    ``route_to_capability`` and persists across turns so a domain is routed
    into once, not on every decision. High-confidence ingress keyword routing
    may select the same profile before the first decision; it never selects an
    action, so ambiguous requests still go through ``route_to_capability``.

    ``pending_interaction`` is the only durable wait-for-user slot. Its
    discriminator prevents a questionnaire and a confirmation proposal, or two
    different proposals, from being active at the same time.

    ``domain_context`` keeps bounded references grouped by their owning domain.
    Compatibility properties retain the former flat read API while persisted
    state has one authoritative location for each migrated field.
    """

    workflow: WorkflowState | None = None
    tool_profile: ToolProfile = "core"
    loaded_capabilities: tuple[str, ...] = ()
    pending_interaction: PendingInteraction | None = None
    domain_context: DomainTaskContext = Field(default_factory=DomainTaskContext)

    @field_validator("loaded_capabilities", mode="before")
    @classmethod
    def _known_loaded_capabilities(cls, value: Any) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple, set, frozenset)):
            return ()
        known = {name for name in value if isinstance(name, str)}
        return tuple(
            name for name, descriptor in CAPABILITIES.items()
            if descriptor.model_callable and name in known
        )

    def add_loaded_capabilities(self, names: tuple[str, ...]) -> "ConversationTaskState":
        known = set(self.loaded_capabilities) | set(names)
        ordered = tuple(
            name for name, descriptor in CAPABILITIES.items()
            if descriptor.model_callable and name in known
        )
        return self.model_copy(update={"loaded_capabilities": ordered})

    @model_validator(mode="before")
    @classmethod
    def _drop_removed_fields(cls, value: Any) -> Any:
        # Resume fact extraction was removed; a stored task may still carry its
        # two slots, and the conversation must stay listable to be deleted.
        if isinstance(value, dict):
            value = dict(value)
            value.pop("active_resume_analysis_id", None)
            value.pop("resume_analysis_status", None)
            value = cls._migrate_domain_context(value)
            value = cls._migrate_pending_interaction(value)
            value = cls._migrate_workflow(value)
        return value

    @classmethod
    def _migrate_domain_context(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Fold former flat domain fields into their owned contexts."""

        raw = value.get("domain_context")
        if isinstance(raw, DomainTaskContext):
            domain = raw.model_dump(mode="python")
        elif isinstance(raw, Mapping):
            domain = dict(raw)
        else:
            domain = {}
        raw_application = domain.get("application")
        application = (
            dict(raw_application) if isinstance(raw_application, Mapping) else {}
        )
        aliases = {
            "active_application_id": "active_id",
            "active_application_status": "active_status",
            "application_candidates": "candidates",
            "email_event_candidates": "email_event_candidates",
            "email_sync_phase": "email_sync_phase",
        }
        for legacy, current in aliases.items():
            legacy_value = value.pop(legacy, None)
            if current not in application and legacy_value is not None:
                application[current] = legacy_value
        if application:
            domain["application"] = application
        cls._fold_legacy_domain(
            value,
            domain,
            "interview",
            {
                "active_interview_round_id": "active_round_id",
                "interview_candidates": "candidates",
                "active_interview_preparation_id": "active_preparation_id",
                "active_calendar_proposal_id": "active_calendar_proposal_id",
                "active_calendar_proposal_expires_at": "calendar_proposal_expires_at",
                "calendar_account_candidates": "calendar_account_candidates",
            },
        )
        cls._fold_legacy_domain(
            value,
            domain,
            "action_center",
            {
                "active_action_item_id": "active_id",
                "action_candidates": "candidates",
            },
        )
        cls._fold_legacy_domain(
            value,
            domain,
            "job",
            {
                "active_job_posting_id": "active_posting_id",
                "active_jd_snapshot_id": "active_jd_snapshot_id",
                "active_saved_job": "active_saved_job",
                "saved_job_candidates": "saved_job_candidates",
                "target_role_candidates": "target_role_candidates",
                "active_job_analysis_id": "active_analysis_id",
                "active_job_analysis_jd_snapshot_id": "active_analysis_jd_snapshot_id",
                "job_analysis_status": "analysis_status",
                "active_job_research_run_id": "active_research_run_id",
                "active_job_research_report_id": "active_research_report_id",
                "job_research_status": "research_status",
            },
        )
        cls._fold_legacy_domain(
            value,
            domain,
            "resume",
            {
                "active_resume_job_match_id": "active_job_match_id",
                "resume_job_match_status": "job_match_status",
                "active_resume_tailoring_draft_id": "active_tailoring_draft_id",
                "resume_tailoring_status": "tailoring_status",
                "active_resume_version_id": "active_version_id",
                "active_resume_artifact_id": "active_artifact_id",
                "resume_candidates": "candidates",
                "resume_version_candidates": "version_candidates",
            },
        )
        if domain:
            value["domain_context"] = domain
        return value

    @staticmethod
    def _fold_legacy_domain(
        value: dict[str, Any],
        domain: dict[str, Any],
        name: str,
        aliases: Mapping[str, str],
    ) -> None:
        raw_context = domain.get(name)
        if isinstance(raw_context, Mapping):
            context = dict(raw_context)
        elif hasattr(raw_context, "model_dump"):
            context = raw_context.model_dump(mode="python")
        else:
            context = {}
        for legacy, current in aliases.items():
            legacy_value = value.pop(legacy, None)
            if current not in context and legacy_value is not None:
                context[current] = legacy_value
        if context:
            domain[name] = context

    @classmethod
    def _migrate_workflow(cls, value: dict[str, Any]) -> dict[str, Any]:
        active = value.pop("active_workflow", "none")
        run_id = value.pop("run_id", None)
        phase = value.pop("phase", None)
        selected = value.pop("selected_result_ref", None)
        query = value.pop("manual_search_query", None)
        candidates = value.pop("candidates", ())
        entry_message = value.pop("workflow_entry_message", None)
        entry_refs = value.pop("workflow_entry_resource_refs", ())
        entry_at = value.pop("workflow_entry_at", None)
        if "workflow" in value:
            return value
        if (active == "none") != (run_id is None):
            raise ValueError(
                "active_workflow and run_id must be set together: a named "
                "workflow needs a run to resume, and a run needs an owner."
            )
        if active == "none":
            return value
        workflow: dict[str, Any] = {
            "kind": active,
            "run_id": run_id,
            "phase": phase,
            "entry_message": entry_message,
            "entry_resource_refs": entry_refs,
            "entry_at": entry_at,
        }
        if active == "job_discovery":
            workflow.update(
                {
                    "selected_result_ref": selected,
                    "manual_search_query": query,
                    "candidates": candidates,
                }
            )
        value["workflow"] = workflow
        return value

    @classmethod
    def _migrate_pending_interaction(
        cls, value: dict[str, Any]
    ) -> dict[str, Any]:
        """Read both former interaction representations safely.

        A unique most-recent proposal is safe to restore. If timestamps tie, no
        proposal is restored: choosing one arbitrarily could authorize a write
        the user was not currently confirming. A questionnaire wins over an old
        confirmation because it resumes a bound workflow; a proposal can safely
        be shown again.
        """
        legacy_questionnaire = value.pop("pending_questionnaire", None)
        current_confirmation = value.pop("pending_confirmation", None)
        legacy_stamps = value.pop("pending_proposed_at", {}) or {}
        bare_target = value.pop("bare_confirmation_target", None)
        candidates: list[tuple[PendingProposalSlot, Any, datetime | None]] = []
        for slot in PENDING_PROPOSAL_SLOTS:
            proposal = value.pop(slot, None)
            if proposal is not None:
                candidates.append((slot, proposal, legacy_stamps.get(slot)))
        if "pending_interaction" in value:
            return value
        if legacy_questionnaire is not None:
            value["pending_interaction"] = {
                "kind": "questionnaire",
                "questionnaire": legacy_questionnaire,
            }
            return value
        if current_confirmation is not None:
            value["pending_interaction"] = current_confirmation
            return value
        if not candidates:
            return value
        if len(candidates) == 1:
            selected = candidates[0]
        else:
            stamped = [item for item in candidates if item[2] is not None]
            if not stamped:
                return value
            newest = max(item[2] for item in stamped)
            latest = [item for item in stamped if item[2] == newest]
            if len(latest) != 1:
                return value
            selected = latest[0]
        slot, proposal, proposed_at = selected
        kind = _PENDING_KIND_BY_SLOT[slot]
        value["pending_interaction"] = {
            "kind": kind,
            "proposal": proposal,
            "proposed_at": proposed_at,
            "bare_confirmation": bare_target == kind,
        }
        return value

    @property
    def active_workflow(self) -> Literal["job_discovery", "mock_interview", "none"]:
        return self.workflow.kind if self.workflow is not None else "none"

    @property
    def active_application_id(self) -> str | None:
        return self.domain_context.application.active_id

    @property
    def active_application_status(self) -> ApplicationStatus | None:
        return self.domain_context.application.active_status

    @property
    def application_candidates(self) -> tuple[ApplicationCandidateContextItem, ...]:
        return self.domain_context.application.candidates

    @property
    def email_event_candidates(self) -> tuple[EmailEventCandidateContextItem, ...]:
        return self.domain_context.application.email_event_candidates

    @property
    def email_sync_phase(self) -> str | None:
        return self.domain_context.application.email_sync_phase

    def update_application_context(self, **updates: Any) -> "ConversationTaskState":
        application = self.domain_context.application.model_copy(update=updates)
        return self.model_copy(
            update={
                "domain_context": self.domain_context.model_copy(
                    update={"application": application}
                )
            }
        )

    @property
    def active_job_posting_id(self) -> str | None:
        return self.domain_context.job.active_posting_id

    @property
    def active_jd_snapshot_id(self) -> str | None:
        return self.domain_context.job.active_jd_snapshot_id

    @property
    def active_saved_job(self) -> ActiveSavedJobContextItem | None:
        return self.domain_context.job.active_saved_job

    @property
    def saved_job_candidates(self) -> tuple[SavedJobCandidateContextItem, ...]:
        return self.domain_context.job.saved_job_candidates

    @property
    def target_role_candidates(self) -> tuple[TargetRoleCandidateContextItem, ...]:
        return self.domain_context.job.target_role_candidates

    @property
    def active_job_analysis_id(self) -> str | None:
        return self.domain_context.job.active_analysis_id

    @property
    def active_job_analysis_jd_snapshot_id(self) -> str | None:
        return self.domain_context.job.active_analysis_jd_snapshot_id

    @property
    def job_analysis_status(self) -> Literal["ready"] | None:
        return self.domain_context.job.analysis_status

    @property
    def active_job_research_run_id(self) -> str | None:
        return self.domain_context.job.active_research_run_id

    @property
    def active_job_research_report_id(self) -> str | None:
        return self.domain_context.job.active_research_report_id

    @property
    def job_research_status(self) -> Literal["current", "outdated", "failed"] | None:
        return self.domain_context.job.research_status

    def update_job_context(self, **updates: Any) -> "ConversationTaskState":
        job = self.domain_context.job.model_copy(update=updates)
        return self.model_copy(
            update={
                "domain_context": self.domain_context.model_copy(update={"job": job})
            }
        )

    @property
    def active_resume_job_match_id(self) -> str | None:
        return self.domain_context.resume.active_job_match_id

    @property
    def resume_job_match_status(self) -> Literal["ready"] | None:
        return self.domain_context.resume.job_match_status

    @property
    def active_resume_tailoring_draft_id(self) -> str | None:
        return self.domain_context.resume.active_tailoring_draft_id

    @property
    def resume_tailoring_status(self) -> Literal[
        "pending", "in_review", "reviewed", "finalized", "superseded"
    ] | None:
        return self.domain_context.resume.tailoring_status

    @property
    def active_resume_version_id(self) -> str | None:
        return self.domain_context.resume.active_version_id

    @property
    def active_resume_artifact_id(self) -> str | None:
        return self.domain_context.resume.active_artifact_id

    @property
    def resume_candidates(self) -> tuple[ResumeCandidateContextItem, ...]:
        return self.domain_context.resume.candidates

    @property
    def resume_version_candidates(self) -> tuple[ResumeVersionCandidateContextItem, ...]:
        return self.domain_context.resume.version_candidates

    def update_resume_context(self, **updates: Any) -> "ConversationTaskState":
        resume = self.domain_context.resume.model_copy(update=updates)
        return self.model_copy(
            update={
                "domain_context": self.domain_context.model_copy(
                    update={"resume": resume}
                )
            }
        )

    @property
    def active_interview_round_id(self) -> str | None:
        return self.domain_context.interview.active_round_id

    @property
    def interview_candidates(self) -> tuple[InterviewCandidateContextItem, ...]:
        return self.domain_context.interview.candidates

    @property
    def active_interview_preparation_id(self) -> str | None:
        return self.domain_context.interview.active_preparation_id

    @property
    def active_calendar_proposal_id(self) -> str | None:
        return self.domain_context.interview.active_calendar_proposal_id

    @property
    def active_calendar_proposal_expires_at(self) -> datetime | None:
        return self.domain_context.interview.calendar_proposal_expires_at

    @property
    def calendar_account_candidates(
        self,
    ) -> tuple[CalendarAccountCandidateContextItem, ...]:
        return self.domain_context.interview.calendar_account_candidates

    def update_interview_context(self, **updates: Any) -> "ConversationTaskState":
        interview = self.domain_context.interview.model_copy(update=updates)
        return self.model_copy(
            update={
                "domain_context": self.domain_context.model_copy(
                    update={"interview": interview}
                )
            }
        )

    @property
    def active_action_item_id(self) -> str | None:
        return self.domain_context.action_center.active_id

    @property
    def action_candidates(self) -> tuple[ActionCandidateContextItem, ...]:
        return self.domain_context.action_center.candidates

    def update_action_center_context(self, **updates: Any) -> "ConversationTaskState":
        action_center = self.domain_context.action_center.model_copy(update=updates)
        return self.model_copy(
            update={
                "domain_context": self.domain_context.model_copy(
                    update={"action_center": action_center}
                )
            }
        )

    @property
    def run_id(self) -> str | None:
        return self.workflow.run_id if self.workflow is not None else None

    @property
    def phase(self) -> str | None:
        return self.workflow.phase if self.workflow is not None else None

    @property
    def selected_result_ref(self) -> str | None:
        workflow = self.workflow
        return (
            workflow.selected_result_ref
            if isinstance(workflow, JobDiscoveryWorkflowState)
            else None
        )

    @property
    def manual_search_query(self) -> str | None:
        workflow = self.workflow
        return (
            workflow.manual_search_query
            if isinstance(workflow, JobDiscoveryWorkflowState)
            else None
        )

    @property
    def candidates(self) -> tuple[CandidateContextItem, ...]:
        workflow = self.workflow
        return (
            workflow.candidates
            if isinstance(workflow, JobDiscoveryWorkflowState)
            else ()
        )

    @property
    def workflow_entry_message(self) -> str | None:
        return self.workflow.entry_message if self.workflow is not None else None

    @property
    def workflow_entry_resource_refs(
        self,
    ) -> tuple[ConversationResourceReference, ...]:
        return self.workflow.entry_resource_refs if self.workflow is not None else ()

    @property
    def workflow_entry_at(self) -> datetime | None:
        return self.workflow.entry_at if self.workflow is not None else None

    @model_validator(mode="after")
    def _validate_saved_job_focus(self) -> "ConversationTaskState":
        focus = self.active_saved_job
        if focus is None:
            return self
        if focus.jd_snapshot_id != self.active_jd_snapshot_id:
            raise ValueError("active_saved_job must pin active_jd_snapshot_id")
        return self

    def focused_saved_job(self) -> ActiveSavedJobContextItem | None:
        """The pinned JD, only while the posting it belongs to is still active.

        Every reducer that moves ``active_job_posting_id`` to another posting
        would otherwise have to remember to clear the pin; checking the pair
        here means a stale pin is simply not used.
        """
        focus = self.active_saved_job
        if focus is None or focus.job_posting_id != self.active_job_posting_id:
            return None
        return focus

    @property
    def pending_job_intent_update(self) -> JobIntentUpdate | None:
        return self._pending_proposal("pending_job_intent_update")

    @property
    def pending_free_text_preference(
        self,
    ) -> FreeTextPreferenceConfirmationProposal | None:
        return self._pending_proposal("pending_free_text_preference")

    @property
    def pending_memory_amendment(self) -> MemoryAmendmentProposal | None:
        return self._pending_proposal("pending_memory_amendment")

    @property
    def pending_memory_tombstone(self) -> MemoryTombstoneProposal | None:
        return self._pending_proposal("pending_memory_tombstone")

    @property
    def pending_career_fact(self) -> CareerFactProposal | None:
        return self._pending_proposal("pending_career_fact")

    @property
    def pending_constraint_retirement(self) -> ConstraintRetirementProposal | None:
        return self._pending_proposal("pending_constraint_retirement")

    @property
    def pending_questionnaire(self) -> PendingQuestionnaire | None:
        pending = self.pending_interaction
        if pending is None or pending.kind != "questionnaire":
            return None
        return pending.questionnaire

    @property
    def pending_confirmation(self) -> PendingConfirmation | None:
        pending = self.pending_interaction
        if pending is None or pending.kind == "questionnaire":
            return None
        return pending

    @property
    def pending_proposed_at(self) -> dict[PendingProposalSlot, datetime]:
        pending = self.pending_confirmation
        if pending is None or pending.proposed_at is None:
            return {}
        return {_PENDING_SLOT_BY_KIND[pending.kind]: pending.proposed_at}

    @property
    def bare_confirmation_target(
        self,
    ) -> Literal["career_fact", "job_intent", "free_text_preference"] | None:
        pending = self.pending_confirmation
        if (
            pending is None
            or not pending.bare_confirmation
            or pending.kind
            not in {"career_fact", "job_intent", "free_text_preference"}
        ):
            return None
        return pending.kind

    def _pending_proposal(self, slot: PendingProposalSlot) -> Any:
        pending = self.pending_confirmation
        if pending is None or pending.kind != _PENDING_KIND_BY_SLOT[slot]:
            return None
        return pending.proposal

    def with_pending_proposal(
        self,
        slot: PendingProposalSlot,
        proposal: Any,
        *,
        bare_confirmation: bool = False,
        proposed_at: datetime | None = None,
    ) -> "ConversationTaskState":
        """Replace the one pending confirmation; two cannot coexist by type."""
        kind = _PENDING_KIND_BY_SLOT[slot]
        pending_type = _PENDING_CONFIRMATION_TYPES[kind]
        pending = pending_type(
            proposal=proposal,
            proposed_at=proposed_at,
            bare_confirmation=bare_confirmation,
        )
        return self.model_copy(update={"pending_interaction": pending})

    def with_pending_questionnaire(
        self, questionnaire: PendingQuestionnaire
    ) -> "ConversationTaskState":
        return self.model_copy(
            update={
                "pending_interaction": PendingQuestionnaireInteraction(
                    questionnaire=questionnaire
                )
            }
        )

    def clear_pending_questionnaire(self) -> "ConversationTaskState":
        if self.pending_questionnaire is None:
            return self
        return self.model_copy(update={"pending_interaction": None})

    def clear_pending_proposal(
        self, slot: PendingProposalSlot
    ) -> "ConversationTaskState":
        if self._pending_proposal(slot) is None:
            return self
        return self.model_copy(update={"pending_interaction": None})

    def disarm_bare_confirmation(self) -> "ConversationTaskState":
        pending = self.pending_confirmation
        if pending is None or not pending.bare_confirmation:
            return self
        return self.model_copy(
            update={
                "pending_interaction": pending.model_copy(
                    update={"bare_confirmation": False}
                )
            }
        )

    def focus_saved_job(
        self, focus: ActiveSavedJobContextItem | None
    ) -> "ConversationTaskState":
        analysis_matches = bool(
            focus is not None
            and self.active_job_analysis_jd_snapshot_id == focus.jd_snapshot_id
        )
        return self.update_job_context(
            active_posting_id=(
                focus.job_posting_id
                if focus is not None
                else self.active_job_posting_id
            ),
            active_jd_snapshot_id=(
                focus.jd_snapshot_id if focus is not None else None
            ),
            active_saved_job=focus,
            active_analysis_id=(
                self.active_job_analysis_id if analysis_matches else None
            ),
            active_analysis_jd_snapshot_id=(
                self.active_job_analysis_jd_snapshot_id
                if analysis_matches
                else None
            ),
            analysis_status=(self.job_analysis_status if analysis_matches else None),
        )

    def pending_proposal_is_live(
        self, slot: PendingProposalSlot, now: datetime
    ) -> bool:
        """Whether ``slot`` holds a proposal shown within the TTL.

        An unstamped proposal predates stamping and its age is unknown, so it
        is treated as expired rather than granted a fresh clock.
        """
        pending = self.pending_confirmation
        proposed_at = self.pending_proposed_at.get(slot)
        return (
            pending is not None
            and pending.kind == _PENDING_KIND_BY_SLOT[slot]
            and proposed_at is not None
            and now - proposed_at <= PENDING_PROPOSAL_TTL
        )

    def stamp_new_proposals(
        self, before: "ConversationTaskState", now: datetime
    ) -> "ConversationTaskState":
        """Start the clock for every proposal a reducer just put in a slot.

        Identity, not equality, decides "just put": a reducer builds a fresh
        proposal from the tool payload, while ``model_copy`` carries untouched
        slots by reference. Re-showing an identical proposal therefore restarts
        its clock, and a reducer that keeps an unrelated slot leaves it alone.
        Stamps for emptied slots are dropped in the same pass.
        """
        pending = self.pending_confirmation
        if pending is None:
            return self
        if pending is before.pending_confirmation:
            return self
        return self.model_copy(
            update={
                "pending_interaction": pending.model_copy(
                    update={"proposed_at": now}
                )
            }
        )

    def expire_stale_proposals(
        self, now: datetime
    ) -> tuple["ConversationTaskState", tuple[PendingProposalSlot, ...]]:
        """Drop proposals left unconfirmed past ``PENDING_PROPOSAL_TTL``."""
        pending = self.pending_confirmation
        if pending is None:
            return self, ()
        slot = _PENDING_SLOT_BY_KIND[pending.kind]
        if self.pending_proposal_is_live(slot, now):
            return self, ()
        return self.model_copy(update={"pending_interaction": None}), (slot,)

    def enter_workflow(
        self,
        workflow: Literal["job_discovery", "mock_interview"],
        *,
        run_id: str,
        phase: str | None = None,
        selected_result_ref: str | None = None,
        manual_search_query: str | None = None,
        candidates: tuple[CandidateContextItem, ...] | None = None,
    ) -> "ConversationTaskState":
        """Claim the workflow slot, replacing any previous occupant's scope.

        Every scoped field is written in one copy so a new workflow can never
        inherit a stale phase or selection from the one it displaced. The held
        request survives re-entry by the same workflow, which is how a run
        advances its phase, but never crosses to a different one.
        """
        current = self.workflow if self.active_workflow == workflow else None
        common = {
            "run_id": run_id,
            "phase": phase,
            "entry_message": current.entry_message if current is not None else None,
            "entry_resource_refs": (
                current.entry_resource_refs if current is not None else ()
            ),
            "entry_at": current.entry_at if current is not None else None,
        }
        next_workflow: WorkflowState
        if workflow == "job_discovery":
            next_workflow = JobDiscoveryWorkflowState(
                **common,
                selected_result_ref=selected_result_ref,
                manual_search_query=manual_search_query,
                candidates=self.candidates if candidates is None else candidates,
            )
        else:
            next_workflow = MockInterviewWorkflowState(**common)
        return self.model_copy(update={"workflow": next_workflow})

    def active_resource_flags(self) -> dict[str, bool]:
        """Whether each active object exists, without naming any of them.

        The explicit projection boundary is intentional: domain context may
        retain opaque identifiers, while the model receives only booleans. A
        test audits every compatibility reference against this allowlist so a
        newly introduced active object cannot silently lose its existence flag.
        """
        return {
            f"has_{name[: -len('_id')]}": getattr(self, name) is not None
            for name in ACTIVE_RESOURCE_ID_FIELDS
        }

    def hold_entry_message(
        self,
        message: str,
        resource_refs: tuple["ConversationResourceReference", ...] = (),
        at: datetime | None = None,
    ) -> "ConversationTaskState":
        """Keep the request a multi-turn workflow has not answered yet.

        The workflow's own turns are not written to the conversation, so the
        reply to this request only exists once the run ends. Holding it here
        keeps the request and its reply in one write instead of leaving the
        conversation mid-exchange for as long as the run lasts.
        """
        if self.workflow is None:
            raise ValueError("cannot hold an entry message without an active workflow")
        return self.model_copy(
            update={
                "workflow": self.workflow.model_copy(
                    update={
                        "entry_message": message,
                        "entry_resource_refs": resource_refs,
                        "entry_at": at,
                    }
                )
            }
        )

    def clear_workflow_entry_message(self) -> "ConversationTaskState":
        if self.workflow is None or self.workflow.entry_message is None:
            return self
        return self.model_copy(
            update={
                "workflow": self.workflow.model_copy(
                    update={
                        "entry_message": None,
                        "entry_resource_refs": (),
                        "entry_at": None,
                    }
                )
            }
        )

    def leave_workflow(self) -> "ConversationTaskState":
        """Release the slot and clear the scoped fields together.

        The held request is scoped to the occupant too. Whoever releases the
        slot has already paired it with a closing reply, so carrying it forward
        would let the next workflow answer this one's opening line.
        """
        return self.model_copy(update={"workflow": None})
