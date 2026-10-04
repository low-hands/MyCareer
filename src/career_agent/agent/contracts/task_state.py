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
from career_agent.agent.contracts.candidates import *
from career_agent.agent.contracts.interactions import *
from career_agent.agent.contracts.interactions import (
    _PENDING_CONFIRMATION_TYPES,
    _PENDING_KIND_BY_SLOT,
    _PENDING_SLOT_BY_KIND,
)
from career_agent.agent.contracts.resources import ConversationResourceReference


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
    """

    workflow: WorkflowState | None = None
    tool_profile: ToolProfile = "core"
    pending_interaction: PendingInteraction | None = None
    active_resume_job_match_id: str | None = None
    resume_job_match_status: Literal["ready"] | None = None
    active_job_analysis_id: str | None = None
    active_job_analysis_jd_snapshot_id: str | None = None
    job_analysis_status: Literal["ready"] | None = None
    active_resume_tailoring_draft_id: str | None = None
    resume_tailoring_status: Literal[
        "pending", "in_review", "reviewed", "finalized", "superseded"
    ] | None = None
    active_resume_version_id: str | None = None
    active_resume_artifact_id: str | None = None
    active_job_posting_id: str | None = None
    active_jd_snapshot_id: str | None = None
    active_saved_job: ActiveSavedJobContextItem | None = None
    active_job_research_run_id: str | None = None
    active_job_research_report_id: str | None = None
    job_research_status: Literal["current", "outdated", "failed"] | None = None
    active_application_id: str | None = None
    active_application_status: ApplicationStatus | None = None
    application_candidates: tuple[ApplicationCandidateContextItem, ...] = ()
    active_interview_round_id: str | None = None
    interview_candidates: tuple[InterviewCandidateContextItem, ...] = ()
    active_interview_preparation_id: str | None = None
    active_action_item_id: str | None = None
    action_candidates: tuple[ActionCandidateContextItem, ...] = ()
    active_calendar_proposal_id: str | None = None
    # Projected to the model, unlike the id beside it: whether a preview is
    # still live decides between executing it and preparing a new one, and
    # a timestamp names no object the model could act on.
    active_calendar_proposal_expires_at: datetime | None = None
    calendar_account_candidates: tuple[CalendarAccountCandidateContextItem, ...] = ()
    saved_job_candidates: tuple[SavedJobCandidateContextItem, ...] = ()
    target_role_candidates: tuple[TargetRoleCandidateContextItem, ...] = ()
    resume_candidates: tuple[ResumeCandidateContextItem, ...] = ()
    resume_version_candidates: tuple[ResumeVersionCandidateContextItem, ...] = ()
    email_event_candidates: tuple[EmailEventCandidateContextItem, ...] = ()
    email_sync_phase: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _drop_removed_fields(cls, value: Any) -> Any:
        # Resume fact extraction was removed; a stored task may still carry its
        # two slots, and the conversation must stay listable to be deleted.
        if isinstance(value, dict):
            value = dict(value)
            value.pop("active_resume_analysis_id", None)
            value.pop("resume_analysis_status", None)
            value = cls._migrate_pending_interaction(value)
            value = cls._migrate_workflow(value)
        return value

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
        return self.model_copy(
            update={
                "active_job_posting_id": (
                    focus.job_posting_id
                    if focus is not None
                    else self.active_job_posting_id
                ),
                "active_jd_snapshot_id": (
                    focus.jd_snapshot_id if focus is not None else None
                ),
                "active_saved_job": focus,
                "active_job_analysis_id": (
                    self.active_job_analysis_id if analysis_matches else None
                ),
                "active_job_analysis_jd_snapshot_id": (
                    self.active_job_analysis_jd_snapshot_id
                    if analysis_matches
                    else None
                ),
                "job_analysis_status": (
                    self.job_analysis_status if analysis_matches else None
                ),
            }
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

        Derived from the ``active_*_id`` field names rather than listed by hand.
        Thirteen such fields existed and none reached the model: the projection
        withheld the ids, which is right, and withheld their existence with
        them, which is not. The system prompt repeatedly directs the model at
        "the active object", so a run could prepare a Calendar preview and then
        be unable to tell, on the next turn, that one was pending — the approval
        gate was unreachable rather than merely awkward.

        Deriving it means a new active object is covered the moment it is
        declared, and that a value can never leak: only ``is not None`` crosses.
        """
        return {
            f"has_{name[: -len('_id')]}": getattr(self, name) is not None
            for name in type(self).model_fields
            if name.startswith("active_") and name.endswith("_id")
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
