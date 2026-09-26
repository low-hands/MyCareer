from __future__ import annotations

from collections.abc import Mapping
import hashlib
import hmac
from datetime import datetime, timedelta, timezone
import json
import re
from typing import Annotated, Any, Literal, NamedTuple, Protocol, get_args

from pydantic import AliasChoices, Field, field_validator, model_validator

from career_agent.agent.summary_text import DELIVERY_SUMMARY_LIMIT, clamp
from career_agent.agent.delivery_policy import is_failed, is_waiting
from career_agent.agent.delivered_body_contracts import (
    BodyDependency,
    DeliveredBodySource,
)
from career_agent.agent.conversation_memory_contracts import (
    SUMMARY_ITEM_MAX_CHARS,
    SUMMARY_SOURCE_MAX_CHARS,
    ConversationSummaryContent,
)
from career_agent.agent.token_budget import serialized_token_count
from career_agent.agent.tool_effects import is_external_write, owner_rule_capabilities
from career_agent.agent.questionnaire_contracts import PendingQuestionnaire, UserQuestion
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


class CurrentTargetContext(ContractModel):
    """Current stored values of one user-created TargetRole."""

    target_role_id: str | None = Field(default=None, exclude=True)
    title: str = Field(min_length=1)
    priority: int = Field(ge=0)
    status: Literal["active"] = Field(default="active", exclude=True)
    city: str | None = Field(default=None, min_length=1, max_length=40)
    salary_expectation: str | None = Field(default=None, min_length=1, max_length=100)
    experience: str | None = Field(default=None, min_length=1, max_length=100)
    education: str | None = Field(default=None, min_length=1, max_length=100)
    city_confirmed_at: datetime | None = Field(default=None, exclude=True)
    salary_expectation_confirmed_at: datetime | None = Field(
        default=None,
        exclude=True,
    )
    experience_confirmed_at: datetime | None = Field(default=None, exclude=True)
    education_confirmed_at: datetime | None = Field(default=None, exclude=True)


class HardConstraintContext(ContractModel):
    """One confirmed person-level job constraint from a closed relation set."""

    relation: Literal[
        "work_arrangement",
        "work_schedule",
        "company_scale",
    ]
    value: str = Field(min_length=1, max_length=500)
    confirmed_at: datetime | None = Field(default=None, exclude=True)


class FreeTextPreferenceContext(ContractModel):
    """A current free-text preference projected with its authority boundary."""

    scope_key: str = Field(min_length=1, max_length=500, exclude=True)
    topic_key: str = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")
    statement: str = Field(min_length=1, max_length=2000)
    status: Literal["quarantined", "active"]
    observed_at: datetime
    confirmed_at: datetime | None = None
    ownership: Literal[
        "person_stable",
        "person_default",
        "person_situational",
        "role",
        "situational",
    ] = "person_default"
    pref_scope: str = Field(default="freeform.person_default", exclude=True)
    layer: Literal["stable", "contextual", "transient"] = Field(
        default="contextual", exclude=True
    )
    timescale: Literal["permanent", "situational"] = Field(
        default="permanent", exclude=True
    )
    valid_until: datetime | None = Field(default=None, exclude=True)
    needs_scope_clarification: bool = Field(default=False, exclude=True)
    update_id: str = Field(
        pattern=r"^intent_update_[a-f0-9]{32}$",
        exclude=True,
    )

    @model_validator(mode="after")
    def confirmation_matches_status(self) -> "FreeTextPreferenceContext":
        if (self.status == "active") != (self.confirmed_at is not None):
            raise ValueError("free-text preference authority is inconsistent")
        return self


class FreeTextPreferenceConfirmationProposal(ContractModel):
    """The exact quarantined revision shown to the user for confirmation."""

    update_id: str = Field(pattern=r"^intent_update_[a-f0-9]{32}$")
    topic_key: str = Field(pattern=r"^[a-z][a-z0-9_]{0,79}$")
    statement: str = Field(min_length=1, max_length=2000)
    ownership: Literal[
        "person_stable",
        "person_default",
        "person_situational",
        "role",
        "situational",
    ] = "person_default"
    pref_scope: str = Field(
        default="freeform.person_default",
        min_length=1,
        max_length=120,
    )
    needs_scope_clarification: bool = False
    base_update_id: str | None = Field(
        default=None,
        pattern=r"^intent_update_[a-f0-9]{32}$",
        description="Active revision being amended from a MEMORY.md review.",
    )
    expected_content_sha256: str | None = Field(
        default=None,
        pattern=r"^sha256:[a-f0-9]{64}$",
    )

    @model_validator(mode="after")
    def review_amendment_is_version_bound(
        self,
    ) -> "FreeTextPreferenceConfirmationProposal":
        if (self.base_update_id is None) != (
            self.expected_content_sha256 is None
        ):
            raise ValueError(
                "a preference amendment requires both base revision and digest"
            )
        return self


class MemoryTelemetryBinding(ContractModel):
    """Version-bound record of one value the prompt exposed, never projected.

    The name undersells it. Besides feeding P1 telemetry, this is the only
    record of which claims a turn put in front of the model, and deletion
    depends on it: a tombstone finds the transcript text to suppress by
    matching these entry ids. Dropping a binding silently weakens deletion
    rather than losing a metric.
    """

    entry_id: str = Field(min_length=1, max_length=440)
    update_id: str = Field(
        pattern=(
            r"^(?:intent_update|career_evidence_update)_[a-f0-9]{32}$"
        )
    )
    content_digest: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    value: str = Field(min_length=1, max_length=32_000)
    revision: int = Field(ge=1)
    lifecycle_status: Literal["current", "superseded"]


class CareerProfileContext(ContractModel):
    """Person-level job intent only.

    Everything that varies between the roles a user is pursuing — salary band,
    the experience and education brackets they are searching within, and a city
    they would accept for one role but not another — lives on TargetRole, which
    already has identity, priority, and the resume families hanging off it.
    Keeping a second free-text role list here only guaranteed that one of the
    two would eventually be written to and the other read.
    """

    user_id: str
    default_city: str | None = None
    default_city_confirmed_at: datetime | None = Field(default=None, exclude=True)
    hard_constraints: tuple[HardConstraintContext, ...] = Field(
        default=(),
        max_length=8,
    )
    current_targets: tuple[CurrentTargetContext, ...] = Field(
        default=(),
        max_length=100,
        exclude=True,
    )
    """Read-time projection from TargetRole; never duplicated in profile storage."""
    current_targets_total: int = Field(default=0, ge=0, exclude=True)
    telemetry_bindings: tuple[MemoryTelemetryBinding, ...] = Field(
        default=(),
        max_length=512,
        exclude=True,
    )
    telemetry_inventory_complete: bool = Field(default=False, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def populate_current_target_total(cls, value: object) -> object:
        if not isinstance(value, dict) or "current_targets_total" in value:
            return value
        targets = value.get("current_targets", ())
        return {**value, "current_targets_total": len(targets)}

    @model_validator(mode="after")
    def hard_constraint_relations_are_unique(self) -> "CareerProfileContext":
        relations = [item.relation for item in self.hard_constraints]
        if len(relations) != len(set(relations)):
            raise ValueError("hard constraint relations must be unique")
        if self.current_targets_total < len(self.current_targets):
            raise ValueError("current_targets_total cannot be smaller than loaded roles")
        return self


class JobIntentUpdate(ContractModel):
    """A proposed change to stated job intent, not yet applied.

    ``target_role_id`` decides the scope. Unset, the update is about the person
    and may only carry ``city``. Set, it is about that one career track, where a
    city is an override of the person-level default.

    The scoping is a validator rather than a prompt instruction because a salary
    recorded against the person cannot be un-mixed later: the two roles it was
    meant to distinguish would already have collapsed into one number.

    Every field is optional because intent is stated a piece at a time, and an
    omitted field leaves the stored value alone rather than clearing it.
    """

    target_role_id: str | None = Field(default=None, min_length=1)
    pref_scope: str = Field(
        default="global",
        pattern=r"^(?:global|[a-z][a-z0-9_.:-]*)$",
        max_length=120,
    )
    timescale: Literal["permanent", "situational"] = "permanent"
    layer: Literal["stable", "contextual", "transient"] | None = None
    valid_until: datetime | None = None
    city: str | None = Field(default=None, min_length=1, max_length=40)
    salary_expectation: str | None = Field(default=None, min_length=1, max_length=100)
    experience: str | None = Field(default=None, min_length=1, max_length=100)
    education: str | None = Field(default=None, min_length=1, max_length=100)
    hard_constraints: tuple[HardConstraintContext, ...] = Field(
        default=(),
        max_length=8,
    )

    @model_validator(mode="after")
    def scope_must_match_the_fields(self) -> "JobIntentUpdate":
        role_scoped = (self.salary_expectation, self.experience, self.education)
        if self.target_role_id is not None and self.hard_constraints:
            raise ValueError("hard constraints belong to the person, not a target role")
        relations = [item.relation for item in self.hard_constraints]
        if len(relations) != len(set(relations)):
            raise ValueError("hard constraint relations must be unique")
        if self.target_role_id is None and any(
            value is not None for value in role_scoped
        ):
            raise ValueError(
                "salary, experience, and education belong to a target role and "
                "need one to be selected"
            )
        if not any(
            value is not None for value in (self.city, *role_scoped)
        ) and not self.hard_constraints:
            raise ValueError("a job intent update must change at least one field")
        if self.timescale == "situational" and self.pref_scope == "global":
            raise ValueError(
                "situational intent requires the named situation in pref_scope"
            )
        if self.timescale == "permanent" and self.valid_until is not None:
            raise ValueError("permanent intent cannot carry valid_until")
        if self.layer == "transient" and self.timescale != "situational":
            raise ValueError("transient intent must be situational")
        return self

    @property
    def is_role_scoped(self) -> bool:
        return self.target_role_id is not None

    def apply_to_profile(self, profile: CareerProfileContext) -> CareerProfileContext:
        if self.is_role_scoped or self.pref_scope != "global":
            return profile
        constraints = {
            constraint.relation: constraint
            for constraint in profile.hard_constraints
        }
        constraints.update(
            {
                constraint.relation: constraint
                for constraint in self.hard_constraints
            }
        )
        changes: dict[str, Any] = {
            "hard_constraints": tuple(
                constraints[relation] for relation in sorted(constraints)
            )
        }
        if self.city is not None:
            changes["default_city"] = self.city
        return profile.model_copy(update=changes)


class MemoryTombstoneProposal(ContractModel):
    """One field-level deletion read back before the irreversible write."""

    target_kind: Literal["career_evidence", "intent_preference"]
    detail_ref: str | None = Field(
        default=None,
        pattern=r"^detail_[a-f0-9]{24}$",
    )
    scope_key: str | None = None
    update_id: str | None = Field(
        default=None,
        pattern=r"^intent_update_[a-f0-9]{32}$",
    )
    pref_scope: str | None = None
    reason: str = Field(min_length=1, max_length=2000)
    expected_content_sha256: str | None = Field(
        default=None,
        pattern=r"^sha256:[a-f0-9]{64}$",
        description=(
            "Store-sealed digest used to reject deletion if the claim changed "
            "after the proposal was shown."
        ),
    )

    @model_validator(mode="after")
    def target_has_exact_identity(self) -> "MemoryTombstoneProposal":
        if self.target_kind == "career_evidence":
            if self.detail_ref is None or any(
                value is not None
                for value in (self.scope_key, self.update_id, self.pref_scope)
            ):
                raise ValueError("career evidence tombstones require only detail_ref")
        elif (
            self.detail_ref is not None
            or self.scope_key is None
            or self.update_id is None
            or self.pref_scope is None
            or self.expected_content_sha256 is None
        ):
            raise ValueError(
                "preference tombstones require scope, track, revision, and digest"
            )
        return self


class ConstraintRetirementProposal(ContractModel):
    """One conversation constraint read back before it stops applying.

    The constraint is carried as its exact text, not a position or a row id,
    because the same text is what the ledger matches on and what the user saw.
    """

    target_kind: Literal["conversation_constraint"]
    constraint: str = Field(min_length=1, max_length=SUMMARY_ITEM_MAX_CHARS)
    reason: str = Field(min_length=1, max_length=2000)


class MemoryAmendmentProposal(ContractModel):
    """One claim correction read back before a new revision is written."""

    target_kind: Literal["career_evidence"]
    detail_ref: str = Field(pattern=r"^detail_[a-f0-9]{24}$")
    new_claim: str = Field(min_length=1, max_length=32_000)
    reason: str = Field(min_length=1, max_length=2000)


class CareerFactProposal(ContractModel):
    """One pending inferred fact read back before it becomes authoritative."""

    career_evidence_id: str = Field(
        pattern=r"^career_evidence_[a-f0-9]{32}$"
    )
    career_record_id: str = Field(
        pattern=r"^career_record_[a-f0-9]{32}$"
    )
    claim: str = Field(min_length=1, max_length=2000)
    reason: str = Field(min_length=1, max_length=2000)


RuleVerdict = Literal["permit", "review", "deny"]
"""What an owner rule says about a capability, ordered least to most restrictive.

``review`` is a first-class outcome, not a soft ``deny``: "you may do this, but
show me first" is what a person means by "ask me before you apply", and it maps
onto the bound interaction this system already has. Collapsing it into ``deny``
would turn a request for oversight into a refusal, and collapsing it into
``permit`` would silently drop the oversight.
"""

_VERDICT_ORDER: dict[str, int] = {"permit": 0, "review": 1, "deny": 2}


def most_restrictive(*verdicts: RuleVerdict) -> RuleVerdict:
    """Compose owner-rule verdicts conservatively: the strictest one wins.

    Several of the owner's rules can bear on one capability, and they are
    combined by taking the most restrictive rather than by order or precedence.
    Precedence would mean a later rule could widen what an earlier one narrowed,
    which is how a permission system stops being one.

    Scoped to owner rules on purpose. Budgets and reachability also gate an
    action, but they are not verdicts on the same lattice: they answer "not this
    turn" and "not with this state", and folding them in here would suggest the
    owner could permit past them.
    """

    return max(verdicts, key=lambda verdict: _VERDICT_ORDER[verdict], default="permit")


def system_capability_verdict(capability: str) -> RuleVerdict:
    """Non-negotiable runtime invariants, never sourced from owner settings.

    The model may formulate a settings change, but it cannot authorize the
    change that governs itself. Keeping this outside ``BehaviorPolicyContext``
    prevents a future owner-facing field from accidentally weakening it.

    External writes sit on the same footing. Once an event is on the user's
    calendar or a message has left the mailbox, no later turn can undo it, so
    the model's reading of "yes, go ahead" is not enough: the owner presses the
    button on the exact sealed arguments. Owner rules may only add restrictions
    on top of this floor.
    """

    if capability == "update_owner_settings" or is_external_write(capability):
        return "review"
    return "permit"


ConfirmBefore = tuple[str, ...]
"""Capabilities the owner wants to approve individually before they run."""


def canonical_confirm_before(value: ConfirmBefore) -> ConfirmBefore:
    """Only WRITE capabilities an owner rule can stop, deduplicated and sorted.

    A rule naming an unknown, read-only or runtime-owned capability would never
    fire, and an owner who typed it believes they are protected. Rejecting it at
    the write boundary (tool arguments, API request, CLI flag) keeps the settings
    document honest. Sorting makes the stored form canonical so a reorder is not
    mistaken for a policy change.

    This is a boundary check, not a storage invariant: a stored document is
    read with ``stored_confirm_before`` so a capability renamed or removed after
    the rule was written still lets the owner load and edit their settings.
    """

    unknown = sorted(set(value) - owner_rule_capabilities())
    if unknown:
        raise ValueError(
            "confirm_before only accepts owner-confirmable WRITE capabilities; unknown: "
            + ", ".join(unknown)
        )
    return stored_confirm_before(value)


def stored_confirm_before(value: ConfirmBefore) -> ConfirmBefore:
    """The canonical stored form, without judging the names against today's registry."""

    return tuple(sorted(set(value)))


class UserPreferencesContext(ContractModel):
    """Soft owner preferences interpreted by the decision model.

    These are deliberately not described as enforcement.  They depend on the
    meaning of the current request (for example, whether search was explicitly
    requested), so the deterministic runtime does not have enough information
    to decide them without inventing a second intent classifier.
    """

    boss_search: Literal["explicit_request_only", "allowed"] = "explicit_request_only"


class BehaviorPolicyContext(ContractModel):
    """Versioned rules the runtime can evaluate before a capability executes."""

    revision: int = Field(default=0, ge=0)
    application_confirmation: Literal["always_ask", "on_user_report"] = "on_user_report"
    confirm_before: ConfirmBefore = ()

    @field_validator("confirm_before")
    @classmethod
    def normalise_confirm_before(cls, value: ConfirmBefore) -> ConfirmBefore:
        # Read side. Names are judged where they are written; a rule for a
        # capability that no longer exists is inert here, not a reason every
        # turn that loads this document fails.
        return stored_confirm_before(value)

    def _owner_rule_verdicts(self, capability: str) -> tuple[RuleVerdict, ...]:
        """Only owner-editable rules; system invariants do not belong here."""

        return tuple(
            verdict
            for applies, verdict in (
                (
                    capability == "create_application"
                    and self.application_confirmation == "always_ask",
                    "review",
                ),
                (capability in self.confirm_before, "review"),
            )
            if applies
        )

    def capability_verdict(self, capability: str) -> RuleVerdict:
        return most_restrictive(*self._owner_rule_verdicts(capability))


class OwnerSettingsContext(ContractModel):
    """Owner state, split into model preferences and runtime policy.

    The two sections share one persisted document so a settings screen can
    update them atomically, but they do not share semantics. ``preferences`` is
    projected to the model. ``behavior_policy`` is evaluated by the runtime and
    carries its own revision, which is sealed into approval records.

    Owner-authorized: the authenticated settings endpoint and CLI may update it
    directly. The model-facing tool can only create a bound proposal; a system
    invariant forces that capability through Review, and the sealed change is
    applied only after an owner interaction receipt. Thus conversation or
    document wording cannot itself relax the rule that constrains the model.

    ``revision`` protects the whole settings document from lost updates.
    ``behavior_policy.revision`` changes only when an enforceable rule changes,
    so editing a soft display/model preference does not invalidate an unrelated
    approval already waiting for the owner.
    """

    revision: int = Field(default=0, ge=0)
    preferences: UserPreferencesContext = UserPreferencesContext()
    behavior_policy: BehaviorPolicyContext = BehaviorPolicyContext()

    @model_validator(mode="before")
    @classmethod
    def migrate_flat_preferences(cls, value: Any) -> Any:
        """Read the pre-split JSON/constructor shape during the v3 migration."""

        if not isinstance(value, dict):
            return value
        if "preferences" in value or "behavior_policy" in value:
            return value
        migrated = dict(value)
        boss_search = migrated.pop("boss_search", "explicit_request_only")
        confirmation = migrated.pop("application_confirmation", "on_user_report")
        migrated["preferences"] = {"boss_search": boss_search}
        migrated["behavior_policy"] = {
            "revision": 0,
            "application_confirmation": confirmation,
        }
        return migrated

    @property
    def boss_search(self) -> Literal["explicit_request_only", "allowed"]:
        return self.preferences.boss_search

    @property
    def application_confirmation(self) -> Literal["always_ask", "on_user_report"]:
        return self.behavior_policy.application_confirmation

    def capability_verdict(self, capability: str) -> RuleVerdict:
        return most_restrictive(
            system_capability_verdict(capability),
            self.behavior_policy.capability_verdict(capability),
        )


# Source compatibility for integrations importing the old name. New code and
# persisted JSON use OwnerSettingsContext and its explicit two-section shape.
AgentPreferencesContext = OwnerSettingsContext


SelectionIndex = Annotated[
    int,
    Field(
        ge=1,
        description=(
            "A 1-based selection_index explicitly shown beside the intended "
            "object in this tool's corresponding candidate list. Never invent "
            "an index or borrow numbering from another list or from the order "
            "of tool observations."
        ),
    ),
]
"""A 1-based pointer into a list the model was shown this turn.

The lower bound belongs to the type, not to each declaration. Written out
per-field it was correct twenty-nine times and missing once — on
``CompareSavedJobsToolArguments.job_selection_indices``, whose element type
carried no bound at all, so index 0 resolved through ``candidates[-1]`` to the
last saved job and a comparison ran against the wrong pair without erroring.
"""


class CandidateContextItem(ContractModel):
    result_ref: str
    title: str
    company_name: str
    city: str | None = None
    salary: str | None = None


class ApplicationCandidateContextItem(ContractModel):
    application_id: str
    title: str
    company_name: str
    status: ApplicationStatus


class InterviewCandidateContextItem(ContractModel):
    interview_round_id: str
    application_id: str
    sequence_number: int
    employer_label: str | None = None
    status: InterviewStatus
    scheduled_start: datetime | None = None


class ActionCandidateContextItem(ContractModel):
    action_item_id: str
    action_type: ActionType
    source_type: ActionSourceType
    source_id: str
    title: str
    status: ActionStatus
    due_at: datetime | None = None


class CalendarAccountCandidateContextItem(ContractModel):
    calendar_account_id: str
    provider: Literal["google"]
    email_address: str
    calendar_id: str


class SavedJobCandidateContextItem(ContractModel):
    job_posting_id: str
    title: str
    company_name: str
    city: str | None = None
    salary: str | None = None
    jd_snapshot_id: str | None = None
    jd_version: int | None = Field(default=None, ge=1)


class ActiveSavedJobContextItem(ContractModel):
    """Which saved job, and which JD version of it, the conversation is on.

    ``active_job_posting_id`` says which posting; this pins the snapshot that
    turn actually read so "this job" keeps resolving to the same text after the
    posting is re-captured as v2. The ids never reach the model — the
    projection shows the title, the version and whether the snapshot can
    still be read.
    """

    job_posting_id: str = Field(min_length=1)
    jd_snapshot_id: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=200)
    company_name: str = Field(min_length=1, max_length=200)
    jd_version: int = Field(ge=1)
    readable: bool = True


class TargetRoleCandidateContextItem(ContractModel):
    target_role_id: str
    title: str
    priority: int
    status: str
    # The intent recorded against this track. It rides along here rather than
    # being flattened into career_profile so the model sees which numbers belong
    # to which role instead of one blended set.
    city: str | None = None
    salary_expectation: str | None = None
    experience: str | None = None
    education: str | None = None


class ResumeCandidateContextItem(ContractModel):
    resume_id: str
    target_role_id: str
    name: str
    status: str
    latest_version_id: str | None = None


class ResumeVersionCandidateContextItem(ContractModel):
    resume_version_id: str
    version_number: int
    source_type: str
    document_format: str
    byte_size: int
    # Set when the list spans several resumes, as the mock-interview resume
    # choice does; a single resume's version list leaves it empty.
    resume_name: str | None = None


class EmailEventCandidateContextItem(ContractModel):
    email_event_id: str
    event_type: str
    status: EmailEventStatus
    summary: str


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


ToolProfile = Literal["core", "job", "resume", "application", "interview", "memory"]
TOOL_PROFILE_NAMES: tuple[ToolProfile, ...] = get_args(ToolProfile)
DOMAIN_TOOL_PROFILES: tuple[ToolProfile, ...] = tuple(
    name for name in TOOL_PROFILE_NAMES if name != "core"
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
    """

    active_workflow: Literal["job_discovery", "mock_interview", "none"] = "none"
    pending_questionnaire: PendingQuestionnaire | None = None
    tool_profile: ToolProfile = "core"
    run_id: str | None = None
    phase: str | None = None
    selected_result_ref: str | None = None
    manual_search_query: str | None = None
    candidates: tuple[CandidateContextItem, ...] = ()
    workflow_entry_message: str | None = None
    workflow_entry_resource_refs: tuple["ConversationResourceReference", ...] = ()
    """The inputs attached to the held request, written back with it on exit."""
    workflow_entry_at: datetime | None = None
    """When the held request was sent, so the transcript shows it before the run
    it started rather than at the moment the run ended and it was written."""
    pending_job_intent_update: JobIntentUpdate | None = None
    pending_free_text_preference: FreeTextPreferenceConfirmationProposal | None = None
    pending_memory_amendment: MemoryAmendmentProposal | None = None
    pending_memory_tombstone: MemoryTombstoneProposal | None = None
    pending_career_fact: CareerFactProposal | None = None
    pending_constraint_retirement: ConstraintRetirementProposal | None = None
    # When each pending proposal was shown, keyed by slot. Kept beside the
    # proposals rather than inside them: JobIntentUpdate is hashed into the
    # intent episode id, so a timestamp there would change what gets recorded.
    pending_proposed_at: dict[PendingProposalSlot, datetime] = Field(
        default_factory=dict
    )
    bare_confirmation_target: Literal[
        "career_fact",
        "job_intent",
        "free_text_preference",
    ] | None = None
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
        if isinstance(value, dict) and (
            "active_resume_analysis_id" in value or "resume_analysis_status" in value
        ):
            value = {
                key: item
                for key, item in value.items()
                if key not in {"active_resume_analysis_id", "resume_analysis_status"}
            }
        return value

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

    @model_validator(mode="after")
    def _validate_workflow_slot(self) -> "ConversationTaskState":
        if (self.active_workflow == "none") != (self.run_id is None):
            raise ValueError(
                "active_workflow and run_id must be set together: a named "
                "workflow needs a run to resume, and a run needs an owner."
            )
        return self

    def pending_proposal_is_live(
        self, slot: PendingProposalSlot, now: datetime
    ) -> bool:
        """Whether ``slot`` holds a proposal shown within the TTL.

        An unstamped proposal predates stamping and its age is unknown, so it
        is treated as expired rather than granted a fresh clock.
        """
        proposed_at = self.pending_proposed_at.get(slot)
        return (
            getattr(self, slot) is not None
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
        stamps: dict[PendingProposalSlot, datetime] = {}
        for slot in PENDING_PROPOSAL_SLOTS:
            value = getattr(self, slot)
            if value is None:
                continue
            if value is not getattr(before, slot):
                stamps[slot] = now
            elif slot in self.pending_proposed_at:
                stamps[slot] = self.pending_proposed_at[slot]
        if stamps == self.pending_proposed_at:
            return self
        return self.model_copy(update={"pending_proposed_at": stamps})

    def expire_stale_proposals(
        self, now: datetime
    ) -> tuple["ConversationTaskState", tuple[PendingProposalSlot, ...]]:
        """Drop proposals left unconfirmed past ``PENDING_PROPOSAL_TTL``."""
        expired = tuple(
            slot
            for slot in PENDING_PROPOSAL_SLOTS
            if getattr(self, slot) is not None
            and not self.pending_proposal_is_live(slot, now)
        )
        if not expired:
            return self, ()
        update: dict[str, Any] = {slot: None for slot in expired}
        update["pending_proposed_at"] = {
            slot: stamp
            for slot, stamp in self.pending_proposed_at.items()
            if slot not in expired
        }
        if (
            self.bare_confirmation_target is not None
            and _BARE_CONFIRMATION_SLOTS[self.bare_confirmation_target] in expired
        ):
            update["bare_confirmation_target"] = None
        return self.model_copy(update=update), expired

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
        return self.model_copy(
            update={
                "active_workflow": workflow,
                "run_id": run_id,
                "phase": phase,
                "selected_result_ref": selected_result_ref,
                "manual_search_query": manual_search_query,
                "candidates": self.candidates if candidates is None else candidates,
                "workflow_entry_message": (
                    self.workflow_entry_message
                    if self.active_workflow == workflow
                    else None
                ),
                "workflow_entry_resource_refs": (
                    self.workflow_entry_resource_refs
                    if self.active_workflow == workflow
                    else ()
                ),
                "workflow_entry_at": (
                    self.workflow_entry_at if self.active_workflow == workflow else None
                ),
            }
        )

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
        return self.model_copy(
            update={
                "workflow_entry_message": message,
                "workflow_entry_resource_refs": resource_refs,
                "workflow_entry_at": at,
            }
        )

    def leave_workflow(self) -> "ConversationTaskState":
        """Release the slot and clear the scoped fields together.

        The held request is scoped to the occupant too. Whoever releases the
        slot has already paired it with a closing reply, so carrying it forward
        would let the next workflow answer this one's opening line.
        """
        return self.model_copy(
            update={
                "active_workflow": "none",
                "run_id": None,
                "phase": None,
                "selected_result_ref": None,
                "manual_search_query": None,
                "workflow_entry_message": None,
                "workflow_entry_resource_refs": (),
                "workflow_entry_at": None,
            }
        )


class ConversationResourceReference(ContractModel):
    """Immutable link from one historical message to its durable resource.

    A statement of what one past turn produced, so it never changes once
    written. This is the opposite of the ``active_*_id`` fields on the task,
    which track what the user is discussing now and are overwritten every time
    the focus moves. Reading back the report a turn produced needs the former;
    resolving "this report" with no antecedent needs the latter.

    ``resume_version`` is the one kind a *user* message carries: the exact
    immutable version the user attached to that turn. The row keeps only the
    id and a display snapshot, never the file, and it keeps pointing at that
    version after the resume gains newer ones or is deleted.
    """

    kind: Literal[
        "job_research_report",
        "mock_interview_report",
        "interview_preparation",
        "interview_retro_report",
        "job_analysis",
        "resume_job_match",
        "resume_tailoring_draft",
        "resume_version",
        "saved_job",
    ]
    resource_id: str = Field(min_length=1)
    """For ``saved_job`` this is the immutable ``jd_snapshot_id``, never the
    posting id: the card in an old turn must keep opening the JD version that
    turn read after the posting is re-captured as a newer one."""
    # Job research alone needs delivery-time render metadata because
    # ``anchored_by_other_job`` is relative to the request that produced this
    # turn and cannot be reconstructed from the report row. Its status is
    # snapshotted with that render bundle. Other kinds intentionally derive
    # current lifecycle state at read time — for example a tailoring card
    # should say that its draft has since been superseded.
    status_at_delivery: Literal["current", "outdated", "superseded"] | None = None
    anchored_by_other_job: bool | None = None
    job_posting_id: str | None = Field(default=None, min_length=1)
    company_key: str | None = Field(default=None, min_length=1, max_length=300)
    """The employer a research report is about, as identity rather than prose.

    ``title`` is for a reader; these are for a check. Whether the handle the
    model passes back is about the company the user asked for is decided by
    comparing the report's ``company_key`` (and the anchoring ``job_posting_id``)
    against the saved job in hand, so a user writing "字节" for a job saved under
    "字节跳动" is matched through the job entity, not through the spelling.
    Filled by job research alone; missing on references written before it was
    recorded, which then fall back to their title.
    """
    title: str | None = Field(
        default=None,
        validation_alias=AliasChoices("title", "label"),
        min_length=1,
        max_length=80,
    )
    """What this resource is *about*, for a reader choosing between several.

    The handle answers "how do I ask for it"; this answers "which one is it".
    Without it a catalogue of twelve research reports projects as twelve
    interchangeable lines that differ only in an opaque suffix, and a model
    asked about one of them has nothing to match against — so it either asks
    which, or picks. MCP's ``ResourceLink`` carries ``title``/``description``
    beside the URI for exactly this reason; we had implemented the URI half and
    left the describing half out.

    Filled by whichever capability produced the resource, because only it knows
    what the thing is: a company and role for research, a version for a resume,
    a session for a mock interview. Same discipline as ``facts`` and the prose
    ``next_action`` — declared where the typed object is, not reconstructed from
    a payload later.

    Missing is allowed for legacy rows and means "not titled yet", not "no title
    exists". New producers fill it. The old field name ``label`` remains a
    validation alias so already persisted references survive the rename, while
    every new serialization uses the MCP-aligned name ``title``.
    """
    description: str | None = Field(default=None, min_length=1, max_length=200)
    """Producer-owned hint about what reading the resource will provide.

    This is deliberately not reconstructed from the assistant's delivery
    message. One turn can deliver several resources and a model-written reply
    can be as generic as "done", so the producer is the only reliable place to
    declare a resource-specific preview.
    """

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_empty_label(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.get("label") == "":
            normalized = dict(value)
            normalized.pop("label")
            return normalized
        return value

    @model_validator(mode="after")
    def scope_job_research_render_context(self) -> "ConversationResourceReference":
        values = (self.status_at_delivery, self.anchored_by_other_job)
        if self.kind == "job_research_report":
            if any(value is None for value in values):
                raise ValueError(
                    "job research references require delivery-time render context"
                )
        elif any(value is not None for value in values):
            raise ValueError(
                "delivery-time render metadata is scoped to job research; "
                "other resource kinds derive current state when read"
            )
        if self.kind == "saved_job":
            if self.job_posting_id is None:
                raise ValueError("saved_job references name their posting")
            if self.company_key is not None:
                raise ValueError("company_key on a reference is scoped to job research")
        elif self.kind != "job_research_report" and (
            self.job_posting_id is not None or self.company_key is not None
        ):
            raise ValueError(
                "company identity on a reference is scoped to job research"
            )
        return self


class ConversationMessageContext(ContractModel):
    role: Literal["user", "assistant"]
    content: str
    content_clipped: bool = False
    """True only on an in-memory recent-window projection.

    Stored messages are complete up to the durable per-message ceiling.  The
    context manager sets this bit on a projected copy when the smaller live
    window clips the content, so the decision prompt never presents a partial
    sentence as the complete message.
    """
    created_at: datetime
    user_interaction_id: str | None = Field(
        default=None, pattern=r"^interaction_[a-f0-9]{20}$"
    )
    resource_refs: tuple[ConversationResourceReference, ...] = ()
    """Every stored report this turn produced, in the order it produced them.

    Plural because a turn is. Four card-backed reads fit inside the read budget,
    so one turn can end holding two reports; a single field would let the live
    stream hand the reader two cards while the reloaded transcript shows one,
    and would leave the earlier report without a handle for the model to read it
    back with.
    """


MAX_CONVERSATION_SPAN_MESSAGES = 8
MAX_CONVERSATION_SPAN_RESOURCE_REFS = 32


class ConversationSpanMessage(ContractModel):
    sequence: int = Field(ge=1)
    role: Literal["user", "assistant"]
    content: str = Field(max_length=SUMMARY_SOURCE_MAX_CHARS)
    content_clipped: bool = False
    created_at: datetime


class ConversationSpanView(ContractModel):
    from_sequence: int = Field(ge=1)
    through_sequence: int = Field(ge=1)
    returned: int = Field(ge=0, le=MAX_CONVERSATION_SPAN_MESSAGES)
    total: int = Field(ge=0)
    body_clipped: bool = False
    resource_ref_total: int = Field(default=0, ge=0)
    resource_refs: tuple[ConversationResourceReference, ...] = Field(
        default=(), max_length=MAX_CONVERSATION_SPAN_RESOURCE_REFS
    )
    messages: tuple[ConversationSpanMessage, ...] = Field(
        default=(), max_length=MAX_CONVERSATION_SPAN_MESSAGES
    )

    @model_validator(mode="after")
    def counts_match_messages(self) -> "ConversationSpanView":
        if self.from_sequence > self.through_sequence:
            raise ValueError("conversation span must move forward")
        if self.returned != len(self.messages):
            raise ValueError("returned must match the messages provided")
        if self.total < self.returned:
            raise ValueError("total cannot be smaller than returned")
        if self.resource_ref_total < len(self.resource_refs):
            raise ValueError("resource_ref_total cannot be smaller than returned refs")
        return self



class CareerMemoryClaim(ContractModel):
    """Confirmed L2 claim with bounded provenance, never an inline quotation."""

    claim: str = Field(min_length=1)
    origin: Literal["resume_extraction", "user_input", "agent_inference"]
    recorded_at: datetime
    """Storage observation time, not the claim's real-world effective date."""
    source_ref: str | None = Field(
        default=None,
        pattern=r"^evidence_[a-f0-9]{24}$",
    )
    revision: int = Field(ge=1)
    detail_ref: str = Field(pattern=r"^detail_[a-f0-9]{24}$")
    telemetry_binding: MemoryTelemetryBinding | None = Field(
        default=None,
        exclude=True,
    )


class CareerMemoryRecord(ContractModel):
    record_id: str | None = Field(
        default=None,
        pattern=r"^career_record_[a-f0-9]{32}$",
        exclude=True,
    )
    record_type: Literal[
        "education",
        "work",
        "internship",
        "project",
        "certification",
    ]
    organization: str | None = None
    title: str
    start_year: int | None = None
    start_month: int | None = None
    end_year: int | None = None
    end_month: int | None = None
    is_current: bool = False
    confirmed_highlights: tuple[CareerMemoryClaim, ...] = ()


class CareerMemoryContext(ContractModel):
    records: tuple[CareerMemoryRecord, ...] = ()
    records_total: int = Field(default=0, ge=0)
    claims_total: int = Field(default=0, ge=0)
    telemetry_bindings: tuple[MemoryTelemetryBinding, ...] = Field(
        default=(),
        max_length=512,
        exclude=True,
    )
    telemetry_inventory_complete: bool = Field(default=False, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def populate_totals(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        records = value.get("records", ())
        updates: dict[str, object] = {}
        if "records_total" not in value:
            updates["records_total"] = len(records)
        if "claims_total" not in value:
            updates["claims_total"] = sum(
                len(
                    record.get("confirmed_highlights", ())
                    if isinstance(record, dict)
                    else record.confirmed_highlights
                )
                for record in records
            )
        return {**value, **updates}

    @model_validator(mode="after")
    def totals_cover_loaded_rows(self) -> "CareerMemoryContext":
        loaded_claims = sum(
            len(record.confirmed_highlights) for record in self.records
        )
        if self.records_total < len(self.records):
            raise ValueError("records_total cannot be smaller than loaded records")
        if self.claims_total < loaded_claims:
            raise ValueError("claims_total cannot be smaller than loaded claims")
        return self

    def tier_one_projection(self, *, token_budget: int) -> dict[str, Any]:
        """Render a bounded, self-describing career-memory index.

        Delivery counters remain visible even at a zero budget so an empty
        projection cannot be mistaken for an empty store. Each candidate is
        counted as the serialized object the model would see: BPE is not
        additive across JSON fragments.
        """

        if token_budget < 0:
            raise ValueError("career-memory token budget cannot be negative")
        empty = _career_memory_projection(
            records=[],
            records_total=self.records_total,
            claims_total=self.claims_total,
        )
        if token_budget == 0:
            return empty

        projected_records: list[dict[str, Any]] = []
        for selection_index, record in enumerate(self.records, start=1):
            serialized_record = record.model_dump(
                mode="json",
                exclude={"confirmed_highlights"},
            )
            serialized_record["selection_index"] = selection_index
            serialized_record["confirmed_highlights"] = []
            candidate_records = [*projected_records, serialized_record]
            candidate = _career_memory_projection(
                records=candidate_records,
                records_total=self.records_total,
                claims_total=self.claims_total,
            )
            if serialized_token_count(candidate) > token_budget:
                break
            projected_records = candidate_records
            accepted_highlights: list[dict[str, Any]] = []
            stopped = False
            for claim in record.confirmed_highlights:
                serialized_claim = claim.model_dump(mode="json")
                serialized_record["confirmed_highlights"] = [
                    *accepted_highlights,
                    serialized_claim,
                ]
                candidate = _career_memory_projection(
                    records=projected_records,
                    records_total=self.records_total,
                    claims_total=self.claims_total,
                )
                if serialized_token_count(candidate) > token_budget:
                    serialized_record["confirmed_highlights"] = (
                        accepted_highlights
                    )
                    stopped = True
                    break
                accepted_highlights.append(serialized_claim)
            serialized_record["confirmed_highlights"] = accepted_highlights
            if stopped:
                break
        return _career_memory_projection(
            records=projected_records,
            records_total=self.records_total,
            claims_total=self.claims_total,
        )


CAREER_RECORDS_TOKEN_BUDGET = 2_800
CAREER_CURRENT_TARGETS_TOKEN_BUDGET = 800
CAREER_HARD_CONSTRAINTS_TOKEN_BUDGET = 600


class CareerProfileBudgets(ContractModel):
    """Career-evidence payload ceiling plus legacy profile-budget inputs.

    Profile facts are now a complete deterministic Markdown projection and do
    not consume either legacy profile budget.  The two fields remain accepted
    while callers migrate their configuration; evidence is still bounded.
    """

    budget_unit: Literal["input_tokens"] = "input_tokens"
    records_input_units: int = Field(
        default=CAREER_RECORDS_TOKEN_BUDGET,
        ge=0,
    )
    current_targets_input_units: int = Field(
        default=CAREER_CURRENT_TARGETS_TOKEN_BUDGET,
        ge=0,
    )
    hard_constraints_input_units: int = Field(
        default=CAREER_HARD_CONSTRAINTS_TOKEN_BUDGET,
        ge=0,
    )

    def estimate_tokens(self, value: Any) -> int:
        return serialized_token_count(value)


def _career_memory_projection(
    *,
    records: list[dict[str, Any]],
    records_total: int,
    claims_total: int,
) -> dict[str, Any]:
    projected: dict[str, Any] = {}
    claims_returned = sum(
        len(record["confirmed_highlights"]) for record in records
    )
    if records:
        projected["records"] = records
    if len(records) < records_total:
        projected["records_returned"] = len(records)
        projected["records_total"] = records_total
    if claims_returned < claims_total:
        projected["claims_returned"] = claims_returned
        projected["claims_total"] = claims_total
    return projected


def confirmation_recency_label(
    confirmed_at: datetime,
    *,
    now: datetime | None = None,
) -> str:
    """Render last corroboration as a Letta/ADK-style relative label."""

    observed_at = now or datetime.now(timezone.utc)
    days = max(
        0,
        int((observed_at - confirmed_at).total_seconds() // 86_400),
    )
    if days == 0:
        return "今天确认"
    return f"{days} 天前确认"


_HARD_CONSTRAINT_LABELS = {
    "work_arrangement": "Work arrangement",
    "work_schedule": "Work schedule",
    "company_scale": "Company scale",
}
_TARGET_INTENT_LABELS = {
    "city": "City",
    "salary_expectation": "Salary expectation",
    "experience": "Experience",
    "education": "Education",
}


def _markdown_value(value: str) -> str:
    """Quote a stored value so its content cannot change Markdown structure."""

    return json.dumps(value, ensure_ascii=False)


def _markdown_fact(
    *,
    label: str,
    value: str | None,
    confirmed_at: datetime | None,
    now: datetime,
    indent: str = "",
) -> list[str]:
    rendered = "Not confirmed" if value is None else _markdown_value(value)
    lines = [f"{indent}- {label}: {rendered}"]
    if value is not None and confirmed_at is not None:
        lines.append(
            f"{indent}  - Last confirmed: "
            f"{confirmation_recency_label(confirmed_at, now=now)}"
        )
    return lines


def career_profile_memory_files(
    profile: CareerProfileContext,
    *,
    now: datetime | None = None,
) -> dict[str, str]:
    """Render complete current profile state as deterministic virtual files.

    These files are a projection, not a retrieval result: every supported
    field is emitted, no score or decay decides whether a fact is present, and
    there is no top-k or token-budget truncation.  Stored values are JSON-quoted
    inside Markdown so untrusted text cannot manufacture headings or bullets.
    """

    observed_at = now or datetime.now(timezone.utc)
    profile_lines = ["# Career profile", "", "## Person-level intent"]
    profile_lines.extend(
        _markdown_fact(
            label="Default city",
            value=profile.default_city,
            confirmed_at=profile.default_city_confirmed_at,
            now=observed_at,
        )
    )
    profile_lines.extend(("", "## Hard constraints"))
    constraints = sorted(
        profile.hard_constraints,
        key=lambda item: (item.relation, item.value),
    )
    if not constraints:
        profile_lines.append("- None confirmed")
    else:
        for constraint in constraints:
            profile_lines.extend(
                _markdown_fact(
                    label=_HARD_CONSTRAINT_LABELS[constraint.relation],
                    value=constraint.value,
                    confirmed_at=constraint.confirmed_at,
                    now=observed_at,
                )
            )

    target_lines = ["# Current career targets"]
    targets = sorted(
        profile.current_targets,
        key=lambda item: (
            item.priority,
            item.title.casefold(),
            item.target_role_id or "",
        ),
    )
    if not targets:
        target_lines.extend(("", "- No active targets"))
    else:
        for index, target in enumerate(targets, start=1):
            target_lines.extend(
                (
                    "",
                    f"## Target {index}",
                    f"- Title: {_markdown_value(target.title)}",
                    f"- Priority: {target.priority}",
                )
            )
            for relation, label in _TARGET_INTENT_LABELS.items():
                target_lines.extend(
                    _markdown_fact(
                        label=label,
                        value=getattr(target, relation),
                        confirmed_at=getattr(target, f"{relation}_confirmed_at"),
                        now=observed_at,
                    )
                )

    return {
        "memory/profile.md": "\n".join(profile_lines) + "\n",
        "memory/current_targets.md": "\n".join(target_lines) + "\n",
    }


def _memory_overflow_notice(memory: Mapping[str, Any]) -> dict[str, Any]:
    sections: list[dict[str, Any]] = []
    records_returned = memory.get("records_returned")
    records_total = memory.get("records_total")
    claims_returned = memory.get("claims_returned")
    claims_total = memory.get("claims_total")
    records_overflow = (
        type(records_returned) is int
        and type(records_total) is int
        and records_returned < records_total
    )
    claims_overflow = (
        type(claims_returned) is int
        and type(claims_total) is int
        and claims_returned < claims_total
    )
    if records_overflow or claims_overflow:
        sections.append(
            {
                "section": "career_memory",
                "strategy": "archive_search",
                "fetch_tool": "search_career_memory",
            }
        )
    if not sections:
        return {}
    return {
        "memory_overflow": {
            "fetch_required": True,
            "sections": sections,
        }
    }
MAX_DECISION_FACTS = 8
# One sentence, not an essay. Long enough for "别再重试，把失败说清楚，或者问用户
# 要不要换个做法"; short enough that a capability cannot annex the decision prompt.
ResourceHandle = Annotated[str, Field(pattern=r"^[a-z]+_[0-9a-f]{6,32}$")]
"""A handle as the model may write it back: the shape, not the membership.

Whether this particular handle was handed out is settled by
``resolve_reference``, which looks it up in what the projection actually
produced. The pattern only refuses text that could never be one.
"""

_HANDLE_PREFIXES = {
    "job_research_report": "report",
    "mock_interview_report": "mock",
    "interview_preparation": "preparation",
    "interview_retro_report": "retro",
    "job_analysis": "analysis",
    "resume_job_match": "match",
    "resume_tailoring_draft": "tailoring",
    "resume_version": "resume",
    "saved_job": "jd",
}
_HANDLE_SUFFIX_LENGTH = 6

ATTACHED_RESUME_EXCERPT_CHARS = 12_000
"""Extracted resume text a turn may show the model, summed over all attachments."""


class AttachedResumeContext(ContractModel):
    """One resume version the user attached to this turn, as the model sees it.

    Built by the runtime after verifying the version belongs to the
    authenticated user; nothing here comes from the request body except the
    id, and the id itself is excluded from the projection. ``excerpt`` is the
    extracted text, cut to this attachment's share of the turn's
    ``ATTACHED_RESUME_EXCERPT_CHARS`` — enough to discuss the resume, without
    the file or the full text ever entering the stored conversation.
    """

    resume_version_id: str = Field(min_length=1, exclude=True)
    resume_id: str = Field(min_length=1, exclude=True)
    resume_name: str = Field(min_length=1, max_length=200)
    version_number: int = Field(ge=1)
    is_latest_version: bool
    target_role: str | None = Field(default=None, max_length=200)
    document_format: Literal["pdf", "text", "markdown"]
    byte_size: int = Field(ge=0)
    uploaded_at: datetime
    excerpt: str | None = Field(default=None, max_length=ATTACHED_RESUME_EXCERPT_CHARS)
    excerpt_truncated: bool = False
    text_unavailable: bool = False
    """True when no text could be read from the file, such as a scanned PDF."""


NEXT_ACTION_LIMIT = 200
# Arguments are model-authored, so unlike a receipt nothing upstream bounds them.
# A window of observations carrying an unbounded dict would break the character budget
# this file declares, so an oversized set is dropped rather than truncated: a
# half-recorded call would read as a call that was made with different arguments,
# which is worse than a call whose arguments are simply not shown.
OBSERVATION_ARGUMENTS_LIMIT = 200
_INTERNAL_ID_VALUE = re.compile(r"\b[0-9a-f]{32}\b")

DecisionFactKey = Annotated[
    str,
    Field(pattern=r"^[a-z][a-z0-9_]{0,39}$"),
]
DecisionFactValue = (
    Annotated[bool, Field(strict=True)]
    | Annotated[int, Field(strict=True, ge=0, le=1_000_000)]
    | Annotated[str, Field(strict=True, min_length=1, max_length=80)]
)


def validate_decision_facts(
    facts: Mapping[str, bool | int | str]
) -> Mapping[str, bool | int | str]:
    """Bound the shape of capability-declared facts, not their membership.

    Facts are declared by the capability that produced the result, next to the
    typed objects it already holds — there is no central state table to keep in
    step with a separate extractor. What stays enforced here is what a central
    table could never check anyway: that a value is a bounded scalar, that no
    key or value carries an internal identifier, and that the set stays small
    enough to be read in a prompt. The declaration itself is the review surface,
    and `tests/agent/test_decision_facts.py` prints it in one place.
    """
    if len(facts) > MAX_DECISION_FACTS:
        raise ValueError(
            f"decision facts may not exceed {MAX_DECISION_FACTS} keys"
        )
    for key, value in facts.items():
        if key == "id" or key.endswith("_id"):
            raise ValueError("decision facts cannot contain internal identifiers")
        if isinstance(value, str) and _INTERNAL_ID_VALUE.search(value):
            raise ValueError(
                f"decision fact {key!r} carries an internal identifier value"
            )
    return facts


class ToolResult(ContractModel):
    """Complete internal result used by reducers and the delivery layer.

    This object is deliberately excluded from ``MainAgentContext`` so adding a
    field to a tool handler can never silently expand the decision prompt.
    """

    tool_name: str
    state: str
    # Summary-row/message-body states use this as their sole durable receipt,
    # so an empty string would make a completed turn disappear after refresh.
    message: str = Field(min_length=1)
    # Internal graph control. Excluding it preserves public result payloads and
    # keeps this execution signal out of the decision model's observation.
    disposition: Literal["completed", "interaction_required", "failed"] = Field(
        default="completed",
        exclude=True,
    )
    execution_outcome: Literal["committed", "not_committed", "unknown"] | None = Field(
        default=None,
        exclude=True,
    )
    """What happened to a WRITE, independently from delivery control.

    ``disposition`` answers where the graph goes next. This field answers what
    the durable execution ledger may claim. They deliberately differ for a
    recoverable input refusal and for an external write whose response was lost.
    """
    # Declared by the capability, next to the typed objects it already holds.
    # Excluded like ``disposition``: it is a decision projection, not part of
    # the public result payload.
    facts: dict[DecisionFactKey, DecisionFactValue] = Field(
        default_factory=dict,
        max_length=MAX_DECISION_FACTS,
        exclude=True,
    )
    next_action: str | None = Field(default=None, max_length=NEXT_ACTION_LIMIT)
    """Advice for the next step, in prose. See ``DecisionObservation``."""
    payload: dict[str, Any] = Field(default_factory=dict)
    body_source: DeliveredBodySource | None = Field(default=None, exclude=True)
    body_dependencies: tuple[BodyDependency, ...] = Field(default=(), exclude=True)
    resource_ref: ConversationResourceReference | None = None
    resource_refs: tuple[ConversationResourceReference, ...] = Field(
        default=(),
        max_length=MAX_CONVERSATION_SPAN_RESOURCE_REFS,
        exclude=True,
    )

    @model_validator(mode="after")
    def declared_facts_stay_bounded_and_identifier_free(self) -> "ToolResult":
        validate_decision_facts(self.facts)
        if is_failed(self.state) and set(self.facts) - {"retryable"}:
            raise ValueError(
                "a failed result may only declare retryability to the model"
            )
        return self

    @model_validator(mode="before")
    @classmethod
    def _derive_control_disposition(cls, value: Any) -> Any:
        """Bind graph control to delivery policy, with explicit emitter exceptions.

        Most waiting states have one meaning everywhere, so the policy registry
        supplies their disposition. A tool may still explicitly require an
        interaction for a state that is not universally waiting — notably a
        tool-specific interaction. Failures are classified here so
        their return to ``decide`` is deliberate rather than a default side
        effect of an unused enum value.
        """
        if not isinstance(value, dict):
            return value
        data = dict(value)
        state = data.get("state")
        declared = data.get("disposition")
        if isinstance(state, str) and is_waiting(state):
            if declared not in {None, "interaction_required"}:
                raise ValueError(
                    f"waiting state {state!r} must require an interaction"
                )
            data["disposition"] = "interaction_required"
        elif declared == "interaction_required":
            raise ValueError(
                "an interaction-required emitter must be declared waiting"
            )
        elif declared is None and isinstance(state, str):
            data["disposition"] = "failed" if is_failed(state) else "completed"
        # Retryability is a uniform property of failure, not a per-capability
        # decision value, so it is derived once here rather than declared by the
        # ten emitters that already put it in their payload. Unknown stays
        # absent: an empty facts object must not read as "not retryable".
        if (
            isinstance(state, str)
            and is_failed(state)
            and not data.get("facts")
            and type((data.get("payload") or {}).get("retryable")) is bool
        ):
            data["facts"] = {"retryable": data["payload"]["retryable"]}
        return data


# Compatibility name for capability handlers. New orchestration code should
# call this a result, not an observation: observations are model-facing.
ToolObservation = ToolResult


# Shared window for the contract, runtime, and trajectory evaluator. Eleven
# holds six reads, one internal write, one external write, two projection
# corrections, and one authorization refusal without forcing unrelated refusal
# classes to share a counter.
MAX_DECISION_OBSERVATIONS = 11
MAX_DECISION_OBSERVATION_BODIES = 1
DECISION_OBSERVATION_RECEIPT_LIMIT = DELIVERY_SUMMARY_LIMIT
DECISION_OBSERVATION_BODY_LIMIT = 6_000
# Raised from 16_000 to 18_000 when observations began recording their
# arguments (ten observations x a 200-char argument bound): without arguments,
# two calls to one capability project identically, so a model that researched
# job 1 and then job 2 could not tell its own two observations apart. Raised
# again to 18_400 when the window grew to eleven for the external-write
# budget; the declared worst shape then measures 18_367. Measured reality is
# far below either figure — a real turn's whole context was 1_340 chars
# against 11_979 of tool schemas.
MAX_DECISION_OBSERVATION_CHARS = 18_400


class DecisionObservation(ContractModel):
    """Closed, bounded observation visible to the Main Agent decision model."""

    tool_name: str = Field(pattern=r"^[a-z0-9_]+$", max_length=80)
    state: str = Field(pattern=r"^[a-z0-9_]+$", max_length=80)
    message: str = Field(
        min_length=1,
        max_length=DECISION_OBSERVATION_RECEIPT_LIMIT,
    )
    body: str | None = Field(
        default=None,
        min_length=1,
        max_length=DECISION_OBSERVATION_BODY_LIMIT,
    )
    facts: dict[DecisionFactKey, DecisionFactValue] = Field(
        default_factory=dict,
        max_length=8,
    )
    arguments: dict[str, Any] = Field(default_factory=dict)
    """The projected arguments this call was made with.

    Without them two calls to one capability are indistinguishable: researching
    job 1 and then job 2 produced two identical lines, so the model could not
    tell which observation belonged to which job — it could not read back what
    it had just done. Anthropic's context editing keeps the ``tool_use`` block,
    arguments and all, and clears only the result; this field is the same record
    for the same reason.

    These are the arguments the model wrote, not the projected ones. Projection
    turns a selector into what the handler needs — live domain objects, and the
    internal ids the boundary exists to keep away from the model — so projected
    arguments are neither safe to show nor recognizable to the model as its own
    call. Showing it back what it wrote is both safe by construction and the
    only form that answers "which of my two calls was this?".
    """
    resource_ref: ConversationResourceReference | None = Field(
        default=None,
        exclude=True,
    )
    """The stored resource this observation produced, if any.

    Excluded from the wire shape: what the model reads is the *number*, which
    only ``MainAgentContext`` can assign — it depends on how many resources the
    conversation already carries. See ``referenced_resources``.
    """
    resource_refs: tuple[ConversationResourceReference, ...] = Field(
        default=(),
        max_length=MAX_CONVERSATION_SPAN_RESOURCE_REFS,
        exclude=True,
    )
    """Resources recovered by a readback, projected only as safe handles."""
    next_action: str | None = Field(
        default=None,
        max_length=NEXT_ACTION_LIMIT,
    )
    """One sentence of advice from the capability, or nothing.

    Prose, not an enum. The pattern used to be ``^[a-z0-9_]+$``, which kept the
    field's shape but threw away the thing that makes it work: a token like
    ``review_job_research`` is not a tool name and does not say what to do with
    it, so the model can only guess, while "别再重试，先问用户" is followed
    directly. The industry pattern this field comes from is a natural-language
    hint for exactly that reason.

    Only say something ``state`` cannot. A hint that restates its state
    (``job_search_page_ready`` → ``browse_and_save_job``) is noise in every
    decision prompt that carries it; thirteen such literals were deleted rather
    than translated. What survives says something the state does not — most of
    it about *not* continuing: the call is a repeat, the retries are spent, the
    budget is gone.

    It is advice, never an instruction. The decision prompt says so, because
    today these strings are our own literals but the same field would carry an
    external MCP tool's words unchanged, and a tool must not be able to steer
    the agent. See ``docs/iteration`` for the tool-to-tool boundary note.
    """

    @model_validator(mode="after")
    def facts_stay_bounded_and_identifier_free(self) -> "DecisionObservation":
        validate_decision_facts(self.facts)
        if self.arguments:
            serialized = json.dumps(self.arguments, ensure_ascii=False, sort_keys=True)
            if len(serialized) > OBSERVATION_ARGUMENTS_LIMIT:
                object.__setattr__(self, "arguments", {})
        return self


def append_decision_observation(
    observations: tuple[DecisionObservation, ...],
    observation: DecisionObservation,
) -> tuple[DecisionObservation, ...]:
    """Append one observation and clear bodies outside the newest keep window.

    Receipts, facts, state and next_action remain intact. Body retention follows
    observation age rather than "last body-bearing result": a later plain
    result makes an earlier body old and clears it as well.
    """

    combined = (*observations, observation)[-MAX_DECISION_OBSERVATIONS:]
    keep_from = max(0, len(combined) - MAX_DECISION_OBSERVATION_BODIES)
    return tuple(
        item
        if index >= keep_from or item.body is None
        else item.model_copy(update={"body": None})
        for index, item in enumerate(combined)
    )


def decision_observation_projection(
    observations: tuple[DecisionObservation, ...],
    reference_handles: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], ...]:
    """The exact observation shape serialized into the decision prompt.

    ``reference_handles`` maps a resource id to the name the model may pass back.
    Without it an observation that stored a report says only that a report
    exists: the line that carries its name is written when the turn commits,
    which has not happened yet. With ``MAX_DECISION_OBSERVATION_BODIES = 1`` a
    second call clears the first observation's body, so a model that wanted to
    re-read the report had nothing to select it with and could only call the
    read tool bare and hope the active resource was still the one it meant.

    This is the same handle discipline the clearing itself follows — drop the
    contents, keep the reference — and the same principle behind removing
    ``REROUTE_FIELDS`` and the ``next_action`` enums: give the model something
    explicit instead of something to infer.
    """

    projected = []
    for observation in observations:
        line = observation.model_dump(mode="json", exclude_none=True)
        reference = observation.resource_ref
        if reference is not None and reference_handles is not None:
            handle = reference_handles.get(reference.resource_id)
            if handle is not None:
                line["reference"] = handle
                # Same field the catalogue and the window carry. Leaving it out
                # here would fail the case the handle was added for: two reports
                # produced in one turn project as two observations that differ
                # only in an opaque suffix, so a model asked about the first has
                # nothing to match on.
                if reference.title:
                    line["title"] = reference.title
                if reference.description:
                    line["description"] = reference.description
        if observation.resource_refs and reference_handles is not None:
            resources = []
            for recovered in observation.resource_refs:
                handle = reference_handles.get(recovered.resource_id)
                if handle is None:
                    continue
                resources.append(
                    {
                        "kind": recovered.kind,
                        "reference": handle,
                        **({"title": recovered.title} if recovered.title else {}),
                        **(
                            {"description": recovered.description}
                            if recovered.description
                            else {}
                        ),
                    }
                )
            if resources:
                line.setdefault("facts", {})["resource_refs"] = resources
        projected.append(line)
    return tuple(projected)


def decision_observation_chars(
    observations: tuple[DecisionObservation, ...],
) -> int:
    """Character cost of observations in the actual JSON prompt projection."""

    return len(
        json.dumps(
            decision_observation_projection(observations),
            ensure_ascii=False,
            sort_keys=True,
        )
    )


class EpisodeProjectionContext(ContractModel):
    detail_ref: str = Field(
        pattern=r"^episode:career_episode_[a-f0-9]{32}$"
    )
    kind: Literal[
        "mock_interview",
        "job_research",
        "application",
        "interview_round",
    ]
    occurred_at: datetime
    title: str = Field(min_length=1, max_length=80)
    synopsis: str = Field(min_length=1, max_length=400)


class WorkingNotesContext(ContractModel):
    markdown: str = Field(max_length=2000)
    revision: str = Field(pattern=r"^(?:empty|[a-f0-9]{12})$")
    clipped: bool = False
    stale_days: int | None = Field(default=None, ge=0)


PREFERENCE_EPISODE_CHAR_BUDGET = 800
_PREFERENCE_CHAR_CAP = 400


def _bounded_markdown(
    lines: list[str],
    *,
    budget: int,
    line_limit: int,
) -> str:
    """Fit ``lines`` into ``budget`` characters, saying how many were dropped.

    Dropping the tail silently let the projection lie by omission, and the lines
    most exposed are the ones a section appends last — including the counts that
    report an earlier cap, so a tight budget could hide the very notice that a
    slot cap had hidden something. Both call sites share the fix: the notice is
    paid for out of the same budget, giving up accepted lines until it fits.
    """

    accepted: list[str] = []
    for index, line in enumerate(lines):
        bounded = line if len(line) <= line_limit else line[: line_limit - 1] + "…"
        candidate = "\n".join((*accepted, bounded))
        if len(candidate) > budget:
            return _with_truncation_notice(
                accepted,
                dropped=len(lines) - index,
                budget=budget,
            )
        accepted.append(bounded)
    return "\n".join(accepted)


def _with_truncation_notice(
    accepted: list[str],
    *,
    dropped: int,
    budget: int,
) -> str:
    """Append a count of hidden lines, buying room for it by hiding more.

    A budget too small for even the first line yields nothing rather than a
    section whose only content is the notice: the caller can drop an empty
    section, and a lone count is not worth the characters it costs.
    """

    kept = list(accepted)
    while kept:
        candidate = "\n".join((*kept, f"- （另有 {dropped} 行未显示）"))
        if len(candidate) <= budget:
            return candidate
        kept.pop()
        dropped += 1
    return ""


class MainAgentContext(ContractModel):
    conversation_id: str
    received_at: datetime | None = Field(default=None, exclude=True)
    """When this turn's user message arrived; harness-only, never projected.

    A workflow's opening request is written only when the run ends, and the
    transcript places it by time, so it needs the moment it was sent rather
    than the moment the turn that started the run finished."""
    spotlight_nonce: str | None = Field(default=None, min_length=32, max_length=32)
    """Harness-only session delimiter; deliberately absent from model_context JSON."""
    profile: CareerProfileContext
    preferences: AgentPreferencesContext = AgentPreferencesContext()
    task: ConversationTaskState = ConversationTaskState()
    career_memory: CareerMemoryContext = CareerMemoryContext()
    free_text_preferences: tuple[FreeTextPreferenceContext, ...] = Field(
        default=(),
        max_length=8,
    )
    free_text_preferences_active_total: int = Field(default=0, ge=0, exclude=True)
    """How many confirmed preferences the projection selected from.

    Carried for the same reason as ``archived_resource_total``: the eight-slot
    cap cannot say it was applied, so a full list reads as everything the user
    has ever confirmed. Rendered as a count of what is missing, not a flag.
    """

    free_text_preferences_quarantined_total: int = Field(
        default=0, ge=0, exclude=True
    )
    """How many quarantined candidates the projection selected from.

    Separate from the active total because the two are rendered in different
    sections under different authority, and because a candidate held back is
    the more consequential omission: it is waiting to be confirmed, and a silent
    cut leaves it waiting indefinitely.
    """

    working_notes: WorkingNotesContext | None = None
    career_episodes: tuple[EpisodeProjectionContext, ...] = Field(
        default=(),
        max_length=5,
    )
    career_profile_budgets: CareerProfileBudgets = Field(
        default_factory=CareerProfileBudgets,
        exclude=True,
    )
    recent_messages: tuple[ConversationMessageContext, ...] = ()
    through_sequence: int = Field(default=0, ge=0)
    """Last durable message covered by ``conversation_summary``; zero if absent."""

    recent_from_sequence: int | None = Field(default=None, ge=1)
    """Sequence of the first raw message projected into the recent window."""

    archived_resource_total: int = Field(default=0, ge=0)
    """How many resources the catalogue would list uncapped.

    Zero when there is no catalogue. Sent because a capped list cannot say it
    was capped: twelve entries and nothing else read as the complete set, and a
    report that scrolled past the cap then looks like it must be one of the
    twelve. Measured behaviour is that the model reaches for the nearest
    plausible entry when it believes the thing it wants is on screen, so the
    fix is to stop the projection from implying that.

    A count rather than a flag, on the same reasoning that made ``next_action``
    prose: "there are more" leaves the model to guess how many it cannot see,
    while 12 of 27 says how far short the list falls.
    """

    archived_resources: tuple[ConversationMessageContext, ...] = Field(
        default=(), max_length=12
    )
    """Messages delivered before the recent window, as a reachable catalogue.

    Bounded by messages rather than references: one message can carry several,
    because one turn can store several reports.

    Without these a report becomes unreachable to the agent the moment its
    turn is summarised away: ``resource_refs`` live only on the original
    message, and the summary carries none. The UI kept working — it reads
    the whole transcript — so the failure was asymmetric and silent, with
    the user looking at a card the model could no longer open.

    Only the reference and bounded producer-owned metadata cross. The report itself stays
    in its entity, which is the whole point of storing a pointer.
    """

    tool_observations: tuple[DecisionObservation, ...] = Field(
        default=(),
        max_length=MAX_DECISION_OBSERVATIONS,
    )
    conversation_summary: ConversationSummaryContent | None = None
    attached_resumes: tuple[AttachedResumeContext, ...] = Field(
        default=(), max_length=8
    )
    """Exact resume versions the current user message is about, already verified."""

    attached_jobs: tuple[SavedJobCandidateContextItem, ...] = Field(
        default=(), max_length=8,
    )

    user_message: str = Field(min_length=1)
    user_interaction_id: str | None = Field(
        default=None, pattern=r"^interaction_[a-f0-9]{20}$", exclude=True
    )
    user_message_source: str | None = Field(default=None, exclude=True)
    """The message as the user sent it, kept only when ``user_message`` was clipped."""

    user_message_clipped: bool = Field(default=False, exclude=True)
    """Whether ``user_message`` is shorter than what the user sent.

    Rendered on the prompt copy the same way a clipped recent-window message
    is, so a cut-off request never reads as the whole of it.
    """

    def stored_user_message(self) -> str:
        """The message to persist or reload from, never the prompt's clipped copy."""
        return (
            self.user_message_source
            if self.user_message_source is not None
            else self.user_message
        )

    def user_input_resource_refs(self) -> tuple[ConversationResourceReference, ...]:
        """The references the stored user message carries for its attachments.

        Only the id and a display snapshot: the excerpt is turn-local and the
        file stays in the resume library.
        """
        return tuple(
            ConversationResourceReference(
                kind="resume_version",
                resource_id=item.resume_version_id,
                title=clamp(
                    f"{item.resume_name} v{item.version_number}", limit=80
                ),
                description=clamp(
                    f"{item.document_format} · {item.byte_size} bytes · "
                    f"上传于 {item.uploaded_at.isoformat()}",
                    limit=200,
                ),
            )
            for item in self.attached_resumes
        ) + tuple(
            ConversationResourceReference(
                kind="saved_job",
                resource_id=item.jd_snapshot_id,
                job_posting_id=item.job_posting_id,
                title=clamp(f"{item.company_name}｜{item.title}", limit=80),
                description=f"JD 第 {item.jd_version} 版",
            )
            for item in self.attached_jobs
            if item.jd_snapshot_id is not None
        )

    @model_validator(mode="before")
    @classmethod
    def populate_free_text_preference_totals(cls, value: object) -> object:
        """Default each total to the count projected, as the uncapped case.

        A caller that never hit the cap should not have to say so twice, and a
        context assembled without the totals would otherwise claim a truncation
        it does not have. ``ContextManager`` passes them explicitly, which is the
        only place they can exceed what is projected.
        """
        if not isinstance(value, dict):
            return value
        projected = value.get("free_text_preferences", ())
        try:
            statuses = [
                item.status
                if hasattr(item, "status")
                else item.get("status")
                for item in projected
            ]
        except (AttributeError, TypeError):
            return value
        filled = dict(value)
        if "free_text_preferences_active_total" not in filled:
            filled["free_text_preferences_active_total"] = sum(
                status == "active" for status in statuses
            )
        if "free_text_preferences_quarantined_total" not in filled:
            filled["free_text_preferences_quarantined_total"] = sum(
                status == "quarantined" for status in statuses
            )
        return filled

    @model_validator(mode="after")
    def observation_bodies_are_only_on_the_newest_item(self) -> "MainAgentContext":
        if self.recent_from_sequence is not None and (
            self.recent_from_sequence <= self.through_sequence
        ):
            raise ValueError("recent messages must begin after the summary boundary")
        stale = self.tool_observations[:-MAX_DECISION_OBSERVATION_BODIES]
        if any(item.body is not None for item in stale):
            raise ValueError(
                "only the newest decision observation may retain a body"
            )
        if decision_observation_chars(self.tool_observations) > (
            MAX_DECISION_OBSERVATION_CHARS
        ):
            raise ValueError("decision observations exceed the character budget")
        if self.free_text_preferences_active_total < sum(
            item.status == "active" for item in self.free_text_preferences
        ):
            raise ValueError(
                "free_text_preferences_active_total cannot be smaller than "
                "projected confirmed preferences"
            )
        if self.free_text_preferences_quarantined_total < sum(
            item.status == "quarantined" for item in self.free_text_preferences
        ):
            raise ValueError(
                "free_text_preferences_quarantined_total cannot be smaller "
                "than projected quarantined preferences"
            )
        return self

    def referenced_resources(self) -> tuple[ConversationResourceReference, ...]:
        """Every resource the model can name this turn.

        Archived catalogue, then the recent window, then what this turn produced
        and no stored line names yet. The order is no longer load-bearing — a
        handle is derived from the resource, not from where it sits — but it is
        kept because it reads the way the conversation happened.

        A resource already named is skipped rather than listed twice: reading an
        old report back produces a reference to something the catalogue already
        holds, and its handle would be identical anyway.
        """
        references: list[ConversationResourceReference] = []
        seen: set[str] = set()
        for reference in (
            *(
                reference
                for message in (*self.archived_resources, *self.recent_messages)
                for reference in message.resource_refs
            ),
            *self.user_input_resource_refs(),
            *(
                observation.resource_ref
                for observation in self.tool_observations
                if observation.resource_ref is not None
            ),
            *(
                reference
                for observation in self.tool_observations
                for reference in observation.resource_refs
            ),
        ):
            if reference.resource_id in seen:
                continue
            seen.add(reference.resource_id)
            references.append(reference)
        return tuple(references)

    def reference_handle(self, reference: ConversationResourceReference) -> str:
        """The name the model may pass back for one resource.

        ``<kind prefix>_<derived suffix>``, following the shape every published
        tool API uses for the same job — ``file_abc123``, ``toolu_01A...``, an
        MCP URI. Two of that shape's three properties are adopted and one is
        not:

        * **Unguessable.** The whole reason for the migration. An ordinal is
          guessable by construction, and a model that has never been given one
          will still write ``1``: recorded behaviour, not a worry. It then
          resolves — to whatever sits first — because the number is real and its
          kind matches. Nothing in a positional scheme can tell "the number I
          was given" from "the number I counted to".
        * **Prefixed by kind.** So a mistake reads as "that is a report handle
          and you used it where a preparation was wanted" instead of "not
          found", and so the model need not pair a handle with a separate
          ``kind`` field to know what it holds.
        * **Not the primary key.** Published APIs expose the id itself; we do
          not, because internal identifiers stay out of the projection. The
          handle is derived one-way from the resource id instead, which also
          means no stored row changes and nothing has to be migrated.

        Derived rather than random so the same report keeps the same handle
        across turns and restarts, with no new column to store it in. The
        conversation id is the salt, which makes handles differ between
        conversations; that is scope hygiene, not a security boundary — the
        property being bought is that a handle cannot be reached by counting,
        not that an adversary holding the transcript could not recompute one.
        """
        digest = hmac.new(
            self.conversation_id.encode("utf-8"),
            reference.resource_id.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"{_HANDLE_PREFIXES[reference.kind]}_{digest[:_HANDLE_SUFFIX_LENGTH]}"

    def reference_handles(self) -> dict[str, str]:
        """Handle → resource id, for everything nameable this turn.

        Built once and read by both the projection and the resolver, so what the
        model is shown and what comes back cannot be two different derivations.

        A collision is resolved by lengthening, not by falling back to order:
        the point of the scheme is that a handle carries no positional meaning.
        Two six-hex-digit suffixes colliding inside one conversation's handful
        of resources is remote, but it would silently hand back the wrong report
        — the exact failure being migrated away from — so it is detected here
        rather than assumed away.
        """
        handles: dict[str, str] = {}
        for reference in self.referenced_resources():
            handle = self.reference_handle(reference)
            length = _HANDLE_SUFFIX_LENGTH
            while handle in handles and handles[handle] != reference.resource_id:
                length += 2
                handle = self._lengthened_handle(reference, length)
            handles[handle] = reference.resource_id
        return handles

    def _lengthened_handle(
        self, reference: ConversationResourceReference, length: int
    ) -> str:
        digest = hmac.new(
            self.conversation_id.encode("utf-8"),
            reference.resource_id.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"{_HANDLE_PREFIXES[reference.kind]}_{digest[:length]}"

    def resolve_reference(self, *, reference: str, kind: str) -> str:
        """Turn a handle the model wrote back into the internal resource id.

        The kind is still checked rather than trusted. The prefix tells the model
        what it is holding, but it is a hint in text the model itself produced;
        the server has to verify against what it actually handed out.
        """
        resource_id = self.reference_handles().get(reference)
        if resource_id is None:
            raise ValueError(f"unknown resource reference '{reference}'")
        held = next(
            item
            for item in self.referenced_resources()
            if item.resource_id == resource_id
        )
        if held.kind != kind:
            raise ValueError(
                f"resource reference '{reference}' is a {held.kind}, not a {kind}"
            )
        return resource_id

    def model_context(self) -> dict[str, Any]:
        model_messages = []
        # One derivation, read here and by the resolver. Nothing depends on the
        # order of this walk any more: a handle names its resource, so the
        # catalogue and the window cannot disagree about what a name means.
        handles = {
            resource_id: handle
            for handle, resource_id in self.reference_handles().items()
        }
        archived = [
            {
                "kind": reference.kind,
                "reference": handles[reference.resource_id],
                "delivered_at": message.created_at.isoformat(),
                **({"title": reference.title} if reference.title else {}),
                **(
                    {"description": reference.description}
                    if reference.description
                    else {}
                ),
            }
            for message in self.archived_resources
            for reference in message.resource_refs
        ]
        unlisted_count = max(0, self.archived_resource_total - len(archived))
        for message in self.recent_messages:
            projected = {
                "role": message.role,
                "content": message.content,
                "created_at": message.created_at.isoformat(),
            }
            if message.resource_refs:
                resources = [
                    {
                        "kind": reference.kind,
                        "reference": handles[reference.resource_id],
                        **({"title": reference.title} if reference.title else {}),
                        **(
                            {"description": reference.description}
                            if reference.description
                            else {}
                        ),
                    }
                    for reference in message.resource_refs
                ]
                # Plural because one turn can store two reports. The key stays
                # ``resources`` in both cases so the model reads one shape.
                projected["resources"] = resources
            model_messages.append(projected)
        if self.profile.current_targets_total != len(self.profile.current_targets):
            raise ValueError(
                "career profile projection requires the complete current-target set"
            )
        career_memory = self.career_memory.tier_one_projection(
            token_budget=self.career_profile_budgets.records_input_units,
        )
        career_memory.update(_memory_overflow_notice(career_memory))
        active_free_text_preferences = [
            item
            for item in self.free_text_preferences
            if item.status == "active" and item.confirmed_at is not None
        ]
        quarantined_free_text_preferences = [
            item
            for item in self.free_text_preferences
            if item.status == "quarantined"
        ][:3]
        preference_lines = ["## 已确认的自由文本偏好（可用于推荐）"]
        preference_lines.extend(
            f"- {item.statement}（确认于 {item.confirmed_at.isoformat()}）"
            for item in active_free_text_preferences
            if item.confirmed_at is not None
        )
        if not active_free_text_preferences:
            preference_lines.append("- 无")
        # Only when something was actually held back. A line that is always
        # present would be a constant, and a constant tells the model nothing
        # about this turn.
        hidden_active = max(
            0,
            self.free_text_preferences_active_total
            - len(active_free_text_preferences),
        )
        if hidden_active:
            preference_lines.append(
                f"- （另有 {hidden_active} 条已确认偏好未列出）"
            )
        preference_lines.append("## 待确认偏好（隔离态，不得用于筛选、排序或推荐）")
        preference_lines.extend(
            f"{index}. {item.statement}"
            for index, item in enumerate(
                quarantined_free_text_preferences,
                start=1,
            )
        )
        if not quarantined_free_text_preferences:
            preference_lines.append("- 无")
        hidden_quarantined = max(
            0,
            self.free_text_preferences_quarantined_total
            - len(quarantined_free_text_preferences),
        )
        if hidden_quarantined:
            preference_lines.append(
                f"- （另有 {hidden_quarantined} 条待确认偏好未列出）"
            )
        preference_markdown = _bounded_markdown(
            preference_lines,
            budget=_PREFERENCE_CHAR_CAP,
            line_limit=220,
        )
        episode_budget = PREFERENCE_EPISODE_CHAR_BUDGET - len(
            preference_markdown
        )
        episode_lines = [
            "## 相关的过往求职事件（渐进披露目录）",
            "这里只是摘要；需要细节时调用 search_career_episodes，并传入 detail_ref。",
        ]
        episode_lines.extend(
            f"- [{item.kind}] {item.title}：{item.synopsis} "
            f"[detail_ref={item.detail_ref}]"
            for item in self.career_episodes
        )
        bounded_episode_markdown = (
            _bounded_markdown(
                episode_lines,
                budget=episode_budget,
                line_limit=220,
            )
            if self.career_episodes
            else ""
        )
        # A catalogue with no entry is two headings and, now that bounding says
        # what it dropped, possibly a count. Checking for an actual entry rather
        # than a line total keeps that count from passing as content.
        episode_markdown = (
            bounded_episode_markdown
            if "detail_ref=" in bounded_episode_markdown
            else ""
        )
        return {
            "career_profile": career_profile_memory_files(self.profile),
            "career_memory": career_memory,
            "free_text_preferences": preference_markdown,
            **(
                {
                    "working_notes": {
                        "revision": self.working_notes.revision,
                        "markdown": self.working_notes.markdown,
                        **({"clipped": True} if self.working_notes.clipped else {}),
                        **(
                            {"stale_days": self.working_notes.stale_days}
                            if self.working_notes.stale_days is not None
                            else {}
                        ),
                    }
                }
                if self.working_notes is not None
                else {}
            ),
            **(
                {"career_episodes": episode_markdown}
                if episode_markdown
                else {}
            ),
            "preferences": {
                "boss_search": self.preferences.boss_search,
            },
            **(
                {
                    "behavior_policy": {
                        **(
                            {
                                "application_confirmation": (
                                    self.preferences.application_confirmation
                                )
                            }
                            if self.preferences.application_confirmation
                            != "on_user_report"
                            else {}
                        ),
                        **(
                            {
                                "confirm_before": list(
                                    self.preferences.behavior_policy.confirm_before
                                )
                            }
                            if self.preferences.behavior_policy.confirm_before
                            else {}
                        ),
                    }
                }
                if self.preferences.application_confirmation != "on_user_report"
                or self.preferences.behavior_policy.confirm_before
                else {}
            ),
            "task": {
                **self.task.active_resource_flags(),
                "active_calendar_proposal_expires_at": (
                    self.task.active_calendar_proposal_expires_at.isoformat()
                    if self.task.active_calendar_proposal_expires_at is not None
                    else None
                ),
                "active_workflow": self.task.active_workflow,
                "active_saved_job": (
                    {
                        "title": focus.title,
                        "company_name": focus.company_name,
                        "jd_version": focus.jd_version,
                        "resource_status": (
                            "readable" if focus.readable else "unavailable"
                        ),
                    }
                    if (focus := self.task.focused_saved_job()) is not None
                    else None
                ),
                "tool_profile": self.task.tool_profile,
                "phase": self.task.phase,
                "email_sync_phase": self.task.email_sync_phase,
                "manual_search_query": self.task.manual_search_query,
                "candidates": [
                    {
                        "selection_index": index,
                        "title": candidate.title,
                        "company_name": candidate.company_name,
                        "city": candidate.city,
                        "salary": candidate.salary,
                    }
                    for index, candidate in enumerate(self.task.candidates, start=1)
                ],
                "resume_job_match_status": self.task.resume_job_match_status,
                "job_analysis_status": self.task.job_analysis_status,
                "resume_tailoring_status": self.task.resume_tailoring_status,
                "active_application_status": self.task.active_application_status,
                "application_candidates": [
                    {
                        "selection_index": index,
                        "title": candidate.title,
                        "company_name": candidate.company_name,
                        "status": candidate.status,
                    }
                    for index, candidate in enumerate(
                        self.task.application_candidates, start=1
                    )
                ],
                "interview_candidates": [
                    {
                        "selection_index": index,
                        "sequence_number": candidate.sequence_number,
                        "employer_label": candidate.employer_label,
                        "status": candidate.status,
                        "scheduled_start": (
                            candidate.scheduled_start.isoformat()
                            if candidate.scheduled_start is not None
                            else None
                        ),
                    }
                    for index, candidate in enumerate(
                        self.task.interview_candidates, start=1
                    )
                ],
                "interview_preparation_ready": (
                    self.task.active_interview_preparation_id is not None
                ),
                "job_research_status": self.task.job_research_status,
                "action_candidates": [
                    {
                        "selection_index": index,
                        "action_type": candidate.action_type,
                        "source_type": candidate.source_type,
                        "title": candidate.title,
                        "status": candidate.status,
                        "due_at": (
                            candidate.due_at.isoformat()
                            if candidate.due_at is not None
                            else None
                        ),
                    }
                    for index, candidate in enumerate(
                        self.task.action_candidates, start=1
                    )
                ],
                "calendar_accounts": [
                    {
                        "selection_index": index,
                        "provider": candidate.provider,
                        "email_address": candidate.email_address,
                        "calendar_id": candidate.calendar_id,
                    }
                    for index, candidate in enumerate(
                        self.task.calendar_account_candidates, start=1
                    )
                ],
                "saved_jobs": [
                    {
                        "selection_index": index,
                        "title": candidate.title,
                        "company_name": candidate.company_name,
                        "city": candidate.city,
                        "salary": candidate.salary,
                    }
                    for index, candidate in enumerate(
                        self.task.saved_job_candidates, start=1
                    )
                ],
                "target_roles": [
                    {
                        "selection_index": index,
                        "title": candidate.title,
                        "priority": candidate.priority,
                        "status": candidate.status,
                    }
                    for index, candidate in enumerate(
                        self.task.target_role_candidates, start=1
                    )
                ],
                "resumes": [
                    {
                        "selection_index": index,
                        "name": candidate.name,
                        "status": candidate.status,
                    }
                    for index, candidate in enumerate(
                        self.task.resume_candidates, start=1
                    )
                ],
                "resume_versions": [
                    {
                        "selection_index": index,
                        **(
                            {"resume_name": candidate.resume_name}
                            if candidate.resume_name is not None
                            else {}
                        ),
                        "version_number": candidate.version_number,
                        "source_type": candidate.source_type,
                        "document_format": candidate.document_format,
                        "byte_size": candidate.byte_size,
                    }
                    for index, candidate in enumerate(
                        self.task.resume_version_candidates, start=1
                    )
                ],
                "email_events": [
                    {
                        "selection_index": index,
                        "event_type": candidate.event_type,
                        "status": candidate.status,
                        "summary": candidate.summary,
                    }
                    for index, candidate in enumerate(
                        self.task.email_event_candidates, start=1
                    )
                ],
            },
            # Internal resource IDs stay in durable messages for the UI and
            # projection layer. The decision model receives only turn-local
            # indexes, matching every other selectable object contract.
            "archived_reports": {
                "items": tuple(archived),
                "unlisted": (
                    f"另有 {unlisted_count} 份更早的调研未列出，无法按引用取回。"
                    if unlisted_count
                    else None
                ),
            },
            "recent_messages": tuple(model_messages),
            **(
                {
                    "through_sequence": self.through_sequence,
                    "recent_from_sequence": self.recent_from_sequence,
                }
                if self.through_sequence
                or (
                    self.recent_from_sequence is not None
                    and self.recent_from_sequence > 1
                )
                else {}
            ),
            "tool_observations": decision_observation_projection(
                self.tool_observations,
                {
                    resource_id: handle
                    for handle, resource_id in self.reference_handles().items()
                }
            ),
            "conversation_summary": (
                self.conversation_summary.model_dump(mode="json")
                if self.conversation_summary
                else None
            ),
            **(
                {
                    "attached_resumes": tuple(
                        {
                            "reference": handles[item.resume_version_id],
                            **item.model_dump(mode="json"),
                        }
                        for item in self.attached_resumes
                    )
                }
                if self.attached_resumes
                else {}
            ),
            "user_message": self.user_message,
        }


class OpenJobSearchToolArguments(ContractModel):
    platform: Literal["boss"] = "boss"
    keyword: str = Field(min_length=1, max_length=100)
    city: str | None = Field(default=None, min_length=1, max_length=40)


class ReadConversationSpanToolArguments(ContractModel):
    from_sequence: int = Field(ge=1)
    through_sequence: int = Field(ge=1)
    query: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description=(
            "Focused terms from the user's request, such as 目标公司 or Rust. "
            "Use this for long omitted ranges; omit it only for an exact "
            "sequence span the user named."
        ),
    )

    @model_validator(mode="after")
    def require_forward_span(self) -> "ReadConversationSpanToolArguments":
        if self.from_sequence > self.through_sequence:
            raise ValueError("from_sequence cannot exceed through_sequence")
        return self


class ResolveClaimSourceToolArguments(ContractModel):
    source_ref: str = Field(pattern=r"^evidence_[a-f0-9]{24}$")


class GetCareerMemoryDetailToolArguments(ContractModel):
    detail_ref: str = Field(pattern=r"^detail_[a-f0-9]{24}$")


class SearchCareerMemoryToolArguments(ContractModel):
    query: str = Field(
        min_length=3,
        max_length=200,
        description=(
            "Focused terms expected inside active career claims omitted from "
            "the bounded Tier-1 window."
        ),
    )
    limit: int = Field(default=8, ge=1, le=20)
    cursor: str | None = Field(
        default=None,
        pattern=r"^memory_[a-f0-9]{8}_[a-f0-9]{8}$",
        description="Opaque next-page cursor returned by an earlier identical query.",
    )


class UpdateWorkingNotesToolArguments(ContractModel):
    expected_revision: str = Field(
        pattern=r"^(?:empty|[a-f0-9]{12})$",
        description="Revision from the current working_notes context projection.",
    )
    markdown: str = Field(
        max_length=2000,
        description=(
            "Complete replacement markdown. It may guide questions and response "
            "style, but never filtering, ranking, applications, or other actions."
        ),
    )


class SearchCareerEpisodesToolArguments(ContractModel):
    """Bounded L1 recall across completed workflows and past conversations."""

    detail_ref: str | None = Field(
        default=None,
        pattern=r"^episode:career_episode_[a-f0-9]{32}$",
        description=(
            "Exact projected episode detail_ref. When present, dereference that "
            "episode instead of running a broad query."
        ),
    )
    query: str = Field(
        default="",
        max_length=200,
        description=(
            "Focused terms from the remembered event. Leave empty only when "
            "listing recent episodes within the supplied filters."
        ),
    )
    start_datetime: datetime | None = Field(
        default=None,
        description="Inclusive lower bound for when the episode occurred.",
    )
    end_datetime: datetime | None = Field(
        default=None,
        description="Inclusive upper bound for when the episode occurred.",
    )
    kinds: tuple[
        Literal[
            "mock_interview",
            "job_research",
            "application",
            "interview_round",
            "resume_analysis",
            "intent_confirmation",
            "resume_tailoring",
        ],
        ...,
    ] = Field(
        default=(),
        max_length=7,
        description="Optional episode-type filters.",
    )
    top_k: int = Field(default=8, ge=1, le=20)

    @model_validator(mode="after")
    def time_window_is_forward(self) -> "SearchCareerEpisodesToolArguments":
        for name, value in (
            ("start_datetime", self.start_datetime),
            ("end_datetime", self.end_datetime),
        ):
            if value is not None and value.utcoffset() is None:
                raise ValueError(f"{name} must include a timezone offset")
        if (
            self.start_datetime is not None
            and self.end_datetime is not None
            and self.start_datetime > self.end_datetime
        ):
            raise ValueError("start_datetime cannot exceed end_datetime")
        return self


class SearchCareerHistoryToolArguments(ContractModel):
    query: str = Field(
        min_length=3,
        max_length=200,
        description=(
            "Focused terms expected inside the earlier claim. Do not pass a "
            "generic request such as 'what did I say before'."
        ),
    )
    limit: int = Field(default=8, ge=1, le=20)
    cursor: str | None = Field(
        default=None,
        pattern=r"^history_[a-f0-9]{8}_[a-f0-9]{8}$",
        description="Opaque next-page cursor returned by an earlier identical query.",
    )


class ProposeMemoryTombstoneToolArguments(ContractModel):
    detail_ref: str = Field(pattern=r"^detail_[a-f0-9]{24}$")
    reason: str = Field(min_length=1, max_length=2000)


class ConfirmMemoryTombstoneToolArguments(ContractModel):
    pass


class ProposeConstraintRetirementToolArguments(ContractModel):
    constraint: str = Field(min_length=1, max_length=SUMMARY_ITEM_MAX_CHARS)
    reason: str = Field(min_length=1, max_length=2000)


class ConfirmConstraintRetirementToolArguments(ContractModel):
    pass


class FetchArchivedConstraintsToolArguments(ContractModel):
    pass


class ProposeMemoryAmendmentToolArguments(ContractModel):
    detail_ref: str = Field(pattern=r"^detail_[a-f0-9]{24}$")
    new_claim: str = Field(min_length=1, max_length=32_000)
    reason: str = Field(min_length=1, max_length=2000)


class ConfirmMemoryAmendmentToolArguments(ContractModel):
    pass


class ProposeCareerFactToolArguments(ContractModel):
    record_selection_index: SelectionIndex
    claim: str = Field(min_length=1, max_length=2000)
    reason: str = Field(min_length=1, max_length=2000)
    user_quote: str | None = Field(
        default=None, min_length=4, max_length=500,
        description=(
            "Exact verbatim excerpt from the user's own message or questionnaire "
            "answer supporting this claim. Omit when the claim is an inference."
        ),
    )


class ConfirmCareerFactToolArguments(ContractModel):
    pass


class FindSavedJobsToolArguments(ContractModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=20)


class GetSavedJobToolArguments(ContractModel):
    job_posting_id: str | None = Field(default=None, min_length=1)
    selection_index: SelectionIndex | None = None


class AnalyzeJobToolArguments(ContractModel):
    job_posting_id: str | None = Field(default=None, min_length=1)
    selection_index: SelectionIndex | None = None


class CorrectJobRequirementTierToolArguments(ContractModel):
    analysis_id: str = Field(min_length=1)
    requirement_id: str = Field(pattern=r"^job_requirement_[a-f0-9]{20}$")
    tier: Literal["S", "A", "B", "C"]
    reason: str = Field(min_length=1, max_length=1000)
    confirmation: Literal["confirm", "correct"] = "correct"


class ResearchJobToolArguments(ContractModel):
    job_posting_id: str | None = Field(default=None, min_length=1)
    selection_index: SelectionIndex | None = None
    focus: str | None = Field(default=None, min_length=1, max_length=1000)
    user_provided_context: str | None = Field(
        default=None,
        min_length=1,
        max_length=4000,
        description=(
            "Relevant business or product clues the user explicitly chose to use, "
            "for example something they heard in an interview. This is unverified "
            "user-reported context, not a public fact."
        ),
    )
    max_sources: int = Field(default=8, ge=2, le=15)


class RetryJobResearchToolArguments(ContractModel):
    run_id: str | None = Field(default=None, min_length=1)


IMPLICIT_REQUEST_KEY = "implicit_request"
"""Set by projection, never by the model: the arguments model forbids it."""


class GetJobResearchToolArguments(ContractModel):
    """Selectors for reading back one job-research report.

    ``reference`` identifies an exact delivered report. ``selection_index``
    selects a saved job and reads its company's latest available report, which
    may differ from a requested historical version. Omitting both uses the
    active report or job. Internal ids are stripped from model-facing schemas
    and supplied by argument projection.
    """

    report_id: str | None = Field(default=None, min_length=1)
    job_posting_id: str | None = Field(default=None, min_length=1)
    selection_index: SelectionIndex | None = None
    reference: ResourceHandle | None = None

    @model_validator(mode="after")
    def validate_selector(self) -> "GetJobResearchToolArguments":
        if self.reference is not None and self.selection_index is not None:
            raise ValueError("use either reference or selection_index")
        return self


class ListTargetRolesToolArguments(ContractModel):
    pass


class ListResumesToolArguments(ContractModel):
    target_role_id: str | None = Field(default=None, min_length=1)
    target_role_selection_index: SelectionIndex | None = None


class GetResumeMetadataToolArguments(ContractModel):
    resume_id: str | None = Field(default=None, min_length=1)
    selection_index: SelectionIndex | None = None


MainAgentSkill = Literal["resume-critique"]
"""Skills the main agent itself may load. Specialist skills (job research,
mock interview, tailoring) belong to their workers and are not listed."""


class LoadSkillToolArguments(ContractModel):
    skill: MainAgentSkill = Field(
        description=(
            "resume-critique: how to critique a resume as a document when the "
            "user asks to review, critique or improve it without naming a job."
        )
    )


class MatchResumeToJobToolArguments(ContractModel):
    resume_version_id: str | None = Field(default=None, min_length=1)
    job_posting_id: str | None = Field(default=None, min_length=1)
    resume_version_selection_index: SelectionIndex | None = None
    job_selection_index: SelectionIndex | None = None


class ProposeJobIntentToolArguments(ContractModel):
    target_role_selection_index: SelectionIndex | None = None
    pref_scope: str = Field(
        default="global",
        pattern=r"^(?:global|[a-z][a-z0-9_.:-]*)$",
        max_length=120,
        description=(
            "Use global unless the user explicitly limits this preference to "
            "a named situation or domain."
        ),
    )
    timescale: Literal["permanent", "situational"] = "permanent"
    city: str | None = Field(default=None, min_length=1, max_length=40)
    salary_expectation: str | None = Field(default=None, min_length=1, max_length=100)
    experience: str | None = Field(default=None, min_length=1, max_length=100)
    education: str | None = Field(default=None, min_length=1, max_length=100)
    hard_constraints: tuple[HardConstraintContext, ...] = Field(
        default=(),
        max_length=8,
    )


class ConfirmJobIntentToolArguments(ContractModel):
    pass


class ProposeFreeTextPreferenceConfirmationToolArguments(ContractModel):
    selection_index: SelectionIndex


class ConfirmFreeTextPreferenceToolArguments(ContractModel):
    scope_choice: Literal[
        "person_stable",
        "person_default",
        "person_situational",
        "role",
        "situational",
    ] | None = None
    scope_domain: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_.:-]{0,79}$",
    )

    @model_validator(mode="after")
    def role_choice_names_its_domain(
        self,
    ) -> "ConfirmFreeTextPreferenceToolArguments":
        if self.scope_choice == "role" and self.scope_domain is None:
            raise ValueError("role scope requires scope_domain")
        if self.scope_choice != "role" and self.scope_domain is not None:
            raise ValueError("scope_domain is only valid for role scope")
        return self


class CompareSavedJobsToolArguments(ContractModel):
    job_selection_indices: tuple[SelectionIndex, ...] = Field(
        min_length=2, max_length=10
    )


class GetResumeJobMatchToolArguments(ContractModel):
    match_id: str | None = Field(default=None, min_length=1)


class DraftResumeTailoringToolArguments(ContractModel):
    match_id: str | None = Field(default=None, min_length=1)
    tailoring_goal: str | None = Field(default=None, min_length=1, max_length=2000)


class GetResumeTailoringDraftToolArguments(ContractModel):
    draft_id: str | None = Field(default=None, min_length=1)


class ReviewResumeTailoringToolArguments(ContractModel):
    draft_id: str | None = Field(default=None, min_length=1)
    accepted_change_indices: tuple[SelectionIndex, ...] = Field(
        default=(), max_length=30
    )
    rejected_change_indices: tuple[SelectionIndex, ...] = Field(
        default=(), max_length=30
    )
    feedback: str | None = Field(default=None, min_length=1, max_length=2000)

    @model_validator(mode="after")
    def validate_decisions(self) -> ReviewResumeTailoringToolArguments:
        accepted = self.accepted_change_indices
        rejected = self.rejected_change_indices
        if not accepted and not rejected:
            raise ValueError("at least one accepted or rejected change index is required")
        if any(index < 1 for index in (*accepted, *rejected)):
            raise ValueError("change indices must be positive")
        if len(set(accepted)) != len(accepted) or len(set(rejected)) != len(rejected):
            raise ValueError("change indices must be unique")
        if set(accepted).intersection(rejected):
            raise ValueError("a change cannot be both accepted and rejected")
        return self


class ReviseResumeTailoringToolArguments(ContractModel):
    draft_id: str | None = Field(default=None, min_length=1)
    feedback: str = Field(min_length=1, max_length=2000)


class FinalizeResumeTailoringToolArguments(ContractModel):
    draft_id: str | None = Field(default=None, min_length=1)


class ExportResumeArtifactToolArguments(ContractModel):
    resume_version_id: str | None = Field(default=None, min_length=1)


class CreateApplicationToolArguments(ContractModel):
    job_posting_id: str | None = Field(default=None, min_length=1)
    resume_version_id: str | None = Field(default=None, min_length=1)
    job_selection_index: SelectionIndex | None = None
    resume_version_selection_index: SelectionIndex | None = None
    submitted_at: datetime | None = None
    note: str | None = Field(default=None, min_length=1, max_length=2000)


class UpdateOwnerSettingsToolArguments(ContractModel):
    """An agent proposal; the runtime always seals it for owner review."""

    boss_search: Literal["explicit_request_only", "allowed"] | None = None
    application_confirmation: Literal["always_ask", "on_user_report"] | None = None
    confirm_before: ConfirmBefore | None = Field(
        default=None,
        description=(
            "Full replacement list of WRITE capability names the owner wants to "
            "approve one by one before they run. Pass an empty list to clear."
        ),
    )

    @model_validator(mode="after")
    def changes_something(self) -> "UpdateOwnerSettingsToolArguments":
        if (
            self.boss_search is None
            and self.application_confirmation is None
            and self.confirm_before is None
        ):
            raise ValueError("an owner-settings proposal must change at least one setting")
        return self

    @field_validator("confirm_before")
    @classmethod
    def normalise_confirm_before(cls, value: ConfirmBefore | None) -> ConfirmBefore | None:
        return None if value is None else canonical_confirm_before(value)


class UpdateApplicationStatusToolArguments(ContractModel):
    application_id: str | None = Field(default=None, min_length=1)
    selection_index: SelectionIndex | None = None
    status: ApplicationStatus
    note: str | None = Field(default=None, min_length=1, max_length=2000)

    @model_validator(mode="after")
    def validate_application_selector(self) -> "UpdateApplicationStatusToolArguments":
        if self.application_id is not None and self.selection_index is not None:
            raise ValueError("use either application_id or selection_index")
        return self


class ListApplicationsToolArguments(ContractModel):
    statuses: tuple[ApplicationStatus, ...] = Field(default=(), max_length=8)
    limit: int = Field(default=20, ge=1, le=50)

    @model_validator(mode="after")
    def validate_statuses(self) -> "ListApplicationsToolArguments":
        if len(set(self.statuses)) != len(self.statuses):
            raise ValueError("application statuses must be unique")
        return self


class GetApplicationToolArguments(ContractModel):
    application_id: str | None = Field(default=None, min_length=1)
    selection_index: SelectionIndex | None = None

    @model_validator(mode="after")
    def validate_application_selector(self) -> "GetApplicationToolArguments":
        if self.application_id is not None and self.selection_index is not None:
            raise ValueError("use either application_id or selection_index")
        return self


class SyncApplicationEmailsToolArguments(ContractModel):
    account_id: str | None = Field(default=None, min_length=1)


class ListEmailEventsToolArguments(ContractModel):
    status: EmailEventStatus | None = None
    limit: int = Field(default=20, ge=1, le=50)


class ResolveEmailEventToolArguments(ContractModel):
    event_id: str | None = Field(default=None, min_length=1)
    selection_index: SelectionIndex | None = None
    approve: bool
    application_id: str | None = Field(default=None, min_length=1)
    interview_round_id: str | None = Field(default=None, min_length=1)


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


class GetDailyBriefToolArguments(ContractModel):
    timezone: str = Field(default="Asia/Shanghai", min_length=1, max_length=100)


class ListActionItemsToolArguments(ContractModel):
    statuses: tuple[ActionStatus, ...] = Field(
        default=("open", "snoozed"), min_length=1, max_length=4
    )
    limit: int = Field(default=50, ge=1, le=100)
    timezone: str = Field(default="Asia/Shanghai", min_length=1, max_length=100)


class ResolveActionItemToolArguments(ContractModel):
    action_item_id: str | None = Field(default=None, min_length=1)
    selection_index: SelectionIndex | None = None

    @model_validator(mode="after")
    def validate_selector(self) -> "ResolveActionItemToolArguments":
        if self.action_item_id is not None and self.selection_index is not None:
            raise ValueError("use either action_item_id or selection_index")
        return self


class SnoozeActionItemToolArguments(ResolveActionItemToolArguments):
    snoozed_until: datetime


class ListCalendarAccountsToolArguments(ContractModel):
    pass


class ListCalendarLinksToolArguments(ContractModel):
    pass


class PrepareInterviewCalendarSyncToolArguments(ContractModel):
    interview_round_id: str | None = Field(default=None, min_length=1)
    interview_selection_index: SelectionIndex | None = None
    calendar_account_id: str | None = Field(default=None, min_length=1)
    calendar_account_selection_index: SelectionIndex | None = None

    @model_validator(mode="after")
    def validate_selectors(self) -> "PrepareInterviewCalendarSyncToolArguments":
        if self.interview_round_id and self.interview_selection_index:
            raise ValueError("use one interview selector")
        if self.calendar_account_id and self.calendar_account_selection_index:
            raise ValueError("use one calendar account selector")
        return self


class GetCalendarProposalToolArguments(ContractModel):
    proposal_id: str | None = Field(default=None, min_length=1)


class ExecuteCalendarProposalToolArguments(GetCalendarProposalToolArguments):
    pass


class StartMockInterviewWorkflowInput(ContractModel):
    user_id: str = Field(min_length=1)
    application_id: str | None = Field(default=None, min_length=1)
    interview_round_id: str | None = Field(default=None, min_length=1)
    interview_type: MockInterviewType
    target_role: str | None = Field(default=None, min_length=1, max_length=300)
    max_primary_questions: int = Field(ge=1, le=20)
    max_follow_ups_per_question: int = Field(ge=0, le=5)
    conversation_id: str | None = Field(default=None, min_length=1)
    # Free practice only. ``required`` means nobody has said which resume to
    # use yet, so the run must not start; the tool asks instead of guessing.
    resume_choice: Literal["chosen", "none", "required"] = "chosen"
    resume_version_id: str | None = Field(default=None, min_length=1)
    # ``check`` means the user named a company but no job: the tool offers any
    # saved jobs at that company before starting company-only practice.
    job_choice: Literal["chosen", "none", "check"] = "none"
    job_posting_id: str | None = Field(default=None, min_length=1)
    jd_snapshot_id: str | None = Field(default=None, min_length=1)
    target_company: str | None = Field(default=None, min_length=1, max_length=200)

    @model_validator(mode="after")
    def validate_resume_choice(self) -> "StartMockInterviewWorkflowInput":
        if (self.resume_choice == "chosen") != (self.resume_version_id is not None) and (
            self.application_id is None
        ):
            raise ValueError("a chosen resume needs its version, and only then")
        return self


class ToolCall(ContractModel):
    name: str
    arguments: dict[str, Any] = {}


class AgentDecision(ContractModel):
    action: Literal["ask_user", "questionnaire", "tool_call", "final"]
    message: str | None = None
    tool_call: ToolCall | None = None
    questions: tuple[UserQuestion, ...] = Field(default=(), max_length=8)
    selection_source: Literal["latest_tool_result"] | None = None

    @model_validator(mode="after")
    def _questionnaire_shape(self) -> "AgentDecision":
        if self.action == "questionnaire":
            if not 2 <= len(self.questions) <= 8:
                raise ValueError("questionnaire needs 2-8 questions")
            if tuple(item.question_id for item in self.questions) != tuple(
                f"q{index}" for index in range(1, len(self.questions) + 1)
            ):
                raise ValueError("questionnaire ids must be ordered q1..qN")
        elif self.questions:
            raise ValueError("questions require questionnaire action")
        if self.selection_source is not None and self.action != "ask_user":
            raise ValueError("selection_source requires ask_user action")
        return self


class DecisionMaker(Protocol):
    def decide(self, context: MainAgentContext, tool_specs: tuple[dict[str, Any], ...]) -> AgentDecision: ...


def _reject_internal_identifiers(name: str, arguments: dict[str, Any]) -> None:
    forbidden = sorted(key for key in arguments if key == "user_id" or key.endswith("_id"))
    if forbidden:
        raise ValueError(
            f"{name} cannot accept internal identifiers: {', '.join(forbidden)}"
        )


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


def project_open_job_search_arguments(
    context: MainAgentContext, arguments: dict[str, Any]
) -> dict[str, Any]:
    _reject_internal_identifiers("open_job_search", arguments)
    model_arguments = OpenJobSearchToolArguments.model_validate(arguments)
    # Falls back to the person-level default only. A target role's city override
    # is deliberately not consulted here: task state has no notion of which role
    # the conversation is working in, so the only available rule would be "any
    # role that happens to have a city", which would silently search the wrong
    # place. Resolving it properly needs an active target role first.
    return {
        **model_arguments.model_copy(
            update={"city": model_arguments.city or context.profile.default_city}
        ).model_dump(),
        # The search is opened on behalf of this conversation; the capture
        # intent it creates has to remember which one, or the job the user
        # saves from it cannot find its way back.
        "user_id": context.profile.user_id,
        "conversation_id": context.conversation_id,
    }


def _report_is_about(
    held: ConversationResourceReference, candidate: SavedJobCandidateContextItem
) -> bool:
    """Whether a held report and a saved job name the same employer.

    Decided on identity when the reference carries it: the anchoring job or the
    company key the report is stored under, which the saved job's formal name
    folds to the same way. References written before identity was recorded
    have only a title to go on and are read the old way.
    """
    if held.job_posting_id is not None or held.company_key is not None:
        if held.job_posting_id == candidate.job_posting_id:
            return True
        return (
            held.company_key is not None
            and bool(candidate.company_name.strip())
            and company_key(candidate.company_name) == held.company_key
        )
    return bool(candidate.company_name) and candidate.company_name in (
        held.title or ""
    )


_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")


def _company_mention(company_name: str, user_message: str) -> str | None:
    """The longest leading part of ``company_name`` the message writes out.

    Users shorten employers from the front — "字节" for "字节跳动", "ByteDance"
    for "ByteDance Ltd" — so a leading part of at least two CJK characters, or
    three otherwise, is taken as naming the company. This only decides what
    the request is *about*, never which report answers it.
    """
    name = company_key(company_name) if company_name.strip() else ""
    message = " ".join(user_message.split()).casefold()
    for end in range(len(name), 1, -1):
        prefix = name[:end].rstrip()
        minimum = 2 if _CJK.search(prefix) else 3
        if len(prefix) >= minimum and prefix in message:
            return prefix
    return None


class _AskedCompanies(NamedTuple):
    jobs: tuple[tuple[int, SavedJobCandidateContextItem], ...]
    ambiguous_mention: str | None


def _companies_asked_for(
    context: MainAgentContext,
    held: ConversationResourceReference,
) -> _AskedCompanies:
    """The saved jobs this turn's request is about, as (selection_index, job).

    A search the model ran this turn already resolved the user's wording —
    short name, alias, or otherwise — to job entities, so when every job it
    returned belongs to one employer, that employer is what the user asked
    about. Absent such a search, a company counts when the message writes out
    its name or a leading part of it; a mixed result list says nothing about
    which company was meant. A leading part that heads more than one company
    — "中国" for both 中国移动 and 中国银行, or a saved job and the held report
    alike — names none of them, and is reported as ambiguous instead.
    """
    numbered = tuple(enumerate(context.task.saved_job_candidates, start=1))
    searched_this_turn = any(
        observation.tool_name == "find_saved_jobs"
        and observation.state == "saved_jobs_found"
        for observation in context.tool_observations
    )
    if searched_this_turn and len(
        {
            company_key(candidate.company_name)
            for _, candidate in numbered
            if candidate.company_name.strip()
        }
    ) == 1:
        return _AskedCompanies(numbered, None)
    held_company = held.company_key or ""
    asked: list[tuple[int, SavedJobCandidateContextItem]] = []
    companies_by_mention: dict[str, set[str]] = {}
    for index, candidate in numbered:
        mention = _company_mention(candidate.company_name, context.user_message)
        if mention is None:
            continue
        asked.append((index, candidate))
        named = companies_by_mention.setdefault(mention, set())
        named.add(company_key(candidate.company_name))
        if held_company.startswith(mention):
            named.add(held_company)
    ambiguous = next(
        (mention for mention, named in companies_by_mention.items() if len(named) > 1),
        None,
    )
    return _AskedCompanies(tuple(asked), ambiguous)


def _reject_borrowed_report_reference(
    context: MainAgentContext, *, reference: str, report_id: str
) -> None:
    """A report handle is bound to a company; the request must be about it.

    ``resolve_reference`` proves the handle was issued, not that it is the one
    the user meant. When the request is about a saved-job company and the held
    report is about a different one, the read would answer about the wrong
    company while looking grounded, so it is refused in favour of the company's
    own selection index. A request that also covers the report's company, or
    is about no saved-job company at all, is left to the model; one whose short
    name fits several companies is refused until the user says which.
    """
    held = next(
        item for item in context.referenced_resources()
        if item.resource_id == report_id
    )
    title = (held.title or "").strip()
    if title and title in context.user_message:
        return
    asked, ambiguous = _companies_asked_for(context, held)
    if not asked:
        return
    selectors = "、".join(
        f"{index}（{candidate.company_name}）" for index, candidate in asked
    )
    if ambiguous is not None:
        raise ValueError(
            f"'{ambiguous}' names more than one saved company, so resource "
            f"reference '{reference}' cannot be read as the one the user meant; "
            f"ask which company is meant (saved: selection_index {selectors}) "
            "instead of guessing"
        )
    if any(_report_is_about(held, candidate) for _, candidate in asked):
        return
    about = f"titled '{title}'" if title else "about another company"
    raise ValueError(
        f"resource reference '{reference}' is {about}, not the company "
        f"the user asked about; use selection_index {selectors} or say that "
        "report is not reachable. A company named by a short name or alias "
        "must be resolved with find_saved_jobs first, never by guessing which "
        "title it means"
    )


def project_job_research_arguments(
    context: MainAgentContext,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    _reject_internal_identifiers(name, arguments)
    if name == "research_job":
        model_arguments = ResearchJobToolArguments.model_validate(arguments)
        payload = model_arguments.model_dump()
        selection_index = payload.pop("selection_index", None)
        job_posting_id = context.task.active_job_posting_id
        if selection_index is not None:
            if not 1 <= selection_index <= len(context.task.saved_job_candidates):
                raise ValueError("saved-job selection index is out of range")
            job_posting_id = context.task.saved_job_candidates[
                selection_index - 1
            ].job_posting_id
        if job_posting_id is None:
            raise ValueError("research_job requires a selected or active saved job")
        payload["job_posting_id"] = job_posting_id
        payload["jd_snapshot_id"] = _pinned_jd_snapshot_id(
            context, job_posting_id=job_posting_id, explicit_selection=selection_index is not None
        )
    elif name == "retry_job_research":
        RetryJobResearchToolArguments.model_validate(arguments)
        if context.task.active_job_research_run_id is None:
            raise ValueError("retry_job_research requires an active failed run")
        payload = {"run_id": context.task.active_job_research_run_id}
    elif name == "get_job_research":
        model_arguments = GetJobResearchToolArguments.model_validate(arguments)
        selection_index = model_arguments.selection_index
        if model_arguments.reference is not None:
            report_id = context.resolve_reference(
                reference=model_arguments.reference,
                kind="job_research_report",
            )
            _reject_borrowed_report_reference(
                context, reference=model_arguments.reference, report_id=report_id
            )
            payload = {"report_id": report_id}
        elif selection_index is not None:
            if not 1 <= selection_index <= len(context.task.saved_job_candidates):
                raise ValueError("saved-job selection index is out of range")
            payload = {
                "job_posting_id": context.task.saved_job_candidates[
                    selection_index - 1
                ].job_posting_id
            }
        elif context.task.active_job_research_report_id is not None:
            # No selector: the active report stands in for "the report". The
            # request travels with it so the read can be refused when the
            # user was asking about a different company.
            payload = {
                "report_id": context.task.active_job_research_report_id,
                IMPLICIT_REQUEST_KEY: context.user_message,
            }
        elif context.task.active_job_posting_id is not None:
            payload = {
                "job_posting_id": context.task.active_job_posting_id,
                IMPLICIT_REQUEST_KEY: context.user_message,
            }
        else:
            raise ValueError("get_job_research requires an active research report or job")
    else:
        raise ValueError(f"Unknown job research tool: {name}")
    return {"user_id": context.profile.user_id, **payload}


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
    # The stuck run is found by user, not named by the model: there is only one
    # unfinished run per user, and naming it would mean exposing its id.
    return {"user_id": context.profile.user_id}


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


def project_calendar_arguments(
    context: MainAgentContext, name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    _reject_internal_identifiers(name, arguments)
    if name == "list_calendar_accounts":
        model_arguments = ListCalendarAccountsToolArguments.model_validate(arguments)
    elif name == "list_calendar_links":
        model_arguments = ListCalendarLinksToolArguments.model_validate(arguments)
    elif name == "prepare_interview_calendar_sync":
        model_arguments = PrepareInterviewCalendarSyncToolArguments.model_validate(arguments)
    elif name == "get_calendar_proposal":
        model_arguments = GetCalendarProposalToolArguments.model_validate(arguments)
    elif name == "execute_calendar_proposal":
        model_arguments = ExecuteCalendarProposalToolArguments.model_validate(arguments)
    else:
        raise ValueError(f"Unknown calendar tool: {name}")
    payload = model_arguments.model_dump()
    if name == "prepare_interview_calendar_sync":
        interview_round_id = payload.get("interview_round_id")
        interview_index = payload.pop("interview_selection_index", None)
        if interview_round_id is None and interview_index is not None:
            if not 1 <= interview_index <= len(context.task.interview_candidates):
                raise ValueError("interview selection index is out of range")
            interview_round_id = context.task.interview_candidates[
                interview_index - 1
            ].interview_round_id
        interview_round_id = interview_round_id or context.task.active_interview_round_id
        if interview_round_id is None:
            raise ValueError("prepare_interview_calendar_sync requires an active interview")
        payload["interview_round_id"] = interview_round_id
        account_id = payload.get("calendar_account_id")
        account_index = payload.pop("calendar_account_selection_index", None)
        if account_id is None and account_index is not None:
            if not 1 <= account_index <= len(context.task.calendar_account_candidates):
                raise ValueError("calendar account selection index is out of range")
            account_id = context.task.calendar_account_candidates[
                account_index - 1
            ].calendar_account_id
        payload["calendar_account_id"] = account_id
    if name in {"get_calendar_proposal", "execute_calendar_proposal"}:
        proposal_id = payload.get("proposal_id") or context.task.active_calendar_proposal_id
        if proposal_id is None:
            raise ValueError(f"{name} requires an active calendar proposal")
        payload["proposal_id"] = proposal_id
    return {"user_id": context.profile.user_id, **payload}
