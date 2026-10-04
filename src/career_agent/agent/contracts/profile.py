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

    External writes and destructive local writes sit on the same footing. Once
    an event is on the user's calendar, or a memory/constraint has been retired,
    the model's reading of "yes, go ahead" is not enough: the owner presses the
    button on the exact sealed arguments. Owner rules may only add restrictions
    on top of this floor. The catalogue is the closed list of those operations.
    """

    if approval_policy(capability) == "always":
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

