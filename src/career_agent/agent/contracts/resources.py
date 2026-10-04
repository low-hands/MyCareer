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

