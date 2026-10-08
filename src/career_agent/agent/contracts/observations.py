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

from career_agent.agent.contracts.resources import *

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

    Tools and workers only return this result; they never write to the
    conversation or the UI stream. The Main Agent runtime owns delivery and
    decides what the user sees.
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
# The evidence envelope adds 864 characters to the eleven-result worst shape.
# Keep the bound explicit and measured, without expanding receipt/body limits.
MAX_DECISION_OBSERVATION_CHARS = 19_300


class DecisionObservation(ContractModel):
    """Closed, bounded observation visible to the Main Agent decision model."""

    tool_name: str = Field(pattern=r"^[a-z0-9_]+$", max_length=80)
    state: str = Field(pattern=r"^[a-z0-9_]+$", max_length=80)
    disposition: Literal["completed", "interaction_required", "failed"] | None = None
    execution_outcome: Literal["committed", "not_committed", "unknown"] | None = None
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


class EvidenceEnvelope(ContractModel):
    """Runtime-authored limits of the evidence currently visible to the model.

    A body is an excerpt, not proof of a complete report. A receipt is not its
    contents. Readback requires a resolvable handle, never a success message.
    """

    body_status: Literal["excerpt", "receipt_only"]
    readback_status: Literal["available", "unavailable"]


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
        # Selection needs these receipts, but the model-facing observation
        # already explains the outcome through state and message.
        line.pop("disposition", None)
        line.pop("execution_outcome", None)
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
        line["evidence"] = EvidenceEnvelope(
            body_status="excerpt" if observation.body else "receipt_only",
            readback_status=(
                "available" if "reference" in line or line.get("facts", {}).get("resource_refs")
                else "unavailable"
            ),
        ).model_dump(mode="json")
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

