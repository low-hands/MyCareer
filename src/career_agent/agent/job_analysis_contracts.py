"""Contracts for the JD-only analysis of a saved job.

The analyser reads one JD snapshot and nothing else: no resume, no stored
preference, no web research. Every conclusion therefore has to be traceable
to the posting itself, which is why each tiered requirement carries the JD
sentence it rests on and says whether it is stated there or inferred from it.
Seniority, and what HR, the hiring manager and interviewers will look for,
are read off the JD's own signals; the contract does not presuppose a campus
or an experienced-hire process.
"""

from __future__ import annotations

import re
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


class JobAnalysisContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


Seniority = Literal["fresh_graduate", "junior", "mid", "senior", "lead"]
RequirementTier = Literal["S", "A", "B", "C"]
RequirementKind = Literal["fact", "inference"]
TierConfidence = Literal["high", "medium", "low"]
ClassificationStatus = Literal["model_assessed", "user_confirmed", "user_corrected"]

SENIORITY_LABELS: dict[Seniority, str] = {
    "fresh_graduate": "应届",
    "junior": "初级",
    "mid": "中级",
    "senior": "高级",
    "lead": "管理",
}
TIER_ORDER: tuple[RequirementTier, ...] = ("S", "A", "B", "C")
TIER_LABELS: dict[RequirementTier, str] = {
    "S": "S · 硬门槛",
    "A": "A · 核心要求",
    "B": "B · 重要加分",
    "C": "C · 锦上添花",
}


class TieredRequirement(JobAnalysisContract):
    """One requirement graded by how much the posting hinges on it.

    S is a gate the JD states outright; A is what the role is really hiring
    for; B distinguishes candidates without disqualifying anyone; C is
    mentioned but incidental. ``kind`` says whether the requirement is
    written in the JD (``fact``) or read between its lines (``inference``);
    either way ``jd_quote`` is the sentence it rests on.
    """

    requirement_id: str | None = Field(
        default=None,
        pattern=r"^job_requirement_[a-f0-9]{20}$",
    )
    text: str = Field(min_length=1, max_length=1000)
    tier: RequirementTier
    kind: RequirementKind
    jd_quote: str = Field(min_length=1, max_length=500)
    # ``tier`` is the effective tier consumed downstream. These fields retain
    # the model's initial judgment and any human correction without changing
    # the stable requirement identity.
    model_tier: RequirementTier | None = None
    tier_source: Literal["model", "rule", "user_corrected"] = "model"
    tier_correction_reason: str | None = Field(default=None, min_length=1, max_length=1000)
    tier_rationale: str = Field(default="", max_length=1500)
    model_tier_rationale: str | None = Field(default=None, min_length=1, max_length=1500)
    tier_evidence: str = Field(default="", max_length=500)
    tier_confidence: TierConfidence = "low"
    classification_status: ClassificationStatus = "model_assessed"

    @property
    def effective_tier(self) -> RequirementTier:
        return self.tier

    @model_validator(mode="after")
    def validate_tier_trace(self) -> TieredRequirement:
        if self.tier_source == "model" and self.model_tier is not None and self.model_tier != self.tier:
            raise ValueError("model tier must equal the effective tier")
        if self.classification_status == "user_corrected" and self.tier_source != "user_corrected":
            raise ValueError("user-corrected status requires a user-corrected tier source")
        if self.tier_source == "user_corrected" and self.classification_status != "user_corrected":
            raise ValueError("user-corrected tier source requires user-corrected status")
        if self.tier_source == "user_corrected":
            if self.tier_correction_reason is None:
                raise ValueError("user correction requires a reason")
        if self.tier_source != "user_corrected" and self.tier_correction_reason is not None:
            raise ValueError("tier correction reason requires a user correction")
        return self


_EXPLICIT_HARD_GATE = re.compile(
    r"(?:required|must have|must|minimum|at least|mandatory|hard requirement|"
    r"\b\d+\s*\+?\s*years?\b|\b\d+\s*years?\s+(?:of\s+)?experience\b|"
    r"bachelor(?:'s|s)?(?:\s+degree)?|master(?:'s|s)?(?:\s+degree)?|"
    r"ph\.?d\.?|degree required|certification required|license required|"
    r"\d+\s*年以上|\d+\s*年经验|本科及以上|硕士及以上|博士及以上|学历要求|"
    r"资格证书|认证要求|持证|硬性要求|必须|至少|硬门槛|必备)",
    re.IGNORECASE,
)


def explicit_hard_gate_language(jd_quote: str) -> bool:
    """Return whether the quoted JD language explicitly states a hard gate."""
    return bool(_EXPLICIT_HARD_GATE.search(jd_quote))


def is_confirmed_hard_gate(requirement: TieredRequirement, *, status: str | None = None) -> bool:
    return (
        getattr(requirement, "tier", None) == "S"
        and getattr(requirement, "kind", None) == "fact"
        and (status is None or status == "missing")
        and getattr(requirement, "tier_confidence", None) == "high"
        and bool((getattr(requirement, "tier_rationale", None) or "").strip())
        and explicit_hard_gate_language(
            getattr(requirement, "tier_evidence", None)
            or getattr(requirement, "jd_quote", "")
        )
    )


class QuotedFinding(JobAnalysisContract):
    """A conclusion that is not stated in the JD, pinned to the text behind it."""

    text: str = Field(min_length=1, max_length=1000)
    jd_quote: str = Field(min_length=1, max_length=500)


class JobAnalysisResult(JobAnalysisContract):
    core_objective: str = Field(min_length=1, max_length=2000)
    seniority: Seniority
    requirements: tuple[TieredRequirement, ...] = Field(default=(), max_length=40)
    core_competencies: tuple[str, ...] = Field(default=(), max_length=15)
    implicit_requirements: tuple[QuotedFinding, ...] = Field(default=(), max_length=15)
    ats_keywords: tuple[str, ...] = Field(default=(), max_length=40)
    hr_focus: tuple[str, ...] = Field(default=(), max_length=10)
    hiring_manager_focus: tuple[str, ...] = Field(default=(), max_length=10)
    likely_interview_topics: tuple[str, ...] = Field(default=(), max_length=15)
    red_flags: tuple[QuotedFinding, ...] = Field(default=(), max_length=10)
    information_gaps: tuple[str, ...] = Field(default=(), max_length=10)
    summary: str = Field(min_length=1, max_length=3000)

    def requirements_in_tier(
        self, tier: RequirementTier
    ) -> tuple[TieredRequirement, ...]:
        return tuple(item for item in self.requirements if item.tier == tier)


class GeneratedTieredRequirement(TieredRequirement):
    """Accept a model's provisional ID; the service replaces it before storage."""

    requirement_id: str | None = None


class JobAnalysisGenerationResult(JobAnalysisResult):
    requirements: tuple[GeneratedTieredRequirement, ...] = Field(default=(), max_length=40)

    def without_model_ids(self) -> JobAnalysisResult:
        payload = self.model_dump(mode="python")
        for requirement in payload["requirements"]:
            requirement["requirement_id"] = None
        return JobAnalysisResult.model_validate(payload)


class JobAnalysisWorker(Protocol):
    def analyze(self, *, jd_text: str) -> JobAnalysisResult: ...
