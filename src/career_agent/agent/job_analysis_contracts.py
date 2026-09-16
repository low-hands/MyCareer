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

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class JobAnalysisContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


Seniority = Literal["fresh_graduate", "junior", "mid", "senior", "lead"]
RequirementTier = Literal["S", "A", "B", "C"]
RequirementKind = Literal["fact", "inference"]

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

    text: str = Field(min_length=1, max_length=1000)
    tier: RequirementTier
    kind: RequirementKind
    jd_quote: str = Field(min_length=1, max_length=500)


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


class JobAnalysisWorker(Protocol):
    def analyze(self, *, jd_text: str) -> JobAnalysisResult: ...
