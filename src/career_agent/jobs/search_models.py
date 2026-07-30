"""Search intent and immutable result snapshot for job discovery.

SearchCriteria expresses what the user asked for.  SearchRun is the
frozen record of what was actually shown to the user — created once,
never updated, referencing persisted JobPosting rows by ID.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from career_agent.jobs.models import JobPosting


class SearchCriteria(BaseModel):
    """User's original search requirements.

    Populated directly by LLM function-calling — no separate NL-parse step.
    Lists (cities, employment_types, …) allow multi-value input like
    "北京或上海" / "全职或实习".  Salary is explicit in 千元/月 units.
    """

    query: str = ""
    cities: list[str] = Field(default_factory=list)
    employment_types: list[str] = Field(default_factory=list)
    salary_min_k: int | None = Field(default=None, ge=0)
    salary_max_k: int | None = Field(default=None, ge=0)
    experience: str = ""
    education: str = ""
    company_industries: list[str] = Field(default_factory=list)
    company_sizes: list[str] = Field(default_factory=list)
    company_stages: list[str] = Field(default_factory=list)

    @field_validator("query", "experience", "education")
    @classmethod
    def _strip_scalar_text(cls, value: str) -> str:
        return value.strip()

    @field_validator(
        "cities",
        "employment_types",
        "company_industries",
        "company_sizes",
        "company_stages",
    )
    @classmethod
    def _normalize_list_values(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            item = value.strip()
            identity = item.casefold()
            if item and identity not in seen:
                normalized.append(item)
                seen.add(identity)
        return normalized

    @model_validator(mode="after")
    def _salary_range_must_be_ordered(self) -> SearchCriteria:
        if self.salary_min_k is not None and self.salary_max_k is not None and self.salary_min_k > self.salary_max_k:
            raise ValueError("salary_min_k must be <= salary_max_k")
        return self


class SearchRun(BaseModel, frozen=True):
    """Immutable snapshot of the job list presented to the user.

    Once created, a SearchRun can never be reordered or modified —
    a new search produces a new SearchRun.  ``job_ids`` uses ``tuple``
    for Python-level immutability; JSON serialisation still emits an array.
    """

    run_id: str
    tenant_id: str
    thread_id: str
    user_query: str
    criteria: SearchCriteria
    job_ids: tuple[str, ...]
    candidate_count: int = Field(default=0, ge=0)
    created_at: datetime

    @field_validator("run_id", "tenant_id", "thread_id", "user_query")
    @classmethod
    def _required_text_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized

    @field_validator("job_ids")
    @classmethod
    def _job_ids_must_be_non_empty_and_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) == 0:
            raise ValueError("job_ids must contain at least one entry")
        if len(value) != len(set(value)):
            raise ValueError("job_ids must not contain duplicates")
        return value

    @model_validator(mode="after")
    def _candidate_count_covers_shown_results(self) -> SearchRun:
        if self.candidate_count < len(self.job_ids):
            raise ValueError(f"candidate_count ({self.candidate_count}) must be >= len(job_ids) ({len(self.job_ids)})")
        return self


class SearchResultSelector(BaseModel):
    """Structured reference extracted from a follow-up user message."""

    position: int | None = Field(default=None, ge=1)
    company_name: str = ""
    title: str = ""
    location: str = ""
    salary: str = ""

    @field_validator("company_name", "title", "location", "salary")
    @classmethod
    def _strip_selector_text(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def _at_least_one_reference_is_required(self) -> SearchResultSelector:
        if self.position is None and not any((self.company_name, self.title, self.location, self.salary)):
            raise ValueError("at least one search-result reference is required")
        return self


class SearchResultCandidate(BaseModel, frozen=True):
    """Display-safe ambiguity candidate without an internal job ID."""

    position: int = Field(ge=1)
    title: str
    company_name: str
    location: str = ""
    salary: str = ""


class ResolvedSearchResult(BaseModel, frozen=True):
    """A user reference resolved to one persisted posting."""

    status: Literal["resolved"] = "resolved"
    run_id: str
    position: int = Field(ge=1)
    posting: JobPosting


class AmbiguousSearchResult(BaseModel, frozen=True):
    """A user reference matched several display-safe candidates."""

    status: Literal["ambiguous"] = "ambiguous"
    run_id: str
    candidates: tuple[SearchResultCandidate, ...] = Field(min_length=2)


class SearchResultNotFound(BaseModel, frozen=True):
    """No visible posting could be resolved from the reference."""

    status: Literal["not_found"] = "not_found"
    run_id: str | None = None
    message: str


SearchResolution = Annotated[
    ResolvedSearchResult | AmbiguousSearchResult | SearchResultNotFound,
    Field(discriminator="status"),
]
