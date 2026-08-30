from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


DimensionId = Literal[
    "resume_fit",
    "requirement_coverage",
    "salary_disclosure",
    "location",
    "availability",
]

# Every dimension carries "unknown" as a real value rather than a missing one.
# A JD that does not state a city and a JD that states a city we cannot match
# are different facts, and neither may be quietly rounded to a middle grade.
DIMENSION_VALUES: dict[DimensionId, tuple[str, ...]] = {
    "resume_fit": (
        "strong",
        "moderate",
        "weak",
        "insufficient_evidence",
        "unknown",
    ),
    "requirement_coverage": ("full", "most", "partial", "little", "unknown"),
    "salary_disclosure": ("disclosed", "undisclosed"),
    "location": ("matches_preference", "differs", "unknown"),
    # Mirrors storage.jobs.AvailabilityStatus so the two never drift apart.
    "availability": ("active", "closed", "unknown"),
}

DIMENSION_ORDER: tuple[DimensionId, ...] = (
    "resume_fit",
    "requirement_coverage",
    "salary_disclosure",
    "location",
    "availability",
)


class JobComparisonContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class ComparisonCell(JobComparisonContract):
    """One job's standing on one dimension, with the reason it says that.

    ``basis`` is not decoration. A grade with no visible derivation invites the
    reader to trust it more than the underlying data supports, which is the
    failure mode this whole comparison is built to avoid.
    """

    dimension: DimensionId
    value: str = Field(min_length=1, max_length=40)
    basis: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def value_must_belong_to_its_dimension(self) -> ComparisonCell:
        allowed = DIMENSION_VALUES[self.dimension]
        if self.value not in allowed:
            raise ValueError(
                f"{self.dimension} cannot take the value {self.value!r}"
            )
        return self


class ComparisonRow(JobComparisonContract):
    job_posting_id: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=500)
    company_name: str = Field(min_length=1, max_length=500)
    cells: tuple[ComparisonCell, ...] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def cover_every_dimension_once(self) -> ComparisonRow:
        seen = tuple(cell.dimension for cell in self.cells)
        if seen != DIMENSION_ORDER:
            raise ValueError(
                "a comparison row must carry every dimension exactly once, in order"
            )
        return self


class JobComparison(JobComparisonContract):
    """A matrix, deliberately not a ranking.

    There is no total, no weight, and no position field. Rows keep the order the
    caller asked for, so nothing in this structure can be mistaken for a verdict
    about which job is better.
    """

    rows: tuple[ComparisonRow, ...] = Field(min_length=1, max_length=10)
    # Dimensions on which every job came back unknown. An all-blank column reads
    # like "these jobs do not differ here", when it actually means the data to
    # tell them apart was never captured.
    uninformative_dimensions: tuple[DimensionId, ...] = Field(default=(), max_length=8)
    # Jobs with no match on record, so the reader knows which blanks are fixable
    # by running match_resume_to_job rather than being inherent to the JD.
    jobs_without_match: tuple[str, ...] = Field(default=(), max_length=10)
    notes: tuple[str, ...] = Field(default=(), max_length=10)
