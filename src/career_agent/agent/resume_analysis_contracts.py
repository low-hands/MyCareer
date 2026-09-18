from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, Mapping, Protocol

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

if TYPE_CHECKING:
    from career_agent.storage.resumes import StoredResumeDocument


# Source evidence is compared verbatim, unlike presentation fields. Do not
# normalize a model-authored near-match into a valid quote or issued locator.
SourceText = Annotated[
    str, StringConstraints(strip_whitespace=False, min_length=1, pattern=r"\S")
]
NumberedSourceLocator = Annotated[int, Field(strict=True, ge=1)]


class ResumeAnalysisContract(BaseModel):
    """Base contract for untrusted, model-produced resume analysis."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class ExtractedCareerEvidence(ResumeAnalysisContract):
    """A resume claim together with the text needed to verify it."""

    claim: str = Field(min_length=1)
    source_locator: SourceText
    source_quote: SourceText


class ExtractedCareerRecord(ResumeAnalysisContract):
    """An unconfirmed career record extracted from a resume."""

    record_type: Literal[
        "education",
        "work",
        "internship",
        "project",
        "certification",
    ]
    organization: str | None = Field(default=None, min_length=1)
    title: str = Field(min_length=1)
    start_year: int | None = Field(default=None, ge=1900, le=2200)
    start_month: int | None = Field(default=None, ge=1, le=12)
    end_year: int | None = Field(default=None, ge=1900, le=2200)
    end_month: int | None = Field(default=None, ge=1, le=12)
    is_current: bool = False
    source_locator: SourceText
    source_quote: SourceText
    evidence: tuple[ExtractedCareerEvidence, ...] = ()

    @model_validator(mode="after")
    def validate_period(self) -> ExtractedCareerRecord:
        if self.start_month is not None and self.start_year is None:
            raise ValueError("start_month requires start_year")
        if self.end_month is not None and self.end_year is None:
            raise ValueError("end_month requires end_year")
        if self.is_current and (
            self.end_year is not None or self.end_month is not None
        ):
            raise ValueError("current records cannot have an end date")
        if self.start_year is not None and self.end_year is not None:
            start = (self.start_year, self.start_month or 1)
            end = (self.end_year, self.end_month or 12)
            if end < start:
                raise ValueError("end date cannot be before start date")
        return self


class ResumeAnalysisResult(ResumeAnalysisContract):
    """Structured output returned by a resume-analysis worker."""

    records: tuple[ExtractedCareerRecord, ...] = ()
    clarification_questions: tuple[Annotated[str, Field(min_length=1)], ...] = ()
    warnings: tuple[Annotated[str, Field(min_length=1)], ...] = ()

    def validate_source_quotes(self, source: Mapping[str, str]) -> None:
        """Validate new worker output against locally issued locators.

        Persisted drafts retain their original contract/locators. This check is
        performed against the exact source at analysis time, not by guessing a
        page from a model-authored description or normalizing a near-match.
        """
        for record in self.records:
            for item in (record, *record.evidence):
                paragraph = source.get(item.source_locator)
                if paragraph is None or item.source_quote not in paragraph:
                    # Do not put model output, document text, or an untrusted
                    # locator into an exception that the Runtime may record.
                    raise ValueError("source_quote_or_locator_mismatch")


class NumberedCareerEvidence(ResumeAnalysisContract):
    """Wire-only evidence using one request's compact paragraph number."""

    claim: str = Field(min_length=1)
    source_locator: NumberedSourceLocator = Field(
        description="An issued paragraph_number integer, never text."
    )


class NumberedCareerRecord(ResumeAnalysisContract):
    """Wire-only record; converted before it can cross the worker boundary."""

    record_type: Literal[
        "education",
        "work",
        "internship",
        "project",
        "certification",
    ]
    organization: str | None = Field(default=None, min_length=1)
    title: str = Field(min_length=1)
    start_year: int | None = Field(default=None, ge=1900, le=2200)
    start_month: int | None = Field(default=None, ge=1, le=12)
    end_year: int | None = Field(default=None, ge=1900, le=2200)
    end_month: int | None = Field(default=None, ge=1, le=12)
    is_current: bool = False
    source_locator: NumberedSourceLocator = Field(
        description="An issued paragraph_number integer, never text."
    )
    evidence: tuple[NumberedCareerEvidence, ...] = ()

    @model_validator(mode="after")
    def validate_period(self) -> NumberedCareerRecord:
        if self.start_month is not None and self.start_year is None:
            raise ValueError("start_month requires start_year")
        if self.end_month is not None and self.end_year is None:
            raise ValueError("end_month requires end_year")
        if self.is_current and (
            self.end_year is not None or self.end_month is not None
        ):
            raise ValueError("current records cannot have an end date")
        if self.start_year is not None and self.end_year is not None:
            start = (self.start_year, self.start_month or 1)
            end = (self.end_year, self.end_month or 12)
            if end < start:
                raise ValueError("end date cannot be before start date")
        return self


class NumberedResumeAnalysisResult(ResumeAnalysisContract):
    """Request-local model output that is never persisted."""

    records: tuple[NumberedCareerRecord, ...] = ()
    clarification_questions: tuple[Annotated[str, Field(min_length=1)], ...] = ()
    warnings: tuple[Annotated[str, Field(min_length=1)], ...] = ()

    def to_persisted(
        self,
        locator_by_number: Mapping[int, str],
        paragraph_by_number: Mapping[int, str],
    ) -> ResumeAnalysisResult:
        """Map issued integers to readable locators and exact local paragraphs."""

        try:
            records = tuple(
                ExtractedCareerRecord(
                    **record.model_dump(
                        exclude={"source_locator", "evidence"},
                        mode="python",
                    ),
                    source_locator=locator_by_number[record.source_locator],
                    source_quote=paragraph_by_number[record.source_locator],
                    evidence=tuple(
                        ExtractedCareerEvidence(
                            **item.model_dump(
                                exclude={"source_locator"},
                                mode="python",
                            ),
                            source_locator=locator_by_number[item.source_locator],
                            source_quote=paragraph_by_number[
                                item.source_locator
                            ],
                        )
                        for item in record.evidence
                    ),
                )
                for record in self.records
            )
        except KeyError:
            # The schema enum is only an upstream hint. This local lookup is the
            # authority when a provider ignores or weakly enforces the schema.
            raise ValueError("source_quote_or_locator_mismatch") from None
        return ResumeAnalysisResult(
            records=records,
            clarification_questions=self.clarification_questions,
            warnings=self.warnings,
        )


class ResumeAnalysisWorker(Protocol):
    """Provider-independent boundary for reading and analysing a resume file."""

    def analyze(self, document: StoredResumeDocument) -> ResumeAnalysisResult: ...
