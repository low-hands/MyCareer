import pytest
from pydantic import ValidationError

from career_agent.agent.resume_analysis_contracts import (
    ExtractedCareerEvidence,
    ExtractedCareerRecord,
    ResumeAnalysisResult,
)


def test_resume_analysis_result_accepts_grounded_nested_records() -> None:
    result = ResumeAnalysisResult(
        records=(
            ExtractedCareerRecord(
                record_type="work",
                organization="Example Inc.",
                title="Product Manager",
                start_year=2022,
                start_month=3,
                is_current=True,
                source_locator="page 1, Experience",
                source_quote="Product Manager, Example Inc., March 2022–Present",
                evidence=(
                    ExtractedCareerEvidence(
                        claim="Led the knowledge-base product roadmap",
                        source_locator="page 1, Experience, bullet 1",
                        source_quote="Led the knowledge-base product roadmap",
                    ),
                ),
            ),
        ),
    )

    assert result.records[0].evidence[0].claim.startswith("Led")


@pytest.mark.parametrize(
    "trusted_field,value",
    [
        ("id", "record-1"),
        ("user_id", "user-1"),
        ("verification_status", "confirmed"),
        ("created_at", "2026-01-01T00:00:00Z"),
    ],
)
def test_extracted_record_rejects_system_owned_fields(
    trusted_field: str,
    value: str,
) -> None:
    data = {
        "record_type": "project",
        "title": "Search assistant",
        "source_locator": "page 1, Projects",
        "source_quote": "Search assistant",
        trusted_field: value,
    }

    with pytest.raises(ValidationError):
        ExtractedCareerRecord.model_validate(data)


def test_extracted_evidence_rejects_system_owned_fields() -> None:
    with pytest.raises(ValidationError):
        ExtractedCareerEvidence(
            claim="Improved retrieval quality",
            source_locator="page 1, bullet 2",
            source_quote="Improved retrieval quality",
            origin="resume_extraction",  # type: ignore[call-arg]
        )


def test_extracted_text_is_stripped_and_must_not_be_blank() -> None:
    evidence = ExtractedCareerEvidence(
        claim="  Improved retrieval quality  ",
        source_locator="  page 1  ",
        source_quote="  Improved retrieval quality  ",
    )
    assert evidence.claim == "Improved retrieval quality"

    with pytest.raises(ValidationError):
        ExtractedCareerEvidence(
            claim="   ",
            source_locator="page 1",
            source_quote="text",
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"start_month": 3},
        {"end_month": 3},
        {"is_current": True, "end_year": 2025},
        {"start_year": 2025, "start_month": 6, "end_year": 2025, "end_month": 5},
    ],
)
def test_extracted_record_rejects_invalid_periods(overrides: dict[str, object]) -> None:
    data: dict[str, object] = {
        "record_type": "work",
        "title": "Engineer",
        "source_locator": "page 1",
        "source_quote": "Engineer",
    }
    data.update(overrides)

    with pytest.raises(ValidationError):
        ExtractedCareerRecord.model_validate(data)


def test_analysis_result_can_request_clarification_without_records() -> None:
    result = ResumeAnalysisResult(
        clarification_questions=("这段经历的结束月份是什么？",),
        warnings=("PDF 第二页无法读取。",),
    )

    assert result.records == ()


def test_analysis_contracts_are_frozen() -> None:
    result = ResumeAnalysisResult()

    with pytest.raises(ValidationError):
        result.records = ()
