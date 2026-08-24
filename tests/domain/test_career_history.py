from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from career_agent.domain.career_history import CareerEvidence, CareerRecord


NOW = datetime(2026, 8, 24, tzinfo=timezone.utc)


def record(**overrides: object) -> CareerRecord:
    values = {
        "id": "record-1",
        "user_id": "user-1",
        "record_type": "work",
        "organization": "Acme",
        "title": "AI Engineer",
        "start_year": 2023,
        "start_month": 7,
        "is_current": True,
        "created_at": NOW,
        "updated_at": NOW,
    }
    values.update(overrides)
    return CareerRecord.model_validate(values)


def evidence(**overrides: object) -> CareerEvidence:
    values = {
        "id": "evidence-1",
        "user_id": "user-1",
        "career_record_id": "record-1",
        "claim": "Improved answer accuracy from 62% to 81%.",
        "origin": "user_input",
        "created_at": NOW,
        "updated_at": NOW,
    }
    values.update(overrides)
    return CareerEvidence.model_validate(values)


def test_career_record_accepts_month_precision() -> None:
    value = record()

    assert value.start_year == 2023
    assert value.start_month == 7
    assert value.end_year is None
    assert value.is_current is True


@pytest.mark.parametrize("field", ["start_month", "end_month"])
@pytest.mark.parametrize("month", [0, 13])
def test_career_record_rejects_invalid_month(field: str, month: int) -> None:
    with pytest.raises(ValidationError):
        record(**{field: month})


@pytest.mark.parametrize(
    ("month_field", "year_field"),
    [("start_month", "start_year"), ("end_month", "end_year")],
)
def test_career_record_month_requires_year(month_field: str, year_field: str) -> None:
    with pytest.raises(ValidationError, match=f"{month_field} requires {year_field}"):
        record(is_current=False, **{month_field: 7, year_field: None})


def test_current_career_record_rejects_end_date() -> None:
    with pytest.raises(ValidationError, match="current record cannot have an end date"):
        record(end_year=2024, end_month=1)


@pytest.mark.parametrize(
    "overrides",
    [
        {"start_year": 2024, "end_year": 2023},
        {"start_year": 2024, "start_month": 8, "end_year": 2024, "end_month": 7},
    ],
)
def test_career_record_rejects_end_before_start(overrides: dict[str, int]) -> None:
    with pytest.raises(ValidationError, match="end date cannot be earlier than start date"):
        record(is_current=False, **overrides)


def test_resume_extraction_requires_source_version_and_locator() -> None:
    with pytest.raises(ValidationError, match="resume extraction requires"):
        evidence(origin="resume_extraction")

    value = evidence(
        origin="resume_extraction",
        source_resume_version_id="resume-version-1",
        source_locator="page=1;start=10;end=42",
    )

    assert value.source_resume_version_id == "resume-version-1"


def test_source_locator_requires_resume_version() -> None:
    with pytest.raises(ValidationError, match="source_locator requires"):
        evidence(source_locator="page=1")


def test_contract_strips_strings_and_rejects_blank_claim() -> None:
    value = evidence(claim="  Built a RAG evaluation pipeline.  ")

    assert value.claim == "Built a RAG evaluation pipeline."
    with pytest.raises(ValidationError):
        evidence(claim="   ")


def test_contracts_are_frozen() -> None:
    value = record()

    with pytest.raises(ValidationError):
        value.title = "Product Manager"
