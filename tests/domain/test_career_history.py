from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from career_agent.domain.career_history import (
    CareerEvidence,
    CareerEvidenceEvent,
    CareerRecord,
)


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


def evidence_event(**overrides: object) -> CareerEvidenceEvent:
    values = {
        "id": "event-1",
        "user_id": "user-1",
        "career_evidence_id": "evidence-1",
        "event_type": "confirmed",
        "previous_status": "pending",
        "new_status": "confirmed",
        "actor_type": "user",
        "occurred_at": NOW,
    }
    values.update(overrides)
    return CareerEvidenceEvent.model_validate(values)


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


def test_resume_extraction_requires_source_version_locator_and_quote() -> None:
    with pytest.raises(ValidationError, match="resume extraction requires"):
        evidence(origin="resume_extraction")

    value = evidence(
        origin="resume_extraction",
        source_resume_version_id="resume-version-1",
        source_locator="page=1;start=10;end=42",
        source_quote="Improved answer accuracy from 62% to 81%.",
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


def test_confirmed_evidence_requires_a_complete_version_binding() -> None:
    binding = {
        "scope_key": "career_evidence/evidence-1/claim",
        "update_id": "career_evidence_update_" + "a" * 32,
        "content_digest": "sha256:" + "b" * 64,
        "revision": 1,
        "valid_from": NOW,
    }
    value = evidence(verification_status="confirmed", **binding)

    assert value.is_current
    with pytest.raises(ValidationError, match="must either all"):
        evidence(
            verification_status="confirmed",
            scope_key=binding["scope_key"],
            revision=1,
            valid_from=NOW,
        )


@pytest.mark.parametrize(
    ("event_type", "previous_status", "new_status"),
    [
        ("created", None, "pending"),
        ("confirmed", "pending", "confirmed"),
        ("rejected", "pending", "rejected"),
    ],
)
def test_career_evidence_event_accepts_valid_transitions(
    event_type: str,
    previous_status: str | None,
    new_status: str,
) -> None:
    value = evidence_event(
        event_type=event_type,
        previous_status=previous_status,
        new_status=new_status,
    )

    assert value.event_type == event_type


def test_career_evidence_event_rejects_invalid_transition() -> None:
    with pytest.raises(ValidationError, match="invalid status transition"):
        evidence_event(previous_status="confirmed", new_status="rejected")


@pytest.mark.parametrize("event_type", ["confirmed", "rejected"])
def test_evidence_decision_requires_user_actor(event_type: str) -> None:
    new_status = "confirmed" if event_type == "confirmed" else "rejected"

    with pytest.raises(ValidationError, match="requires user actor"):
        evidence_event(
            event_type=event_type,
            new_status=new_status,
            actor_type="agent",
        )


def test_lineage_event_requires_mutation_and_related_evidence() -> None:
    with pytest.raises(ValidationError, match="mutation and related"):
        evidence_event(
            event_type="corrected",
            previous_status="confirmed",
            new_status="confirmed",
        )

    value = evidence_event(
        event_type="corrected",
        previous_status="confirmed",
        new_status="confirmed",
        mutation_id="career_evidence_mutation_" + "a" * 32,
        related_evidence_id="evidence-0",
    )
    assert value.event_type == "corrected"
