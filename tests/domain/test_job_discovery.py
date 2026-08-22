from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from career_agent.domain.job_discovery import JobDetail, Provenance, WaitlistStatus
from career_agent.services.job_discovery import JobDiscoveryService, PromotionRequest, PromotionSuccess
from career_agent.storage.memory import InMemoryJobRepository


NOW = datetime(2026, 8, 17, tzinfo=timezone.utc)


def detail(*, title: str = "Backend Engineer", company: str = "Acme", description: str = "Build reliable services.", source_job_id: str | None = "job-1") -> JobDetail:
    return JobDetail(
        source_name="boss",
        source_job_id=source_job_id,
        security_id="security-1",
        title=title,
        company_name=company,
        description=description,
        captured_at=NOW,
        provenance=Provenance(
            source_name="boss",
            captured_at=NOW,
            operation="detail",
            adapter_version="test",
            source_job_id=source_job_id,
        ),
    )


def request(job: JobDetail, *, key: str = "request-1", decision=None, duplicate_id=None) -> PromotionRequest:
    return PromotionRequest(
        user_id="user-1",
        detail=job,
        target_role_id="role-backend",
        target_role_title="Backend Engineer",
        idempotency_key=key,
        dedup_decision=decision,
        duplicate_job_posting_id=duplicate_id,
    )


def test_promotion_creates_immutable_snapshot_and_active_waitlist() -> None:
    repository = InMemoryJobRepository()
    result = JobDiscoveryService(repository).confirm_and_add_to_waitlist(request(detail()), now=NOW)

    assert isinstance(result, PromotionSuccess)
    assert result.records.waitlist.status is WaitlistStatus.ACTIVE
    assert result.records.snapshot.version == 1
    assert result.records.posting.source_url is None
    assert result.records.snapshot.content == "Build reliable services."


def test_empty_description_never_persists() -> None:
    repository = InMemoryJobRepository()
    result = JobDiscoveryService(repository).confirm_and_add_to_waitlist(request(detail(description="   ")), now=NOW)

    assert result.status == "waiting_input"
    assert result.code == "INCOMPLETE_JD"
    assert not repository.postings
    assert not repository.waitlist


def test_target_role_is_required() -> None:
    repository = InMemoryJobRepository()
    invalid = PromotionRequest(
        user_id="user-1",
        detail=detail(),
        target_role_id="",
        target_role_title="",
        idempotency_key="request-1",
    )

    result = JobDiscoveryService(repository).confirm_and_add_to_waitlist(invalid, now=NOW)

    assert result.code == "TARGET_ROLE_REQUIRED"
    assert not repository.postings


def test_duplicate_requires_explicit_user_decision() -> None:
    repository = InMemoryJobRepository()
    service = JobDiscoveryService(repository)
    first = service.confirm_and_add_to_waitlist(request(detail(), key="first"), now=NOW)
    assert isinstance(first, PromotionSuccess)

    duplicate = service.confirm_and_add_to_waitlist(request(detail(), key="second"), now=NOW)

    assert duplicate.status == "waiting_input"
    assert duplicate.code == "DUPLICATE_DECISION_REQUIRED"
    assert duplicate.candidates[0].job_posting_id == first.records.posting.id


def test_duplicate_can_merge_after_user_decision() -> None:
    repository = InMemoryJobRepository()
    service = JobDiscoveryService(repository)
    first = service.confirm_and_add_to_waitlist(request(detail(), key="first"), now=NOW)
    assert isinstance(first, PromotionSuccess)

    merged = service.confirm_and_add_to_waitlist(
        request(detail(description="Build reliable services with Python."), key="second", decision="merge_existing", duplicate_id=first.records.posting.id),
        now=NOW,
    )

    assert isinstance(merged, PromotionSuccess)
    assert merged.status == "merged"
    assert merged.records.posting.id == first.records.posting.id
    assert len(repository.snapshots_for(first.records.posting.id)) == 2


def test_idempotency_reuses_existing_promotion() -> None:
    repository = InMemoryJobRepository()
    service = JobDiscoveryService(repository)
    first = service.confirm_and_add_to_waitlist(request(detail(), key="same"), now=NOW)
    replay = service.confirm_and_add_to_waitlist(request(detail(), key="same"), now=NOW)

    assert isinstance(first, PromotionSuccess)
    assert isinstance(replay, PromotionSuccess)
    assert replay.records.waitlist.id == first.records.waitlist.id
    assert len(repository.postings) == 1




def test_contract_rejects_unknown_fields() -> None:
    payload = detail().model_dump()
    payload["unexpected"] = "must not cross the boundary"

    with pytest.raises(ValidationError):
        JobDetail.model_validate(payload)


def test_archiving_waitlist_requires_reason_and_timestamp() -> None:
    repository = InMemoryJobRepository()
    result = JobDiscoveryService(repository).confirm_and_add_to_waitlist(request(detail()), now=NOW)
    assert isinstance(result, PromotionSuccess)

    archived = result.records.waitlist.archive("user_removed", NOW)

    assert archived.status is WaitlistStatus.ARCHIVED
    assert archived.archive_reason == "user_removed"
    assert archived.archived_at == NOW


def test_batch_keeps_success_when_another_item_is_incomplete() -> None:
    repository = InMemoryJobRepository()
    service = JobDiscoveryService(repository)

    results = service.confirm_and_add_batch_to_waitlist((request(detail(), key="good"), request(detail(title="Data Engineer", description=""), key="bad")), now=NOW)

    assert results[0].status == "created"
    assert results[1].code == "INCOMPLETE_JD"
    assert len(repository.postings) == 1
