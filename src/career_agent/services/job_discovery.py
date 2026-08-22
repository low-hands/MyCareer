from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Sequence

from career_agent.domain.job_discovery import (
    ContractModel,
    DuplicateCandidate,
    IncompleteJobDetail,
    JobDetail,
    validate_job_detail,
)
from career_agent.storage.memory import InMemoryJobRepository, PromotionRecords


DedupDecision = Literal["merge_existing", "keep_separate"]


class PromotionRequest(ContractModel):
    user_id: str
    detail: JobDetail
    target_role_id: str
    target_role_title: str
    idempotency_key: str
    dedup_decision: DedupDecision | None = None
    duplicate_job_posting_id: str | None = None


class PromotionSuccess(ContractModel):
    status: Literal["created", "merged"]
    records: PromotionRecords


class PromotionFailure(ContractModel):
    status: Literal["failed", "waiting_input"]
    code: str
    message: str
    recoverable: bool
    candidates: tuple[DuplicateCandidate, ...] = ()


PromotionResult = PromotionSuccess | PromotionFailure


class JobDiscoveryService:
    def __init__(self, repository: InMemoryJobRepository) -> None:
        self.repository = repository

    def confirm_and_add_to_waitlist(self, request: PromotionRequest, *, now: datetime | None = None) -> PromotionResult:
        if not request.target_role_id or not request.target_role_title.strip():
            return PromotionFailure(status="waiting_input", code="TARGET_ROLE_REQUIRED", message="A confirmed target role is required.", recoverable=True)

        existing = self.repository.get_idempotent(request.idempotency_key)
        if existing is not None:
            return PromotionSuccess(status="created", records=existing)

        try:
            normalized = validate_job_detail(request.detail)
        except IncompleteJobDetail as error:
            return PromotionFailure(status="waiting_input", code=error.code, message=str(error), recoverable=error.recoverable)

        candidates = tuple(self.repository.find_duplicates(request.user_id, request.detail, normalized))
        if candidates and request.dedup_decision is None:
            return PromotionFailure(
                status="waiting_input",
                code="DUPLICATE_DECISION_REQUIRED",
                message="A possible duplicate needs a user decision.",
                recoverable=True,
                candidates=candidates,
            )

        timestamp = now or datetime.now(timezone.utc)
        if candidates and request.dedup_decision == "merge_existing":
            candidate_id = request.duplicate_job_posting_id or candidates[0].job_posting_id
            if candidate_id not in {candidate.job_posting_id for candidate in candidates}:
                return PromotionFailure(status="failed", code="INVALID_DUPLICATE_REFERENCE", message="The duplicate reference is not a candidate.", recoverable=False)
            records = self.repository.build_records(
                user_id=request.user_id,
                detail=request.detail,
                normalized_description=normalized,
                target_role_id=request.target_role_id,
                target_role_title=request.target_role_title,
                now=timestamp,
                posting_id=candidate_id,
            )
            existing_waitlist = self.repository.active_waitlist_for(request.user_id, candidate_id)
            if existing_waitlist is not None and existing_waitlist.target_role_id == request.target_role_id:
                records = PromotionRecords(posting=records.posting, snapshot=records.snapshot, waitlist=existing_waitlist)
            return PromotionSuccess(status="merged", records=self.repository.save_records(records, request.idempotency_key))

        records = self.repository.build_records(
            user_id=request.user_id,
            detail=request.detail,
            normalized_description=normalized,
            target_role_id=request.target_role_id,
            target_role_title=request.target_role_title,
            now=timestamp,
        )
        return PromotionSuccess(status="created", records=self.repository.save_records(records, request.idempotency_key))

    def confirm_and_add_batch_to_waitlist(self, requests: Sequence[PromotionRequest], *, now: datetime | None = None) -> tuple[PromotionResult, ...]:
        return tuple(self.confirm_and_add_to_waitlist(request, now=now) for request in requests)
