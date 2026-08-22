from __future__ import annotations

from datetime import datetime
from typing import Iterable

from career_agent.domain.job_discovery import (
    ContractModel,
    DuplicateCandidate,
    JDSnapshot,
    JobDetail,
    JobPosting,
    WaitlistItem,
    WaitlistStatus,
    company_title_fingerprint,
    content_fingerprint,
    jd_content_hash,
    new_id,
)


class PromotionRecords(ContractModel):
    posting: JobPosting
    snapshot: JDSnapshot
    waitlist: WaitlistItem


class InMemoryJobRepository:
    """Replaceable repository for contract and service tests."""

    def __init__(self) -> None:
        self.postings: dict[str, JobPosting] = {}
        self.snapshots: dict[str, list[JDSnapshot]] = {}
        self.waitlist: dict[str, WaitlistItem] = {}
        self._idempotency: dict[str, PromotionRecords] = {}

    def get_idempotent(self, key: str) -> PromotionRecords | None:
        return self._idempotency.get(key)

    def find_duplicates(self, user_id: str, detail: JobDetail, normalized_description: str) -> list[DuplicateCandidate]:
        source_id = detail.source_job_id
        title_fingerprint = company_title_fingerprint(detail.title, detail.company_name)
        body_fingerprint = content_fingerprint(detail.title, detail.company_name, normalized_description)
        matches: list[DuplicateCandidate] = []
        for posting in self.postings.values():
            if posting.user_id != user_id:
                continue
            if source_id and posting.source_name == detail.source_name and posting.source_job_id == source_id:
                matches.append(DuplicateCandidate(job_posting_id=posting.id, rule="source_id", title=posting.title, company_name=posting.company_name))
            elif posting.company_title_fingerprint == title_fingerprint:
                matches.append(DuplicateCandidate(job_posting_id=posting.id, rule="company_title", title=posting.title, company_name=posting.company_name))
            elif posting.content_fingerprint == body_fingerprint:
                matches.append(DuplicateCandidate(job_posting_id=posting.id, rule="content", title=posting.title, company_name=posting.company_name))
        return matches

    def snapshots_for(self, posting_id: str) -> list[JDSnapshot]:
        return list(self.snapshots.get(posting_id, ()))

    def active_waitlist_for(self, user_id: str, posting_id: str) -> WaitlistItem | None:
        return next(
            (item for item in self.waitlist.values() if item.user_id == user_id and item.job_posting_id == posting_id and item.status is WaitlistStatus.ACTIVE),
            None,
        )

    def save_records(self, records: PromotionRecords, idempotency_key: str) -> PromotionRecords:
        self.postings[records.posting.id] = records.posting
        self.snapshots.setdefault(records.posting.id, []).append(records.snapshot)
        self.waitlist[records.waitlist.id] = records.waitlist
        self._idempotency[idempotency_key] = records
        return records

    def all_waitlist(self) -> Iterable[WaitlistItem]:
        return self.waitlist.values()

    def next_snapshot_version(self, posting_id: str) -> int:
        return len(self.snapshots.get(posting_id, ())) + 1

    def build_records(
        self,
        *,
        user_id: str,
        detail: JobDetail,
        normalized_description: str,
        target_role_id: str,
        target_role_title: str,
        now: datetime,
        posting_id: str | None = None,
    ) -> PromotionRecords:
        posting_id = posting_id or new_id("job")
        snapshot_id = new_id("jd")
        posting = JobPosting(
            id=posting_id,
            user_id=user_id,
            title=detail.title.strip(),
            company_name=detail.company_name.strip(),
            source_name=detail.source_name,
            source_job_id=detail.source_job_id,
            source_url=detail.source_url,
            external_status="active",
            persisted_at=now,
            last_seen_at=now,
            latest_snapshot_id=snapshot_id,
            company_title_fingerprint=company_title_fingerprint(detail.title, detail.company_name),
            content_fingerprint=content_fingerprint(detail.title, detail.company_name, normalized_description),
        )
        snapshot = JDSnapshot(
            id=snapshot_id,
            job_posting_id=posting_id,
            version=self.next_snapshot_version(posting_id),
            content=normalized_description,
            content_hash=jd_content_hash(normalized_description),
            captured_at=detail.captured_at,
            provenance=detail.provenance,
            normalizer_version="jd-text-v1",
        )
        waitlist = WaitlistItem(
            id=new_id("wait"),
            user_id=user_id,
            job_posting_id=posting_id,
            target_role_id=target_role_id,
            target_role_title=target_role_title,
            status=WaitlistStatus.ACTIVE,
            added_at=now,
        )
        return PromotionRecords(posting=posting, snapshot=snapshot, waitlist=waitlist)
