from __future__ import annotations

import re

from career_agent.agent.main_agent_contracts import (
    CareerMemoryClaim,
    CareerMemoryContext,
    CareerMemoryRecord,
)
from career_agent.domain.career_history import CareerEvidence
from career_agent.storage.career_history import CareerHistoryStore


class CareerContextProjector:
    """Projects durable confirmed career facts into bounded agent working memory."""

    def __init__(
        self,
        store: CareerHistoryStore,
        *,
        candidate_record_limit: int = 100,
        candidate_claim_limit_per_record: int = 100,
    ) -> None:
        if candidate_record_limit < 1 or candidate_claim_limit_per_record < 1:
            raise ValueError("Career context safety limits must be positive")
        self._store = store
        self._candidate_record_limit = candidate_record_limit
        self._candidate_claim_limit_per_record = candidate_claim_limit_per_record

    def project(self, *, user_id: str, query: str) -> CareerMemoryContext:
        records_total = self._store.count_records(user_id=user_id)
        claims_total = self._store.count_evidence(
            user_id=user_id,
            verification_status="confirmed",
        )
        records = self._store.list_records(
            user_id=user_id,
            limit=self._candidate_record_limit,
        )
        query_terms = self._terms(query)
        candidates = []
        for recency, record in enumerate(records):
            evidence = self._store.list_evidence(
                user_id=user_id,
                career_record_id=record.id,
                verification_status="confirmed",
                limit=self._candidate_claim_limit_per_record,
            )
            ranked_evidence = sorted(
                evidence,
                key=lambda item: (
                    -self._overlap(query_terms, self._terms(item.claim)),
                    item.created_at,
                    item.id,
                ),
            )
            highlights = tuple(
                CareerMemoryClaim(
                    claim=item.claim,
                    origin=item.origin,
                    recorded_at=item.created_at,
                    source_ref=item.source_ref,
                    revision=item.revision,
                    detail_ref=item.detail_ref,
                )
                for item in ranked_evidence
            )
            searchable = " ".join(
                value
                for value in (
                    record.title,
                    record.organization,
                    *(highlight.claim for highlight in highlights),
                )
                if value
            )
            relevance = self._overlap(query_terms, self._terms(searchable))
            candidates.append((record, highlights, relevance, recency))

        selected = sorted(
            candidates,
            key=lambda item: (
                -int(item[0].is_current),
                -item[2],
                item[3],
            ),
        )
        return CareerMemoryContext(
            records=tuple(
                CareerMemoryRecord(
                    record_type=record.record_type,
                    organization=record.organization,
                    title=record.title,
                    start_year=record.start_year,
                    start_month=record.start_month,
                    end_year=record.end_year,
                    end_month=record.end_month,
                    is_current=record.is_current,
                    confirmed_highlights=highlights,
                )
                for record, highlights, _, _ in selected
            ),
            records_total=records_total,
            claims_total=claims_total,
        )

    def resolve_source_ref(
        self, *, user_id: str, source_ref: str
    ) -> CareerEvidence | None:
        """Dereference provenance only when a caller explicitly requests it."""

        return self._store.get_evidence_by_source_ref(
            user_id=user_id,
            source_ref=source_ref,
        )

    @staticmethod
    def _terms(value: str) -> frozenset[str]:
        normalized = value.casefold()
        latin = re.findall(r"[a-z0-9][a-z0-9.+#-]*", normalized)
        chinese_chunks = re.findall(r"[\u4e00-\u9fff]+", normalized)
        chinese = [
            chunk[index : index + 2]
            for chunk in chinese_chunks
            for index in range(max(1, len(chunk) - 1))
        ]
        return frozenset((*latin, *chinese))

    @staticmethod
    def _overlap(left: frozenset[str], right: frozenset[str]) -> int:
        return len(left.intersection(right))
