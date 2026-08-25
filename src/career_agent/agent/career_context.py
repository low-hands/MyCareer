from __future__ import annotations

import re

from career_agent.agent.main_agent_contracts import (
    CareerMemoryContext,
    CareerMemoryRecord,
)
from career_agent.storage.career_history import CareerHistoryStore


class CareerContextProjector:
    """Projects durable confirmed career facts into bounded agent working memory."""

    def __init__(
        self,
        store: CareerHistoryStore,
        *,
        max_records: int = 5,
        max_highlights_per_record: int = 3,
    ) -> None:
        if max_records < 1 or max_highlights_per_record < 1:
            raise ValueError("Career context limits must be positive")
        self._store = store
        self._max_records = max_records
        self._max_highlights_per_record = max_highlights_per_record

    def project(self, *, user_id: str, query: str) -> CareerMemoryContext:
        records = self._store.list_records(user_id=user_id)
        query_terms = self._terms(query)
        candidates = []
        for recency, record in enumerate(records):
            evidence = self._store.list_evidence(
                user_id=user_id,
                career_record_id=record.id,
                verification_status="confirmed",
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
                item.claim
                for item in ranked_evidence[: self._max_highlights_per_record]
            )
            searchable = " ".join(
                value
                for value in (record.title, record.organization, *highlights)
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
        )[: self._max_records]
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
            )
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
