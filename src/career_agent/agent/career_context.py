from __future__ import annotations

from collections import Counter
from datetime import datetime
import math
import re

from career_agent.agent.main_agent_contracts import (
    CareerMemoryClaim,
    CareerMemoryContext,
    CareerMemoryRecord,
    MemoryTelemetryBinding,
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
        tier_one_record_limit: int = 15,
        tier_one_claim_limit: int = 15,
        relevance_candidate_limit: int = 45,
        recency_half_life_days: float = 14.0,
    ) -> None:
        if candidate_record_limit < 1 or candidate_claim_limit_per_record < 1:
            raise ValueError("Career context safety limits must be positive")
        if not 1 <= tier_one_record_limit <= 15:
            raise ValueError("Tier-1 record limit must be between 1 and 15")
        if not 5 <= tier_one_claim_limit <= 15:
            raise ValueError("Tier-1 claim limit must be between 5 and 15")
        if not tier_one_claim_limit <= relevance_candidate_limit <= 100:
            raise ValueError(
                "relevance over-recall must cover Tier-1 and stay bounded"
            )
        if recency_half_life_days <= 0:
            raise ValueError("recency half-life must be positive")
        self._store = store
        self._candidate_record_limit = candidate_record_limit
        self._candidate_claim_limit_per_record = candidate_claim_limit_per_record
        self._tier_one_record_limit = tier_one_record_limit
        self._tier_one_claim_limit = tier_one_claim_limit
        self._relevance_candidate_limit = relevance_candidate_limit
        self._recency_half_life_days = recency_half_life_days

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
        relevance_hits = self._store.rank_current_evidence(
            user_id=user_id,
            query=query,
            limit=self._relevance_candidate_limit,
        )
        relevance_rank = {
            evidence.id: index
            for index, evidence in enumerate(relevance_hits)
        }
        loaded = []
        for recency, record in enumerate(records):
            evidence = self._store.list_evidence(
                user_id=user_id,
                career_record_id=record.id,
                verification_status="confirmed",
                limit=self._candidate_claim_limit_per_record,
            )
            loaded.append((record, evidence, recency))
        newest_evidence_at = max(
            (
                item.created_at
                for _, evidence, _ in loaded
                for item in evidence
            ),
            default=None,
        )
        short_query_relevance = (
            self._bounded_bm25_relevance(
                query_terms=query_terms,
                evidence=tuple(
                    item
                    for _, record_evidence, _ in loaded
                    for item in record_evidence
                ),
            )
            if not self._has_fts_query_token(query)
            else {}
        )
        candidates = []
        for record, evidence, recency in loaded:
            scored_evidence = tuple(
                (
                    item,
                    self._claim_score(
                        item,
                        query_terms=query_terms,
                        relevance_rank=relevance_rank,
                        relevance_candidate_count=len(relevance_hits),
                        fallback_relevance=short_query_relevance,
                        newest_evidence_at=newest_evidence_at,
                        record_is_current=record.is_current,
                    ),
                )
                for item in evidence
            )
            ranked_evidence = tuple(
                item
                for item, _ in sorted(
                    scored_evidence,
                    key=lambda pair: (
                        -pair[1],
                        -pair[0].created_at.timestamp(),
                        pair[0].id,
                    ),
                )
            )
            highlights = tuple(
                CareerMemoryClaim(
                    claim=item.claim,
                    origin=item.origin,
                    recorded_at=item.created_at,
                    source_ref=item.source_ref,
                    revision=item.revision,
                    detail_ref=item.detail_ref,
                    telemetry_binding=self._evidence_binding(item),
                )
                for item in ranked_evidence
            )
            record_metadata = " ".join(
                value
                for value in (
                    record.title,
                    record.organization,
                )
                if value
            )
            relevance = max(
                (score for _, score in scored_evidence),
                default=self._record_metadata_score(
                    query_terms=query_terms,
                    record_metadata=record_metadata,
                    record_is_current=record.is_current,
                ),
            )
            candidates.append((record, highlights, relevance, recency))

        selected = list(sorted(
            candidates,
            key=lambda item: (
                -item[2],
                item[3],
            ),
        ))
        current_anchor = next(
            (candidate for candidate in selected if candidate[0].is_current),
            None,
        )
        # Preserve the strongest current-role anchor without making it outrank
        # the best query match. A second-place guarantee keeps "what I do now"
        # visible while relevance still owns the first slot.
        if current_anchor is not None:
            current_index = selected.index(current_anchor)
            if current_index > 1:
                selected.pop(current_index)
                selected.insert(1, current_anchor)

        remaining_claims = self._tier_one_claim_limit
        projected = []
        for record, highlights, _, _ in selected:
            if len(projected) >= self._tier_one_record_limit:
                break
            is_current_anchor = (
                current_anchor is not None and record.id == current_anchor[0].id
            )
            anchor_pending = (
                current_anchor is not None
                and not any(item.is_current for item in projected)
            )
            if remaining_claims == 0 and not is_current_anchor:
                break
            reserved_for_anchor = int(
                anchor_pending
                and not is_current_anchor
                and bool(current_anchor[1])
            )
            # Depth-first packing remains the default, except that one claim is
            # reserved for the current anchor so the preceding record cannot
            # consume the entire claim budget.
            available_claims = (
                remaining_claims
                if is_current_anchor
                else max(0, remaining_claims - reserved_for_anchor)
            )
            selected_highlights = highlights[:available_claims]
            projected.append(
                CareerMemoryRecord(
                    record_type=record.record_type,
                    organization=record.organization,
                    title=record.title,
                    start_year=record.start_year,
                    start_month=record.start_month,
                    end_year=record.end_year,
                    end_month=record.end_month,
                    is_current=record.is_current,
                    confirmed_highlights=selected_highlights,
                )
            )
            remaining_claims -= len(selected_highlights)
        projected_records = tuple(projected)
        projected_claims = tuple(
            claim
            for record in projected_records
            for claim in record.confirmed_highlights
        )
        scope_keys = tuple(
            dict.fromkeys(
                binding.entry_id
                for claim in projected_claims
                if (binding := claim.telemetry_binding) is not None
            )
        )
        selected_scopes = scope_keys[:400]
        versions = self._store.list_evidence_versions(
            user_id=user_id,
            scope_keys=selected_scopes,
            limit=513,
        )
        clipped = len(scope_keys) > 400 or len(versions) > 512
        version_bindings = tuple(
            binding
            for item in versions[:512]
            if (binding := self._evidence_binding(item)) is not None
        )
        active_scopes = {
            binding.entry_id
            for binding in version_bindings
            if binding.lifecycle_status == "current"
        }
        return CareerMemoryContext(
            records=projected_records,
            records_total=records_total,
            claims_total=claims_total,
            telemetry_bindings=version_bindings,
            telemetry_inventory_complete=(
                not clipped
                and all(
                    claim.telemetry_binding is not None
                    for claim in projected_claims
                )
                and len(version_bindings) == min(len(versions), 512)
                and set(scope_keys) <= active_scopes
            ),
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
    def _evidence_binding(
        evidence: CareerEvidence,
    ) -> MemoryTelemetryBinding | None:
        if (
            evidence.scope_key is None
            or evidence.update_id is None
            or evidence.content_digest is None
            or evidence.revision is None
            or len(evidence.claim) > 32_000
        ):
            return None
        return MemoryTelemetryBinding(
            entry_id=evidence.scope_key,
            update_id=evidence.update_id,
            content_digest=evidence.content_digest,
            value=evidence.claim,
            revision=evidence.revision,
            lifecycle_status=(
                "rolled_back"
                if evidence.rolled_back_at is not None
                else "superseded"
                if evidence.superseded_by is not None
                else "current"
            ),
        )

    @staticmethod
    def _terms(value: str) -> frozenset[str]:
        return frozenset(CareerContextProjector._term_sequence(value))

    @staticmethod
    def _term_sequence(value: str) -> tuple[str, ...]:
        normalized = value.casefold()
        latin = re.findall(r"[a-z0-9][a-z0-9.+#-]*", normalized)
        chinese_chunks = re.findall(r"[\u4e00-\u9fff]+", normalized)
        chinese = [
            chunk[index : index + 2]
            for chunk in chinese_chunks
            for index in range(max(1, len(chunk) - 1))
        ]
        return tuple((*latin, *chinese))

    @staticmethod
    def _overlap(left: frozenset[str], right: frozenset[str]) -> int:
        return len(left.intersection(right))

    @staticmethod
    def _has_fts_query_token(value: str) -> bool:
        normalized = value.casefold()
        return bool(
            re.search(r"[a-z0-9+#.-]{3,}", normalized)
            or re.search(r"[\u4e00-\u9fff]{3,}", normalized)
        )

    @classmethod
    def _bounded_bm25_relevance(
        cls,
        *,
        query_terms: frozenset[str],
        evidence: tuple[CareerEvidence, ...],
    ) -> dict[str, float]:
        """Length-normalized relevance for queries the trigram index cannot serve."""

        if not query_terms or not evidence:
            return {}
        documents = {
            item.id: cls._term_sequence(item.claim) for item in evidence
        }
        average_length = sum(map(len, documents.values())) / len(documents)
        document_frequency = {
            term: sum(term in tokens for tokens in documents.values())
            for term in query_terms
        }
        raw_scores: dict[str, float] = {}
        k1 = 1.2
        length_weight = 0.75
        for evidence_id, tokens in documents.items():
            frequencies = Counter(tokens)
            document_length = len(tokens)
            score = 0.0
            for term in query_terms:
                frequency = frequencies[term]
                if not frequency:
                    continue
                inverse_document_frequency = math.log(
                    1
                    + (
                        len(documents)
                        - document_frequency[term]
                        + 0.5
                    )
                    / (document_frequency[term] + 0.5)
                )
                denominator = frequency + k1 * (
                    1
                    - length_weight
                    + length_weight
                    * document_length
                    / max(1.0, average_length)
                )
                score += (
                    inverse_document_frequency
                    * frequency
                    * (k1 + 1)
                    / denominator
                )
            raw_scores[evidence_id] = score
        maximum = max(raw_scores.values(), default=0.0)
        if maximum <= 0:
            return {}
        return {
            evidence_id: score / maximum
            for evidence_id, score in raw_scores.items()
        }

    @classmethod
    def _record_metadata_score(
        cls,
        *,
        query_terms: frozenset[str],
        record_metadata: str,
        record_is_current: bool,
    ) -> float:
        relevance = min(
            1.0,
            cls._overlap(query_terms, cls._terms(record_metadata))
            / max(1, len(query_terms)),
        )
        structural_importance = 1.0 if record_is_current else 0.5
        # No claim timestamp means there is no honest recency signal to score.
        return 0.4 * relevance + 0.3 * structural_importance

    def _claim_score(
        self,
        evidence: CareerEvidence,
        *,
        query_terms: frozenset[str],
        relevance_rank: dict[str, int],
        relevance_candidate_count: int,
        fallback_relevance: dict[str, float],
        newest_evidence_at: datetime | None,
        record_is_current: bool,
    ) -> float:
        """Blend available relevance, recency, and structural importance."""

        rank = relevance_rank.get(evidence.id)
        if rank is not None and relevance_candidate_count:
            relevance = (
                relevance_candidate_count - rank
            ) / relevance_candidate_count
        elif evidence.id in fallback_relevance:
            relevance = fallback_relevance[evidence.id]
        else:
            relevance = min(
                1.0,
                self._overlap(query_terms, self._terms(evidence.claim))
                / max(1, len(query_terms)),
            )
        recency = 1.0
        if newest_evidence_at is not None:
            age_days = max(
                0.0,
                (
                    newest_evidence_at - evidence.created_at
                ).total_seconds()
                / 86_400,
            )
            recency = 0.5 ** (age_days / self._recency_half_life_days)
        structural_importance = 1.0 if record_is_current else 0.5
        return (
            0.4 * relevance
            + 0.3 * recency
            + 0.3 * structural_importance
        )
