from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from career_agent.agent.main_agent_contracts import (
    CareerMemoryClaim,
    CareerMemoryContext,
    CareerMemoryRecord,
    MemoryTelemetryBinding,
)
from career_agent.domain.career_history import CareerEvidence
from career_agent.storage.career_history import CareerHistoryStore


class CareerEvidenceSemanticRetriever(Protocol):
    """Embedding-backed channel; results must be best-first evidence ids."""

    def rank_current_evidence_ids(
        self,
        *,
        user_id: str,
        query: str,
        limit: int,
    ) -> Sequence[str]: ...


def reciprocal_rank_fusion(
    *rankings: Sequence[str],
    rank_constant: int = 60,
) -> dict[str, float]:
    """Fuse independent retrieval rankings without score calibration."""

    if rank_constant < 1:
        raise ValueError("RRF rank constant must be positive")
    fused: dict[str, float] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, evidence_id in enumerate(ranking, start=1):
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            fused[evidence_id] = fused.get(evidence_id, 0.0) + 1.0 / (
                rank_constant + rank
            )
    return fused


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
        semantic_retriever: CareerEvidenceSemanticRetriever | None = None,
        rrf_rank_constant: int = 60,
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
        if rrf_rank_constant < 1:
            raise ValueError("RRF rank constant must be positive")
        self._store = store
        self._candidate_record_limit = candidate_record_limit
        self._candidate_claim_limit_per_record = candidate_claim_limit_per_record
        self._tier_one_record_limit = tier_one_record_limit
        self._tier_one_claim_limit = tier_one_claim_limit
        self._relevance_candidate_limit = relevance_candidate_limit
        self._semantic_retriever = semantic_retriever
        self._rrf_rank_constant = rrf_rank_constant

    def project(self, *, user_id: str, query: str) -> CareerMemoryContext:
        records_total = self._store.count_records(user_id=user_id)
        claims_total = self._store.count_evidence(
            user_id=user_id,
            verification_status="confirmed",
        )
        query_terms = self._store.current_evidence_query_terms(
            user_id=user_id,
            query=query,
        )
        lexical_hits = self._store.rank_current_evidence(
            user_id=user_id,
            query_terms=query_terms,
            limit=self._relevance_candidate_limit,
        )
        semantic_ids = tuple(
            dict.fromkeys(
                self._semantic_retriever.rank_current_evidence_ids(
                    user_id=user_id,
                    query=query,
                    limit=self._relevance_candidate_limit,
                )
            )
        ) if self._semantic_retriever is not None else ()
        evidence_by_id = {item.id: item for item in lexical_hits}
        for evidence_id in semantic_ids:
            if evidence_id in evidence_by_id:
                continue
            evidence = self._store.get_evidence(
                user_id=user_id,
                career_evidence_id=evidence_id,
            )
            if (
                evidence is not None
                and evidence.verification_status == "confirmed"
                and evidence.superseded_by is None
                and evidence.tombstoned_at is None
            ):
                evidence_by_id[evidence.id] = evidence

        records_by_id = {}
        for hit in evidence_by_id.values():
            if hit.career_record_id in records_by_id:
                continue
            if len(records_by_id) >= self._candidate_record_limit:
                break
            record = self._store.get_record(
                user_id=user_id,
                career_record_id=hit.career_record_id,
            )
            if record is not None:
                records_by_id[record.id] = record

        lexical_ids = tuple(
            item.id
            for item in sorted(
                lexical_hits,
                key=lambda item: (
                    0
                    if records_by_id.get(item.career_record_id) is not None
                    and records_by_id[item.career_record_id].is_current
                    else 1,
                    -item.created_at.timestamp(),
                    item.id,
                ),
            )
            if item.career_record_id in records_by_id
        )
        eligible_semantic_ids = tuple(
            evidence_id
            for evidence_id in semantic_ids
            if evidence_id in evidence_by_id
            and evidence_by_id[evidence_id].career_record_id in records_by_id
        )
        fused = reciprocal_rank_fusion(
            lexical_ids,
            eligible_semantic_ids,
            rank_constant=self._rrf_rank_constant,
        )
        ranked_evidence = sorted(
            (
                item
                for item in evidence_by_id.values()
                if item.career_record_id in records_by_id
            ),
            key=lambda item: (
                -fused[item.id],
                0 if records_by_id[item.career_record_id].is_current else 1,
                -item.created_at.timestamp(),
                item.id,
            ),
        )
        evidence_by_record: dict[str, list[CareerEvidence]] = {}
        for item in ranked_evidence:
            bucket = evidence_by_record.setdefault(item.career_record_id, [])
            if len(bucket) < self._candidate_claim_limit_per_record:
                bucket.append(item)

        candidates = []
        for record_id, evidence in evidence_by_record.items():
            record = records_by_id[record_id]
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
                for item in evidence
            )
            candidates.append((record, highlights))

        remaining_claims = self._tier_one_claim_limit
        projected = []
        for record, highlights in candidates:
            if len(projected) >= self._tier_one_record_limit:
                break
            if remaining_claims == 0:
                break
            selected_highlights = highlights[:remaining_claims]
            projected.append(
                CareerMemoryRecord(
                    record_id=record.id,
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
                "superseded" if evidence.superseded_by is not None else "current"
            ),
        )
