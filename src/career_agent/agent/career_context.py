from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import math

from career_agent.agent.main_agent_contracts import (
    CareerMemoryClaim,
    CareerMemoryContext,
    CareerMemoryRecord,
    MemoryTelemetryBinding,
)
from career_agent.domain.career_history import CareerEvidence
from career_agent.storage.career_history import (
    CareerEvidenceQueryTerms,
    CareerHistoryStore,
    RankedCareerEvidence,
)

# Generative Agents (Park et al., UIST 2023) does not justify a 0.4/0.3/0.3
# split: the paper uses equal alpha values and normalizes all three components,
# while its released implementation uses gw=[0.5, 3, 2]. The only claimed
# property of this provisional split is the safety constraint below. The
# specific triple still needs a judgment set.
_RELEVANCE_WEIGHT = 0.6
_RECENCY_WEIGHT = 0.2
_STRUCTURAL_WEIGHT = 0.2


def _weights_preserve_hit_precedence(
    relevance: float,
    recency: float,
    structural: float,
) -> bool:
    relevance_decimal = Decimal(str(relevance))
    required = 2 * Decimal(str(recency)) + Decimal(str(structural))
    return relevance_decimal >= required


if not _weights_preserve_hit_precedence(
    _RELEVANCE_WEIGHT,
    _RECENCY_WEIGHT,
    _STRUCTURAL_WEIGHT,
):
    raise AssertionError(
        "weights let a fresh current non-match outrank a historical hit"
    )


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
        query_terms = self._store.current_evidence_query_terms(
            user_id=user_id,
            query=query,
        )
        ranked_hits = self._store.rank_current_evidence(
            user_id=user_id,
            query_terms=query_terms,
            limit=self._relevance_candidate_limit,
        )
        fts_relevance = self._fts_relevance(
            ranked_hits,
            query_terms=query_terms,
        )
        loaded = []
        for recency, record in enumerate(records):
            evidence = self._store.list_evidence(
                user_id=user_id,
                career_record_id=record.id,
                verification_status="confirmed",
                limit=self._candidate_claim_limit_per_record,
            )
            loaded.append((record, evidence, recency))
        observed_at = datetime.now(timezone.utc)
        candidates = []
        for record, evidence, recency in loaded:
            scored_evidence = tuple(
                (
                    item,
                    self._claim_score(
                        item,
                        fts_relevance=fts_relevance,
                        observed_at=observed_at,
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
            relevance = max(
                (score for _, score in scored_evidence),
                default=self._record_metadata_score(
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
                "superseded" if evidence.superseded_by is not None else "current"
            ),
        )

    @staticmethod
    def _fts_relevance(
        hits: tuple[RankedCareerEvidence, ...],
        *,
        query_terms: CareerEvidenceQueryTerms,
    ) -> dict[str, float]:
        """Map BM25 through a dimensionless, result-set-independent sigmoid.

        A hit maps to [0.5, 1), while an item outside the hit set remains 0.
        Dividing by the query's corpus-level IDF sum removes BM25's dominant
        query-length and collection-size scale without using hit-set statistics.
        """

        if not hits:
            return {}
        if not query_terms.match_tokens or query_terms.idf_sum <= 0:
            raise ValueError("FTS hits require query terms with positive IDF")
        return {
            hit.evidence.id: CareerContextProjector._stable_sigmoid(
                max(0.0, -hit.bm25_score) / query_terms.idf_sum
            )
            for hit in hits
        }
    @staticmethod
    def _stable_sigmoid(value: float) -> float:
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        exponent = math.exp(value)
        return exponent / (1.0 + exponent)

    @staticmethod
    def _record_metadata_score(*, record_is_current: bool) -> float:
        # No claim in the FTS candidate set, so relevance is 0. Recency has
        # nothing to attach to. Structural importance still distinguishes a
        # current role from a historical empty record.
        return _STRUCTURAL_WEIGHT * (1.0 if record_is_current else 0.5)

    def _claim_score(
        self,
        evidence: CareerEvidence,
        *,
        fts_relevance: dict[str, float],
        observed_at: datetime,
        record_is_current: bool,
    ) -> float:
        """Blend FTS relevance with recency and structural importance.

        Recency and structural importance only rerank; they cannot pull a
        claim into the relevance candidate set.
        """

        relevance = fts_relevance.get(evidence.id, 0.0)
        age_days = max(
            0.0,
            (observed_at - evidence.created_at).total_seconds() / 86_400,
        )
        recency = 0.5 ** (age_days / self._recency_half_life_days)
        structural_importance = 1.0 if record_is_current else 0.5
        return (
            _RELEVANCE_WEIGHT * relevance
            + _RECENCY_WEIGHT * recency
            + _STRUCTURAL_WEIGHT * structural_importance
        )
