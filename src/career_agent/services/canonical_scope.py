from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol
import re
import unicodedata

from career_agent.domain.memory_scope import (
    CanonicalScope,
    ScopeFamily,
    ScopeProposal,
    ScopeResolution,
)


class SemanticRelationNormalizer(Protocol):
    """Proposes one relation from a closed registry; it never returns a key."""

    def normalize(
        self,
        *,
        family: ScopeFamily,
        surface_relation: str,
        allowed_relations: tuple[str, ...],
    ) -> str | None: ...


_DEFAULT_RELATIONS: dict[ScopeFamily, frozenset[str]] = {
    "person_intent": frozenset({"default_city"}),
    "target_role_intent": frozenset(
        {"city", "salary_expectation", "experience", "education"}
    ),
    # Free-text career evidence has no safe built-in predicate registry yet.
    # M2b may add domain predicates deliberately; hashing the whole claim here
    # would turn every correction into a different key and defeat supersession.
    "career_evidence": frozenset(),
}


class CanonicalScopeResolver:
    """Resolve a proposal to a stable key before any version is allocated."""

    def __init__(
        self,
        *,
        relation_aliases: Mapping[tuple[ScopeFamily, str], str] | None = None,
        evidence_relations: frozenset[str] = frozenset(),
        semantic_normalizer: SemanticRelationNormalizer | None = None,
    ) -> None:
        relations = dict(_DEFAULT_RELATIONS)
        relations["career_evidence"] = frozenset(
            self._canonical_relation(item) for item in evidence_relations
        )
        self._relations = relations
        self._aliases = {
            (family, self._surface(alias)): self._canonical_relation(relation)
            for (family, alias), relation in (relation_aliases or {}).items()
        }
        self._semantic_normalizer = semantic_normalizer

    def resolve(self, proposal: ScopeProposal) -> ScopeResolution:
        relation = self._resolve_relation(proposal)
        if relation is None:
            candidates = tuple(
                f"{proposal.family}/{proposal.subject_id}/{candidate}"
                for candidate in sorted(self._relations[proposal.family])
            )
            return ScopeResolution(
                proposal=proposal,
                reason="No canonical relation matched the proposed memory field.",
                candidate_scope_keys=candidates,
            )
        if not self._safe_subject(proposal.subject_id):
            return ScopeResolution(
                proposal=proposal,
                reason="The subject identifier is not safe for canonical key derivation.",
            )
        if proposal.family == "person_intent" and proposal.subject_id != "self":
            return ScopeResolution(
                proposal=proposal,
                reason="Person-level intent must use the per-user 'self' subject.",
            )
        scope = CanonicalScope(
            family=proposal.family,
            subject_id=proposal.subject_id,
            relation=relation,
            scope_key=f"{proposal.family}/{proposal.subject_id}/{relation}",
        )
        return ScopeResolution(proposal=proposal, canonical_scope=scope)

    def _resolve_relation(self, proposal: ScopeProposal) -> str | None:
        surface = self._surface(proposal.relation)
        alias = self._aliases.get((proposal.family, surface))
        if alias in self._relations[proposal.family]:
            return alias
        direct = self._canonical_relation(proposal.relation)
        if direct in self._relations[proposal.family]:
            return direct
        if self._semantic_normalizer is None:
            return None
        proposed = self._semantic_normalizer.normalize(
            family=proposal.family,
            surface_relation=proposal.relation,
            allowed_relations=tuple(sorted(self._relations[proposal.family])),
        )
        if proposed is None:
            return None
        canonical = self._canonical_relation(proposed)
        return canonical if canonical in self._relations[proposal.family] else None

    @staticmethod
    def _surface(value: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", value).casefold().split())

    @staticmethod
    def _canonical_relation(value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value).strip().casefold()
        return re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")

    @staticmethod
    def _safe_subject(value: str) -> bool:
        return re.fullmatch(r"[A-Za-z0-9_.:-]+", value) is not None
