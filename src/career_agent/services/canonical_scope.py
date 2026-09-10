from __future__ import annotations

from collections.abc import Mapping
import re
import unicodedata

from career_agent.domain.memory_scope import (
    CanonicalScope,
    ScopeFamily,
    ScopeProposal,
    ScopeResolution,
)


_DEFAULT_RELATIONS: dict[ScopeFamily, frozenset[str]] = {
    "person_intent": frozenset(
        {
            "default_city",
            "work_arrangement",
            "work_schedule",
            "company_scale",
        }
    ),
    "target_role_intent": frozenset(
        {"city", "salary_expectation", "experience", "education"}
    ),
}


class CanonicalScopeResolver:
    """Resolve a proposal to a stable key before any version is allocated."""

    def __init__(
        self,
        *,
        relation_aliases: Mapping[tuple[ScopeFamily, str], str] | None = None,
    ) -> None:
        self._relations = dict(_DEFAULT_RELATIONS)
        self._aliases = {
            (family, self._surface(alias)): self._canonical_relation(relation)
            for (family, alias), relation in (relation_aliases or {}).items()
        }

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
        return None

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
