from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from pydantic import BaseModel, ConfigDict

from career_agent.domain.intent_memory import IntentMemoryVersion


class PreferenceResolutionContext(BaseModel):
    """Named scopes that are true for the recommendation being assembled."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_role_id: str | None = None
    role_domains: tuple[str, ...] = ()
    job_posting_id: str | None = None
    conversation_id: str | None = None

    def matching_scope_names(self) -> frozenset[str]:
        names = {
            *(f"role.{value}" for value in self.role_domains),
        }
        if self.target_role_id:
            names.add(f"role.{self.target_role_id.casefold()}")
        if self.job_posting_id:
            names.add(f"job.{self.job_posting_id.casefold()}")
        if self.conversation_id:
            names.add(f"conversation.{self.conversation_id.casefold()}")
        return frozenset(names)


def resolve_effective_preferences(
    versions: Iterable[IntentMemoryVersion],
    *,
    context: PreferenceResolutionContext,
    now: datetime | None = None,
) -> tuple[IntentMemoryVersion, ...]:
    """Resolve active preference layers before anything reaches a consumer.

    Stable person-level values are identity-like and always lead the result.
    For every other canonical scope, all rows at the narrowest matching layer
    win together. Keeping ties is what lets the structured and free-text tracks
    carry the same scope without one silently deleting the other.
    """

    observed_at = now or datetime.now(timezone.utc)
    matching_names = context.matching_scope_names()
    eligible = tuple(
        item
        for item in versions
        if item.superseded_at is None
        and item.admission_status == "active"
        and (
            item.valid_until is None
            or observed_at < item.valid_until
        )
    )
    stable = tuple(
        sorted(
            (item for item in eligible if item.layer == "stable"),
            key=_result_key,
        )
    )
    by_scope: dict[str, list[tuple[int, IntentMemoryVersion]]] = {}
    for item in eligible:
        if item.layer == "stable":
            continue
        rank = _matching_rank(item, matching_names)
        if rank is None:
            continue
        by_scope.setdefault(item.scope_key, []).append((rank, item))

    resolved: list[IntentMemoryVersion] = []
    for scope_key in sorted(by_scope):
        candidates = by_scope[scope_key]
        winning_rank = max(rank for rank, _ in candidates)
        resolved.extend(
            sorted(
                (
                    item
                    for rank, item in candidates
                    if rank == winning_rank
                ),
                key=_result_key,
            )
        )
    return (*stable, *resolved)


def preference_scope_name(pref_scope: str) -> str:
    """Remove the free-text track prefix from one ownership scope."""

    if pref_scope == "freeform":
        return "person_default"
    if pref_scope.startswith("freeform."):
        return pref_scope.removeprefix("freeform.")
    if pref_scope == "global":
        return "person_stable"
    return pref_scope


def _matching_rank(
    item: IntentMemoryVersion,
    matching_names: frozenset[str],
) -> int | None:
    scope_name = preference_scope_name(item.pref_scope)
    if item.layer == "transient" or item.timescale == "situational":
        if scope_name == "person_situational":
            return 3
        return 3 if scope_name in matching_names else None
    if scope_name.startswith("role."):
        return 2 if scope_name in matching_names else None
    if scope_name in {"person_default", "global"} or (
        item.pref_scope == "global" and item.layer == "contextual"
    ):
        return 1
    return None


def _result_key(item: IntentMemoryVersion) -> tuple[float, str, str, int]:
    # Newer corroboration wins a same-rank tie before the existing deterministic
    # scope and structured/free-text fallbacks.
    corroborated_at = item.last_corroborated_at
    if corroborated_at.utcoffset() is None:
        corroborated_at = corroborated_at.replace(tzinfo=timezone.utc)
    return (
        -corroborated_at.timestamp(),
        item.scope_key,
        "1" if item.pref_scope.startswith("freeform") else "0",
        item.revision,
    )
