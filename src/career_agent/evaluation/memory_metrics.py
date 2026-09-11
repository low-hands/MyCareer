from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from career_agent.harness.observability import RunEvent


Comparability = Literal["COMPARABLE", "BEST_EFFORT", "NONCOMPARABLE"]


@dataclass(frozen=True)
class MetricResult:
    value: float | int | None
    measurable: bool
    comparability: Comparability
    reason: str | None = None


@dataclass(frozen=True)
class MemoryMetricsSummary:
    context_churn_rate: MetricResult
    context_churn_by_slot: dict[str, MetricResult]
    version_observation_count: int
    p1_complete_observation_count: int
    p1_complete_ratio: MetricResult
    version_context_observation_count: int
    p1_complete_context_observation_count: int
    version_context_entry_count: int
    zombie_exposure: MetricResult
    supersedence_exposure: MetricResult
    working_notes_influence: MetricResult


def summarize_memory_metrics(
    events: Sequence[RunEvent | Mapping[str, Any]],
) -> MemoryMetricsSummary:
    """Compute context diagnostics and version-bound exposure when complete."""

    previous_slots: dict[str, dict[str, str]] = {}
    changed_slots = {name: 0 for name in _CONTEXT_SLOT_NAMES}
    compared_slots = {name: 0 for name in _CONTEXT_SLOT_NAMES}
    version_context_entries: list[Mapping[str, Any]] = []
    version_context_observation_count = 0
    p1_complete_context_observation_count = 0
    tombstoned_update_ids: set[str] = set()
    post_tombstone_observation_count = 0
    zombie_observation_count = 0
    working_notes_guard_observation_count = 0
    working_notes_only_argument_count = 0

    for event in events:
        event_type = _event_type(event)
        details = _details(event)
        if event_type == "memory_tombstone_observed":
            for entry in details.get("entries", ()):
                if (
                    isinstance(entry, Mapping)
                    and entry.get("lifecycle_status") == "tombstoned"
                    and isinstance(entry.get("update_id"), str)
                ):
                    tombstoned_update_ids.add(str(entry["update_id"]))
        elif event_type == "memory_context_observed":
            blocked = details.get("working_notes_only_argument")
            if type(blocked) is int and blocked in {0, 1}:
                working_notes_guard_observation_count += 1
                working_notes_only_argument_count += blocked
            version_context_observation_count += 1
            if _is_complete_p1_observation(details):
                p1_complete_context_observation_count += 1
                entries = _version_entries(details)
                version_context_entries.extend(entries)
                if tombstoned_update_ids:
                    post_tombstone_observation_count += 1
                    zombie_observation_count += any(
                        entry.get("update_id") in tombstoned_update_ids
                        for entry in entries
                    )
            key = details.get("conversation_key")
            slots = details.get("slot_fingerprints")
            if not isinstance(key, str) or not isinstance(slots, Mapping):
                continue
            current = {
                str(name): str(digest)
                for name, digest in slots.items()
                if isinstance(name, str) and isinstance(digest, str)
            }
            previous = previous_slots.get(key)
            if previous is not None:
                for name in previous.keys() & current.keys():
                    if name not in compared_slots:
                        continue
                    compared_slots[name] += 1
                    changed_slots[name] += previous[name] != current[name]
            previous_slots[key] = current

    total_changed = sum(changed_slots.values())
    total_compared = sum(compared_slots.values())
    churn = (
        MetricResult(
            value=total_changed / total_compared,
            measurable=True,
            comparability="NONCOMPARABLE",
            reason=(
                "Diagnostic aggregate only, not a memory-change signal. It measures "
                "projected context-slot churn rather than bound memory versions, and "
                "career_profile churn is primarily driven by query-relevance reranking."
            ),
        )
        if total_compared
        else MetricResult(
            value=None,
            measurable=False,
            comparability="NONCOMPARABLE",
            reason="At least two context observations for one conversation are required.",
        )
    )
    churn_by_slot = {
        name: (
            MetricResult(
                value=changed_slots[name] / compared_slots[name],
                measurable=True,
                comparability="BEST_EFFORT",
                reason=(
                    "Projected career_memory churn is primarily driven by "
                    "query-relevance reranking, not by memory writes."
                    if name == "career_memory"
                    else (
                        "Low-churn identity block selected for the prompt-cache "
                        "prefix from measured slot stability."
                        if name == "career_identity"
                        else "Measures projected slot churn, not bound memory-version churn."
                    )
                ),
            )
            if compared_slots[name]
            else MetricResult(
                value=None,
                measurable=False,
                comparability="NONCOMPARABLE",
                reason=(
                    "At least two observations containing this slot are required."
                ),
            )
        )
        for name in _CONTEXT_SLOT_NAMES
    }
    version_observation_count = version_context_observation_count
    p1_complete_observation_count = p1_complete_context_observation_count
    p1_complete_ratio = _rate(
        p1_complete_observation_count,
        version_observation_count,
        comparability="BEST_EFFORT",
        empty_reason="No memory-context observations were supplied.",
        reason=(
            "Share of supplied memory-context observations "
            "with complete P1 version binding; incomplete observations stay "
            "visible in the denominator instead of invalidating unrelated "
            "observations."
        ),
    )
    if p1_complete_context_observation_count == 0:
        unavailable = MetricResult(
            value=None,
            measurable=False,
            comparability="NONCOMPARABLE",
            reason=(
                "Requires at least one complete P1 memory-context observation; "
                "see p1_complete_ratio for coverage."
            ),
        )
        supersedence = unavailable
    else:
        superseded_context = sum(
            entry.get("lifecycle_status") == "superseded"
            for entry in version_context_entries
        )
        supersedence = _rate(
            superseded_context,
            len(version_context_entries),
            comparability="BEST_EFFORT",
            empty_reason="No version-bound memory values were projected.",
            reason=(
                "P1 script-aware surface exposure over complete observations "
                "only; see p1_complete_ratio and the entry denominator."
            ),
        )
    zombie = (
        MetricResult(
            value=zombie_observation_count / post_tombstone_observation_count,
            measurable=True,
            comparability="BEST_EFFORT",
            reason=(
                "Share of complete P1 context observations after a durable "
                "M3 tombstone that still expose one of its update_ids."
            ),
        )
        if post_tombstone_observation_count
        else MetricResult(
            value=None,
            measurable=False,
            comparability="NONCOMPARABLE",
            reason=(
                "Requires a durable M3 tombstone followed by at least one "
                "complete P1 context observation."
            ),
        )
    )
    working_notes_influence = _rate(
        working_notes_only_argument_count,
        working_notes_guard_observation_count,
        comparability="BEST_EFFORT",
        empty_reason=(
            "No decisions with non-empty working notes and guard telemetry "
            "were supplied."
        ),
        reason=(
            "Share of decisions with non-empty working notes whose literal "
            "tool arguments were blocked by the best-effort notes guard; "
            "semantic paraphrases are outside this detector."
        ),
    )
    return MemoryMetricsSummary(
        context_churn_rate=churn,
        context_churn_by_slot=churn_by_slot,
        version_observation_count=version_observation_count,
        p1_complete_observation_count=p1_complete_observation_count,
        p1_complete_ratio=p1_complete_ratio,
        version_context_observation_count=version_context_observation_count,
        p1_complete_context_observation_count=(
            p1_complete_context_observation_count
        ),
        version_context_entry_count=len(version_context_entries),
        zombie_exposure=zombie,
        supersedence_exposure=supersedence,
        working_notes_influence=working_notes_influence,
    )


def _event_type(event: RunEvent | Mapping[str, Any]) -> str | None:
    value = event.event_type if isinstance(event, RunEvent) else event.get("event_type")
    return value if isinstance(value, str) else None


def _details(event: RunEvent | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(event, RunEvent):
        return event.details
    value = event.get("details")
    return value if isinstance(value, Mapping) else event


def _rate(
    numerator: int,
    denominator: int,
    *,
    comparability: Comparability,
    empty_reason: str,
    reason: str,
) -> MetricResult:
    if denominator == 0:
        return MetricResult(
            value=None,
            measurable=False,
            comparability="NONCOMPARABLE",
            reason=empty_reason,
        )
    return MetricResult(
        value=numerator / denominator,
        measurable=True,
        comparability=comparability,
        reason=reason,
    )


def _version_entries(
    details: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    entries = details.get("entries")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return ()
    return tuple(
        entry
        for entry in entries
        if isinstance(entry, Mapping)
        and isinstance(entry.get("entry_id"), str)
        and isinstance(entry.get("update_id"), str)
        and isinstance(entry.get("content_digest"), str)
        and entry.get("lifecycle_status") in {"current", "superseded"}
    )


def _version_entries_are_complete(details: Mapping[str, Any]) -> bool:
    entries = details.get("entries")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return False
    return len(_version_entries(details)) == len(entries)


def _is_complete_p1_observation(details: Mapping[str, Any]) -> bool:
    return (
        details.get("binding_profile") == "p1"
        and details.get("version_inventory_complete") is True
        and _version_entries_are_complete(details)
    )

_CONTEXT_SLOT_NAMES = (
    "career_identity",
    "career_memory",
    "task",
    "conversation_summary",
    "recent_messages",
)
