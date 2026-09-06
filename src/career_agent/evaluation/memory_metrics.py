from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import median
from typing import Any, Literal
import unicodedata

from career_agent.domain.memory_scope import ScopeResolutionQueueItem
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
    uptake_rate: MetricResult
    context_churn_rate: MetricResult
    staleness_exposure: MetricResult
    zombie_exposure: MetricResult
    supersedence_exposure: MetricResult


@dataclass(frozen=True)
class StateDriftProbeResult:
    classification: Literal["current", "superseded", "both", "no_use"]
    stale_surfaces: tuple[str, ...]
    current_surfaces: tuple[str, ...]

    @property
    def action_used_stale(self) -> bool:
        return self.classification in {"superseded", "both"}


@dataclass(frozen=True)
class PairedMemoryEvaluation:
    pair_count: int
    memory_on_success_rate: float
    memory_off_success_rate: float
    success_rate_delta: float
    comparability: Comparability = "COMPARABLE"


@dataclass(frozen=True)
class UnresolvedKeyBacklogSummary:
    open_count: int
    unresolved_count: int
    clarification_requested_count: int
    oldest_open_age_seconds: float | None
    resolved_count: int
    median_resolution_latency_seconds: float | None


def summarize_memory_metrics(
    events: Sequence[RunEvent | Mapping[str, Any]],
) -> MemoryMetricsSummary:
    """Compute only P2-safe metrics and fail closed on version-sensitive ones."""

    pending_writes: set[tuple[str, str]] = set()
    used_writes: set[tuple[str, str]] = set()
    previous_slots: dict[str, dict[str, str]] = {}
    changed_slots = 0
    compared_slots = 0

    for event in events:
        event_type = _event_type(event)
        details = _details(event)
        if event_type == "memory_write_observed":
            for identity in _entry_identities(details):
                pending_writes.add(identity)
        elif event_type == "memory_use_observed":
            for identity in _entry_identities(details):
                if identity in pending_writes:
                    used_writes.add(identity)
        elif event_type == "memory_context_observed":
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
                    compared_slots += 1
                    changed_slots += previous[name] != current[name]
            previous_slots[key] = current

    uptake = (
        MetricResult(
            value=len(used_writes) / len(pending_writes),
            measurable=True,
            comparability="BEST_EFFORT",
            reason="P2 binding has entry_id and digest but no update_id.",
        )
        if pending_writes
        else MetricResult(
            value=None,
            measurable=False,
            comparability="NONCOMPARABLE",
            reason="No observable memory writes were supplied.",
        )
    )
    churn = (
        MetricResult(
            value=changed_slots / compared_slots,
            measurable=True,
            comparability="BEST_EFFORT",
            reason="Measures projected context-slot churn, not bound memory-version churn.",
        )
        if compared_slots
        else MetricResult(
            value=None,
            measurable=False,
            comparability="NONCOMPARABLE",
            reason="At least two context observations for one conversation are required.",
        )
    )
    unavailable = MetricResult(
        value=None,
        measurable=False,
        comparability="NONCOMPARABLE",
        reason="Requires entry_id, update_id, and content_digest version binding.",
    )
    return MemoryMetricsSummary(
        uptake_rate=uptake,
        context_churn_rate=churn,
        staleness_exposure=unavailable,
        zombie_exposure=unavailable,
        supersedence_exposure=unavailable,
    )


def probe_state_drift(
    *,
    current_value: str,
    superseded_values: Sequence[str],
    surfaces: Mapping[str, str],
    action: str,
) -> StateDriftProbeResult:
    """Classify exact-surface state use independently from retrieval success."""

    current = _surface(current_value)
    stale = tuple(
        candidate
        for candidate in dict.fromkeys(_surface(value) for value in superseded_values)
        if candidate and candidate != current
    )
    normalized_surfaces = {
        name: _surface(value) for name, value in surfaces.items()
    }
    stale_surfaces = tuple(
        sorted(
            name
            for name, value in normalized_surfaces.items()
            if any(candidate in value for candidate in stale)
        )
    )
    current_surfaces = tuple(
        sorted(
            name for name, value in normalized_surfaces.items() if current in value
        )
    )
    normalized_action = _surface(action)
    used_current = bool(current) and current in normalized_action
    used_stale = any(candidate in normalized_action for candidate in stale)
    classification: Literal["current", "superseded", "both", "no_use"]
    if used_current and used_stale:
        classification = "both"
    elif used_stale:
        classification = "superseded"
    elif used_current:
        classification = "current"
    else:
        classification = "no_use"
    return StateDriftProbeResult(
        classification=classification,
        stale_surfaces=stale_surfaces,
        current_surfaces=current_surfaces,
    )


def compare_paired_runs(
    *,
    memory_on_successes: Sequence[bool],
    memory_off_successes: Sequence[bool],
) -> PairedMemoryEvaluation:
    if len(memory_on_successes) != len(memory_off_successes):
        raise ValueError("paired memory evaluations require equal-length arms")
    if not memory_on_successes:
        raise ValueError("paired memory evaluations require at least one pair")
    count = len(memory_on_successes)
    on_rate = sum(memory_on_successes) / count
    off_rate = sum(memory_off_successes) / count
    return PairedMemoryEvaluation(
        pair_count=count,
        memory_on_success_rate=on_rate,
        memory_off_success_rate=off_rate,
        success_rate_delta=on_rate - off_rate,
    )


def summarize_unresolved_key_backlog(
    items: Sequence[ScopeResolutionQueueItem],
    *,
    now: datetime | None = None,
) -> UnresolvedKeyBacklogSummary:
    current_time = now or datetime.now(timezone.utc)
    open_items = tuple(
        item
        for item in items
        if item.status in {"unresolved", "clarification_requested"}
    )
    ages = tuple(
        max(0.0, (current_time - item.created_at).total_seconds())
        for item in open_items
    )
    resolved = tuple(
        item
        for item in items
        if item.status == "resolved" and item.resolved_at is not None
    )
    latencies = tuple(
        max(0.0, (item.resolved_at - item.created_at).total_seconds())
        for item in resolved
        if item.resolved_at is not None
    )
    return UnresolvedKeyBacklogSummary(
        open_count=len(open_items),
        unresolved_count=sum(item.status == "unresolved" for item in open_items),
        clarification_requested_count=sum(
            item.status == "clarification_requested" for item in open_items
        ),
        oldest_open_age_seconds=max(ages) if ages else None,
        resolved_count=len(resolved),
        median_resolution_latency_seconds=(
            float(median(latencies)) if latencies else None
        ),
    )


def _event_type(event: RunEvent | Mapping[str, Any]) -> str | None:
    value = event.event_type if isinstance(event, RunEvent) else event.get("event_type")
    return value if isinstance(value, str) else None


def _details(event: RunEvent | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(event, RunEvent):
        return event.details
    value = event.get("details")
    return value if isinstance(value, Mapping) else event


def _entry_identities(details: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    entries = details.get("entries")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return ()
    identities = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        entry_id = entry.get("entry_id")
        digest = entry.get("content_digest")
        if isinstance(entry_id, str) and isinstance(digest, str):
            identities.append((entry_id, digest))
    return tuple(identities)


def _surface(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())
