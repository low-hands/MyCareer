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
    context_churn_by_slot: dict[str, MetricResult]
    version_observation_count: int
    p1_complete_observation_count: int
    p1_complete_ratio: MetricResult
    version_context_observation_count: int
    p1_complete_context_observation_count: int
    version_use_observation_count: int
    p1_complete_use_observation_count: int
    version_context_entry_count: int
    version_use_entry_count: int
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
class UnresolvedKeyBacklogSummary:
    open_count: int
    unresolved_count: int
    clarification_requested_count: int
    oldest_open_age_seconds: float | None
    resolved_count: int
    median_resolution_latency_seconds: float | None


@dataclass(frozen=True)
class MemoryBudgetCohort:
    records_input_unit_budget: int
    current_targets_input_unit_budget: int
    hard_constraints_input_unit_budget: int
    career_memory_enabled: bool
    turn_count: int
    truncated_turn_count: int
    budget_expansion_turn_count: int
    model_invisible_truncation_count: int
    technical_outcome_observed_count: int
    technical_failure_count: int
    fetch_turn_count: int
    fetch_required_turn_count: int
    fetch_hit_count: int
    fetch_miss_count: int
    truncation_rate: MetricResult
    technical_failure_rate: MetricResult
    fetch_hit_rate: MetricResult
    fetch_miss_rate: MetricResult


@dataclass(frozen=True)
class MemoryBudgetSummary:
    observed_run_count: int
    eligible_run_count: int
    missing_budget_run_count: int
    cohorts: tuple[MemoryBudgetCohort, ...]


def summarize_memory_metrics(
    events: Sequence[RunEvent | Mapping[str, Any]],
) -> MemoryMetricsSummary:
    """Compute P2 diagnostics and P1 version-bound exposure when complete."""

    pending_writes: set[tuple[str, str]] = set()
    used_writes: set[tuple[str, str]] = set()
    previous_slots: dict[str, dict[str, str]] = {}
    changed_slots = {name: 0 for name in _CONTEXT_SLOT_NAMES}
    compared_slots = {name: 0 for name in _CONTEXT_SLOT_NAMES}
    version_context_entries: list[Mapping[str, Any]] = []
    version_use_entries: list[Mapping[str, Any]] = []
    version_context_observation_count = 0
    p1_complete_context_observation_count = 0
    version_use_observation_count = 0
    p1_complete_use_observation_count = 0

    for event in events:
        event_type = _event_type(event)
        details = _details(event)
        if event_type == "memory_write_observed":
            for identity in _entry_identities(details):
                pending_writes.add(identity)
        elif event_type == "memory_use_observed":
            version_use_observation_count += 1
            for identity in _entry_identities(details):
                if identity in pending_writes:
                    used_writes.add(identity)
            if _is_complete_p1_observation(details):
                p1_complete_use_observation_count += 1
                version_use_entries.extend(_version_entries(details))
        elif event_type == "memory_context_observed":
            version_context_observation_count += 1
            if _is_complete_p1_observation(details):
                p1_complete_context_observation_count += 1
                version_context_entries.extend(_version_entries(details))
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
    version_observation_count = (
        version_context_observation_count + version_use_observation_count
    )
    p1_complete_observation_count = (
        p1_complete_context_observation_count
        + p1_complete_use_observation_count
    )
    p1_complete_ratio = _rate(
        p1_complete_observation_count,
        version_observation_count,
        comparability="BEST_EFFORT",
        empty_reason="No memory-context or memory-use observations were supplied.",
        reason=(
            "Share of supplied memory-context and memory-use observations "
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
        staleness = unavailable
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
        stale_use = sum(
            entry.get("lifecycle_status") in {"superseded", "rolled_back"}
            for entry in version_use_entries
        )
        staleness = _rate(
            stale_use,
            len(version_use_entries),
            comparability="BEST_EFFORT",
            empty_reason="No version-bound memory use was observed.",
            reason=(
                "P1 script-aware use of superseded or rolled-back versions "
                "from complete observations only."
            ),
        )
    zombie = MetricResult(
        value=None,
        measurable=False,
        comparability="NONCOMPARABLE",
        reason=(
            "M3 tombstones are not implemented; rolled-back corrections are "
            "not deletion and must not be reported as zombie exposure."
        ),
    )
    return MemoryMetricsSummary(
        uptake_rate=uptake,
        context_churn_rate=churn,
        context_churn_by_slot=churn_by_slot,
        version_observation_count=version_observation_count,
        p1_complete_observation_count=p1_complete_observation_count,
        p1_complete_ratio=p1_complete_ratio,
        version_context_observation_count=version_context_observation_count,
        p1_complete_context_observation_count=(
            p1_complete_context_observation_count
        ),
        version_use_observation_count=version_use_observation_count,
        p1_complete_use_observation_count=p1_complete_use_observation_count,
        version_context_entry_count=len(version_context_entries),
        version_use_entry_count=len(version_use_entries),
        staleness_exposure=staleness,
        zombie_exposure=zombie,
        supersedence_exposure=supersedence,
    )


def summarize_memory_budget_metrics(
    events: Sequence[RunEvent | Mapping[str, Any]],
) -> MemoryBudgetSummary:
    """Describe bounded delivery and layered-fetch behavior per production turn.

    These cohorts are operational telemetry, not a budget estimator. Budget
    changes are validated with paired cassette replay, never inferred from this
    observational queue.
    """

    runs: dict[str, list[RunEvent | Mapping[str, Any]]] = {}
    for event in events:
        run_id = _run_id(event)
        if run_id is not None:
            runs.setdefault(run_id, []).append(event)

    cohort_runs: dict[tuple[int, int, int, bool], list[dict[str, Any]]] = {}
    missing_budget = 0
    for run_events in runs.values():
        indexed_contexts = [
            (index, _details(event))
            for index, event in enumerate(run_events)
            if _event_type(event) == "memory_context_observed"
        ]
        contexts = [details for _, details in indexed_contexts]
        configurations = {
            configuration
            for details in contexts
            if (
                configuration := _budget_configuration(details)
            )
            is not None
        }
        if not contexts or len(configurations) != 1:
            missing_budget += 1
            continue
        configuration = next(iter(configurations))
        truncated = any(_memory_truncated(details) for details in contexts)
        invisible = any(
            _memory_truncated(details)
            and not _memory_truncation_visible(details)
            for details in contexts
        )
        budget_expanded = any(
            isinstance(
                details.get("career_profile_truncation"),
                Mapping,
            )
            and details["career_profile_truncation"].get(
                "any_budget_expanded"
            )
            is True
            for details in contexts
        )
        visible_overflow = [
            (index, _memory_required_fetch_tools(details))
            for index, details in indexed_contexts
            if _memory_required_fetch_tools(details)
        ]
        first_visible_overflow = (
            min(index for index, _ in visible_overflow)
            if visible_overflow
            else None
        )
        required_fetch_tools = set().union(
            *(tools for _, tools in visible_overflow)
        )
        model_events = [
            (index, _details(event))
            for index, event in enumerate(run_events)
            if _event_type(event) == "model_succeeded"
            and getattr(event, "model_call_category", None)
            in {None, "orchestrator_decision"}
        ]
        failed = any(_event_type(event) == "turn_failed" for event in run_events)
        completed = any(
            _event_type(event) == "turn_completed" for event in run_events
        )
        fetches = [
            (index, str(details.get("tool_name")))
            for index, details in model_events
            if details.get("tool_name")
            in {
                "search_career_memory",
                "list_target_roles",
            }
        ]
        fetched = bool(fetches)
        fetch_required = first_visible_overflow is not None
        fetched_after_overflow = {
            tool_name
            for index, tool_name in fetches
            if (
                first_visible_overflow is not None
                and index > first_visible_overflow
            )
        }
        fetch_hit = (
            fetch_required
            and required_fetch_tools.issubset(fetched_after_overflow)
        )
        cohort_runs.setdefault(configuration, []).append(
            {
                "truncated": truncated,
                    "budget_expanded": budget_expanded,
                "invisible": invisible,
                "technical_outcome_observed": failed or completed,
                "failed": failed,
                "fetched": fetched,
                "fetch_required": fetch_required,
                "fetch_hit": fetch_hit,
            }
        )

    cohorts = tuple(
        _memory_budget_cohort(configuration, samples)
        for configuration, samples in sorted(cohort_runs.items())
    )
    return MemoryBudgetSummary(
        observed_run_count=len(runs),
        eligible_run_count=sum(cohort.turn_count for cohort in cohorts),
        missing_budget_run_count=missing_budget,
        cohorts=cohorts,
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


def _run_id(event: RunEvent | Mapping[str, Any]) -> str | None:
    value = event.run_id if isinstance(event, RunEvent) else event.get("run_id")
    return value if isinstance(value, str) and value else None


def _budget_configuration(
    details: Mapping[str, Any],
) -> tuple[int, int, int, bool] | None:
    budgets = details.get("career_profile_budgets")
    if not isinstance(budgets, Mapping):
        return None
    values = tuple(
        budgets.get(name)
        for name in (
            "records_input_units",
            "current_targets_input_units",
            "hard_constraints_input_units",
        )
    )
    enabled = details.get("career_memory_enabled")
    if (
        any(type(value) is not int or value < 0 for value in values)
        or budgets.get("budget_unit") != "estimated_input_tokens"
        or type(enabled) is not bool
    ):
        return None
    return values[0], values[1], values[2], enabled


def _memory_truncated(details: Mapping[str, Any]) -> bool:
    delivery = details.get("career_profile_delivery")
    if not isinstance(delivery, Mapping):
        return False
    return any(
        type(delivery.get(key)) is int and delivery[key] > 0
        for key in (
            "records_dropped",
            "claims_dropped",
            "current_targets_dropped",
            "hard_constraints_dropped",
        )
    )


def _memory_truncation_visible(details: Mapping[str, Any]) -> bool:
    truncation = details.get("career_profile_truncation")
    if not isinstance(truncation, Mapping):
        return False
    return truncation.get("all_truncation_model_visible") is True


def _memory_required_fetch_tools(
    details: Mapping[str, Any],
) -> frozenset[str]:
    truncation = details.get("career_profile_truncation")
    if not isinstance(truncation, Mapping):
        return frozenset()
    tools = truncation.get("required_fetch_tools")
    if isinstance(tools, Sequence) and not isinstance(tools, (str, bytes)):
        return frozenset(
            str(tool)
            for tool in tools
            if tool in {"search_career_memory", "list_target_roles"}
        )
    inferred = set()
    records = truncation.get("records")
    targets = truncation.get("current_targets")
    if isinstance(records, Mapping) and records.get("fetch_required") is True:
        inferred.add("search_career_memory")
    if isinstance(targets, Mapping) and targets.get("fetch_required") is True:
        inferred.add("list_target_roles")
    return frozenset(inferred)


def _memory_budget_cohort(
    configuration: tuple[int, int, int, bool],
    samples: Sequence[Mapping[str, Any]],
) -> MemoryBudgetCohort:
    records_budget, targets_budget, constraints_budget, memory_enabled = (
        configuration
    )
    turn_count = len(samples)
    truncated = sum(item["truncated"] is True for item in samples)
    expanded = sum(item["budget_expanded"] is True for item in samples)
    invisible = sum(item["invisible"] is True for item in samples)
    technical_outcomes = [
        item for item in samples if item["technical_outcome_observed"] is True
    ]
    failures = sum(item["failed"] is True for item in technical_outcomes)
    fetched = sum(item["fetched"] is True for item in samples)
    fetch_required = sum(item["fetch_required"] is True for item in samples)
    fetch_hits = sum(item["fetch_hit"] is True for item in samples)
    fetch_misses = fetch_required - fetch_hits
    return MemoryBudgetCohort(
        records_input_unit_budget=records_budget,
        current_targets_input_unit_budget=targets_budget,
        hard_constraints_input_unit_budget=constraints_budget,
        career_memory_enabled=memory_enabled,
        turn_count=turn_count,
        truncated_turn_count=truncated,
        budget_expansion_turn_count=expanded,
        model_invisible_truncation_count=invisible,
        technical_outcome_observed_count=len(technical_outcomes),
        technical_failure_count=failures,
        fetch_turn_count=fetched,
        fetch_required_turn_count=fetch_required,
        fetch_hit_count=fetch_hits,
        fetch_miss_count=fetch_misses,
        truncation_rate=MetricResult(
            value=truncated / turn_count,
            measurable=True,
            comparability="COMPARABLE",
            reason="One deterministic bounded-delivery observation per production turn.",
        ),
        technical_failure_rate=_rate(
            failures,
            len(technical_outcomes),
            comparability="COMPARABLE",
            empty_reason="No completed or failed turns were observed.",
            reason="Counts durable turn_failed outcomes only.",
        ),
        fetch_hit_rate=_rate(
            fetch_hits,
            fetch_required,
            comparability="COMPARABLE",
            empty_reason="No model-visible overflow turns were observed.",
            reason=(
                "Share of overflow-bearing turns that used a layered memory "
                "fetch path; retrieval is successful fallback, not damage."
            ),
        ),
        fetch_miss_rate=_rate(
            fetch_misses,
            fetch_required,
            comparability="COMPARABLE",
            empty_reason="No model-visible overflow turns were observed.",
            reason=(
                "Share of overflow-bearing turns that did not use the required "
                "layered fetch path."
            ),
        ),
    )


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
        and entry.get("lifecycle_status")
        in {"current", "superseded", "rolled_back"}
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


def _surface(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


_CONTEXT_SLOT_NAMES = (
    "career_identity",
    "career_memory",
    "task",
    "conversation_summary",
    "recent_messages",
)
