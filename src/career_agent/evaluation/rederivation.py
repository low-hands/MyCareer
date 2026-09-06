from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from career_agent.harness.observability import RunEvent


@dataclass(frozen=True)
class ReDerivationSummary:
    compaction_count: int
    tool_call_count: int
    post_compaction_tool_call_count: int
    rederivation_count: int
    measurable: bool


def tool_call_fingerprint(tool_name: str, arguments: Mapping[str, Any]) -> str:
    """Return the privacy-safe identity persisted for a model tool call."""

    canonical = json.dumps(
        {"name": tool_name, "arguments": arguments},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def count_rederivations(
    events: Sequence[RunEvent | Mapping[str, Any]],
) -> int:
    """Count repeated tool calls after at least one compaction point.

    Events must be supplied in execution order. Production traces expose calls
    as ``model_succeeded`` events with a name and argument digest; tests and
    offline evaluators may instead supply explicit ``tool_call`` events with
    their argument mapping.
    """

    return summarize_rederivations(events).rederivation_count


def summarize_rederivations(
    events: Sequence[RunEvent | Mapping[str, Any]],
) -> ReDerivationSummary:
    """Describe whether a cross-turn trace can measure repeated work."""

    seen_calls: set[tuple[str, str]] = set()
    calls_before_compaction: set[tuple[str, str]] = set()
    compaction_count = 0
    tool_call_count = 0
    post_compaction_tool_call_count = 0
    rederivation_count = 0
    has_compacted = False
    for event in events:
        if _event_type(event) == "context_compacted":
            compaction_count += 1
            has_compacted = True
            calls_before_compaction.update(seen_calls)
            continue
        identity = _tool_call_identity(event)
        if identity is None:
            continue
        tool_call_count += 1
        if has_compacted:
            post_compaction_tool_call_count += 1
        if identity in calls_before_compaction:
            rederivation_count += 1
        seen_calls.add(identity)
    return ReDerivationSummary(
        compaction_count=compaction_count,
        tool_call_count=tool_call_count,
        post_compaction_tool_call_count=post_compaction_tool_call_count,
        rederivation_count=rederivation_count,
        measurable=bool(calls_before_compaction)
        and post_compaction_tool_call_count > 0,
    )


def _event_type(event: RunEvent | Mapping[str, Any]) -> str | None:
    value = (
        event.event_type
        if isinstance(event, RunEvent)
        else event.get("event_type")
    )
    return value if isinstance(value, str) else None


def _event_details(
    event: RunEvent | Mapping[str, Any],
) -> Mapping[str, Any]:
    if isinstance(event, RunEvent):
        return event.details
    details = event.get("details")
    return details if isinstance(details, Mapping) else event


def _tool_call_identity(
    event: RunEvent | Mapping[str, Any],
) -> tuple[str, str] | None:
    event_type = _event_type(event)
    if event_type not in {"model_succeeded", "tool_call"}:
        return None
    details = _event_details(event)
    tool_name = details.get("tool_name")
    if not isinstance(tool_name, str) or not tool_name:
        return None
    fingerprint = details.get("tool_arguments_fingerprint")
    if isinstance(fingerprint, str) and fingerprint:
        return tool_name, fingerprint
    arguments = details.get("arguments")
    if not isinstance(arguments, Mapping):
        return None
    return tool_name, tool_call_fingerprint(tool_name, arguments)
