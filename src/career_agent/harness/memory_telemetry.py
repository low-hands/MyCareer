from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import unicodedata
from typing import Any

from career_agent.agent.decision_messages import decision_context_chars
from career_agent.harness.observability import conversation_trace_key


def content_digest(value: Any) -> str:
    if isinstance(value, str):
        value = " ".join(unicodedata.normalize("NFKC", value).split())
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def memory_context_observation(
    context: Any, *, career_memory_enabled: bool
) -> dict[str, Any]:
    """Return P2-safe fingerprints without persisting projected user text."""

    projection = context.model_context()
    slot_values = {
        name: projection.get(name)
        for name in _CONTEXT_SLOT_NAMES
    }
    slots = {
        name: content_digest(value) for name, value in slot_values.items()
    }
    slot_chars = {
        name: _serialized_chars(value) for name, value in slot_values.items()
    }
    dynamic_context_chars = decision_context_chars(context)
    career_profile_chars = slot_chars["career_profile"]
    entries: list[dict[str, str]] = []
    default_city = context.profile.default_city
    if default_city:
        entries.append(
            {
                "entry_id": "person_intent/self/default_city",
                "content_digest": content_digest(default_city),
            }
        )
    observation = {
        "conversation_id": context.conversation_id,
        "conversation_key": conversation_trace_key(
            context.profile.user_id, context.conversation_id
        ),
        "career_memory_enabled": career_memory_enabled,
        "binding_profile": "p2",
        "slot_fingerprints": slots,
        "slot_chars": slot_chars,
        "dynamic_context_chars": dynamic_context_chars,
        "career_profile_dynamic_ratio": (
            career_profile_chars / dynamic_context_chars
            if dynamic_context_chars
            else 0.0
        ),
        "career_profile_chars": _json_char_composition(slot_values["career_profile"]),
        "career_profile_source_chars": _career_profile_source_chars(
            slot_values["career_profile"]
        ),
        "career_profile_schema_chars": _onto_schema_chars(
            slot_values["career_profile"]
        ),
        "career_profile_delivery": _career_profile_delivery(
            context,
            slot_values["career_profile"],
        ),
        "entries": entries,
    }
    return observation


def memory_use_observation(context: Any, decision: Any) -> dict[str, Any] | None:
    """Detect exact use of currently typed values; deliberately BEST_EFFORT."""

    values: tuple[tuple[str, str], ...] = tuple(
        (entry_id, value)
        for entry_id, value in (
            ("person_intent/self/default_city", context.profile.default_city),
        )
        if isinstance(value, str) and value.strip()
    )
    if not values:
        return None
    if hasattr(decision, "model_dump"):
        payload = decision.model_dump(mode="json")
    elif isinstance(decision, Mapping):
        payload = dict(decision)
    else:
        payload = str(decision)
    rendered = _surface(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    )
    used = [
        {"entry_id": entry_id, "content_digest": content_digest(value)}
        for entry_id, value in values
        if _surface(value) in rendered
    ]
    if not used:
        return None
    return {
        "conversation_id": context.conversation_id,
        "conversation_key": conversation_trace_key(
            context.profile.user_id, context.conversation_id
        ),
        "binding_profile": "p2",
        "detection": "exact_surface",
        "entries": used,
    }


def _surface(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


_CONTEXT_SLOT_NAMES = (
    "career_profile",
    "task",
    "conversation_summary",
    "recent_messages",
)


def _serialized_chars(value: Any) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    )


def _json_char_composition(value: Any) -> dict[str, int]:
    """Split exact serialized size into keys, scalar values, and punctuation.

    ONTO ``fields`` labels are counted as values by the walker (they live in a
    list). Move that slice into ``schema_chars`` so a keys-ratio drop is not
    mistaken for a saving. The four reported buckets still sum to the exact
    serialized size; ``career_profile_schema_chars`` separately reports both
    full field-array cost and label-only cost.
    """

    total = _serialized_chars(value)
    keys, values = _key_and_value_chars(value)
    schema = _onto_schema_chars(value)
    return {
        "keys": keys,
        "values": values - schema["label_chars"],
        "punctuation": total - keys - values,
        "schema_chars": schema["label_chars"],
    }


def _career_profile_source_chars(value: Any) -> dict[str, int]:
    """Assign every top-level character to a source or shared JSON framing."""

    if not isinstance(value, Mapping):
        return {
            "records": 0,
            "current_targets": 0,
            "hard_constraints": 0,
            "shared": _serialized_chars(value),
        }
    groups = {
        "records": {
            "records",
            "claims",
            "records_returned",
            "records_total",
            "claims_returned",
            "claims_total",
        },
        "current_targets": {
            "current_targets",
            "current_targets_returned",
            "current_targets_total",
        },
        "hard_constraints": {
            "hard_constraints",
            "hard_constraints_returned",
            "hard_constraints_total",
        },
    }
    totals = {name: 0 for name in groups}
    assigned = 0
    for key, item in value.items():
        # Strip the braces from a one-entry object. This owns the key, colon,
        # spaces, and value while commas and the outer braces remain shared.
        chars = _serialized_chars({key: item}) - 2
        owner = next(
            (name for name, keys in groups.items() if key in keys),
            None,
        )
        if owner is not None:
            totals[owner] += chars
            assigned += chars
    totals["shared"] = _serialized_chars(value) - assigned
    return totals


def _career_profile_delivery(context: Any, value: Any) -> dict[str, int]:
    """Report bounded-delivery counts without persisting projected user text."""

    profile = value if isinstance(value, Mapping) else {}
    records = profile.get("records")
    claims = profile.get("claims")
    record_rows = (
        records.get("rows", ())
        if isinstance(records, Mapping)
        else ()
    )
    claim_rows = (
        claims.get("rows", ())
        if isinstance(claims, Mapping)
        else ()
    )
    targets = profile.get("current_targets")
    target_rows = (
        targets.get("roles", ())
        if isinstance(targets, Mapping)
        else ()
    )
    constraints = profile.get("hard_constraints", ())
    if not isinstance(constraints, list):
        constraints = ()

    records_returned = len(record_rows)
    claims_returned = len(claim_rows)
    targets_returned = len(target_rows)
    constraints_returned = len(constraints)
    records_total = int(context.career_memory.records_total)
    claims_total = int(context.career_memory.claims_total)
    targets_total = int(context.profile.current_targets_total)
    constraints_total = len(context.profile.hard_constraints)
    return {
        "records_returned": records_returned,
        "records_total": records_total,
        "records_dropped": max(0, records_total - records_returned),
        "claims_returned": claims_returned,
        "claims_total": claims_total,
        "claims_dropped": max(0, claims_total - claims_returned),
        "current_targets_returned": targets_returned,
        "current_targets_total": targets_total,
        "current_targets_dropped": max(0, targets_total - targets_returned),
        "hard_constraints_returned": constraints_returned,
        "hard_constraints_total": constraints_total,
        "hard_constraints_dropped": max(
            0,
            constraints_total - constraints_returned,
        ),
    }


def _onto_schema_chars(value: Any) -> dict[str, int]:
    """Measure ONTO ``fields`` arrays without treating them as payload values."""

    if _is_onto_table(value):
        fields = value["fields"]
        schema_chars = _serialized_chars(fields)
        label_chars = sum(_serialized_chars(name) for name in fields)
        nested = [
            _onto_schema_chars(item)
            for key, item in value.items()
            if key != "fields"
        ]
        return {
            "schema_chars": schema_chars + sum(item["schema_chars"] for item in nested),
            "label_chars": label_chars + sum(item["label_chars"] for item in nested),
        }
    if isinstance(value, Mapping):
        parts = [_onto_schema_chars(item) for item in value.values()]
        return {
            "schema_chars": sum(item["schema_chars"] for item in parts),
            "label_chars": sum(item["label_chars"] for item in parts),
        }
    if isinstance(value, (list, tuple)):
        parts = [_onto_schema_chars(item) for item in value]
        return {
            "schema_chars": sum(item["schema_chars"] for item in parts),
            "label_chars": sum(item["label_chars"] for item in parts),
        }
    return {"schema_chars": 0, "label_chars": 0}


def _is_onto_table(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and isinstance(value.get("fields"), list)
        and isinstance(value.get("rows"), list)
        and all(isinstance(name, str) for name in value["fields"])
    )


def _key_and_value_chars(value: Any) -> tuple[int, int]:
    if isinstance(value, Mapping):
        key_chars = 0
        value_chars = 0
        for key, item in value.items():
            key_chars += len(json.dumps(str(key), ensure_ascii=False))
            nested_keys, nested_values = _key_and_value_chars(item)
            key_chars += nested_keys
            value_chars += nested_values
        return key_chars, value_chars
    if isinstance(value, (list, tuple)):
        key_chars = 0
        value_chars = 0
        for item in value:
            nested_keys, nested_values = _key_and_value_chars(item)
            key_chars += nested_keys
            value_chars += nested_values
        return key_chars, value_chars
    return 0, _serialized_chars(value)
