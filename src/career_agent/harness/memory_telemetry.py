from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import re
import unicodedata
from typing import Any

from career_agent.agent.decision_messages import (
    context_churn_slot_values,
    decision_context_chars,
)
from career_agent.harness.observability import conversation_trace_key


_MIN_CJK_SUBSTRING_SURFACE_CHARS = 4


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
    career_profile = projection.get("career_profile")
    slot_values = context_churn_slot_values(projection)
    slots = {
        name: content_digest(value) for name, value in slot_values.items()
    }
    slot_chars = {
        name: _serialized_chars(value) for name, value in slot_values.items()
    }
    slot_chars["career_profile"] = _serialized_chars(career_profile)
    dynamic_context_chars = decision_context_chars(context)
    career_profile_chars = slot_chars["career_profile"]
    delivery = _career_profile_delivery(
        context,
        career_profile,
    )
    bindings, inventory_complete = _telemetry_bindings(context)
    entries = _bound_context_entries(bindings, slot_values)
    default_city = context.profile.default_city
    if default_city and not any(
        entry["entry_id"] == "person_intent/self/default_city"
        for entry in entries
    ):
        entries.append(
            {
                "entry_id": "person_intent/self/default_city",
                "content_digest": content_digest(default_city),
            }
        )
    observation = {
        "conversation_key": conversation_trace_key(
            context.profile.user_id, context.conversation_id
        ),
        "career_memory_enabled": career_memory_enabled,
        "binding_profile": "p1" if inventory_complete else "p2",
        "version_inventory_complete": inventory_complete,
        "slot_fingerprints": slots,
        "slot_chars": slot_chars,
        "dynamic_context_chars": dynamic_context_chars,
        "career_profile_dynamic_ratio": (
            career_profile_chars / dynamic_context_chars
            if dynamic_context_chars
            else 0.0
        ),
        "career_profile_chars": _json_char_composition(career_profile),
        "career_profile_source_chars": _career_profile_source_chars(
            career_profile
        ),
        "career_profile_schema_chars": _onto_schema_chars(
            career_profile
        ),
        "career_profile_budgets": context.career_profile_budgets.model_dump(
            mode="json"
        ),
        "career_profile_delivery": delivery,
        "career_profile_truncation": _career_profile_truncation(
            career_profile,
            delivery,
        ),
        "entries": entries,
    }
    return observation


def memory_use_observation(context: Any, decision: Any) -> dict[str, Any] | None:
    """Detect exact use of currently typed values; deliberately BEST_EFFORT."""

    bindings, inventory_complete = _telemetry_bindings(context)
    if hasattr(decision, "model_dump"):
        payload = decision.model_dump(mode="json")
    elif isinstance(decision, Mapping):
        payload = dict(decision)
    else:
        payload = str(decision)
    rendered = _surface(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    )
    projection = context.model_context()
    slot_values = context_churn_slot_values(projection)
    visible_update_ids = {
        entry["update_id"]
        for entry in _bound_context_entries(
            bindings,
            slot_values,
        )
    }
    used = [
        _binding_entry(binding)
        for binding in bindings
        if binding.update_id in visible_update_ids
        and _surface_matches(rendered, _surface(binding.value))
    ]
    default_city = context.profile.default_city
    if (
        default_city
        and _surface_matches(rendered, _surface(default_city))
        and not any(
            entry["entry_id"] == "person_intent/self/default_city"
            for entry in used
        )
    ):
        used.append(
            {
                "entry_id": "person_intent/self/default_city",
                "content_digest": content_digest(default_city),
            }
        )
    if not used:
        return None
    return {
        "conversation_key": conversation_trace_key(
            context.profile.user_id, context.conversation_id
        ),
        "binding_profile": "p1" if inventory_complete else "p2",
        "version_inventory_complete": inventory_complete,
        "detection": "exact_surface",
        "entries": used,
    }


def _surface(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _surface_matches(rendered: str, value: str) -> bool:
    """Match normalized values without treating compounds as exact use."""

    if not value:
        return False
    if any(character.isascii() and character.isalnum() for character in value):
        return (
            re.search(
                rf"(?<![0-9a-z_]){re.escape(value)}(?![0-9a-z_])",
                rendered,
            )
            is not None
        )
    cjk_characters = tuple(character for character in value if _is_cjk(character))
    if cjk_characters and len(cjk_characters) == len(value.replace(" ", "")):
        if len(cjk_characters) >= _MIN_CJK_SUBSTRING_SURFACE_CHARS:
            return value in rendered
        return any(
            not _is_cjk(rendered[index - 1] if index else "")
            and not _is_cjk(
                rendered[index + len(value)]
                if index + len(value) < len(rendered)
                else ""
            )
            for index in _surface_occurrences(rendered, value)
        )
    return value in rendered


def _surface_occurrences(rendered: str, value: str) -> tuple[int, ...]:
    starts = []
    offset = 0
    while (index := rendered.find(value, offset)) >= 0:
        starts.append(index)
        offset = index + 1
    return tuple(starts)


def _is_cjk(character: str) -> bool:
    return bool(character) and (
        "\u3400" <= character <= "\u4dbf"
        or "\u4e00" <= character <= "\u9fff"
        or "\uf900" <= character <= "\ufaff"
    )


def _telemetry_bindings(context: Any) -> tuple[tuple[Any, ...], bool]:
    profile_bindings = tuple(
        getattr(context.profile, "telemetry_bindings", ())
    )
    career_bindings = tuple(
        getattr(context.career_memory, "telemetry_bindings", ())
    )
    selected: dict[tuple[str, str], Any] = {}
    update_ids_by_surface: dict[tuple[str, str], set[str]] = {}
    for binding in (*profile_bindings, *career_bindings):
        identity = (binding.entry_id, binding.content_digest)
        update_ids_by_surface.setdefault(identity, set()).add(binding.update_id)
        previous = selected.get(identity)
        if (
            previous is None
            or binding.lifecycle_status == "current"
            or (
                previous.lifecycle_status != "current"
                and binding.revision > previous.revision
            )
        ):
            selected[identity] = binding
    profile_has_versioned_values = bool(
        context.profile.default_city
        or context.profile.hard_constraints
        or any(
            value is not None
            for target in context.profile.current_targets
            for value in (
                target.city,
                target.salary_expectation,
                target.experience,
                target.education,
            )
        )
    )
    profile_complete = bool(
        getattr(context.profile, "telemetry_inventory_complete", False)
    ) or not profile_has_versioned_values
    career_complete = bool(
        getattr(context.career_memory, "telemetry_inventory_complete", False)
    ) or context.career_memory.claims_total == 0
    entries_by_value: dict[str, set[str]] = {}
    for binding in (*profile_bindings, *career_bindings):
        value = _surface(binding.value)
        entries_by_value.setdefault(value, set()).add(binding.entry_id)
    ambiguous_lineage_surface = any(
        len(update_ids) > 1
        for update_ids in update_ids_by_surface.values()
    )
    overlapping_surface = any(
        left != right and (left in right or right in left)
        for left in entries_by_value
        for right in entries_by_value
    )
    cross_scope_surface = any(
        len(entry_ids) > 1 for entry_ids in entries_by_value.values()
    )
    low_entropy_surface = any(
        len(value) < 2 for value in entries_by_value
    )
    return (
        tuple(
            sorted(
                selected.values(),
                key=lambda item: (item.entry_id, item.revision, item.update_id),
            )
        ),
        profile_complete
        and career_complete
        and not ambiguous_lineage_surface
        and not overlapping_surface
        and not cross_scope_surface
        and not low_entropy_surface,
    )


def _bound_context_entries(
    bindings: tuple[Any, ...],
    slot_values: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rendered_slots = {
        name: _surface(
            json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        )
        for name, value in slot_values.items()
    }
    entries = []
    for binding in bindings:
        value = _surface(binding.value)
        surfaces = sorted(
            name
            for name, rendered in rendered_slots.items()
            if _surface_matches(rendered, value)
        )
        if surfaces:
            entries.append(
                {
                    **_binding_entry(binding),
                    "surfaces": surfaces,
                }
            )
    return entries


def _binding_entry(binding: Any) -> dict[str, Any]:
    return {
        "entry_id": binding.entry_id,
        "update_id": binding.update_id,
        "content_digest": binding.content_digest,
        "revision": binding.revision,
        "lifecycle_status": binding.lifecycle_status,
    }


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
            "hard_constraints_budget_expanded",
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


def _career_profile_truncation(
    value: Any,
    delivery: Mapping[str, int],
) -> dict[str, Any]:
    """Distinguish model-visible truncation from an operational dropped count."""

    profile = value if isinstance(value, Mapping) else {}
    targets = profile.get("current_targets")
    target_body = targets if isinstance(targets, Mapping) else {}
    overflow = profile.get("memory_overflow")
    overflow_body = overflow if isinstance(overflow, Mapping) else {}
    overflow_sections = overflow_body.get("sections")
    fetch_tools = (
        {
            str(item.get("section")): str(item.get("fetch_tool"))
            for item in overflow_sections
            if isinstance(item, Mapping)
            and isinstance(item.get("section"), str)
            and isinstance(item.get("fetch_tool"), str)
        }
        if isinstance(overflow_sections, Sequence)
        else {}
    )
    sections = {
        "records": (
            delivery["records_dropped"] > 0 or delivery["claims_dropped"] > 0,
            all(
                key in profile
                for key in (
                    "records_returned",
                    "records_total",
                    "claims_returned",
                    "claims_total",
                )
            ),
            delivery["records_total"] > 0,
            delivery["records_returned"] == 0,
            fetch_tools.get("career_memory"),
        ),
        "current_targets": (
            delivery["current_targets_dropped"] > 0,
            (
                all(
                    key in target_body
                    for key in ("roles_returned", "roles_total")
                )
                or all(
                    key in profile
                    for key in (
                        "current_targets_returned",
                        "current_targets_total",
                    )
                )
            ),
            delivery["current_targets_total"] > 0,
            delivery["current_targets_returned"] == 0,
            fetch_tools.get("current_targets"),
        ),
        "hard_constraints": (
            delivery["hard_constraints_dropped"] > 0,
            all(
                key in profile
                for key in (
                    "hard_constraints_returned",
                    "hard_constraints_total",
                )
            ),
            delivery["hard_constraints_total"] > 0,
            delivery["hard_constraints_returned"] == 0,
            None,
        ),
    }
    result: dict[str, Any] = {}
    for name, (
        truncated,
        counters_present,
        has_stored,
        returned_none,
        fetch_tool,
    ) in (
        sections.items()
    ):
        visible = not truncated or counters_present
        result[name] = {
            "truncated": truncated,
            "model_visible": visible,
            "indistinguishable_from_empty": (
                truncated and has_stored and returned_none and not counters_present
            ),
            "fetch_required": truncated and fetch_tool is not None,
            **(
                {"fetch_tool": fetch_tool}
                if truncated and fetch_tool is not None
                else {}
            ),
        }
        if name == "hard_constraints":
            result[name]["budget_expanded"] = (
                profile.get("hard_constraints_budget_expanded") is True
            )
    any_truncated = any(
        section["truncated"] for section in result.values()
    )
    all_visible = all(
        section["model_visible"] for section in result.values()
    )
    any_fetch_required = any(
        section["fetch_required"] for section in result.values()
    )
    any_budget_expanded = any(
        bool(section.get("budget_expanded"))
        for section in result.values()
    )
    result["any_truncated"] = any_truncated
    result["any_fetch_required"] = any_fetch_required
    result["any_budget_expanded"] = any_budget_expanded
    result["required_fetch_tools"] = sorted(
        {
            section["fetch_tool"]
            for section in result.values()
            if isinstance(section, Mapping)
            and isinstance(section.get("fetch_tool"), str)
        }
    )
    result["all_truncation_model_visible"] = all_visible
    return result


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
