from __future__ import annotations

from collections.abc import Mapping
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


_ASCII_TOKEN = re.compile(r"[0-9A-Za-z_]+")


def normalized_surface(value: str) -> str:
    """Normalize a detector surface without changing its lexical meaning."""

    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def surface_contains_token(rendered: str, token: str) -> bool:
    """Match one normalized token with M6b's ASCII boundary discipline."""

    if not token:
        return False
    if all(character.isascii() for character in token):
        return (
            re.search(
                rf"(?<![0-9a-z_]){re.escape(token)}(?![0-9a-z_])",
                rendered,
            )
            is not None
        )
    # 2/3-grams supply the boundary for CJK: a token from a longer authoritative
    # run (for example 通勤 in 通勤太远) is intentionally considered present.
    return token in rendered


def surface_tokens(value: str) -> tuple[str, ...]:
    """Return the script-aware tokens used by best-effort memory detectors.

    ASCII words keep the same ``[0-9a-z_]`` boundaries as the M6b exact-use
    detector and are useful only from four characters onward.  CJK has no
    whitespace word boundary, so bounded 2/3-grams provide the deliberately
    lexical (not semantic) comparison surface.
    """

    normalized = normalized_surface(value)
    tokens: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        if token not in seen:
            seen.add(token)
            tokens.append(token)

    for match in _ASCII_TOKEN.finditer(normalized):
        token = match.group(0)
        if len(token) >= 4:
            add(token)
    run: list[str] = []

    def flush_cjk() -> None:
        if not run:
            return
        for width in (2, 3):
            for index in range(len(run) - width + 1):
                add("".join(run[index : index + width]))
        run.clear()

    for character in normalized:
        if _is_cjk(character):
            run.append(character)
        else:
            flush_cjk()
    flush_cjk()
    return tuple(tokens)


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
    context: Any,
    *,
    career_memory_enabled: bool,
    working_notes_only_tokens: int | None = None,
    working_notes_only_argument: bool | None = None,
) -> dict[str, Any]:
    """Return P2-safe fingerprints without persisting projected user text."""

    projection = context.model_context()
    career_profile = projection.get("career_profile")
    career_memory = projection.get("career_memory")
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
        career_memory,
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
        "career_profile_budgets": context.career_profile_budgets.model_dump(
            mode="json"
        ),
        "career_profile_delivery": delivery,
        "career_profile_truncation": _career_profile_truncation(
            career_memory,
            delivery,
        ),
        "entries": entries,
        "working_notes_chars": len(
            getattr(getattr(context, "working_notes", None), "markdown", "")
        ),
    }
    if observation["working_notes_chars"]:
        observation["working_notes_only_tokens"] = int(
            working_notes_only_tokens or 0
        )
        observation["working_notes_only_argument"] = int(
            bool(working_notes_only_argument)
        )
    hidden_preferences = _hidden_free_text_preferences(context)
    if hidden_preferences:
        observation["free_text_preferences_hidden"] = hidden_preferences
    return observation


def _hidden_free_text_preferences(context: Any) -> int:
    """How many preferences the cap kept off the projection this turn.

    Emitted only when the answer is non-zero, on the same reasoning as
    ``working_notes_only_tokens``: a key that is always present carries no
    signal. M6b reads it to decide whether a truncation the model could not see
    disqualifies a lower-bound claim about preference use.
    """

    projected = tuple(getattr(context, "free_text_preferences", ()))
    active_shown = sum(item.status == "active" for item in projected)
    quarantined_shown = min(
        sum(item.status == "quarantined" for item in projected),
        # The projection renders at most three candidates regardless of how many
        # were carried, so a fourth is hidden from the model even though the
        # context holds it.
        3,
    )
    return max(
        0,
        int(getattr(context, "free_text_preferences_active_total", 0))
        - active_shown,
    ) + max(
        0,
        int(getattr(context, "free_text_preferences_quarantined_total", 0))
        - quarantined_shown,
    )


def _surface(value: str) -> str:
    return normalized_surface(value)


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
    for binding in (*profile_bindings, *career_bindings):
        identity = (binding.entry_id, binding.content_digest)
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
    return (
        tuple(
            sorted(
                selected.values(),
                key=lambda item: (item.entry_id, item.revision, item.update_id),
            )
        ),
        profile_complete and career_complete,
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
            if value and value in rendered
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


def _career_profile_delivery(context: Any, value: Any) -> dict[str, int]:
    """Report evidence bounds and complete deterministic profile delivery."""

    profile = value if isinstance(value, Mapping) else {}
    records = profile.get("records")
    record_rows = records if isinstance(records, list) else ()
    claim_rows = tuple(
        claim
        for record in record_rows
        if isinstance(record, Mapping)
        for claim in (
            record.get("confirmed_highlights", ())
            if isinstance(record.get("confirmed_highlights"), list)
            else ()
        )
    )
    records_returned = len(record_rows)
    claims_returned = len(claim_rows)
    targets_returned = len(context.profile.current_targets)
    constraints_returned = len(context.profile.hard_constraints)
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
) -> dict[str, bool]:
    """Expose only truncation presence and whether counters disclose it."""

    profile = value if isinstance(value, Mapping) else {}
    any_truncated = any(
        delivery[key] > 0
        for key in (
            "records_dropped",
            "claims_dropped",
            "current_targets_dropped",
            "hard_constraints_dropped",
        )
    )
    records_visible = delivery["records_dropped"] == 0 or all(
        key in profile for key in ("records_returned", "records_total")
    )
    claims_visible = delivery["claims_dropped"] == 0 or all(
        key in profile for key in ("claims_returned", "claims_total")
    )
    # Profile projection is complete by construction. A mismatch raises before
    # telemetry is produced instead of being represented as a recoverable
    # archive overflow.
    targets_visible = delivery["current_targets_dropped"] == 0
    constraints_visible = delivery["hard_constraints_dropped"] == 0
    return {
        "any_truncated": any_truncated,
        "all_truncation_model_visible": (
            records_visible
            and claims_visible
            and targets_visible
            and constraints_visible
        ),
    }
