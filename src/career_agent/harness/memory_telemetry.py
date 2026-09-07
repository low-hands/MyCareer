from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import unicodedata
from typing import Any

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
    entries: list[dict[str, str]] = []
    default_city = context.profile.default_city
    if default_city:
        entries.append(
            {
                "entry_id": "person_intent/self/default_city",
                "content_digest": content_digest(default_city),
            }
        )
    return {
        "conversation_id": context.conversation_id,
        "conversation_key": conversation_trace_key(
            context.profile.user_id, context.conversation_id
        ),
        "career_memory_enabled": career_memory_enabled,
        "binding_profile": "p2",
        "slot_fingerprints": slots,
        "slot_chars": slot_chars,
        "career_profile_chars": _json_char_composition(
            slot_values["career_profile"]
        ),
        "entries": entries,
    }


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
    """Split exact serialized size into keys, scalar values, and punctuation."""

    total = _serialized_chars(value)
    keys, values = _key_and_value_chars(value)
    return {
        "keys": keys,
        "values": values,
        "punctuation": total - keys - values,
    }


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
