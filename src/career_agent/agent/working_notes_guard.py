from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

from career_agent.agent.main_agent_contracts import (
    MainAgentContext,
    career_profile_memory_files,
)
from career_agent.harness.memory_telemetry import (
    normalized_surface,
    surface_contains_token,
    surface_tokens,
)


_OPAQUE_ID = re.compile(r"^[a-z_]+_[a-f0-9]{16,}$")
_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
# ``unresolved_questions`` is excluded: a question is not a source, and it is
# where an assistant's "do you prefer Rust?" lands once it is compacted out of
# the recent window.  The other fields are still model-written from both roles;
# the summary prompt restricts them to what was explicitly stated.
_SUMMARY_FIELDS = (
    "user_goals",
    "confirmed_decisions",
    "active_constraints",
)
_NON_AUTHORITY_OBSERVATION_STATES = frozenset(
    {"working_notes_derived_argument", "career_history_found"}
)


def working_notes_only_tokens(
    *, arguments: dict[str, Any], context: MainAgentContext
) -> tuple[str, ...]:
    """Find lexical argument tokens sourced only from the scratchpad.

    This is a best-effort influence tripwire, not a provenance proof.  It can
    catch a literal carried from notes into a tool argument; paraphrases and
    reasoning that leave no lexical trace remain outside its scope.
    """

    notes = context.working_notes
    if notes is None or not notes.markdown:
        return ()
    argument_tokens = _tokens_from_values(arguments)
    if not argument_tokens:
        return ()
    note_surface = normalized_surface(notes.markdown)
    authoritative_surface = normalized_surface(
        "\n".join(_authoritative_text(context))
    )
    return tuple(
        token
        for token in argument_tokens
        if surface_contains_token(note_surface, token)
        and not surface_contains_token(authoritative_surface, token)
    )


def _tokens_from_values(
    value: Any, *, field_name: str | None = None
) -> tuple[str, ...]:
    tokens: list[str] = []
    seen: set[str] = set()

    def visit(item: Any, name: str | None = None) -> None:
        if isinstance(item, str):
            stripped = item.strip()
            if (
                name == "selection_index"
                or not stripped
                or _NUMBER.fullmatch(stripped)
                or stripped.startswith("sha256:")
                or _OPAQUE_ID.fullmatch(stripped)
            ):
                return
            for token in surface_tokens(stripped):
                if token not in seen:
                    seen.add(token)
                    tokens.append(token)
            return
        if isinstance(item, Mapping):
            for key, child in item.items():
                visit(child, str(key))
            return
        model_dump = getattr(item, "model_dump", None)
        if callable(model_dump):
            visit(model_dump(mode="python"), name)
            return
        if isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            for child in item:
                visit(child, name)

    visit(value, field_name)
    return tuple(tokens)


def _authoritative_text(context: MainAgentContext) -> tuple[str, ...]:
    text: list[str] = [context.user_message]
    # Only the user's own words ground a token.  An assistant reply that echoed
    # a note (for example while asking the user to confirm it) would otherwise
    # launder note content into authority on the very next decision.
    text.extend(
        message.content
        for message in context.recent_messages
        if message.role == "user"
    )
    if context.conversation_summary is not None:
        for field in _SUMMARY_FIELDS:
            text.extend(getattr(context.conversation_summary, field))
    text.extend(_string_values(career_profile_memory_files(context.profile)))
    text.extend(item.statement for item in context.free_text_preferences)
    text.extend(_candidate_titles(context))
    # A guard refusal repeats the suspicious tokens to explain itself.  It is
    # not a new authority source and must not make a paraphrased retry pass.
    # Superseded history claims are explicitly not current values; letting them
    # ground a token would revive the stale state M2b removed from projection.
    for observation in context.tool_observations:
        if observation.state in _NON_AUTHORITY_OBSERVATION_STATES:
            continue
        text.append(observation.message)
        if observation.body is not None:
            text.append(observation.body)
    return tuple(text)


def _candidate_titles(context: MainAgentContext) -> tuple[str, ...]:
    titles: list[str] = []
    task = context.task
    for field_name in type(task).model_fields:
        if field_name != "candidates" and not field_name.endswith("_candidates"):
            continue
        candidates = getattr(task, field_name)
        if not isinstance(candidates, Sequence):
            continue
        for candidate in candidates:
            title = (
                candidate.get("title")
                if isinstance(candidate, Mapping)
                else getattr(candidate, "title", None)
            )
            if isinstance(title, str):
                titles.append(title)
    return tuple(titles)


def _string_values(value: Any) -> tuple[str, ...]:
    values: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, str):
            values.append(item)
        elif isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            for child in item:
                visit(child)

    visit(value)
    return tuple(values)
