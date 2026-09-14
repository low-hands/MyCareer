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


# Phrasings by which a user hands the choice to what the agent has stored about
# them.  "按我的偏好" without a memory verb is included: the user is not stating
# a preference in that sentence, so the referent has to come from memory too.
_REMEMBERED_PREFERENCE_APPEALS = (
    "你记得的",
    "你记得我",
    "你记住的",
    "你了解的我",
    "你知道的我",
    "按我的偏好",
    "按照我的偏好",
    "根据我的偏好",
    "按我偏好",
    "按我的喜好",
    "根据我的喜好",
    "按我平时的",
    "按我一贯的",
)


def remembered_preference_without_authority(context: MainAgentContext) -> bool:
    """Whether "by the preference you remember" can only mean the scratchpad.

    The lexical guard sees a note literal in an argument; it cannot see a call
    such as ``compare_saved_jobs([1, 2])`` whose whole reason is a note. This
    is the complementary check on the request instead of the arguments: the
    user delegated the choice to remembered preference, no confirmed source
    holds one, and the working notes do hold something. The only preference
    the call could act on is then an unconfirmed guess, so the model is asked
    to confirm it first.

    Recent user messages are not consulted. A preference the user stated a
    moment ago is authoritative, but this check cannot tell which sentence
    it was; a request phrased as a memory appeal with the answer in the window
    costs one confirming question, which is the safe side of the error.
    """

    notes = context.working_notes
    if notes is None or not notes.markdown.strip():
        return False
    if not any(
        appeal in context.user_message for appeal in _REMEMBERED_PREFERENCE_APPEALS
    ):
        return False
    return not _notes_preference_is_confirmed(context, notes.markdown)


# Scratchpad boilerplate that says a note is about a preference without saying
# which one. Shared with a confirmed statement it proves nothing.
_PREFERENCE_BOILERPLATE = frozenset(
    {
        "偏好", "喜好", "喜欢", "倾向", "更想", "想要", "希望",
        "用户", "可能", "似乎", "大概", "观察", "记录", "确认", "未确认",
        "待确认", "已确认", "岗位", "公司", "工作",
    }
)


_OBSERVATION_SEPARATORS = re.compile(r"[\n\r、；;，,。.]+")


def _note_observations(markdown: str) -> tuple[tuple[str, ...], ...]:
    """Split the scratchpad into single observations, each as its tokens.

    Headings are structure, not remembered facts. A line that lists several
    preferences is several observations, so confirming one of them cannot vouch
    for its neighbours on the same line.
    """

    observations: list[tuple[str, ...]] = []
    for line in markdown.splitlines():
        if line.lstrip().startswith("#"):
            continue
        for clause in _OBSERVATION_SEPARATORS.split(line):
            tokens = tuple(
                token
                for token in surface_tokens(clause)
                if token not in _PREFERENCE_BOILERPLATE
            )
            if tokens:
                observations.append(tokens)
    return tuple(observations)


def _notes_preference_is_confirmed(context: MainAgentContext, markdown: str) -> bool:
    """Whether confirmed sources state everything the scratchpad remembers.

    Existence of *some* confirmed fact is not enough: a confirmed city says
    nothing about a note's "prefers large companies", and acting on the note
    because the city exists would be the unconfirmed guess the guard is for.
    Nor is one confirmed observation enough for the rest: the request is for
    "my remembered preferences" as a whole, and the guard cannot tell which
    observation the model will lean on, so each one needs its own confirmed
    source overlapping it in a token that carries the preference itself rather
    than the word "preference".
    """

    observations = _note_observations(markdown)
    if not observations:
        return False
    confirmed_surface = normalized_surface(
        "\n".join(_confirmed_preference_text(context))
    )
    if not confirmed_surface:
        return False
    return all(
        any(surface_contains_token(confirmed_surface, token) for token in tokens)
        for tokens in observations
    )


def _confirmed_preference_text(context: MainAgentContext) -> tuple[str, ...]:
    text: list[str] = [
        item.statement
        for item in context.free_text_preferences
        if item.status == "active"
    ]
    profile = context.profile
    if profile.default_city:
        text.append(profile.default_city)
    text.extend(constraint.value for constraint in profile.hard_constraints)
    for target in profile.current_targets:
        text.extend(
            value
            for value in (
                target.title,
                target.city,
                target.salary_expectation,
                target.experience,
                target.education,
            )
            if value
        )
    summary = context.conversation_summary
    if summary is not None:
        text.extend(summary.user_goals)
        text.extend(summary.active_constraints)
    return tuple(text)


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
