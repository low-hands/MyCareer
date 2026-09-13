"""The committed contract document must describe the code that is running.

Four things can drift and each has its own assertion: the file on disk can
fall behind the models; a route can start returning a model the document does
not describe; the examples the web parser is tested against can cover a
subset of the literals the models accept; and a nullable field can be non-null
in every example, so the web client's ``satisfies`` check never sees the
``null`` its type forgot. Each is a way the web client's tests would stay
green while the real API had already moved.
"""

from __future__ import annotations

import json

from career_agent.api.app import create_app
from career_agent.api.public_contract import (
    CONTRACT_PATH,
    public_contract,
    render_public_contract,
)

REGENERATE = "run `uv run python -m career_agent.api.public_contract` and commit the result"


def test_the_committed_contract_matches_the_models() -> None:
    assert CONTRACT_PATH.exists(), f"{CONTRACT_PATH} is missing; {REGENERATE}"
    committed = CONTRACT_PATH.read_text(encoding="utf-8")
    assert committed == render_public_contract(), (
        f"{CONTRACT_PATH.name} is stale; {REGENERATE}"
    )


def test_the_rendering_is_deterministic_and_carries_the_contract_verbatim() -> None:
    # The web suite type-checks this file; a timestamp or unordered key would
    # make every regeneration a spurious change.
    rendered = render_public_contract()
    assert rendered == render_public_contract()
    prefix = "export const API_CONTRACT = "
    start = rendered.index(prefix) + len(prefix)
    end = rendered.rindex(" as const;")
    assert json.loads(rendered[start:end]) == public_contract()


def _response_models() -> set[str]:
    """Every model a 2xx JSON response is declared to carry, read from the
    OpenAPI document so included routers and list endpoints are both seen."""

    names: set[str] = set()
    for operations in create_app().openapi()["paths"].values():
        for operation in operations.values():
            for status, response in operation.get("responses", {}).items():
                if not status.startswith("2"):
                    continue
                schema = response.get("content", {}).get("application/json", {}).get("schema")
                if schema is None:
                    continue
                # ``tuple[View, ...]`` list endpoints: the contract describes the item.
                if schema.get("type") == "array":
                    schema = schema.get("items", {})
                reference = schema.get("$ref")
                if reference:
                    names.add(reference.rsplit("/", 1)[-1])
    return names


def test_every_route_response_model_is_in_the_contract() -> None:
    contract = public_contract()
    described = set(contract["reads"])
    served = _response_models()
    assert served, "no routes declare a response_model; the scan is broken"
    missing = served - described
    assert not missing, f"routes return models the contract does not describe: {sorted(missing)}"
    # And nothing described that no route serves, or the web client would be
    # pinning itself to a shape it can never receive.
    orphaned = described - served
    assert not orphaned, f"contract describes models no route returns: {sorted(orphaned)}"


def test_stream_examples_cover_every_event_type() -> None:
    contract = public_contract()["stream_events"]
    mapping = contract["schema"]["discriminator"]["mapping"]
    exampled = {event["type"] for event in contract["examples"]}
    assert exampled == set(mapping), (
        f"event types without an example: {sorted(set(mapping) - exampled)}"
    )


def _enum(schema: dict, definition: str, field: str) -> set[str]:
    prop = schema["$defs"][definition]["properties"][field]
    if "enum" in prop:
        return set(prop["enum"])
    for arm in prop.get("anyOf", ()):
        if "enum" in arm:
            return set(arm["enum"])
    raise AssertionError(f"{definition}.{field} has no enum in the schema")


def test_stream_examples_cover_every_behaviour_selecting_literal() -> None:
    """The literals the web client branches on must each appear at least once,
    so the parser is exercised on every value rather than the one the author
    happened to think of."""

    contract = public_contract()["stream_events"]
    schema = contract["schema"]
    examples = contract["examples"]

    def values(event_type: str, field: str) -> set[str]:
        return {
            str(event[field])
            for event in examples
            if event["type"] == event_type and event.get(field) is not None
        }

    assert values("progress", "stage") == _enum(schema, "ProgressEvent", "stage")
    assert values("capability_started", "capability") == _enum(
        schema, "CapabilityStartedEvent", "capability"
    )
    assert values("capability_completed", "capability") == _enum(
        schema, "CapabilityCompletedEvent", "capability"
    )
    assert values("interaction_required", "kind") == _enum(
        schema, "InteractionRequiredEvent", "kind"
    )
    assert values("interaction_required", "scope") == _enum(
        schema, "InteractionRequiredEvent", "scope"
    )
    assert values("report_ready", "kind") == _enum(schema, "ReportReadyEvent", "kind")
    assert values("report_ready", "status_at_delivery") == _enum(
        schema, "ReportReadyEvent", "status_at_delivery"
    )


def _resolve(schema: dict, root: dict) -> dict:
    if "$ref" in schema:
        schema = root["$defs"][schema["$ref"].rsplit("/", 1)[-1]]
    arms = [arm for arm in schema.get("anyOf", ()) if arm.get("type") != "null"]
    if len(arms) == 1:
        return _resolve(arms[0], root)
    return schema


def _nullable_fields(schema: dict, root: dict, seen: set[str]) -> set[tuple[str, str]]:
    """``(definition, field)`` for every ``X | None`` property reachable from
    ``schema``, naming the definition by its schema title."""

    schema = _resolve(schema, root)
    title = schema.get("title", "")
    if title in seen:
        return set()
    seen.add(title)
    found: set[tuple[str, str]] = set()
    for field, prop in schema.get("properties", {}).items():
        if any(arm.get("type") == "null" for arm in prop.get("anyOf", ())):
            found.add((title, field))
        found |= _nullable_fields(prop, root, seen)
    if schema.get("type") == "array":
        found |= _nullable_fields(schema["items"], root, seen)
    return found


def _null_fields(schema: dict, value: object, root: dict) -> set[tuple[str, str]]:
    """``(definition, field)`` for every property that is ``None`` somewhere in
    ``value``, walked in step with the schema."""

    schema = _resolve(schema, root)
    found: set[tuple[str, str]] = set()
    if isinstance(value, dict):
        title = schema.get("title", "")
        for field, item in value.items():
            prop = schema.get("properties", {}).get(field)
            if prop is None:
                continue
            if item is None:
                found.add((title, field))
            else:
                found |= _null_fields(prop, item, root)
    elif isinstance(value, list) and schema.get("type") == "array":
        for item in value:
            found |= _null_fields(schema["items"], item, root)
    return found


def test_every_nullable_field_is_null_in_some_example() -> None:
    """The web client checks each example against its TypeScript type with
    ``satisfies``; that only notices a missing ``| null`` if some example
    actually carries the null."""

    declared: set[tuple[str, str]] = set()
    exercised: set[tuple[str, str]] = set()
    contract = public_contract()
    for entry in contract["reads"].values():
        schema = entry["schema"]
        declared |= _nullable_fields(schema, schema, set())
        for example in entry["examples"]:
            exercised |= _null_fields(schema, example, schema)
    # A transcript's ``pending_interaction`` is the stream's event; the client
    # types and parses both the same way, so the stream examples count too.
    stream = contract["stream_events"]["schema"]
    mapping = stream["discriminator"]["mapping"]
    for example in contract["stream_events"]["examples"]:
        exercised |= _null_fields({"$ref": mapping[example["type"]]}, example, stream)
    never_null = sorted(declared - exercised)
    assert not never_null, f"nullable fields no example sets to null: {never_null}"


def test_the_wire_schema_omits_internal_fields() -> None:
    # ``ContentDeltaEvent.delivery`` is transport pacing metadata excluded from
    # serialisation; the client must not be told it exists.
    schema = public_contract()["stream_events"]["schema"]
    assert "delivery" not in schema["$defs"]["ContentDeltaEvent"]["properties"]
    for event in public_contract()["stream_events"]["examples"]:
        assert "delivery" not in event
