"""Bounded strict-schema Chat or explicitly selected Responses text calls.

Both protocols share the same local strict validation. No protocol fallback,
JSON-object downgrade, file upload, tools or automatic retry is permitted.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Protocol, TypeVar

from openai import (
    APIConnectionError,
    APIStatusError,
    OpenAI,
)
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from openai.types.responses import (
    Response,
    ResponseInputItemParam,
    ResponseOutputMessage,
    ResponseTextConfigParam,
)
from openai.types.shared_params import ResponseFormatJSONSchema
from pydantic import BaseModel, ValidationError

from career_agent.agent.openai_compatible_client import (
    AgentWorkerError,
    provider_worker_error,
)

T = TypeVar("T", bound=BaseModel)


class ChatCompletions(Protocol):
    def create(
        self,
        *,
        model: str,
        messages: Iterable[ChatCompletionMessageParam],
        response_format: ResponseFormatJSONSchema,
        max_tokens: int,
        timeout: float,
        extra_body: Mapping[str, object] | None = None,
    ) -> ChatCompletion: ...


class ChatResource(Protocol):
    @property
    def completions(self) -> ChatCompletions: ...


class StructuredChatClient(Protocol):
    @property
    def chat(self) -> ChatResource: ...


class TextResponses(Protocol):
    def create(
        self,
        *,
        model: str,
        instructions: str,
        input: Iterable[ResponseInputItemParam],
        text: ResponseTextConfigParam,
        max_output_tokens: int,
        timeout: float,
        store: bool,
        extra_body: Mapping[str, object] | None = None,
    ) -> Response: ...


class StructuredResponsesClient(Protocol):
    @property
    def responses(self) -> TextResponses: ...


def strict_json_schema(
    output_type: type[BaseModel],
    *,
    field_enums: Mapping[str, tuple[int, ...]] | None = None,
) -> dict[str, object]:
    """Require every field (nullable where appropriate), including nested ones.

    Defaults remain useful to persisted domain contracts but are not a part of
    the provider strict-schema dialect. Work on a fresh schema, not the model.
    """
    schema = output_type.model_json_schema()

    def strict(node: object) -> None:
        if isinstance(node, dict):
            node.pop("default", None)
            properties = node.get("properties")
            if isinstance(properties, dict):
                node["additionalProperties"] = False
                node["required"] = list(properties)
                for field, values in (field_enums or {}).items():
                    candidate = properties.get(field)
                    if isinstance(candidate, dict):
                        candidate["enum"] = list(values)
            for value in node.values():
                strict(value)
        elif isinstance(node, list):
            for value in node:
                strict(value)

    strict(schema)
    return schema


def _field_names(schema: object) -> set[str]:
    names: set[str] = set()
    if isinstance(schema, dict):
        properties = schema.get("properties")
        if isinstance(properties, dict):
            names.update(properties)
        for value in schema.values():
            names.update(_field_names(value))
    elif isinstance(schema, list):
        for value in schema:
            names.update(_field_names(value))
    return names


def _validation_detail(error: ValidationError, schema: dict[str, object]) -> str:
    known_fields = _field_names(schema)
    # Even an extra-field location is untrusted model output. Retain only
    # schema-owned names/indices; never Pydantic's message, input, or context.
    return json.dumps(
        [
            {
                "type": item["type"],
                "loc": [
                    part if isinstance(part, int) or part in known_fields else "<extra>"
                    for part in item["loc"]
                ],
            }
            for item in error.errors(
                include_url=False, include_context=False, include_input=False
            )[:20]
        ],
        sort_keys=True,
    )


def _complete_fields(value: object) -> bool:
    if isinstance(value, BaseModel):
        return value.model_fields_set == set(type(value).model_fields) and all(
            _complete_fields(item) for _, item in value
        )
    if isinstance(value, (tuple, list)):
        return all(_complete_fields(item) for item in value)
    return True


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError("Non-JSON numeric constant")


def structured_chat_completion(
    client: OpenAI | StructuredChatClient,
    *,
    model: str,
    timeout_seconds: float,
    instructions: str,
    content: str,
    output_type: type[T],
    schema_name: str,
    max_output_tokens: int,
    max_input_tokens: int,
    code_prefix: str,
    subject: str,
    field_enums: Mapping[str, tuple[int, ...]] | None = None,
    extra_body: Mapping[str, object] | None = None,
) -> T:
    schema = strict_json_schema(output_type, field_enums=field_enums)
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": content},
    ]
    response_format: ResponseFormatJSONSchema = {
        "type": "json_schema",
        "json_schema": {"name": schema_name, "strict": True, "schema": schema},
    }
    budget_payload: dict[str, object] = {
        "messages": messages,
        "response_format": response_format,
    }
    if extra_body is not None:
        budget_payload["extra_body"] = dict(extra_body)
    _check_request_budget(
        budget_payload,
        max_output_tokens=max_output_tokens,
        max_input_tokens=max_input_tokens,
        code_prefix=code_prefix,
        subject=subject,
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format=response_format,
            max_tokens=max_output_tokens,
            timeout=timeout_seconds,
            **({"extra_body": extra_body} if extra_body is not None else {}),
        )
    except (APIConnectionError, APIStatusError) as error:
        raise provider_worker_error(code_prefix, error) from None

    if len(response.choices) != 1:
        raise AgentWorkerError(
            f"{code_prefix}_EMPTY_RESPONSE",
            f"{subject} model returned no single structured output.",
        )
    choice = response.choices[0]
    if choice.message.refusal:
        raise AgentWorkerError(
            f"{code_prefix}_REFUSED", f"{subject} model declined the analysis."
        )
    text = choice.message.content
    if (
        choice.finish_reason != "stop"
        or choice.message.tool_calls
        or (_chat_output_budget_exhausted(response, max_output_tokens) and not text)
    ):
        raise AgentWorkerError(
            f"{code_prefix}_INCOMPLETE_RESPONSE",
            f"{subject} model did not complete structured output.",
        )
    return _validate_output(
        text,
        output_type=output_type,
        schema=schema,
        code_prefix=code_prefix,
        subject=subject,
    )


def _check_request_budget(
    payload: dict[str, object],
    *,
    max_output_tokens: int,
    max_input_tokens: int,
    code_prefix: str,
    subject: str,
) -> None:
    # UTF-8 bytes conservatively bound tokens without guessing a tokenizer.
    # Include schema and framing, plus a separate output reservation.
    input_bound = len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) + 256
    if input_bound + max_output_tokens > max_input_tokens:
        raise AgentWorkerError(
            f"{code_prefix}_TOKEN_BUDGET_EXCEEDED",
            f"{subject} exceeds the complete request token budget. Select a shorter document or check the input budget configuration.",
        )


def structured_text_response(
    client: OpenAI | StructuredResponsesClient,
    *,
    model: str,
    timeout_seconds: float,
    instructions: str,
    content: str,
    output_type: type[T],
    schema_name: str,
    max_output_tokens: int,
    max_input_tokens: int,
    code_prefix: str,
    subject: str,
    field_enums: Mapping[str, tuple[int, ...]] | None = None,
    extra_body: Mapping[str, object] | None = None,
) -> T:
    """Strict Responses adapter for a smoke-verified, explicitly configured endpoint."""
    schema = strict_json_schema(output_type, field_enums=field_enums)
    inputs: list[ResponseInputItemParam] = [
        {"role": "user", "content": [{"type": "input_text", "text": content}]}
    ]
    text_config: ResponseTextConfigParam = {
        "format": {
            "type": "json_schema",
            "name": schema_name,
            "strict": True,
            "schema": schema,
        }
    }
    budget_payload: dict[str, object] = {
        "instructions": instructions,
        "input": inputs,
        "text": text_config,
    }
    if extra_body is not None:
        budget_payload["extra_body"] = dict(extra_body)
    _check_request_budget(
        budget_payload,
        max_output_tokens=max_output_tokens,
        max_input_tokens=max_input_tokens,
        code_prefix=code_prefix,
        subject=subject,
    )
    try:
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=inputs,
            text=text_config,
            max_output_tokens=max_output_tokens,
            timeout=timeout_seconds,
            store=False,
            **({"extra_body": extra_body} if extra_body is not None else {}),
        )
    except (APIConnectionError, APIStatusError) as error:
        raise provider_worker_error(code_prefix, error) from None

    if (
        response.status != "completed"
        or response.error is not None
        or response.incomplete_details is not None
    ):
        raise AgentWorkerError(
            f"{code_prefix}_INCOMPLETE_RESPONSE",
            f"{subject} model did not complete structured output.",
        )
    messages: list[ResponseOutputMessage] = []
    for item in response.output:
        if item.type == "reasoning":
            # Reasoning metadata is neither evidence nor the final JSON result.
            continue
        if (
            item.type != "message"
            or item.status != "completed"
            or item.role != "assistant"
        ):
            raise AgentWorkerError(
                f"{code_prefix}_INCOMPLETE_RESPONSE",
                f"{subject} model returned an unexpected output item.",
            )
        messages.append(item)
    if len(messages) != 1:
        if _responses_output_budget_exhausted(response, max_output_tokens):
            raise AgentWorkerError(
                f"{code_prefix}_INCOMPLETE_RESPONSE",
                f"{subject} model did not complete structured output.",
            )
        raise AgentWorkerError(
            f"{code_prefix}_EMPTY_RESPONSE",
            f"{subject} model returned no single structured output.",
        )
    parts = messages[0].content
    if any(part.type == "refusal" for part in parts):
        raise AgentWorkerError(
            f"{code_prefix}_REFUSED", f"{subject} model declined the analysis."
        )
    if len(parts) != 1 or parts[0].type != "output_text":
        raise AgentWorkerError(
            f"{code_prefix}_EMPTY_RESPONSE",
            f"{subject} model returned no single structured output.",
        )
    return _validate_output(
        parts[0].text,
        output_type=output_type,
        schema=schema,
        code_prefix=code_prefix,
        subject=subject,
    )


def _chat_output_budget_exhausted(
    response: ChatCompletion, max_output_tokens: int
) -> bool:
    """Use total completion tokens, which already include reasoning tokens."""
    usage = response.usage
    return bool(
        usage is not None
        and type(usage.completion_tokens) is int
        and usage.completion_tokens >= max_output_tokens
    )


def _responses_output_budget_exhausted(
    response: Response, max_output_tokens: int
) -> bool:
    """Responses output_tokens is the total reasoning-plus-visible budget."""
    usage = response.usage
    return bool(
        usage is not None
        and type(usage.output_tokens) is int
        and usage.output_tokens >= max_output_tokens
    )


def _validate_output(
    text: str | None,
    *,
    output_type: type[T],
    schema: dict[str, object],
    code_prefix: str,
    subject: str,
) -> T:
    if not text or not text.strip():
        raise AgentWorkerError(
            f"{code_prefix}_EMPTY_RESPONSE",
            f"{subject} model returned no structured output.",
        )
    try:
        output_bytes = len(text.encode("utf-8"))
    except UnicodeEncodeError:
        raise AgentWorkerError(
            f"{code_prefix}_INVALID_RESPONSE",
            f"{subject} model returned invalid Unicode.",
        ) from None
    if output_bytes > 128_000:
        raise AgentWorkerError(
            f"{code_prefix}_INVALID_RESPONSE",
            f"{subject} model output exceeds the validation limit.",
        )
    try:
        # Pydantic accepts duplicate keys (last value wins). Reject ambiguous
        # wire output before validation rather than silently losing a claim.
        json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (ValueError, RecursionError):
        raise AgentWorkerError(
            f"{code_prefix}_INVALID_RESPONSE",
            f"{subject} model returned invalid JSON.",
            detail="invalid_json",
        ) from None
    try:
        result = output_type.model_validate_json(text, strict=True)
    except ValidationError as error:
        raise AgentWorkerError(
            f"{code_prefix}_INVALID_RESPONSE",
            f"{subject} model returned invalid structured output.",
            detail=_validation_detail(error, schema),
        ) from None
    if not _complete_fields(result):
        raise AgentWorkerError(
            f"{code_prefix}_INVALID_RESPONSE",
            f"{subject} model omitted required structured fields.",
            detail="missing_required_fields",
        )
    return result
