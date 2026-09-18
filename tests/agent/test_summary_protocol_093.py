from __future__ import annotations

import json
from dataclasses import replace

import httpx
import pytest
from openai import OpenAI

from career_agent.agent.conversation_memory_contracts import (
    HARNESS_SUMMARY_COUNTER_FIELDS,
    SummaryMessage,
)
from career_agent.agent.openai_compatible_client import (
    AgentWorkerError,
    OpenAICompatibleAgentConfig,
)
from career_agent.agent.openai_conversation_summary_worker import (
    OpenAIConversationSummaryWorker,
    summary_response_format,
)


def config() -> OpenAICompatibleAgentConfig:
    return OpenAICompatibleAgentConfig(
        endpoint="https://synthetic.test/v1/chat/completions",
        api_key="synthetic-key",
        model="synthetic-summary",
        timeout_seconds=12,
    )


def summary_object() -> dict[str, object]:
    return {
        "user_goals": ["Track synthetic applications"],
        "confirmed_decisions": ["Use SQLite"],
        "unresolved_questions": [],
        "active_constraints": ["Require approval"],
        "long_term_memory_candidates": [],
    }


def test_wire_request_is_chat_json_schema_with_independent_output_and_no_tools() -> (
    None
):
    requests: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "synthetic",
                "created": 1,
                "object": "chat.completion",
                "model": "synthetic-summary",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(summary_object()),
                        },
                    }
                ],
            },
        )

    with OpenAI(
        api_key="synthetic-key",
        base_url="https://synthetic.test/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as client:
        result = OpenAIConversationSummaryWorker(
            config(), client=client, max_output_tokens=2048
        ).summarize(
            previous=None,
            messages=(SummaryMessage(sequence=1, role="user", content="Use SQLite."),),
        )
    assert result.confirmed_decisions == ("Use SQLite",)
    assert len(requests) == 1
    assert requests[0]["model"] == "synthetic-summary"
    assert requests[0]["max_tokens"] == 2048
    assert requests[0]["response_format"] == summary_response_format()
    assert "tools" not in requests[0]
    assert "input" not in requests[0]
    schema = summary_response_format()["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert HARNESS_SUMMARY_COUNTER_FIELDS.isdisjoint(schema["properties"])
    assert set(schema["required"]) == set(summary_object())
    candidate = schema["$defs"]["DistilledFreeTextPreferenceCandidate"]
    assert candidate["additionalProperties"] is False
    assert set(candidate["required"]) == set(candidate["properties"])
    assert '"default":' not in json.dumps(schema)


@pytest.mark.parametrize(
    "content,finish_reason,code",
    [
        (
            "synthetic-private-provider-output",
            "stop",
            "CONVERSATION_SUMMARY_INVALID_RESPONSE",
        ),
        ("{}", "stop", "CONVERSATION_SUMMARY_INVALID_RESPONSE"),
        (
            json.dumps(summary_object() | {"omitted_active_constraint_count": 99}),
            "stop",
            "CONVERSATION_SUMMARY_INVALID_RESPONSE",
        ),
        (
            json.dumps(summary_object()),
            "length",
            "CONVERSATION_SUMMARY_INCOMPLETE_RESPONSE",
        ),
        (None, "stop", "CONVERSATION_SUMMARY_EMPTY_RESPONSE"),
        (
            json.dumps(
                summary_object()
                | {"user_goals": ["synthetic-private-provider-output" * 30]}
            ),
            "stop",
            "CONVERSATION_SUMMARY_INVALID_RESPONSE",
        ),
    ],
)
def test_invalid_partial_or_truncated_summary_fails_closed_without_content_in_error(
    content: str | None, finish_reason: str, code: str
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "synthetic",
                "created": 1,
                "object": "chat.completion",
                "model": "synthetic-summary",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": finish_reason,
                        "message": {"role": "assistant", "content": content},
                    }
                ],
            },
        )

    with (
        OpenAI(
            api_key="synthetic-key",
            base_url="https://synthetic.test/v1",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        ) as client,
        pytest.raises(AgentWorkerError) as raised,
    ):
        OpenAIConversationSummaryWorker(config(), client=client).summarize(
            previous=None,
            messages=(
                SummaryMessage(sequence=1, role="user", content="Synthetic input."),
            ),
        )
    assert raised.value.code == code
    assert raised.value.detail is None
    assert "synthetic-private-provider-output" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_oversized_summary_input_including_schema_is_rejected_before_provider_call() -> (
    None
):
    calls = 0

    def unexpected(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError("oversized request reached the provider")

    with (
        OpenAI(
            api_key="synthetic-key",
            base_url="https://synthetic.test/v1",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(unexpected)),
        ) as client,
        pytest.raises(AgentWorkerError) as raised,
    ):
        OpenAIConversationSummaryWorker(
            replace(config(), max_input_tokens=1024), client=client
        ).summarize(
            previous=None,
            messages=tuple(
                SummaryMessage(
                    sequence=index + 1, role="user", content="合成示例" * 1000
                )
                for index in range(8)
            ),
        )
    assert calls == 0
    assert raised.value.code == "CONVERSATION_SUMMARY_INPUT_BUDGET_EXCEEDED"
    assert raised.value.retryable is False


def test_nonretryable_provider_rejection_makes_one_call_not_silent_model_fallback() -> (
    None
):
    calls = 0

    def rejected(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "synthetic-private-provider-body",
                    "code": "InvalidParameter",
                }
            },
        )

    with (
        OpenAI(
            api_key="synthetic-key",
            base_url="https://synthetic.test/v1",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(rejected)),
        ) as client,
        pytest.raises(AgentWorkerError) as raised,
    ):
        OpenAIConversationSummaryWorker(config(), client=client).summarize(
            previous=None,
            messages=(
                SummaryMessage(sequence=1, role="user", content="Synthetic input."),
            ),
        )
    assert calls == 1
    assert raised.value.code == "CONVERSATION_SUMMARY_REJECTED_400"
    assert raised.value.retryable is False
    assert raised.value.detail is None
    assert "synthetic-private-provider-body" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__
    assert raised.value.provider is not None
    assert raised.value.provider.status == 400
    assert raised.value.provider.code == "InvalidParameter"


@pytest.mark.parametrize(
    "status,category,retryable",
    [
        (400, "configuration", False),
        (401, "configuration", False),
        (429, "rate_limit", True),
        (500, "upstream", False),
        (502, "upstream", True),
        (503, "upstream", True),
        (504, "upstream", True),
    ],
)
def test_summary_provider_errors_keep_only_shared_safe_metadata(
    status: int,
    category: str,
    retryable: bool,
) -> None:
    calls = 0

    def rejected(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            status,
            json={
                "error": {
                    "message": "synthetic-private-provider-body",
                    "code": "InvalidParameter",
                    "param": "response_format.json_schema",
                    "type": "invalid_request_error",
                }
            },
        )

    with (
        OpenAI(
            api_key="synthetic-key",
            base_url="https://synthetic.test/v1",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(rejected)),
        ) as client,
        pytest.raises(AgentWorkerError) as raised,
    ):
        OpenAIConversationSummaryWorker(config(), client=client).summarize(
            previous=None,
            messages=(
                SummaryMessage(sequence=1, role="user", content="Synthetic input."),
            ),
        )
    assert calls == 1
    assert raised.value.retryable is retryable
    assert raised.value.provider is not None
    assert raised.value.provider.as_dict() == {
        "status": status,
        "code": "InvalidParameter",
        "param": "response_format.json_schema",
        "type": "invalid_request_error",
        "category": category,
        "retryable": retryable,
    }
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__


@pytest.mark.parametrize(
    "transport_error,code,category",
    [
        (httpx.ReadTimeout, "CONVERSATION_SUMMARY_TIMEOUT", "timeout"),
        (httpx.ConnectError, "CONVERSATION_SUMMARY_TRANSPORT_ERROR", "transport"),
    ],
)
def test_summary_transport_errors_are_bounded_and_sanitized(
    transport_error: type[httpx.TransportError],
    code: str,
    category: str,
) -> None:
    calls = 0

    def disconnected(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise transport_error("synthetic-private-transport-detail", request=request)

    with (
        OpenAI(
            api_key="synthetic-key",
            base_url="https://synthetic.test/v1",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(disconnected)),
        ) as client,
        pytest.raises(AgentWorkerError) as raised,
    ):
        OpenAIConversationSummaryWorker(config(), client=client).summarize(
            previous=None,
            messages=(
                SummaryMessage(sequence=1, role="user", content="Synthetic input."),
            ),
        )
    assert calls == 1
    assert raised.value.code == code
    assert raised.value.retryable is True
    assert raised.value.provider is not None
    assert raised.value.provider.category == category
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__


@pytest.mark.parametrize("timeout", [0, 121, float("nan"), float("inf")])
def test_direct_summary_worker_rejects_invalid_timeout_before_client_creation(
    timeout: float,
) -> None:
    with pytest.raises(ValueError, match="timeout"):
        OpenAIConversationSummaryWorker(replace(config(), timeout_seconds=timeout))
