from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from career_agent.agent.openai_compatible_client import (
    AgentConfigurationError,
    AgentWorkerError,
)
from career_agent.agent.openai_resume_analysis_worker import OpenAIResumeAnalysisWorker
from career_agent.agent.resume_analysis_contracts import (
    NumberedResumeAnalysisResult,
)
from career_agent.agent.structured_chat_completions import strict_json_schema

import test_resume_093_worker as contracts
from test_resume_093_extraction import document, synthetic_pdf


class ResponsesHarness(contracts.ChatHarness):
    def __init__(self) -> None:
        super().__init__()
        self.response_status = "completed"
        self.message_status = "completed"
        self.parts: list[dict[str, object]] | None = None
        self.extra_items: list[dict[str, object]] = []
        self.error: dict[str, object] | None = None
        self.incomplete_details: dict[str, object] | None = None

    def respond(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        self.paths.append(request.url.path)
        if self.transport_error is not None:
            raise self.transport_error
        if self.status != 200:
            return httpx.Response(self.status, json=self.error_body)
        output = self.outputs.pop(0) if self.outputs is not None else self.output
        parts = self.parts
        if parts is None:
            parts = (
                [{"type": "refusal", "refusal": self.refusal}]
                if self.refusal
                else [{"type": "output_text", "text": output, "annotations": []}]
            )
        payload = {
            "id": "synthetic-response",
            "object": "response",
            "created_at": 0,
            "model": contracts.CONFIG.model,
            "status": self.response_status,
            "error": self.error,
            "incomplete_details": self.incomplete_details,
            "output": [
                {"id": "reasoning", "type": "reasoning", "summary": []},
                {
                    "id": "message",
                    "type": "message",
                    "role": "assistant",
                    "status": self.message_status,
                    "content": parts,
                },
                *self.extra_items,
            ],
        }
        return httpx.Response(
            200,
            content=json.dumps(payload).encode(),
            headers={"content-type": "application/json"},
        )

    def worker(
        self,
        *,
        max_input_tokens: int = 32_000,
        disable_thinking: bool = False,
    ) -> OpenAIResumeAnalysisWorker:
        return OpenAIResumeAnalysisWorker(
            replace(contracts.CONFIG, max_input_tokens=max_input_tokens),
            client=self.client,
            protocol="responses",
            disable_thinking=disable_thinking,
        )


@pytest.fixture
def responses() -> Iterator[ResponsesHarness]:
    harness = ResponsesHarness()
    try:
        yield harness
    finally:
        harness.client.close()


@pytest.mark.parametrize("is_pdf", [False, True])
def test_explicit_responses_adapter_uses_strict_schema_and_only_text(
    responses: ResponsesHarness, is_pdf: bool
) -> None:
    raw = synthetic_pdf((contracts.SOURCE,)) if is_pdf else contracts.SOURCE.encode()
    result = responses.worker().analyze(document(raw, "pdf" if is_pdf else "text"))
    assert result == contracts.result()
    assert responses.paths == ["/v1/responses"]
    request = responses.requests[0]
    assert set(request) == {
        "model",
        "instructions",
        "input",
        "text",
        "max_output_tokens",
        "store",
    }
    assert request["store"] is False
    assert request["max_output_tokens"] == 8192
    assert request["text"] == {
        "format": {
            "type": "json_schema",
            "name": "resume_analysis_result",
            "strict": True,
            "schema": strict_json_schema(
                NumberedResumeAnalysisResult,
                field_enums={"source_locator": (1, 2)},
            ),
        }
    }
    inputs = request["input"]
    assert isinstance(inputs, list)
    assert len(inputs) == 1 and inputs[0]["role"] == "user"
    assert len(inputs[0]["content"]) == 1
    assert inputs[0]["content"][0]["type"] == "input_text"
    assert json.loads(inputs[0]["content"][0]["text"]) == {
        "source_paragraphs": [
            {"paragraph_number": index, "source_text": line}
            for index, line in enumerate(contracts.SOURCE.splitlines(), start=1)
        ]
    }
    wire = json.dumps(request)
    assert not any(
        term in wire for term in ("input_file", "file_data", "base64", "file_url")
    )


def test_responses_disable_thinking_is_explicit_provider_option(
    responses: ResponsesHarness,
) -> None:
    responses.worker(disable_thinking=True).analyze(
        document(contracts.SOURCE.encode(), "text")
    )
    assert responses.requests[0]["enable_thinking"] is False


@pytest.mark.parametrize(
    "status", ["incomplete", "failed", "queued", "in_progress", "cancelled"]
)
def test_noncompleted_response_cannot_create_draft(
    responses: ResponsesHarness, status: str
) -> None:
    responses.response_status = status
    with pytest.raises(AgentWorkerError) as caught:
        responses.worker().analyze(document(contracts.SOURCE.encode(), "text"))
    assert caught.value.code == "RESUME_ANALYSIS_INCOMPLETE_RESPONSE"


@pytest.mark.parametrize(
    "case",
    ["message_status", "error", "incomplete_details", "tool", "multiple_messages"],
)
def test_response_envelope_is_checked_before_json(
    responses: ResponsesHarness, case: str
) -> None:
    if case == "message_status":
        responses.message_status = "incomplete"
    elif case == "error":
        responses.error = {"code": "server_error", "message": "private-value"}
    elif case == "incomplete_details":
        responses.incomplete_details = {"reason": "max_output_tokens"}
    elif case == "tool":
        responses.extra_items = [
            {
                "type": "function_call",
                "id": "tool",
                "call_id": "call",
                "name": "unauthorized",
                "arguments": "{}",
            }
        ]
    else:
        responses.extra_items = [
            {
                "type": "message",
                "id": "other",
                "role": "assistant",
                "status": "completed",
                "content": [],
            }
        ]
    with pytest.raises(AgentWorkerError) as caught:
        responses.worker().analyze(document(contracts.SOURCE.encode(), "text"))
    assert caught.value.code in {
        "RESUME_ANALYSIS_INCOMPLETE_RESPONSE",
        "RESUME_ANALYSIS_EMPTY_RESPONSE",
    }
    assert "private-value" not in str(caught.value)


@pytest.mark.parametrize(
    "parts",
    [
        [],
        [
            {"type": "output_text", "text": "{}", "annotations": []},
            {"type": "output_text", "text": "{}", "annotations": []},
        ],
    ],
)
def test_no_ambiguous_output_concatenation(
    responses: ResponsesHarness, parts: list[dict[str, object]]
) -> None:
    responses.parts = parts
    with pytest.raises(AgentWorkerError) as caught:
        responses.worker().analyze(document(contracts.SOURCE.encode(), "text"))
    assert caught.value.code == "RESUME_ANALYSIS_EMPTY_RESPONSE"


@pytest.mark.parametrize(
    "output",
    [
        "not-json",
        "{}",
        '{"records": [], "records": [], "clarification_questions": [], "warnings": []}',
        '{"records": [], "clarification_questions": [], "warnings": [], "private-extra-key": true}',
        '{"records": [], "clarification_questions": [], "warnings": []}',
        "x" * 128_001,
        "\ud800",
    ],
)
def test_responses_retains_same_strict_json_validation(
    responses: ResponsesHarness, output: str
) -> None:
    contracts.test_invalid_wire_output_fails_closed_with_safe_detail(responses, output)


def test_responses_invalid_sample_is_retried_once(
    responses: ResponsesHarness,
) -> None:
    responses.outputs = ["not-json", contracts.numbered_result().model_dump_json()]

    assert responses.worker().analyze(
        document(contracts.SOURCE.encode(), "text")
    ) == contracts.result()
    assert responses.paths == ["/v1/responses", "/v1/responses"]


def test_responses_invalid_retry_exhaustion_is_terminal(
    responses: ResponsesHarness,
) -> None:
    responses.outputs = ["not-json", "still-not-json"]

    with pytest.raises(AgentWorkerError) as caught:
        responses.worker().analyze(document(contracts.SOURCE.encode(), "text"))

    assert caught.value.code == "RESUME_ANALYSIS_INVALID_RESPONSE"
    assert caught.value.retryable is False
    assert responses.paths == ["/v1/responses", "/v1/responses"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_locator", 9),
    ],
)
def test_responses_retains_exact_evidence_validation(
    responses: ResponsesHarness, field: str, value: object
) -> None:
    contracts.test_unissued_locator_is_rejected(
        responses, field, value
    )


def test_responses_requires_nullable_fields_and_nested_evidence(
    responses: ResponsesHarness,
) -> None:
    contracts.test_nullable_default_fields_are_still_required_on_wire(responses)
    contracts.test_nested_evidence_is_also_validated(responses)


@pytest.mark.parametrize("status,retryable", [(400, False), (429, True), (503, True)])
def test_responses_provider_error_does_not_fallback_or_retry(
    responses: ResponsesHarness, status: int, retryable: bool
) -> None:
    contracts.test_provider_errors_retain_safe_metadata_without_retries(
        responses, status, retryable
    )
    assert responses.paths == ["/v1/responses"]


def test_responses_budget_failure_precedes_provider_request(
    responses: ResponsesHarness,
) -> None:
    contracts.test_complete_request_budget_includes_schema_framing_and_output(responses)


def test_responses_refusal_is_safe(responses: ResponsesHarness) -> None:
    contracts.test_refusal_does_not_expose_model_text(responses)


def test_responses_formal_draft_persists_and_requires_confirmation(
    responses: ResponsesHarness, tmp_path: Path
) -> None:
    contracts.test_formal_analysis_persists_draft_then_requires_owned_one_time_confirmation(
        responses, tmp_path
    )


def test_responses_failure_never_persists_draft(
    responses: ResponsesHarness, tmp_path: Path
) -> None:
    contracts.test_worker_failure_cannot_create_draft_or_history(responses, tmp_path)


@pytest.mark.parametrize(
    "protocol,path",
    [("responses", "/v1/responses"), ("chat_completions", "/v1/chat/completions")],
)
def test_env_protocol_selection_is_explicit(protocol: str, path: str) -> None:
    harness = ResponsesHarness() if protocol == "responses" else contracts.ChatHarness()
    try:
        worker = OpenAIResumeAnalysisWorker.from_env(
            environ={
                "CUSTOM_BASE_URL": "https://provider.invalid/v1",
                "CUSTOM_API_KEY": "synthetic-not-a-credential",
                "CUSTOM_MODEL": "not-a-capability-heuristic",
                "CUSTOM_API_PROTOCOL": protocol,
                "CUSTOM_DISABLE_THINKING": "true",
            },
            prefix="CUSTOM",
            client=harness.client,
        )
        worker.analyze(document(contracts.SOURCE.encode(), "text"))
        assert harness.paths == [path]
        assert harness.requests[0]["enable_thinking"] is False
    finally:
        harness.client.close()


@pytest.mark.parametrize(
    "protocol", ["auto", "json_object", "", "private-invalid-value"]
)
def test_invalid_protocol_is_configuration_error_not_a_fallback(protocol: str) -> None:
    with pytest.raises(AgentConfigurationError) as caught:
        OpenAIResumeAnalysisWorker.from_env(
            environ={
                "RESUME_ANALYSIS_AGENT_BASE_URL": "https://provider.invalid/v1",
                "RESUME_ANALYSIS_AGENT_API_KEY": "synthetic-not-a-credential",
                "RESUME_ANALYSIS_AGENT_MODEL": "model",
                "RESUME_ANALYSIS_AGENT_API_PROTOCOL": protocol,
            }
        )
    assert caught.value.code == "AGENT_CONFIGURATION_INVALID"
    assert "private-invalid-value" not in str(caught.value)


@pytest.mark.parametrize("value", ["", "1", "yes", "private-invalid-value"])
def test_invalid_disable_thinking_is_configuration_error(value: str) -> None:
    with pytest.raises(AgentConfigurationError) as caught:
        OpenAIResumeAnalysisWorker.from_env(
            environ={
                "RESUME_ANALYSIS_AGENT_BASE_URL": "https://provider.invalid/v1",
                "RESUME_ANALYSIS_AGENT_API_KEY": "synthetic-not-a-credential",
                "RESUME_ANALYSIS_AGENT_MODEL": "model",
                "RESUME_ANALYSIS_AGENT_DISABLE_THINKING": value,
            }
        )
    assert caught.value.code == "AGENT_CONFIGURATION_INVALID"
    assert "private-invalid-value" not in str(caught.value)
