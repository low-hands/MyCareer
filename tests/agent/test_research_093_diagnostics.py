import json

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError
import pytest

from career_agent.agent.job_research_provider_diagnostics import (
    ProviderRequestObserver,
    ProviderRequestStructure,
    request_structure,
    trace_research_request,
)
from career_agent.agent.openai_compatible_client import (
    AgentWorkerError,
    ProviderErrorMetadata,
    provider_error_metadata,
    provider_worker_error,
)
from career_agent.harness.capability_steps import CapabilityStep, observing_capability_steps
from career_agent.harness.observability import (
    ACTIVE_TRACE_CONTEXT,
    CapabilityModelTraceCallback,
    InMemoryTraceRecorder,
    traced_model_call,
)


PRIVATE = "PrivateCompanyJDSecret123"


def _status_error(status: int, fields: dict[str, object]) -> APIStatusError:
    return APIStatusError(
        PRIVATE,
        response=httpx.Response(status, request=httpx.Request("POST", "https://provider.test/responses")),
        body={"error": {**fields, "message": PRIVATE}},
    )


def test_serialized_shape_contains_protocol_not_payload_values() -> None:
    schema = {
        "type": "object", "properties": {PRIVATE: {"type": "string", "description": PRIVATE}},
        "required": [PRIVATE], "additionalProperties": False,
    }
    request = httpx.Request(
        "POST", f"https://provider.test/responses?key={PRIVATE}",
        headers={"Authorization": f"Bearer {PRIVATE}"},
        json={
            "model": PRIVATE, "input": [{"role": "user", "content": PRIVATE}],
            "metadata": {PRIVATE: PRIVATE},
            "tools": [{"type": "function", "name": PRIVATE, "parameters": schema},
                      {"type": "web_search"}, {"type": PRIVATE}],
            "text": {"format": {"type": "json_schema", "name": PRIVATE, "schema": schema}},
        },
    )
    shape = request_structure(request)
    assert shape.protocol == "responses"
    assert shape.tool_types == ("function", "other", "web_search")
    assert shape.schema_bytes == 2 * len(json.dumps(schema, ensure_ascii=False).encode())
    assert "input[].content" in shape.field_paths
    assert "text.format.schema.properties.*.description" in shape.field_paths
    assert "metadata.*" in shape.field_paths
    assert PRIVATE not in json.dumps(shape.as_dict())
    assert set(shape.as_dict()["sdk_versions"]) == {
        "openai", "langchain-openai", "langchain", "deepagents", "httpx",
    }


def test_schema_property_names_are_masked_even_when_they_match_protocol_fields() -> None:
    schema = {"type": "object", "properties": {
        "title": {"type": "string"}, "parameters": {"type": "string"},
    }}
    shape = request_structure(httpx.Request("POST", "https://provider.test/responses", json={
        "metadata": {"model": PRIVATE},
        "text": {"format": {"schema": schema}},
    }))
    assert "metadata.model" not in shape.field_paths
    assert "metadata.*" in shape.field_paths
    assert "text.format.schema.properties.title" not in shape.field_paths
    assert "text.format.schema.properties.*.type" in shape.field_paths
    assert shape.schema_bytes == len(json.dumps(schema, ensure_ascii=False).encode())


def test_observer_is_best_effort_and_unknown_tool_values_are_not_echoed() -> None:
    structures: list[ProviderRequestStructure] = []
    observer = ProviderRequestObserver(structures.append)
    observer(httpx.Request("POST", "https://provider.test/responses", content=PRIVATE))
    assert not structures
    observer(httpx.Request("POST", "https://provider.test/unknown", json={"tools": [{"type": {PRIVATE: PRIVATE}}]}))
    assert structures[0].protocol == "unknown"
    assert structures[0].tool_types == ("other",)
    assert PRIVATE not in json.dumps(structures[0].as_dict())

    def unavailable_sink(structure: ProviderRequestStructure) -> None:
        raise RuntimeError(PRIVATE)

    ProviderRequestObserver(unavailable_sink)(
        httpx.Request("POST", "https://provider.test/responses", json={"input": PRIVATE})
    )


@pytest.mark.parametrize("status,category,retryable", [
    (400, "configuration", False), (401, "configuration", False),
    (403, "configuration", False), (429, "rate_limit", True),
    (500, "upstream", False), (502, "upstream", True), (503, "upstream", True),
])
def test_provider_status_is_preserved_without_message(status: int, category: str, retryable: bool) -> None:
    error = _status_error(status, {
        "code": "InvalidParameter", "param": "tools[0].type", "type": "invalid_request_error",
    })
    metadata = provider_error_metadata(error)
    assert metadata is not None
    assert metadata.as_dict() == {
        "status": status, "code": "InvalidParameter", "param": "tools[0].type",
        "type": "invalid_request_error", "category": category, "retryable": retryable,
    }
    translated = provider_worker_error("JOB_RESEARCH", error)
    assert translated.provider == metadata
    assert translated.retryable is retryable
    assert PRIVATE not in str(translated)
    assert PRIVATE not in json.dumps(metadata.as_dict())


@pytest.mark.parametrize("value", [PRIVATE, "sk-secret123", "https://private.test", "text.PrivateCompanyJDSecret123", "tools[secret].type"])
def test_identifiers_are_allowlisted_not_just_regex_checked(value: str) -> None:
    metadata = provider_error_metadata(_status_error(400, {"code": value, "param": value, "type": value}))
    assert metadata is not None
    assert metadata.code is metadata.param is metadata.type is None
    assert ProviderErrorMetadata(code=value, param=value, type=value).as_dict()["code"] is None


def test_connection_and_timeout_categories_are_distinct() -> None:
    request = httpx.Request("POST", "https://provider.test/responses")
    timeout = provider_worker_error("JOB_RESEARCH", APITimeoutError(request=request))
    transport = provider_worker_error("JOB_RESEARCH", APIConnectionError(request=request))
    assert timeout.code == "JOB_RESEARCH_TIMEOUT" and timeout.retryable
    assert transport.code == "JOB_RESEARCH_TRANSPORT_ERROR" and transport.retryable


def test_raw_sdk_code_never_bypasses_safe_metadata_in_model_trace() -> None:
    recorder = InMemoryTraceRecorder()
    callback = CapabilityModelTraceCallback(stage="job_research", worker="DeepAgentJobResearchWorker")
    steps: list[CapabilityStep] = []
    token = ACTIVE_TRACE_CONTEXT.set((recorder, "synthetic-turn"))
    try:
        with observing_capability_steps(steps.append):
            callback.on_chat_model_start({}, [[PRIVATE]], run_id="synthetic-call")
            callback.on_llm_error(_status_error(400, {"code": PRIVATE}), run_id="synthetic-call")
        trace_research_request(request_structure(httpx.Request(
            "POST", "https://provider.test/responses", json={"input": PRIVATE},
        )))
    finally:
        ACTIVE_TRACE_CONTEXT.reset(token)
    events = recorder.snapshot("synthetic-turn").events
    assert [event.event_type for event in events] == ["model_attempt", "model_failed", "provider_request"]
    assert events[1].error_code == "APIStatusError"
    assert events[1].recoverable is False
    assert events[1].details["provider"]["status"] == 400
    assert not any(step.kind == "retry" for step in steps)
    assert PRIVATE not in recorder.snapshot("synthetic-turn").model_dump_json()


@pytest.mark.parametrize("error,retryable", [
    (ValueError(PRIVATE), False),
    (_status_error(400, {"code": "InvalidParameter"}), False),
    (_status_error(429, {"code": "rate_limit_exceeded"}), True),
])
def test_callback_only_announces_retry_for_known_retryable_failures(
    error: Exception, retryable: bool,
) -> None:
    callback = CapabilityModelTraceCallback(stage="job_research", worker="DeepAgentJobResearchWorker")
    steps: list[CapabilityStep] = []
    with observing_capability_steps(steps.append):
        callback.on_chat_model_start({}, [[PRIVATE]], run_id="synthetic-call")
        callback.on_llm_error(error, run_id="synthetic-call")
    assert any(step.kind == "retry" for step in steps) is retryable


def test_decorated_worker_keeps_stable_code_and_discards_exception_payload() -> None:
    class Worker:
        @traced_model_call("job_research")
        def run(self) -> None:
            raise AgentWorkerError("JOB_RESEARCH_REJECTED_400", PRIVATE, detail=PRIVATE,
                                   provider=ProviderErrorMetadata(status=400, code="InvalidParameter"))

    recorder = InMemoryTraceRecorder()
    token = ACTIVE_TRACE_CONTEXT.set((recorder, "synthetic-turn"))
    try:
        with pytest.raises(AgentWorkerError):
            Worker().run()
    finally:
        ACTIVE_TRACE_CONTEXT.reset(token)
    trace = recorder.snapshot("synthetic-turn")
    assert trace.events[-1].error_code == "JOB_RESEARCH_REJECTED_400"
    assert trace.events[-1].recoverable is False
    assert PRIVATE not in trace.model_dump_json()
