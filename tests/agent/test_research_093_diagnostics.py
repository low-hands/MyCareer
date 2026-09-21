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
    ProviderErrorCategory,
    ProviderErrorMetadata,
    provider_error_metadata,
    provider_worker_error,
    public_error_code,
    user_facing_worker_failure,
    worker_failure_reason,
)
from career_agent.harness.capability_steps import CapabilityStep, observing_capability_steps
from career_agent.harness.observability import (
    ACTIVE_TRACE_CONTEXT,
    CapabilityModelTraceCallback,
    InMemoryTraceRecorder,
    traced_model_call,
)


PRIVATE = "PrivateCompanyJDSecret123"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("RESUME_ANALYSIS_REJECTED_400", "RESUME_ANALYSIS_REJECTED_400"),
        ("bad code with spaces", "CAPABILITY_FAILED"),
        ("<script>alert(1)</script>", "CAPABILITY_FAILED"),
        (None, "CAPABILITY_FAILED"),
    ],
)
def test_public_error_code_accepts_only_stable_identifiers(raw, expected) -> None:
    assert public_error_code(raw) == expected


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
    (520, "upstream", False), (522, "upstream", False), (523, "upstream", False),
    (524, "time_budget", False), (598, "time_budget", False),
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


@pytest.mark.parametrize(
    ("category", "retryable", "expected"),
    [
        ("configuration", False, "当前模型配置不支持该能力"),
        ("rate_limit", True, "模型服务正在限流"),
        ("upstream", True, "上游模型服务暂时不可用"),
        ("upstream", False, "上游模型服务未能完成请求"),
        ("transport", True, "当前无法连接模型服务"),
        ("timeout", True, "模型服务响应超时"),
        ("time_budget", False, "本次请求超过服务端时间预算"),
    ],
)
def test_user_failure_reason_follows_provider_category(
    category: ProviderErrorCategory, retryable: bool, expected: str
) -> None:
    error = AgentWorkerError(
        "JOB_RESEARCH_REJECTED_400",
        PRIVATE,
        retryable=retryable,
        provider=ProviderErrorMetadata(
            category=category, retryable=retryable
        ),
    )
    message = user_facing_worker_failure("岗位研究", error)
    assert expected in message
    assert "JOB_RESEARCH_REJECTED_400" in message
    assert PRIVATE not in message


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


def test_gateway_time_budget_preserves_status_without_suggesting_retry() -> None:
    error = provider_worker_error("JOB_RESEARCH", _status_error(524, {}))
    # The status survives, the word "rejected" must not: the provider took the
    # request and the gateway cut the idle connection before it answered, so a
    # code reading REJECTED sends an operator to check a healthy endpoint.
    assert error.code == "JOB_RESEARCH_TIME_BUDGET_524"
    assert "REJECTED" not in error.code
    assert error.provider is not None and error.provider.category == "time_budget"
    assert error.retryable is False
    message = user_facing_worker_failure("岗位研究", error)
    assert "超过服务端时间预算" in message
    assert "稍后重试" not in message and "配置不支持" not in message


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


def test_the_model_reads_the_reason_and_the_user_reads_the_code() -> None:
    """The code is for the person who has to report it, not for the model.

    An identifier tells the model nothing it can act on — retry or stop is
    carried by ``retryable`` — but it is reliably copied into the sentence the
    model writes, so the user ends up reading an error code mid-answer instead
    of the operator reading it in a place that stays put. Both readers get the
    same explanation; only one of them gets the identifier.
    """

    for status in (400, 429, 503, 524):
        error = provider_worker_error("JOB_RESEARCH", _status_error(status, {}))
        reason = worker_failure_reason(error)
        shown = user_facing_worker_failure("岗位研究", error)

        assert error.code not in reason
        assert "错误码" not in reason
        assert error.code in shown
        assert reason in shown


def test_the_failure_observation_handed_to_the_model_carries_no_error_code() -> None:
    """Asserting the two functions differ does not pin which one is wired in.

    The observation is what the model reads and then paraphrases, so this
    reaches through the registry's own factory rather than re-deriving the
    string: swapping the call site back to the user-facing wording has to turn
    this red, and the code must still reach the caller through the payload.
    """

    from career_agent.agent.main_agent_tools import MainAgentToolRegistry
    from career_agent.services.job_research import JobResearchExecutionError

    error = JobResearchExecutionError(
        run_id="run-1",
        code="JOB_RESEARCH_TIME_BUDGET_524",
        retryable=False,
        detail=PRIVATE,
        provider=ProviderErrorMetadata(
            status=524, category="time_budget", retryable=False
        ),
    )

    observation = MainAgentToolRegistry._job_research_failure(
        tool_name="research_job", error=error, job_posting_id="job-1"
    )

    assert "JOB_RESEARCH_TIME_BUDGET_524" not in observation.message
    assert "错误码" not in observation.message
    assert "本次请求超过服务端时间预算" in observation.message
    assert observation.payload["error_code"] == "JOB_RESEARCH_TIME_BUDGET_524"
    assert PRIVATE not in observation.model_dump_json()
