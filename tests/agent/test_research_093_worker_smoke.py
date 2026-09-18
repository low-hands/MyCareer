import json
from pathlib import Path
from unittest.mock import patch

import httpx
from langchain.agents.structured_output import ProviderStrategy, StructuredOutputValidationError
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from openai import APIStatusError
from openai.types.responses import Response
import pytest

from career_agent.agent.deepagent_job_research_worker import DeepAgentJobResearchWorker, _base_url
from career_agent.agent.job_research_contracts import JobResearchWorkerRequest
from career_agent.agent.job_research_provider_diagnostics import ProviderRequestStructure
from career_agent.agent import job_research_provider_smoke as smoke
from career_agent.agent.openai_compatible_client import AgentWorkerError, OpenAICompatibleAgentConfig
from career_agent.domain.job_research import JobResearchDraft, JobResearchScope


PRIVATE = "PrivateModelOutputSecret123"
DRAFT: dict[str, object] = {
    "summary": "Microsoft Azure provides public cloud services.",
    "sources": [{
        "source_key": "S1", "url": "https://azure.microsoft.com/en-us/",
        "title": "Azure", "publisher": "Microsoft", "published_at": None,
        "relevant_excerpt": "Azure is Microsoft's cloud platform.",
    }],
    "findings": [{
        "topic": "Public product", "statement": "Microsoft offers Azure cloud services.",
        "evidence_type": "fact", "source_keys": ["S1"], "confidence": "high",
    }],
    "open_questions": [], "limitations": ["Synthetic test, not role-specific evidence."],
}


class Checkpointer:
    def delete_thread(self, thread_id: str) -> None:
        pass


class Agent:
    def __init__(self, *, state: object = None, error: Exception | None = None) -> None:
        self.state = state if state is not None else {"structured_response": DRAFT}
        self.error = error
        self.calls: list[tuple[object, dict[str, object]]] = []

    def invoke(self, payload: object, *, config: dict[str, object]) -> object:
        self.calls.append((payload, config))
        if self.error is not None:
            raise self.error
        return self.state


def _config() -> OpenAICompatibleAgentConfig:
    return OpenAICompatibleAgentConfig(
        endpoint="https://synthetic-provider.test/v1/chat/completions",
        api_key="synthetic-test-key", model="synthetic-model", timeout_seconds=2,
    )


def _request() -> JobResearchWorkerRequest:
    return JobResearchWorkerRequest(
        company_name="Microsoft", role_title="Cloud engineer",
        jd_text="Synthetic Azure cloud role.", scope=JobResearchScope(max_sources=2),
    )


def test_worker_bounds_execution_and_does_not_retry_400() -> None:
    error = APIStatusError(
        PRIVATE,
        response=httpx.Response(400, request=httpx.Request("POST", "https://provider.test/responses")),
        body={"code": "InvalidParameter", "param": "text.format", "type": "invalid_request_error"},
    )
    agent = Agent(error=error)
    worker = DeepAgentJobResearchWorker(_config(), skills_root=Path("skills"), checkpointer=Checkpointer(),
                                        agent=agent, recursion_limit=12)
    with pytest.raises(AgentWorkerError) as raised:
        worker.research(run_id="synthetic-run", request=_request())
    assert raised.value.code == "JOB_RESEARCH_REJECTED_400"
    assert raised.value.retryable is False
    assert raised.value.provider is not None and raised.value.provider.param == "text.format"
    assert len(agent.calls) == 1
    assert agent.calls[0][1]["recursion_limit"] == 12
    assert PRIVATE not in str(raised.value)


@pytest.mark.parametrize("limit", [0, 1, 129])
def test_worker_rejects_unbounded_step_configuration(limit: int) -> None:
    with pytest.raises(ValueError, match="recursion_limit"):
        DeepAgentJobResearchWorker(_config(), skills_root=Path("skills"), checkpointer=Checkpointer(),
                                   agent=Agent(), recursion_limit=limit)


def test_framework_validation_never_becomes_persistable_model_output() -> None:
    error = StructuredOutputValidationError("JobResearchDraft", ValueError(PRIVATE), AIMessage(content=PRIVATE))
    worker = DeepAgentJobResearchWorker(_config(), skills_root=Path("skills"), checkpointer=Checkpointer(),
                                        agent=Agent(error=error))
    with pytest.raises(AgentWorkerError) as raised:
        worker.research(run_id="synthetic-run", request=_request())
    assert raised.value.code == "JOB_RESEARCH_INVALID_RESPONSE"
    assert not raised.value.retryable
    assert PRIVATE not in str(raised.value)
    assert PRIVATE not in (raised.value.detail or "")


def test_pydantic_extra_field_location_is_not_model_output_in_diagnostics() -> None:
    worker = DeepAgentJobResearchWorker(_config(), skills_root=Path("skills"), checkpointer=Checkpointer(),
                                        agent=Agent(state={"structured_response": {**DRAFT, PRIVATE: PRIVATE}}))
    with pytest.raises(AgentWorkerError) as raised:
        worker.research(run_id="synthetic-run", request=_request())
    assert PRIVATE not in (raised.value.detail or "")
    assert json.loads(raised.value.detail or "[]") == [{"type": "extra_forbidden", "loc": ["*"]}]


def test_citation_closure_rejection_does_not_echo_unknown_source_key() -> None:
    draft = JobResearchDraft.model_validate(DRAFT)
    invalid = draft.model_copy(update={"findings": (
        draft.findings[0].model_copy(update={"source_keys": ("PRIVATEKEY",)}),
    )})
    worker = DeepAgentJobResearchWorker(_config(), skills_root=Path("skills"), checkpointer=Checkpointer(),
                                        agent=Agent(state={"structured_response": invalid}))
    with pytest.raises(AgentWorkerError) as raised:
        worker.research(run_id="synthetic-run", request=_request())
    assert raised.value.code == "JOB_RESEARCH_INVALID_RESPONSE"
    assert "PRIVATEKEY" not in str(raised.value)


def test_recursion_limit_has_stable_nonretryable_error() -> None:
    worker = DeepAgentJobResearchWorker(_config(), skills_root=Path("skills"), checkpointer=Checkpointer(),
                                        agent=Agent(error=GraphRecursionError(PRIVATE)))
    with pytest.raises(AgentWorkerError) as raised:
        worker.research(run_id="synthetic-run", request=_request())
    assert raised.value.code == "JOB_RESEARCH_STEP_LIMIT" and not raised.value.retryable
    assert PRIVATE not in str(raised.value)


@pytest.mark.parametrize("observed", [False, True])
def test_actual_agent_requires_native_search_not_merely_valid_json(observed: bool) -> None:
    captured: dict[str, object] = {}
    blocks: list[str | dict[str, object]] = [{"type": "text", "text": PRIVATE}]
    if observed:
        blocks.append({"type": "web_search_call", "status": "completed", "id": "synthetic-search"})
    agent = Agent(state={"structured_response": DRAFT, "messages": [AIMessage(content=blocks)]})

    def factory(**kwargs: object) -> Agent:
        captured.update(kwargs)
        return agent

    worker = DeepAgentJobResearchWorker(_config(), skills_root=Path("skills"), checkpointer=Checkpointer(),
                                        agent_factory=factory)
    try:
        strategy = captured["response_format"]
        assert isinstance(strategy, ProviderStrategy) and strategy.schema is JobResearchDraft
        assert strategy.to_model_kwargs()["response_format"]["json_schema"]["strict"] is True
        assert captured["tools"] == [{"type": "web_search"}]
        model = captured["model"]
        assert isinstance(model, ChatOpenAI)
        assert model.output_version == "responses/v1" and model.max_retries == 0
        if observed:
            assert worker.research(run_id="synthetic-run", request=_request()).sources
        else:
            with pytest.raises(AgentWorkerError) as raised:
                worker.research(run_id="synthetic-run", request=_request())
            assert raised.value.code == "JOB_RESEARCH_SEARCH_UNVERIFIED"
            assert not raised.value.retryable
    finally:
        worker.close()


@pytest.mark.parametrize("suffix", ["", "/", "/responses", "/chat/completions"])
def test_endpoint_normalization_is_not_a_capability_heuristic(suffix: str) -> None:
    assert _base_url("https://synthetic-provider.test/v1" + suffix) == "https://synthetic-provider.test/v1"


def _response(*, output: list[dict[str, object]], status: str = "completed") -> Response:
    return Response.model_validate({
        "id": "synthetic-response", "object": "response", "created_at": 0,
        "model": "synthetic-model", "status": status, "output": output,
        "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
    })


def _message(text: str) -> dict[str, object]:
    return {"id": "synthetic-message", "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}


def test_smoke_rejects_200_without_native_search_and_without_json_boolean() -> None:
    result: dict[str, object] = {}
    with pytest.raises(smoke.SmokeValidationError, match="native_web_search_missing"):
        smoke._validate_response("web_search", _response(output=[_message(PRIVATE)]), result)
    assert result["native_search_observed"] is False and result["url_citations"] == 0
    assert PRIVATE not in json.dumps(result)
    with pytest.raises(smoke.SmokeValidationError, match="structured_output"):
        smoke._validate_response("structured", _response(output=[_message('{"ok":1}')]), {})
    with pytest.raises(smoke.SmokeValidationError, match="response_completed"):
        smoke._validate_response("responses_basic", _response(output=[_message("ok")], status="incomplete"), {})


@pytest.mark.parametrize("stage", ["web_search", "web_search_structured"])
def test_smoke_rejects_unfinished_native_search(stage: smoke.MatrixStage) -> None:
    response = _response(output=[_message('{"ok":true}'), {
        "id": "synthetic-search", "type": "web_search_call", "status": "failed",
        "action": {"type": "search", "query": PRIVATE},
    }])
    result: dict[str, object] = {}
    with pytest.raises(smoke.SmokeValidationError, match="native_web_search_incomplete"):
        smoke._validate_response(stage, response, result)
    assert result["native_search_observed"] is True
    assert result["native_search_completed"] is False
    assert PRIVATE not in json.dumps(result)


def test_matrix_makes_each_probe_once_and_prints_no_provider_body(
    capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    calls: list[httpx.Request] = []

    def reject(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(400, json={"error": {
            "code": "InvalidParameter", "param": "text.format", "type": "invalid_request_error", "message": PRIVATE,
        }})

    class Client(httpx.Client):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(transport=httpx.MockTransport(reject), **kwargs)

    with patch.object(smoke.httpx, "Client", Client), patch.object(smoke, "_full_agent", return_value={"synthetic_only": True}):
        results = smoke.run_matrix(_config(), report_path=tmp_path / "report.json")
    assert json.loads((tmp_path / "report.json").read_text()) == results
    assert len(results) == 6 and len(calls) == 5
    assert all(result["passed"] is False for result in results[:5])
    assert all(result["request_count"] == 1 for result in results[:5])
    assert all(result["provider"]["retryable"] is False for result in results[:5])
    assert PRIVATE not in capsys.readouterr().out


def test_synthetic_full_smoke_checks_durable_report_sources_and_refresh(tmp_path: Path) -> None:
    class SyntheticWorker(DeepAgentJobResearchWorker):
        def __init__(self, config: OpenAICompatibleAgentConfig, **kwargs: object) -> None:
            super().__init__(config, agent=Agent(), **kwargs)

    shapes: list[ProviderRequestStructure] = []
    with patch.object(smoke, "DeepAgentJobResearchWorker", SyntheticWorker):
        result = smoke._full_agent(_config(), shapes.append, work_root=tmp_path)
    assert result == {
        "report_persisted": True, "sources": 1, "findings": 1, "citation_closure": True,
        "company_binding": True, "posting_binding": True, "jd_binding": True,
        "refresh_readback": True, "owner_isolation": True,
    }
    assert not shapes  # This is deterministic persistence coverage, not live compatibility.
    directories = list(tmp_path.glob("research093-smoke-*"))
    assert len(directories) == 1
    assert (directories[0] / "research.sqlite3").is_file()
    assert (directories[0] / "jobs.sqlite3").is_file()


def test_full_smoke_requires_explicit_synthetic_artifact_directory() -> None:
    with pytest.raises(smoke.SmokeValidationError, match="synthetic_work_root_required"):
        smoke._full_agent(_config(), lambda structure: None)


def test_worker_closes_its_http_client_if_agent_construction_fails() -> None:
    clients: list[httpx.Client] = []

    def reject(**kwargs: object) -> Agent:
        model = kwargs["model"]
        assert isinstance(model, ChatOpenAI) and isinstance(model.http_client, httpx.Client)
        clients.append(model.http_client)
        raise ValueError("synthetic-construction-failure")

    with pytest.raises(ValueError, match="synthetic-construction-failure"):
        DeepAgentJobResearchWorker(_config(), skills_root=Path("skills"), checkpointer=Checkpointer(),
                                   agent_factory=reject)
    assert len(clients) == 1 and clients[0].is_closed


def test_failure_chain_is_bounded_and_never_formats_exception_messages() -> None:
    outer = RuntimeError(PRIVATE)
    inner = ValueError(PRIVATE)
    outer.__cause__ = inner
    inner.__cause__ = outer
    result = smoke._failure_metadata(outer)
    assert result["error_chain"] == ["RuntimeError", "ValueError"]
    assert PRIVATE not in json.dumps(result)
