from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from openai import OpenAI

from career_agent.agent.openai_compatible_client import (
    AgentWorkerError,
    OpenAICompatibleAgentConfig,
)
from career_agent.agent.openai_resume_analysis_worker import OpenAIResumeAnalysisWorker
from career_agent.agent.resume_analysis_contracts import (
    ExtractedCareerEvidence,
    ExtractedCareerRecord,
    NumberedCareerEvidence,
    NumberedCareerRecord,
    NumberedResumeAnalysisResult,
    ResumeAnalysisResult,
)
from career_agent.agent.structured_chat_completions import strict_json_schema
from career_agent.services.resume_analysis import (
    ResumeAnalysisNotFoundError,
    ResumeAnalysisNotPendingError,
    ResumeAnalysisService,
    ResumeAnalysisWorkerNotCommittedError,
    ResumeVersionNotFoundError,
)
from career_agent.storage.career_history import CareerHistoryStore
from career_agent.storage.resume_analysis import SQLiteResumeAnalysisDraftStore
from career_agent.storage.resumes import ResumeStore

from test_resume_093_extraction import document, synthetic_pdf


SOURCE = "示例公司 后端工程师 2022.03-2024.05\n负责检索系统"
CONFIG = OpenAICompatibleAgentConfig(
    endpoint="https://provider.invalid/v1/chat/completions",
    api_key="synthetic-not-a-credential",
    model="synthetic-model",
    timeout_seconds=2,
)


def result() -> ResumeAnalysisResult:
    return ResumeAnalysisResult(
        records=(
            ExtractedCareerRecord(
                record_type="work",
                organization="示例公司",
                title="后端工程师",
                start_year=2022,
                start_month=3,
                end_year=2024,
                end_month=5,
                source_locator="page 1, paragraph 1",
                source_quote=SOURCE.splitlines()[0],
                evidence=(
                    ExtractedCareerEvidence(
                        claim="负责检索系统",
                        source_locator="page 1, paragraph 2",
                        source_quote="负责检索系统",
                    ),
                ),
            ),
        ),
    )


def numbered_result() -> NumberedResumeAnalysisResult:
    return NumberedResumeAnalysisResult(
        records=(
            NumberedCareerRecord(
                record_type="work",
                organization="示例公司",
                title="后端工程师",
                start_year=2022,
                start_month=3,
                end_year=2024,
                end_month=5,
                source_locator=1,
                evidence=(
                    NumberedCareerEvidence(
                        claim="负责检索系统",
                        source_locator=2,
                    ),
                ),
            ),
        ),
    )


class ChatHarness:
    """Exercise SDK serialization/error handling without a real network request."""

    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []
        self.paths: list[str] = []
        self.output: str | None = numbered_result().model_dump_json()
        self.finish_reason = "stop"
        self.refusal: str | None = None
        self.completion_tokens: int | None = None
        self.reasoning_tokens: int | None = None
        self.status = 200
        self.error_body: dict[str, object] = {
            "error": {
                "code": "InvalidParameter",
                "param": "response_format.json_schema",
                "type": "invalid_request_error",
                "message": "DO_NOT_LOG_PROVIDER_BODY",
            }
        }
        self.transport_error: httpx.TransportError | None = None
        self.client = OpenAI(
            api_key=CONFIG.api_key,
            base_url="https://provider.invalid/v1",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(self.respond)),
        )

    def respond(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        self.paths.append(request.url.path)
        if self.transport_error is not None:
            raise self.transport_error
        if self.status != 200:
            return httpx.Response(self.status, json=self.error_body)
        payload: dict[str, object] = {
            "id": "synthetic-completion",
            "created": 0,
            "object": "chat.completion",
            "model": CONFIG.model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": self.finish_reason,
                    "message": {
                        "role": "assistant",
                        "content": self.output,
                        "refusal": self.refusal,
                    },
                }
            ],
        }
        if self.completion_tokens is not None:
            payload["usage"] = {
                "prompt_tokens": 100,
                "completion_tokens": self.completion_tokens,
                "total_tokens": 100 + self.completion_tokens,
                "completion_tokens_details": {
                    "reasoning_tokens": self.reasoning_tokens or 0
                },
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
            replace(CONFIG, max_input_tokens=max_input_tokens),
            client=self.client,
            disable_thinking=disable_thinking,
        )


@pytest.fixture
def chat() -> Iterator[ChatHarness]:
    harness = ChatHarness()
    try:
        yield harness
    finally:
        harness.client.close()


@pytest.mark.parametrize("is_pdf", [False, True])
def test_real_sdk_chat_json_schema_request_has_only_located_text(
    chat: ChatHarness, is_pdf: bool
) -> None:
    raw = synthetic_pdf((SOURCE,)) if is_pdf else SOURCE.encode()
    analyzed = chat.worker().analyze(document(raw, "pdf" if is_pdf else "text"))
    assert analyzed == result()
    assert chat.paths == ["/v1/chat/completions"]
    assert len(chat.requests) == 1
    request = chat.requests[0]
    assert set(request) == {"model", "messages", "response_format", "max_tokens"}
    assert request["max_tokens"] == 8192
    assert request["model"] == CONFIG.model
    messages = request["messages"]
    assert isinstance(messages, list)
    assert [message["role"] for message in messages] == ["system", "user"]
    assert all(isinstance(message["content"], str) for message in messages)
    assert "untrusted data" in messages[0]["content"]
    assert "explicit user confirmation" in messages[0]["content"]
    assert json.loads(messages[1]["content"]) == {
        "source_paragraphs": [
            {"paragraph_number": 1, "source_text": SOURCE.splitlines()[0]},
            {"paragraph_number": 2, "source_text": SOURCE.splitlines()[1]},
        ]
    }
    assert request["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "resume_analysis_result",
            "strict": True,
            "schema": strict_json_schema(
                NumberedResumeAnalysisResult,
                field_enums={"source_locator": (1, 2)},
            ),
        },
    }
    wire = json.dumps(request)
    assert "input_file" not in wire
    assert "base64" not in wire
    assert "file_data" not in wire


def test_explicit_disable_thinking_is_sent_without_model_name_guessing(
    chat: ChatHarness,
) -> None:
    chat.worker(disable_thinking=True).analyze(document(SOURCE.encode(), "text"))
    assert chat.requests[0]["enable_thinking"] is False


def test_strict_schema_requires_all_fields_without_mutating_persisted_defaults() -> (
    None
):
    def verify(node: object) -> None:
        if isinstance(node, dict):
            assert "default" not in node
            if "properties" in node:
                assert node["required"] == list(node["properties"])
                assert node["additionalProperties"] is False
            for value in node.values():
                verify(value)
        elif isinstance(node, list):
            for item in node:
                verify(item)

    schema = strict_json_schema(
        NumberedResumeAnalysisResult,
        field_enums={"source_locator": (1, 2)},
    )
    verify(schema)
    assert schema["$defs"]["NumberedCareerRecord"]["properties"][
        "source_locator"
    ]["enum"] == [1, 2]
    assert schema["$defs"]["NumberedCareerEvidence"]["properties"][
        "source_locator"
    ]["enum"] == [1, 2]
    assert ResumeAnalysisResult().records == ()
    assert result().records[0].is_current is False


def test_large_locator_set_omits_enum_but_local_lookup_still_rejects_unissued_number(
    chat: ChatHarness,
) -> None:
    source = "\n".join(f"paragraph {number}" for number in range(1, 301))
    record = numbered_result().records[0].model_dump(mode="json")
    record["source_locator"] = 301
    chat.output = json.dumps(
        {"records": [record], "clarification_questions": [], "warnings": []}
    )

    with pytest.raises(AgentWorkerError) as raised:
        chat.worker().analyze(document(source.encode(), "text"))
    assert raised.value.code == "RESUME_ANALYSIS_INVALID_EVIDENCE"

    schema = chat.requests[0]["response_format"]["json_schema"]["schema"]
    assert "enum" not in schema["$defs"]["NumberedCareerRecord"]["properties"][
        "source_locator"
    ]
    assert "enum" not in schema["$defs"]["NumberedCareerEvidence"]["properties"][
        "source_locator"
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_locator", 99),
    ],
)
def test_unissued_locator_is_rejected(
    chat: ChatHarness, field: str, value: object
) -> None:
    record = numbered_result().records[0].model_dump(mode="json")
    record[field] = value
    chat.output = json.dumps(
        {"records": [record], "clarification_questions": [], "warnings": []}
    )
    with pytest.raises(AgentWorkerError) as caught:
        chat.worker().analyze(document(SOURCE.encode(), "text"))
    assert caught.value.code == "RESUME_ANALYSIS_INVALID_EVIDENCE"
    assert caught.value.detail == "source_quote_or_locator_mismatch"
    assert str(value) not in str(caught.value)


def test_nested_evidence_is_also_validated(chat: ChatHarness) -> None:
    record = numbered_result().records[0].model_dump(mode="json")
    record["evidence"] = [
        {
            "claim": "fabrication",
            "source_locator": 99,
        }
    ]
    chat.output = json.dumps(
        {"records": [record], "clarification_questions": [], "warnings": []}
    )
    with pytest.raises(AgentWorkerError, match="evidence") as caught:
        chat.worker().analyze(document(SOURCE.encode(), "text"))
    assert caught.value.code == "RESUME_ANALYSIS_INVALID_EVIDENCE"


def test_persisted_contract_still_rejects_nonverbatim_quotes() -> None:
    original = result()
    record = original.records[0].model_copy(
        update={"source_quote": "invented achievement"}
    )
    with pytest.raises(ValueError, match="mismatch"):
        original.model_copy(update={"records": (record,)}).validate_source_quotes(
            {
                "page 1, paragraph 1": SOURCE.splitlines()[0],
                "page 1, paragraph 2": SOURCE.splitlines()[1],
            }
        )


@pytest.mark.parametrize(
    "output",
    [
        "not-json",
        '{"records": [], "records": [], "warnings": ["ambiguous"], "clarification_questions": []}',
        '{"records": [], "warnings": [], "clarification_questions": [], "private-extra-key": true}',
        '{"records": [], "warnings": ["clarify"]}',
        '{"records": [], "warnings": [" "], "clarification_questions": []}',
        '{"records": [], "warnings": [], "clarification_questions": []}',
        '{"records": NaN, "warnings": [], "clarification_questions": []}',
        "[" * 1100 + "]" * 1100,
        "x" * 128_001,
        "\ud800",
    ],
)
def test_invalid_wire_output_fails_closed_with_safe_detail(
    chat: ChatHarness, output: str
) -> None:
    chat.output = output
    with pytest.raises(AgentWorkerError) as caught:
        chat.worker().analyze(document(SOURCE.encode(), "text"))
    assert caught.value.code == "RESUME_ANALYSIS_INVALID_RESPONSE"
    assert "private-extra-key" not in str(caught.value.detail)


@pytest.mark.parametrize(
    "field,value",
    [
        ("start_year", "2022"),
        ("is_current", 0),
        ("source_quote", "model-authored quote is forbidden"),
        ("user_id", "malicious-user"),
        ("evidence", None),
    ],
)
def test_output_types_and_system_owned_fields_are_strict(
    chat: ChatHarness, field: str, value: object
) -> None:
    record = numbered_result().records[0].model_dump(mode="json")
    record[field] = value
    chat.output = json.dumps(
        {"records": [record], "clarification_questions": [], "warnings": []}
    )
    with pytest.raises(AgentWorkerError) as caught:
        chat.worker().analyze(document(SOURCE.encode(), "text"))
    assert caught.value.code == "RESUME_ANALYSIS_INVALID_RESPONSE"
    assert "malicious-user" not in str(caught.value.detail)


def test_nullable_default_fields_are_still_required_on_wire(chat: ChatHarness) -> None:
    record = numbered_result().records[0].model_dump(mode="json")
    del record["end_month"]
    chat.output = json.dumps(
        {"records": [record], "clarification_questions": [], "warnings": []}
    )
    with pytest.raises(AgentWorkerError) as caught:
        chat.worker().analyze(document(SOURCE.encode(), "text"))
    assert caught.value.code == "RESUME_ANALYSIS_INVALID_RESPONSE"
    assert caught.value.detail == "missing_required_fields"


@pytest.mark.parametrize("finish_reason", ["length", "content_filter", "tool_calls"])
def test_incomplete_completion_never_saves_partial_facts(
    chat: ChatHarness, finish_reason: str
) -> None:
    chat.finish_reason = finish_reason
    with pytest.raises(AgentWorkerError) as caught:
        chat.worker().analyze(document(SOURCE.encode(), "text"))
    assert caught.value.code == "RESUME_ANALYSIS_INCOMPLETE_RESPONSE"


@pytest.mark.parametrize("output", [None, "", "  "])
def test_empty_completion_is_not_a_draft(chat: ChatHarness, output: str | None) -> None:
    chat.output = output
    with pytest.raises(AgentWorkerError) as caught:
        chat.worker().analyze(document(SOURCE.encode(), "text"))
    assert caught.value.code == "RESUME_ANALYSIS_EMPTY_RESPONSE"


def test_reasoning_tokens_exhausting_output_budget_is_incomplete(
    chat: ChatHarness,
) -> None:
    chat.output = None
    chat.finish_reason = "stop"
    chat.completion_tokens = 8192
    chat.reasoning_tokens = 8010
    with pytest.raises(AgentWorkerError) as caught:
        chat.worker().analyze(document(SOURCE.encode(), "text"))
    assert caught.value.code == "RESUME_ANALYSIS_INCOMPLETE_RESPONSE"


def test_refusal_does_not_expose_model_text(chat: ChatHarness) -> None:
    chat.refusal = "private-refusal"
    with pytest.raises(AgentWorkerError) as caught:
        chat.worker().analyze(document(SOURCE.encode(), "text"))
    assert caught.value.code == "RESUME_ANALYSIS_REFUSED"
    assert "private-refusal" not in str(caught.value)


@pytest.mark.parametrize(
    "status,retryable",
    [
        (400, False),
        (401, False),
        (403, False),
        (429, True),
        (502, True),
        (503, True),
        (504, True),
    ],
)
def test_provider_errors_retain_safe_metadata_without_retries(
    chat: ChatHarness, status: int, retryable: bool
) -> None:
    chat.status = status
    with pytest.raises(AgentWorkerError) as caught:
        chat.worker().analyze(document(SOURCE.encode(), "text"))
    error = caught.value
    suffix = "RATE_LIMITED" if status == 429 else f"REJECTED_{status}"
    assert error.code == f"RESUME_ANALYSIS_{suffix}"
    assert error.retryable is retryable
    assert error.provider is not None
    assert error.provider.status == status
    assert error.provider.param == "response_format.json_schema"
    assert error.provider.code == "InvalidParameter"
    assert error.provider.type == "invalid_request_error"
    assert "DO_NOT_LOG_PROVIDER_BODY" not in str(error)
    assert "DO_NOT_LOG_PROVIDER_BODY" not in json.dumps(error.provider.as_dict())
    assert len(chat.requests) == 1


@pytest.mark.parametrize("timeout", [False, True])
def test_transport_errors_are_safe_and_retryable(
    chat: ChatHarness, timeout: bool
) -> None:
    chat.transport_error = (
        httpx.ReadTimeout("private") if timeout else httpx.ConnectError("private")
    )
    with pytest.raises(AgentWorkerError) as caught:
        chat.worker().analyze(document(SOURCE.encode(), "text"))
    suffix = "TIMEOUT" if timeout else "TRANSPORT_ERROR"
    assert caught.value.code == f"RESUME_ANALYSIS_{suffix}"
    assert caught.value.retryable is True
    assert "private" not in str(caught.value)
    assert len(chat.requests) == 1


def test_complete_request_budget_includes_schema_framing_and_output(
    chat: ChatHarness,
) -> None:
    with pytest.raises(AgentWorkerError) as caught:
        chat.worker(max_input_tokens=9000).analyze(document(b"Engineer", "text"))
    assert caught.value.code == "RESUME_ANALYSIS_TOKEN_BUDGET_EXCEEDED"
    assert chat.requests == []


@pytest.mark.parametrize(
    "raw", [synthetic_pdf((None,)), b"%PDF-damaged", b"a" * 60_001]
)
def test_local_failures_never_call_provider(chat: ChatHarness, raw: bytes) -> None:
    with pytest.raises(AgentWorkerError):
        chat.worker().analyze(
            document(raw, "pdf" if raw.startswith(b"%PDF") else "text")
        )
    assert chat.requests == []


def test_formal_analysis_persists_draft_then_requires_owned_one_time_confirmation(
    chat: ChatHarness, tmp_path: Path
) -> None:
    path = tmp_path / "synthetic.sqlite3"
    store = ResumeStore(path)
    role = store.create_target_role(user_id="owner", title="Engineer", priority=1)
    _, version = store.import_document(
        user_id="owner",
        name="Synthetic",
        target_role_id=role.id,
        content=synthetic_pdf((SOURCE,)),
        document_format="pdf",
    )
    history = CareerHistoryStore(path)
    drafts = SQLiteResumeAnalysisDraftStore(path)
    service = ResumeAnalysisService(store, chat.worker(), drafts, history)
    with pytest.raises(ResumeVersionNotFoundError):
        service.analyze_version(user_id="other", resume_version_id=version.id)
    assert chat.requests == []
    draft = service.analyze_version(user_id="owner", resume_version_id=version.id)
    assert draft.status == "pending"
    assert history.list_records(user_id="owner") == ()
    assert history.list_evidence(user_id="owner") == ()
    # Reopen every store to demonstrate that the confirmation is durable, not
    # dependent on the just-returned object or another provider request.
    reopened = ResumeAnalysisService(
        ResumeStore(path),
        chat.worker(),
        SQLiteResumeAnalysisDraftStore(path),
        CareerHistoryStore(path),
    )
    assert reopened.get_analysis(user_id="owner", analysis_id=draft.id) == draft
    with pytest.raises(ResumeAnalysisNotFoundError):
        reopened.confirm_analysis(user_id="other", analysis_id=draft.id)
    imported = reopened.confirm_analysis(user_id="owner", analysis_id=draft.id)
    assert len(imported.records) == 1
    assert len(imported.evidence) >= 1
    assert all(item.verification_status == "confirmed" for item in imported.evidence)
    assert all(
        item.source_resume_version_id == version.id for item in imported.evidence
    )
    assert all(item.origin == "resume_extraction" for item in imported.evidence)
    assert draft.result.records[0].source_quote == SOURCE.splitlines()[0]
    assert draft.result.records[0].evidence[0].source_quote == SOURCE.splitlines()[1]
    assert history.list_records(user_id="owner") == imported.records
    assert history.list_records(user_id="other") == ()
    with pytest.raises(ResumeAnalysisNotPendingError):
        reopened.confirm_analysis(user_id="owner", analysis_id=draft.id)
    assert len(chat.requests) == 1


def test_worker_failure_cannot_create_draft_or_history(
    chat: ChatHarness, tmp_path: Path
) -> None:
    path = tmp_path / "synthetic.sqlite3"
    store = ResumeStore(path)
    role = store.create_target_role(user_id="owner", title="Engineer", priority=1)
    _, version = store.import_document(
        user_id="owner",
        name="Synthetic",
        target_role_id=role.id,
        content=SOURCE.encode(),
        document_format="text",
    )
    chat.output = "invalid"
    drafts = SQLiteResumeAnalysisDraftStore(path)
    history = CareerHistoryStore(path)
    service = ResumeAnalysisService(store, chat.worker(), drafts, history)
    with pytest.raises(ResumeAnalysisWorkerNotCommittedError):
        service.analyze_version(user_id="owner", resume_version_id=version.id)
    assert history.list_records(user_id="owner") == ()
    assert history.list_evidence(user_id="owner") == ()
    # Query only this disposable test database, not a production database.
    with sqlite3.connect(path) as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM resume_analysis_drafts"
            ).fetchone()[0]
            == 0
        )
