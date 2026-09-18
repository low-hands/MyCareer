from __future__ import annotations

from collections.abc import Iterator
import json
import logging
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from openai import OpenAI

from compaction_smoke_093 import main, probe_summary_provider
from career_agent.agent.context_deployment_config import ConversationSummaryAgentConfig
from career_agent.agent.openai_compatible_client import OpenAICompatibleAgentConfig


@pytest.fixture(autouse=True)
def restore_process_logging_level() -> Iterator[None]:
    previous = logging.root.manager.disable
    try:
        yield
    finally:
        logging.disable(previous)


def synthetic_config() -> ConversationSummaryAgentConfig:
    return ConversationSummaryAgentConfig(
        provider=OpenAICompatibleAgentConfig(
            endpoint="https://synthetic.test/v1/chat/completions",
            api_key="synthetic-key",
            model="synthetic-summary",
            timeout_seconds=120,
        )
    )


def completion(content: str) -> httpx.Response:
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
                    "message": {"role": "assistant", "content": content},
                }
            ],
        },
    )


@pytest.mark.parametrize("minimal", ['{"ok":true}', '{"ok":1}', "not-json"])
def test_provider_probe_requires_valid_structured_content_not_just_http_200(
    minimal: str,
) -> None:
    requests: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) < 3:
            return completion(minimal)
        return completion(json.dumps({
            "user_goals": [], "confirmed_decisions": ["Use SQLite"],
            "unresolved_questions": [], "active_constraints": [],
            "long_term_memory_candidates": [],
        }))

    with OpenAI(
        api_key="synthetic-key", base_url="https://synthetic.test/v1", max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as client, patch("compaction_smoke_093.OpenAI", return_value=client) as factory:
        report = probe_summary_provider(synthetic_config())
    assert factory.call_args.kwargs["max_retries"] == 0
    assert report["timeout_seconds_per_request"] == 20
    assert report["passed"] is (minimal == '{"ok":true}')
    assert len(requests) == 3
    assert "response_format" not in requests[0]
    assert requests[1]["response_format"]["json_schema"]["name"] == "synthetic_probe"
    assert requests[2]["response_format"]["json_schema"]["name"] == "conversation_summary"
    serialized = json.dumps(report)
    assert "Use SQLite" not in serialized
    assert "synthetic-key" not in serialized
    assert "not-json" not in serialized
    assert report["automatic_fallback_used"] is False


def test_provider_probe_records_only_safe_errors_with_one_attempt_per_shape() -> None:
    calls = 0

    def rejected(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"error": {
            "code": "InvalidParameter", "param": "response_format.json_schema",
            "type": "invalid_request_error", "message": "synthetic-private-provider-body",
        }})

    with OpenAI(
        api_key="synthetic-key", base_url="https://synthetic.test/v1", max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(rejected)),
    ) as client, patch("compaction_smoke_093.OpenAI", return_value=client):
        report = probe_summary_provider(synthetic_config())
    assert calls == 3
    assert report["passed"] is False
    for result in report["results"]:
        assert result["status"] == 400
        assert result["code"] == "InvalidParameter"
        assert result["error_code"] == "CONVERSATION_SUMMARY_REJECTED_400"
        assert result["retryable"] is False
    assert "synthetic-private-provider-body" not in json.dumps(report)


def test_smoke_cli_requires_explicit_scratch_root_instead_of_os_temporary_files() -> None:
    with patch("sys.argv", ["compaction_smoke_093.py"]), pytest.raises(SystemExit) as raised:
        main()
    assert raised.value.code == 2


def test_smoke_cli_retains_synthetic_artifacts_under_requested_root(tmp_path: Path) -> None:
    output = tmp_path / "metrics.json"
    with patch("sys.argv", [
        "compaction_smoke_093.py", "--work-dir", str(tmp_path), "--output", str(output),
    ]):
        assert main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["exact_private_trajectory_replay"] is False
    assert len(list(tmp_path.glob("compaction-093-synthetic-*/*.sqlite3"))) == 2
