from __future__ import annotations

import argparse
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from career_agent.agent.context_deployment_config import (
    ContextDeploymentConfig,
    ConversationSummaryAgentConfig,
    validate_model_window,
)
from career_agent.agent.openai_compatible_client import (
    AgentConfigurationError,
    OpenAICompatibleAgentConfig,
)
from career_agent.agent.openai_conversation_summary_worker import OpenAIConversationSummaryWorker

PREFIX = "CONVERSATION_SUMMARY_AGENT"


def main_config() -> OpenAICompatibleAgentConfig:
    return OpenAICompatibleAgentConfig(
        endpoint="https://main.example.test/v1/chat/completions",
        api_key="synthetic-main-key", model="synthetic-main", timeout_seconds=120,
    )


def independent_env() -> dict[str, str]:
    return {
        f"{PREFIX}_BASE_URL": "https://summary.example.test/v1",
        f"{PREFIX}_API_KEY": "synthetic-summary-key",
        f"{PREFIX}_MODEL": "synthetic-summary",
    }


def test_production_context_defaults_and_supported_boundaries() -> None:
    config = ContextDeploymentConfig.from_env(environ={})
    assert (config.recent_message_limit, config.summary_batch_size) == (16, 8)
    assert config.compact_occupancy_threshold == 0.75
    for recent, batch, threshold in ((2, 2, 0.7), (64, 32, 0.9)):
        actual = ContextDeploymentConfig.from_env(environ={
            "CONTEXT_RECENT_MESSAGE_LIMIT": str(recent),
            "CONTEXT_SUMMARY_BATCH_SIZE": str(batch),
            "CONTEXT_COMPACT_OCCUPANCY_THRESHOLD": str(threshold),
        })
        assert (actual.recent_message_limit, actual.summary_batch_size) == (recent, batch)
        assert actual.compact_occupancy_threshold == threshold


@pytest.mark.parametrize("key,value", [
    ("CONTEXT_RECENT_MESSAGE_LIMIT", value) for value in ("", "1", "65", "16.5", "bad")
] + [
    ("CONTEXT_SUMMARY_BATCH_SIZE", value) for value in ("", "1", "33", "8.5")
] + [
    ("CONTEXT_COMPACT_OCCUPANCY_THRESHOLD", value) for value in ("", "0.69", "0.91", "NaN", "inf", "-inf", "bad")
] + [
    ("MAIN_AGENT_CONTEXT_WINDOW_TOKENS", value) for value in ("", "2047", "2000001")
])
def test_context_settings_reject_invalid_values_without_echoing_them(key: str, value: str) -> None:
    with pytest.raises(AgentConfigurationError) as raised:
        ContextDeploymentConfig.from_env(environ={key: value})
    assert raised.value.code == "AGENT_CONFIGURATION_INVALID"
    assert key in str(raised.value)


def test_absent_summary_namespace_deliberately_reuses_main_connection_not_timeout() -> None:
    main = main_config()
    summary = ConversationSummaryAgentConfig.from_env(main_config=main, environ={})
    assert summary.provider.endpoint == main.endpoint
    assert summary.provider.api_key == main.api_key
    assert summary.provider.model == main.model
    assert summary.provider.max_input_tokens == main.max_input_tokens
    assert summary.provider.timeout_seconds == 30
    assert main.timeout_seconds == 120
    assert summary.max_output_tokens == 1200


def test_summary_disable_thinking_alone_overrides_main_connection_fallback() -> None:
    main = main_config()
    summary = ConversationSummaryAgentConfig.from_env(
        main_config=main,
        main_context_window_tokens=48_000,
        environ={f"{PREFIX}_DISABLE_THINKING": "true"},
    )
    assert summary.provider.endpoint == main.endpoint
    assert summary.provider.api_key == main.api_key
    assert summary.provider.model == main.model
    assert summary.provider.timeout_seconds == 30
    assert summary.context_window_tokens == 48_000
    assert summary.disable_thinking is True


def test_summary_connection_timeout_and_budgets_are_independent() -> None:
    env = independent_env() | {
        f"{PREFIX}_TIMEOUT_SECONDS": "12.5",
        f"{PREFIX}_MAX_INPUT_TOKENS": "8192",
        f"{PREFIX}_MAX_OUTPUT_TOKENS": "2048",
        f"{PREFIX}_CONTEXT_WINDOW_TOKENS": "10240",
        f"{PREFIX}_DISABLE_THINKING": "true",
        "RESUME_ANALYSIS_AGENT_MODEL": "unrelated-specialist",
    }
    result = ConversationSummaryAgentConfig.from_env(main_config=main_config(), environ=env)
    assert result.provider.endpoint == "https://summary.example.test/v1/chat/completions"
    assert result.provider.api_key == "synthetic-summary-key"
    assert result.provider.model == "synthetic-summary"
    assert result.provider.timeout_seconds == 12.5
    assert result.provider.max_input_tokens == 8192
    assert result.max_output_tokens == 2048
    assert result.context_window_tokens == 10240
    assert result.disable_thinking is True


@pytest.mark.parametrize("env", [
    {f"{PREFIX}_MODEL": "partial"},
    {f"{PREFIX}_TIMEOUT_SECONDS": "30"},
    {f"{PREFIX}_API_KEY": ""},
    {f"{PREFIX}_UNKNOWN": "synthetic-private-value"},
    independent_env() | {f"{PREFIX}_BASE_URL": "http://summary.example.test"},
    independent_env() | {f"{PREFIX}_MODEL": " "},
    independent_env() | {f"{PREFIX}_API_KEY": " "},
])
def test_partial_blank_unknown_or_insecure_summary_configuration_never_falls_back(env: dict[str, str]) -> None:
    with pytest.raises(AgentConfigurationError) as raised:
        ConversationSummaryAgentConfig.from_env(main_config=main_config(), environ=env)
    assert raised.value.code in {"AGENT_CONFIGURATION_MISSING", "AGENT_CONFIGURATION_INVALID"}
    assert "synthetic-private-value" not in str(raised.value)


@pytest.mark.parametrize("suffix,value", [
    ("TIMEOUT_SECONDS", value) for value in ("", "0", "121", "nan", "inf", "bad")
] + [
    ("MAX_INPUT_TOKENS", value) for value in ("", "1023", "2000001", "bad")
] + [
    ("MAX_OUTPUT_TOKENS", value) for value in ("", "255", "16385", "1.5")
] + [
    ("CONTEXT_WINDOW_TOKENS", value) for value in ("", "2047", "2000001")
] + [
    ("DISABLE_THINKING", value) for value in ("", "1", "yes", "bad")
])
def test_invalid_independent_summary_budgets_fail_startup(suffix: str, value: str) -> None:
    with pytest.raises(AgentConfigurationError):
        ConversationSummaryAgentConfig.from_env(
            main_config=main_config(), environ=independent_env() | {f"{PREFIX}_{suffix}": value},
        )


def test_main_and_summary_input_plus_output_must_fit_actual_declared_window() -> None:
    validate_model_window(input_tokens=32000, output_tokens=16384, context_window_tokens=48384, prefix="MAIN_AGENT")
    with pytest.raises(AgentConfigurationError, match="exceed"):
        validate_model_window(input_tokens=32000, output_tokens=16384, context_window_tokens=48383, prefix="MAIN_AGENT")
    with pytest.raises(AgentConfigurationError, match="exceed"):
        ConversationSummaryAgentConfig.from_env(
            main_config=main_config(), environ=independent_env() | {f"{PREFIX}_CONTEXT_WINDOW_TOKENS": "32000"},
        )


def test_production_factory_passes_independent_summary_and_context_settings(tmp_path: Path) -> None:
    from career_agent import cli

    class FactoryObserved(Exception):
        pass

    def capture_context(*args: object, **kwargs: object) -> None:
        worker = kwargs["summary_worker"]
        assert isinstance(worker, OpenAIConversationSummaryWorker)
        assert worker._config.model == "synthetic-summary"
        assert worker._config.timeout_seconds == 11
        assert worker._max_output_tokens == 2048
        assert worker._disable_thinking is True
        assert kwargs["recent_message_limit"] == 20
        assert kwargs["summary_batch_size"] == 10
        assert kwargs["compact_occupancy_threshold"] == 0.8
        raise FactoryObserved

    env = independent_env() | {
        "MAIN_AGENT_BASE_URL": "https://main.example.test/v1",
        "MAIN_AGENT_API_KEY": "synthetic-main-key", "MAIN_AGENT_MODEL": "synthetic-main",
        "CONTEXT_RECENT_MESSAGE_LIMIT": "20", "CONTEXT_SUMMARY_BATCH_SIZE": "10",
        "CONTEXT_COMPACT_OCCUPANCY_THRESHOLD": "0.8",
        f"{PREFIX}_TIMEOUT_SECONDS": "11", f"{PREFIX}_MAX_OUTPUT_TOKENS": "2048",
        f"{PREFIX}_DISABLE_THINKING": "true",
    }
    args = argparse.Namespace(
        main_agent_timeout_seconds=120,
        context_store=str(tmp_path / "context.sqlite3"), resume_store=str(tmp_path / "resume.sqlite3"),
    )
    with (
        patch.dict(os.environ, env, clear=True),
        patch("career_agent.agent.context_deployment_config.load_dotenv"),
        patch("career_agent.agent.openai_compatible_client.load_dotenv"),
        patch.object(cli, "ContextManager", side_effect=capture_context),
        pytest.raises(FactoryObserved),
    ):
        cli.build_main_agent_runtime(args)
