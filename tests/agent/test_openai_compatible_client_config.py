import pytest

from career_agent.agent.openai_compatible_client import (
    AgentConfigurationError,
    OpenAICompatibleAgentConfig,
)


def _fallback() -> OpenAICompatibleAgentConfig:
    return OpenAICompatibleAgentConfig(
        endpoint="https://example.test/v1/chat/completions",
        api_key="secret",
        model="baseline",
        timeout_seconds=120.0,
    )


def test_model_lane_overlays_model_and_timeout_without_copying_credentials() -> None:
    fallback = _fallback()
    result = OpenAICompatibleAgentConfig.with_env_overrides(
        prefix="MOCK_INTERVIEW_AGENT",
        fallback=fallback,
        environ={
            "MOCK_INTERVIEW_AGENT_MODEL": "fast-model",
            "MOCK_INTERVIEW_AGENT_TIMEOUT_SECONDS": "45",
        },
    )

    assert result.model == "fast-model"
    assert result.timeout_seconds == 45.0
    assert result.endpoint == fallback.endpoint
    assert result.api_key == fallback.api_key


def test_unset_model_lane_returns_same_fallback() -> None:
    fallback = _fallback()

    assert OpenAICompatibleAgentConfig.with_env_overrides(
        prefix="MOCK_INTERVIEW_AGENT",
        fallback=fallback,
        environ={},
    ) is fallback


@pytest.mark.parametrize("value", ["0", "121", "not-a-number"])
def test_model_lane_rejects_invalid_timeout(value: str) -> None:
    with pytest.raises(AgentConfigurationError) as caught:
        OpenAICompatibleAgentConfig.with_env_overrides(
            prefix="MOCK_INTERVIEW_AGENT",
            fallback=_fallback(),
            environ={"MOCK_INTERVIEW_AGENT_TIMEOUT_SECONDS": value},
        )

    assert caught.value.code == "AGENT_CONFIGURATION_INVALID"


def test_model_lane_accepts_a_full_independent_connection() -> None:
    result = OpenAICompatibleAgentConfig.with_env_overrides(
        prefix="RESUME_TAILORING_AGENT",
        fallback=_fallback(),
        environ={
            "RESUME_TAILORING_AGENT_BASE_URL": "https://official.test/v1",
            "RESUME_TAILORING_AGENT_API_KEY": "official-secret",
            "RESUME_TAILORING_AGENT_MODEL": "official-fast",
            "RESUME_TAILORING_AGENT_TIMEOUT_SECONDS": "30",
        },
    )

    assert result.endpoint == "https://official.test/v1/chat/completions"
    assert result.api_key == "official-secret"
    assert result.model == "official-fast"
    assert result.timeout_seconds == 30.0


def test_model_lane_rejects_partial_independent_connection() -> None:
    with pytest.raises(AgentConfigurationError) as caught:
        OpenAICompatibleAgentConfig.with_env_overrides(
            prefix="MOCK_INTERVIEW_AGENT",
            fallback=_fallback(),
            environ={"MOCK_INTERVIEW_AGENT_BASE_URL": "https://other.test/v1"},
        )

    assert caught.value.code == "AGENT_CONFIGURATION_MISSING"
