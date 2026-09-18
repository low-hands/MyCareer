import pytest

from career_agent.agent.deepagent_job_research_worker import _base_url
from career_agent.agent.job_research_config import job_research_config_from_env
from career_agent.agent.openai_compatible_client import (
    AgentConfigurationError,
    OpenAICompatibleAgentConfig,
)


ENV = {
    "JOB_RESEARCH_AGENT_BASE_URL": "https://research-provider.test/v1",
    "JOB_RESEARCH_AGENT_API_KEY": "synthetic-key",
    "JOB_RESEARCH_AGENT_MODEL": "synthetic-model",
}
FALLBACK = OpenAICompatibleAgentConfig(
    endpoint="https://legacy-provider.test/v1/chat/completions",
    api_key="synthetic-legacy-key", model="synthetic-legacy-model",
)


@pytest.mark.parametrize("suffix", ["", "/", "/responses", "/responses/", "/chat/completions"])
def test_explicit_independent_config_accepts_base_and_protocol_urls(suffix: str) -> None:
    config = job_research_config_from_env(environ={
        **ENV, "JOB_RESEARCH_AGENT_BASE_URL": "https://research-provider.test/v1" + suffix,
        "JOB_RESEARCH_AGENT_TIMEOUT_SECONDS": "45",
    }, fallback=FALLBACK)
    assert config.endpoint == "https://research-provider.test/v1/chat/completions"
    assert _base_url(config.endpoint) == "https://research-provider.test/v1"
    assert config.api_key == ENV["JOB_RESEARCH_AGENT_API_KEY"]
    assert config.model == ENV["JOB_RESEARCH_AGENT_MODEL"]
    assert config.timeout_seconds == 45


def test_legacy_fallback_is_explicit_and_only_when_independent_config_is_absent() -> None:
    assert job_research_config_from_env(environ={}, fallback=FALLBACK) is FALLBACK
    assert job_research_config_from_env(environ={"JOB_RESEARCH_AGENT_MODEL": " "}, fallback=FALLBACK) is FALLBACK
    with pytest.raises(AgentConfigurationError) as raised:
        job_research_config_from_env(environ={})
    assert raised.value.code == "AGENT_CONFIGURATION_MISSING"


@pytest.mark.parametrize("key", list(ENV) + ["JOB_RESEARCH_AGENT_TIMEOUT_SECONDS"])
def test_partial_independent_config_never_reuses_legacy_credentials(key: str) -> None:
    with pytest.raises(AgentConfigurationError) as raised:
        job_research_config_from_env(environ={key: "synthetic-partial-value"}, fallback=FALLBACK)
    assert raised.value.code == "AGENT_CONFIGURATION_MISSING"
    assert "synthetic-partial-value" not in str(raised.value)


@pytest.mark.parametrize("value", ["0", "-1", "121", "nan", "inf", "", "invalid"])
def test_independent_timeout_is_finite_and_bounded(value: str) -> None:
    with pytest.raises(AgentConfigurationError) as raised:
        job_research_config_from_env(environ={**ENV, "JOB_RESEARCH_AGENT_TIMEOUT_SECONDS": value})
    assert raised.value.code == "AGENT_CONFIGURATION_INVALID"
