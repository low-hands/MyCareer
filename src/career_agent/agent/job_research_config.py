"""Explicit, independent research provider configuration; no capability guessing."""
from __future__ import annotations

from dataclasses import replace
import math
import os
from typing import Mapping

from dotenv import load_dotenv

from career_agent.agent.openai_compatible_client import (
    AgentConfigurationError,
    OpenAICompatibleAgentConfig,
)


_PREFIX = "JOB_RESEARCH_AGENT"


def job_research_config_from_env(
    *,
    environ: Mapping[str, str] | None = None,
    fallback: OpenAICompatibleAgentConfig | None = None,
) -> OpenAICompatibleAgentConfig:
    """Use an independent endpoint when explicitly configured.

    The application may supply its legacy specialist config as ``fallback``.
    Any nonempty JOB_RESEARCH_AGENT_* setting opts into independent config:
    require the full URL/key/model trio rather than mixing credentials or
    silently falling back after a configuration error. Passing configuration
    does not certify native search support; run the six-stage smoke first.
    """
    if environ is None:
        load_dotenv()
        environ = os.environ
    configured = any(
        key.startswith(f"{_PREFIX}_") and value.strip()
        for key, value in environ.items()
    )
    if not configured and fallback is not None:
        return fallback
    config = OpenAICompatibleAgentConfig.from_env(environ=environ, prefix=_PREFIX)
    try:
        timeout = float(environ.get(f"{_PREFIX}_TIMEOUT_SECONDS", "30"))
    except ValueError as error:
        raise AgentConfigurationError(
            "AGENT_CONFIGURATION_INVALID",
            "JOB_RESEARCH_AGENT_TIMEOUT_SECONDS must be between 1 and 120 seconds.",
        ) from error
    if not math.isfinite(timeout) or not 1 <= timeout <= 120:
        raise AgentConfigurationError(
            "AGENT_CONFIGURATION_INVALID",
            "JOB_RESEARCH_AGENT_TIMEOUT_SECONDS must be between 1 and 120 seconds.",
        )
    return replace(config, timeout_seconds=timeout)
