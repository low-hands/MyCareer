"""Production wiring honours the 093 specialist settings.

The workers read ``RESUME_ANALYSIS_AGENT_API_PROTOCOL`` and
``JOB_RESEARCH_AGENT_*`` correctly in isolation; these tests build the real
runtime graph up to each service and check the worker it is handed.
"""

from __future__ import annotations

from contextlib import ExitStack
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from career_agent import cli
from career_agent.agent.openai_compatible_client import AgentConfigurationError


class Observed(Exception):
    pass


BASE_ENV = {
    "MAIN_AGENT_BASE_URL": "https://main.example.test/v1",
    "MAIN_AGENT_API_KEY": "synthetic-main-key",
    "MAIN_AGENT_MODEL": "synthetic-main",
    "RESUME_ANALYSIS_AGENT_BASE_URL": "https://specialist.example.test/v1",
    "RESUME_ANALYSIS_AGENT_API_KEY": "synthetic-specialist-key",
    "RESUME_ANALYSIS_AGENT_MODEL": "synthetic-specialist",
}


def _args(tmp_path: Path):
    args = cli.build_parser().parse_args(
        ["chat", "--user-id", "u1", "--session-id", "s1", "--message", "hi"]
    )
    for name, value in vars(args).items():
        if name.endswith("_store"):
            setattr(args, name, str(tmp_path / Path(str(value)).name))
    args.agent_timeout_seconds = 42
    return args


def _build(tmp_path: Path, env: dict[str, str], **patches):
    with ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, env, clear=True))
        for module in ("context_deployment_config", "openai_compatible_client", "job_research_config"):
            stack.enter_context(patch(f"career_agent.agent.{module}.load_dotenv"))
        for name, effect in patches.items():
            stack.enter_context(patch.object(cli, name, side_effect=effect))
        cli.build_main_agent_runtime(_args(tmp_path))


def _research_config(tmp_path: Path, env: dict[str, str]):
    seen = []

    def capture(config, **kwargs):
        seen.append(config)
        raise Observed

    with pytest.raises(Observed):
        _build(tmp_path, env, DeepAgentJobResearchWorker=capture)
    return seen[0]


def test_research_reuses_the_specialist_endpoint_when_not_configured(tmp_path):
    config = _research_config(tmp_path, BASE_ENV)
    assert config.model == "synthetic-specialist"
    assert config.timeout_seconds == 42


def test_research_uses_its_own_endpoint_when_configured(tmp_path):
    config = _research_config(tmp_path, BASE_ENV | {
        "JOB_RESEARCH_AGENT_BASE_URL": "https://research.example.test/v1",
        "JOB_RESEARCH_AGENT_API_KEY": "synthetic-research-key",
        "JOB_RESEARCH_AGENT_MODEL": "synthetic-research",
        "JOB_RESEARCH_AGENT_TIMEOUT_SECONDS": "25",
    })
    assert config.model == "synthetic-research"
    assert config.api_key == "synthetic-research-key"
    assert config.timeout_seconds == 25


def test_a_partial_research_group_fails_startup_instead_of_mixing(tmp_path):
    with pytest.raises(AgentConfigurationError):
        _build(tmp_path, BASE_ENV | {"JOB_RESEARCH_AGENT_MODEL": "only-model"})
