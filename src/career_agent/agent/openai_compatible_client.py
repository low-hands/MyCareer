from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping
from urllib.parse import urlsplit

from dotenv import load_dotenv


class AgentWorkerError(Exception):
    def __init__(self, code: str, message: str, *, retryable: bool = False, detail: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.detail = detail


class AgentConfigurationError(AgentWorkerError):
    pass


@dataclass(frozen=True)
class OpenAICompatibleAgentConfig:
    endpoint: str
    api_key: str
    model: str
    timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls, *, environ: Mapping[str, str] | None = None, prefix: str = "JOB_DISCOVERY_AGENT") -> "OpenAICompatibleAgentConfig":
        if environ is None:
            load_dotenv()
            environ = os.environ
        base_url = environ.get(f"{prefix}_BASE_URL", "").strip()
        api_key = environ.get(f"{prefix}_API_KEY", "").strip()
        model = environ.get(f"{prefix}_MODEL", "").strip()
        if not base_url or not api_key or not model:
            raise AgentConfigurationError("AGENT_CONFIGURATION_MISSING", f"{prefix}_BASE_URL, {prefix}_API_KEY, and {prefix}_MODEL are required.")
        parts = urlsplit(base_url)
        if parts.scheme != "https" or not parts.netloc:
            raise AgentConfigurationError("AGENT_CONFIGURATION_INVALID", "Agent endpoint must be an HTTPS URL.")
        endpoint = base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint = f"{endpoint}/chat/completions"
        return cls(endpoint=endpoint, api_key=api_key, model=model)
