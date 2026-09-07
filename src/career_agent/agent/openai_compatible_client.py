from __future__ import annotations

import os
from dataclasses import dataclass
import math
from typing import Literal, Mapping
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
    max_input_tokens: int = 32000
    prompt_cache: Literal["disabled", "implicit", "explicit"] = "implicit"
    input_token_safety_factor: float = 1.1
    cjk_tokens_per_char: float = 1.8
    ascii_chars_per_token: float = 4.0

    def __post_init__(self) -> None:
        if self.max_input_tokens < 1024:
            raise ValueError("max_input_tokens must be at least 1024")
        if self.prompt_cache not in {"disabled", "implicit", "explicit"}:
            raise ValueError(
                "prompt_cache must be disabled, implicit, or explicit"
            )
        if (
            not math.isfinite(self.input_token_safety_factor)
            or not 1.0 <= self.input_token_safety_factor <= 2.0
        ):
            raise ValueError(
                "input_token_safety_factor must be between 1.0 and 2.0"
            )
        if (
            not math.isfinite(self.cjk_tokens_per_char)
            or not 0.5 <= self.cjk_tokens_per_char <= 3.0
        ):
            raise ValueError("cjk_tokens_per_char must be between 0.5 and 3.0")
        if (
            not math.isfinite(self.ascii_chars_per_token)
            or not 1.0 <= self.ascii_chars_per_token <= 8.0
        ):
            raise ValueError("ascii_chars_per_token must be between 1.0 and 8.0")

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
        raw_max_input_tokens = environ.get(
            f"{prefix}_MAX_INPUT_TOKENS", "32000"
        ).strip()
        try:
            max_input_tokens = int(raw_max_input_tokens)
        except ValueError as error:
            raise AgentConfigurationError(
                "AGENT_CONFIGURATION_INVALID",
                f"{prefix}_MAX_INPUT_TOKENS must be an integer.",
            ) from error
        if max_input_tokens < 1024:
            raise AgentConfigurationError(
                "AGENT_CONFIGURATION_INVALID",
                f"{prefix}_MAX_INPUT_TOKENS must be at least 1024.",
            )
        prompt_cache = environ.get(
            f"{prefix}_PROMPT_CACHE", "implicit"
        ).strip().lower()
        if prompt_cache not in {"disabled", "implicit", "explicit"}:
            raise AgentConfigurationError(
                "AGENT_CONFIGURATION_INVALID",
                f"{prefix}_PROMPT_CACHE must be disabled, implicit, or explicit.",
            )
        raw_safety_factor = environ.get(
            f"{prefix}_INPUT_TOKEN_SAFETY_FACTOR", "1.1"
        ).strip()
        try:
            input_token_safety_factor = float(raw_safety_factor)
        except ValueError as error:
            raise AgentConfigurationError(
                "AGENT_CONFIGURATION_INVALID",
                f"{prefix}_INPUT_TOKEN_SAFETY_FACTOR must be a number.",
            ) from error
        if (
            not math.isfinite(input_token_safety_factor)
            or not 1.0 <= input_token_safety_factor <= 2.0
        ):
            raise AgentConfigurationError(
                "AGENT_CONFIGURATION_INVALID",
                f"{prefix}_INPUT_TOKEN_SAFETY_FACTOR must be between 1.0 and 2.0.",
            )
        try:
            cjk_tokens_per_char = float(
                environ.get(
                    f"{prefix}_CJK_TOKENS_PER_CHAR",
                    "1.8",
                ).strip()
            )
            ascii_chars_per_token = float(
                environ.get(
                    f"{prefix}_ASCII_CHARS_PER_TOKEN",
                    "4.0",
                ).strip()
            )
        except ValueError as error:
            raise AgentConfigurationError(
                "AGENT_CONFIGURATION_INVALID",
                f"{prefix} tokenizer calibration values must be numbers.",
            ) from error
        if (
            not math.isfinite(cjk_tokens_per_char)
            or not 0.5 <= cjk_tokens_per_char <= 3.0
            or not math.isfinite(ascii_chars_per_token)
            or not 1.0 <= ascii_chars_per_token <= 8.0
        ):
            raise AgentConfigurationError(
                "AGENT_CONFIGURATION_INVALID",
                f"{prefix} tokenizer calibration is outside its safe range.",
            )
        return cls(
            endpoint=endpoint,
            api_key=api_key,
            model=model,
            max_input_tokens=max_input_tokens,
            prompt_cache=prompt_cache,
            input_token_safety_factor=input_token_safety_factor,
            cjk_tokens_per_char=cjk_tokens_per_char,
            ascii_chars_per_token=ascii_chars_per_token,
        )
