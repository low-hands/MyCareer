from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Literal, Mapping
from urllib.parse import urlsplit

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError


ProviderErrorCategory = Literal[
    "configuration", "rate_limit", "upstream", "transport", "timeout"
]
_SAFE_PROVIDER_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_.\[\]-]{0,127}\Z")
_RETRYABLE_PROVIDER_STATUS = frozenset({429, 502, 503, 504})
# Provider-controlled strings can contain user data even when they look like
# identifiers. Only known protocol vocabulary may cross the diagnostic boundary.
_PROVIDER_CODES = frozenset({
    "InvalidParameter", "InvalidParameterValue", "InvalidApiKey", "InvalidRequest",
    "UnsupportedOperation", "AccessDenied", "Unauthorized", "Throttling",
    "Throttling.RateQuota", "Throttling.AllocationQuota", "InternalError",
    "ServiceUnavailable", "BadRequest", "invalid_request_error", "bad_request",
    "invalid_api_key", "insufficient_quota", "rate_limit_exceeded",
    "context_length_exceeded", "content_policy_violation", "invalid_value",
    "invalid_type", "missing_required_parameter", "unsupported_parameter",
    "unsupported_value", "model_not_found", "server_error",
})
_PROVIDER_TYPES = frozenset({
    "invalid_request_error", "authentication_error", "permission_error",
    "rate_limit_error", "server_error", "insufficient_quota", "api_error",
})
_PROVIDER_PARAMS = frozenset("""
model input messages role content type text tools name description parameters
strict tool_choice parallel_tool_calls response_format json_schema format schema
properties required additionalProperties items anyOf allOf oneOf enum const
store stream temperature top_p max_output_tokens max_tokens max_completion_tokens
reasoning effort summary include truncation instructions metadata service_tier
previous_response_id stream_options verbosity function arguments call_id id output
prompt_cache_key safety_identifier user context_size user_location file_data
file_url input_file search_context_size filters allowed_domains
""".split())


def _provider_identifier(value: object, *, field: str) -> str | None:
    if not isinstance(value, str) or not _SAFE_PROVIDER_IDENTIFIER.fullmatch(value):
        return None
    if field == "code":
        return value if value in _PROVIDER_CODES else None
    if field == "type":
        return value if value in _PROVIDER_TYPES else None
    # Keep useful indexed paths (tools[0].type), not arbitrary schema property
    # names, query strings or provider-echoed content.
    parts = re.sub(r"\[\d{1,4}\]", "", value).split(".")
    return value if all(part in _PROVIDER_PARAMS for part in parts) else None


@dataclass(frozen=True)
class ProviderErrorMetadata:
    """Small safe boundary shared by specialist adapters and trace callbacks."""

    status: int | None = None
    code: str | None = None
    param: str | None = None
    type: str | None = None
    category: ProviderErrorCategory = "configuration"
    retryable: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "code": _provider_identifier(self.code, field="code"),
            "param": _provider_identifier(self.param, field="param"),
            "type": _provider_identifier(self.type, field="type"),
            "category": self.category,
            "retryable": self.retryable,
        }


class AgentWorkerError(Exception):
    def __init__(self, code: str, message: str, *, retryable: bool = False, detail: str | None = None, provider: ProviderErrorMetadata | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.detail = detail
        self.provider = provider


def provider_error_metadata(error: BaseException) -> ProviderErrorMetadata | None:
    """Extract only safe provider identifiers, never exception text or messages."""
    if isinstance(error, AgentWorkerError):
        return error.provider
    if isinstance(error, APITimeoutError):
        return ProviderErrorMetadata(category="timeout", retryable=True)
    if isinstance(error, APIConnectionError):
        return ProviderErrorMetadata(category="transport", retryable=True)
    if not isinstance(error, APIStatusError):
        return None
    body = error.body
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        body = body["error"]
    fields = body if isinstance(body, dict) else {}
    status = error.status_code
    category: ProviderErrorCategory = (
        "rate_limit" if status == 429 else "upstream" if status >= 500 else "configuration"
    )
    return ProviderErrorMetadata(
        status=status,
        code=_provider_identifier(fields.get("code"), field="code"),
        param=_provider_identifier(fields.get("param"), field="param"),
        type=_provider_identifier(fields.get("type"), field="type"),
        category=category,
        retryable=status in _RETRYABLE_PROVIDER_STATUS,
    )


def provider_worker_error(prefix: str, error: BaseException) -> AgentWorkerError:
    """Translate SDK exceptions without losing their safe diagnostic category."""
    metadata = provider_error_metadata(error)
    if metadata is None:
        raise TypeError("Expected an OpenAI provider error")
    suffix = {
        "timeout": "TIMEOUT",
        "transport": "TRANSPORT_ERROR",
        "rate_limit": "RATE_LIMITED",
    }.get(metadata.category, f"REJECTED_{metadata.status}")
    return AgentWorkerError(
        f"{prefix}_{suffix}",
        "Specialist provider request failed.",
        retryable=metadata.retryable,
        provider=metadata,
    )


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

    def __post_init__(self) -> None:
        if self.max_input_tokens < 1024:
            raise ValueError("max_input_tokens must be at least 1024")
        if self.prompt_cache not in {"disabled", "implicit", "explicit"}:
            raise ValueError(
                "prompt_cache must be disabled, implicit, or explicit"
            )

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
        endpoint = base_url.rstrip("/").removesuffix("/responses")
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
        return cls(
            endpoint=endpoint,
            api_key=api_key,
            model=model,
            max_input_tokens=max_input_tokens,
            prompt_cache=prompt_cache,
        )
