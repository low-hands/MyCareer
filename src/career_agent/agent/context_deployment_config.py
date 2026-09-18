"""Bounded deployment policy; model capabilities are configured, never guessed."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from typing import Mapping

from dotenv import load_dotenv

from career_agent.agent.openai_compatible_client import (
    AgentConfigurationError,
    OpenAICompatibleAgentConfig,
)

DEFAULT_RECENT_MESSAGE_LIMIT = 16
DEFAULT_SUMMARY_BATCH_SIZE = 8
DEFAULT_COMPACT_OCCUPANCY_THRESHOLD = 0.75
DEFAULT_CONTEXT_WINDOW_TOKENS = 65536
DEFAULT_SUMMARY_MAX_OUTPUT_TOKENS = 1200


def _environment(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    if environ is not None:
        return environ
    load_dotenv()
    return os.environ


def _integer(
    environ: Mapping[str, str], key: str, default: int, minimum: int, maximum: int
) -> int:
    try:
        value = int(environ.get(key, str(default)).strip())
    except ValueError:
        raise AgentConfigurationError(
            "AGENT_CONFIGURATION_INVALID", f"{key} must be an integer."
        ) from None
    if not minimum <= value <= maximum:
        raise AgentConfigurationError(
            "AGENT_CONFIGURATION_INVALID",
            f"{key} must be between {minimum} and {maximum}.",
        )
    return value


def _number(
    environ: Mapping[str, str], key: str, default: float, minimum: float, maximum: float
) -> float:
    try:
        value = float(environ.get(key, str(default)).strip())
    except ValueError:
        raise AgentConfigurationError(
            "AGENT_CONFIGURATION_INVALID", f"{key} must be a finite number."
        ) from None
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise AgentConfigurationError(
            "AGENT_CONFIGURATION_INVALID",
            f"{key} must be between {minimum} and {maximum}.",
        )
    return value


def _boolean(
    environ: Mapping[str, str], key: str, default: bool = False
) -> bool:
    value = environ.get(key, str(default)).strip().lower()
    if value not in {"true", "false"}:
        raise AgentConfigurationError(
            "AGENT_CONFIGURATION_INVALID",
            f"{key} must be true or false.",
        )
    return value == "true"


def validate_model_window(
    *, input_tokens: int, output_tokens: int, context_window_tokens: int, prefix: str
) -> None:
    if any(
        type(value) is not int or value < 1
        for value in (input_tokens, output_tokens, context_window_tokens)
    ):
        raise AgentConfigurationError(
            "AGENT_CONFIGURATION_INVALID",
            f"{prefix} input, output, and context window budgets must be positive integers.",
        )
    if input_tokens + output_tokens > context_window_tokens:
        raise AgentConfigurationError(
            "AGENT_CONFIGURATION_INVALID",
            f"{prefix} input and output budgets exceed its configured context window.",
        )


@dataclass(frozen=True)
class ContextDeploymentConfig:
    recent_message_limit: int = DEFAULT_RECENT_MESSAGE_LIMIT
    summary_batch_size: int = DEFAULT_SUMMARY_BATCH_SIZE
    compact_occupancy_threshold: float = DEFAULT_COMPACT_OCCUPANCY_THRESHOLD
    main_context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS

    def __post_init__(self) -> None:
        if (
            type(self.recent_message_limit) is not int
            or not 2 <= self.recent_message_limit <= 64
        ):
            raise ValueError("recent_message_limit must be between 2 and 64")
        if (
            type(self.summary_batch_size) is not int
            or not 2 <= self.summary_batch_size <= 32
        ):
            raise ValueError("summary_batch_size must be between 2 and 32")
        if not 0.7 <= self.compact_occupancy_threshold <= 0.9:
            raise ValueError("compact occupancy threshold must be between 0.7 and 0.9")
        if (
            type(self.main_context_window_tokens) is not int
            or not 2048 <= self.main_context_window_tokens <= 2_000_000
        ):
            raise ValueError("main context window must be between 2048 and 2000000")

    @classmethod
    def from_env(
        cls, *, environ: Mapping[str, str] | None = None
    ) -> ContextDeploymentConfig:
        env = _environment(environ)
        return cls(
            recent_message_limit=_integer(
                env, "CONTEXT_RECENT_MESSAGE_LIMIT", DEFAULT_RECENT_MESSAGE_LIMIT, 2, 64
            ),
            summary_batch_size=_integer(
                env, "CONTEXT_SUMMARY_BATCH_SIZE", DEFAULT_SUMMARY_BATCH_SIZE, 2, 32
            ),
            compact_occupancy_threshold=_number(
                env,
                "CONTEXT_COMPACT_OCCUPANCY_THRESHOLD",
                DEFAULT_COMPACT_OCCUPANCY_THRESHOLD,
                0.7,
                0.9,
            ),
            main_context_window_tokens=_integer(
                env,
                "MAIN_AGENT_CONTEXT_WINDOW_TOKENS",
                DEFAULT_CONTEXT_WINDOW_TOKENS,
                2048,
                2_000_000,
            ),
        )


@dataclass(frozen=True)
class ConversationSummaryAgentConfig:
    provider: OpenAICompatibleAgentConfig
    max_output_tokens: int = DEFAULT_SUMMARY_MAX_OUTPUT_TOKENS
    context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS
    disable_thinking: bool = False

    def __post_init__(self) -> None:
        if type(self.disable_thinking) is not bool:
            raise ValueError("summary disable_thinking must be a boolean")
        if (
            type(self.max_output_tokens) is not int
            or not 256 <= self.max_output_tokens <= 16384
        ):
            raise ValueError("summary output budget must be between 256 and 16384")
        if (
            type(self.provider.max_input_tokens) is not int
            or not 1024 <= self.provider.max_input_tokens <= 2_000_000
        ):
            raise ValueError("summary input budget must be between 1024 and 2000000")
        if (
            type(self.context_window_tokens) is not int
            or not 2048 <= self.context_window_tokens <= 2_000_000
        ):
            raise ValueError("summary context window must be between 2048 and 2000000")
        if not math.isfinite(self.provider.timeout_seconds) or not (
            1 <= self.provider.timeout_seconds <= 120
        ):
            raise ValueError("summary timeout must be between 1 and 120 seconds")
        validate_model_window(
            input_tokens=self.provider.max_input_tokens,
            output_tokens=self.max_output_tokens,
            context_window_tokens=self.context_window_tokens,
            prefix="CONVERSATION_SUMMARY_AGENT",
        )

    @classmethod
    def from_env(
        cls,
        *,
        main_config: OpenAICompatibleAgentConfig,
        main_context_window_tokens: int = DEFAULT_CONTEXT_WINDOW_TOKENS,
        environ: Mapping[str, str] | None = None,
    ) -> ConversationSummaryAgentConfig:
        env = _environment(environ)
        prefix = "CONVERSATION_SUMMARY_AGENT"
        configured = {key for key in env if key.startswith(f"{prefix}_")}
        supported = {
            f"{prefix}_{suffix}"
            for suffix in (
                "BASE_URL",
                "API_KEY",
                "MODEL",
                "TIMEOUT_SECONDS",
                "MAX_INPUT_TOKENS",
                "MAX_OUTPUT_TOKENS",
                "CONTEXT_WINDOW_TOKENS",
                "PROMPT_CACHE",
                "DISABLE_THINKING",
            )
        }
        if configured - supported:
            raise AgentConfigurationError(
                "AGENT_CONFIGURATION_INVALID",
                "Unknown CONVERSATION_SUMMARY_AGENT configuration field.",
            )
        # Absence is deliberate backward compatibility, not fallback on an
        # invalid endpoint, an empty secret, a typo, or a provider failure.
        # DISABLE_THINKING is a request option rather than a connection field,
        # so it may independently override the Main-connection fallback.
        fallback_options = {f"{prefix}_DISABLE_THINKING"}
        if configured <= fallback_options:
            return cls(
                provider=replace(main_config, timeout_seconds=30.0),
                context_window_tokens=main_context_window_tokens,
                disable_thinking=_boolean(
                    env, f"{prefix}_DISABLE_THINKING"
                ),
            )
        # Parse numeric values here first: errors must not chain the raw env
        # value from the shared provider parser into startup logs.
        timeout_seconds = _number(env, f"{prefix}_TIMEOUT_SECONDS", 30.0, 1.0, 120.0)
        max_input_tokens = _integer(
            env, f"{prefix}_MAX_INPUT_TOKENS", 32000, 1024, 2_000_000
        )
        provider = replace(
            OpenAICompatibleAgentConfig.from_env(environ=env, prefix=prefix),
            timeout_seconds=timeout_seconds,
            max_input_tokens=max_input_tokens,
        )
        return cls(
            provider=provider,
            max_output_tokens=_integer(
                env,
                f"{prefix}_MAX_OUTPUT_TOKENS",
                DEFAULT_SUMMARY_MAX_OUTPUT_TOKENS,
                256,
                16384,
            ),
            context_window_tokens=_integer(
                env,
                f"{prefix}_CONTEXT_WINDOW_TOKENS",
                DEFAULT_CONTEXT_WINDOW_TOKENS,
                2048,
                2_000_000,
            ),
            disable_thinking=_boolean(
                env, f"{prefix}_DISABLE_THINKING"
            ),
        )
