"""Vendor-specific request options for the Main Agent's Chat Completions calls.

Each vendor that speaks the OpenAI-compatible protocol still names its own
extensions differently. Only these differences live here; the decision loop
itself is vendor-neutral.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from career_agent.agent.providers.openai_client import AgentConfigurationError


VendorName = Literal["qwen", "deepseek"]


@dataclass(frozen=True)
class QwenMainAgentVendor:
    """Alibaba Model Studio (DashScope) compatible mode."""

    name: VendorName = "qwen"
    # Qwen accepts the OpenAI-style prompt_cache_key in extra_body.
    sends_prompt_cache_key: bool = True
    # Chat Completions documents required as unreliable in non-thinking mode.
    supports_required_tool_choice: bool = False

    def request_extra_body(self, *, thinking: bool | None) -> dict[str, Any]:
        # Unset keeps the endpoint's default, which is what Qwen recordings use.
        if thinking is None:
            return {}
        return {"enable_thinking": thinking}

    def validate(self, *, thinking: bool | None) -> None:
        return None


@dataclass(frozen=True)
class DeepSeekMainAgentVendor:
    """DeepSeek's OpenAI-compatible API."""

    name: VendorName = "deepseek"
    # DeepSeek caches context automatically and does not document
    # prompt_cache_key; do not send an unknown extension.
    sends_prompt_cache_key: bool = False
    # Main Agent disables DeepSeek thinking, where required would be rejected.
    supports_required_tool_choice: bool = True

    def request_extra_body(self, *, thinking: bool | None) -> dict[str, Any]:
        # DeepSeek enables thinking by default, so unset must send "disabled".
        return {"thinking": {"type": "enabled" if thinking else "disabled"}}

    def validate(self, *, thinking: bool | None) -> None:
        if thinking:
            raise AgentConfigurationError(
                "MAIN_AGENT_UNSUPPORTED_THINKING",
                "DeepSeek thinking mode is not supported for the Main Agent: it "
                "rejects a forced tool_choice, which the interaction fallback "
                "needs, and requires reasoning_content to be passed back on "
                "every later request, which the Main Agent does not do.",
            )


MainAgentVendor = QwenMainAgentVendor | DeepSeekMainAgentVendor

_VENDORS: dict[str, MainAgentVendor] = {
    "qwen": QwenMainAgentVendor(),
    "deepseek": DeepSeekMainAgentVendor(),
}


def main_agent_vendor(name: str) -> MainAgentVendor:
    vendor = _VENDORS.get(name.strip().lower())
    if vendor is None:
        raise AgentConfigurationError(
            "MAIN_AGENT_INVALID_VENDOR",
            "MAIN_AGENT_VENDOR must be qwen or deepseek",
        )
    return vendor
