"""Value-free diagnostics from the serialized HTTP request, not model kwargs."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
import json
from typing import Callable

import httpx


# Only protocol/schema vocabulary is observable. Arbitrary schema property names,
# tool names, header names/values and all scalar content stay out of diagnostics.
_FIELDS = frozenset("""
model input messages role content type text tools name description parameters
strict tool_choice parallel_tool_calls response_format json_schema format schema
properties required additionalProperties items anyOf allOf oneOf enum const $defs
$ref title default minLength maxLength minItems maxItems minimum maximum pattern
store stream temperature top_p max_output_tokens max_tokens max_completion_tokens
reasoning effort summary include truncation instructions metadata service_tier
previous_response_id stream_options verbosity function arguments call_id id output
cache_control prompt_cache_key safety_identifier user context_size user_location
country city region timezone search_context_size filters allowed_domains
""".split())
_TOOL_TYPES = frozenset({
    "function", "web_search", "web_search_preview", "file_search",
    "computer", "computer_use_preview", "code_interpreter", "mcp", "image_generation",
})


@lru_cache(maxsize=1)
def sdk_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("openai", "langchain-openai", "langchain", "deepagents", "httpx"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "unavailable"
    return result


@dataclass(frozen=True)
class ProviderRequestStructure:
    field_paths: tuple[str, ...]
    tool_types: tuple[str, ...]
    schema_bytes: int
    protocol: str

    def as_dict(self) -> dict[str, object]:
        return {
            "field_paths": list(self.field_paths),
            "tool_types": list(self.tool_types),
            "schema_bytes": self.schema_bytes,
            "protocol": self.protocol,
            "sdk_versions": dict(sdk_versions()),
        }


def request_structure(request: httpx.Request) -> ProviderRequestStructure:
    payload = json.loads(request.content)
    paths: set[str] = set()
    schema_bytes = 0

    def visit(
        value: object, path: str = "", depth: int = 0, *, named_map: bool = False,
    ) -> None:
        nonlocal schema_bytes
        if depth > 24:
            return
        if isinstance(value, dict):
            for key, child in value.items():
                field = key if not named_map and key in _FIELDS else "*"
                child_path = f"{path}.{field}" if path else field
                paths.add(child_path)
                if not named_map and key in {"schema", "parameters"} and isinstance(child, dict):
                    schema_bytes += len(json.dumps(child, ensure_ascii=False).encode())
                visit(
                    child, child_path, depth + 1,
                    named_map=not named_map and key in {"properties", "$defs", "metadata"},
                )
        elif isinstance(value, list):
            for child in value:
                visit(child, f"{path}[]", depth + 1)

    visit(payload)
    tools = payload.get("tools", []) if isinstance(payload, dict) else []
    types: set[str] = set()
    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, dict):
                tool_type = tool.get("type")
                types.add(tool_type if isinstance(tool_type, str) and tool_type in _TOOL_TYPES else "other")
    protocol = (
        "responses" if request.url.path.endswith("/responses")
        else "chat_completions" if request.url.path.endswith("/chat/completions")
        else "unknown"
    )
    return ProviderRequestStructure(tuple(sorted(paths)), tuple(sorted(types)), schema_bytes, protocol)


class ProviderRequestObserver:
    def __init__(self, sink: Callable[[ProviderRequestStructure], None]) -> None:
        self._sink = sink

    def __call__(self, request: httpx.Request) -> None:
        try:
            self._sink(request_structure(request))
        except Exception:
            # Diagnostics must not change the capability's behavior, and cannot
            # print a parse exception that might contain request content.
            return


def trace_research_request(structure: ProviderRequestStructure) -> None:
    from career_agent.harness.observability import record_active_trace

    record_active_trace(
        "provider_request", "job_research", outcome="started", details=structure.as_dict()
    )
