"""Match each provider call before consuming its recorded response.

Only transport/cache hints and randomized spotlight delimiters are normalized.
The evaluation clock is fixed by the runner; all model-visible data is matched.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from types import SimpleNamespace

from career_agent.agent.providers.openai_client import AgentWorkerError
from career_agent.agent.runtime.decision_messages import (
    CONTROL_CONTEXT_LABEL, CONTROL_REMINDER_TAG,
    DATA_CONTEXT_LABEL, STABLE_DATA_CONTEXT_LABEL, TURN_OBSERVATION_LABEL,
)


class RequestMismatch(ValueError):
    pass


def request_fingerprint(request):
    value = deepcopy(request)
    value.pop("timeout", None)
    extra = value.get("extra_body", {})
    for key in ("prompt_cache_key", "prompt_cache_options"):
        extra.pop(key, None)
    if not extra:
        value.pop("extra_body", None)
    # Normalize only runtime-generated outer delimiters, never user text.
    for index, message in enumerate(value.get("messages", [])):
        content = message.get("content")
        # Explicit caching wraps the same stable text in one SDK text block.
        # Remove only that runtime-generated transport wrapper.
        if (index == 1 and isinstance(content, list) and len(content) == 1
                and set(content[0]) <= {"type", "text", "prompt_cache_breakpoint"}
                and content[0].get("type") == "text"
                and isinstance(content[0].get("text"), str)
                and content[0]["text"].startswith(STABLE_DATA_CONTEXT_LABEL + "\n")):
            content = message["content"] = content[0]["text"]
        if not isinstance(content, str):
            continue
        for label in (DATA_CONTEXT_LABEL, STABLE_DATA_CONTEXT_LABEL, TURN_OBSERVATION_LABEL):
            prefix = label + '\n<untrusted-data nonce="'
            if content.startswith(prefix):
                match = re.match(r'[0-9a-f]{32}', content[len(prefix):])
                if match:
                    nonce = match.group()
                    closing = f'</untrusted-data nonce="{nonce}">'
                    if content.endswith(closing):
                        message["content"] = (prefix + '0' * 32
                            + content[len(prefix) + 32:-len(closing)]
                            + '</untrusted-data nonce="' + '0' * 32 + '">')
                break
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _response_dict(response):
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json")
    # Scripted clients and legacy reconstructed decisions have the same shape.
    return {"choices": [{"message": {
        "content": choice.message.content,
        "tool_calls": [{"function": {"name": call.function.name,
                                         "arguments": call.function.arguments}}
                       for call in (choice.message.tool_calls or [])],
    }} for choice in response.choices]}


def _response_object(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _response_object(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_response_object(item) for item in value]
    return value


class RequestTape:
    def __init__(self, recording=None, *, clock=None):
        self.clock = clock
        self.expected = recording
        self.entries = []
        self.mismatch = None

    def attach(self, maker):
        original = maker._request

        def request(*, messages, tools, request_options):
            messages = deepcopy(messages)
            if self.clock is not None:
                # Replace only the generated control block immediately before
                # the volatile data projection; user text is never rewritten.
                prefix = f"<{CONTROL_REMINDER_TAG}>\n{CONTROL_CONTEXT_LABEL}\n"
                suffix = f"\n</{CONTROL_REMINDER_TAG}>"
                for index, message in enumerate(messages):
                    if index and isinstance(message.get("content"), str) and message["content"].startswith(DATA_CONTEXT_LABEL + "\n"):
                        control = messages[index - 1].get("content", "")
                        if control.startswith(prefix) and control.endswith(suffix):
                            data = json.loads(control[len(prefix):-len(suffix)])
                            data["runtime_clock"] = self.clock
                            messages[index - 1]["content"] = prefix + json.dumps(data, ensure_ascii=False, sort_keys=True) + suffix
                            break
            fingerprint = request_fingerprint({
                "model": maker._config.model, "max_tokens": maker._max_output_tokens,
                "messages": messages, "tools": list(tools), "parallel_tool_calls": False,
                "tool_choice": request_options.get("tool_choice", "auto"),
                **{key: value for key, value in request_options.items() if key != "tool_choice"},
            })
            index = len(self.entries)
            if self.expected is not None:
                if index >= len(self.expected) or self.expected[index]["fingerprint"] != fingerprint:
                    self.mismatch = f"model request {index + 1} changed or has no recording"
                    raise RequestMismatch(self.mismatch)
                entry = self.expected[index]
                self.entries.append(entry)
                if "error" in entry:
                    raise AgentWorkerError(entry["error"]["code"], "Recorded provider failure.",
                                           retryable=entry["error"]["retryable"])
                return _response_object(entry["response"])
            try:
                response = original(messages=messages, tools=tools, request_options=request_options)
            except AgentWorkerError as error:
                self.entries.append({"fingerprint": fingerprint,
                                     "error": {"code": error.code, "retryable": error.retryable}})
                raise
            self.entries.append({"fingerprint": fingerprint, "response": _response_dict(response)})
            return response

        maker._request = request

    def verify(self):
        if self.mismatch:
            raise RequestMismatch(self.mismatch)
        if self.expected is not None and len(self.entries) != len(self.expected):
            raise RequestMismatch("recorded model requests were not all consumed")
