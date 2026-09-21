"""One bounded retry for schema-invalid, side-effect-free model samples."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from career_agent.agent.openai_compatible_client import AgentWorkerError
from career_agent.harness.observability import record_active_trace


T = TypeVar("T")
INVALID_RESPONSE_RETRIES = 1


def retry_invalid_response(call: Callable[[], T], *, code_prefix: str) -> T:
    """Retry one invalid sample, then expose an exhausted terminal failure.

    The inner retry preserves already completed upstream work. Once that budget
    is exhausted, ``retryable`` is forced false so the Main Agent cannot retry
    the whole capability and multiply an expensive workflow.
    """

    invalid_code = f"{code_prefix}_INVALID_RESPONSE"
    for attempt in range(1, INVALID_RESPONSE_RETRIES + 2):
        try:
            return call()
        except AgentWorkerError as error:
            if error.code != invalid_code:
                raise
            if attempt > INVALID_RESPONSE_RETRIES:
                error.retryable = False
                raise
            record_active_trace(
                "model_retry",
                "structured_response",
                attempt=attempt + 1,
                outcome="started",
                error_code=invalid_code,
                recoverable=True,
                details={"code_prefix": code_prefix},
                model_call_category="capability_agent",
            )

    raise AssertionError("structured response retry loop exhausted")
