"""Bounded retries for side-effect-free structured model calls.

Two failures get one retry each: a schema-invalid sample, and a connection the
provider dropped before answering. A timeout is not retried: it has already
spent the whole time budget, and a second one would double the wait.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from openai import APITimeoutError

from career_agent.agent.openai_compatible_client import AgentWorkerError
from career_agent.harness.observability import record_active_trace


T = TypeVar("T")
INVALID_RESPONSE_RETRIES = 1
TRANSPORT_RETRIES = 1


def retry_invalid_response(call: Callable[[], T], *, code_prefix: str) -> T:
    """Retry one invalid sample and one dropped connection, then give up.

    The inner retry preserves already completed upstream work. Once the
    invalid-sample budget is exhausted, ``retryable`` is forced false so the
    Main Agent cannot retry the whole capability and multiply an expensive
    workflow. An exhausted transport failure keeps its own retryability: the
    provider may recover, and the user can ask again later.
    """

    invalid_code = f"{code_prefix}_INVALID_RESPONSE"
    transport_code = f"{code_prefix}_TRANSPORT_ERROR"
    invalid_retries = 0
    transport_retries = 0
    while True:
        try:
            return call()
        except AgentWorkerError as error:
            if error.code == invalid_code:
                if invalid_retries >= INVALID_RESPONSE_RETRIES:
                    error.retryable = False
                    raise
                invalid_retries += 1
            elif error.code == transport_code and not isinstance(
                error.__cause__, APITimeoutError
            ):
                if transport_retries >= TRANSPORT_RETRIES:
                    raise
                transport_retries += 1
            else:
                raise
            record_active_trace(
                "model_retry",
                "structured_response",
                attempt=invalid_retries + transport_retries + 1,
                outcome="started",
                error_code=error.code,
                recoverable=True,
                details={"code_prefix": code_prefix},
                model_call_category="capability_agent",
            )
