"""One structured model call, and the five ways it can fail.

Six capability workers each wrote this out: the same ``responses.create`` with
the same four ``except`` branches, differing only in an error-code prefix, a
human noun, the instructions, the schema name, the output type and a token
limit. Between them that came to fifty-two ``raise AgentWorkerError`` sites and
three byte-identical copies of ``_provider_code``.

The cost of that was not length. It was drift: three of the six appended the
provider's own error code to ``{PREFIX}_REJECTED_{status}`` and three did not,
so the same class of failure produced a less diagnosable code in half the
workers for no reason anyone chose. Nothing pinned the difference — there is no
test anywhere on ``REJECTED_`` — which is how it survived.

The boundary is narrow on purpose. This owns *the call and its five failure
modes*. Each worker keeps what is genuinely its own: validating its inputs
(``_EMPTY_DOCUMENT``, ``_EMPTY_JD``, ``_INVALID_TEXT_ENCODING`` and friends),
choosing its instructions, and naming its output type. A helper that also tried
to own those would need a branch per capability, which is the fake abstraction
this one exists to avoid.

Deliberately not extended to ``openai_compatible_agent_worker`` or the Main
Agent decision maker: they call ``chat.completions``, whose request and response
shapes differ. Folding two protocols into one function would put the difference
inside the abstraction rather than beside it.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from openai import APIConnectionError, APIStatusError, RateLimitError
from pydantic import BaseModel

from career_agent.agent.openai_compatible_client import AgentWorkerError
from career_agent.agent.structured_response_retry import retry_invalid_response


T = TypeVar("T", bound=BaseModel)

_PROVIDER_CODE = re.compile(r"[A-Za-z0-9_.-]{1,64}")


def provider_code(error: APIStatusError) -> str:
    """The provider's own error code, when it sent one that is safe to echo.

    Appended to the status so an operator can tell "the model refused this
    content" from "the key is out of quota" without opening the provider's
    dashboard. Bounded by a character class because it lands in an error code
    that is logged and compared: an unbounded string from a remote service does
    not belong in an identifier.
    """
    body = getattr(error, "body", None)
    candidate = (
        body.get("error", {}).get("code")
        if isinstance(body, dict) and isinstance(body.get("error"), dict)
        else None
    )
    if isinstance(candidate, str) and _PROVIDER_CODE.fullmatch(candidate):
        return f"_{candidate}"
    return ""


def validation_detail(error: ValueError) -> str:
    """Why parsing failed, in a form that carries none of what was parsed.

    ``str(error)`` on a pydantic ``ValidationError`` embeds the offending input,
    which here is model output derived from a resume, a JD or an interview
    answer. Only the structural fields — which field, what kind of problem —
    survive, so a detail can be logged and shown without becoming a second copy
    of the document.

    Three of the six workers passed no detail at all. They gain one: it is
    strictly more diagnostic and, by the same construction, still carries
    nothing from the payload.
    """
    errors = getattr(error, "errors", lambda: ())()
    if not isinstance(errors, list):
        return type(error).__name__
    return json.dumps(
        [
            {
                "type": item.get("type"),
                "loc": item.get("loc"),
                "msg": item.get("msg"),
            }
            for item in errors
            if isinstance(item, dict)
        ],
        ensure_ascii=False,
        sort_keys=True,
    )


def structured_response(
    client: Any,
    *,
    model: str,
    timeout_seconds: float,
    instructions: str,
    content: Any,
    output_type: type[T],
    schema_name: str,
    max_output_tokens: int,
    code_prefix: str,
    subject: str,
    include_validation_feedback: bool = False,
) -> T:
    """Ask for one JSON-schema-shaped answer, or raise a classified failure.

    ``code_prefix`` and ``subject`` keep each capability's own vocabulary in its
    errors (``RESUME_ANALYSIS_RATE_LIMITED``, "Resume analysis model is rate
    limited."), because those codes are what an operator greps for and a caller
    branches on. What is shared is which five failures exist and how each is
    classified — including which are retryable, a judgement that was previously
    made six times.
    """
    repair_detail: str | None = None

    def request_once() -> T:
        nonlocal repair_detail
        request_instructions = instructions
        if include_validation_feedback and repair_detail is not None:
            request_instructions += (
                "\nThe previous response failed schema validation. Return a complete "
                "corrected response conforming to the schema. Structural errors: "
                + repair_detail
            )
        try:
            response = client.responses.create(
                model=model,
                instructions=request_instructions,
                input=[{"role": "user", "content": content}],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "schema": output_type.model_json_schema(),
                        "strict": False,
                    }
                },
                max_output_tokens=max_output_tokens,
                timeout=timeout_seconds,
            )
        except RateLimitError as error:
            raise AgentWorkerError(
                f"{code_prefix}_RATE_LIMITED",
                f"{subject} model is rate limited.",
                retryable=True,
            ) from error
        except APIConnectionError as error:
            raise AgentWorkerError(
                f"{code_prefix}_TRANSPORT_ERROR",
                f"{subject} model transport failed.",
                retryable=True,
            ) from error
        except APIStatusError as error:
            raise AgentWorkerError(
                f"{code_prefix}_REJECTED_{error.status_code}{provider_code(error)}",
                f"{subject} model rejected the request.",
            ) from error

        output_text = getattr(response, "output_text", None)
        if not isinstance(output_text, str) or not output_text.strip():
            raise AgentWorkerError(
                f"{code_prefix}_EMPTY_RESPONSE",
                f"{subject} model returned no structured output.",
            )
        try:
            return output_type.model_validate_json(output_text)
        except ValueError as error:
            repair_detail = validation_detail(error)
            raise AgentWorkerError(
                f"{code_prefix}_INVALID_RESPONSE",
                f"{subject} model returned invalid structured output.",
                detail=repair_detail,
            ) from error

    return retry_invalid_response(request_once, code_prefix=code_prefix)
