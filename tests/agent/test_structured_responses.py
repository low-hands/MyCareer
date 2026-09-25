"""The five ways one structured model call can fail.

Written when the six workers collapsed into one helper, because the collapse
made the gap visible: three of the five branches — rate limit, transport,
provider rejection — had **no test anywhere** before or after, in any worker.
The suite was green through a migration that rewrote them.

That is also where the migration's most consequential fix landed: three workers
had silently dropped the provider's error code from ``REJECTED_{status}``, and
nothing would have noticed either the drift or its repair.

One place instead of six is what makes covering all five affordable.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
from pydantic import BaseModel

from career_agent.agent.openai_compatible_client import AgentWorkerError
from career_agent.agent.structured_response_retry import (
    INVALID_RESPONSE_RETRIES,
    retry_invalid_response,
)
from career_agent.agent.structured_responses import structured_response
from career_agent.harness.observability import (
    ACTIVE_TRACE_CONTEXT,
    InMemoryTraceRecorder,
)


class Answer(BaseModel):
    verdict: str
    score: int


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://example.invalid/v1/responses")


def _status_error(status_code: int, body: object) -> APIStatusError:
    return APIStatusError(
        "rejected",
        response=httpx.Response(status_code, request=_request()),
        body=body,
    )


class _Client:
    """A responses client that returns text, or raises what a provider raises."""

    def __init__(self, *, output_text: str | None = None, error: Exception | None = None):
        self._output_text = output_text
        self._error = error
        self.calls: list[dict] = []
        self.responses = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return type("Response", (), {"output_text": self._output_text})()


class _SequenceClient:
    """A responses client whose samples change between bounded attempts."""

    def __init__(self, *output_texts: str):
        self._output_texts = list(output_texts)
        self.calls: list[dict] = []
        self.responses = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("Response", (), {"output_text": self._output_texts.pop(0)})()


def _call(client, **overrides):
    arguments = {
        "model": "test-model",
        "timeout_seconds": 30.0,
        "instructions": "答题",
        "content": "内容",
        "output_type": Answer,
        "schema_name": "answer",
        "max_output_tokens": 512,
        "code_prefix": "RESUME_ANALYSIS",
        "subject": "Resume analysis",
    }
    arguments.update(overrides)
    return structured_response(client, **arguments)


def test_a_valid_answer_is_parsed_into_the_declared_type() -> None:
    client = _Client(output_text='{"verdict": "strong", "score": 8}')

    answer = _call(client)

    assert answer == Answer(verdict="strong", score=8)
    request = client.calls[0]
    # The schema travels with the request, so the provider constrains the shape
    # rather than the parse discovering the mismatch afterwards.
    assert request["text"]["format"]["name"] == "answer"
    assert request["text"]["format"]["schema"] == Answer.model_json_schema()
    assert request["timeout"] == 30.0



@pytest.mark.parametrize(
    ("error", "code", "retryable"),
    (
        (
            RateLimitError(
                "slow down", response=httpx.Response(429, request=_request()), body=None
            ),
            "RESUME_ANALYSIS_RATE_LIMITED",
            True,
        ),
        (
            APIConnectionError(request=_request()),
            "RESUME_ANALYSIS_TRANSPORT_ERROR",
            True,
        ),
    ),
)
def test_a_transient_provider_failure_is_classified_retryable(
    error, code, retryable
) -> None:
    """Retryability is decided once here; it used to be decided six times.

    The distinction matters upstream: the loop spends its retry budget on these
    and refuses to spend it on the rest.
    """
    with pytest.raises(AgentWorkerError) as raised:
        _call(_Client(error=error))

    assert raised.value.code == code
    assert raised.value.retryable is retryable


def test_a_rejection_carries_the_status_and_the_providers_own_code() -> None:
    """The half of the migration that had no coverage at all.

    Three of the six workers appended the provider code and three did not, so
    the same failure was less diagnosable in half the system. Nothing tested
    ``REJECTED_`` anywhere, which is how the difference survived — and how its
    repair would also have gone unverified.
    """
    error = _status_error(400, {"error": {"code": "content_policy_violation"}})

    with pytest.raises(AgentWorkerError) as raised:
        _call(_Client(error=error))

    assert raised.value.code == (
        "RESUME_ANALYSIS_REJECTED_400_content_policy_violation"
    )
    assert raised.value.retryable is False


@pytest.mark.parametrize(
    "body",
    (
        None,
        {"error": "not-a-dict"},
        {"error": {"code": "has spaces"}},
        {"error": {"code": "x" * 65}},
    ),
)
def test_an_unusable_provider_code_leaves_the_status_alone(body) -> None:
    """The code lands in an identifier that gets logged and compared.

    A remote service can put anything in that field, so anything that is not a
    short, plain token is dropped rather than concatenated. The status still
    identifies the failure.
    """
    with pytest.raises(AgentWorkerError) as raised:
        _call(_Client(error=_status_error(503, body)))

    assert raised.value.code == "RESUME_ANALYSIS_REJECTED_503"


@pytest.mark.parametrize("output_text", (None, "", "   ", 42))
def test_an_answer_that_is_not_text_is_reported_as_empty(output_text) -> None:
    with pytest.raises(AgentWorkerError) as raised:
        _call(_Client(output_text=output_text))

    assert raised.value.code == "RESUME_ANALYSIS_EMPTY_RESPONSE"


def test_an_unparseable_answer_reports_why_without_quoting_the_payload() -> None:
    """The detail says which field and what kind of problem, and nothing else.

    Model output here derives from a resume, a JD or an interview answer, so
    ``str(ValidationError)`` — which embeds the offending input — cannot be the
    detail. This was nearly the migration's one regression: the first version of
    the helper used exactly that.
    """
    secret = "候选人机密履历内容"
    client = _Client(output_text=f'{{"verdict": "ok", "score": "{secret}"}}')

    with pytest.raises(AgentWorkerError) as raised:
        _call(client)

    assert raised.value.code == "RESUME_ANALYSIS_INVALID_RESPONSE"
    assert raised.value.retryable is False
    assert len(client.calls) == 2
    assert raised.value.detail is not None
    assert secret not in raised.value.detail
    assert "score" in raised.value.detail


def test_an_invalid_sample_is_retried_once_and_the_next_valid_sample_wins() -> None:
    private_output = "PRIVATE INVALID MODEL OUTPUT"
    client = _SequenceClient(
        f'{{"verdict": "ok", "score": "{private_output}"}}',
        '{"verdict": "strong", "score": 8}',
    )
    recorder = InMemoryTraceRecorder()
    token = ACTIVE_TRACE_CONTEXT.set((recorder, "turn-retry-success"))
    try:
        answer = _call(client)
    finally:
        ACTIVE_TRACE_CONTEXT.reset(token)

    assert answer == Answer(verdict="strong", score=8)
    assert len(client.calls) == 2
    retries = recorder.snapshot("turn-retry-success").events
    assert len(retries) == 1
    retry = retries[0]
    assert retry.event_type == "model_retry"
    assert retry.attempt == 2
    assert retry.error_code == "RESUME_ANALYSIS_INVALID_RESPONSE"
    assert retry.recoverable is True
    assert retry.details == {"code_prefix": "RESUME_ANALYSIS"}
    # Why the sample was rejected survives the retry, and only structurally.
    assert '"loc": ["score"]' in (retry.error_detail or "")
    assert private_output not in retry.model_dump_json()


def test_two_invalid_samples_stop_after_one_retry_and_trace_it_once() -> None:
    client = _SequenceClient(
        '{"verdict": "ok", "score": "first"}',
        '{"verdict": "ok", "score": "second"}',
    )
    recorder = InMemoryTraceRecorder()
    token = ACTIVE_TRACE_CONTEXT.set((recorder, "turn-retry-exhausted"))
    try:
        with pytest.raises(AgentWorkerError) as raised:
            _call(client)
    finally:
        ACTIVE_TRACE_CONTEXT.reset(token)

    assert raised.value.code == "RESUME_ANALYSIS_INVALID_RESPONSE"
    assert raised.value.retryable is False
    assert len(client.calls) == 2
    retries = recorder.snapshot("turn-retry-exhausted").events
    assert len(retries) == 1
    assert retries[0].event_type == "model_retry"
    assert retries[0].attempt == 2


class _FlakyClient:
    """A responses client that raises the queued errors, then answers."""

    def __init__(self, *errors: Exception, output_text: str = '{"verdict": "ok", "score": 1}'):
        self._errors = list(errors)
        self._output_text = output_text
        self.calls: list[dict] = []
        self.responses = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._errors:
            raise self._errors.pop(0)
        return type("Response", (), {"output_text": self._output_text})()


def test_a_dropped_connection_is_retried_once_and_the_next_answer_wins() -> None:
    client = _FlakyClient(APIConnectionError(request=_request()))
    recorder = InMemoryTraceRecorder()
    token = ACTIVE_TRACE_CONTEXT.set((recorder, "turn-transport-retry"))
    try:
        answer = _call(client)
    finally:
        ACTIVE_TRACE_CONTEXT.reset(token)

    assert answer == Answer(verdict="ok", score=1)
    assert len(client.calls) == 2
    retries = recorder.snapshot("turn-transport-retry").events
    assert [(item.event_type, item.attempt, item.error_code) for item in retries] == [
        ("model_retry", 2, "RESUME_ANALYSIS_TRANSPORT_ERROR")
    ]


def test_two_dropped_connections_stop_after_one_retry_and_stay_retryable() -> None:
    client = _FlakyClient(
        APIConnectionError(request=_request()),
        APIConnectionError(request=_request()),
    )

    with pytest.raises(AgentWorkerError) as raised:
        _call(client)

    assert raised.value.code == "RESUME_ANALYSIS_TRANSPORT_ERROR"
    assert raised.value.retryable is True
    assert len(client.calls) == 2


def test_a_timeout_is_not_retried() -> None:
    """A timeout already spent the whole budget; a retry would double the wait."""
    client = _FlakyClient(APITimeoutError(request=_request()))

    with pytest.raises(AgentWorkerError) as raised:
        _call(client)

    assert raised.value.code == "RESUME_ANALYSIS_TRANSPORT_ERROR"
    assert len(client.calls) == 1


def test_each_capability_keeps_its_own_error_vocabulary() -> None:
    """Shared implementation, unshared identifiers.

    Operators grep these codes and callers branch on them, so unifying how the
    five failures are produced must not rename any of them.
    """
    with pytest.raises(AgentWorkerError) as raised:
        _call(
            _Client(output_text=""),
            code_prefix="MOCK_INTERVIEW",
            subject="Mock interview",
        )

    assert raised.value.code == "MOCK_INTERVIEW_EMPTY_RESPONSE"
    assert str(raised.value).startswith("Mock interview")


_MIGRATED_WORKERS = {
    "openai_resume_analysis_worker": "RESUME_ANALYSIS",
    "openai_resume_job_match_worker": "RESUME_JOB_MATCH",
    "openai_job_analysis_worker": "JOB_ANALYSIS",
    "openai_email_tracking_worker": "EMAIL_TRACKING",
    "openai_interview_preparation_worker": "INTERVIEW_PREPARATION",
    "openai_resume_tailoring_reviewer": "RESUME_REVIEW",
    "openai_mock_interview_worker": "MOCK_INTERVIEW",
}
"""Each worker and the error vocabulary its capability owns."""


def test_every_worker_wires_its_own_prefix_and_nothing_else_calls_the_provider() -> None:
    """Read from source, because the defect this prevents is an omission.

    The tests above prove the helper classifies five failures correctly. They
    say nothing about whether a worker hands it the right ``code_prefix`` — a
    copy-pasted new worker inheriting the previous one's prefix would produce
    correctly shaped, wrongly attributed errors, and every existing test would
    still pass.

    The second half is the reason the helper exists: a worker that goes back to
    calling ``responses.create`` itself gets its own copy of five branches to
    keep in step, which is how three of them silently lost the provider code the
    first time.
    """
    import ast

    agent = Path(__file__).resolve().parents[2] / "src" / "career_agent" / "agent"
    wired: dict[str, str] = {}
    direct_callers = []
    # The shared helpers: Responses (``structured_responses``) and, since 093,
    # Chat Completions JSON Schema plus the explicit Responses text adapter
    # (``structured_chat_completions``). Only they may call the provider.
    helpers = {"structured_responses", "structured_chat_completions"}
    for source in agent.glob("*.py"):
        if source.stem in helpers:
            continue
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = ast.unparse(node.func)
            if called.endswith("responses.create"):
                direct_callers.append(source.stem)
            # A worker may call a helper directly or through a protocol-selected
            # partial; either way its literal ``code_prefix`` is its wiring.
            if called == "structured_response" or any(
                keyword.arg == "code_prefix" and isinstance(keyword.value, ast.Constant)
                for keyword in node.keywords
            ):
                prefixes = [
                    ast.literal_eval(keyword.value)
                    for keyword in node.keywords
                    if keyword.arg == "code_prefix"
                ]
                if prefixes:
                    wired[source.stem] = prefixes[0]

    assert wired == _MIGRATED_WORKERS
    # The operator smoke probes provider capabilities stage by stage and must
    # see the raw response; it is a diagnostic entry point, not a worker.
    assert direct_callers == ["job_research_provider_smoke"]


@pytest.mark.parametrize("prefix", sorted(set(_MIGRATED_WORKERS.values())))
@pytest.mark.parametrize(
    ("failure", "suffix"),
    (
        ("rate_limit", "RATE_LIMITED"),
        ("transport", "TRANSPORT_ERROR"),
        ("rejected", "REJECTED_400_bad_request"),
        ("empty", "EMPTY_RESPONSE"),
        ("invalid", "INVALID_RESPONSE"),
    ),
)
def test_every_capability_reaches_all_five_codes_under_its_own_prefix(
    prefix, failure, suffix
) -> None:
    """Thirty combinations, because the migration changed all thirty at once.

    Before the collapse each worker owned its five branches and none of the
    three provider-exception ones was tested anywhere. Driving the matrix is
    what makes "the codes did not change" a checked statement rather than a
    claim about a diff.
    """
    clients = {
        "rate_limit": _Client(
            error=RateLimitError(
                "slow down", response=httpx.Response(429, request=_request()), body=None
            )
        ),
        "transport": _Client(error=APIConnectionError(request=_request())),
        "rejected": _Client(
            error=_status_error(400, {"error": {"code": "bad_request"}})
        ),
        "empty": _Client(output_text=""),
        "invalid": _Client(output_text='{"verdict": "ok", "score": "not-a-number"}'),
    }

    with pytest.raises(AgentWorkerError) as raised:
        _call(clients[failure], code_prefix=prefix, subject="Capability")

    assert raised.value.code == f"{prefix}_{suffix}"


def test_an_exhausted_invalid_response_cannot_be_retried_by_the_caller() -> None:
    """The bounded inner budget must be the only one spent on a bad sample.

    A retryable exhausted failure would let the Main Agent re-run the whole
    capability, so an expensive workflow would repeat every completed step to
    reach the same last call: today's tailoring draft costs 88-165s before the
    review it dies in. The inner raise happens not to mark the error retryable,
    which is why forcing it has to be asserted against one that does — the
    previous revision of this retry did exactly that.
    """

    attempts: list[int] = []

    def always_invalid() -> None:
        attempts.append(1)
        raise AgentWorkerError(
            "CAPABILITY_INVALID_RESPONSE",
            "Capability model returned invalid structured output.",
            retryable=True,
        )

    with pytest.raises(AgentWorkerError) as raised:
        retry_invalid_response(always_invalid, code_prefix="CAPABILITY")

    assert raised.value.retryable is False
    assert len(attempts) == INVALID_RESPONSE_RETRIES + 1


def test_a_failure_that_is_not_an_invalid_sample_keeps_its_own_retryability() -> None:
    """Only the invalid-sample class is bounded here.

    Rate limiting and transport loss are retryable for reasons this retry knows
    nothing about, and they must pass through untouched and uncounted rather
    than being re-sampled against a provider that just asked us to slow down.
    """

    attempts: list[int] = []

    def rate_limited() -> None:
        attempts.append(1)
        raise AgentWorkerError(
            "CAPABILITY_RATE_LIMITED",
            "Capability model is rate limited.",
            retryable=True,
        )

    with pytest.raises(AgentWorkerError) as raised:
        retry_invalid_response(rate_limited, code_prefix="CAPABILITY")

    assert raised.value.code == "CAPABILITY_RATE_LIMITED"
    assert raised.value.retryable is True
    assert len(attempts) == 1


class _LostMessageClient:
    """A relay that answers "completed" but sometimes drops the message item."""

    def __init__(self, *outcomes: str | None):
        self._outcomes = list(outcomes)
        self.calls = 0
        self.responses = self

    def create(self, **kwargs):
        self.calls += 1
        text = self._outcomes.pop(0)
        reasoning = type("Item", (), {"type": "reasoning"})()
        message = type("Item", (), {"type": "message"})()
        return type("Response", (), {
            "status": "completed",
            "output": [reasoning] if text is None else [reasoning, message],
            "output_text": "" if text is None else text,
        })()


def test_an_answer_that_lost_its_message_is_retried_and_the_next_one_wins() -> None:
    client = _LostMessageClient(None, None, '{"verdict": "strong", "score": 8}')

    assert _call(client) == Answer(verdict="strong", score=8)
    assert client.calls == 3


def test_empty_answers_stop_after_two_retries_and_say_what_arrived() -> None:
    client = _LostMessageClient(None, None, None, '{"verdict": "strong", "score": 8}')

    with pytest.raises(AgentWorkerError) as raised:
        _call(client)

    assert client.calls == 3
    assert raised.value.code == "RESUME_ANALYSIS_EMPTY_RESPONSE"
    assert raised.value.detail == "status=completed; output=reasoning"
