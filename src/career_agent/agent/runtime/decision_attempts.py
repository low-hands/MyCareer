"""Who gets told when the decision model is asked again.

The decision maker retries a stalled or refused request itself, and a retry
is the one thing the person waiting can otherwise never see: the wait just
gets longer. The runtime owns the stream, the decision maker owns the retry
loop, and neither should import the other for this, so the loop announces
each attempt through a context-local observer the runtime installs around
``decide``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionAttempt:
    attempt: int
    max_attempts: int
    elapsed_seconds: float
    previous_error_code: str | None = None


DecisionAttemptObserver = Callable[[DecisionAttempt], None]

_OBSERVER: ContextVar[DecisionAttemptObserver | None] = ContextVar(
    "decision_attempt_observer", default=None
)


def notify_decision_attempt(attempt: DecisionAttempt) -> None:
    observer = _OBSERVER.get()
    if observer is None:
        return
    try:
        observer(attempt)
    except Exception:
        # Presentation only; a faulty observer must not fail the decision.
        return


@contextmanager
def observing_decision_attempts(
    observer: DecisionAttemptObserver | None,
) -> Iterator[None]:
    token = _OBSERVER.set(observer)
    try:
        yield
    finally:
        _OBSERVER.reset(token)
