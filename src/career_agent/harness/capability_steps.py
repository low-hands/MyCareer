"""Who gets told what a capability is doing while it runs.

A capability can spend a minute inside one tool call: several model requests,
a few web searches, a mailbox scan. The runtime only sees the call start and
end, and the workers doing the work must not know about the stream. So each
worker announces its steps through a context-local observer the runtime
installs around the tool call, carrying an internal stage label and never
the text being worked on.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

StepKind = Literal["model", "tool", "io", "retry"]


@dataclass(frozen=True)
class CapabilityStep:
    stage: str
    kind: StepKind = "model"
    index: int | None = None
    total: int | None = None


CapabilityStepObserver = Callable[[CapabilityStep], None]

_OBSERVER: ContextVar[CapabilityStepObserver | None] = ContextVar(
    "capability_step_observer", default=None
)


def notify_capability_step(
    stage: str,
    *,
    kind: StepKind = "model",
    index: int | None = None,
    total: int | None = None,
) -> None:
    observer = _OBSERVER.get()
    if observer is None:
        return
    try:
        observer(CapabilityStep(stage=stage, kind=kind, index=index, total=total))
    except Exception:
        # Presentation only; a faulty observer must not fail the capability.
        return


@contextmanager
def observing_capability_steps(
    observer: CapabilityStepObserver | None,
) -> Iterator[None]:
    token = _OBSERVER.set(observer)
    try:
        yield
    finally:
        _OBSERVER.reset(token)
