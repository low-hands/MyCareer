from __future__ import annotations

SUMMARY_LIMIT = 220

DELIVERY_SUMMARY_LIMIT = 600
"""How long a report-shaped turn's prose may be, writer or no writer.

Larger than ``SUMMARY_LIMIT`` because this is a written answer to a question
rather than a mechanical first line, and small enough that ten of them fit in
the recent window with room to spare. The report body is not what this bounds:
that goes to the report card, which is where a reader can take it at their own
length.
"""


MODEL_REPLY_LIMIT = 8_000
"""How long the model's own reply may be when nothing else bounds it.

``DELIVERY_SUMMARY_LIMIT`` is the ceiling for prose that accompanies a card,
where the body is delivered elsewhere and the message is only about it. A reply
with no card behind it is the whole answer — an explanation, a multi-step
summary, resume advice — and holding it to the card ceiling would truncate
answers the old writer never touched, because the writer only ever restated
card-backed reports. This bound exists to keep one turn from filling the recent
window on its own, not to summarise; it stays well under the context manager's
32k message ceiling.
"""


def condense(text: str, *, limit: int = SUMMARY_LIMIT) -> str:
    """Reduce a report's own summary to one bounded line.

    Used for the copy that goes into the durable conversation row when no
    answer writer composed a reply. That row is carried into every later turn's
    recent window, so it has to be short; it also has to be produced without a
    model call, because the case it exists for is the writer not running.

    Takes the first line rather than re-summarising: a report's ``summary``
    field already opens with its own headline, so the first line is the closest
    thing to an abstract that is available deterministically.
    """
    stripped = text.strip()
    if not stripped:
        return ""
    first = stripped.splitlines()[0].strip()
    if len(first) <= limit:
        return first
    return first[: limit - 1].rstrip() + "…"


def clamp(text: str, *, limit: int = DELIVERY_SUMMARY_LIMIT) -> str:
    """Bound a written answer without reshaping it.

    Distinct from ``condense``, which takes the first line: that is right for a
    report's own headline and wrong for prose, where it would silently drop
    every paragraph after the first instead of marking a cut.
    """
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 1].rstrip() + "…"
