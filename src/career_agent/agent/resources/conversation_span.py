"""Recognise a user message that names an exact conversation sequence span.

A user who writes "把序号 100 到 110 的对话读回来" has already supplied the
only two arguments ``read_conversation_span`` needs. Leaving that call to
the model let it ask the user to repeat the range instead, so the runtime
reads the named span first and hands the observation to the model.

Only an unambiguous request qualifies: the numbers must be introduced as
sequence numbers (序号/编号/第 N 条) and that very phrase must name the
conversation (对话/消息/聊天记录...), so "薪资 30 到 50K" or "第 3 到 5 条
岗位" never turn into a history read. A conversation word elsewhere in the
message is not enough: "对话里序号 3 到 5 的岗位" and "第 3 轮面试的聊天
记录" are about jobs and an interview, not message numbers. Anything
unclear is left to the model.
"""

from __future__ import annotations

import re
from typing import NamedTuple

_RANGE_SEPARATOR = r"\s*(?:到|至|~|～|-|—|–|through|thru|to)\s*"
_CONVERSATION_NOUN = (
    r"(?:对话|消息|聊天记录|聊天|会话消息|会话记录|会话|发言|"
    r"conversation|messages?|chat|history)"
)
# 的对话 / 那条消息 / 这句对话 / 里的消息
_NOUN_LINK = r"\s*(?:的|那|这|里的|中的)?\s*(?:[一两几]?(?:条|句))?\s*"

# 序号 100 到 110 的对话 / 编号100-110的消息 / 会话序号 12 / conversation sequence 3 through 7
_KEYED = re.compile(
    rf"(?:(?P<pre>{_CONVERSATION_NOUN})\s*)?"
    r"(?:序号|编号|sequence|seq\.?)\s*#?\s*(?P<start>\d+)"
    rf"(?:{_RANGE_SEPARATOR}(?:序号|编号|#)?\s*(?P<end>\d+))?"
    rf"(?:{_NOUN_LINK}(?P<post>{_CONVERSATION_NOUN}))?",
    re.IGNORECASE,
)
# 第 3 到 7 条消息 / 第3至7句对话 / 第 5 条聊天记录
_ORDINAL = re.compile(
    r"第\s*(?P<start>\d+)"
    rf"(?:{_RANGE_SEPARATOR}(?:第\s*)?(?P<end>\d+))?"
    r"\s*(?:条|句)\s*(?:的)?\s*"
    rf"(?P<post>{_CONVERSATION_NOUN})?",
    re.IGNORECASE,
)


class ExplicitSequenceSpan(NamedTuple):
    from_sequence: int
    through_sequence: int


def explicit_sequence_span(user_message: str) -> ExplicitSequenceSpan | None:
    """The single sequence span the message names, or ``None``.

    ``None`` also covers a reversed or zero-based span, a message naming
    more than one span, and a numbered phrase that does not itself name the
    conversation: those need the user or the model, not a guess.
    """

    spans: set[ExplicitSequenceSpan] = set()
    bound: set[ExplicitSequenceSpan] = set()
    for pattern in (_KEYED, _ORDINAL):
        for match in pattern.finditer(user_message):
            start = int(match.group("start"))
            end = int(match.group("end")) if match.group("end") is not None else start
            span = ExplicitSequenceSpan(start, end)
            spans.add(span)
            if match.group("post") or match.groupdict().get("pre"):
                bound.add(span)
    if len(spans) != 1:
        return None
    span = next(iter(spans))
    if span not in bound:
        return None
    if span.from_sequence < 1 or span.from_sequence > span.through_sequence:
        return None
    return span
