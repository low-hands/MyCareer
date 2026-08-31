"""What a report-shaped turn shows and stores, with the writer running.

053 bounded the durable row for turns the writer did not compose. It left the
composed branch alone, and that branch was the worse of the two: the writer is
told to "rewrite the supplied grounded_draft" with a 4096-token budget, and for
these states grounded_draft is the whole rendered report. A full second copy of
the report was therefore the writer's *designed* output — displayed beside the
card that already held one, and stored verbatim into every later turn's window.

The delivery matrix has three intentional shapes: a full durable message; a
bounded message plus an entity-backed card; and a full ephemeral message whose
durable row is bounded. Live and refreshed prose are equal only in the first
two. The third is used by Daily Brief and Resume Analysis, whose complete body
must be shown now but should not occupy every later recent window.
"""

from __future__ import annotations

import pytest

from career_agent.agent.answer_writer import AnswerCompositionRequest
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    CareerProfileContext,
    ConversationResourceReference,
    ConversationTaskState,
    MainAgentContext,
    ToolObservation,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime, MainAgentTurnResult
from career_agent.agent.summary_text import DELIVERY_SUMMARY_LIMIT
from career_agent.harness.streaming import ContentDeltaEvent

_REPORT_BODY = "# 公司调研\n\n这家公司主营企业级搜索。\n" + "细节段落。" * 800


class _Writer:
    """A writer that ignores the length instruction, the way a model can."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.request: AnswerCompositionRequest | None = None

    def stream(self, request):
        self.request = request
        for index in range(0, len(self.text), 40):
            yield self.text[index : index + 40]


def _result(*, state: str = "job_research_ready") -> MainAgentTurnResult:
    observation = ToolObservation(
        tool_name="research_job",
        state=state,
        message="岗位研究已完成。这家公司主营企业级搜索。",
        resource_ref=(
            ConversationResourceReference(
                kind="job_research_report",
                resource_id="report-1",
                status_at_delivery="current",
                anchored_by_other_job=False,
            )
            if state == "job_research_ready"
            else None
        ),
    )
    return MainAgentTurnResult(
        decision=AgentDecision(action="final", message=""),
        context=MainAgentContext(
            conversation_id="c1",
            profile=CareerProfileContext(user_id="u1"),
            task=ConversationTaskState(),
            user_message="这家公司怎么样",
        ),
        assistant_message=_REPORT_BODY,
        tool_result=observation,
        tool_results=(observation,),
    )


def _runtime(writer) -> MainAgentRuntime:
    return MainAgentRuntime(
        context_manager=object(),
        decision_maker=object(),
        tools=object(),
        answer_writer=writer,
    )


def _stream(agent, result) -> list[str]:
    import career_agent.agent.main_agent_runtime as runtime_module

    seen: list[str] = []
    token = runtime_module._STREAM_SINK.set(
        lambda event: seen.append(event.delta)
        if isinstance(event, ContentDeltaEvent)
        else None
    )
    try:
        agent._stream_answer_if_eligible(
            result=result, conversation_id="c1", user_request="这家公司怎么样"
        )
    finally:
        runtime_module._STREAM_SINK.reset(token)
    return seen


def test_the_writer_is_told_the_body_is_delivered_elsewhere() -> None:
    writer = _Writer("这家公司主营企业级搜索，调研覆盖产品线与竞争格局。")
    _runtime(writer)._stream_answer_if_eligible(
        result=_result(), conversation_id="c1", user_request="这家公司怎么样"
    )

    assert writer.request.max_chars == DELIVERY_SUMMARY_LIMIT
    # The full report still goes in, because a grounded summary has to be
    # written from the thing it summarises.
    assert writer.request.grounded_draft == _REPORT_BODY


def test_an_overlong_answer_is_cut_rather_than_stored_whole() -> None:
    """The instruction is a request; this is the enforcement."""
    writer = _Writer(_REPORT_BODY)
    result = _result()

    _runtime(writer)._stream_answer_if_eligible(
        result=result, conversation_id="c1", user_request="这家公司怎么样"
    )

    assert result.content_streamed is True
    assert len(result.assistant_message) <= DELIVERY_SUMMARY_LIMIT
    assert result.assistant_message.endswith("…")
    # And the row that the cut answer produces is that same cut answer.
    stored = MainAgentRuntime._conversation_content(
        result.tool_result, screen=result.assistant_message, composed=True
    )
    assert stored == result.assistant_message


def test_a_short_answer_passes_through_untouched() -> None:
    text = "这家公司主营企业级搜索，报告覆盖产品线、竞争格局与公开风险。"
    writer = _Writer(text)
    result = _result()

    _runtime(writer)._stream_answer_if_eligible(
        result=result, conversation_id="c1", user_request="这家公司怎么样"
    )

    assert result.assistant_message == text
    assert "…" not in result.assistant_message


def test_a_turn_that_is_not_report_shaped_keeps_the_full_writer_budget() -> None:
    """The cap belongs to states whose body has somewhere else to go."""
    writer = _Writer("普通回答。" * 300)
    result = _result(state="interview_retro_recorded")
    result.tool_result = ToolObservation(
        tool_name="get_saved_job",
        state="saved_job_ready",
        message="已读取这个岗位。",
    )
    result.decision = AgentDecision(action="final", message="x")

    _runtime(writer)._stream_answer_if_eligible(
        result=result, conversation_id="c1", user_request="看看这个岗位"
    )

    # Not report-shaped, so the writer never ran at all here — and if it had,
    # it would not have been capped.
    assert writer.request is None or writer.request.max_chars is None


@pytest.mark.parametrize("composed", [True, False])
def test_the_stored_row_is_bounded_even_when_handed_an_unbounded_screen(
    composed,
) -> None:
    """The row's own guard, independent of the stream that normally cuts.

    Both branches are fed the whole report. Without the writer that is the real
    input; with the writer it is the input only if the stream cap were bypassed
    — which is exactly the case a second check exists for.
    """
    result = _result()

    stored = MainAgentRuntime._conversation_content(
        result.tool_result, screen=_REPORT_BODY, composed=composed
    )

    assert len(stored) <= DELIVERY_SUMMARY_LIMIT
    if composed:
        # Clamped prose keeps its opening, so body text appearing here is
        # correct — what matters is that it is bounded and says it was cut.
        assert stored.endswith("…")
    else:
        # No writer ran, so the row is the tool's own line and carries none of
        # the body at all.
        assert stored == result.tool_result.message
        assert "细节段落" not in stored


def _streamed(result) -> str:
    """Everything the reader actually receives as message content."""
    import career_agent.agent.main_agent_runtime as runtime

    deltas: list[str] = []
    token = runtime._STREAM_SINK.set(
        lambda event: deltas.append(event.delta)
        if isinstance(event, ContentDeltaEvent)
        else None
    )
    try:
        MainAgentRuntime._deliver_stream_events(
            _runtime(None), result=result, turn_id="t1", conversation_id="c1"
        )
    finally:
        runtime._STREAM_SINK.reset(token)
    return "".join(deltas)


def test_a_state_with_no_card_streams_its_body_in_full() -> None:
    """The regression that merging the two axes caused.

    A single mock interview question is quoted verbatim because that is what was
    asked for, and no card exists to hold it. Its row still has to be bounded —
    the answer runs to twenty thousand characters — but compressing the delivery
    on the strength of that leaves the reader a one-line receipt and nowhere to
    read the rest.
    """
    body = "主问题\n讲一个项目。\n\n你的回答\n" + "我设计了离线评估集。" * 300
    observation = ToolObservation(
        tool_name="get_mock_interview_result",
        state="mock_interview_question_found",
        message="已读取模拟面试第 3 题的完整问答。讲一个项目。",
    )
    result = _result()
    result.tool_result = observation
    result.tool_results = (observation,)
    result.assistant_message = body

    streamed = _streamed(result)

    assert streamed == body
    # And the row is still the bounded line, which is the other half.
    assert MainAgentRuntime._conversation_content(
        observation, screen=body, composed=False
    ) == observation.message


def test_a_state_with_a_card_streams_the_row_and_not_the_body() -> None:
    """The pairing that makes live and refresh agree where a card exists."""
    result = _result()

    streamed = _streamed(result)

    assert streamed == result.tool_result.message
    assert "细节段落" not in streamed


def test_the_writer_ceiling_follows_the_card_not_the_row() -> None:
    """A bounded row is not a reason to shorten the answer.

    Today the summary-without-card states have no presenter, so their screen and
    their row are the same short string and the ceiling costs nothing either
    way. That is exactly why this is asserted rather than left to be noticed:
    the moment one of them gains a body, keying the ceiling on the row would
    truncate a delivery that has nowhere else to go.
    """
    observation = ToolObservation(
        tool_name="analyze_resume",
        state="resume_analysis_ready",
        message="已分析该简历版本，提取出 12 段候选经历。",
    )
    result = _result()
    result.tool_result = observation
    result.tool_results = (observation,)
    result.assistant_message = "候选经历明细。" * 400

    writer = _Writer("这份简历提取出 12 段经历。")
    _runtime(writer)._stream_answer_if_eligible(
        result=result, conversation_id="c1", user_request="分析一下我的简历"
    )

    assert writer.request is not None
    assert writer.request.max_chars is None


def test_a_message_body_with_writer_is_full_live_and_an_exact_receipt_after_refresh() -> None:
    """The one cell where live and durable prose intentionally differ.

    The body has no card, so the writer is uncapped and every emitted character
    reaches the current screen. The row is still summary-mode, but an arbitrary
    prefix is not a summary: commit keeps the tool's accurate receipt instead
    of making tomorrow's context carry a partial brief that looks complete.
    """
    observation = ToolObservation(
        tool_name="get_daily_brief",
        state="daily_brief_ready",
        message="今日职业简报包含 18 个待办事项。",
    )
    result = _result()
    result.tool_result = observation
    result.tool_results = (observation,)
    full_body = (
        "## 已逾期\n" + "跟进投递。" * 180
        + "\n## 等待对方回复\n这是不能从历史记录中悄悄消失的最后一节。"
    )
    writer = _Writer(full_body)

    live = "".join(_stream(_runtime(writer), result))
    stored = MainAgentRuntime._conversation_content(
        observation,
        screen=result.assistant_message,
        composed=result.content_streamed,
    )

    assert writer.request is not None
    assert writer.request.max_chars is None
    assert live == full_body
    assert result.assistant_message == full_body
    assert live != stored
    assert stored == observation.message
    assert "## 已逾期" not in stored
    assert "等待对方回复" not in stored
    assert not stored.endswith("…")


def test_a_summarising_writer_is_not_also_told_to_preserve_every_fact() -> None:
    """Two instructions that cannot both be obeyed leave the model to choose.

    "Preserve every factual value" and a character ceiling contradict each other
    when grounded_draft is a whole report. The summary rules ask for accuracy
    and retained uncertainty on what is said, not for saying everything.
    """
    writer = _Writer("这家公司主营企业级搜索。")
    _runtime(writer)._stream_answer_if_eligible(
        result=_result(), conversation_id="c1", user_request="这家公司怎么样"
    )

    rules = " ".join(writer.request.required_rules)
    assert "Preserve every factual value" not in rules
    assert "Do not add facts" in rules
    assert "keep the uncertainty" in rules
    assert "warning" in rules
