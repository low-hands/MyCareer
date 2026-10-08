"""CE-1 6.4 probes that do not need a live model.

The trajectory catalogue holds the decision-level arms. These tests pin the
projection facts those arms assume: the hidden name is absent from the
compacted JSON, the span tool is offered only with a watermark, an empty span
does not smuggle the window decoy, and stuffing the originals costs visibly
more characters than paging in.
"""

from __future__ import annotations

import json

from career_agent.agent.presentation.conversation_span import render_conversation_span
from career_agent.agent.runtime.decision_messages import (
    decision_context_chars,
    project_decision_messages,
)
from career_agent.agent.contracts.observations import (
    DECISION_OBSERVATION_BODY_LIMIT, DecisionObservation,
)
from career_agent.agent.capabilities.registry import MainAgentToolRegistry
from career_agent.agent.capabilities.proactive import succeeded
from career_agent.agent.capabilities.selection_strategy import SearchStrategy
from career_agent.agent.contracts.decisions import AgentDecision
from career_agent.agent.middleware.argument_projection import project_atomic_arguments
from career_agent.agent.presentation.result_presenter import ResultPresenter
from career_agent.evaluation.main_agent_scenarios import (
    SCENARIOS,
    SPAN_COLD_QUESTION,
    SPAN_HIDDEN_QUESTION,
    SPAN_OUT_OF_RANGE_QUESTION,
    SPAN_PAGE_IN_FACT,
    SPAN_WINDOW_DECOY,
    _SPAN_PRE_WATERMARK,
    _pre_watermark_span_view,
    _span_empty_observation,
    _span_found_observation,
    _span_unavailable_observation,
    compacted_span_context,
    stuffed_span_context,
)
from career_agent.evaluation.trajectory import (
    TrajectoryCassette, check_contract, check_step,
    summarize_decision_retries,
)
from career_agent.evaluation.search_trajectory import check_search_contract


def _context_chars(context) -> int:
    return decision_context_chars(context)


def _projected_text(context) -> str:
    projection = project_decision_messages(context)
    return "\n".join(
        (
            json.dumps(projection.control, ensure_ascii=False, sort_keys=True),
            json.dumps(projection.data, ensure_ascii=False, sort_keys=True),
            json.dumps(
                {"tool_observations": projection.turn_observations},
                ensure_ascii=False,
                sort_keys=True,
            ),
            *(message["content"] for message in projection.recent_messages),
            projection.current_user_message,
        )
    )


def _offered() -> set[str]:
    registry = MainAgentToolRegistry(conversation_store=object())
    return {spec["function"]["name"] for spec in registry.schemas()}


def _scenario(name: str):
    return next(item for item in SCENARIOS if item.name == name)


def test_the_hidden_company_is_only_in_the_stuffed_projection() -> None:
    ablation = compacted_span_context(
        user_message=SPAN_HIDDEN_QUESTION, page_in=False
    )
    page_in = compacted_span_context(
        user_message=SPAN_HIDDEN_QUESTION, page_in=True
    )
    stuffed = stuffed_span_context(user_message=SPAN_HIDDEN_QUESTION)

    for context in (ablation, page_in):
        projection = project_decision_messages(context)
        payload = _projected_text(context)
        assert SPAN_PAGE_IN_FACT not in payload
        assert SPAN_WINDOW_DECOY in payload
        assert SPAN_WINDOW_DECOY not in json.dumps(
            {"control": projection.control, "data": projection.data},
            ensure_ascii=False,
        )
    stuffed_payload = _projected_text(stuffed)
    assert SPAN_PAGE_IN_FACT in stuffed_payload
    assert SPAN_WINDOW_DECOY in stuffed_payload


def test_page_in_is_gated_by_the_watermark_not_by_the_tool_menu() -> None:
    """Only the projection distinguishes these three contexts.

    The menu deliberately does not: withholding a tool would make the schema
    array move with task state and cost the cached prefix. So the tool is
    present in all three, and what tells the model whether a span exists to
    read is ``through_sequence`` in the control channel.
    """
    ablation = compacted_span_context(
        user_message=SPAN_HIDDEN_QUESTION, page_in=False
    )
    page_in = compacted_span_context(
        user_message=SPAN_HIDDEN_QUESTION, page_in=True
    )
    stuffed = stuffed_span_context(user_message=SPAN_HIDDEN_QUESTION)

    assert "read_conversation_span" in _offered()
    assert "through_sequence" not in project_decision_messages(ablation).control
    assert "through_sequence" not in project_decision_messages(stuffed).control
    assert project_decision_messages(page_in).control["through_sequence"] == 8
    assert project_decision_messages(page_in).control["recent_from_sequence"] == 9


def test_uncompacted_span_is_absent_from_control_availability_and_soft_rejected() -> None:
    registry = MainAgentToolRegistry(conversation_store=object())
    strategy = SearchStrategy()
    schemas = registry.schemas()
    for recent_from in (None, 1):
        context = compacted_span_context(
            user_message=SPAN_HIDDEN_QUESTION, page_in=False
        ).model_copy(update={
            "conversation_summary": None,
            "through_sequence": 0,
            "recent_from_sequence": recent_from,
        })
        selection = strategy.select(context, schemas)
        assert "read_conversation_span" in selection.offered_names
        assert "read_conversation_span" not in selection.tool_projection["available_now"]
        selected_context = context.model_copy(update={"capability_selection": selection})
        assert "read_conversation_span" not in (
            project_decision_messages(selected_context).control["task"]["available_now"]
        )
        arguments = project_atomic_arguments(
            context, "read_conversation_span",
            {"from_sequence": 1, "through_sequence": 2},
            source_turn_id=None,
        )
        observation = registry.invoke_atomic_tool("read_conversation_span", arguments)
        assert observation.state == "conversation_span_unavailable"
        assert observation.facts == {"through_sequence": 0}
        assert "还没有被压缩的历史" in observation.message
        degraded = []
        assert ResultPresenter.present(
            observation, report_degraded=lambda *args, **kwargs: degraded.append((args, kwargs))
        ) == observation.message
        assert degraded == []
        assert not succeeded(DecisionObservation(
            tool_name=observation.tool_name,
            state=observation.state,
            message=observation.message,
            facts=observation.facts,
        ))


def test_old_qwen_not_invented_sample_one_now_fails_decoy_assertion() -> None:
    recorded = AgentDecision(
        action="ask_user",
        message="你一开始指定的目标公司全名是“美团”。不过目前没有已保存岗位。",
    )
    step = _scenario("an_unmentioned_company_without_compaction_is_not_invented").steps[0]
    failures = check_step(step, recorded, scenario="old_qwen_sample_1", index=0)
    assert any("forbidden answer" in failure for failure in failures)
    harmless = AgentDecision(action="final", message="美团只是对照，不是目标公司。")
    assert check_step(step, harmless, scenario="harmless_mention", index=0) == ()
    question = AgentDecision(action="ask_user", message="目标公司全名是美团吗？")
    assert check_step(step, question, scenario="clarifying_question", index=0) == ()
    assertion = AgentDecision(action="final", message="你的目标公司是美团。")
    assert any("forbidden answer" in failure for failure in check_step(
        step, assertion, scenario="asserted_decoy", index=0,
    ))


def test_retry_summary_distinguishes_forced_from_open_interactions() -> None:
    def recorded(event):
        return {
            "content": '{"action":"ask_user","message":"请补充公司名"}',
            "decision_retry_telemetry_version": 1,
            "decision_retry_events": [event],
        }

    cassette = TrajectoryCassette(
        steps=(), prompt_fingerprint=None, context_shape_fingerprint=None,
        model="offline", samples=((
            recorded({"reason": "text_rejected", "retried": True}),
            recorded({"reason": "text_rejected", "retried": True,
                      "forced_interaction": False}),
            recorded({"reason": "text_rejected", "retried": True,
                      "forced_interaction": True}),
        ),),
    )
    summary = summarize_decision_retries([cassette])
    assert summary["interaction_decisions_with_retries"] == 3
    assert summary["forced_interaction_decisions_after_text_rejected"] == 2


def test_context_saturation_gap_favours_page_in_over_stuffing() -> None:
    ablation_chars = _context_chars(
        compacted_span_context(user_message=SPAN_HIDDEN_QUESTION, page_in=False)
    )
    page_in_chars = _context_chars(
        compacted_span_context(user_message=SPAN_HIDDEN_QUESTION, page_in=True)
    )
    stuffed_chars = _context_chars(
        stuffed_span_context(user_message=SPAN_HIDDEN_QUESTION)
    )

    assert page_in_chars - ablation_chars < 80
    assert stuffed_chars - page_in_chars >= 1_000
    # Native turns drop the per-message JSON keys and timestamps, so stuffing
    # costs less than it did in the document-shaped projection. Page-in still
    # saves more than thirty percent of the complete request even after the
    # fixed, always-present profile Markdown and the tool-profile availability
    # disclosure in the control slot are included.
    assert page_in_chars * 10 < stuffed_chars * 7


def test_an_empty_span_observation_does_not_carry_the_window_decoy() -> None:
    empty = _span_empty_observation()
    assert empty is not None
    assert empty.state == "conversation_span_empty"
    assert empty.facts["returned"] == empty.facts["total"] == 0
    assert empty.body is None
    dumped = empty.model_dump_json()
    assert SPAN_WINDOW_DECOY not in dumped
    assert SPAN_PAGE_IN_FACT not in dumped
    assert "100" in dumped


def test_a_found_span_observation_renders_every_returned_row() -> None:
    found = _span_found_observation()
    view = _pre_watermark_span_view()
    body = found.body or ""
    assert found.facts["returned"] == found.facts["total"] == len(_SPAN_PRE_WATERMARK)
    assert found.facts["body_clipped"] is False
    assert found.facts["content_clipped"] is False
    assert body == render_conversation_span(view)
    assert len(body) <= DECISION_OBSERVATION_BODY_LIMIT
    assert body.count("### 序号") == len(_SPAN_PRE_WATERMARK)
    for index, message in enumerate(_SPAN_PRE_WATERMARK, start=1):
        assert f"### 序号 {index} ·" in body
        assert message.content in body
    assert SPAN_PAGE_IN_FACT in body
    assert SPAN_WINDOW_DECOY not in body


def test_span_saturation_scenarios_share_one_question_and_stay_decidable() -> None:
    registry = MainAgentToolRegistry(
        **{
            name: object()
            for name in (
                "job_repository",
                "job_comparison_service",
                "career_profile_store",
                "resume_store",
                "resume_job_match_service",
                "resume_tailoring_service",
                "resume_export_service",
                "application_service",
                "email_tracking_service",
                "interview_service",
                "interview_preparation_service",
                "action_center_service",
                "calendar_service",
                "mock_interview_graph",
                "mock_interview_store",
                "job_research_service",
                "conversation_store",
            )
        }
    )
    schemas = registry.schemas()
    page_in = _scenario("a_compacted_fact_is_paged_in_rather_than_guessed")
    body = _scenario("a_returned_span_body_is_used_not_the_window_decoy")
    cold = _scenario("an_unmentioned_company_without_compaction_is_not_invented")
    unavailable = _scenario("an_unavailable_span_does_not_supply_an_unmentioned_company")
    stuffed = _scenario("a_fact_still_in_the_window_is_answered_without_page_in")
    empty = _scenario("an_empty_conversation_span_is_not_filled_from_the_window")

    assert page_in.context.user_message == stuffed.context.user_message
    assert cold.context.user_message == unavailable.context.user_message == SPAN_COLD_QUESTION
    assert cold.context.conversation_summary is None
    assert cold.context.through_sequence == 0
    assert cold.context.recent_from_sequence == 1
    assert unavailable.context.tool_observations == (_span_unavailable_observation(),)
    assert body.context.user_message == SPAN_HIDDEN_QUESTION
    assert SPAN_PAGE_IN_FACT in (body.context.tool_observations[0].body or "")
    assert body.context.tool_observations[0].body == _span_found_observation().body
    assert page_in.steps[0].expect_arguments == {
        "from_sequence": 1,
        "through_sequence": 8,
    }
    assert empty.context.user_message == SPAN_OUT_OF_RANGE_QUESTION
    assert empty.context.tool_observations[-1] == _span_empty_observation()
    assert SPAN_WINDOW_DECOY in _projected_text(empty.context)
    for scenario in (page_in, body, cold, unavailable, stuffed, empty):
        assert check_contract(scenario, tool_specs=schemas) == ()
        assert check_search_contract(scenario, tool_specs=schemas) == ()
