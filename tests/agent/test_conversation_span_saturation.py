"""CE-1 6.4 probes that do not need a live model.

The trajectory catalogue holds the decision-level arms. These tests pin the
projection facts those arms assume: the hidden name is absent from the
compacted JSON, the span tool is offered only with a watermark, an empty span
does not smuggle the window decoy, and stuffing the originals costs visibly
more characters than paging in.
"""

from __future__ import annotations

import json

from career_agent.agent.conversation_span_presenter import render_conversation_span
from career_agent.agent.decision_messages import (
    decision_context_chars,
    project_decision_messages,
)
from career_agent.agent.main_agent_contracts import DECISION_OBSERVATION_BODY_LIMIT
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.evaluation.main_agent_scenarios import (
    SCENARIOS,
    SPAN_HIDDEN_QUESTION,
    SPAN_OUT_OF_RANGE_QUESTION,
    SPAN_PAGE_IN_FACT,
    SPAN_WINDOW_DECOY,
    _SPAN_PRE_WATERMARK,
    _pre_watermark_span_view,
    _span_empty_observation,
    _span_found_observation,
    compacted_span_context,
    stuffed_span_context,
)
from career_agent.evaluation.trajectory import check_contract


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
    ablation = _scenario("a_compacted_fact_without_page_in_is_not_invented")
    stuffed = _scenario("a_fact_still_in_the_window_is_answered_without_page_in")
    empty = _scenario("an_empty_conversation_span_is_not_filled_from_the_window")

    assert page_in.context.user_message == ablation.context.user_message == (
        stuffed.context.user_message
    )
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
    for scenario in (page_in, body, ablation, stuffed, empty):
        assert check_contract(scenario, tool_specs=schemas) == ()
