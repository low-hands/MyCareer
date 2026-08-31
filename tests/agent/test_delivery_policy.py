"""The registry that decides how a tool result is delivered.

These properties used to be four independent tables in ``MainAgentRuntime``.
Nothing held them to the same set of states, so one could be extended and the
others not — and one was, which is how a report-shaped readback ended up in the
answer-writer allow list without the bounded ``message`` that list assumes.

The tests here are the thing that was missing: they hold the registry to exactly
the states the tool layer emits, and hold every report-shaped state to the
promise its policy makes.
"""

from __future__ import annotations

import inspect
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from career_agent.agent.delivery_policy import (
    DELIVERY_POLICIES,
    DeliveryPolicy,
    condenses_message,
    is_waiting,
    policy_for,
    response_type_for,
    uses_answer_writer,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.agent.mock_interview_contracts import (
    MockInterviewExchange,
    MockInterviewQuestionSummary,
    MockInterviewQuestionView,
    MockInterviewResultView,
)
from career_agent.agent.mock_interview_presenter import (
    render_mock_interview_question,
    render_mock_interview_result,
    summarize_mock_interview_question,
    summarize_mock_interview_result,
)
from career_agent.agent.summary_text import SUMMARY_LIMIT

# Located from the module object rather than a hardcoded path, so moving the
# tool layer fails this test loudly instead of silently checking nothing.
_TOOLS_SOURCE = Path(inspect.getsourcefile(MainAgentToolRegistry))


def _emitted_states() -> set[str]:
    """Every state literal the tool layer can put on an observation.

    Read from the source rather than by exercising every tool: the point is to
    fail when a state is added, and a new state has no test to exercise it yet.
    The second pattern catches the mock interview graph mapping, whose states
    are dict values rather than ``state=`` keywords.
    """
    source = _TOOLS_SOURCE.read_text()
    states = set(re.findall(r'state="([a-z_]+)"', source))
    mapping = re.search(
        r'state = \{(.*?)\}\[result\.state\]', source, re.DOTALL
    )
    assert mapping is not None, "the graph state mapping moved or was renamed"
    states.update(re.findall(r'"(mock_interview_[a-z_]+)"', mapping.group(1)))
    return states


def test_the_registry_covers_exactly_the_states_the_tools_emit() -> None:
    emitted = _emitted_states()
    registered = set(DELIVERY_POLICIES)

    # A state the tools emit but nobody classified would silently take the
    # plain default, which is how the writer allow list and the waiting set
    # drifted apart in the first place.
    assert emitted - registered == set()
    # And the other direction, which is what removed "selection_required",
    # "waiting_user", and "detail_unavailable": three states the waiting set
    # carried that no tool had ever produced.
    assert registered - emitted == set()


def test_a_card_delivered_body_is_not_also_kept_in_the_row() -> None:
    """The second axis, and the invariant that keeps the two from re-merging.

    A card renders the body from its entity, so the row keeps prose about it.
    Declaring a card state with a full row would put the body in both places —
    the shape 052 set out to remove.
    """
    with pytest.raises(ValueError):
        DeliveryPolicy(body_delivery="resource_card")

    for state, policy in DELIVERY_POLICIES.items():
        if policy.delivers_body_elsewhere:
            assert policy.durable_message == "summary", state


def test_a_bounded_row_does_not_by_itself_shorten_the_delivery() -> None:
    """The distinction that was briefly collapsed into one flag.

    Some states need a bounded row *and* a full delivery: a verbatim readback is
    quoted because that is what was asked for, while the answer it quotes is far
    too long for the transcript to carry. Deciding both from one flag reduced
    that delivery to a receipt with nowhere to read the rest.
    """
    summarised_without_card = {
        state
        for state, policy in DELIVERY_POLICIES.items()
        if policy.condensed_message and not policy.delivers_body_elsewhere
    }
    assert "mock_interview_question_found" in summarised_without_card
    # And the reverse pairing has to keep existing, or the split is pointless.
    assert any(
        policy.delivers_body_elsewhere for policy in DELIVERY_POLICIES.values()
    )


def test_a_writer_eligible_state_must_condense_its_durable_row() -> None:
    """The invariant the mock interview readback violated.

    The writer shares a model and a key with ``decide``, so it fails for real
    reasons. When it does, the durable row falls back to ``message`` — which has
    to be a bounded line, or the fallback writes the whole report into every
    later turn's recent window.
    """
    with pytest.raises(ValueError):
        DeliveryPolicy(response_type="job_research")

    for state, policy in DELIVERY_POLICIES.items():
        if policy.uses_answer_writer:
            assert policy.condensed_message, state


def test_a_waiting_state_is_never_handed_to_the_writer() -> None:
    with pytest.raises(ValueError):
        DeliveryPolicy(waiting=True, durable_message="summary", response_type="general")

    for state, policy in DELIVERY_POLICIES.items():
        assert not (policy.waiting and policy.uses_answer_writer), state


def test_an_unregistered_state_is_delivered_plainly_rather_than_raising() -> None:
    """A result is already in hand, and may have had durable effects."""
    assert policy_for("a_state_from_the_future") == DeliveryPolicy()
    assert not is_waiting("a_state_from_the_future")
    assert not condenses_message("a_state_from_the_future")
    assert not uses_answer_writer("a_state_from_the_future")
    assert response_type_for("a_state_from_the_future") == "general"


def test_the_runtime_reads_waiting_and_writer_rules_from_the_registry() -> None:
    """No second copy of these tables survives on the runtime."""
    assert not hasattr(MainAgentRuntime, "_WAITING_STATES")
    assert is_waiting("calendar_approval_required")
    assert not is_waiting("saved_jobs_found")
    assert response_type_for("mock_interview_result_found") == "interview_report"


def _long_result_view() -> MockInterviewResultView:
    return MockInterviewResultView(
        interview_type="technical",
        status="completed",
        questions=(
            MockInterviewQuestionSummary(
                plan_item_number=1,
                question="讲一个你负责的检索可靠性改进。" * 20,
                rating="strong",
                follow_up_count=2,
            ),
        ),
        answered_count=1,
        report_id="rep-1",
        report_summary="整体表现稳定。\n" + "细节展开。" * 500,
    )


def test_the_readback_row_stays_bounded_however_long_the_report_is() -> None:
    view = _long_result_view()
    row = summarize_mock_interview_result(view)

    # The row is carried into every later turn's window, so its size cannot
    # follow the report's.
    assert len(row) < SUMMARY_LIMIT + 60
    assert "细节展开。细节展开。" not in row
    # The screen is where the report goes, and it is not the same string.
    screen = render_mock_interview_result(view)
    assert "整体表现稳定。" in screen
    assert len(screen) > len(row)


def test_a_readback_of_one_exchange_keeps_the_answer_off_the_row() -> None:
    view = MockInterviewQuestionView(
        question_number=2,
        exchanges=(
            MockInterviewExchange(
                turn_type="primary",
                question="讲一个你负责的检索可靠性改进。",
                answer="我设计了离线评估集。" * 800,
                rating="strong",
                evaluation_summary="回答具体。",
            ),
        ),
    )

    row = summarize_mock_interview_question(view)
    assert "我设计了离线评估集。" not in row
    assert len(row) < SUMMARY_LIMIT + 60
    assert "第 2 题" in row
    # Asked for verbatim, so the screen quotes it in full.
    assert "我设计了离线评估集。" in render_mock_interview_question(view)


def test_a_card_policy_without_a_reference_fails_open_to_the_full_body() -> None:
    """A broken observation must not turn a completed report into a receipt."""
    from career_agent.agent.main_agent_contracts import ToolObservation

    observation = ToolObservation(
        tool_name="get_mock_interview_result",
        state="mock_interview_result_found",
        message="已读取模拟面试（技术面，1 题）。整体表现稳定。",
        payload=_long_result_view().model_dump(mode="json"),
    )
    screen = MainAgentRuntime._assistant_message(observation)

    assert MainAgentRuntime._conversation_content(
        observation, screen=screen, composed=False
    ) == screen
    # The writer's result is still the actual screen copy and can be stored.
    assert (
        MainAgentRuntime._conversation_content(
            observation, screen="写手的摘要。", composed=True
        )
        == "写手的摘要。"
    )


def _states_that_attach_a_resource_ref() -> set[str]:
    """States the tool layer builds with a ``resource_ref``, read from source.

    Scanned rather than exercised because the point is to fail when a state is
    added, and a new one has no test to exercise it yet. Matches each
    ``ToolObservation(...)`` construction that mentions
    ``ConversationResourceReference`` and takes the ``state=`` inside it.
    """
    source = _TOOLS_SOURCE.read_text()
    states = set()
    for block in re.split(r"return ToolObservation\(|= ToolObservation\(", source)[1:]:
        head = block[: block.find("\n    def ") if "\n    def " in block else len(block)]
        if "ConversationResourceReference" not in head:
            continue
        found = re.search(r'state="([a-z_]+)"', head)
        if found:
            states.add(found.group(1))
    return states


def _graph_states_with_a_reference() -> set[str]:
    """The mock interview states, checked by building the observation.

    Their ``state`` comes from a mapping rather than a literal, so the source
    scan cannot see it. Rather than granting them a hand-written exemption —
    which is the kind of maintained list this file exists to avoid — the
    observation is actually built and asked whether it carries a reference.
    """
    from career_agent.agent.main_agent_tools import MainAgentToolRegistry
    from career_agent.agent.mock_interview_contracts import MockInterviewGraphResult
    from career_agent.domain.mock_interviews import (
        MockInterviewQuestionResult,
        MockInterviewReport,
    )

    report = MockInterviewReport(
        id="rep-1",
        session_id="sess-1",
        completion_reason="plan_completed",
        summary="整体稳定。",
        strengths=("结构清晰",),
        development_areas=("深度不足",),
        practice_actions=("多练系统设计",),
        question_results=(
            MockInterviewQuestionResult(
                plan_item_number=1,
                question="讲一个项目。",
                final_rating="adequate",
                summary="结构清晰。",
                follow_up_count=0,
            ),
        ),
        created_at=datetime(2026, 8, 31, tzinfo=timezone.utc),
    )
    states = set()
    for graph_state, kwargs in (
        ("completed", {"report_id": "rep-1", "report": report}),
        ("cancelled", {}),
        ("running", {}),
    ):
        observation = MainAgentToolRegistry._mock_interview_observation(
            MockInterviewGraphResult(
                session_id="sess-1", state=graph_state, message="…", **kwargs
            )
        )
        if observation.resource_ref is not None:
            states.add(observation.state)
    return states


def test_a_card_is_declared_only_where_a_reference_is_actually_attached() -> None:
    """The check that would have caught the two axes being merged.

    ``body_delivery="resource_card"`` is a claim that the UI can render the body
    from somewhere else. If the tool never attaches a reference, no card is
    emitted and no ``report_ready`` fires, so compressing the delivery on the
    strength of that claim discards the content instead of relocating it.
    """
    declared = {
        state
        for state, policy in DELIVERY_POLICIES.items()
        if policy.delivers_body_elsewhere
    }
    attached = _states_that_attach_a_resource_ref() | _graph_states_with_a_reference()

    unbacked = declared - attached
    assert unbacked == set(), (
        f"{sorted(unbacked)} claim a card but attach no resource_ref, so their "
        "body would be compressed with nowhere to read it"
    )
