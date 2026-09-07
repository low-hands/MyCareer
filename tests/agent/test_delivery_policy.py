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

import ast
import inspect
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from career_agent.agent.delivery_policy import (
    DELIVERY_POLICIES,
    DeliveryPolicy,
    condenses_message,
    is_failed,
    is_waiting,
    policy_for,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_runtime import MainAgentTurnResult, ModelDecision
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    CareerProfileContext,
    ConversationTaskState,
    MainAgentContext,
    ToolObservation,
)
from career_agent.agent.job_comparison_presenter import DIMENSION_ORDER
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
    # ``invalid_input`` is emitted by the runtime's projection-refusal path,
    # not spelled as a tool-layer state literal. It belongs to the same
    # single-source-of-truth check, so the runtime source joins the scan.
    runtime_source = Path(inspect.getsourcefile(MainAgentRuntime)).read_text()
    states.update(re.findall(r'state="([a-z_]+)"', runtime_source))
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


def test_an_unregistered_state_is_delivered_plainly_rather_than_raising() -> None:
    """A result is already in hand, and may have had durable effects."""
    assert policy_for("a_state_from_the_future") == DeliveryPolicy()
    assert not is_waiting("a_state_from_the_future")
    assert not condenses_message("a_state_from_the_future")


def test_body_eligibility_is_exactly_condensed_delivery_and_never_raw_payload(
    monkeypatch,
) -> None:
    """H reuses the presenter boundary; internal payload IDs never bypass it."""

    internal_id = "internal-job-posting-id-should-not-cross"
    monkeypatch.setattr(
        MainAgentRuntime,
        "_assistant_message",
        staticmethod(lambda result: "SAFE PRESENTER BODY"),
    )
    for state, policy in DELIVERY_POLICIES.items():
        observation = MainAgentRuntime._tool_observation(
            "probe",
            ToolObservation(
                tool_name="probe",
                state=state,
                message="安全收据。",
                payload={"job_posting_id": internal_id},
            ),
        )
        assert (observation.body is not None) == policy.condensed_message, state
        assert internal_id not in observation.model_dump_json(), state


def test_real_condensed_presenters_do_not_render_internal_identifiers() -> None:
    """The presenter boundary itself must stay clean, not only hide raw payloads.

    The monkeypatched registry test above proves that payload cannot bypass the
    presenter. This complementary test runs every real condensed presenter with
    a UUID-shaped internal identifier in an actual payload field. A distinct
    visible marker also proves that validation reached the intended presenter
    instead of silently degrading to the receipt.
    """

    internal_id = "a" * 32
    cases = {
        "conversation_span_found": (
            {
                "from_sequence": 1,
                "through_sequence": 1,
                "returned": 1,
                "total": 1,
                "messages": [
                    {
                        "sequence": 1,
                        "role": "user",
                        "content": "真实会话回读标记",
                        "created_at": "2026-08-31T00:00:00Z",
                    }
                ],
            },
            "真实会话回读标记",
        ),
        "daily_brief_ready": (
            {
                "due_today": [
                    {"title": "真实日报标记", "summary": "今天处理", "id": internal_id}
                ],
                "job_posting_id": internal_id,
            },
            "真实日报标记",
        ),
        "interview_preparation_ready": (
            {
                "preparation": {
                    "summary": "真实面试准备标记",
                    "focus_areas": [],
                    "evidence_stories": [],
                    "likely_questions": [],
                    "gaps": [],
                    "questions_to_ask": [],
                    "checklist": [],
                    "limitations": [],
                },
                "application_id": internal_id,
            },
            "真实面试准备标记",
        ),
        "interview_retro_recorded": (
            {
                "report_id": internal_id,
                "source_notes": "真实复盘原始标记",
                "summary": "真实复盘摘要标记",
                "self_assessment": "uncertain",
            },
            "真实复盘摘要标记",
        ),
        "job_research_ready": (
            {
                "report_id": internal_id,
                "research": {
                    "summary": "真实调研标记",
                    "findings": [
                        {
                            "topic": "产品线",
                            "statement": "真实调研结论标记",
                            "evidence_type": "fact",
                            "source_keys": ["S1"],
                            "confidence": "high",
                        }
                    ],
                    "open_questions": [],
                    "limitations": [],
                },
                "sources": [
                    {
                        "source_key": "S1",
                        "url": "https://example.com/product",
                        "title": "公开产品资料",
                        "publisher": "示例科技",
                        "relevant_excerpt": "公开资料中的真实来源标记。",
                    }
                ],
            },
            "真实调研标记",
        ),
        "mock_interview_completed": (
            {
                "session_id": internal_id,
                "state": "completed",
                "message": "真实模拟面试内部状态",
            },
            "模拟面试完成",
        ),
        "mock_interview_result_found": (
            {
                "interview_type": "真实模拟结果标记",
                "status": "completed",
                "questions": [],
                "answered_count": 0,
                "report_id": internal_id,
                "report_summary": "真实模拟总结标记",
            },
            "真实模拟结果标记",
        ),
        "mock_interview_question_found": (
            {
                "question_number": 1,
                "exchanges": [
                    {
                        "turn_type": "primary",
                        "question": "真实模拟问题标记",
                        "answer": "真实模拟回答标记",
                        }
                    ],
            },
            "真实模拟问题标记",
        ),
        "resume_analysis_ready": (
            {
                "analysis_id": internal_id,
                "records": [],
                "clarification_questions": [],
                "warnings": ["真实简历分析标记"],
            },
            "真实简历分析标记",
        ),
        "resume_job_match_ready": (
            {
                "match_id": internal_id,
                "overall_fit": "moderate",
                "summary": "真实匹配标记",
                "requirements": [],
                "recommendations": [],
                "clarification_questions": [],
                "limitations": [],
            },
            "真实匹配标记",
        ),
        "resume_tailoring_draft_ready": (
            {
                "draft_id": internal_id,
                "strategy_summary": "真实定制标记",
                "changes": [],
                "preserved_strengths": [],
                "unresolved_gaps": [],
                "clarification_questions": [],
                "warnings": [],
            },
            "真实定制标记",
        ),
        "saved_jobs_compared": (
            {
                "job_posting_id": internal_id,
                "comparison": {
                    "rows": [
                        {
                            # The row model itself carries an internal id, so
                            # this state proves the presenter drops one it holds
                            # rather than merely one left in the raw payload.
                            "job_posting_id": internal_id,
                            "title": "真实对比标记",
                            "company_name": "示例科技",
                            "cells": [
                                {
                                    "dimension": dimension,
                                    "value": value,
                                    "basis": "示例依据",
                                }
                                for dimension, value in zip(
                                    DIMENSION_ORDER,
                                    (
                                        "insufficient_evidence",
                                        "partial",
                                        "undisclosed",
                                        "differs",
                                        "active",
                                    ),
                                )
                            ],
                        }
                    ],
                    "uninformative_dimensions": [],
                    "jobs_without_match": [],
                    "notes": [],
                },
            },
            "真实对比标记",
        ),
        "saved_job_ready": (
            {
                "job_posting_id": internal_id,
                "jd_snapshot": {"content": "真实 JD 正文标记"},
            },
            "真实 JD 正文标记",
        ),
        "claim_source_found": (
            {
                "source_quote": "真实来源引文标记",
                "source_ref": f"evidence_{'b' * 24}",
                "body_clipped": False,
            },
            "真实来源引文标记",
        ),
        "career_memory_detail_found": (
            {
                "body": "真实职业声明详情标记",
                "lineage_ref": f"lineage_{'c' * 24}",
                "supported_by": [],
                "lineage": [],
            },
            "真实职业声明详情标记",
        ),
        "career_memory_search_found": (
            {
                "body": "真实当前职业声明标记",
                "query": "retrieval",
                "items": [],
                "total": 1,
            },
            "真实当前职业声明标记",
        ),
        "career_history_found": (
            {
                "body": "真实历史声明标记",
                "query": "retrieval",
                "items": [],
                "total": 1,
            },
            "真实历史声明标记",
        ),
    }
    condensed = {
        state
        for state, policy in DELIVERY_POLICIES.items()
        if policy.condensed_message
    }
    assert set(cases) == condensed

    internal_id_pattern = re.compile(r"\b[0-9a-f]{32}\b")
    for state, (payload, marker) in cases.items():
        observation = MainAgentRuntime._tool_observation(
            "probe",
            ToolObservation(
                tool_name="probe",
                state=state,
                message="如果 presenter 失败就只能看到这条收据。",
                payload=payload,
            ),
        )
        assert observation.body is not None, state
        assert marker in observation.body, state
        assert internal_id_pattern.search(observation.body) is None, state


def test_the_runtime_reads_waiting_rules_from_the_registry() -> None:
    """No second copy of these tables survives on the runtime."""
    assert not hasattr(MainAgentRuntime, "_WAITING_STATES")
    assert is_waiting("calendar_approval_required")
    assert not is_waiting("saved_jobs_found")


def test_waiting_policy_and_control_disposition_cannot_drift() -> None:
    waiting = {
        state for state, policy in DELIVERY_POLICIES.items() if policy.waiting
    }

    for state in waiting:
        assert ToolObservation(
            tool_name="emitter",
            state=state,
            message="需要用户继续。",
        ).disposition == "interaction_required"

    # The sole state whose meaning depends on its emitter: producing a new
    # analysis asks for confirmation, reading the same immutable result does
    # not. Every other explicit interaction must first be declared waiting.
    assert ToolObservation(
        tool_name="analyze_resume",
        state="resume_analysis_ready",
        message="等待确认。",
        disposition="interaction_required",
    ).disposition == "interaction_required"
    with pytest.raises(ValueError, match="must be declared waiting"):
        ToolObservation(
            tool_name="future_emitter",
            state="saved_jobs_found",
            message="错误地等待用户。",
            disposition="interaction_required",
        )

    source = _TOOLS_SOURCE.read_text()
    explicit_states = set()
    for block in re.split(r"ToolObservation\(", source)[1:]:
        constructor = block[: block.find(")\n")]
        if 'disposition="interaction_required"' not in constructor:
            continue
        found = re.search(r'state="([a-z_]+)"', constructor)
        assert found is not None, "explicit interaction must name a reviewable state"
        explicit_states.add(found.group(1))
    assert explicit_states == {"resume_analysis_ready"}


def test_a_match_receipt_must_embed_the_value_a_scenario_decides_on() -> None:
    """The one accepted receipt-wording dependency, made to fail loudly.

    `weak_match_routes_to_resume_tailoring` decides on
    ``tool_observations.0.message`` because ``overall_fit`` fits in the receipt,
    and the registration rule says a value the receipt can carry is not
    duplicated into ``facts``. That leaves the wording load-bearing: rewrite the
    line without the value and the scenario keeps passing while testing nothing.
    This asserts every emitter of that state interpolates the fit rather than
    describing it, so the drift is a red test instead of a silent one.
    """
    source = _TOOLS_SOURCE.read_text()
    receipts = re.findall(r'message=f"[^"]*整体匹配度为[^"]*"', source)

    assert receipts, "no resume_job_match receipt found; did the wording change?"
    for receipt in receipts:
        assert "{stored.result.overall_fit}" in receipt, receipt


def test_failure_disposition_is_intentional_and_not_waiting() -> None:
    assert not is_waiting("failed")
    assert ToolObservation(
        tool_name="emitter", state="failed", message="执行失败。"
    ).disposition == "failed"
    assert ToolObservation(
        tool_name="emitter", state="job_research_failed", message="调研失败。"
    ).disposition == "failed"
    for state in {
        "calendar_sync_not_available",
        "calendar_write_failed",
        "mock_interview_checkpoint_missing",
        "mock_interview_graph_incompatible",
        "mock_interview_restart_failed",
        "resume_tailoring_not_ready",
    }:
        assert is_failed(state), state
        assert ToolObservation(
            tool_name="emitter", state=state, message="执行失败。"
        ).disposition == "failed"


def test_every_failed_emitter_declares_retryability_in_its_payload() -> None:
    tree = ast.parse(_TOOLS_SOURCE.read_text())
    checked = 0
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ToolObservation"
        ):
            continue
        keywords = {item.arg: item.value for item in node.keywords if item.arg}
        state_node = keywords.get("state")
        if not (
            isinstance(state_node, ast.Constant)
            and isinstance(state_node.value, str)
            and is_failed(state_node.value)
        ):
            continue
        payload = keywords.get("payload")
        assert isinstance(payload, ast.Dict), state_node.value
        payload_keys = {
            key.value
            for key in payload.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        assert "retryable" in payload_keys, state_node.value
        checked += 1

    assert checked >= 11


def test_every_interaction_emitter_state_constructs_a_renderer() -> None:
    """Configuration omissions fail in CI, not as a production turn error."""
    waiting = {
        state for state, policy in DELIVERY_POLICIES.items() if policy.waiting
    }
    expected = waiting | {"resume_analysis_ready"}
    assert MainAgentRuntime._INTERACTION_RENDERER_STATES == expected

    for state in sorted(expected):
        task = ConversationTaskState()
        disposition = None
        payload: dict[str, object] = {}
        if state == "capability_confirmation_required":
            # The durable seal is what the interaction is keyed on, so the
            # renderer needs it. Supplied here for the same reason the analysis
            # branch below supplies task state: the state alone is not the gate.
            payload = {"confirmation_id": "c" * 32}
        if state == "resume_analysis_ready":
            task = task.model_copy(
                update={
                    "active_resume_analysis_id": "analysis-1",
                    "resume_analysis_status": "pending",
                }
            )
            disposition = "interaction_required"
        observation = ToolObservation(
            tool_name="analyze_resume" if disposition else "emitter",
            state=state,
            message="请继续。",
            payload=payload,
            **({"disposition": disposition} if disposition else {}),
        )
        context = MainAgentContext(
            conversation_id="c1",
            profile=CareerProfileContext(user_id="u1"),
            task=task,
            user_message="继续",
        )
        turn = MainAgentTurnResult(
            origin=ModelDecision(AgentDecision(action="final", message="请继续。")),
            context=context,
            assistant_message="请继续。",
            tool_result=observation,
        )
        assert MainAgentRuntime._interaction_event(
            result=turn,
            conversation_id="c1",
        ) is not None, state


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


def test_mock_interview_readbacks_use_restricted_markdown_for_dynamic_text() -> None:
    question_view = MockInterviewQuestionView(
        question_number=1,
        exchanges=(
            MockInterviewExchange(
                turn_type="primary",
                question="## 题目 `code` [链接](https://evil.example)",
                answer="**回答** www.evil.example",
                rating="adequate",
                evaluation_summary="<b>评价</b>",
            ),
        ),
    )
    result_view = MockInterviewResultView(
        interview_type="technical",
        status="completed",
        questions=(
            MockInterviewQuestionSummary(
                plan_item_number=1,
                question="## 题目 https://evil.example",
                rating="adequate",
                follow_up_count=0,
            ),
        ),
        answered_count=1,
        report_summary="[总结](https://evil.example)",
    )

    question_screen = render_mock_interview_question(question_view)
    result_screen = render_mock_interview_result(result_view)

    assert r"\## 题目" in question_screen
    assert "`code`" in question_screen
    assert "**回答**" in question_screen
    assert "&lt;b>评价&lt;/b>" in question_screen
    assert "https://evil.example" not in question_screen
    assert "www.evil.example" not in question_screen
    assert r"\[总结\]" in result_screen
    assert "https://evil.example" not in result_screen


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

    Known blind spot: a conditional construction counts as attached even when
    the ``else`` branch yields ``None``, because the scan only asks whether the
    name appears in the block. That is the right trade for what this guard is
    for — a new state that forgot a reference entirely — but it cannot see a
    state whose reference is merely usually present. No live instance remains:
    ``mock_interview_completed`` was the last, and the graph now raises rather
    than projecting a completed run without its report. Exercising the branch
    is what ``_graph_states_with_a_reference`` does, for the states where the
    distinction matters.
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
        observation = MainAgentToolRegistry()._mock_interview_observation(
            MockInterviewGraphResult(
                session_id="sess-1", state=graph_state, message="…", **kwargs
            ),
            "start_mock_interview",
            user_id="u1",
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
