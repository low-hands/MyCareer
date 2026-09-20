from datetime import datetime, timezone
import pytest

from career_agent.agent.career_context import CareerContextProjector
from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    CareerFactProposal,
    ConversationTaskState,
    JobIntentUpdate,
    ToolCall,
    project_career_fact_arguments,
)
from career_agent.agent.questionnaire_contracts import QuestionAnswer, UserQuestion
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.storage.career_history import CareerHistoryStore
from career_agent.storage.context import CareerContextStore
from career_agent.harness.streaming import InteractionResponse
from conftest import enter_tool_profile


class SequenceDecisionMaker:
    def __init__(self, *decisions: AgentDecision) -> None:
        self.decisions = list(decisions)
        self.contexts = []

    def decide(self, context, tool_specs):
        self.contexts.append(context)
        return self.decisions.pop(0)


def _tool(name: str, arguments: dict | None = None) -> AgentDecision:
    return AgentDecision(
        action="tool_call",
        tool_call=ToolCall(name=name, arguments=arguments or {}),
    )


def _final() -> AgentDecision:
    return AgentDecision(action="final", message="")


def _runtime(tmp_path, *decisions, profile=None):
    context = CareerContextStore(tmp_path / "context.sqlite3")
    history = CareerHistoryStore(tmp_path / "career.sqlite3")
    if profile is not None:
        enter_tool_profile(context, profile)
    maker = SequenceDecisionMaker(*decisions)
    runtime = MainAgentRuntime(
        context_manager=ContextManager(context),
        decision_maker=maker,
        tools=MainAgentToolRegistry(
            career_history_store=history,
            conversation_store=context,
        ),
        career_context_projector=CareerContextProjector(history),
    )
    return runtime, maker, history


def test_career_fact_stays_quarantined_until_next_turn_confirmation(
    tmp_path,
) -> None:
    _, _, history = _runtime(tmp_path)
    record = history.create_record(
        user_id="u1",
        record_type="work",
        organization="Example",
        title="ML Engineer",
        is_current=True,
    )
    history.confirm_evidence(
        user_id="u1",
        career_evidence_id=history.create_evidence(
            user_id="u1",
            career_record_id=record.id,
            claim="在这份工作里负责推荐系统。",
            origin="user_input",
        ).id,
    )

    runtime, _, history = _runtime(
        tmp_path,
        _tool(
            "propose_career_fact",
            {
                "record_selection_index": 1,
                "claim": "曾带领 5 人机器学习团队。",
                "reason": "用户明确补充了这段经历的团队规模。",
                "user_quote": "带过 5 人团队",
            },
        ),
        profile="memory",
    )
    proposed = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="我在这份工作里带过 5 人团队。",
    )

    assert proposed.tool_result is not None
    assert proposed.tool_result.state == "career_fact_proposed"
    proposal = proposed.context.task.pending_career_fact
    assert proposal is not None
    pending = history.get_evidence(
        user_id="u1",
        career_evidence_id=proposal.career_evidence_id,
    )
    assert pending is not None
    assert pending.origin == "user_input"
    assert pending.source_user_quote == "带过 5 人团队"
    assert pending.verification_status == "pending"

    before_maker = SequenceDecisionMaker(_final())
    before = MainAgentRuntime(
        context_manager=ContextManager(
            CareerContextStore(tmp_path / "context.sqlite3")
        ),
        decision_maker=before_maker,
        tools=MainAgentToolRegistry(
            career_history_store=history,
            conversation_store=CareerContextStore(
                tmp_path / "context.sqlite3"
            ),
        ),
        career_context_projector=CareerContextProjector(history),
    )
    before.run_turn(
        user_id="u1",
        conversation_id="c2",
        user_message="我的机器学习团队是什么情况？",
    )
    assert "曾带领 5 人机器学习团队" not in str(
        before_maker.contexts[0].model_context()
    )

    runtime, maker, history = _runtime(tmp_path)
    confirmed_turn = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="确认",
    )

    assert maker.contexts == []
    assert confirmed_turn.origin.label == "policy:career_fact_confirmation"
    assert confirmed_turn.tool_result is not None
    assert confirmed_turn.tool_result.state == "career_fact_confirmed"
    assert confirmed_turn.context.task.pending_career_fact is None
    confirmed = history.get_evidence(
        user_id="u1",
        career_evidence_id=proposal.career_evidence_id,
    )
    assert confirmed is not None
    assert confirmed.origin == "user_input"
    assert confirmed.verification_status == "confirmed"

    after_runtime, after_maker, _ = _runtime(tmp_path, _final())
    after_runtime.run_turn(
        user_id="u1",
        conversation_id="c3",
        user_message="我的机器学习团队是什么情况？",
    )
    assert "曾带领 5 人机器学习团队" in str(
        after_maker.contexts[0].model_context()
    )


def test_stale_bare_confirmation_cannot_confirm_an_older_fact(
    tmp_path,
) -> None:
    _, _, history = _runtime(tmp_path)
    record = history.create_record(
        user_id="u1",
        record_type="work",
        organization="Example",
        title="ML Engineer",
        is_current=True,
    )
    history.confirm_evidence(
        user_id="u1",
        career_evidence_id=history.create_evidence(
            user_id="u1",
            career_record_id=record.id,
            claim="在这份工作里负责推荐系统。",
            origin="user_input",
        ).id,
    )
    proposed_runtime, _, _ = _runtime(
        tmp_path,
        _tool(
            "propose_career_fact",
            {
                "record_selection_index": 1,
                "claim": "曾带领 5 人机器学习团队。",
                "reason": "用户明确补充了这段经历的团队规模。",
            },
        ),
        profile="memory",
    )
    proposed = proposed_runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="我在这份工作里带过 5 人团队。",
    )
    proposal = proposed.context.task.pending_career_fact
    assert proposal is not None

    unrelated_runtime, unrelated_maker, _ = _runtime(
        tmp_path, _final()
    )
    unrelated_runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="先帮我看看新岗位",
    )
    assert len(unrelated_maker.contexts) == 1

    confirmation_runtime, confirmation_maker, history = _runtime(
        tmp_path, _final()
    )
    result = confirmation_runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="可以",
    )

    assert len(confirmation_maker.contexts) == 1
    assert result.origin.label == "model:final"
    assert result.tool_result is None
    evidence = history.get_evidence(
        user_id="u1",
        career_evidence_id=proposal.career_evidence_id,
    )
    assert evidence is not None
    assert evidence.origin == "agent_inference"
    assert evidence.verification_status == "pending"
    assert result.context.task.pending_career_fact == proposal
    assert result.context.task.bare_confirmation_target is None


def test_questionnaire_answer_carries_user_interaction_provenance_into_career_fact(tmp_path) -> None:
    history = CareerHistoryStore(tmp_path / "career.sqlite3")
    record = history.create_record(
        user_id="u1", record_type="work", organization="Example",
        title="Engineer", is_current=True,
    )
    history.confirm_evidence(
        user_id="u1",
        career_evidence_id=history.create_evidence(
            user_id="u1", career_record_id=record.id,
            claim="在 Example 开发内部工具。", origin="user_input",
        ).id,
    )
    runtime, maker, history = _runtime(
        tmp_path,
        AgentDecision(action="questionnaire", message="请补充工具使用情况。", questions=(
            UserQuestion(question_id="q1", prompt="使用哪些 AI 编程工具？", kind="free_text"),
            UserQuestion(question_id="q2", prompt="其他补充？", kind="free_text"),
        )),
        _final(),
        _tool("propose_career_fact", {
            "record_selection_index": 1,
            "claim": "日常使用 Claude Code 和 Cursor 开发内部工具。",
            "reason": "用户要求将刚才回答的工具使用情况记为职业事实。",
            "user_quote": "Claude Code 和 Cursor 开发内部工具",
        }),
        profile="memory",
    )
    events = []
    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="先问我工具经验",
                     event_sink=events.append)
    interaction = next(event for event in events if event.type == "interaction_required")
    response = InteractionResponse(
        interaction_id=interaction.interaction_id, scope="questionnaire", action="submit",
        answers=(
            QuestionAnswer(question_id="q1", free_text="Claude Code 和 Cursor 开发内部工具"),
            QuestionAnswer(question_id="q2", skipped=True),
        ),
    )
    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="已提交问卷",
                     interaction_response=response)
    proposed = runtime.run_turn(
        user_id="u1", conversation_id="c1",
        user_message="把刚才回答的工具使用情况记到 Example Engineer 这份工作经历。",
    )
    assert proposed.tool_result is not None
    assert proposed.tool_result.state == "career_fact_proposed"
    proposal = proposed.context.task.pending_career_fact
    assert proposal is not None
    pending = history.get_evidence(user_id="u1", career_evidence_id=proposal.career_evidence_id)
    assert pending is not None
    assert pending.verification_status == "pending"
    assert pending.origin == "user_input"
    assert pending.source_user_quote == "Claude Code 和 Cursor 开发内部工具"
    assert pending.source_user_interaction_id == interaction.interaction_id
    assert "来源：用户原话" in proposed.tool_result.message

    confirmed = runtime.run_turn(user_id="u1", conversation_id="c1", user_message="确认")
    assert confirmed.tool_result is not None
    evidence = history.get_evidence(user_id="u1", career_evidence_id=pending.id)
    assert evidence is not None
    assert evidence.origin == "user_input"
    assert evidence.verification_status == "confirmed"
    assert evidence.source_user_interaction_id == interaction.interaction_id
    assert len(maker.contexts) == 3
    with pytest.raises(ValueError, match="user_quote is absent"):
        project_career_fact_arguments(maker.contexts[2], "propose_career_fact", {
            "record_selection_index": 1, "claim": "虚构成果。", "reason": "测试",
            "user_quote": "用户从未说过的虚构成果",
        })
    with pytest.raises(ValueError):
        project_career_fact_arguments(maker.contexts[2], "propose_career_fact", {
            "record_selection_index": 1, "claim": "虚构成果。", "reason": "测试",
            "origin": "user_input",
        })


def test_career_fact_pending_reuse_keeps_user_and_inference_sources_separate(tmp_path) -> None:
    history = CareerHistoryStore(tmp_path / "career.sqlite3")
    record = history.create_record(
        user_id="u1", record_type="work", organization="Example", title="Engineer",
    )
    tools = MainAgentToolRegistry(career_history_store=history)
    common = {
        "user_id": "u1", "career_record_id": record.id,
        "claim": "日常使用 Cursor。", "reason": "待确认",
    }
    direct = {
        **common, "origin": "user_input", "source_user_quote": "日常使用 Cursor",
        "source_user_interaction_id": "interaction_" + "a" * 20,
    }
    first = tools._propose_career_fact(direct).payload["proposal"]["career_evidence_id"]
    again = tools._propose_career_fact(direct).payload["proposal"]["career_evidence_id"]
    inferred = tools._propose_career_fact({**common, "origin": "agent_inference"}).payload["proposal"]["career_evidence_id"]
    assert first == again
    assert inferred != first
    confirmed_inference = history.confirm_evidence(user_id="u1", career_evidence_id=inferred)
    assert confirmed_inference.origin == "agent_inference"
    assert confirmed_inference.verification_status == "confirmed"


def test_bare_confirmation_uses_last_shown_pending_type_not_fixed_priority(
    tmp_path,
) -> None:
    context = CareerContextStore(tmp_path / "context.sqlite3")
    history = CareerHistoryStore(tmp_path / "career.sqlite3")
    record = history.create_record(
        user_id="u1",
        record_type="work",
        organization="Example",
        title="ML Engineer",
    )
    evidence = history.create_evidence(
        user_id="u1",
        career_record_id=record.id,
        claim="曾带领 5 人机器学习团队。",
        origin="agent_inference",
    )
    fact = CareerFactProposal(
        career_evidence_id=evidence.id,
        career_record_id=record.id,
        claim=evidence.claim,
        reason="待用户确认",
    )
    context.upsert_task(
        user_id="u1",
        conversation_id="c1",
        task=ConversationTaskState(
            pending_career_fact=fact,
            pending_job_intent_update=JobIntentUpdate(city="上海"),
            pending_proposed_at={
                "pending_career_fact": datetime.now(timezone.utc),
                "pending_job_intent_update": datetime.now(timezone.utc),
            },
            bare_confirmation_target="job_intent",
        ),
    )
    maker = SequenceDecisionMaker()
    runtime = MainAgentRuntime(
        context_manager=ContextManager(context),
        decision_maker=maker,
        tools=MainAgentToolRegistry(
            career_history_store=history,
            conversation_store=context,
            career_profile_store=context,
        ),
        career_context_projector=CareerContextProjector(history),
    )

    result = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="确认",
    )

    assert maker.contexts == []
    assert result.origin.label == "policy:job_intent_confirmation"
    assert context.get_profile("u1").default_city == "上海"
    still_pending = history.get_evidence(
        user_id="u1",
        career_evidence_id=evidence.id,
    )
    assert still_pending is not None
    assert still_pending.verification_status == "pending"
    assert result.context.task.pending_career_fact == fact
