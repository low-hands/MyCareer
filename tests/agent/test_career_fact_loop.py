from career_agent.agent.career_context import CareerContextProjector
from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    ToolCall,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.storage.career_history import CareerHistoryStore
from career_agent.storage.context import CareerContextStore


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


def _runtime(tmp_path, *decisions):
    context = CareerContextStore(tmp_path / "context.sqlite3")
    history = CareerHistoryStore(tmp_path / "career.sqlite3")
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
            },
        ),
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
    assert pending.origin == "agent_inference"
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
