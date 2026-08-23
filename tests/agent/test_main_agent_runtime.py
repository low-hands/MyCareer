from __future__ import annotations

import pytest

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.job_discovery_contracts import JDAnalysis
from career_agent.agent.job_discovery_gateway import GatewayJobItem, JobDiscoveryGatewayResult
from career_agent.agent.main_agent_contracts import AgentDecision, CareerProfileContext, ConversationTaskState, ToolCall
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.storage.context import CareerContextStore


class Gateway:
    def __init__(self) -> None:
        self.calls = []

    def advance(self, **kwargs):
        self.calls.append(kwargs)
        task = kwargs["task"]
        if task.phase == "selection_required":
            return JobDiscoveryGatewayResult(run_id=task.run_id or "run-1", state="analysis_ready", message="Analysis ready.", selected_result_ref="r1", analysis=JDAnalysis(result_ref="r1", job_summary="LLM 应用落地。", responsibilities=("建设 LLM 应用",), required_skills=("Python",), preferred_qualifications=("RAG 经验",), clarification_questions=("经验年限不明确",)))
        if task.phase == "detail_unavailable":
            return JobDiscoveryGatewayResult(run_id=task.run_id or "run-1", state="analysis_ready", message="Analysis ready.", selected_result_ref="r1", analysis=JDAnalysis(result_ref="r1", job_summary="LLM 应用落地。", responsibilities=("建设 LLM 应用",), required_skills=("Python",), preferred_qualifications=("RAG 经验",), clarification_questions=("经验年限不明确",)))
        return JobDiscoveryGatewayResult(run_id="run-1", state="selection_required", message="Select a result.", items=(GatewayJobItem(result_ref="r1", title="AI Engineer", company_name="Acme"),), next_action="select_result")


class AlwaysReadyGateway:
    def __init__(self) -> None:
        self.calls = []

    def advance(self, **kwargs):
        self.calls.append(kwargs)
        return JobDiscoveryGatewayResult(
            run_id="run-1",
            state="analysis_ready",
            message="Analysis ready.",
            selected_result_ref="r1",
            analysis=JDAnalysis(result_ref="r1", job_summary="Summary"),
        )


class DecisionMaker:
    def __init__(self, decision: AgentDecision) -> None:
        self.decision = decision

    def decide(self, context, tool_names):
        assert tuple(spec["function"]["name"] for spec in tool_names) == ("job_discovery",)
        return self.decision


class SequenceDecisionMaker:
    def __init__(self, *decisions: AgentDecision) -> None:
        self.decisions = list(decisions)
        self.contexts = []

    def decide(self, context, tool_names):
        self.contexts.append(context)
        if not self.decisions:
            raise AssertionError("Main Agent requested more decisions than expected")
        return self.decisions.pop(0)


def build_runtime(tmp_path, decision: AgentDecision, gateway: Gateway | None = None):
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1", target_roles=("AI Engineer",), default_city="Shanghai", salary_preference="25K以上"))
    gateway = gateway or Gateway()
    return MainAgentRuntime(context_manager=manager, decision_maker=DecisionMaker(decision), tools=MainAgentToolRegistry(gateway)), gateway, manager


def test_initial_workflow_call_projects_profile_defaults(tmp_path) -> None:
    agent, gateway, _ = build_runtime(tmp_path, AgentDecision(action="tool_call", tool_call=ToolCall(name="job_discovery", arguments={})))

    result = agent.run_turn(user_id="u1", conversation_id="c1", user_message="Find jobs.")

    call = gateway.calls[0]
    assert call["research_request"].conversation_id == "c1"
    assert call["research_request"].city == "Shanghai"
    assert call["research_request"].salary == "25K以上"
    assert call["user_message"] == "Find jobs."
    assert result.context.task.run_id == "run-1"
    assert result.context.task.candidates[0].result_ref == "r1"
    assert len(gateway.calls) == 1


def test_tool_observation_returns_to_model_before_final_answer(tmp_path) -> None:
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1", target_roles=("AI Engineer",)))
    gateway = Gateway()
    decisions = SequenceDecisionMaker(
        AgentDecision(action="tool_call", tool_call=ToolCall(name="job_discovery", arguments={})),
        AgentDecision(action="ask_user", message="我找到了一个岗位，要查看第 1 个吗？"),
    )
    agent = MainAgentRuntime(context_manager=manager, decision_maker=decisions, tools=MainAgentToolRegistry(gateway))

    result = agent.run_turn(user_id="u1", conversation_id="c1", user_message="帮我找工作")

    assert result.decision.action == "ask_user"
    assert result.assistant_message == "我找到了一个岗位，要查看第 1 个吗？"
    assert len(gateway.calls) == 1
    assert len(decisions.contexts) == 2
    observation = decisions.contexts[1].model_context()["tool_observations"][0]
    assert observation["tool_name"] == "job_discovery"
    assert observation["state"] == "selection_required"
    assert observation["payload"]["items"][0]["selection_index"] == 1
    serialized = str(observation)
    assert "run-1" not in serialized
    assert "r1" not in serialized


def test_repeated_tool_call_is_stopped_without_duplicate_execution(tmp_path) -> None:
    agent, gateway, _ = build_runtime(
        tmp_path,
        AgentDecision(action="tool_call", tool_call=ToolCall(name="job_discovery", arguments={})),
    )

    result = agent.run_turn(user_id="u1", conversation_id="c1", user_message="Find work.")

    assert len(gateway.calls) == 1
    assert result.assistant_message == "Select a result."


def test_tool_loop_stops_at_configured_limit(tmp_path) -> None:
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    gateway = AlwaysReadyGateway()
    decisions = SequenceDecisionMaker(
        AgentDecision(action="tool_call", tool_call=ToolCall(name="job_discovery", arguments={"target_role": "Role A"})),
        AgentDecision(action="tool_call", tool_call=ToolCall(name="job_discovery", arguments={"target_role": "Role B"})),
        AgentDecision(action="tool_call", tool_call=ToolCall(name="job_discovery", arguments={"target_role": "Role C"})),
    )
    agent = MainAgentRuntime(
        context_manager=manager,
        decision_maker=decisions,
        tools=MainAgentToolRegistry(gateway),
        max_tool_calls=2,
    )

    result = agent.run_turn(user_id="u1", conversation_id="c1", user_message="Research several roles.")

    assert len(gateway.calls) == 2
    assert result.assistant_message.startswith("岗位摘要\nSummary")


def test_workflow_selection_uses_index_not_internal_result_ref(tmp_path) -> None:
    first, gateway, _ = build_runtime(tmp_path, AgentDecision(action="tool_call", tool_call=ToolCall(name="job_discovery", arguments={})))
    first.run_turn(user_id="u1", conversation_id="c1", user_message="Find work.")
    second, _, _ = build_runtime(tmp_path, AgentDecision(action="tool_call", tool_call=ToolCall(name="job_discovery", arguments={"selection_index": 1})), gateway)

    result = second.run_turn(user_id="u1", conversation_id="c1", user_message="Show the first one.")

    assert gateway.calls[1]["selection_index"] == 1
    assert gateway.calls[1]["task"].run_id == "run-1"
    assert result.context.task.phase == "analysis_ready"
    assert result.assistant_message == "岗位摘要\nLLM 应用落地。\n\n工作职责\n- 建设 LLM 应用\n\n必备技能\n- Python\n\n加分项\n- RAG 经验\n\n待确认问题\n- 经验年限不明确"


def test_target_and_search_overrides_do_not_mutate_profile(tmp_path) -> None:
    agent, gateway, manager = build_runtime(tmp_path, AgentDecision(action="tool_call", tool_call=ToolCall(name="job_discovery", arguments={"target_role": "Backend Engineer", "city": "Hangzhou", "salary": "30K以上", "experience": "3-5年", "education": "本科"})))

    agent.run_turn(user_id="u1", conversation_id="c1", user_message="Search backend roles in Hangzhou this time.")

    request = gateway.calls[0]["research_request"]
    assert request.target_role == "Backend Engineer"
    assert request.city == "Hangzhou"
    assert request.salary == "30K以上"
    assert request.experience == "3-5年"
    assert request.education == "本科"
    profile = manager.load_for_turn(user_id="u1", conversation_id="c1", user_message="next").profile
    assert profile.target_roles == ("AI Engineer",)
    assert profile.default_city == "Shanghai"


def test_detail_unavailable_injects_current_user_message_as_jd(tmp_path) -> None:
    agent, gateway, manager = build_runtime(tmp_path, AgentDecision(action="tool_call", tool_call=ToolCall(name="job_discovery", arguments={})))
    seed = manager.load_for_turn(user_id="u1", conversation_id="c1", user_message="seed")
    manager.commit_turn(context=seed, task=ConversationTaskState(active_workflow="job_discovery", run_id="run-1", phase="detail_unavailable", selected_result_ref="r1"), assistant_message="Detail unavailable.")
    jd = "Build reliable LLM systems and operate agent workflows in production."

    agent.run_turn(user_id="u1", conversation_id="c1", user_message=jd)

    call = gateway.calls[0]
    assert call["user_id"] == "u1"
    assert call["conversation_id"] == "c1"
    assert call["task"].phase == "detail_unavailable"
    assert call["task"].selected_result_ref == "r1"
    assert call["user_message"] == jd
    assert call["selection_index"] is None


def test_analysis_formatter_uses_placeholder_for_empty_sections():
    result = JobDiscoveryGatewayResult(run_id="run-1", state="analysis_ready", message="Analysis ready.", analysis=JDAnalysis(result_ref="r1", job_summary="摘要"))

    assert MainAgentRuntime._assistant_message(result) == "岗位摘要\n摘要\n\n工作职责\n- 暂无明确说明\n\n必备技能\n- 暂无明确说明\n\n加分项\n- 暂无明确说明\n\n待确认问题\n- 暂无明确说明"


def test_normal_answer_commits_history_without_tool(tmp_path) -> None:
    agent, gateway, manager = build_runtime(tmp_path, AgentDecision(action="final", message="AI Engineers build AI products."))

    result = agent.run_turn(user_id="u1", conversation_id="c1", user_message="What is an AI Engineer?")
    loaded = manager.load_for_turn(user_id="u1", conversation_id="c1", user_message="next")

    assert result.tool_result is None
    assert gateway.calls == []
    assert [message.content for message in loaded.recent_messages] == ["What is an AI Engineer?", "AI Engineers build AI products."]


@pytest.mark.parametrize("forbidden", ["user_id", "conversation_id", "run_id", "result_ref", "security_id", "job_id", "jd_text"])
def test_internal_arguments_are_rejected_without_commit(tmp_path, forbidden) -> None:
    agent, _, manager = build_runtime(tmp_path, AgentDecision(action="tool_call", tool_call=ToolCall(name="job_discovery", arguments={forbidden: "hidden"})))

    with pytest.raises(ValueError, match="internal arguments"):
        agent.run_turn(user_id="u1", conversation_id="c1", user_message="Do it.")

    assert manager.load_for_turn(user_id="u1", conversation_id="c1", user_message="next").recent_messages == ()


def test_unknown_tool_is_rejected_without_commit(tmp_path) -> None:
    agent, _, manager = build_runtime(tmp_path, AgentDecision(action="tool_call", tool_call=ToolCall(name="boss.detail", arguments={})))

    with pytest.raises(ValueError, match="Unknown main-agent tool"):
        agent.run_turn(user_id="u1", conversation_id="c1", user_message="Do it.")

    assert manager.load_for_turn(user_id="u1", conversation_id="c1", user_message="next").recent_messages == ()
