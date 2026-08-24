from __future__ import annotations

import pytest

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import AgentDecision, CareerProfileContext, ToolCall
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.storage.context import CareerContextStore
from career_agent.storage.resumes import ResumeStore


class UnusedGateway:
    def advance(self, **kwargs):
        raise AssertionError("Resume metadata tools must not enter Job Discovery")


class SequenceDecisionMaker:
    def __init__(self, *decisions: AgentDecision) -> None:
        self.decisions = list(decisions)
        self.contexts = []

    def decide(self, context, tool_specs):
        self.contexts.append(context)
        if not self.decisions:
            raise AssertionError("Main Agent requested more decisions than expected")
        return self.decisions.pop(0)


def seed_resume(store: ResumeStore, *, user_id: str = "u1", title: str = "AI Engineer", name: str = "AI Base"):
    role = store.create_target_role(user_id=user_id, title=title, priority=1)
    resume, first = store.import_document(
        user_id=user_id,
        target_role_id=role.id,
        name=name,
        content=b"PRIVATE RESUME CONTENT v1",
        document_format="text",
    )
    _, second = store.import_document(
        user_id=user_id,
        resume_id=resume.id,
        content=b"PRIVATE RESUME CONTENT v2",
        document_format="markdown",
    )
    return role, resume, first, second


def build_agent(tmp_path, store: ResumeStore, decisions: SequenceDecisionMaker, *, user_id: str = "u1"):
    manager = ContextManager(CareerContextStore(tmp_path / f"{user_id}-context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id=user_id))
    tools = MainAgentToolRegistry(UnusedGateway(), resume_store=store)
    return MainAgentRuntime(context_manager=manager, decision_maker=decisions, tools=tools), tools


def test_resume_tools_list_roles_resumes_and_safe_version_metadata(tmp_path) -> None:
    store = ResumeStore(tmp_path / "resumes.sqlite3")
    role, resume, first, second = seed_resume(store)
    seed_resume(store, user_id="other", title="Backend Engineer", name="Other Private Resume")
    decisions = SequenceDecisionMaker(
        AgentDecision(action="tool_call", tool_call=ToolCall(name="list_target_roles", arguments={})),
        AgentDecision(action="tool_call", tool_call=ToolCall(name="list_resumes", arguments={"target_role_id": role.id})),
        AgentDecision(action="tool_call", tool_call=ToolCall(name="get_resume_metadata", arguments={"resume_id": resume.id})),
        AgentDecision(action="final", message="你有一份 AI Engineer 简历，共两个版本。"),
    )
    agent, tools = build_agent(tmp_path, store, decisions)

    result = agent.run_turn(user_id="u1", conversation_id="c1", user_message="我有哪些 AI Engineer 简历和版本？")

    assert tools.names == ("job_discovery", "list_target_roles", "list_resumes", "get_resume_metadata")
    assert all("user_id" not in spec["function"]["parameters"].get("properties", {}) for spec in tools.schemas())
    role_observation = decisions.contexts[1].tool_observations[-1]
    assert role_observation.payload["items"] == [{"selection_index": 1, "target_role_id": role.id, "title": "AI Engineer", "priority": 1, "status": "active"}]
    resume_observation = decisions.contexts[2].tool_observations[-1]
    assert resume_observation.payload["items"][0]["resume_id"] == resume.id
    assert "Other Private Resume" not in resume_observation.model_dump_json()
    metadata_observation = decisions.contexts[3].tool_observations[-1]
    assert metadata_observation.payload["resume"]["resume_id"] == resume.id
    assert [item["resume_version_id"] for item in metadata_observation.payload["versions"]] == [second.id, first.id]
    serialized = metadata_observation.model_dump_json()
    assert "PRIVATE RESUME CONTENT" not in serialized
    assert first.content_sha256 not in serialized
    assert result.assistant_message == "你有一份 AI Engineer 简历，共两个版本。"


def test_get_resume_metadata_hides_foreign_resume(tmp_path) -> None:
    store = ResumeStore(tmp_path / "resumes.sqlite3")
    _, foreign_resume, _, _ = seed_resume(store, user_id="other")
    decisions = SequenceDecisionMaker(
        AgentDecision(action="tool_call", tool_call=ToolCall(name="get_resume_metadata", arguments={"resume_id": foreign_resume.id})),
        AgentDecision(action="final", message="没有找到这份简历。"),
    )
    agent, _ = build_agent(tmp_path, store, decisions)

    agent.run_turn(user_id="u1", conversation_id="c1", user_message="读取这份简历")

    observation = decisions.contexts[1].tool_observations[-1]
    assert observation.state == "resume_not_found"
    assert "Other" not in observation.model_dump_json()


@pytest.mark.parametrize(
    "tool_name,arguments",
    [
        ("list_target_roles", {"user_id": "other"}),
        ("list_resumes", {"user_id": "other"}),
        ("get_resume_metadata", {"resume_id": "resume-1", "user_id": "other"}),
    ],
)
def test_resume_tools_reject_model_supplied_user_id(tmp_path, tool_name, arguments) -> None:
    store = ResumeStore(tmp_path / "resumes.sqlite3")
    decisions = SequenceDecisionMaker(
        AgentDecision(action="tool_call", tool_call=ToolCall(name=tool_name, arguments=arguments))
    )
    agent, _ = build_agent(tmp_path, store, decisions)

    with pytest.raises(ValueError, match="cannot accept internal argument"):
        agent.run_turn(user_id="u1", conversation_id="c1", user_message="越权读取")

