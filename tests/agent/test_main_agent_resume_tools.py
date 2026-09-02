from __future__ import annotations

import pytest

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import AgentDecision, CareerProfileContext, ToolCall
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.agent.resume_analysis_contracts import (
    ExtractedCareerEvidence,
    ExtractedCareerRecord,
    ResumeAnalysisResult,
)
from career_agent.harness.streaming import (
    ContentDeltaEvent,
    InteractionRequiredEvent,
    InteractionResponse,
    interaction_id,
)
from career_agent.services.resume_analysis import ResumeAnalysisService
from career_agent.storage.context import CareerContextStore
from career_agent.storage.resumes import ResumeStore
from career_agent.storage.resume_analysis import SQLiteResumeAnalysisDraftStore
from career_agent.storage.career_history import CareerHistoryStore


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


def build_agent(
    tmp_path,
    store: ResumeStore,
    decisions: SequenceDecisionMaker,
    *,
    user_id: str = "u1",
    resume_analysis_service: ResumeAnalysisService | None = None,
    max_read_calls: int = 6,
    max_write_calls: int = 1,
):
    manager = ContextManager(CareerContextStore(tmp_path / f"{user_id}-context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id=user_id))
    tools = MainAgentToolRegistry(
        resume_store=store,
        resume_analysis_service=resume_analysis_service,
    )
    return (
        MainAgentRuntime(
            context_manager=manager,
            decision_maker=decisions,
            tools=tools,
            max_read_calls=max_read_calls,
            max_write_calls=max_write_calls,
        ),
        tools,
    )


def test_resume_tools_list_roles_resumes_and_safe_version_metadata(tmp_path) -> None:
    store = ResumeStore(tmp_path / "resumes.sqlite3")
    role, resume, first, second = seed_resume(store)
    seed_resume(store, user_id="other", title="Backend Engineer", name="Other Private Resume")
    decisions = SequenceDecisionMaker(
        AgentDecision(action="tool_call", tool_call=ToolCall(name="list_target_roles", arguments={})),
        AgentDecision(action="tool_call", tool_call=ToolCall(name="list_resumes", arguments={"target_role_selection_index": 1})),
        AgentDecision(action="tool_call", tool_call=ToolCall(name="get_resume_metadata", arguments={"selection_index": 1})),
        AgentDecision(action="final", message="你有一份 AI Engineer 简历，共两个版本。"),
    )
    agent, tools = build_agent(tmp_path, store, decisions)

    result = agent.run_turn(user_id="u1", conversation_id="c1", user_message="我有哪些 AI Engineer 简历和版本？")

    assert tools.names == ("open_job_search", "list_target_roles", "list_resumes", "get_resume_metadata")
    assert all("user_id" not in spec["function"]["parameters"].get("properties", {}) for spec in tools.schemas())
    role_observation = result.tool_results[0]
    assert role_observation.payload["items"] == [{
        "selection_index": 1, "target_role_id": role.id, "title": "AI Engineer",
        "priority": 1, "status": "active",
        # Intent scoped to this track, unset until the user states it.
        "city": None, "salary_expectation": None, "experience": None, "education": None,
    }]
    resume_observation = result.tool_results[1]
    assert resume_observation.payload["items"][0]["resume_id"] == resume.id
    assert "Other Private Resume" not in resume_observation.model_dump_json()
    metadata_observation = result.tool_results[2]
    assert metadata_observation.payload["resume"]["resume_id"] == resume.id
    assert [item["resume_version_id"] for item in metadata_observation.payload["versions"]] == [second.id, first.id]
    serialized = metadata_observation.model_dump_json()
    assert "PRIVATE RESUME CONTENT" not in serialized
    assert first.content_sha256 not in serialized
    assert result.assistant_message == "已读取简历“AI Base”及其 2 个版本的元数据。"


def test_get_resume_metadata_hides_foreign_resume(tmp_path) -> None:
    store = ResumeStore(tmp_path / "resumes.sqlite3")
    _, foreign_resume, _, _ = seed_resume(store, user_id="other")
    agent, tools = build_agent(tmp_path, store, SequenceDecisionMaker(AgentDecision(action="final", message="")))

    observation = tools.invoke_atomic_tool(
        "get_resume_metadata",
        {"user_id": "u1", "resume_id": foreign_resume.id},
    )
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

    with pytest.raises(ValueError, match="cannot accept internal identifier"):
        agent.run_turn(user_id="u1", conversation_id="c1", user_message="越权读取")


class RecordingResumeAnalysisWorker:
    def __init__(self) -> None:
        self.documents = []

    def analyze(self, document):
        self.documents.append(document)
        return ResumeAnalysisResult(
            records=(
                ExtractedCareerRecord(
                    record_type="work",
                    organization="Example Inc.",
                    title="Product Manager",
                    start_year=2022,
                    is_current=True,
                    source_locator="page 1, Experience",
                    source_quote="Example Inc. — Product Manager",
                    evidence=(
                        ExtractedCareerEvidence(
                            claim="Led knowledge-base product planning",
                            source_locator="page 1, bullet 1",
                            source_quote="Led knowledge-base product planning",
                        ),
                    ),
                ),
            ),
            clarification_questions=("What was the start month?",),
        )


def test_analyze_resume_tool_loads_owned_document_and_returns_only_analysis(tmp_path) -> None:
    store = ResumeStore(tmp_path / "resumes.sqlite3")
    _, _, _, version = seed_resume(store)
    worker = RecordingResumeAnalysisWorker()
    draft_store = SQLiteResumeAnalysisDraftStore(tmp_path / "drafts.sqlite3")
    history_store = CareerHistoryStore(tmp_path / "resumes.sqlite3")
    service = ResumeAnalysisService(
        store,
        worker,
        draft_store,
        history_store,
    )
    decisions = SequenceDecisionMaker(
        AgentDecision(action="tool_call", tool_call=ToolCall(name="list_resumes", arguments={})),
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(
                name="get_resume_metadata",
                arguments={"selection_index": 1},
            ),
        ),
        AgentDecision(action="tool_call", tool_call=ToolCall(name="analyze_resume", arguments={"selection_index": 1})),
        # A faulty decision model must not be able to turn its own extraction
        # into confirmed evidence before the user has seen it.
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name="confirm_resume_analysis", arguments={}),
        ),
    )
    agent, tools = build_agent(
        tmp_path,
        store,
        decisions,
        resume_analysis_service=service,
        # Keep budget out of the assertion: without the tool-level turn
        # barrier, the scripted confirmation would otherwise execute.
        max_read_calls=5,
    )

    events = []
    result = agent.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="分析最新版本的简历",
        event_sink=events.append,
    )

    assert tools.names[-2:] == (
        "analyze_resume",
        "get_resume_analysis",
    )
    assert "confirm_resume_analysis" not in tools.names
    schema = next(
        spec for spec in tools.schemas() if spec["function"]["name"] == "analyze_resume"
    )
    assert "user_id" not in schema["function"]["parameters"].get("properties", {})
    assert worker.documents[0].raw_bytes == b"PRIVATE RESUME CONTENT v2"
    observation = result.tool_results[-1]
    assert observation.state == "resume_analysis_ready"
    assert observation.payload["analysis_id"].startswith("resume_analysis_")
    assert observation.payload["records"][0]["title"] == "Product Manager"
    assert observation.payload["clarification_questions"] == ("What was the start month?",)
    serialized = observation.model_dump_json()
    assert "PRIVATE RESUME CONTENT" not in serialized
    assert "raw_bytes" not in serialized
    assert result.context.task.resume_analysis_status == "pending"
    assert result.context.task.active_resume_analysis_id == observation.payload["analysis_id"]
    assert len(result.tool_results) == 3
    assert len(decisions.decisions) == 1
    assert decisions.decisions[0].tool_call.name == "confirm_resume_analysis"
    assert history_store.list_records(user_id="u1") == ()
    assert history_store.list_evidence(user_id="u1") == ()
    assert result.assistant_message.startswith("# 简历分析结果")
    assert "Product Manager · Example Inc. · 工作经历" in result.assistant_message
    assert "What was the start month?" in result.assistant_message
    assert "尚未写入职业事实库" in result.assistant_message
    interaction_index = next(
        index
        for index, event in enumerate(events)
        if isinstance(event, InteractionRequiredEvent)
    )
    streamed = "".join(
        event.delta
        for event in events[:interaction_index]
        if isinstance(event, ContentDeltaEvent)
    )
    assert streamed == result.assistant_message
    assert events[interaction_index].scope == "resume_analysis_confirmation"


def test_analyze_resume_tool_hides_foreign_version(tmp_path) -> None:
    store = ResumeStore(tmp_path / "resumes.sqlite3")
    _, _, _, foreign_version = seed_resume(store, user_id="other")
    worker = RecordingResumeAnalysisWorker()
    agent, tools = build_agent(
        tmp_path,
        store,
        SequenceDecisionMaker(AgentDecision(action="final", message="")),
        resume_analysis_service=ResumeAnalysisService(
            store,
                worker,
                SQLiteResumeAnalysisDraftStore(tmp_path / "drafts.sqlite3"),
                CareerHistoryStore(tmp_path / "resumes.sqlite3"),
        ),
    )

    observation = tools.invoke_atomic_tool(
        "analyze_resume",
        {"user_id": "u1", "resume_version_id": foreign_version.id},
    )
    assert observation.state == "resume_version_not_found"
    assert worker.documents == []


def test_analyze_resume_tool_rejects_model_supplied_user_id(tmp_path) -> None:
    store = ResumeStore(tmp_path / "resumes.sqlite3")
    worker = RecordingResumeAnalysisWorker()
    decisions = SequenceDecisionMaker(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(
                name="analyze_resume",
                arguments={"resume_version_id": "version-1", "user_id": "other"},
            ),
        )
    )
    agent, _ = build_agent(
        tmp_path,
        store,
        decisions,
        resume_analysis_service=ResumeAnalysisService(
            store,
                worker,
                SQLiteResumeAnalysisDraftStore(tmp_path / "drafts.sqlite3"),
                CareerHistoryStore(tmp_path / "resumes.sqlite3"),
        ),
    )

    with pytest.raises(ValueError, match="cannot accept internal identifier"):
        agent.run_turn(user_id="u1", conversation_id="c1", user_message="越权分析")


def test_resume_analysis_can_be_reviewed_and_confirmed_across_turns(tmp_path) -> None:
    path = tmp_path / "resumes.sqlite3"
    store = ResumeStore(path)
    _, _, _, version = seed_resume(store)
    worker = RecordingResumeAnalysisWorker()
    draft_store = SQLiteResumeAnalysisDraftStore(path)
    history_store = CareerHistoryStore(path)
    service = ResumeAnalysisService(store, worker, draft_store, history_store)

    analyze_decisions = SequenceDecisionMaker(
        AgentDecision(action="tool_call", tool_call=ToolCall(name="list_resumes", arguments={})),
        AgentDecision(action="tool_call", tool_call=ToolCall(name="get_resume_metadata", arguments={"selection_index": 1})),
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name="analyze_resume", arguments={"selection_index": 1}),
        ),
        AgentDecision(action="final", message="请确认这次分析结果。"),
    )
    analyze_agent, _ = build_agent(
        tmp_path,
        store,
        analyze_decisions,
        resume_analysis_service=service,
    )
    analyze_result = analyze_agent.run_turn(
        user_id="u1", conversation_id="c1", user_message="分析这份简历"
    )
    analysis_id = analyze_result.context.task.active_resume_analysis_id
    assert analysis_id is not None

    review_decisions = SequenceDecisionMaker(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name="get_resume_analysis", arguments={}),
        ),
        AgentDecision(action="final", message="有一段候选经历等待确认。"),
    )
    review_agent, _ = build_agent(
        tmp_path,
        store,
        review_decisions,
        resume_analysis_service=service,
    )
    review_result = review_agent.run_turn(
        user_id="u1", conversation_id="c1", user_message="给我再看一下"
    )
    review_observation = review_result.tool_result
    assert review_observation.payload["analysis_id"] == analysis_id
    assert review_observation.payload["status"] == "pending"
    assert len(review_decisions.contexts) == 2

    confirm_decisions = SequenceDecisionMaker()
    confirm_agent, _ = build_agent(
        tmp_path,
        store,
        confirm_decisions,
        resume_analysis_service=service,
    )
    stale = confirm_agent.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="确认旧版本",
        interaction_response=InteractionResponse(
            interaction_id=interaction_id(
                "c1", "resume_analysis_confirmation", "older-analysis"
            ),
            scope="resume_analysis_confirmation",
            action="confirm",
        ),
    )
    assert stale.tool_result.state == "resume_analysis_decision_expired"
    assert len(history_store.list_records(user_id="u1")) == 0
    assert confirm_decisions.contexts == []

    confirmed = confirm_agent.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="确认并导入",
        interaction_response=InteractionResponse(
            interaction_id=interaction_id(
                "c1", "resume_analysis_confirmation", analysis_id
            ),
            scope="resume_analysis_confirmation",
            action="confirm",
        ),
    )

    confirm_observation = confirmed.tool_result
    assert confirm_observation.state == "resume_analysis_confirmed"
    assert confirm_observation.payload["analysis_id"] == analysis_id
    assert len(history_store.list_records(user_id="u1")) == 1
    assert len(history_store.list_evidence(user_id="u1")) == 2
    assert confirmed.context.task.resume_analysis_status == "confirmed"
    assert confirm_decisions.contexts == []

    repeated = confirm_agent.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="再次确认",
        interaction_response=InteractionResponse(
            interaction_id=interaction_id(
                "c1", "resume_analysis_confirmation", analysis_id
            ),
            scope="resume_analysis_confirmation",
            action="confirm",
        ),
    )
    assert repeated.tool_result.state == "resume_analysis_decision_expired"
    assert len(history_store.list_records(user_id="u1")) == 1
    assert len(history_store.list_evidence(user_id="u1")) == 2
    assert confirm_decisions.contexts == []
