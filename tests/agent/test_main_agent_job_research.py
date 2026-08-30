from datetime import datetime, timezone

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    CareerProfileContext,
    ConversationTaskState,
    MainAgentContext,
    SavedJobCandidateContextItem,
    ToolCall,
    project_job_research_arguments,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.agent.openai_compatible_main_agent import (
    OpenAICompatibleMainAgentDecisionMaker,
)
from career_agent.agent.job_research_presenter import render_job_research
from career_agent.domain.job_research import (
    JobResearchDraft,
    JobResearchFinding,
    JobResearchFindingDraft,
    JobResearchReport,
    JobResearchRun,
    JobResearchScope,
    JobResearchSource,
    JobResearchSourceDraft,
)
from career_agent.services.job_research import JobResearchResult
from career_agent.storage.context import CareerContextStore
from career_agent.storage.jobs import StoredJobSummary


NOW = datetime(2026, 8, 30, tzinfo=timezone.utc)


class Gateway:
    def advance(self, **kwargs):
        raise AssertionError("job discovery should not run")


class Jobs:
    def search_saved_jobs(self, **kwargs):
        return (
            StoredJobSummary(
                job_posting_id="job-secret",
                title="RAG Engineer",
                company_name="Example Corp",
                source_name="test",
                availability_status="active",
                captured_at=NOW,
                last_checked_at=NOW,
            ),
        )


def _result() -> JobResearchResult:
    scope = JobResearchScope(focus="technical context", max_sources=5)
    run = JobResearchRun(
        id="run-secret",
        user_id="u1",
        job_posting_id="job-secret",
        jd_snapshot_id="jd-secret",
        scope=scope,
        status="completed",
        input_fingerprint="a" * 64,
        worker_version="v1",
        report_id="report-secret",
        started_at=NOW,
        completed_at=NOW,
        updated_at=NOW,
    )
    report = JobResearchReport(
        id="report-secret",
        run_id="run-secret",
        user_id="u1",
        job_posting_id="job-secret",
        jd_snapshot_id="jd-secret",
        status="current",
        scope=scope,
        summary="该岗位与企业检索可靠性直接相关。",
        findings=(JobResearchFinding(
            topic="产品场景",
            statement="公开产品材料显示其服务企业检索场景。",
            evidence_type="fact",
            source_keys=("S1",),
            confidence="high",
        ),),
        open_questions=("具体团队如何衡量检索质量？",),
        created_at=NOW,
    )
    source = JobResearchSource(
        id="source-secret",
        run_id="run-secret",
        source_key="S1",
        url="https://example.com/product",
        normalized_url="https://example.com/product",
        title="Enterprise Retrieval Product",
        publisher="Example Corp",
        retrieved_at=NOW,
        relevant_excerpt="The product supports enterprise retrieval.",
        content_sha256="b" * 64,
    )
    return JobResearchResult(run=run, report=report, sources=(source,), cached=False)


class Research:
    def __init__(self):
        self.calls = []

    def research(self, **kwargs):
        self.calls.append(kwargs)
        return _result()


class Decisions:
    def __init__(self):
        self.values = [
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(name="find_saved_jobs", arguments={"query": "RAG"}),
            ),
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(
                    name="research_job",
                    arguments={
                        "selection_index": 1,
                        "focus": "technical context",
                        "user_provided_context": "一面提到企业知识库产品线。",
                        "max_sources": 5,
                    },
                ),
            ),
            AgentDecision(action="final", message="研究完成。"),
        ]

    def decide(self, context, tool_specs):
        return self.values.pop(0)


def test_main_agent_runs_research_and_delivers_full_report_outside_context(
    tmp_path,
) -> None:
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    research = Research()
    tools = MainAgentToolRegistry(
        Gateway(),
        job_repository=Jobs(),
        job_research_service=research,
    )
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=Decisions(),
        tools=tools,
    )

    result = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="研究第一个岗位",
    )

    assert research.calls == [{
        "user_id": "u1",
        "job_posting_id": "job-secret",
        "focus": "technical context",
        "user_provided_context": "一面提到企业知识库产品线。",
        "max_sources": 5,
    }]
    assert result.assistant_message.startswith("# 岗位研究")
    assert "[S1]" in result.assistant_message
    assert "https://example.com/product" in result.assistant_message
    assert "run-secret" not in result.assistant_message
    assert "report-secret" not in result.assistant_message
    assert result.context.task.active_job_research_report_id == "report-secret"
    assert result.context.model_context()["task"]["job_research_status"] == "current"
    assert tools.capability_kind("research_job") == "workflow"
    assert tools.capability_kind("get_job_research") == "atomic_tool"
    schemas = {item["function"]["name"]: item for item in tools.schemas()}
    assert "job_posting_id" not in schemas["research_job"]["function"]["parameters"]["properties"]
    assert "run_id" not in schemas["retry_job_research"]["function"]["parameters"]["properties"]
    assert "report_id" not in schemas["get_job_research"]["function"]["parameters"]["properties"]

    loaded = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="继续"
    )
    stored_reply = loaded.recent_messages[-1].content
    assert stored_reply.startswith("岗位研究已完成。")
    assert "Enterprise Retrieval Product" not in stored_reply


def test_job_research_projection_uses_indexes_and_hides_internal_ids() -> None:
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        task=ConversationTaskState(
            saved_job_candidates=(SavedJobCandidateContextItem(
                job_posting_id="job-secret",
                title="RAG Engineer",
                company_name="Example Corp",
            ),),
            active_job_research_run_id="run-secret",
            job_research_status="failed",
        ),
        user_message="重试",
    )

    assert project_job_research_arguments(
        context,
        "research_job",
        {"selection_index": 1},
    )["job_posting_id"] == "job-secret"
    assert project_job_research_arguments(
        context,
        "retry_job_research",
        {},
    )["run_id"] == "run-secret"


def test_job_research_is_described_as_an_explicit_optional_capability() -> None:
    tools = MainAgentToolRegistry(
        Gateway(),
        job_repository=Jobs(),
        job_research_service=Research(),
    )
    schemas = {item["function"]["name"]: item for item in tools.schemas()}
    description = schemas["research_job"]["function"]["description"]
    prompt = OpenAICompatibleMainAgentDecisionMaker._system_prompt(tuple(schemas))

    assert "only when the user explicitly asks" in description
    assert "never start it automatically" in description
    assert "optional user-requested add-on" in prompt
    assert "generic JD does not prove" in prompt
    properties = schemas["research_job"]["function"]["parameters"]["properties"]
    assert "user_provided_context" in properties


def test_presenter_labels_user_context_as_an_unverified_search_lead() -> None:
    result = _result()
    draft = JobResearchDraft(
        summary=result.report.summary,
        sources=(JobResearchSourceDraft(
            source_key="S1",
            url=result.sources[0].url,
            title=result.sources[0].title,
            publisher=result.sources[0].publisher,
            relevant_excerpt=result.sources[0].relevant_excerpt,
        ),),
        findings=(JobResearchFindingDraft(
            topic=result.report.findings[0].topic,
            statement=result.report.findings[0].statement,
            evidence_type=result.report.findings[0].evidence_type,
            source_keys=result.report.findings[0].source_keys,
            confidence=result.report.findings[0].confidence,
        ),),
        open_questions=result.report.open_questions,
    )

    rendered = render_job_research(
        draft,
        status="current",
        user_provided_context="一面提到知识库产品。\n尚未公开确认。",
    )

    assert "## 用户提供的检索线索" in rendered
    assert "> 一面提到知识库产品。\n> 尚未公开确认。" in rendered
    assert "不视为事实" in rendered
