import json
from datetime import datetime, timezone
from types import SimpleNamespace

from career_agent.agent.summary_text import DELIVERY_SUMMARY_LIMIT
from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    CareerProfileContext,
    ConversationTaskState,
    MainAgentContext,
    SavedJobCandidateContextItem,
    ToolCall,
    ToolResult,
    project_job_research_arguments,
)
from career_agent.agent.main_agent_reducers import reduce_task_state
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
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
                    name="route_to_capability", arguments={"domain": "job"}
                ),
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
    # F: the message is the model's. The report itself is delivered by the
    # card and is the same text H puts in the observation body, so the content
    # and leak assertions belong to that rendering, not to the reply.
    # Declared by the handler from the typed run result it already holds.
    assert result.tool_result.facts == {
        "cached": result.tool_result.payload["cached"],
        "finding_count": len(result.tool_result.payload["research"]["findings"]),
        "status": result.tool_result.payload["status"],
    }
    rendered = MainAgentRuntime._assistant_message(result.tool_result)
    assert rendered.startswith("# 公司调研")
    assert "[S1]" in rendered
    assert "https://example.com/product" in rendered
    assert "run-secret" not in rendered
    assert "report-secret" not in rendered
    assert "run-secret" not in result.assistant_message
    assert "report-secret" not in result.assistant_message
    assert result.context.task.active_job_research_report_id == "report-secret"
    assert result.tool_result.resource_ref is not None
    assert result.tool_result.resource_ref.title == "岗位研究报告"
    assert result.tool_result.resource_ref.description == (
        "公司调研；该岗位与企业检索可靠性直接相关。"
    )
    # The reference carries the report's identity for later entity checks,
    # while the model-facing projection still never sees the ids.
    assert result.tool_result.resource_ref.job_posting_id == "job-secret"
    assert result.tool_result.resource_ref.company_key is None
    assert "job-secret" not in json.dumps(
        result.context.model_context(), ensure_ascii=False
    )
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
    # Since F the row keeps the model's reply, not a receipt. What must not
    # change is that the report body stays in the entity behind the card.
    assert stored_reply == "研究完成。"
    assert "Enterprise Retrieval Product" not in stored_reply
    assert len(stored_reply) <= DELIVERY_SUMMARY_LIMIT


COMPANIES = {"job-h": "历史科技甲", "job-s": "示例科技"}


class TwoCompanyJobs:
    """A saved job at each company; search filters by the company named."""

    def search_saved_jobs(self, *, user_id, query, limit=20, include_dismissed=False):
        return tuple(
            StoredJobSummary(
                job_posting_id=job_posting_id,
                title="算法工程师",
                company_name=company,
                source_name="test",
                availability_status="active",
                captured_at=NOW,
                last_checked_at=NOW,
            )
            for job_posting_id, company in COMPANIES.items()
            if company in query
        )

    def get_job(self, *, user_id, job_posting_id):
        company = COMPANIES.get(job_posting_id)
        if company is None:
            return None
        return SimpleNamespace(
            posting=SimpleNamespace(
                id=job_posting_id, title="算法工程师", company_name=company
            )
        )


def _company_result(job_posting_id: str) -> JobResearchResult:
    base = _result()
    company = COMPANIES[job_posting_id]
    run = base.run.model_copy(
        update={
            "id": f"run-{job_posting_id}",
            "job_posting_id": job_posting_id,
            "report_id": f"report-{job_posting_id}",
        }
    )
    report = base.report.model_copy(
        update={
            "id": f"report-{job_posting_id}",
            "run_id": run.id,
            "job_posting_id": job_posting_id,
            "summary": f"{company}的主要竞争对手是{company}竞品。",
        }
    )
    return JobResearchResult(run=run, report=report, sources=base.sources, cached=False)


class TwoCompanyResearch:
    def __init__(self) -> None:
        self.reads: list[dict] = []

    def research(self, *, user_id, job_posting_id, **kwargs):
        return _company_result(job_posting_id)

    def get_report(self, *, user_id, report_id=None, job_posting_id=None):
        self.reads.append({"report_id": report_id, "job_posting_id": job_posting_id})
        if report_id is not None:
            job_posting_id = report_id.removeprefix("report-")
        return _company_result(job_posting_id)


class ScriptedDecisions:
    """Decisions that may look at the context they are deciding on."""

    def __init__(self, *steps) -> None:
        self.steps = list(steps)
        self.contexts: list[MainAgentContext] = []

    def decide(self, context, tool_specs):
        self.contexts.append(context)
        step = self.steps.pop(0)
        return step(context) if callable(step) else step


def _call(name: str, **arguments) -> AgentDecision:
    return AgentDecision(
        action="tool_call", tool_call=ToolCall(name=name, arguments=arguments)
    )


def test_a_report_handle_for_another_company_is_refused_before_it_is_read(
    tmp_path,
) -> None:
    """End to end: the borrowed handle never reaches the service.

    The trajectory ``a_report_made_this_turn_without_an_index_cannot_be_named``
    records the model answering a question about 示例科技 through the handle
    of 历史科技甲's earlier report. The refusal comes back as a soft
    ``invalid_input`` naming the grounded selection index, the model retries
    with it, and the report that is read is the company the user asked about.
    """
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    research = TwoCompanyResearch()
    tools = MainAgentToolRegistry(
        job_repository=TwoCompanyJobs(),
        job_research_service=research,
    )

    def borrow(context: MainAgentContext) -> AgentDecision:
        handles = tuple(context.reference_handles())
        assert len(handles) == 1
        return _call("get_job_research", reference=handles[0])

    decisions = ScriptedDecisions(
        _call("find_saved_jobs", query="历史科技甲"),
        _call("route_to_capability", domain="job"),
        _call("research_job", selection_index=1, focus="competitors"),
        AgentDecision(action="final", message="历史科技甲的调研好了。"),
        _call("find_saved_jobs", query="示例科技"),
        borrow,
        _call("get_job_research", selection_index=1),
        AgentDecision(action="final", message="示例科技的竞争对手如上。"),
    )
    runtime = MainAgentRuntime(
        context_manager=manager, decision_maker=decisions, tools=tools
    )
    runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="调研一下历史科技甲"
    )

    result = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="示例科技那份调研里，他们的主要竞争对手是谁？",
    )

    refused = decisions.contexts[-2].tool_observations[-1]
    assert refused.tool_name == "get_job_research"
    assert refused.state == "invalid_input"
    assert "selection_index 1（示例科技）" in refused.message
    # The service saw exactly one read, and it was the right company's.
    assert research.reads == [{"report_id": None, "job_posting_id": "job-s"}]
    assert result.tool_result is not None
    assert result.tool_result.state == "job_research_ready"
    assert result.context.task.active_job_research_report_id == "report-job-s"
    assert result.assistant_message == "示例科技的竞争对手如上。"
    assert "历史科技甲竞品" not in MainAgentRuntime._assistant_message(
        result.tool_result
    )


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


def test_reused_company_report_keeps_the_requested_job_active() -> None:
    result = _result()

    payload = MainAgentToolRegistry._job_research_payload(
        result, requested_job_posting_id="job-current"
    )

    assert payload["job_posting_id"] == "job-current"
    assert payload["anchor_job_posting_id"] == "job-secret"
    assert payload["anchored_by_other_job"] is True

    task = reduce_task_state(
        ConversationTaskState(active_job_posting_id="job-current"),
        ToolResult(
            tool_name="get_job_research",
            state="job_research_ready",
            message="已读取岗位研究报告。",
            payload=payload,
        ),
    )
    assert task.active_job_posting_id == "job-current"
    assert task.active_job_research_report_id == "report-secret"


def test_job_research_is_described_as_an_explicit_optional_capability() -> None:
    tools = MainAgentToolRegistry(
        job_repository=Jobs(),
        job_research_service=Research(),
    )
    schemas = {item["function"]["name"]: item for item in tools.schemas()}
    description = schemas["research_job"]["function"]["description"]

    assert "only when the user explicitly asks" in description
    assert "never start it automatically" in description
    assert "Optional current public-web research" in description
    assert "A generic JD cannot establish" in description
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
