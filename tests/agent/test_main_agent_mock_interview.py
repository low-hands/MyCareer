from __future__ import annotations

from datetime import datetime, timezone

import pytest

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    ApplicationCandidateContextItem,
    AttachedResumeContext,
    CareerProfileContext,
    ConversationTaskState,
    MainAgentContext,
    ResumeVersionCandidateContextItem,
    ToolCall,
    project_mock_interview_arguments,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime, RuntimeAction
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.agent.mock_interview_contracts import MockInterviewGraphResult
from career_agent.agent.mock_interview_graph import (
    MockInterviewCheckpointMissingError,
    MockInterviewInputRoutingError,
)
from career_agent.domain.applications import Application
from career_agent.domain.job_discovery import JobDetail, Provenance
from career_agent.domain.mock_interviews import (
    MockInterviewQuestionResult,
    MockInterviewReport,
)
from career_agent.harness.observability import InMemoryTraceRecorder
from career_agent.services.applications import ApplicationService
from career_agent.storage.applications import SQLiteApplicationStore
from career_agent.storage.context import CareerContextStore
from career_agent.storage.jobs import SQLiteJobPostingRepository
from career_agent.storage.resumes import ResumeStore


NOW = datetime(2026, 8, 27, tzinfo=timezone.utc)


class OneDecision:
    def __init__(self, decision: AgentDecision) -> None:
        self.decision = decision
        self.calls = 0

    def decide(self, context, tool_specs):
        self.calls += 1
        if self.calls > 1:
            raise AssertionError("Mock interview result must be presented without re-deciding")
        return self.decision


class FakeMockInterviewGraph:
    def __init__(self) -> None:
        self.starts = []
        self.resumes = []

    def start(self, request):
        self.starts.append(request)
        return MockInterviewGraphResult(
            session_id="mock-session-1",
            state="awaiting_answer",
            message="Answer the current mock interview question.",
            turn_id="turn-1",
            question="请介绍一个你亲自负责的 RAG 可靠性改进。",
        )

    def resume(self, *, user_id, session_id, answer):
        self.resumes.append(
            {"user_id": user_id, "session_id": session_id, "answer": answer}
        )
        return MockInterviewGraphResult(
            session_id=session_id,
            state="completed",
            message="Mock interview completed.",
            report_id="report-1",
            report=MockInterviewReport(
                id="report-1",
                session_id=session_id,
                completion_reason="plan_completed",
                summary="回答覆盖了核心问题，但量化验证还不够具体。",
                question_results=(
                    MockInterviewQuestionResult(
                        plan_item_number=1,
                        question="请介绍一个你亲自负责的 RAG 可靠性改进。",
                        final_rating="adequate",
                        summary="职责清楚，验证不足。",
                        follow_up_count=0,
                    ),
                ),
                strengths=("能说明个人职责",),
                development_areas=("补充验证指标",),
                practice_actions=("用 STAR 重写回答",),
                created_at=NOW,
            ),
        )

    def handle_input(self, *, user_id, session_id, message):
        return self.resume(user_id=user_id, session_id=session_id, answer=message)


def _application_setup(tmp_path):
    resumes = ResumeStore(tmp_path / "resumes.sqlite3")
    role = resumes.create_target_role(user_id="u1", title="AI Engineer", priority=1)
    _, version = resumes.import_document(
        user_id="u1",
        target_role_id=role.id,
        name="AI Resume",
        content=b"PRIVATE RESUME",
        document_format="text",
    )
    jobs = SQLiteJobPostingRepository(tmp_path / "jobs.sqlite3")
    stored = jobs.save_detail(
        user_id="u1",
        run_id="run-1",
        result_ref="ref-1",
        selection_index=1,
        detail=JobDetail(
            source_name="test",
            source_job_id="job-1",
            title="RAG Engineer",
            company_name="Acme",
            description="PRIVATE JD",
            captured_at=NOW,
            provenance=Provenance(
                source_name="test",
                source_job_id="job-1",
                captured_at=NOW,
                operation="detail",
                adapter_version="test-v1",
            ),
        ),
    )
    service = ApplicationService(
        SQLiteApplicationStore(tmp_path / "applications.sqlite3"), jobs, resumes
    )
    application = service.create_application(
        user_id="u1",
        job_posting_id=stored.posting.id,
        resume_version_id=version.id,
    ).application
    return service, application


def test_projection_selects_application_by_index_and_rejects_internal_ids() -> None:
    application = Application(
        id="app-2",
        user_id="u1",
        job_posting_id="job-2",
        jd_snapshot_id="snapshot-2",
        resume_version_id="resume-2",
        status="submitted",
        submitted_at=NOW,
        created_at=NOW,
        updated_at=NOW,
    )
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        task=ConversationTaskState(
            application_candidates=(
                ApplicationCandidateContextItem(
                    application_id=application.id,
                    title="RAG Engineer",
                    company_name="Acme",
                    status="submitted",
                ),
            )
        ),
        user_message="开始模拟面试",
    )

    projected = project_mock_interview_arguments(
        context,
        {"application_selection_index": 1, "interview_type": "technical"},
    )

    assert projected["application_id"] == "app-2"
    assert projected["interview_type"] == "technical"
    with pytest.raises(ValueError, match="internal identifiers"):
        project_mock_interview_arguments(context, {"application_id": "app-2"})


def test_free_practice_ignores_an_active_application_and_requires_type() -> None:
    context = MainAgentContext(
        conversation_id="c1", profile=CareerProfileContext(user_id="u1"),
        task=ConversationTaskState(active_application_id="app-1"),
        user_message="随便练练",
    )
    projected = project_mock_interview_arguments(
        context, {"practice_scope": "free", "interview_type": "behavioral"}
    )
    assert projected["application_id"] is None
    with pytest.raises(ValueError, match="requires interview_type"):
        project_mock_interview_arguments(context, {"practice_scope": "free"})


def test_runtime_starts_then_resumes_mock_interview_through_the_main_graph(
    tmp_path,
) -> None:
    service, application = _application_setup(tmp_path)
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    seed = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="选择这次投递"
    )
    manager.commit_turn(
        context=seed,
        task=ConversationTaskState(
            active_application_id=application.id, tool_profile="interview"
        ),
        assistant_message="已选择投递。",
    )
    graph = FakeMockInterviewGraph()
    decision_maker = OneDecision(
        AgentDecision(
            action="tool_call",
            tool_call=ToolCall(
                name="start_mock_interview",
                arguments={
                    "interview_type": "technical",
                    "max_primary_questions": 1,
                },
            ),
        )
    )
    tools = MainAgentToolRegistry(
        application_service=service,
        mock_interview_graph=graph,
    )
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=decision_maker,
        tools=tools,
    )

    started = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="开始技术模拟面试"
    )

    assert decision_maker.calls == 1
    assert graph.starts[0].application_id == application.id
    assert graph.starts[0].job_posting_id == application.job_posting_id
    assert graph.starts[0].jd_snapshot_id == application.jd_snapshot_id
    assert graph.starts[0].resume_version_id == application.resume_version_id
    assert started.context.task.active_workflow == "mock_interview"
    assert started.context.task.run_id == "mock-session-1"
    assert started.assistant_message == (
        "请介绍一个你亲自负责的 RAG 可靠性改进。"
    )
    # Mid-run the conversation has no trace of the interview at all, not even a
    # request waiting for an answer. It is held, so the whole run can be written
    # as one exchange when it ends and the stored history is never mid-turn.
    after_start = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="inspect"
    )
    assert [message.content for message in after_start.recent_messages] == [
        "选择这次投递",
        "已选择投递。",
    ]
    assert after_start.task.workflow_entry_message == "开始技术模拟面试"

    recorder = InMemoryTraceRecorder()
    events: list[object] = []
    completed = MainAgentRuntime(
        context_manager=manager,
        decision_maker=decision_maker,
        tools=tools,
        trace_recorder=recorder,
    ).run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="我负责设计离线评估集，并监控召回率。",
        event_sink=events.append,
    )

    assert decision_maker.calls == 1
    assert graph.resumes == [
        {
            "user_id": "u1",
            "session_id": "mock-session-1",
            "answer": "我负责设计离线评估集，并监控召回率。",
        }
    ]
    assert completed.context.task.active_workflow == "none"
    assert completed.context.task.run_id is None
    assert completed.delegated_write_count == 1
    trace = recorder.snapshot(events[0].turn_id)
    turn_completed = next(
        event for event in trace.events if event.event_type == "turn_completed"
    )
    assert turn_completed.details["write_call_count"] == 1
    assert not [
        event
        for event in trace.events
        if event.model_call_category == "orchestrator_decision"
    ]
    assert completed.assistant_message.startswith("模拟面试完成。")
    assert "补充验证指标" in completed.assistant_message
    # The run's own turns stay out of the main thread, but its outcome does not:
    # the candidate's answer is absent while the reply that closes the request is
    # present, carrying a reference to the report rather than a copy of it, so a
    # later turn can act on the interview without replaying it.
    after_completion = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="inspect"
    )
    contents = [message.content for message in after_completion.recent_messages]
    assert contents[:3] == ["选择这次投递", "已选择投递。", "开始技术模拟面试"]
    assert contents[3].startswith("模拟面试完成。")
    # A headline, not the report: bounded, and produced without a model call so
    # it is the same line whether or not the answer writer ran.
    assert len(contents[3]) <= 240
    assert "待提升" not in contents[3]
    reference = after_completion.recent_messages[3].resource_refs[0]
    assert reference is not None
    assert reference.kind == "mock_interview_report"
    # One request, one reply. Neither the questions nor the answers survive.
    assert len(contents) == 4
    assert not any("我负责设计离线评估集" in item for item in contents)
    assert not any("模拟面试题" in item for item in contents)


def test_start_schema_exposes_only_selection_indexes_not_internal_ids(tmp_path) -> None:
    service, _ = _application_setup(tmp_path)
    tools = MainAgentToolRegistry(
        application_service=service,
        mock_interview_graph=FakeMockInterviewGraph(),
    )

    schema = next(
        item for item in tools.schemas() if item["function"]["name"] == "start_mock_interview"
    )
    properties = schema["function"]["parameters"]["properties"]
    assert "application_selection_index" in properties
    assert "interview_selection_index" in properties
    assert all(not key.endswith("_id") for key in properties)


def test_runtime_mock_interview_entries_are_not_model_tools(tmp_path) -> None:
    service, _ = _application_setup(tmp_path)
    tools = MainAgentToolRegistry(
        application_service=service,
        mock_interview_graph=FakeMockInterviewGraph(),
    )

    public_names = {item["function"]["name"] for item in tools.schemas()}
    assert "handle_mock_interview_input" not in public_names
    assert "retry_mock_interview" not in public_names
    assert "handle_mock_interview_input" not in tools.names
    assert "retry_mock_interview" not in tools.names
    assert set(tools.runtime_workflow_names) == {
        "handle_mock_interview_input",
        "retry_mock_interview",
    }


class ReplayDecisions:
    """Answer every turn, unlike OneDecision which forbids a second call."""

    def __init__(self, decision: AgentDecision) -> None:
        self.decision = decision
        self.calls = 0

    def decide(self, context, tool_specs):
        self.calls += 1
        return self.decision


class LostCheckpointGraph(FakeMockInterviewGraph):
    """Business rows outlive the resumable thread."""

    def resume(self, *, user_id, session_id, answer):
        raise MockInterviewCheckpointMissingError("thread is gone")

    def retry(self, *, user_id, session_id):
        raise MockInterviewCheckpointMissingError("thread is gone")


class RoutingFailureGraph(FakeMockInterviewGraph):
    def handle_input(self, *, user_id, session_id, message):
        raise MockInterviewInputRoutingError(
            "INPUT_ROUTE_UNAVAILABLE",
            "router unavailable",
            retryable=True,
        )


def _runtime_with_graph(tmp_path, graph, decision_maker, runtime_class=MainAgentRuntime):
    service, application = _application_setup(tmp_path)
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    seed = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="选择这次投递"
    )
    manager.commit_turn(
        context=seed,
        task=ConversationTaskState(
            active_application_id=application.id, tool_profile="interview"
        ),
        assistant_message="已选择投递。",
    )
    tools = MainAgentToolRegistry(
        application_service=service,
        mock_interview_graph=graph,
    )
    runtime = runtime_class(
        context_manager=manager,
        decision_maker=decision_maker,
        tools=tools,
    )
    return runtime, manager


def _start_decision() -> AgentDecision:
    return AgentDecision(
        action="tool_call",
        tool_call=ToolCall(
            name="start_mock_interview",
            arguments={"interview_type": "technical", "max_primary_questions": 1},
        ),
    )


def test_a_dead_checkpoint_hands_the_conversation_back_with_a_trace(tmp_path) -> None:
    decision_maker = ReplayDecisions(_start_decision())
    runtime, manager = _runtime_with_graph(
        tmp_path, LostCheckpointGraph(), decision_maker
    )
    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="开始技术模拟面试")

    lost = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="我的回答"
    )

    assert lost.context.task.phase == "mock_interview_checkpoint_missing"
    # The run is unreachable, so the next message must reach the decision model.
    # Keeping the slot without this would loop on the same dead thread forever.
    calls_before = decision_maker.calls
    runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="那算了，帮我看看简历"
    )
    assert decision_maker.calls > calls_before

    contents = [
        message.content
        for message in manager.load_for_turn(
            user_id="u1", conversation_id="c1", user_message="inspect"
        ).recent_messages
    ]
    # The failure is visible and the answer is not: Main Agent can explain why
    # the interview stopped without the transcript being copied over.
    assert any("执行断点已经丢失" in item for item in contents)
    assert not any("我的回答" in item for item in contents)


def test_input_routing_failure_keeps_the_question_and_does_not_store_the_input(
    tmp_path,
) -> None:
    runtime, manager = _runtime_with_graph(
        tmp_path, RoutingFailureGraph(), ReplayDecisions(_start_decision())
    )
    started = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="开始技术模拟面试"
    )

    failed = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="这条回答不要落库"
    )

    assert started.context.task.phase == "mock_interview_answer_required"
    assert failed.context.task.phase == "mock_interview_answer_required"
    assert failed.tool_result.state == "mock_interview_input_retry_required"
    assert "尚未保存" in failed.assistant_message
    contents = [
        item.content
        for item in manager.load_for_turn(
            user_id="u1", conversation_id="c1", user_message="inspect"
        ).recent_messages
    ]
    assert not any("这条回答不要落库" in item for item in contents)


class CancellingGraph(FakeMockInterviewGraph):
    """Ends the run without a report, the other way out of the loop."""

    def resume(self, *, user_id, session_id, answer):
        return MockInterviewGraphResult(
            session_id=session_id,
            state="cancelled",
            message="模拟面试已取消。",
        )


def test_a_cancelled_run_still_leaves_its_ending_in_the_main_thread(tmp_path) -> None:
    runtime, manager = _runtime_with_graph(
        tmp_path, CancellingGraph(), ReplayDecisions(_start_decision())
    )
    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="开始技术模拟面试")

    cancelled = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="不想练了，取消"
    )

    assert cancelled.context.task.active_workflow == "none"
    contents = [
        message.content
        for message in manager.load_for_turn(
            user_id="u1", conversation_id="c1", user_message="inspect"
        ).recent_messages
    ]
    # Every exit writes a trace, not just the one that produces a report.
    assert any("取消" in item for item in contents)
    assert not any("不想练了" in item for item in contents)


@pytest.mark.parametrize(
    "graph_factory, closing",
    [
        (FakeMockInterviewGraph, "模拟面试完成。"),
        (CancellingGraph, "模拟面试已取消。"),
        (LostCheckpointGraph, "执行断点已经丢失"),
    ],
)
def test_every_exit_pairs_the_request_and_releases_the_hold(
    tmp_path, graph_factory, closing
) -> None:
    runtime, manager = _runtime_with_graph(
        tmp_path, graph_factory(), ReplayDecisions(_start_decision())
    )
    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="开始技术模拟面试")
    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="我的回答")

    loaded = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="inspect"
    )
    contents = [message.content for message in loaded.recent_messages]
    assert contents[-2] == "开始技术模拟面试"
    assert closing in contents[-1]
    # The hold is released on every exit, including the one that keeps the slot
    # to record why the run died. A leftover request would be answered by the
    # next run's report instead of by its own.
    assert loaded.task.workflow_entry_message is None


class MaximalReportGraph(FakeMockInterviewGraph):
    """A report with every field at its contract maximum."""

    def resume(self, *, user_id, session_id, answer):
        return MockInterviewGraphResult(
            session_id=session_id,
            state="completed",
            message="Mock interview completed.",
            report_id="report-1",
            report=MockInterviewReport(
                id="report-1",
                session_id=session_id,
                completion_reason="plan_completed",
                summary="回" * 3000,
                question_results=(
                    MockInterviewQuestionResult(
                        plan_item_number=1,
                        question="q",
                        final_rating="adequate",
                        summary="s",
                        follow_up_count=0,
                    ),
                ),
                strengths=tuple("亮" * 200 for _ in range(10)),
                development_areas=tuple(f"待{index}" + "提" * 200 for index in range(10)),
                practice_actions=tuple(f"练{index}" + "习" * 200 for index in range(10)),
                created_at=NOW,
            ),
        )


def test_a_long_report_reaches_the_screen_and_is_referenced_in_history(
    tmp_path,
) -> None:
    """The screen gets the whole report; history gets a reference to it.

    Storing the rendered report would put a section-shaped blob in the recent
    window, where a per-message cap could drop exactly the trailing sections the
    next turn needs. The durable row instead names the outcome and points at the
    report entity, so nothing has to be condensed to fit.
    """
    runtime, manager = _runtime_with_graph(
        tmp_path, MaximalReportGraph(), ReplayDecisions(_start_decision())
    )
    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="开始技术模拟面试")
    result = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="我的回答"
    )

    # Nothing is withheld from the candidate.
    assert "总结" in result.assistant_message
    assert "待提升" in result.assistant_message
    assert "练习建议" in result.assistant_message
    assert "练0" in result.assistant_message

    loaded = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="inspect"
    )
    stored = loaded.recent_messages[-1]
    # The report's own summary is at its 3000-character limit here, so this
    # pins the condensing: the durable row stays bounded no matter how long the
    # report is, which is the property the recent window depends on.
    assert stored.content.startswith("模拟面试完成。")
    assert len(stored.content) <= 240
    assert "待提升" not in stored.content
    assert stored.resource_refs
    assert stored.resource_refs[0].kind == "mock_interview_report"
    assert stored.resource_refs[0].resource_id == "report-1"
    # The model sees that a report exists without ever seeing its id.
    projected = loaded.model_context()["recent_messages"][-1]
    assert projected["resources"] == [
        {
            "kind": "mock_interview_report",
            "reference": loaded.reference_handle(stored.resource_refs[0]),
            "title": "模拟面试报告",
            "description": "回" * 199 + "…",
        }
    ]
    assert projected["resources"][0]["reference"].startswith("mock_")



def test_the_live_turn_streams_the_reference_the_transcript_will_carry(
    tmp_path,
) -> None:
    """The card must appear during the turn, not only after a reload.

    The stream is the only path to it mid-turn: the reply the user reads is a
    summary, and without this event the report would be reachable only by
    refreshing the page. Emitting the same reference the row stores is what
    keeps the live card and the restored one pointing at one report.
    """
    runtime, _ = _runtime_with_graph(
        tmp_path, MaximalReportGraph(), ReplayDecisions(_start_decision())
    )
    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="开始技术模拟面试")
    events: list[object] = []
    runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="我的回答",
        event_sink=events.append,
    )

    references = [event for event in events if event.type == "report_ready"]
    assert [(item.kind, item.resource_id) for item in references] == [
        ("mock_interview_report", "report-1")
    ]
    streamed = "".join(
        event.delta for event in events if event.type == "content_delta"
    )
    stored = runtime.context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="继续"
    ).recent_messages[-1]
    assert streamed == stored.content
    assert "待提升" not in streamed
    assert events[-1].type == "turn_completed"


def test_a_turn_with_no_report_streams_no_reference(tmp_path) -> None:
    """A card with nothing behind it is worse than none, so the turn is checked."""
    runtime, _ = _runtime_with_graph(
        tmp_path, FakeMockInterviewGraph(), ReplayDecisions(_start_decision())
    )
    events: list[object] = []
    runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="开始技术模拟面试",
        event_sink=events.append,
    )

    assert not [event for event in events if event.type == "report_ready"]


def test_every_entry_reports_the_call_that_actually_happened(tmp_path) -> None:
    """The decision model must not be told about a call that never occurred.

    ``_drive_mock_interview`` hardcoded ``tool_name="start_mock_interview"`` in
    all four entries and in every one of its error branches. That string is not
    telemetry: ``_tool_observation`` copies it straight into the decision
    context, so consuming a candidate's answer produced an observation reading
    "this turn called start_mock_interview" — naming a capability nobody
    invoked, on the very turn the answer was being consumed.

    Independent of G, which changes *who* drives these calls, not what they
    call themselves.
    """
    class RetryableGraph(FakeMockInterviewGraph):
        def retry(self, *, user_id, session_id):
            return self.resume(user_id=user_id, session_id=session_id, answer="stored")

    service, _ = _application_setup(tmp_path)
    tools = MainAgentToolRegistry(
        application_service=service,
        mock_interview_graph=RetryableGraph(),
    )

    consumed = tools.handle_mock_interview_input(
        user_id="u1", session_id="s1", message="我做过检索系统的端到端优化。"
    )
    retried = tools.retry_mock_interview(user_id="u1", session_id="s1")

    assert consumed.tool_name == "handle_mock_interview_input"
    assert retried.tool_name == "retry_mock_interview"

    # The failure branches carry the same obligation: a closed observation is
    # still an observation the model reads.
    failing = MainAgentToolRegistry(
        application_service=service,
        mock_interview_graph=RoutingFailureGraph(),
    )
    refused = failing.handle_mock_interview_input(
        user_id="u1", session_id="s1", message="我做过检索系统的端到端优化。"
    )
    assert refused.state == "mock_interview_input_retry_required"
    assert refused.tool_name == "handle_mock_interview_input"


@pytest.mark.parametrize(
    ("phase", "handler"),
    (
        ("mock_interview_running", "handle_mock_interview_input"),
        ("failed", "retry_mock_interview"),
    ),
)
def test_the_origin_names_the_workflow_not_the_handler_that_advanced_it(
    tmp_path, phase, handler
) -> None:
    """One workflow, two internal handlers, one reported origin.

    ``origin.label`` is published by the CLI and hashed into interaction ids, so
    it is an external surface. Naming the handler there would make that surface
    change whenever the runtime's own routing is refactored, and would expose a
    name that means nothing outside this module. Which handler ran is in the
    tool result, where an internal name belongs — asserted here so the claim is
    checked rather than assumed.
    """
    class RetryableGraph(FakeMockInterviewGraph):
        def retry(self, *, user_id, session_id):
            return self.resume(user_id=user_id, session_id=session_id, answer="stored")

    service, _ = _application_setup(tmp_path)
    runtime = MainAgentRuntime(
        context_manager=ContextManager(CareerContextStore(tmp_path / f"{phase}.sqlite3")),
        decision_maker=_never_called_decision_maker(),
        tools=MainAgentToolRegistry(
            application_service=service, mock_interview_graph=RetryableGraph()
        ),
    )
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        task=ConversationTaskState(
            active_workflow="mock_interview", run_id="s1", phase=phase
        ),
        user_message="[workflow-owned input withheld]",
    )

    result = runtime._run_owned_workflow_turn(context=context, user_message="继续")

    assert result.origin == RuntimeAction(workflow="mock_interview")
    assert result.origin.label == "workflow:mock_interview"
    assert result.tool_result.tool_name == handler


def test_a_turn_the_model_never_decided_says_so(tmp_path) -> None:
    """A bound workflow action uses the graph without consulting Main Agent."""
    class RetryableGraph(FakeMockInterviewGraph):
        def retry(self, *, user_id, session_id):
            return self.resume(user_id=user_id, session_id=session_id, answer="stored")

    service, _ = _application_setup(tmp_path)
    tools = MainAgentToolRegistry(
        application_service=service,
        mock_interview_graph=RetryableGraph(),
    )
    runtime = MainAgentRuntime(
        context_manager=ContextManager(CareerContextStore(tmp_path / "ctx.sqlite3")),
        decision_maker=_never_called_decision_maker(),
        tools=tools,
    )
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        task=ConversationTaskState(
            active_workflow="mock_interview", run_id="s1", phase="mock_interview_running"
        ),
        user_message="[workflow-owned input withheld]",
    )

    result = runtime._run_owned_workflow_turn(
        context=context, user_message="我做过检索系统的端到端优化。"
    )

    # The user supplied the answer; the runtime, not the user and not the model,
    # decided it belongs to the workflow it owns. That distinction is now the
    # variant rather than a second enum beside a fabricated decision — and the
    # fabrication is gone, so there is no model decision to misread.
    assert result.origin == RuntimeAction(workflow="mock_interview")
    assert result.requested_by == "user"
    assert result.model_decision is None
    assert result.delegated_write_count == 1
    assert result.context.tool_observations[-1].arguments == {}
    assert "我做过检索系统的端到端优化。" not in str(
        result.context.model_context()
    )


def test_runtime_workflow_action_obeys_the_standard_write_budget(tmp_path) -> None:
    graph = FakeMockInterviewGraph()
    service, _ = _application_setup(tmp_path)
    runtime = MainAgentRuntime(
        context_manager=ContextManager(CareerContextStore(tmp_path / "ctx.sqlite3")),
        decision_maker=_never_called_decision_maker(),
        tools=MainAgentToolRegistry(
            application_service=service,
            mock_interview_graph=graph,
        ),
    )
    context = MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        task=ConversationTaskState(
            active_workflow="mock_interview",
            run_id="s1",
            phase="mock_interview_running",
        ),
        user_message="[workflow-owned input withheld]",
    )
    decision = AgentDecision(
        action="tool_call",
        tool_call=ToolCall(name="handle_mock_interview_input", arguments={}),
    )

    state = runtime._graph.invoke(
        {
            "context": context,
            "decision": decision,
            "pending": {
                "name": "handle_mock_interview_input",
                "runtime_owned": True,
                "arguments": {"message": "PRIVATE ANSWER"},
            },
            "tool_results": (),
            "artifact_ids": (),
            "control": {
                "read_calls": 0,
                "write_calls": 1,
                "projection_refusals": 0,
                "authorization_refusals": 0,
                "fingerprints": (),
                "retryable_fingerprints": (),
                "retry_counts": {},
            },
        }
    )

    assert graph.resumes == []
    assert state["control"]["write_calls"] == 1
    assert state["control"]["authorization_refusals"] == 1
    assert "预算已经用完" in state["assistant_message"]
    assert "PRIVATE ANSWER" not in str(state["context"].model_context())


class _never_called_decision_maker:
    def decide(self, context, schemas):  # pragma: no cover - must not be reached
        raise AssertionError("a workflow-owned turn consults no Main Agent model")


def test_the_progress_events_name_the_entry_that_actually_ran(tmp_path) -> None:
    """The standard act node announces the runtime entry that actually ran."""
    announced: list[str] = []

    class RecordingRuntime(MainAgentRuntime):
        def _emit_capability_started(self, name):
            announced.append(name)

        def _emit_capability_completed(self, name, state):
            announced.append(name)

    runtime, _ = _runtime_with_graph(
        tmp_path,
        FakeMockInterviewGraph(),
        ReplayDecisions(_start_decision()),
        runtime_class=RecordingRuntime,
    )
    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="开始技术模拟面试")
    announced.clear()
    runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="我的回答"
    )

    assert announced == [
        "handle_mock_interview_input",
        "handle_mock_interview_input",
    ]


class ScriptedDecisions:
    def __init__(self, *decisions: AgentDecision) -> None:
        self.decisions = list(decisions)

    def decide(self, context, tool_specs):
        return self.decisions.pop(0)


def _free_start(**arguments) -> AgentDecision:
    return AgentDecision(
        action="tool_call",
        tool_call=ToolCall(
            name="start_mock_interview",
            arguments={"practice_scope": "free", "interview_type": "behavioral", **arguments},
        ),
    )


def test_free_practice_asks_which_resume_before_starting_anything(tmp_path) -> None:
    resumes = ResumeStore(tmp_path / "resumes.sqlite3")
    role = resumes.create_target_role(user_id="u1", title="PM", priority=1)
    _, fixture = resumes.import_document(
        user_id="u1", target_role_id=role.id, name="测试简历",
        content=b"fixture", document_format="text",
    )
    _, real = resumes.import_document(
        user_id="u1", target_role_id=role.id, name="正式简历",
        content=b"real", document_format="text",
    )
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    seed = manager.load_for_turn(user_id="u1", conversation_id="c1", user_message="想练面试")
    manager.commit_turn(
        context=seed, task=ConversationTaskState(tool_profile="interview"),
        assistant_message="好的。",
    )
    graph = FakeMockInterviewGraph()
    tools = MainAgentToolRegistry(
        application_service=object(), mock_interview_graph=graph, resume_store=resumes,
    )
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=ScriptedDecisions(
            _free_start(), _free_start(resume_version_selection_index=2)
        ),
        tools=tools,
    )

    asked = runtime.run_turn(user_id="u1", conversation_id="c1", user_message="来一场行为面")

    # Nothing started: the latest resume is not picked on the user's behalf.
    assert graph.starts == []
    assert asked.context.task.active_workflow == "none"
    card = MainAgentRuntime._interaction_event(result=asked, conversation_id="c1")
    assert card is not None and card.kind == "single_selection"
    assert card.accepts_upload == "resume"
    assert [option.label for option in card.options] == [
        "《正式简历》v1", "《测试简历》v1", "不用简历",
    ]

    chosen = runtime.run_turn(user_id="u1", conversation_id="c1", user_message="《测试简历》v1")

    assert [request.resume_version_id for request in graph.starts] == [fixture.id]
    assert chosen.context.task.active_workflow == "mock_interview"
    assert real.id != fixture.id


def test_free_practice_resume_comes_only_from_the_user() -> None:
    def context(**updates) -> MainAgentContext:
        return MainAgentContext(
            conversation_id="c1", profile=CareerProfileContext(user_id="u1"),
            task=updates.pop("task", None) or ConversationTaskState(
                active_resume_version_id="resume-seen-earlier",
                resume_version_candidates=(
                    ResumeVersionCandidateContextItem(
                        resume_version_id="resume-offered", version_number=1,
                        source_type="user_import", document_format="pdf", byte_size=1,
                        resume_name="正式简历",
                    ),
                ),
            ),
            user_message="练一下", **updates,
        )

    free = {"practice_scope": "free", "interview_type": "behavioral"}
    unchosen = project_mock_interview_arguments(context(), free)
    assert (unchosen["resume_choice"], unchosen["resume_version_id"]) == ("required", None)
    picked = project_mock_interview_arguments(
        context(), {**free, "resume_version_selection_index": 1}
    )
    assert picked["resume_version_id"] == "resume-offered"
    none = project_mock_interview_arguments(context(), {**free, "without_resume": True})
    assert (none["resume_choice"], none["resume_version_id"]) == ("none", None)
    attached = project_mock_interview_arguments(
        context(attached_resumes=(
            AttachedResumeContext(
                resume_version_id="resume-attached", resume_id="r", resume_name="附件",
                version_number=1, is_latest_version=True, document_format="pdf",
                byte_size=1, uploaded_at=NOW,
            ),
        )),
        free,
    )
    assert attached["resume_version_id"] == "resume-attached"
    with pytest.raises(ValueError, match="free practice only"):
        project_mock_interview_arguments(
            context(task=ConversationTaskState(active_application_id="app-1")),
            {"interview_type": "technical", "without_resume": True},
        )


def _save_job(jobs, *, source_job_id: str, company: str, title: str, jd: str):
    return jobs.save_detail(
        user_id="u1", run_id=f"run-{source_job_id}", result_ref=f"ref-{source_job_id}",
        selection_index=1,
        detail=JobDetail(
            source_name="test", source_job_id=source_job_id, title=title,
            company_name=company, description=jd, captured_at=NOW,
            provenance=Provenance(
                source_name="test", source_job_id=source_job_id, captured_at=NOW,
                operation="detail", adapter_version="test-v1",
            ),
        ),
    )


class FakeResearch:
    """Research keyed exactly by company name, like the real store."""

    def __init__(self, reports: dict[str, object]) -> None:
        self.reports = reports
        self.looked_up: list[str] = []

    def latest_company_report(self, *, user_id, company_name):
        self.looked_up.append(company_name)
        return self.reports.get(company_name)

    def find_report(self, *, user_id, report_id):
        return next(
            (report for report in self.reports.values() if report.id == report_id), None
        )


class _Report:
    def __init__(self, report_id: str) -> None:
        self.id = report_id
        self.created_at = NOW


def _company_runtime(tmp_path, decisions, *, research):
    resumes = ResumeStore(tmp_path / "resumes.sqlite3")
    role = resumes.create_target_role(user_id="u1", title="PM", priority=1)
    _, resume = resumes.import_document(
        user_id="u1", target_role_id=role.id, name="正式简历",
        content=b"real", document_format="text",
    )
    jobs = SQLiteJobPostingRepository(tmp_path / "jobs.sqlite3")
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    seed = manager.load_for_turn(user_id="u1", conversation_id="c1", user_message="想练面试")
    manager.commit_turn(
        context=seed, task=ConversationTaskState(tool_profile="interview"),
        assistant_message="好的。",
    )
    graph = FakeMockInterviewGraph()
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=ScriptedDecisions(*decisions),
        tools=MainAgentToolRegistry(
            application_service=object(), mock_interview_graph=graph,
            resume_store=resumes, job_repository=jobs, job_research_service=research,
        ),
    )
    return runtime, graph, jobs, resume


def test_naming_a_company_offers_its_saved_jobs_then_practises_on_the_chosen_jd(
    tmp_path,
) -> None:
    research = FakeResearch({"字节跳动": _Report("research-bytedance")})
    runtime, graph, jobs, resume = _company_runtime(
        tmp_path,
        (
            _free_start(company_name="字节"),
            _free_start(job_selection_index=1, company_name="字节"),
            _free_start(job_selection_index=1, resume_version_selection_index=1),
        ),
        research=research,
    )
    job = _save_job(
        jobs, source_job_id="bd-1", company="字节跳动", title="AI 产品经理", jd="负责大模型产品。",
    )
    _save_job(jobs, source_job_id="other", company="阿里巴巴", title="产品经理", jd="电商。")

    offered = runtime.run_turn(user_id="u1", conversation_id="c1", user_message="模拟面试，字节")
    job_card = MainAgentRuntime._interaction_event(result=offered, conversation_id="c1")
    assert graph.starts == []
    assert [option.label for option in job_card.options] == [
        "字节跳动 · AI 产品经理", "不针对具体岗位，只按「字节」",
    ]

    asked = runtime.run_turn(user_id="u1", conversation_id="c1", user_message="字节跳动 · AI 产品经理")
    # The job is settled; the resume is still the user's to choose.
    assert graph.starts == []
    assert asked.tool_result.state == "mock_interview_resume_choice_required"

    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="《正式简历》v1")

    [request] = graph.starts
    assert (request.job_posting_id, request.jd_snapshot_id) == (job.posting.id, job.snapshot.id)
    assert request.resume_version_id == resume.id
    assert request.target_company is None and request.application_id is None
    # Research is looked up by the job's own company name, and pinned.
    assert request.company_research_report_id == "research-bytedance"


def test_company_only_practice_keeps_the_name_and_uses_no_research_it_does_not_have(
    tmp_path,
) -> None:
    research = FakeResearch({"字节跳动": _Report("research-bytedance")})
    runtime, graph, _, _ = _company_runtime(
        tmp_path,
        (_free_start(company_name="字节", without_resume=True),),
        research=research,
    )

    runtime.run_turn(user_id="u1", conversation_id="c1", user_message="按字节的风格练，不用简历")

    [request] = graph.starts
    assert (request.target_company, request.job_posting_id) == ("字节", None)
    # "字节" is not guessed to be "字节跳动": no research is used.
    assert request.company_research_report_id is None
    assert research.looked_up == ["字节"]
