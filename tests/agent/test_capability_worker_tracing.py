"""Capability model calls share the outer turn trace without leaking content."""

from __future__ import annotations

from types import SimpleNamespace

from pathlib import Path

import pytest

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.deepagent_job_research_worker import (
    DeepAgentJobResearchWorker,
)
from career_agent.agent.deepagent_resume_tailoring_worker import (
    DeepAgentResumeFinalizationWorker,
    DeepAgentResumeTailoringWorker,
)
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    CareerProfileContext,
    ToolCall,
    ToolObservation,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.agent.openai_compatible_agent_worker import (
    OpenAICompatibleAgentWorker,
)
from career_agent.agent.openai_compatible_client import AgentWorkerError
from career_agent.agent.openai_email_tracking_worker import OpenAIEmailTrackingWorker
from career_agent.agent.openai_interview_preparation_worker import (
    OpenAIInterviewPreparationWorker,
)
from career_agent.agent.openai_mock_interview_worker import OpenAIMockInterviewWorker
from career_agent.agent.openai_resume_analysis_worker import OpenAIResumeAnalysisWorker
from career_agent.agent.openai_resume_job_match_worker import (
    OpenAIResumeJobMatchWorker,
)
from career_agent.agent.openai_resume_tailoring_reviewer import (
    OpenAIResumeTailoringReviewer,
)
from career_agent.harness.observability import (
    ACTIVE_TRACE_CONTEXT,
    CapabilityModelTraceCallback,
    InMemoryTraceRecorder,
    traced_model_call,
)
from career_agent.storage.context import CareerContextStore


@pytest.mark.parametrize(
    ("worker", "method"),
    (
        (OpenAICompatibleAgentWorker, "decide"),
        (OpenAIEmailTrackingWorker, "assess"),
        (OpenAIInterviewPreparationWorker, "prepare"),
        (OpenAIResumeAnalysisWorker, "analyze"),
        (OpenAIResumeJobMatchWorker, "match"),
        (OpenAIMockInterviewWorker, "_request_structured"),
        (OpenAIResumeTailoringReviewer, "_review"),
        (DeepAgentJobResearchWorker, "research"),
        (DeepAgentResumeTailoringWorker, "tailor"),
        (DeepAgentResumeFinalizationWorker, "finalize"),
    ),
)
def test_every_production_capability_model_entry_is_traced(worker, method) -> None:
    """A new call site must make an explicit trace decision in this matrix."""

    assert hasattr(getattr(worker, method), "__wrapped__")


def test_capability_events_share_the_outer_turn_without_input_or_output(
    tmp_path: Path,
) -> None:
    class Worker:
        @traced_model_call("synthetic_capability_worker")
        def run(self, sensitive_input: str) -> str:
            assert sensitive_input == "PRIVATE RESUME AND JD"
            return "PRIVATE MODEL OUTPUT"

    class Registry(MainAgentToolRegistry):
        def __init__(self) -> None:
            super().__init__(job_repository=object())
            self.worker = Worker()

        def invoke_atomic_tool(self, name, arguments):
            assert name == "find_saved_jobs"
            self.worker.run("PRIVATE RESUME AND JD")
            return ToolObservation(
                tool_name=name,
                state="no_saved_jobs_found",
                message="没有找到匹配的已保存岗位。",
                payload={"items": []},
            )

    class Decisions:
        def __init__(self) -> None:
            self.values = [
                AgentDecision(
                    action="tool_call",
                    tool_call=ToolCall(
                        name="find_saved_jobs", arguments={"query": "AI"}
                    ),
                ),
                AgentDecision(action="final", message="查询完成。"),
            ]

        def decide(self, context, tool_specs):
            return self.values.pop(0)

    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    recorder = InMemoryTraceRecorder()
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=Decisions(),
        tools=Registry(),
        trace_recorder=recorder,
    )

    runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="查一下保存的岗位"
    )

    assert len(recorder._events) == 1
    events = next(iter(recorder._events.values()))
    capability = [
        event
        for event in events
        if event.stage == "synthetic_capability_worker"
    ]
    assert [event.event_type for event in capability] == [
        "model_attempt",
        "model_succeeded",
    ]
    assert all(
        event.model_call_category == "capability_agent" for event in capability
    )
    serialized = " ".join(event.model_dump_json() for event in capability)
    assert "PRIVATE RESUME AND JD" not in serialized
    assert "PRIVATE MODEL OUTPUT" not in serialized


def test_capability_failure_keeps_worker_error_classification() -> None:
    class Worker:
        @traced_model_call("failing_capability_worker")
        def run(self) -> None:
            raise AgentWorkerError(
                "CAPABILITY_RATE_LIMITED",
                "public explanation",
                detail="private provider detail",
                retryable=True,
            )

    recorder = InMemoryTraceRecorder()
    token = ACTIVE_TRACE_CONTEXT.set((recorder, "turn-1"))
    try:
        with pytest.raises(AgentWorkerError):
            Worker().run()
    finally:
        ACTIVE_TRACE_CONTEXT.reset(token)

    events = recorder.snapshot("turn-1").events
    assert [event.event_type for event in events] == [
        "model_attempt",
        "model_failed",
    ]
    assert events[-1].error_code == "CAPABILITY_RATE_LIMITED"
    assert events[-1].recoverable is True
    assert events[-1].error_detail == "AgentWorkerError"
    assert "private provider detail" not in events[-1].model_dump_json()


def test_deep_agent_callback_counts_each_inner_model_request_without_content() -> None:
    recorder = InMemoryTraceRecorder()
    callback = CapabilityModelTraceCallback(
        stage="job_research",
        worker="DeepAgentJobResearchWorker",
    )
    token = ACTIVE_TRACE_CONTEXT.set((recorder, "turn-deep"))
    try:
        callback.on_chat_model_start(
            {}, [["PRIVATE JD"]], run_id="inner-1"
        )
        callback.on_llm_end("PRIVATE RESULT", run_id="inner-1")
        callback.on_chat_model_start(
            {}, [["PRIVATE FOLLOW-UP"]], run_id="inner-2"
        )
        callback.on_llm_end("PRIVATE RESULT 2", run_id="inner-2")
    finally:
        ACTIVE_TRACE_CONTEXT.reset(token)

    events = recorder.snapshot("turn-deep").events
    assert [event.event_type for event in events] == [
        "model_attempt",
        "model_succeeded",
        "model_attempt",
        "model_succeeded",
    ]
    assert all(event.stage == "job_research" for event in events)
    assert all(event.model_call_category == "capability_agent" for event in events)
    serialized = " ".join(event.model_dump_json() for event in events)
    assert "PRIVATE JD" not in serialized
    assert "PRIVATE FOLLOW-UP" not in serialized
    assert "PRIVATE RESULT" not in serialized


def test_a_provider_run_search_is_announced_once_by_id_and_carries_no_query() -> None:
    """The one step the request-level callbacks cannot produce.

    A provider-hosted search happens inside a single model request, so without
    this the user watches "正在调研岗位背景" for the whole minute. Streaming
    surfaces the search as a ``web_search_call`` block, which repeats across
    chunks as the block accumulates: the same search must announce itself once,
    and a second search must be distinguishable from a repeat of the first.

    The block also carries the query the model chose. Search terms derive from
    the JD and the conversation, so the assertion is not only "a step appeared"
    but "the step contains nothing from the block except its ordinal".
    """
    from career_agent.harness.capability_steps import (
        CapabilityStep,
        observing_capability_steps,
    )

    callback = CapabilityModelTraceCallback(
        stage="job_research", worker="DeepAgentJobResearchWorker"
    )

    def chunk(*blocks: dict) -> object:
        return SimpleNamespace(message=SimpleNamespace(content=list(blocks)))

    secret = "量霸科技 Agent 开发实习 招聘"
    first = {"type": "web_search_call", "id": "ws_1", "query": secret}
    steps: list[CapabilityStep] = []
    with observing_capability_steps(steps.append):
        callback.on_llm_new_token("", chunk={"not": "a message"})
        callback.on_llm_new_token("", chunk=chunk({"type": "text", "text": secret}))
        callback.on_llm_new_token("", chunk=chunk(first))
        callback.on_llm_new_token("", chunk=chunk(first))  # same block, later chunk
        callback.on_llm_new_token(
            "", chunk=chunk({"type": "web_search_call", "id": "ws_2", "query": secret})
        )
        callback.on_llm_new_token("", chunk=chunk({"type": "web_search_call"}))

    assert steps == [
        CapabilityStep(stage="job_research.web_search", kind="io", index=1),
        CapabilityStep(stage="job_research.web_search", kind="io", index=2),
    ]
    assert secret not in repr(steps)


def test_the_search_step_reads_as_progress_not_as_an_internal_name() -> None:
    """An unlabelled stage stays silent, so the label is part of the feature."""
    from career_agent.harness.capability_steps import CapabilityStep

    step = CapabilityStep(stage="job_research.web_search", kind="io", index=2)
    label = MainAgentRuntime._capability_step_label(step)
    assert label == "正在检索公开资料"
    # An io step counts searches, so it must not borrow the model-call wording.
    assert MainAgentRuntime._capability_step_message(label, step) == (
        "正在检索公开资料（第 2 次）……"
    )
    first = CapabilityStep(stage="job_research.web_search", kind="io", index=1)
    assert MainAgentRuntime._capability_step_message(label, first) == "正在检索公开资料……"


def test_capability_steps_reach_the_installed_observer_without_a_trace() -> None:
    from career_agent.harness.capability_steps import (
        CapabilityStep,
        observing_capability_steps,
    )
    from career_agent.harness.observability import CapabilityToolStepCallback

    class Worker:
        @traced_model_call("resume_analysis")
        def run(self) -> str:
            return "PRIVATE RESULT"

    model_callback = CapabilityModelTraceCallback(
        stage="job_research", worker="DeepAgentJobResearchWorker"
    )
    tool_callback = CapabilityToolStepCallback(stage="job_research")
    steps: list[CapabilityStep] = []

    assert ACTIVE_TRACE_CONTEXT.get() is None
    with observing_capability_steps(steps.append):
        assert Worker().run() == "PRIVATE RESULT"
        model_callback.on_chat_model_start({}, [["PRIVATE JD"]], run_id="r1")
        model_callback.on_llm_error(
            AgentWorkerError("JOB_RESEARCH_TIMEOUT", "timed out", retryable=True),
            run_id="r1",
        )
        model_callback.on_chat_model_start({}, [["PRIVATE JD"]], run_id="r2")
        tool_callback.on_tool_start({"name": "read_file"}, "PRIVATE PATH")
        model_callback.on_llm_end("PRIVATE RESULT", run_id="r2")
        # An error of unknown retryability is not announced as a retry (093):
        # the user would be told "retrying" before a failure that will not be.
        model_callback.on_chat_model_start({}, [["PRIVATE JD"]], run_id="r3")
        model_callback.on_llm_error(RuntimeError("boom"), run_id="r3")
    Worker().run()

    assert steps == [
        CapabilityStep(stage="resume_analysis", kind="model"),
        CapabilityStep(stage="job_research", kind="model", index=1),
        CapabilityStep(stage="job_research", kind="retry"),
        CapabilityStep(stage="job_research", kind="model", index=2),
        CapabilityStep(stage="job_research.read_file", kind="tool"),
        CapabilityStep(stage="job_research", kind="model", index=3),
    ]
    assert "PRIVATE" not in repr(steps)
