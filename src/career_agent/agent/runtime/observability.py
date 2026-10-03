from __future__ import annotations

from collections.abc import Callable
from threading import Event, Thread
from time import perf_counter
from typing import Any, Literal
from uuid import uuid4

from career_agent.agent.resources.input import (
    InputResourceNotFoundError,
    InputResourceRejectedError,
)
from career_agent.agent.runtime.interaction_coordinator import QuestionnaireContinuationError
from career_agent.agent.capabilities.registry import MainAgentToolOutput
from career_agent.agent.runtime.state import PendingAction
from career_agent.agent.providers.openai_client import (
    AgentWorkerError,
    public_error_code,
    worker_failure_reason,
)
from career_agent.agent.runtime.turn_coordinator import STREAM_SINK
from career_agent.agent.contracts.turn import MainAgentTurnResult
from career_agent.harness.capability_steps import (
    CapabilityStep,
    observing_capability_steps,
)
from career_agent.harness.observability import (
    EventType,
    ModelCallCategory,
    TraceRecorder,
    conversation_trace_key,
    record_active_trace,
)
from career_agent.harness.streaming import (
    CapabilityCompletedEvent,
    CapabilityStartedEvent,
    ProgressEvent,
    PublicStreamEvent,
    StreamEventSink,
    TurnFailedEvent,
)


class RuntimeObservability:
    """Best-effort tracing and progress for the synchronous agent runtime."""

    CAPABILITY_LABELS = {
        "job_search": "正在处理岗位检索……",
        "job_research": "正在调研岗位相关业务信息……",
        "resume": "正在处理简历……",
        "application_tracking": "正在处理投递进展……",
        "interview": "正在处理面试任务……",
        "calendar": "正在准备日历操作……",
        "action_center": "正在整理待办事项……",
        "career_task": "正在执行职业任务……",
    }
    CAPABILITY_COMPLETED_LABELS = {
        "job_search": "岗位检索步骤已完成。",
        "job_research": "岗位调研步骤已完成。",
        "resume": "简历处理步骤已完成。",
        "application_tracking": "投递进展处理已完成。",
        "interview": "面试处理步骤已完成。",
        "calendar": "日历准备步骤已完成。",
        "action_center": "待办整理步骤已完成。",
        "career_task": "职业任务步骤已完成。",
    }
    COMPACTION_MESSAGES = {
        "load": ("loading_context", "正在压缩较早的对话记录，稍后开始判断……"),
        "commit": ("saving", "正在压缩较早的对话记录，回复已送达，可以先看……"),
    }
    CAPABILITY_STEP_MESSAGES = {
        "resume_transcription": "正在识别简历文字",
        "resume_document_prepare": "正在准备简历文档",
        "resume_job_match": "正在比对简历与岗位要求",
        "job_analysis": "正在分析岗位 JD",
        "resume_job_match_state_audit": "正在核对简历比对结果",
        "resume_tailoring": "正在起草定制简历",
        "resume_draft_review": "正在审校简历草稿",
        "resume_finalization": "正在定稿简历",
        "resume_final_review": "正在审校定稿简历",
        "job_research": "正在调研岗位背景",
        "job_research.web_search": "正在检索公开资料",
        "interview_preparation": "正在准备面试资料",
        "mock_interview_plan": "正在规划模拟面试",
        "mock_interview_input_route": "正在理解你的回答",
        "mock_interview_ask": "正在生成面试问题",
        "mock_interview_follow_up": "正在判断是否需要追问",
        "mock_interview_evaluate": "正在逐题评估你的回答",
        "mock_interview_report": "正在整理面试报告",
        "email_tracking_assess": "正在识别招聘邮件",
        "email_sync.fetch": "正在读取邮箱",
        "email_sync.scan": "正在扫描邮件",
    }
    CAPABILITY_TOOL_MESSAGES = {
        "read_file": "正在阅读工作指南",
        "ls": "正在查找工作指南",
        "glob": "正在查找工作指南",
        "grep": "正在检索工作指南",
    }

    def __init__(self, *, trace_recorder: TraceRecorder | None) -> None:
        self._trace_recorder = trace_recorder

    @staticmethod
    def emit(event: PublicStreamEvent) -> None:
        sink = STREAM_SINK.get()
        if sink is None:
            return
        try:
            sink(event)
        except Exception:
            return

    @staticmethod
    def public_capability(name: str) -> str:
        if name in {"open_job_search", "find_saved_jobs", "get_saved_job", "analyze_job"}:
            return "job_search"
        if "job_research" in name or name in {"research_job", "retry_job_research"}:
            return "job_research"
        if "resume" in name:
            return "resume"
        if "calendar" in name:
            return "calendar"
        if "interview" in name:
            return "interview"
        if "application" in name or "email" in name:
            return "application_tracking"
        if "action" in name or "daily_brief" in name:
            return "action_center"
        return "career_task"

    @classmethod
    def emit_capability_started(cls, name: str) -> None:
        capability = cls.public_capability(name)
        cls.emit(CapabilityStartedEvent(
            capability=capability,
            message=cls.CAPABILITY_LABELS[capability],
        ))

    @classmethod
    def emit_capability_completed(cls, name: str, state: str) -> None:
        capability = cls.public_capability(name)
        cls.emit(CapabilityCompletedEvent(
            capability=capability,
            state=state,
            message=cls.CAPABILITY_COMPLETED_LABELS[capability],
        ))

    @classmethod
    def emit_turn_failure(
        cls,
        *,
        turn_id: str,
        error: Exception,
        reply_delivered: bool,
    ) -> None:
        if reply_delivered:
            event = TurnFailedEvent(
                turn_id=turn_id,
                code="TURN_COMMIT_FAILED",
                message="回复已生成，但本轮状态未能保存；刷新后这条回复可能不会保留。",
            )
        elif isinstance(error, InputResourceNotFoundError):
            event = TurnFailedEvent(
                turn_id=turn_id,
                code="INPUT_RESOURCE_NOT_FOUND",
                message="附带的简历版本不存在或不属于当前用户，请重新选择后再发送。",
            )
        elif isinstance(error, InputResourceRejectedError):
            event = TurnFailedEvent(
                turn_id=turn_id,
                code="INPUT_RESOURCE_REJECTED",
                message="当前模拟面试进行中，此时不会读取附带的简历。请完成或退出当前流程后再发送。",
            )
        elif isinstance(error, QuestionnaireContinuationError):
            event = TurnFailedEvent(
                turn_id=turn_id, code=error.code, message=error.user_message
            )
        elif isinstance(error, AgentWorkerError):
            event = TurnFailedEvent(
                turn_id=turn_id,
                code=public_error_code(error),
                message=f"本轮任务未完成：{worker_failure_reason(error)}",
            )
        else:
            event = TurnFailedEvent(
                turn_id=turn_id,
                code="TURN_EXECUTION_FAILED",
                message="本轮处理失败，请稍后重试。",
            )
        cls.emit(event)

    @staticmethod
    def emit_trace(
        event_type: Literal["capability_failed", "presentation_degraded"],
        stage: str,
        *,
        error_code: str | None = None,
        error_detail: str | None = None,
        details: dict[str, Any] | None = None,
        recoverable: bool | None = None,
    ) -> None:
        RuntimeObservability.record_trace_event(
            event_type,
            stage,
            outcome="failed",
            error_code=error_code,
            error_detail=error_detail,
            details=details,
            recoverable=recoverable,
        )

    @staticmethod
    def record_trace_event(
        event_type: EventType,
        stage: str,
        *,
        outcome: Literal["started", "succeeded", "failed", "interrupted"],
        duration_ms: int | None = None,
        error_code: str | None = None,
        error_detail: str | None = None,
        details: dict[str, Any] | None = None,
        recoverable: bool | None = None,
        model_call_category: ModelCallCategory | None = None,
    ) -> None:
        record_active_trace(
            event_type,
            stage,
            outcome=outcome,
            duration_ms=duration_ms,
            error_code=error_code,
            error_detail=error_detail,
            details=details,
            recoverable=recoverable,
            model_call_category=model_call_category,
        )

    def record_turn(
        self, *, turn_id: str, conversation_id: str, result: MainAgentTurnResult
    ) -> None:
        if self._trace_recorder is None:
            return
        try:
            self._trace_recorder.record(
                turn_id,
                "turn_completed",
                "turn",
                outcome="succeeded",
                details={
                    "conversation_id": conversation_id,
                    "tool_call_count": result.delegated_read_count + result.delegated_write_count,
                    "read_call_count": result.delegated_read_count,
                    "write_call_count": result.delegated_write_count,
                    "final_state": (
                        result.tool_result.state
                        if result.tool_result is not None
                        else result.origin.label
                    ),
                },
            )
        except Exception:
            return

    def record_rejected_turn(self, *, user_id: str, conversation_id: str) -> None:
        if self._trace_recorder is None:
            return
        try:
            self._trace_recorder.record(
                uuid4().hex,
                "turn_rejected",
                "gate",
                outcome="interrupted",
                details={"conversation_id": conversation_id, "user_id": user_id},
            )
        except Exception:
            return

    def record_capture_continuation(
        self,
        *,
        user_id: str,
        conversation_id: str | None,
        capture_event_id: str | None,
        phase: Literal["intent", "settled"],
        status: str,
        turn_id: str | None = None,
        error_code: str | None = None,
    ) -> None:
        if self._trace_recorder is None:
            return
        details: dict[str, Any] = {"phase": phase, "status": status}
        if conversation_id is not None:
            details["conversation_key"] = conversation_trace_key(user_id, conversation_id)
        if turn_id is not None:
            details["turn_id"] = turn_id
        try:
            self._trace_recorder.record(
                capture_event_id or uuid4().hex,
                "capture_continuation",
                "job_capture",
                outcome="failed" if status == "failed" else "succeeded",
                details=details,
                error_code=error_code,
            )
        except Exception:
            return

    def record_turn_failed(
        self,
        *,
        turn_id: str,
        conversation_id: str,
        error: Exception,
        reply_delivered: bool = False,
    ) -> None:
        if self._trace_recorder is None:
            return
        error_code = getattr(error, "code", None)
        if not isinstance(error_code, str) or not error_code.strip():
            error_code = "TURN_EXECUTION_FAILED"
        retryable = getattr(error, "retryable", None)
        if not isinstance(retryable, bool):
            retryable = None
        details: dict[str, Any] = {
            "conversation_id": conversation_id,
            "error_type": type(error).__name__,
        }
        if reply_delivered:
            details["reply_delivered"] = True
        try:
            self._trace_recorder.record(
                turn_id,
                "turn_failed",
                "turn",
                outcome="failed",
                details=details,
                error_code=error_code,
                error_detail=str(error),
                recoverable=retryable,
            )
        except Exception:
            return

    @classmethod
    def announce_compaction(cls, phase: str) -> None:
        stage, message = cls.COMPACTION_MESSAGES[phase]
        cls.emit(ProgressEvent(stage=stage, message=message))

    @staticmethod
    def heartbeat(
        sink: StreamEventSink | None,
        *,
        interval: float,
        stage: str,
        describe: Callable[[int], str],
        step_fields: Callable[[], dict[str, str]] | None = None,
    ) -> Event:
        stop = Event()
        if sink is None:
            return stop
        started = perf_counter()

        def beat() -> None:
            while not stop.wait(interval):
                waited = int(perf_counter() - started)
                try:
                    sink(ProgressEvent(
                        stage=stage,
                        message=describe(waited),
                        **(step_fields() if step_fields is not None else {}),
                    ))
                except Exception:
                    return

        Thread(target=beat, name=f"{stage}-heartbeat", daemon=True).start()
        return stop

    @classmethod
    def capability_step_label(cls, step: CapabilityStep) -> str | None:
        if step.kind == "tool":
            _, _, tool = step.stage.rpartition(".")
            return cls.CAPABILITY_TOOL_MESSAGES.get(tool)
        return cls.CAPABILITY_STEP_MESSAGES.get(step.stage)

    @staticmethod
    def capability_step_message(label: str, step: CapabilityStep) -> str:
        if step.kind == "retry":
            return f"{label}时请求失败，正在重试……"
        if step.index is not None and step.total is not None:
            return f"{label}（第 {step.index}/{step.total} 项）……"
        if step.kind == "io" and step.index is not None and step.index > 1:
            return f"{label}（第 {step.index} 次）……"
        if step.index is not None and step.index > 1:
            return f"{label}（第 {step.index} 次调用模型）……"
        return f"{label}……"

    @classmethod
    def run_capability(
        cls,
        pending: PendingAction,
        run: Callable[[], MainAgentToolOutput],
        *,
        heartbeat_interval: float,
    ) -> MainAgentToolOutput:
        capability = cls.public_capability(pending["name"])
        latest: tuple[str | None, str] = (
            None,
            cls.CAPABILITY_LABELS[capability].rstrip("…"),
        )

        def on_step(step: CapabilityStep) -> None:
            nonlocal latest
            label = cls.capability_step_label(step)
            if label is None:
                return
            latest = (step.stage, label)
            cls.emit(ProgressEvent(
                stage="running_capability",
                message=cls.capability_step_message(label, step),
                step_key=step.stage,
                step_label=label,
            ))

        def current_step_fields() -> dict[str, str]:
            key, label = latest
            return {"step_key": key, "step_label": label} if key is not None else {}

        heartbeat = cls.heartbeat(
            STREAM_SINK.get(),
            interval=heartbeat_interval,
            stage="running_capability",
            describe=lambda waited: f"{latest[1]}（已等待 {waited} 秒）……",
            step_fields=current_step_fields,
        )
        try:
            with observing_capability_steps(on_step):
                return run()
        finally:
            heartbeat.set()
