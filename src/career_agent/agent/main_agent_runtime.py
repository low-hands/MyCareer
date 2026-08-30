from __future__ import annotations

import json
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.career_context import CareerContextProjector
from career_agent.agent.job_discovery_gateway import JobDiscoveryGatewayResult
from career_agent.agent.main_agent_contracts import AgentDecision, CandidateContextItem, ConversationTaskState, DecisionMaker, DecisionObservation, MainAgentContext, ToolCall, ToolObservation, project_action_center_arguments, project_calendar_arguments, project_email_arguments, project_interview_arguments, project_interview_preparation_arguments, project_job_discovery_arguments, project_job_research_arguments, project_mock_interview_arguments, project_mock_interview_result_arguments, project_restart_mock_interview_arguments, project_resume_arguments, project_saved_job_arguments
from career_agent.agent.main_agent_reducers import reduce_task_state
from career_agent.agent.main_agent_tools import MainAgentToolOutput, MainAgentToolRegistry
from career_agent.agent.interview_preparation_presenter import render_interview_preparation
from career_agent.agent.job_research_presenter import render_job_research
from career_agent.agent.mock_interview_contracts import MockInterviewGraphResult
from career_agent.domain.interview_preparation import InterviewPreparationResult
from career_agent.domain.job_research import (
    JobResearchDraft,
    JobResearchFindingDraft,
    JobResearchSourceDraft,
)
from career_agent.domain.resume import ResumeArtifactDelivery

# Bounds for the stored copy of a report. Chosen so the rendered message stays
# well inside a single message's share of the recent-context budget even when
# every field arrives at its contract maximum.
_HISTORY_SUMMARY_CHARS = 600
_HISTORY_ITEM_CHARS = 120
_HISTORY_ITEMS_PER_SECTION = 5


def _clip(text: str, limit: int) -> str:
    """Cut to a length, marking the cut so a reader can tell it happened."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1] + "…"


def _clip_items(items: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        _clip(item, _HISTORY_ITEM_CHARS)
        for item in items[:_HISTORY_ITEMS_PER_SECTION]
    )


class MainAgentState(TypedDict, total=False):
    context: MainAgentContext
    decision: AgentDecision
    pending_capability_name: str
    pending_tool_result: MainAgentToolOutput
    last_tool_result: MainAgentToolOutput
    tool_results: tuple[MainAgentToolOutput, ...]
    tool_call_fingerprints: tuple[str, ...]
    artifact_ids: tuple[str, ...]
    tool_call_count: int
    assistant_message: str


class MainAgentTurnResult:
    def __init__(self, *, decision: AgentDecision, context: MainAgentContext, assistant_message: str, tool_result: MainAgentToolOutput | None = None, tool_results: tuple[MainAgentToolOutput, ...] = (), artifacts: tuple[ResumeArtifactDelivery, ...] = ()) -> None:
        self.decision = decision
        self.context = context
        self.assistant_message = assistant_message
        self.tool_result = tool_result
        self.tool_results = tool_results
        self.artifacts = artifacts


class MainAgentRuntime:
    _WAITING_STATES = frozenset({"selection_required", "waiting_user", "detail_unavailable", "email_events_pending", "calendar_approval_required", "resume_tailoring_review_blocked", "resume_final_review_blocked", "resume_tailoring_superseded", "failed"})

    def __init__(self, *, context_manager: ContextManager, decision_maker: DecisionMaker, tools: MainAgentToolRegistry, career_context_projector: CareerContextProjector | None = None, max_tool_calls: int = 3, owned_resources: tuple[Any, ...] = ()) -> None:
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least one")
        self._context_manager = context_manager
        # Exposed for the CLI's post-turn maintenance notice, which is an
        # operator concern and deliberately never reaches the decision model.
        self.context_manager = context_manager
        self._decision_maker = decision_maker
        self._tools = tools
        self._career_context_projector = career_context_projector
        self._max_tool_calls = max_tool_calls
        self._owned_resources = owned_resources
        self._closed = False

        graph = StateGraph(MainAgentState)
        graph.add_node("hydrate_career_context", self._hydrate_career_context)
        graph.add_node("decide", self._decide)
        graph.add_node("invoke_atomic_tool", self._invoke_atomic_tool)
        graph.add_node("run_workflow", self._run_workflow)
        graph.add_node("observe", self._observe)
        graph.add_node("finish", self._finish)
        graph.add_node("present_workflow", self._present_workflow)
        graph.add_node("fallback", self._fallback)
        graph.add_edge(START, "hydrate_career_context")
        graph.add_edge("hydrate_career_context", "decide")
        graph.add_conditional_edges(
            "decide",
            self._after_decision,
            {
                "invoke_atomic_tool": "invoke_atomic_tool",
                "run_workflow": "run_workflow",
                "finish": "finish",
                "fallback": "fallback",
            },
        )
        graph.add_edge("invoke_atomic_tool", "observe")
        graph.add_edge("run_workflow", "observe")
        graph.add_conditional_edges(
            "observe",
            self._after_observe,
            {"decide": "decide", "present_workflow": "present_workflow"},
        )
        graph.add_edge("finish", END)
        graph.add_edge("present_workflow", END)
        graph.add_edge("fallback", END)
        self._graph = graph.compile()

    def close(self) -> None:
        if self._closed:
            return
        for resource in reversed(self._owned_resources):
            close = getattr(resource, "close", None)
            if close is not None:
                close()
        self._closed = True

    def run_turn(self, *, user_id: str, conversation_id: str, user_message: str) -> MainAgentTurnResult:
        routing_task = self._context_manager.get_task(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if self._owns_next_turn(routing_task):
            context = self._context_manager.load_for_workflow_turn(
                user_id=user_id,
                conversation_id=conversation_id,
                task=routing_task,
            )
            result = self._run_active_mock_interview(
                context=context,
                user_message=user_message,
            )
            # One decision point for every way a run can end, so no exit path
            # can forget to leave a trace. The test is whether the workflow will
            # still be driving the next turn, not whether it still holds the
            # slot: a dead checkpoint keeps the slot to record why it died, yet
            # hands the conversation back, and that turn needs a trace too.
            if self._owns_next_turn(result.context.task):
                self._context_manager.commit_workflow_turn(
                    context=context,
                    task=result.context.task,
                )
            else:
                self._context_manager.commit_workflow_exit(
                    context=context,
                    task=result.context.task,
                    # The candidate keeps the full report on screen; only the
                    # stored copy is condensed, because only it has to fit
                    # alongside the rest of the conversation next turn.
                    assistant_message=self._history_message(
                        result.tool_result,
                        screen=result.assistant_message,
                    ),
                )
            return result

        context = self._context_manager.load_for_turn(
            user_id=user_id,
            conversation_id=conversation_id,
            user_message=user_message,
        )
        result = self._run_loaded_context(context)
        # This input belonged to Main Agent even when its result hands future
        # turns to a workflow. Ownership is an ingress property, not something
        # that can be inferred from the task state after execution. The reply,
        # however, did come from the workflow: it is the run's first question,
        # withheld on the same grounds as every question after it.
        if self._owns_next_turn(result.context.task):
            held = self._context_manager.commit_workflow_entry(
                context=context,
                task=result.context.task,
            )
            # Report the state that was stored, or the next turn would resume
            # from a task whose held request the caller never saw.
            result.context = result.context.model_copy(update={"task": held})
        else:
            self._context_manager.commit_turn(
                context=context,
                task=result.context.task,
                assistant_message=self._history_message(
                    result.tool_result,
                    screen=result.assistant_message,
                ),
            )
        return result

    @staticmethod
    def _owns_next_turn(task: ConversationTaskState) -> bool:
        """Whether the mock interview will consume the next user message.

        Holding the workflow slot is not enough. These two phases keep it only
        to record why the run cannot continue; the run itself is unreachable, so
        the next message has to reach the decision model or the conversation
        would have no way out.
        """
        return task.active_workflow == "mock_interview" and task.phase not in {
            "mock_interview_checkpoint_missing",
            "mock_interview_graph_incompatible",
        }

    def _run_loaded_context(self, context: MainAgentContext) -> MainAgentTurnResult:
        state = self._graph.invoke(
            {
                "context": context,
                "tool_call_fingerprints": (),
                "artifact_ids": (),
                "tool_results": (),
                "tool_call_count": 0,
            }
        )
        tool_result = state.get("last_tool_result")
        artifacts = tuple(
            self._tools.deliver_resume_artifact(
                user_id=context.profile.user_id,
                artifact_id=artifact_id,
            )
            for artifact_id in state.get("artifact_ids", ())
        )
        return MainAgentTurnResult(
            decision=state["decision"],
            context=state["context"],
            assistant_message=state["assistant_message"],
            tool_result=tool_result,
            tool_results=state.get("tool_results", ()),
            artifacts=artifacts,
        )

    def _decide(self, state: MainAgentState) -> MainAgentState:
        return {"decision": self._decision_maker.decide(state["context"], self._tools.schemas())}

    def _run_active_mock_interview(
        self, *, context: MainAgentContext, user_message: str
    ) -> MainAgentTurnResult:
        session_id = context.task.run_id
        if session_id is None:
            raise ValueError("Active mock interview has no resumable session")
        if context.task.phase == "failed":
            # The answer for this turn is already durable; the step after it
            # failed. Re-drive from the store rather than treating this message
            # as a new answer, which the turn would reject as conflicting and
            # leave the candidate unable to leave the failed phase at all.
            result = self._tools.retry_mock_interview(
                user_id=context.profile.user_id,
                session_id=session_id,
            )
        else:
            result = self._tools.handle_mock_interview_input(
                user_id=context.profile.user_id,
                session_id=session_id,
                message=user_message,
            )
        updated = self._update_mock_interview_task(context, result)
        updated = updated.model_copy(
            update={
                "tool_observations": (
                    *updated.tool_observations,
                    self._tool_observation("start_mock_interview", result),
                )[-3:]
            }
        )
        decision = AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name="start_mock_interview", arguments={}),
        )
        return MainAgentTurnResult(
            decision=decision,
            context=updated,
            assistant_message=self._assistant_message(result),
            tool_result=result,
            tool_results=(result,),
        )

    def _hydrate_career_context(self, state: MainAgentState) -> MainAgentState:
        if self._career_context_projector is None:
            return {}
        context = state["context"]
        memory = self._career_context_projector.project(
            user_id=context.profile.user_id,
            query=context.user_message,
        )
        return {"context": context.model_copy(update={"career_memory": memory})}

    @staticmethod
    def _tool_call_fingerprint(decision: AgentDecision) -> str:
        if decision.tool_call is None:
            return ""
        return json.dumps(
            {"name": decision.tool_call.name, "arguments": decision.tool_call.arguments},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _after_decision(self, state: MainAgentState) -> Literal["invoke_atomic_tool", "run_workflow", "finish", "fallback"]:
        decision = state["decision"]
        if decision.action != "tool_call":
            return "finish"
        if decision.tool_call is None:
            raise ValueError("tool_call action requires tool_call arguments")
        if state.get("tool_call_count", 0) >= self._max_tool_calls:
            return "fallback"
        if self._tool_call_fingerprint(decision) in state.get("tool_call_fingerprints", ()):
            return "fallback"
        last_result = state.get("last_tool_result")
        if last_result is not None and last_result.state in self._WAITING_STATES:
            return "fallback"
        kind = self._tools.capability_kind(decision.tool_call.name)
        if kind == "atomic_tool":
            return "invoke_atomic_tool"
        return "run_workflow"

    def _invoke_atomic_tool(self, state: MainAgentState) -> MainAgentState:
        context = state["context"]
        decision = state["decision"]
        if decision.tool_call is None:
            raise ValueError("tool_call action requires tool_call arguments")
        arguments = self._project_atomic_tool_arguments(context, decision.tool_call.name, decision.tool_call.arguments)
        result = self._tools.invoke_atomic_tool(decision.tool_call.name, arguments)
        return {"pending_capability_name": decision.tool_call.name, "pending_tool_result": result}

    def _run_workflow(self, state: MainAgentState) -> MainAgentState:
        context = state["context"]
        decision = state["decision"]
        if decision.tool_call is None:
            raise ValueError("tool_call action requires tool_call arguments")
        name = decision.tool_call.name
        if name == "job_discovery":
            arguments = project_job_discovery_arguments(context, decision.tool_call.arguments)
        elif name == "sync_application_emails":
            arguments = project_email_arguments(context, name, decision.tool_call.arguments)
        elif name in {"research_job", "retry_job_research"}:
            arguments = project_job_research_arguments(
                context,
                name,
                decision.tool_call.arguments,
            )
        elif name == "start_mock_interview":
            arguments = project_mock_interview_arguments(
                context, decision.tool_call.arguments
            )
        elif name == "restart_mock_interview":
            arguments = project_restart_mock_interview_arguments(
                context, decision.tool_call.arguments
            )
        else:
            raise ValueError(f"Unknown main-agent workflow: {name}")
        result = self._tools.invoke_workflow(name, arguments)
        return {"pending_capability_name": name, "pending_tool_result": result}

    def _observe(self, state: MainAgentState) -> MainAgentState:
        context = state["context"]
        result = state["pending_tool_result"]
        capability_name = state["pending_capability_name"]
        if isinstance(result, JobDiscoveryGatewayResult):
            updated = self._update_task(context, result)
        elif isinstance(result, ToolObservation) and result.tool_name in {
            "start_mock_interview",
            "restart_mock_interview",
        }:
            updated = self._update_mock_interview_task(context, result)
        else:
            updated = self._update_atomic_task(context, result)
        observation = self._tool_observation(capability_name, result)
        updated = updated.model_copy(update={"tool_observations": (*updated.tool_observations, observation)[-3:]})
        fingerprint = self._tool_call_fingerprint(state["decision"])
        artifact_ids = state.get("artifact_ids", ())
        if isinstance(result, ToolObservation) and result.state == "resume_artifact_ready":
            artifact_id = result.payload.get("artifact_id")
            if isinstance(artifact_id, str) and artifact_id not in artifact_ids:
                artifact_ids = (*artifact_ids, artifact_id)
        return {
            "context": updated,
            "last_tool_result": result,
            "tool_results": (*state.get("tool_results", ()), result),
            "tool_call_fingerprints": (*state.get("tool_call_fingerprints", ()), fingerprint),
            "tool_call_count": state.get("tool_call_count", 0) + 1,
            "artifact_ids": artifact_ids,
        }

    @staticmethod
    def _after_observe(
        state: MainAgentState,
    ) -> Literal["decide", "present_workflow"]:
        # Both entries into a run end the turn on the question they just asked.
        # A restart is a start with a retirement in front of it, so routing it
        # back to the decision model would put the first question of the new run
        # behind another tool call instead of in front of the candidate.
        if state.get("pending_capability_name") in {
            "start_mock_interview",
            "restart_mock_interview",
        }:
            return "present_workflow"
        return "decide"

    @staticmethod
    def _present_workflow(state: MainAgentState) -> MainAgentState:
        return {
            "assistant_message": MainAgentRuntime._assistant_message(
                state["last_tool_result"]
            )
        }

    @staticmethod
    def _finish(state: MainAgentState) -> MainAgentState:
        decision = state["decision"]
        result = state.get("last_tool_result")
        if result is not None and decision.action == "final":
            # The decision model only saw DecisionObservation, never the full
            # result. A final answer about that result must therefore come from
            # the authoritative presenter rather than ungrounded model prose.
            message = MainAgentRuntime._assistant_message(result)
        else:
            message = decision.message or ""
            if not message and result is not None:
                message = MainAgentRuntime._assistant_message(result)
        return {"assistant_message": message}

    @staticmethod
    def _fallback(state: MainAgentState) -> MainAgentState:
        result = state.get("last_tool_result")
        if result is not None:
            return {"assistant_message": MainAgentRuntime._assistant_message(result)}
        return {"assistant_message": "本轮可执行步骤已达到上限，请确认后继续。"}

    @staticmethod
    def _tool_observation(name: str, result: MainAgentToolOutput) -> DecisionObservation:
        if isinstance(result, ToolObservation):
            return DecisionObservation(
                tool_name=result.tool_name,
                state=result.state,
                next_action=result.next_action,
            )
        return DecisionObservation(
            tool_name=name,
            state=result.state,
            next_action=result.next_action,
        )

    @staticmethod
    def _history_message(result: MainAgentToolOutput, *, screen: str) -> str:
        """Render the run's outcome small enough to survive as history.

        The screen copy is unbounded, but the stored copy shares a budget with
        every other message the next turn reads, so it is cut at a fixed length
        on the way in. A full report can exceed that, and the sections lost are
        the ones at the end: what to work on and what to practise. Rebuilding
        the message from bounded parts keeps every section present instead of
        keeping the first half of the first one.
        """
        if not isinstance(result, ToolObservation):
            return screen
        if result.state == "mock_interview_completed":
            report = MockInterviewGraphResult.model_validate(result.payload).report
            if report is None:
                return screen
            sections = (
                ("总结", (_clip(report.summary, _HISTORY_SUMMARY_CHARS),)),
                ("待提升", _clip_items(report.development_areas)),
                ("练习建议", _clip_items(report.practice_actions)),
            )
            heading = "模拟面试完成。"
        elif result.state == "interview_preparation_ready":
            preparation = MainAgentRuntime._interview_preparation_result(result)
            if preparation is None:
                return screen
            sections = (
                ("总结", (_clip(preparation.summary, _HISTORY_SUMMARY_CHARS),)),
                (
                    "准备重点",
                    _clip_items(tuple(item.topic for item in preparation.focus_areas)),
                ),
                (
                    "待补差距",
                    _clip_items(tuple(item.gap for item in preparation.gaps)),
                ),
            )
            heading = "面试准备材料已生成。"
        elif result.state == "job_research_ready":
            research = MainAgentRuntime._job_research_draft(result)
            if research is None:
                return screen
            sections = (
                ("总结", (_clip(research.summary, _HISTORY_SUMMARY_CHARS),)),
                (
                    "关键结论",
                    _clip_items(
                        tuple(item.statement for item in research.findings)
                    ),
                ),
                ("待确认", _clip_items(research.open_questions)),
            )
            heading = "岗位研究已完成。"
        else:
            return screen
        blocks = [
            f"{title}\n" + "\n".join(f"- {line}" for line in lines)
            for title, lines in sections
            if lines
        ]
        return "\n\n".join((heading, *blocks))

    @staticmethod
    def _assistant_message(result: MainAgentToolOutput) -> str:
        if isinstance(result, ToolObservation):
            if result.state == "job_research_ready":
                research = MainAgentRuntime._job_research_draft(result)
                if research is not None:
                    return render_job_research(
                        research,
                        status=str(result.payload.get("status") or "current"),
                        user_provided_context=(
                            str(result.payload["user_provided_context"])
                            if result.payload.get("user_provided_context") is not None
                            else None
                        ),
                    )
            if result.state == "interview_preparation_ready":
                preparation = MainAgentRuntime._interview_preparation_result(result)
                if preparation is not None:
                    return render_interview_preparation(preparation)
            if result.state == "calendar_approval_required":
                payload = result.payload.get("payload")
                if isinstance(payload, dict):
                    return (
                        "请确认是否执行以下 Calendar 变更：\n"
                        f"- 操作：{result.payload.get('operation')}\n"
                        f"- 标题：{payload.get('title')}\n"
                        f"- 开始：{payload.get('start_at')}\n"
                        f"- 结束：{payload.get('end_at')}\n"
                        f"- 时区：{payload.get('timezone')}\n"
                        f"- 地点：{payload.get('location') or '未提供'}\n"
                        f"- 预览失效时间：{result.payload.get('expires_at')}\n"
                        "只有你明确确认后才会写入外部 Calendar。"
                    )
                return (
                    "请确认是否取消这条 Calendar 事件。"
                    f"预览失效时间：{result.payload.get('expires_at')}。"
                )
            return result.message
        analyses = result.analysis_items or ((result.analysis,) if result.analysis else ())
        if result.state not in {"analysis_ready", "partial_analysis_ready"} or not analyses:
            return result.message

        def section(title: str, items: tuple[str, ...]) -> str:
            content = "\n".join(f"- {item}" for item in items) if items else "- 暂无明确说明"
            return f"{title}\n{content}"

        blocks = []
        analysis_indices = result.analysis_selection_indices or tuple(range(1, len(analyses) + 1))
        for selection_index, analysis in zip(analysis_indices, analyses):
            item = result.items[selection_index - 1] if selection_index <= len(result.items) else None
            heading = f"岗位 {selection_index}\n" + (f"{item.title} — {item.company_name}\n" if item else "") if len(analyses) > 1 else ""
            sections = (
                f"岗位摘要\n{analysis.job_summary}",
                section("工作职责", analysis.responsibilities),
                section("必备技能", analysis.required_skills),
                section("加分项", analysis.preferred_qualifications),
                section("待确认问题", analysis.clarification_questions),
            )
            blocks.append((heading.strip() + "\n\n" if heading else "") + "\n\n".join(sections))
        return "\n\n".join(blocks)

    @staticmethod
    def _interview_preparation_result(
        result: ToolObservation,
    ) -> InterviewPreparationResult | None:
        raw = result.payload.get("preparation")
        if not isinstance(raw, dict):
            return None
        try:
            return InterviewPreparationResult.model_validate(raw)
        except ValueError:
            return None

    @staticmethod
    def _job_research_draft(result: ToolObservation) -> JobResearchDraft | None:
        raw = result.payload.get("research")
        raw_sources = result.payload.get("sources")
        if not isinstance(raw, dict) or not isinstance(raw_sources, list):
            return None
        try:
            sources = tuple(
                JobResearchSourceDraft(
                    source_key=item["source_key"],
                    url=item["url"],
                    title=item["title"],
                    publisher=item.get("publisher"),
                    published_at=item.get("published_at"),
                    relevant_excerpt=item["relevant_excerpt"],
                )
                for item in raw_sources
                if isinstance(item, dict)
            )
            findings = tuple(
                JobResearchFindingDraft.model_validate(item)
                for item in raw.get("findings", ())
            )
            return JobResearchDraft(
                summary=raw["summary"],
                sources=sources,
                findings=findings,
                open_questions=tuple(raw.get("open_questions", ())),
                limitations=tuple(raw.get("limitations", ())),
            )
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _project_atomic_tool_arguments(context: MainAgentContext, name: str, arguments: dict[str, object]) -> dict[str, object]:
        if name in {"find_saved_jobs", "get_saved_job"}:
            return project_saved_job_arguments(context, name, arguments)
        if name == "get_job_research":
            return project_job_research_arguments(context, name, arguments)
        if name in {"list_email_events", "resolve_email_event"}:
            return project_email_arguments(context, name, arguments)
        if name in {
            "list_interviews",
            "get_interview",
            "create_interview",
            "update_interview",
            "complete_interview",
            "record_interview_retro",
        }:
            return project_interview_arguments(context, name, arguments)
        if name in {"prepare_interview", "get_interview_preparation"}:
            return project_interview_preparation_arguments(context, name, arguments)
        if name == "get_mock_interview_result":
            return project_mock_interview_result_arguments(context, arguments)
        if name in {
            "get_daily_brief",
            "list_action_items",
            "complete_action_item",
            "dismiss_action_item",
            "snooze_action_item",
        }:
            return project_action_center_arguments(context, name, arguments)
        if name in {
            "list_calendar_accounts",
            "list_calendar_links",
            "prepare_interview_calendar_sync",
            "get_calendar_proposal",
            "execute_calendar_proposal",
        }:
            return project_calendar_arguments(context, name, arguments)
        if name in {
            "list_target_roles",
            "list_resumes",
            "get_resume_metadata",
            "analyze_resume",
            "get_resume_analysis",
            "confirm_resume_analysis",
            "match_resume_to_job",
            "get_resume_job_match",
            "draft_resume_tailoring",
            "get_resume_tailoring_draft",
            "review_resume_tailoring",
            "revise_resume_tailoring",
            "finalize_resume_tailoring",
            "export_resume_artifact",
            "create_application",
            "update_application_status",
            "list_applications",
            "get_application",
        }:
            return project_resume_arguments(context, name, arguments)
        return arguments

    @staticmethod
    def _update_task(context: MainAgentContext, result: JobDiscoveryGatewayResult) -> MainAgentContext:
        task = context.task
        # A run-less result never claims the slot: the gateway returns one when it
        # refuses to start, and overwriting a live run with it would strand it.
        if not result.run_id:
            return context.model_copy(update={"task": task})
        if result.state == "selection_required":
            candidates = tuple(CandidateContextItem(result_ref=item.result_ref, title=item.title, company_name=item.company_name, city=item.city, salary=item.salary) for item in result.items)
            task = task.enter_workflow("job_discovery", run_id=result.run_id, phase=result.state, candidates=candidates)
        elif result.state == "analysis_ready":
            task = task.enter_workflow("job_discovery", run_id=result.run_id, phase=result.state, selected_result_ref=result.selected_result_ref)
        elif result.state == "detail_unavailable":
            task = task.enter_workflow("job_discovery", run_id=result.run_id, phase=result.state, selected_result_ref=result.selected_result_ref, manual_search_query=result.manual_search_query)
        elif result.state in {"failed", "waiting_user"}:
            task = task.enter_workflow("job_discovery", run_id=result.run_id, phase=result.state)
        return context.model_copy(update={"task": task})

    @staticmethod
    def _update_mock_interview_task(
        context: MainAgentContext, result: ToolObservation
    ) -> MainAgentContext:
        task = context.task
        session_id = result.payload.get("session_id")
        if result.state in {
            "mock_interview_answer_required",
            "mock_interview_running",
        }:
            if not isinstance(session_id, str) or not session_id:
                raise ValueError("Mock interview result has no session_id")
            task = task.enter_workflow(
                "mock_interview",
                run_id=session_id,
                phase=result.state,
                candidates=(),
            )
        elif result.state in {
            "mock_interview_completed",
            "mock_interview_cancelled",
            "mock_interview_restart_failed",
            "no_mock_interview_to_restart",
        }:
            if task.active_workflow == "mock_interview":
                task = task.leave_workflow()
        elif result.state == "failed" and task.active_workflow == "mock_interview":
            # A persisted answer can be retried. Keep ownership instead of
            # stranding the graph after a transient Worker failure.
            task = task.enter_workflow(
                "mock_interview",
                run_id=task.run_id or str(session_id),
                phase="failed",
                candidates=(),
            )
        elif result.state in {
            "mock_interview_checkpoint_missing",
            "mock_interview_graph_incompatible",
        } and task.active_workflow == "mock_interview":
            task = task.enter_workflow(
                "mock_interview",
                run_id=task.run_id or str(session_id),
                phase=result.state,
                candidates=(),
            )
        return context.model_copy(update={"task": task})

    @staticmethod
    def _update_atomic_task(
        context: MainAgentContext, result: ToolObservation
    ) -> MainAgentContext:
        return context.model_copy(
            update={"task": reduce_task_state(context.task, result)}
        )
