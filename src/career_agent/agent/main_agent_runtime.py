from __future__ import annotations

import json
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.career_context import CareerContextProjector
from career_agent.agent.job_discovery_gateway import JobDiscoveryGatewayResult
from career_agent.agent.main_agent_contracts import AgentDecision, CandidateContextItem, DecisionMaker, DecisionObservation, MainAgentContext, ToolObservation, project_action_center_arguments, project_calendar_arguments, project_email_arguments, project_interview_arguments, project_interview_preparation_arguments, project_job_discovery_arguments, project_resume_arguments, project_saved_job_arguments
from career_agent.agent.main_agent_reducers import reduce_task_state
from career_agent.agent.main_agent_tools import MainAgentToolOutput, MainAgentToolRegistry
from career_agent.domain.resume import ResumeArtifactDelivery


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

    def __init__(self, *, context_manager: ContextManager, decision_maker: DecisionMaker, tools: MainAgentToolRegistry, career_context_projector: CareerContextProjector | None = None, max_tool_calls: int = 3) -> None:
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least one")
        self._context_manager = context_manager
        self._decision_maker = decision_maker
        self._tools = tools
        self._career_context_projector = career_context_projector
        self._max_tool_calls = max_tool_calls

        graph = StateGraph(MainAgentState)
        graph.add_node("hydrate_career_context", self._hydrate_career_context)
        graph.add_node("decide", self._decide)
        graph.add_node("invoke_atomic_tool", self._invoke_atomic_tool)
        graph.add_node("run_workflow", self._run_workflow)
        graph.add_node("observe", self._observe)
        graph.add_node("finish", self._finish)
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
        graph.add_edge("observe", "decide")
        graph.add_edge("finish", END)
        graph.add_edge("fallback", END)
        self._graph = graph.compile()

    def run_turn(self, *, user_id: str, conversation_id: str, user_message: str) -> MainAgentTurnResult:
        context = self._context_manager.load_for_turn(user_id=user_id, conversation_id=conversation_id, user_message=user_message)
        result = self._run_loaded_context(context)
        self._context_manager.commit_turn(context=context, task=result.context.task, assistant_message=result.assistant_message)
        return result

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
    def _assistant_message(result: MainAgentToolOutput) -> str:
        if isinstance(result, ToolObservation):
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
    def _project_atomic_tool_arguments(context: MainAgentContext, name: str, arguments: dict[str, object]) -> dict[str, object]:
        if name in {"find_saved_jobs", "get_saved_job"}:
            return project_saved_job_arguments(context, name, arguments)
        if name in {"list_email_events", "resolve_email_event"}:
            return project_email_arguments(context, name, arguments)
        if name in {
            "list_interviews",
            "get_interview",
            "create_interview",
            "update_interview",
            "complete_interview",
        }:
            return project_interview_arguments(context, name, arguments)
        if name in {"prepare_interview", "get_interview_preparation"}:
            return project_interview_preparation_arguments(context, name, arguments)
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
    def _update_atomic_task(
        context: MainAgentContext, result: ToolObservation
    ) -> MainAgentContext:
        return context.model_copy(
            update={"task": reduce_task_state(context.task, result)}
        )
