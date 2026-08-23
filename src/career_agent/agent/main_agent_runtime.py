from __future__ import annotations

import json
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.job_discovery_gateway import JobDiscoveryGatewayResult
from career_agent.agent.main_agent_contracts import AgentDecision, CandidateContextItem, DecisionMaker, MainAgentContext, ToolObservation, project_job_discovery_arguments, project_saved_job_arguments
from career_agent.agent.main_agent_tools import MainAgentToolOutput, MainAgentToolRegistry


class MainAgentState(TypedDict, total=False):
    context: MainAgentContext
    decision: AgentDecision
    last_tool_result: MainAgentToolOutput
    tool_call_fingerprints: tuple[str, ...]
    tool_call_count: int
    assistant_message: str


class MainAgentTurnResult:
    def __init__(self, *, decision: AgentDecision, context: MainAgentContext, assistant_message: str, tool_result: MainAgentToolOutput | None = None) -> None:
        self.decision = decision
        self.context = context
        self.assistant_message = assistant_message
        self.tool_result = tool_result


class MainAgentRuntime:
    _WAITING_STATES = frozenset({"selection_required", "waiting_user", "detail_unavailable", "failed"})

    def __init__(self, *, context_manager: ContextManager, decision_maker: DecisionMaker, tools: MainAgentToolRegistry, max_tool_calls: int = 3) -> None:
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least one")
        self._context_manager = context_manager
        self._decision_maker = decision_maker
        self._tools = tools
        self._max_tool_calls = max_tool_calls

        graph = StateGraph(MainAgentState)
        graph.add_node("decide", self._decide)
        graph.add_node("invoke_tool", self._invoke_tool)
        graph.add_node("finish", self._finish)
        graph.add_node("fallback", self._fallback)
        graph.add_edge(START, "decide")
        graph.add_conditional_edges(
            "decide",
            self._after_decision,
            {"invoke_tool": "invoke_tool", "finish": "finish", "fallback": "fallback"},
        )
        graph.add_edge("invoke_tool", "decide")
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
                "tool_call_count": 0,
            }
        )
        return MainAgentTurnResult(
            decision=state["decision"],
            context=state["context"],
            assistant_message=state["assistant_message"],
            tool_result=state.get("last_tool_result"),
        )

    def _decide(self, state: MainAgentState) -> MainAgentState:
        return {"decision": self._decision_maker.decide(state["context"], self._tools.schemas())}

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

    def _after_decision(self, state: MainAgentState) -> Literal["invoke_tool", "finish", "fallback"]:
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
        return "invoke_tool"

    def _invoke_tool(self, state: MainAgentState) -> MainAgentState:
        context = state["context"]
        decision = state["decision"]
        if decision.tool_call is None:
            raise ValueError("tool_call action requires tool_call arguments")
        arguments = self._project_arguments(context, decision.tool_call.name, decision.tool_call.arguments)
        result = self._tools.invoke(decision.tool_call.name, arguments)
        updated = self._update_task(context, result) if isinstance(result, JobDiscoveryGatewayResult) else context
        observation = self._tool_observation(decision.tool_call.name, result)
        updated = updated.model_copy(update={"tool_observations": (*updated.tool_observations, observation)[-3:]})
        fingerprint = self._tool_call_fingerprint(decision)
        return {
            "context": updated,
            "last_tool_result": result,
            "tool_call_fingerprints": (*state.get("tool_call_fingerprints", ()), fingerprint),
            "tool_call_count": state.get("tool_call_count", 0) + 1,
        }

    @staticmethod
    def _finish(state: MainAgentState) -> MainAgentState:
        decision = state["decision"]
        message = decision.message or ""
        if not message and state.get("last_tool_result") is not None:
            message = MainAgentRuntime._assistant_message(state["last_tool_result"])
        return {"assistant_message": message}

    @staticmethod
    def _fallback(state: MainAgentState) -> MainAgentState:
        result = state.get("last_tool_result")
        if result is not None:
            return {"assistant_message": MainAgentRuntime._assistant_message(result)}
        return {"assistant_message": "本轮可执行步骤已达到上限，请确认后继续。"}

    @staticmethod
    def _tool_observation(name: str, result: MainAgentToolOutput) -> ToolObservation:
        if isinstance(result, ToolObservation):
            return result
        payload: dict[str, Any] = {
            "items": [
                {
                    "selection_index": index,
                    "title": item.title,
                    "company_name": item.company_name,
                    "city": item.city,
                    "salary": item.salary,
                    "rationale": item.rationale,
                    "cautions": item.cautions,
                }
                for index, item in enumerate(result.items, start=1)
            ],
            "error_code": result.error_code,
            "error_stage": result.error_stage,
            "error_detail": result.error_detail,
            "recovery_action": result.recovery_action,
            "manual_search_query": result.manual_search_query,
        }
        analyses = result.analysis_items or ((result.analysis,) if result.analysis else ())
        if analyses:
            analysis_indices = result.analysis_selection_indices or tuple(range(1, len(analyses) + 1))
            payload["analyses"] = [
                {
                    "selection_index": selection_index,
                    "job_summary": analysis.job_summary,
                    "responsibilities": analysis.responsibilities,
                    "required_skills": analysis.required_skills,
                    "preferred_qualifications": analysis.preferred_qualifications,
                    "clarification_questions": analysis.clarification_questions,
                }
                for selection_index, analysis in zip(analysis_indices, analyses)
            ]
        if result.comparison is not None:
            payload["comparison"] = result.comparison.model_dump(mode="json")
        return ToolObservation(tool_name=name, state=result.state, message=result.message, next_action=result.next_action, payload=payload)

    @staticmethod
    def _assistant_message(result: MainAgentToolOutput) -> str:
        if isinstance(result, ToolObservation):
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
    def _project_arguments(context: MainAgentContext, name: str, arguments: dict[str, object]) -> dict[str, object]:
        if name == "job_discovery":
            return project_job_discovery_arguments(context, arguments)
        if name in {"find_saved_jobs", "get_saved_job"}:
            return project_saved_job_arguments(context, name, arguments)
        return arguments

    @staticmethod
    def _update_task(context: MainAgentContext, result: JobDiscoveryGatewayResult) -> MainAgentContext:
        task = context.task
        if result.state == "selection_required":
            candidates = tuple(CandidateContextItem(result_ref=item.result_ref, title=item.title, company_name=item.company_name, city=item.city, salary=item.salary) for item in result.items)
            task = task.model_copy(update={"active_workflow": "job_discovery", "run_id": result.run_id, "phase": result.state, "selected_result_ref": None, "candidates": candidates})
        elif result.state == "analysis_ready":
            task = task.model_copy(update={"active_workflow": "job_discovery", "run_id": result.run_id, "phase": result.state, "selected_result_ref": result.selected_result_ref})
        elif result.state == "detail_unavailable":
            task = task.model_copy(update={"active_workflow": "job_discovery", "run_id": result.run_id, "phase": result.state, "selected_result_ref": result.selected_result_ref, "manual_search_query": result.manual_search_query})
        elif result.state in {"failed", "waiting_user"}:
            task = task.model_copy(update={"active_workflow": "job_discovery", "run_id": result.run_id, "phase": result.state})
        return context.model_copy(update={"task": task})
