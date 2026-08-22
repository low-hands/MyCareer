from __future__ import annotations

from typing import Any

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.job_discovery_gateway import JobDiscoveryGatewayResult
from career_agent.agent.main_agent_contracts import AgentDecision, CandidateContextItem, ConversationTaskState, DecisionMaker, MainAgentContext, project_job_discovery_arguments
from career_agent.agent.main_agent_tools import MainAgentToolRegistry


class MainAgentTurnResult:
    def __init__(self, *, decision: AgentDecision, context: MainAgentContext, assistant_message: str, tool_result: JobDiscoveryGatewayResult | None = None) -> None:
        self.decision = decision
        self.context = context
        self.assistant_message = assistant_message
        self.tool_result = tool_result


class MainAgentRuntime:
    def __init__(self, *, context_manager: ContextManager, decision_maker: DecisionMaker, tools: MainAgentToolRegistry) -> None:
        self._context_manager = context_manager
        self._decision_maker = decision_maker
        self._tools = tools

    def run_turn(self, *, user_id: str, conversation_id: str, user_message: str) -> MainAgentTurnResult:
        context = self._context_manager.load_for_turn(user_id=user_id, conversation_id=conversation_id, user_message=user_message)
        result = self._run_loaded_context(context)
        self._context_manager.commit_turn(context=context, task=result.context.task, assistant_message=result.assistant_message)
        return result

    def _run_loaded_context(self, context: MainAgentContext) -> MainAgentTurnResult:
        decision = self._decision_maker.decide(context, self._tools.schemas())
        if decision.action != "tool_call":
            return MainAgentTurnResult(decision=decision, context=context, assistant_message=decision.message or "")
        if decision.tool_call is None:
            raise ValueError("tool_call action requires tool_call arguments")
        arguments = self._project_arguments(context, decision.tool_call.name, decision.tool_call.arguments)
        result = self._tools.invoke(decision.tool_call.name, arguments)
        updated = self._update_task(context, result)
        return MainAgentTurnResult(decision=decision, context=updated, tool_result=result, assistant_message=self._assistant_message(result))

    @staticmethod
    def _assistant_message(result: JobDiscoveryGatewayResult) -> str:
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
