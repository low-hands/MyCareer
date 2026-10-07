from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from career_agent.agent.contracts.interactions import CONFIRMATION_SPECS
from career_agent.agent.contracts.observations import ToolObservation
from career_agent.agent.contracts.task_state import ConversationTaskState
from career_agent.agent.capabilities.registry import MainAgentToolOutput
from career_agent.agent.runtime.state import MainAgentState
from career_agent.agent.contracts.questionnaire import PendingQuestionnaire
from career_agent.agent.contracts.turn import MainAgentTurnResult
from career_agent.harness.streaming import (
    InteractionOption,
    InteractionRequiredEvent,
    capability_confirmation_event,
    interaction_id,
    questionnaire_event,
)


class InteractionRenderer:
    """Render graph interruptions and completed turns as UI interactions."""

    RENDERER_STATES = frozenset(
        {
            "calendar_approval_required",
            "capability_confirmation_required",
            "email_events_pending",
            "constraint_retirement_proposed",
            "memory_amendment_proposed",
            "memory_tombstone_proposed",
            "free_text_preference_confirmation_proposed",
            "free_text_preference_confirmed_structured_proposed",
            "career_fact_proposed",
            "mock_interview_answer_required",
            "mock_interview_running",
            "mock_interview_resume_choice_required",
            "mock_interview_job_choice_required",
            "resume_final_review_blocked",
            "resume_tailoring_review_blocked",
            "resume_tailoring_superseded",
        }
    )

    def __init__(
        self,
        *,
        active_turn_id: Callable[[], str | None],
        assistant_message: Callable[[MainAgentToolOutput], str],
        renderer_states: frozenset[str] | None = None,
        has_interaction_renderer: Callable[[str], bool] | None = None,
    ) -> None:
        self._active_turn_id_callback = active_turn_id
        self._assistant_message_callback = assistant_message
        self._renderer_states = (
            self.RENDERER_STATES if renderer_states is None else renderer_states
        )
        self._has_renderer_callback = has_interaction_renderer

    def event(
        self,
        *,
        result: MainAgentTurnResult,
        conversation_id: str,
    ) -> InteractionRequiredEvent | None:
        tool_result = result.tool_result
        prompt = result.assistant_message
        task = result.context.task
        stable_parts = (
            conversation_id,
            task.active_workflow,
            task.run_id or "",
            task.phase or "",
            tool_result.state if tool_result is not None else result.origin.label,
        )

        decision = result.model_decision
        questionnaire = task.pending_questionnaire
        if (
            decision is not None
            and decision.action == "questionnaire"
            and questionnaire is not None
            and not any(
                item.disposition == "failed" for item in result.tool_results
            )
            and (
                tool_result is None
                or tool_result.state
                not in {
                    "capability_confirmation_required",
                    "calendar_approval_required",
                    "email_events_pending",
                }
            )
        ):
            return questionnaire_event(questionnaire)

        if tool_result is not None:
            event = self._tool_interaction(
                tool_result=tool_result,
                prompt=prompt,
                task=task,
                stable_parts=stable_parts,
                conversation_id=conversation_id,
            )
            if event is not None:
                return event

        if (
            decision is not None
            and decision.action == "ask_user"
            and not any(
                item.disposition == "failed" for item in result.tool_results
            )
        ):
            options = (
                self._selection_options(tool_result, task)
                if decision.selection_source == "latest_tool_result"
                else ()
            )
            if options:
                return InteractionRequiredEvent(
                    interaction_id=interaction_id(*stable_parts, prompt),
                    kind="single_selection",
                    prompt=prompt,
                    options=options,
                    allow_free_text=True,
                )
            return InteractionRequiredEvent(
                interaction_id=interaction_id(*stable_parts, prompt),
                kind="free_text",
                prompt=prompt,
                allow_free_text=True,
            )
        return None

    def interrupt(self, state: MainAgentState) -> MainAgentState:
        decision = state["decision"]
        if decision.action == "ask_user":
            return {
                "assistant_message": decision.message
                or "请补充下一步所需的信息。"
            }
        if decision.action == "questionnaire":
            context = state["context"]
            task = context.task
            offered = state.get("control", {}).get("offered_tool_names", ())
            # The model proposes a continuation. Bind it only when the same
            # decision actually saw that capability's full schema. A name from
            # the directory alone cannot promote an undisclosed write tool.
            continuation = (
                decision.continuation_capability
                if decision.continuation_capability in offered else None
            )
            now = datetime.now(timezone.utc)
            questionnaire = PendingQuestionnaire(
                interaction_id=interaction_id(
                    context.conversation_id,
                    "questionnaire",
                    self._active_turn_id() or uuid4().hex,
                ),
                prompt=decision.message or "请补充以下信息。",
                questions=decision.questions,
                created_at=now,
                expires_at=now + timedelta(days=7),
                active_workflow=task.active_workflow,
                continuation_capability=continuation,
                resume_version_id=task.active_resume_version_id,
                resume_job_match_id=task.active_resume_job_match_id,
                job_posting_id=task.active_job_posting_id,
                jd_snapshot_id=task.active_jd_snapshot_id,
            )
            return {
                "assistant_message": decision.message or "请补充以下信息。",
                "context": context.model_copy(
                    update={
                        "task": task.with_pending_questionnaire(questionnaire)
                    }
                ),
            }
        result = self._last_result(state)
        if result is None:
            raise ValueError("capability interaction requires an observed result")
        if not self._has_interaction_renderer(result.state):
            raise ValueError(
                "interaction_required result has no interaction renderer: "
                f"{result.tool_name}/{result.state}"
            )
        return {"assistant_message": self._assistant_message(result)}

    def _tool_interaction(
        self,
        *,
        tool_result: MainAgentToolOutput,
        prompt: str,
        task: ConversationTaskState,
        stable_parts: tuple[str, ...],
        conversation_id: str,
    ) -> InteractionRequiredEvent | None:
        if tool_result.state == "capability_confirmation_required":
            return capability_confirmation_event(
                conversation_id=conversation_id,
                confirmation_id=tool_result.payload["confirmation_id"],
                prompt=prompt,
            )
        if tool_result.payload.get("confirmation_id") and any(
            spec.requires_seal and spec.proposed_state == tool_result.state
            for spec in CONFIRMATION_SPECS.values()
        ):
            return capability_confirmation_event(
                conversation_id=conversation_id,
                confirmation_id=tool_result.payload["confirmation_id"],
                prompt=(
                    f"{tool_result.payload['confirmation_summary']}\n"
                    "这是删除或停用操作，请亲自确认是否执行。"
                ),
            )
        if tool_result.state == "calendar_approval_required":
            return InteractionRequiredEvent(
                interaction_id=interaction_id(*stable_parts),
                kind="approval",
                prompt=prompt,
                options=(
                    InteractionOption(value="confirm", label="确认执行"),
                    InteractionOption(value="cancel", label="暂不执行"),
                ),
            )
        if tool_result.state == "mock_interview_resume_choice_required":
            return self._mock_interview_resume_choice_event(
                interaction_id(*stable_parts, prompt), prompt, task
            )
        if tool_result.state == "mock_interview_job_choice_required":
            company = str(tool_result.payload.get("company_name") or "这家公司")
            return InteractionRequiredEvent(
                interaction_id=interaction_id(*stable_parts, prompt),
                kind="single_selection",
                prompt=prompt,
                options=(
                    *(
                        InteractionOption(
                            selection_index=index,
                            label=f"{item.company_name} · {item.title}",
                            description="，".join(
                                value for value in (item.city, item.salary) if value
                            )
                            or None,
                        )
                        for index, item in enumerate(
                            task.saved_job_candidates, start=1
                        )
                    ),
                    InteractionOption(
                        value="without_job",
                        label=f"不针对具体岗位，只按「{company}」",
                        description="没有 JD，按目标岗位出题；常见大厂会参考其面试风格",
                    ),
                ),
                allow_free_text=True,
            )
        if tool_result.state == "email_events_pending":
            return InteractionRequiredEvent(
                interaction_id=interaction_id(*stable_parts),
                kind="confirmation",
                prompt=prompt,
                options=(
                    InteractionOption(value="review", label="查看并确认"),
                    InteractionOption(value="later", label="稍后处理"),
                ),
                allow_free_text=True,
            )
        if tool_result.state in {
            "constraint_retirement_proposed",
            "memory_amendment_proposed",
            "memory_tombstone_proposed",
            "free_text_preference_confirmation_proposed",
            "free_text_preference_confirmed_structured_proposed",
            "career_fact_proposed",
            "mock_interview_answer_required",
            "mock_interview_running",
            "resume_tailoring_review_blocked",
            "resume_final_review_blocked",
            "resume_tailoring_superseded",
        }:
            return InteractionRequiredEvent(
                interaction_id=interaction_id(*stable_parts),
                kind="free_text",
                prompt=prompt,
                allow_free_text=True,
            )
        return None

    def _active_turn_id(self) -> str | None:
        return self._active_turn_id_callback()

    @staticmethod
    def _last_result(state: MainAgentState) -> MainAgentToolOutput | None:
        results = state.get("tool_results", ())
        return results[-1] if results else None

    def _has_interaction_renderer(self, state: str) -> bool:
        if self._has_renderer_callback is not None:
            return self._has_renderer_callback(state)
        return state in self._renderer_states

    def _assistant_message(self, result: MainAgentToolOutput) -> str:
        return self._assistant_message_callback(result)

    @staticmethod
    def _mock_interview_resume_choice_event(
        event_id: str,
        prompt: str,
        task: ConversationTaskState,
    ) -> InteractionRequiredEvent:
        """Render every live resume, no-resume, and upload as one choice."""

        return InteractionRequiredEvent(
            interaction_id=event_id,
            kind="single_selection",
            prompt=prompt,
            options=(
                *(
                    InteractionOption(
                        selection_index=index,
                        label=f"《{item.resume_name or '简历'}》v{item.version_number}",
                        description=item.document_format.upper(),
                    )
                    for index, item in enumerate(
                        task.resume_version_candidates, start=1
                    )
                ),
                InteractionOption(
                    value="without_resume",
                    label="不用简历",
                    description="只问通用题和专业基础",
                ),
            ),
            allow_free_text=True,
            accepts_upload="resume",
        )

    @staticmethod
    def _selection_options(
        tool_result: MainAgentToolOutput | None,
        task: ConversationTaskState,
    ) -> tuple[InteractionOption, ...]:
        if not isinstance(tool_result, ToolObservation):
            return ()
        name = tool_result.tool_name
        if name == "find_saved_jobs":
            return tuple(
                InteractionOption(
                    selection_index=index,
                    label=f"{item.title}｜{item.company_name}",
                    description="，".join(
                        value for value in (item.city, item.salary) if value
                    )
                    or None,
                )
                for index, item in enumerate(task.saved_job_candidates, start=1)
            )
        if name == "list_target_roles":
            return tuple(
                InteractionOption(
                    selection_index=index,
                    label=item.title,
                    description=f"优先级 {item.priority}，状态 {item.status}",
                )
                for index, item in enumerate(task.target_role_candidates, start=1)
            )
        if name == "list_resumes":
            return tuple(
                InteractionOption(
                    selection_index=index,
                    label=item.name,
                    description=f"状态：{item.status}",
                )
                for index, item in enumerate(task.resume_candidates, start=1)
            )
        if name == "get_resume_metadata":
            return tuple(
                InteractionOption(
                    selection_index=index,
                    label=f"版本 {item.version_number}",
                    description=(
                        f"{item.document_format}，{item.source_type}，"
                        f"{item.byte_size} bytes"
                    ),
                )
                for index, item in enumerate(
                    task.resume_version_candidates, start=1
                )
            )
        if name == "list_applications":
            return tuple(
                InteractionOption(
                    selection_index=index,
                    label=f"{item.title}｜{item.company_name}",
                    description=f"状态：{item.status}",
                )
                for index, item in enumerate(task.application_candidates, start=1)
            )
        if name == "list_interviews":
            return tuple(
                InteractionOption(
                    selection_index=index,
                    label=(
                        f"{item.employer_label or '面试'}｜第 {item.sequence_number} 轮"
                    ),
                    description="，".join(
                        value
                        for value in (
                            f"状态 {item.status}",
                            (
                                item.scheduled_start.isoformat()
                                if item.scheduled_start is not None
                                else None
                            ),
                        )
                        if value
                    ),
                )
                for index, item in enumerate(task.interview_candidates, start=1)
            )
        if name == "list_action_items":
            return tuple(
                InteractionOption(
                    selection_index=index,
                    label=item.title,
                    description=f"{item.action_type}，状态 {item.status}",
                )
                for index, item in enumerate(task.action_candidates, start=1)
            )
        if name == "list_calendar_accounts":
            return tuple(
                InteractionOption(
                    selection_index=index,
                    label=item.email_address,
                    description=f"{item.provider} Calendar",
                )
                for index, item in enumerate(
                    task.calendar_account_candidates, start=1
                )
            )
        if name == "list_email_events":
            return tuple(
                InteractionOption(
                    selection_index=index,
                    label=item.summary,
                    description=f"{item.event_type}，状态 {item.status}",
                )
                for index, item in enumerate(task.email_event_candidates, start=1)
            )
        return ()
