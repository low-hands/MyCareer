from __future__ import annotations

from collections.abc import Callable

from career_agent.agent.interaction_coordinator import InteractionCoordinator
from career_agent.agent.main_agent_contracts import (
    ConversationTaskState,
    ToolObservation,
)
from career_agent.agent.main_agent_tools import MainAgentToolOutput
from career_agent.agent.main_state import MainAgentState
from career_agent.agent.turn_models import MainAgentTurnResult
from career_agent.harness.streaming import (
    InteractionOption,
    InteractionRequiredEvent,
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
        return InteractionCoordinator.event(
            result=result,
            conversation_id=conversation_id,
            host=self,
        )

    def interrupt(self, state: MainAgentState) -> MainAgentState:
        return InteractionCoordinator.interrupt(state, host=self)

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
