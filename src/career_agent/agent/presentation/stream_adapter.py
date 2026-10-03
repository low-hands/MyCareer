from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from career_agent.agent.main_agent_contracts import (
    ConversationResourceReference,
    ToolObservation,
)
from career_agent.agent.presentation.interaction_renderer import InteractionRenderer
from career_agent.agent.presentation.presenter import TurnPresenter
from career_agent.agent.turn_models import MainAgentTurnResult
from career_agent.harness.streaming import (
    ArtifactReadyEvent,
    ClientActionEvent,
    ContentDeltaEvent,
    JobResourceReadyEvent,
    ProgressEvent,
    PublicStreamEvent,
    ReportReadyEvent,
    TurnCompletedEvent,
    TurnSuspendedEvent,
    iter_content_deltas,
)


class StreamAdapter:
    """Translate a completed turn into the public streaming protocol."""

    def __init__(
        self,
        *,
        interaction_renderer: InteractionRenderer,
        presenter: TurnPresenter,
        emit: Callable[[PublicStreamEvent], None],
    ) -> None:
        self._interaction_renderer = interaction_renderer
        self._presenter = presenter
        self._emit = emit

    def deliver_events(
        self,
        *,
        result: MainAgentTurnResult,
        turn_id: str,
        conversation_id: str,
    ) -> None:
        self._emit_client_actions(result)

        interaction = self._interaction_renderer.event(
            result=result,
            conversation_id=conversation_id,
        )
        if interaction is not None:
            self._emit(interaction)
            self._emit(
                TurnSuspendedEvent(
                    turn_id=turn_id,
                    interaction_id=interaction.interaction_id,
                )
            )
            return

        for artifact in result.artifacts:
            reference = artifact.reference
            self._emit(
                ArtifactReadyEvent(
                    artifact_id=reference.id,
                    filename=reference.filename,
                    media_type=reference.media_type,
                    byte_size=reference.byte_size,
                )
            )
        for reference in self._presenter.turn_resource_refs(result.tool_results):
            self._emit(self.resource_ready_event(reference))
        self._emit(TurnCompletedEvent(turn_id=turn_id))

    def deliver_reply(
        self,
        *,
        result: MainAgentTurnResult,
        conversation_id: str,
    ) -> None:
        """Emit reply text before the coordinator commits the completed turn."""

        interaction = self._interaction_renderer.event(
            result=result,
            conversation_id=conversation_id,
        )
        if interaction is not None or result.content_streamed:
            return

        self._emit(ProgressEvent(stage="presenting", message="正在整理交付内容……"))
        streamed_message = (
            self._presenter.conversation_content(
                result.tool_result,
                screen=self._presenter.durable_screen(result),
                composed=bool(result.model_message),
            )
            if self._presenter.turn_is_card_backed(result.tool_results)
            else result.assistant_message
        )
        for delta in iter_content_deltas(streamed_message):
            self._emit(ContentDeltaEvent(delta=delta, delivery="synthetic"))

    def _emit_client_actions(self, result: MainAgentTurnResult) -> None:
        for tool_result in result.tool_results:
            if not isinstance(tool_result, ToolObservation):
                continue
            action = tool_result.payload.get("client_action")
            if not isinstance(action, dict) or action.get("type") != "open_url":
                continue
            intent_id = action.get("capture_intent_id")
            expires_at = action.get("capture_intent_expires_at")
            self._emit(
                ClientActionEvent(
                    action="open_url",
                    url=str(action.get("url", "")),
                    label=str(action.get("label", "打开岗位搜索页")),
                    capture_intent_id=str(intent_id) if intent_id else None,
                    capture_intent_expires_at=(
                        datetime.fromisoformat(str(expires_at)) if expires_at else None
                    ),
                )
            )

    @staticmethod
    def resource_ready_event(
        reference: ConversationResourceReference,
    ) -> ReportReadyEvent | JobResourceReadyEvent:
        if reference.kind == "saved_job":
            if reference.job_posting_id is None:
                raise ValueError("saved_job references name their posting")
            return JobResourceReadyEvent(
                resource_id=reference.resource_id,
                job_posting_id=reference.job_posting_id,
                title=reference.title,
                description=reference.description,
            )
        if reference.kind == "resume_version":
            raise ValueError("resume_version references ride on user messages")
        return ReportReadyEvent(
            kind=reference.kind,
            resource_id=reference.resource_id,
            status_at_delivery=reference.status_at_delivery,
            anchored_by_other_job=reference.anchored_by_other_job,
        )
