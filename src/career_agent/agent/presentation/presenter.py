from __future__ import annotations

from collections.abc import Callable

from career_agent.agent.presentation.body_contracts import (
    BodyDependency,
    SavedJobBodySource,
)
from career_agent.agent.presentation.delivery_policy import (
    condenses_message,
    delivers_body_elsewhere,
    policy_for,
)
from career_agent.agent.contracts.resources import ConversationResourceReference
from career_agent.agent.capabilities.registry import MainAgentToolOutput
from career_agent.agent.runtime.state import MainAgentState
from career_agent.agent.providers.openai_client import public_error_code
from career_agent.agent.contracts.turn import MainAgentTurnResult
from career_agent.storage.context import DeliveredBodyDraft


class TurnPresenter:
    """Compose one turn's durable and live delivery surfaces."""

    def __init__(
        self,
        *,
        render_result: Callable[[MainAgentToolOutput], str],
    ) -> None:
        self._render_result = render_result

    def _assistant_message(self, result: MainAgentToolOutput) -> str:
        return self._render_result(result)

    @staticmethod
    def _last_result(state: MainAgentState) -> MainAgentToolOutput | None:
        results = state.get("tool_results", ())
        return results[-1] if results else None

    def _screen_message(self, result: MainAgentToolOutput) -> str:
        """Presenter text for a turn the model did not close with prose."""

        if result.state == "saved_job_ready" and result.resource_ref is not None:
            return result.message
        if policy_for(result.state).body_delivery == "model":
            return result.message
        screen = self._assistant_message(result)
        if result.disposition != "failed":
            return screen
        code = public_error_code(result.payload.get("error_code"))
        if code == "CAPABILITY_FAILED" or code in screen:
            return screen
        return f"{screen}（错误码：{code}）"

    @staticmethod
    def turn_resource_refs(
        results: tuple[MainAgentToolOutput, ...],
        input_refs: tuple[ConversationResourceReference, ...] = (),
    ) -> tuple[ConversationResourceReference, ...]:
        """Return every durable resource once, preserving delivery order."""

        references: list[ConversationResourceReference] = []
        seen: set[str] = set()
        for result in results:
            reference = result.resource_ref
            if reference is None or reference.resource_id in seen:
                continue
            seen.add(reference.resource_id)
            references.append(reference)
        for reference in input_refs:
            if reference.kind == "saved_job" and reference.resource_id not in seen:
                seen.add(reference.resource_id)
                references.append(reference)
        return tuple(references)

    @staticmethod
    def has_backed_card(result: MainAgentToolOutput) -> bool:
        return (
            delivers_body_elsewhere(result.state)
            and result.resource_ref is not None
        )

    def turn_is_card_backed(
        self, results: tuple[MainAgentToolOutput, ...]
    ) -> bool:
        """Whether every delivery in the turn is retrievable through a card."""

        return bool(results) and all(self.has_backed_card(item) for item in results)

    def _turn_is_card_backed(
        self, results: tuple[MainAgentToolOutput, ...]
    ) -> bool:
        """Compatibility alias for the original runtime-facing renderer port."""

        return self.turn_is_card_backed(results)

    @staticmethod
    def durable_screen(result: MainAgentTurnResult) -> str:
        """Text shared by the live row and the durable transcript row."""

        return result.model_message or result.assistant_message

    def _undelivered_bodies(
        self, results: tuple[MainAgentToolOutput, ...]
    ) -> str:
        """Presenter bodies for which no card or model-authored reply exists."""

        bodies: list[str] = []
        for result in results:
            policy = policy_for(result.state)
            if (
                not policy.condensed_message
                or policy.body_delivery == "model"
                or self.has_backed_card(result)
            ):
                continue
            rendered = self._assistant_message(result)
            if rendered and rendered not in bodies:
                bodies.append(rendered)
        return "\n\n".join(bodies)

    def delivered_bodies(
        self,
        results: tuple[MainAgentToolOutput, ...],
    ) -> tuple[DeliveredBodyDraft, ...]:
        drafts: list[DeliveredBodyDraft] = []
        for result in results:
            policy = policy_for(result.state)
            if policy.body_retention == "none" or policy.body_title is None:
                continue
            if policy.body_retention == "source" and result.body_source is None:
                continue
            draft = DeliveredBodyDraft(
                kind=result.state,
                title=policy.body_title,
                retention=policy.body_retention,
                body=(
                    self._assistant_message(result)
                    if policy.body_retention == "snapshot"
                    else ""
                ),
                source=result.body_source,
                dependencies=(
                    (
                        *result.body_dependencies,
                        BodyDependency(
                            kind="job",
                            resource_id=result.body_source.job_posting_id,
                        ),
                    )
                    if isinstance(result.body_source, SavedJobBodySource)
                    else result.body_dependencies
                ),
            )
            if draft not in drafts:
                drafts.append(draft)
        return tuple(drafts)

    @staticmethod
    def conversation_content(
        result: MainAgentToolOutput | None,
        *,
        screen: str,
        composed: bool,
    ) -> str:
        """Choose the durable row for each delivery shape."""

        if composed:
            return screen
        if result is None or not condenses_message(result.state):
            return screen
        if delivers_body_elsewhere(result.state) and result.resource_ref is None:
            return screen
        return result.message
