from __future__ import annotations

from career_agent.agent.career_context import CareerContextProjector
from career_agent.agent.main_agent_contracts import MainAgentContext
from career_agent.agent.main_state import MainAgentState, PendingAction
from career_agent.storage.intent_versions import intent_entry_id


class ContextHydrator:
    """Project career memory and record every memory scope shown this turn."""

    def __init__(
        self,
        *,
        career_context_projector: CareerContextProjector | None,
    ) -> None:
        self._career_context_projector = career_context_projector

    def project_context(self, context: MainAgentContext) -> MainAgentContext:
        projector = self._career_context_projector
        if projector is None:
            return context
        memory = projector.project(
            user_id=context.profile.user_id,
            query=context.user_message,
        )
        return context.model_copy(update={"career_memory": memory})

    def hydrate(self, state: MainAgentState) -> MainAgentState:
        context = state["context"]
        free_text_scope_keys = self.free_text_preference_scope_keys(context)
        # A prelude read that brought the turn here has been observed; the
        # model's first decision must not inherit its policy ownership.
        pending: PendingAction = {}
        if self._career_context_projector is None:
            return {
                "pending": pending,
                "career_memory_scope_keys": free_text_scope_keys,
            }
        context = self.project_context(context)
        return {
            "pending": pending,
            "context": context,
            "career_memory_scope_keys": tuple(
                dict.fromkeys(
                    (
                        *(
                            binding.entry_id
                            for binding in context.career_memory.telemetry_bindings
                        ),
                        *free_text_scope_keys,
                    )
                )
            ),
        }

    @staticmethod
    def free_text_preference_scope_keys(
        context: MainAgentContext,
    ) -> tuple[str, ...]:
        """Preference tracks whose values were exposed in this turn's prompt."""

        return tuple(
            dict.fromkeys(
                intent_entry_id(item.scope_key, item.pref_scope)
                for item in context.free_text_preferences
            )
        )
