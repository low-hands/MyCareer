"""Pure checks for model proposals; execution guards remain authoritative."""
from __future__ import annotations

from career_agent.agent.capabilities.effects import is_notes_guarded
from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.capabilities.catalog import CAPABILITIES
from career_agent.agent.middleware.working_notes import working_notes_only_tokens


def proposal_rejection(context: MainAgentContext, name: str, arguments: dict) -> str | None:
    if is_notes_guarded(name) and working_notes_only_tokens(arguments=arguments, context=context):
        return (
            "Arguments contain an unconfirmed preference found only in working notes. "
            "Nothing executed. Ask the user to confirm the preference before using it."
        )
    descriptor = CAPABILITIES.get(name)
    if descriptor is not None:
        try:
            descriptor.validate_arguments(context, arguments)
        except ValueError:
            return (
                "The proposed arguments do not satisfy this capability's input contract "
                "or resource binding. Nothing executed. Follow its schema, documented "
                "prerequisites and supplied references. Clarify missing or "
                "ambiguous inputs instead of inventing selectors."
            )
    return None
