"""Per-model-call capability offer and prompt projection."""

from __future__ import annotations

from dataclasses import replace
import json
import logging
from typing import Any, Literal

from career_agent.agent.capabilities.catalog import CAPABILITIES
from career_agent.agent.capabilities.proactive import proactive_tool_names
from career_agent.agent.capabilities.reachability import STATE_GATED_TOOLS, reachable
from career_agent.agent.capabilities.search import (
    CapabilityScorer, search_catalog, searchable_capabilities,
)
from career_agent.agent.capabilities.selection import (
    ALWAYS_OFFERED_TOOLS, CapabilitySelection, prepare_capability_selection,
)
from career_agent.agent.capabilities.waiting import waiting_tool_names
from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.contracts.task_state import ConversationTaskState
from career_agent.agent.providers.token_budget import count_tokens


MAX_SEARCH_CALLS_PER_TURN = 5
_LOGGER = logging.getLogger(__name__)


def capability_directory() -> str:
    groups: dict[str, list[str]] = {}
    for item in searchable_capabilities():
        groups.setdefault(item.namespace or "control", []).append(
            f"- {item.name}: {item.summary}"
        )
    return "\n".join(
        f"[{namespace}]\n" + "\n".join(lines)
        for namespace, lines in groups.items()
    )


class SearchStrategy:
    def __init__(self, semantic_index: CapabilityScorer | None = None) -> None:
        self._semantic_index = semantic_index
        self._schema_cache: dict[tuple[str, ...], tuple[dict[str, Any], ...]] = {}
        self._intent_cache: dict[tuple[str, object, str], tuple[str, ...]] = {}
        self._searchable_names = frozenset(
            item.name for item in searchable_capabilities()
        )
        self._directory = capability_directory()

    def resolve_unoffered(
        self, name: str, context: MainAgentContext,
    ) -> Literal["load", "refuse", "reject"]:
        """Classify a discoverable name before deciding whether to load it.

        A waiting or unreachable tool goes to authorization under its original
        name. Authorization can then issue a deterministic refusal before
        argument projection or execution; loading it would only repeat the
        same unavailable state.
        """
        if name not in self._searchable_names:
            return "reject"
        if (
            name in waiting_tool_names(context.tool_observations)
            or not reachable(name, context.task)
        ):
            return "refuse"
        return "load"

    def _intent_names(self, context: MainAgentContext) -> tuple[str, ...]:
        if not context.user_message.strip():
            return ()
        key = (context.conversation_id, context.received_at, context.user_message)
        if key not in self._intent_cache:
            # The same user turn can contain several model calls. Keep the
            # result for that turn, without persisting it in task state.
            if len(self._intent_cache) >= 1024:
                self._intent_cache.pop(next(iter(self._intent_cache)))
            query = context.user_message[:200]
            semantic_scores = None
            if self._semantic_index is not None:
                try:
                    semantic_scores = self._semantic_index.scores(query)
                except Exception as error:
                    # The embedding service is optional; lexical discovery must
                    # remain available when a query fails or times out.
                    _LOGGER.warning(
                        "capability intent embedding unavailable; lexical fallback: %s",
                        type(error).__name__,
                    )
                    semantic_scores = None
            if semantic_scores is None:
                self._intent_cache[key] = search_catalog(query=query, limit=5)
            else:
                self._intent_cache[key] = search_catalog(
                    query=query, limit=5, semantic_scores=semantic_scores,
                )
        return self._intent_cache[key]

    def select(
        self, context: MainAgentContext,
        registered: tuple[dict[str, Any], ...],
    ) -> CapabilitySelection:
        task = context.task
        waiting = waiting_tool_names(context.tool_observations)
        state_needed = tuple(
            name for name in CAPABILITIES
            if name in STATE_GATED_TOOLS and reachable(name, task)
        )
        if context.turn_continuation_capability is not None:
            state_needed = (*state_needed, context.turn_continuation_capability)
        registered_names = frozenset(
            schema["function"]["name"] for schema in registered
        )
        source_names = (
            ("always", ALWAYS_OFFERED_TOOLS),
            ("loaded", task.loaded_capabilities),
            ("state", state_needed),
            ("intent", tuple(
                name for name in self._intent_names(context)
                if name in registered_names
            )),
            ("proactive", proactive_tool_names(context)),
        )
        sources: dict[str, str] = {}
        for source, names in source_names:
            for name in names:
                if name in registered_names and CAPABILITIES[name].model_callable:
                    sources.setdefault(name, source)
        selected = tuple(name for name in sources if name not in waiting)
        selection = prepare_capability_selection(
            selected, task=task, registered_schemas=registered,
        )
        cached = self._schema_cache.get(selection.offered_names)
        if cached is None:
            cached = selection.schemas
            self._schema_cache[selection.offered_names] = cached
        result = CapabilitySelection(
            selected_names=selection.selected_names,
            offered_names=selection.offered_names,
            blocked_requirements=selection.blocked_requirements,
            schemas=cached,
            sources=tuple((name, sources[name]) for name in selection.selected_names),
            waiting_suppressed=tuple(name for name in CAPABILITIES if name in waiting and name in sources),
        )
        projection = self.tool_context(task, result)
        if context.through_sequence == 0 and context.recent_from_sequence in (None, 1):
            projection["available_now"] = [
                name for name in projection["available_now"]
                if name != "read_conversation_span"
            ]
        return replace(result, tool_projection=projection)

    @staticmethod
    def tool_context(task: ConversationTaskState, selection: CapabilitySelection) -> dict[str, object]:
        sources = dict(selection.sources)
        blocked = sorted(
            selection.blocked_requirements,
            key=lambda item: (sources.get(item[0]) not in {"always", "loaded"}, item[0]),
        )[:5]
        return {
            "available_now": list(selection.offered_names),
            "loaded_capabilities": list(task.loaded_capabilities),
            "blocked": [f"{name}: {requirement}" for name, requirement in blocked],
        }

    def offers_tool(self, context: MainAgentContext, name: str) -> bool:
        # The prelude is evaluated before decide; its stable tool is always on.
        return name in ALWAYS_OFFERED_TOOLS and reachable(name, context.task)

    def tool_policy(self) -> str:
        guidance = (
            "The capability directory lists possible work, but only offered tools "
            "can be called now. To use another capability, call search_capabilities: "
            "use names when known, including several known tools needed for the "
            "current request; otherwise query for the next need. "
            "Loaded tools become available from the next decision and remain "
            "loaded for this conversation. Satisfy a blocked prerequisite before "
            "calling that tool. Loading does not grant permission; execution "
            "still checks authorization and approval. Reviewed follow-up and "
            "bound-resource read tools may also be offered as state changes; "
            "offer alone is not a reason to call them. "
            "When a questionnaire pauses a requested capability, set its "
            "continuation_capability to that exact tool name so the resumed "
            "turn can offer it; omit it for a standalone questionnaire. "
        )
        return guidance + "\n\nCapability directory:\n" + self._directory + "\n\n"


def selection_trace(selection: CapabilitySelection) -> dict[str, object]:
    counts = {source: sum(kind == source for _, kind in selection.sources)
              for source in ("always", "loaded", "state", "intent", "proactive")}
    return {
        "selection_sources": counts,
        "blocked_requirements": dict(selection.blocked_requirements),
        "waiting_suppressed": selection.waiting_suppressed,
        "offered_tool_count": len(selection.offered_names),
        "tool_schema_tokens_proxy": count_tokens(
            json.dumps(selection.schemas, ensure_ascii=False, sort_keys=True)
        ),
    }
