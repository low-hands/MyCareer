from __future__ import annotations

from dataclasses import dataclass

from career_agent.domain.memory_scope import (
    CanonicalScope,
    ScopeProposal,
    ScopeResolutionQueueItem,
)
from career_agent.harness.observability import (
    conversation_trace_key,
    record_active_trace,
)
from career_agent.services.canonical_scope import CanonicalScopeResolver
from career_agent.storage.scope_resolution import SQLiteScopeResolutionStore


def _conversation_scope(proposal: ScopeProposal) -> dict[str, str]:
    """The join key every conversation-scoped trace event carries.

    Without it these events persist but no conversation-scoped reader can see
    them, which reads as "no memory writes happened" rather than as missing
    instrumentation.
    """

    if proposal.conversation_id is None:
        return {}
    return {
        "conversation_key": conversation_trace_key(
            proposal.user_id, proposal.conversation_id
        )
    }


class UnresolvedMemoryScopeError(ValueError):
    def __init__(self, item: ScopeResolutionQueueItem) -> None:
        super().__init__(
            "Memory write requires a canonical scope; the proposal was queued "
            "for clarification."
        )
        self.queue_item = item


@dataclass(frozen=True)
class ScopeAdmission:
    canonical_scope: CanonicalScope | None
    queue_item: ScopeResolutionQueueItem | None

    @property
    def admitted(self) -> bool:
        return self.canonical_scope is not None


class MemoryScopeWriteGate:
    """Fail-closed boundary shared by current and future memory writers."""

    def __init__(
        self,
        resolver: CanonicalScopeResolver,
        queue: SQLiteScopeResolutionStore,
    ) -> None:
        self._resolver = resolver
        self._queue = queue

    def admit(self, proposal: ScopeProposal) -> ScopeAdmission:
        resolution = self._resolver.resolve(proposal)
        if resolution.canonical_scope is not None:
            scope = resolution.canonical_scope
            record_active_trace(
                "memory_scope_resolved",
                "memory_scope",
                outcome="succeeded",
                details={
                    "scope_key": scope.scope_key,
                    "family": scope.family,
                    "source_kind": proposal.source_kind,
                    "content_digest": self._queue.content_digest(
                        proposal.proposed_value
                    ),
                    "binding_profile": "p2",
                    **_conversation_scope(proposal),
                },
            )
            return ScopeAdmission(canonical_scope=scope, queue_item=None)
        item = self._queue.enqueue(resolution)
        record_active_trace(
            "memory_scope_unresolved",
            "memory_scope",
            outcome="succeeded",
            details={
                "queue_item_id": item.id,
                "family": item.family,
                "source_kind": item.source_kind,
                "content_digest": item.content_digest,
                "reason": item.reason,
                "binding_profile": "p2",
                **_conversation_scope(proposal),
            },
        )
        return ScopeAdmission(canonical_scope=None, queue_item=item)

    def require(self, proposal: ScopeProposal) -> CanonicalScope:
        admission = self.admit(proposal)
        if admission.canonical_scope is None:
            if admission.queue_item is None:
                raise RuntimeError("unresolved admission lost its queue item")
            raise UnresolvedMemoryScopeError(admission.queue_item)
        return admission.canonical_scope

    @staticmethod
    def record_committed(
        scopes_and_values: tuple[tuple[CanonicalScope, str], ...],
        *,
        proposals: tuple[ScopeProposal, ...] = (),
        versions: tuple[object, ...] = (),
    ) -> None:
        if not scopes_and_values:
            return
        current_versions = {
            version.scope_key: version
            for version in versions
            if all(
                hasattr(version, name)
                for name in (
                    "scope_key",
                    "update_id",
                    "content_digest",
                    "revision",
                    "superseded_at",
                )
            )
            and version.superseded_at is None
        }
        entries = []
        for scope, value in scopes_and_values:
            entry = {
                "entry_id": scope.scope_key,
                "content_digest": SQLiteScopeResolutionStore.content_digest(
                    value
                ),
            }
            version = current_versions.get(scope.scope_key)
            if version is not None:
                entry.update(
                    {
                        "update_id": version.update_id,
                        "revision": version.revision,
                        "lifecycle_status": "current",
                    }
                )
            entries.append(entry)
        complete = all(
            scope.scope_key in current_versions
            for scope, _ in scopes_and_values
        )
        record_active_trace(
            "memory_write_observed",
            "memory_scope",
            outcome="succeeded",
            details={
                "binding_profile": "p1" if complete else "p2",
                "version_inventory_complete": complete,
                **(_conversation_scope(proposals[0]) if proposals else {}),
                "entries": entries,
            },
        )
