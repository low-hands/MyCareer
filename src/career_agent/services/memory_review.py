from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
import sqlite3
from typing import Any, Literal, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

from career_agent.agent.main_agent_contracts import (
    FreeTextPreferenceConfirmationProposal,
    MemoryAmendmentProposal,
    MemoryTombstoneProposal,
)
from career_agent.domain.career_history import CareerEvidence, CareerRecord
from career_agent.domain.intent_memory import IntentMemoryVersion
from career_agent.services.free_text_preferences import (
    preference_ownership_from_storage,
    preference_topic_key,
)
from career_agent.storage.intent_versions import intent_content_digest, intent_entry_id


_DOCUMENT = re.compile(
    r'^<!-- memory:document export_id="(?P<export_id>memory_export_[a-f0-9]{32})" -->$'
)
_RECORD = re.compile(
    r'^<!-- memory:record record_id="(?P<record_id>career_record_[a-f0-9]{32})" -->$'
)
_START = re.compile(
    r'^<!-- memory:start scope_key="(?P<scope_key>[^"]+)" '
    r'update_id="(?P<update_id>(?:career_evidence_update|intent_update)_[a-f0-9]{32})" -->$'
)
_END = re.compile(
    r'^<!-- memory:end scope_key="(?P<scope_key>[^"]+)" '
    r'update_id="(?P<update_id>(?:career_evidence_update|intent_update)_[a-f0-9]{32})" -->$'
)
_LOOSE_UPDATE_ID = re.compile(
    r'update_id="((?:career_evidence_update|intent_update)_[a-f0-9]{32})"'
)
_MEMORY_MARKER = "<!-- memory:"
# Rendered under a record with no confirmed fact, so the record keeps an anchor
# the user can add its first fact beneath. It is not an item: it never enters the
# snapshot, and reading new facts back skips it.
_EMPTY_RECORD_PLACEHOLDER = "- （尚无已确认事实，可在此行下新增）"


class MemoryReviewSnapshotItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str = Field(pattern=r"^(?:career_fact|free_text_preference)$")
    scope_key: str
    update_id: str
    value: str
    revision: int = Field(ge=1)
    record_id: str | None = None
    detail_ref: str | None = None
    pref_scope: str | None = None
    layer: str | None = None
    timescale: str | None = None
    semantic_stance: str | None = None


class MemoryReviewNewFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    career_record_id: str = Field(pattern=r"^career_record_[a-f0-9]{32}$")
    claim: str = Field(min_length=1, max_length=2000)
    reason: str = Field(min_length=1, max_length=2000)


class MemoryReviewAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    export_id: str
    amendments: tuple[MemoryAmendmentProposal, ...] = ()
    tombstones: tuple[MemoryTombstoneProposal, ...] = ()
    preference_amendments: tuple[FreeTextPreferenceConfirmationProposal, ...] = ()
    new_facts: tuple[MemoryReviewNewFact, ...] = ()
    warnings: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()

    @property
    def proposal_count(self) -> int:
        return sum(
            map(
                len,
                (
                    self.amendments,
                    self.tombstones,
                    self.preference_amendments,
                    self.new_facts,
                ),
            )
        )


class MemoryReviewDiffItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["amend", "delete", "add"]
    target_kind: Literal["career_evidence", "intent_preference"]
    identity: str
    before: str | None = None
    after: str | None = None


class PreparedMemoryReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    analysis: MemoryReviewAnalysis
    changes: tuple[MemoryReviewDiffItem, ...]
    confirmation_digest: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")


class AppliedMemoryReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    confirmation_digest: str
    amendments: int = 0
    tombstones: int = 0
    preference_amendments: int = 0
    new_facts: int = 0
    working_notes_cleared: bool = False
    cleanup_incomplete: tuple[str, ...] = ()


class ReviewManifestStore(Protocol):
    def create_memory_review_export(
        self, *, user_id: str, items: Sequence[dict[str, Any]]
    ) -> str: ...

    def get_memory_review_export(
        self, *, user_id: str, export_id: str
    ) -> tuple[dict[str, Any], ...] | None: ...

    def list_free_text_preferences(
        self, *, user_id: str, statuses: Sequence[str] = ...
    ) -> tuple[IntentMemoryVersion, ...]: ...

    def get_active_free_text_preference(
        self, *, user_id: str, update_id: str
    ) -> IntentMemoryVersion | None: ...

    def get_current_free_text_preference_track(
        self, *, user_id: str, scope_key: str, pref_scope: str
    ) -> IntentMemoryVersion | None: ...

    def confirm_free_text_preference_amendment(self, **kwargs: Any) -> IntentMemoryVersion | None: ...

    def tombstone_free_text_preference(self, **kwargs: Any) -> bool: ...

    def purge_derived_memory(self, **kwargs: Any) -> dict[str, int]: ...


class CareerHistoryReviewStore(Protocol):
    def list_records(
        self, *, user_id: str, limit: int | None = ...
    ) -> tuple[CareerRecord, ...]: ...

    def list_evidence(
        self,
        *,
        user_id: str,
        career_record_id: str | None = ...,
        verification_status: str | None = ...,
        include_historical: bool = ...,
        include_tombstoned: bool = ...,
        limit: int | None = ...,
    ) -> tuple[CareerEvidence, ...]: ...

    def get_current_evidence(
        self, *, user_id: str, scope_key: str
    ) -> CareerEvidence | None: ...

    def create_evidence(self, **kwargs: Any) -> CareerEvidence: ...

    def get_evidence_by_detail_ref(self, **kwargs: Any) -> CareerEvidence | None: ...

    def list_evidence_lineage(self, **kwargs: Any) -> tuple[CareerEvidence, ...]: ...

    def correct_evidence(self, **kwargs: Any) -> Any: ...

    def confirm_evidence(self, **kwargs: Any) -> CareerEvidence: ...

    def tombstone_evidence(self, **kwargs: Any) -> Any: ...


class WorkingNotesCleaner(Protocol):
    def clear(self, *, user_id: str) -> bool: ...


@dataclass(frozen=True)
class _ParsedBlock:
    scope_key: str
    update_id: str
    value: str
    first_line: int
    last_line: int


class MemoryReviewService:
    """Render an editable view and turn edits into non-authoritative proposals."""

    def __init__(
        self,
        *,
        context_store: ReviewManifestStore,
        career_history_store: CareerHistoryReviewStore,
    ) -> None:
        self._context = context_store
        self._history = career_history_store

    def export(self, *, user_id: str) -> tuple[str, str, int]:
        records = self._history.list_records(user_id=user_id)
        evidence = self._history.list_evidence(
            user_id=user_id,
            verification_status="confirmed",
            include_historical=False,
        )
        by_record: dict[str, list[CareerEvidence]] = {}
        snapshot: list[MemoryReviewSnapshotItem] = []
        for item in evidence:
            if (
                item.scope_key is None
                or item.update_id is None
                or item.revision is None
                or item.detail_ref is None
            ):
                continue
            by_record.setdefault(item.career_record_id, []).append(item)
            snapshot.append(
                MemoryReviewSnapshotItem(
                    kind="career_fact",
                    scope_key=item.scope_key,
                    update_id=item.update_id,
                    value=item.claim,
                    revision=item.revision,
                    record_id=item.career_record_id,
                    detail_ref=item.detail_ref,
                )
            )
        preferences = self._context.list_free_text_preferences(
            user_id=user_id,
            statuses=("active",),
        )
        for item in preferences:
            snapshot.append(
                MemoryReviewSnapshotItem(
                    kind="free_text_preference",
                    scope_key=item.scope_key,
                    update_id=item.update_id,
                    value=item.value,
                    revision=item.revision,
                    pref_scope=item.pref_scope,
                    layer=item.layer,
                    timescale=item.timescale,
                    semantic_stance=item.semantic_stance,
                )
            )
        export_id = self._context.create_memory_review_export(
            user_id=user_id,
            items=[item.model_dump(mode="json") for item in snapshot],
        )
        lines = [
            "# MEMORY.md",
            "",
            f'<!-- memory:document export_id="{export_id}" -->',
            "",
            "> 这是长期记忆的可编辑视图。修改不会直接写入权威库。",
            "> 通过 CLI 查看完整 diff 后，可用一次显式确认整批应用。",
            "> 请勿修改 HTML 注释锚点。要删除一条记忆，可删除整块或只删除块内正文。",
            "",
            "## 已确认事实",
        ]
        for record in records:
            items = by_record.get(record.id, ())
            label = " · ".join(
                part for part in (record.title, record.organization) if part
            )
            lines.extend(
                (
                    "",
                    f"### {label}",
                    f'<!-- memory:record record_id="{record.id}" -->',
                )
            )
            if not items:
                lines.append(_EMPTY_RECORD_PLACEHOLDER)
            for item in items:
                lines.extend(self._render_item(item.scope_key or "", item.update_id or "", item.claim))
        if not records:
            lines.extend(("", "（无）"))

        ownership_titles = (
            ("person_stable", "① 人级恒定层"),
            ("person_default", "② 人级默认层"),
            ("role", "③ 岗位类型层"),
            ("situational", "④ 单次情境层"),
        )
        lines.extend(("", "## 已确认偏好（按四层）"))
        grouped: dict[str, list[IntentMemoryVersion]] = {
            name: [] for name, _ in ownership_titles
        }
        for item in preferences:
            ownership = preference_ownership_from_storage(
                pref_scope=item.pref_scope,
                layer=item.layer,
                timescale=item.timescale,
            )
            if ownership == "person_situational":
                ownership = "situational"
            grouped.setdefault(ownership, []).append(item)
        for ownership, title in ownership_titles:
            lines.extend(("", f"### {title}"))
            items = grouped.get(ownership, ())
            if not items:
                lines.append("（无）")
                continue
            for item in items:
                lines.extend(self._render_item(item.scope_key, item.update_id, item.value))
        return "\n".join(lines).rstrip() + "\n", export_id, len(snapshot)

    def analyze(self, *, user_id: str, markdown: str) -> MemoryReviewAnalysis:
        lines = markdown.splitlines()
        document_lines = [
            (index, match)
            for index, line in enumerate(lines)
            if (match := _DOCUMENT.fullmatch(line.strip())) is not None
        ]
        if len(document_lines) != 1:
            raise ValueError("MEMORY.md 的文档锚点缺失或已损坏；请重新导出。")
        export_id = document_lines[0][1].group("export_id")
        raw_snapshot = self._context.get_memory_review_export(
            user_id=user_id,
            export_id=export_id,
        )
        if raw_snapshot is None:
            raise ValueError("MEMORY.md 导出快照不存在或不属于当前用户；请重新导出。")
        snapshot = tuple(MemoryReviewSnapshotItem.model_validate(item) for item in raw_snapshot)
        expected = {item.update_id: item for item in snapshot}
        blocks, occupied, invalid_ids, warnings = self._parse_blocks(lines)
        parsed: dict[str, _ParsedBlock] = {}
        for block in blocks:
            item = expected.get(block.update_id)
            if item is None or item.scope_key != block.scope_key:
                invalid_ids.add(block.update_id)
                warnings.append(
                    f"第 {block.first_line + 1} 行锚点不属于该导出，整块已忽略。"
                )
                continue
            if block.update_id in parsed:
                invalid_ids.add(block.update_id)
                parsed.pop(block.update_id, None)
                warnings.append(
                    f"update_id={block.update_id} 重复出现，相关块全部忽略。"
                )
                continue
            parsed[block.update_id] = block

        amendments: list[MemoryAmendmentProposal] = []
        tombstones: list[MemoryTombstoneProposal] = []
        preference_amendments: list[FreeTextPreferenceConfirmationProposal] = []
        conflicts: list[str] = []
        for base in snapshot:
            if base.update_id in invalid_ids:
                continue
            block = parsed.get(base.update_id)
            proposed_value = block.value.strip() if block is not None else ""
            current = self._current_version(user_id=user_id, item=base)
            if current is None:
                if not proposed_value:
                    continue
                conflicts.append(
                    f"{base.scope_key} 已有新 revision；本次编辑已忽略，请重新导出。"
                )
                continue
            if current.update_id != base.update_id:
                if proposed_value and intent_content_digest(
                    self._version_value(current)
                ) == intent_content_digest(proposed_value):
                    continue
                conflicts.append(
                    f"{base.scope_key} 已有新 revision；本次编辑已忽略，请重新导出。"
                )
                continue
            if not proposed_value:
                tombstones.append(self._tombstone(base))
                continue
            if intent_content_digest(proposed_value) == intent_content_digest(base.value):
                continue
            if base.kind == "career_fact":
                amendments.append(
                    MemoryAmendmentProposal(
                        target_kind="career_evidence",
                        detail_ref=base.detail_ref,
                        new_claim=proposed_value,
                        reason="用户在 MEMORY.md 审阅面修改了该条目。",
                    )
                )
            else:
                preference_amendments.append(
                    FreeTextPreferenceConfirmationProposal(
                        update_id=base.update_id,
                        base_update_id=base.update_id,
                        expected_content_sha256=intent_content_digest(base.value),
                        topic_key=preference_topic_key(base.scope_key),
                        statement=proposed_value,
                        ownership=preference_ownership_from_storage(
                            pref_scope=base.pref_scope or "freeform.person_default",
                            layer=base.layer or "contextual",
                            timescale=base.timescale or "permanent",
                        ),
                        pref_scope=base.pref_scope or "freeform.person_default",
                    )
                )

        new_facts = self._new_fact_proposals(
            user_id=user_id,
            lines=lines,
            occupied=occupied,
            known_values={intent_content_digest(item.value) for item in snapshot},
        )
        return MemoryReviewAnalysis(
            export_id=export_id,
            amendments=tuple(amendments),
            tombstones=tuple(tombstones),
            preference_amendments=tuple(preference_amendments),
            new_facts=tuple(new_facts),
            warnings=tuple(warnings),
            conflicts=tuple(conflicts),
        )

    def prepare(self, *, user_id: str, markdown: str) -> PreparedMemoryReview:
        """Build the complete human-facing diff and bind it to one digest."""

        analysis = self.analyze(user_id=user_id, markdown=markdown)
        snapshot = self._snapshot_by_update_id(
            user_id=user_id,
            export_id=analysis.export_id,
        )
        changes: list[MemoryReviewDiffItem] = []
        for proposal in analysis.amendments:
            base = next(
                (
                    item
                    for item in snapshot.values()
                    if item.detail_ref == proposal.detail_ref
                ),
                None,
            )
            changes.append(
                MemoryReviewDiffItem(
                    action="amend",
                    target_kind="career_evidence",
                    identity=proposal.detail_ref,
                    before=base.value if base is not None else None,
                    after=proposal.new_claim,
                )
            )
        for proposal in analysis.preference_amendments:
            base = snapshot.get(str(proposal.base_update_id))
            changes.append(
                MemoryReviewDiffItem(
                    action="amend",
                    target_kind="intent_preference",
                    identity=str(proposal.base_update_id),
                    before=base.value if base is not None else None,
                    after=proposal.statement,
                )
            )
        for proposal in analysis.new_facts:
            changes.append(
                MemoryReviewDiffItem(
                    action="add",
                    target_kind="career_evidence",
                    identity=(
                        f"{proposal.career_record_id}:"
                        f"{intent_content_digest(proposal.claim)}"
                    ),
                    after=proposal.claim,
                )
            )
        for proposal in analysis.tombstones:
            identity = str(proposal.update_id or proposal.detail_ref)
            base = (
                snapshot.get(str(proposal.update_id))
                if proposal.update_id is not None
                else next(
                    (
                        item
                        for item in snapshot.values()
                        if item.detail_ref == proposal.detail_ref
                    ),
                    None,
                )
            )
            changes.append(
                MemoryReviewDiffItem(
                    action="delete",
                    target_kind=proposal.target_kind,
                    identity=identity,
                    before=base.value if base is not None else None,
                )
            )
        canonical = json.dumps(
            {
                "export_id": analysis.export_id,
                "changes": [item.model_dump(mode="json") for item in changes],
                "warnings": analysis.warnings,
                "conflicts": analysis.conflicts,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return PreparedMemoryReview(
            analysis=analysis,
            changes=tuple(changes),
            confirmation_digest=digest,
        )

    def apply(
        self,
        *,
        user_id: str,
        markdown: str,
        confirmation_digest: str,
        working_notes: WorkingNotesCleaner | None = None,
    ) -> AppliedMemoryReview:
        """Apply exactly the previously displayed diff after one confirmation."""

        prepared = self.prepare(user_id=user_id, markdown=markdown)
        if prepared.confirmation_digest != confirmation_digest:
            raise ValueError(
                "MEMORY.md diff changed after review; review it again before confirming."
            )
        analysis = prepared.analysis
        if analysis.conflicts:
            raise ValueError("MEMORY.md contains stale revisions; export and review again.")

        snapshot = self._snapshot_by_update_id(
            user_id=user_id,
            export_id=analysis.export_id,
        )
        # Preflight every CAS target before the first authoritative write. This
        # cannot make two SQLite files one transaction, but it removes ordinary
        # stale-batch partial writes; a rerun treats already-applied values as no-op.
        for proposal in analysis.amendments:
            base = next(
                (
                    item
                    for item in snapshot.values()
                    if item.detail_ref == proposal.detail_ref
                ),
                None,
            )
            current = self._history.get_evidence_by_detail_ref(
                user_id=user_id,
                detail_ref=proposal.detail_ref,
            )
            if (
                base is None
                or current is None
                or current.content_digest != intent_content_digest(base.value)
            ):
                raise ValueError("A MEMORY.md amendment target changed; review again.")

        for proposal in analysis.preference_amendments:
            current = self._context.get_active_free_text_preference(
                user_id=user_id,
                update_id=str(proposal.base_update_id),
            )
            if (
                current is None
                or current.content_digest != proposal.expected_content_sha256
            ):
                raise ValueError("A MEMORY.md preference changed; review again.")

        for proposal in analysis.tombstones:
            if proposal.target_kind == "intent_preference":
                current = self._context.get_active_free_text_preference(
                    user_id=user_id,
                    update_id=str(proposal.update_id),
                )
                if (
                    current is None
                    or current.content_digest != proposal.expected_content_sha256
                ):
                    raise ValueError("A MEMORY.md deletion target changed; review again.")
            else:
                current = self._history.get_evidence_by_detail_ref(
                    user_id=user_id,
                    detail_ref=str(proposal.detail_ref),
                )
                if (
                    current is None
                    or current.content_digest != proposal.expected_content_sha256
                ):
                    raise ValueError("A MEMORY.md deletion target changed; review again.")

        for proposal in analysis.amendments:
            evidence = self._history.get_evidence_by_detail_ref(
                user_id=user_id,
                detail_ref=proposal.detail_ref,
            )
            if evidence is None:
                raise ValueError("A MEMORY.md amendment target changed; review again.")
            self._history.correct_evidence(
                user_id=user_id,
                career_evidence_id=evidence.id,
                new_claim=proposal.new_claim,
                reason=proposal.reason,
            )

        for proposal in analysis.preference_amendments:
            amended = self._context.confirm_free_text_preference_amendment(
                user_id=user_id,
                base_update_id=str(proposal.base_update_id),
                expected_content_sha256=str(proposal.expected_content_sha256),
                statement=proposal.statement,
            )
            if amended is None:
                raise ValueError("A MEMORY.md preference changed; review again.")

        for proposal in analysis.new_facts:
            digest = intent_content_digest(proposal.claim)
            existing = next(
                (
                    item
                    for item in self._history.list_evidence(
                        user_id=user_id,
                        career_record_id=proposal.career_record_id,
                        include_historical=False,
                    )
                    if intent_content_digest(item.claim) == digest
                ),
                None,
            )
            if existing is not None and existing.verification_status == "confirmed":
                continue
            pending = existing or self._history.create_evidence(
                user_id=user_id,
                career_record_id=proposal.career_record_id,
                claim=proposal.claim,
                origin="user_input",
            )
            self._history.confirm_evidence(
                user_id=user_id,
                career_evidence_id=pending.id,
                reason=proposal.reason,
            )

        notes_cleared = False
        cleanup_incomplete: list[str] = []

        def clear_working_notes() -> None:
            nonlocal notes_cleared
            if working_notes is None:
                return
            try:
                notes_cleared = (
                    working_notes.clear(user_id=user_id) or notes_cleared
                )
            except OSError:
                if "working_notes" not in cleanup_incomplete:
                    cleanup_incomplete.append("working_notes")

        for proposal in analysis.tombstones:
            if proposal.target_kind == "intent_preference":
                committed = self._context.tombstone_free_text_preference(
                    user_id=user_id,
                    scope_key=str(proposal.scope_key),
                    pref_scope=str(proposal.pref_scope),
                    update_id=str(proposal.update_id),
                    expected_content_sha256=str(
                        proposal.expected_content_sha256
                    ),
                    reason=proposal.reason,
                )
                if not committed:
                    raise ValueError("A MEMORY.md deletion target changed; review again.")
                clear_working_notes()
                try:
                    self._context.purge_derived_memory(
                        user_id=user_id,
                        scope_key=intent_entry_id(
                            str(proposal.scope_key),
                            str(proposal.pref_scope),
                        ),
                        update_ids=(str(proposal.update_id),),
                    )
                except (OSError, sqlite3.Error, ValueError):
                    cleanup_incomplete.append(
                        f"derived_memory:{proposal.update_id}"
                    )
            else:
                evidence = self._history.get_evidence_by_detail_ref(
                    user_id=user_id,
                    detail_ref=str(proposal.detail_ref),
                )
                if evidence is None or evidence.scope_key is None:
                    raise ValueError("A MEMORY.md deletion target changed; review again.")
                lineage = self._history.list_evidence_lineage(
                    user_id=user_id,
                    scope_key=evidence.scope_key,
                )
                tombstone = self._history.tombstone_evidence(
                    user_id=user_id,
                    career_evidence_id=evidence.id,
                    reason=proposal.reason,
                    actor_type="user",
                    expected_content_sha256=proposal.expected_content_sha256,
                )
                clear_working_notes()
                try:
                    self._context.purge_derived_memory(
                        user_id=user_id,
                        scope_key=tombstone.scope_key,
                        lineage_markers=tuple(
                            marker
                            for item in lineage
                            for marker in (item.detail_ref, item.source_ref)
                            if marker is not None
                        ),
                        update_ids=tuple(
                            item.update_id
                            for item in lineage
                            if item.update_id is not None
                        ),
                    )
                except (OSError, sqlite3.Error, ValueError):
                    cleanup_incomplete.append(
                        f"derived_memory:{proposal.detail_ref}"
                    )
        return AppliedMemoryReview(
            confirmation_digest=confirmation_digest,
            amendments=len(analysis.amendments),
            tombstones=len(analysis.tombstones),
            preference_amendments=len(analysis.preference_amendments),
            new_facts=len(analysis.new_facts),
            working_notes_cleared=notes_cleared,
            cleanup_incomplete=tuple(cleanup_incomplete),
        )

    def _snapshot_by_update_id(
        self,
        *,
        user_id: str,
        export_id: str,
    ) -> dict[str, MemoryReviewSnapshotItem]:
        raw = self._context.get_memory_review_export(
            user_id=user_id,
            export_id=export_id,
        )
        if raw is None:
            raise ValueError("MEMORY.md export snapshot no longer exists.")
        return {
            item.update_id: item
            for item in (
                MemoryReviewSnapshotItem.model_validate(value) for value in raw
            )
        }

    @staticmethod
    def _render_item(scope_key: str, update_id: str, value: str) -> tuple[str, ...]:
        return (
            "",
            f'<!-- memory:start scope_key="{scope_key}" update_id="{update_id}" -->',
            f"- {value}",
            f'<!-- memory:end scope_key="{scope_key}" update_id="{update_id}" -->',
        )

    def _current_version(
        self, *, user_id: str, item: MemoryReviewSnapshotItem
    ) -> CareerEvidence | IntentMemoryVersion | None:
        if item.kind == "career_fact":
            return self._history.get_current_evidence(
                user_id=user_id,
                scope_key=item.scope_key,
            )
        return self._context.get_current_free_text_preference_track(
            user_id=user_id,
            scope_key=item.scope_key,
            pref_scope=item.pref_scope or "freeform.person_default",
        )

    @staticmethod
    def _version_value(item: CareerEvidence | IntentMemoryVersion) -> str:
        return item.claim if isinstance(item, CareerEvidence) else item.value

    @staticmethod
    def _tombstone(item: MemoryReviewSnapshotItem) -> MemoryTombstoneProposal:
        if item.kind == "career_fact":
            return MemoryTombstoneProposal(
                target_kind="career_evidence",
                detail_ref=item.detail_ref,
                expected_content_sha256=intent_content_digest(item.value),
                reason="用户在 MEMORY.md 审阅面删除了该条目。",
            )
        return MemoryTombstoneProposal(
            target_kind="intent_preference",
            scope_key=item.scope_key,
            update_id=item.update_id,
            pref_scope=item.pref_scope,
            expected_content_sha256=intent_content_digest(item.value),
            reason="用户在 MEMORY.md 审阅面删除了该条目。",
        )

    @staticmethod
    def _parse_blocks(
        lines: list[str],
    ) -> tuple[list[_ParsedBlock], set[int], set[str], list[str]]:
        blocks: list[_ParsedBlock] = []
        occupied: set[int] = set()
        invalid_ids: set[str] = set()
        warnings: list[str] = []
        index = 0
        while index < len(lines):
            stripped = lines[index].strip()
            start = _START.fullmatch(stripped)
            if start is None:
                if _MEMORY_MARKER in stripped and not (
                    _DOCUMENT.fullmatch(stripped) or _RECORD.fullmatch(stripped)
                ):
                    loose = _LOOSE_UPDATE_ID.search(stripped)
                    if loose:
                        invalid_ids.add(loose.group(1))
                    begin = index
                    index += 1
                    while index < len(lines):
                        occupied.add(index)
                        if _END.fullmatch(lines[index].strip()) is not None:
                            index += 1
                            break
                        if _START.fullmatch(lines[index].strip()) is not None:
                            break
                        index += 1
                    occupied.update(range(begin, index))
                    warnings.append(
                        f"第 {begin + 1} 行记忆锚点无法解析，整块已忽略。"
                    )
                    continue
                index += 1
                continue
            scope_key = start.group("scope_key")
            update_id = start.group("update_id")
            begin = index
            index += 1
            body: list[str] = []
            valid = False
            while index < len(lines):
                stripped = lines[index].strip()
                end = _END.fullmatch(stripped)
                if end is not None:
                    valid = (
                        end.group("scope_key") == scope_key
                        and end.group("update_id") == update_id
                    )
                    index += 1
                    break
                if _MEMORY_MARKER in stripped:
                    loose = _LOOSE_UPDATE_ID.search(stripped)
                    if loose:
                        invalid_ids.add(loose.group(1))
                    index += 1
                    break
                body.append(lines[index])
                index += 1
            occupied.update(range(begin, index))
            if not valid:
                invalid_ids.add(update_id)
                warnings.append(
                    f"第 {begin + 1} 行开始的记忆锚点无法解析，整块已忽略。"
                )
                continue
            value = "\n".join(body).strip()
            if value.startswith("- "):
                value = value[2:].strip()
            blocks.append(
                _ParsedBlock(scope_key, update_id, value, begin, index - 1)
            )
        return blocks, occupied, invalid_ids, warnings

    def _new_fact_proposals(
        self,
        *,
        user_id: str,
        lines: list[str],
        occupied: set[int],
        known_values: set[str],
    ) -> list[MemoryReviewNewFact]:
        proposals: list[MemoryReviewNewFact] = []
        evidence_by_record: dict[str, dict[str, CareerEvidence]] = {}
        record_id: str | None = None
        in_facts = False
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "## 已确认事实":
                in_facts = True
                record_id = None
                continue
            if stripped.startswith("## ") and stripped != "## 已确认事实":
                in_facts = False
                record_id = None
            record = _RECORD.fullmatch(stripped)
            if record is not None:
                record_id = record.group("record_id")
                continue
            if not in_facts or index in occupied or record_id is None:
                continue
            if not stripped.startswith("- ") or stripped == _EMPTY_RECORD_PLACEHOLDER:
                continue
            claim = stripped[2:].strip()
            digest = intent_content_digest(claim)
            if not claim or digest in known_values:
                continue
            if record_id not in evidence_by_record:
                evidence_by_record[record_id] = {
                    intent_content_digest(item.claim): item
                    for item in self._history.list_evidence(
                        user_id=user_id,
                        career_record_id=record_id,
                        include_historical=False,
                    )
                }
            existing = evidence_by_record[record_id].get(digest)
            if existing is not None and existing.verification_status == "confirmed":
                known_values.add(digest)
                continue
            proposals.append(
                MemoryReviewNewFact(
                    career_record_id=record_id,
                    claim=claim,
                    reason="用户在 MEMORY.md 审阅面新增了无锚点事实。",
                )
            )
            known_values.add(digest)
        return proposals
