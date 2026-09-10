from __future__ import annotations

import hashlib
import json

from career_agent.agent.resume_job_match_contracts import (
    ConfirmedResumeFact,
    IntentStateAnchor,
    IntentStateTransition,
    ResumeJobMatchAuditProposal,
    ResumeJobMatchResult,
    ResumeJobMatchWorker,
)
from career_agent.storage.context import CareerProfileStore
from career_agent.storage.career_history import CareerHistoryStore
from career_agent.storage.jobs import JobPostingRepository
from career_agent.storage.resumes import ResumeStore
from career_agent.storage.resume_job_matches import (
    SQLiteResumeJobMatchStore,
    StoredResumeJobMatch,
)
from career_agent.services.free_text_preferences import preference_scope_domain
from career_agent.services.preference_resolution import (
    PreferenceResolutionContext,
    resolve_effective_preferences,
)


class ResumeJobMatchInputNotFoundError(ValueError):
    """Raised when either user-owned input cannot be resolved."""

    def __init__(self, input_kind: str) -> None:
        self.input_kind = input_kind
        super().__init__(f"{input_kind} not found or does not belong to the current user")


class ResumeJobMatchService:
    """Keeps complete resume/JD documents behind a narrow matching boundary."""

    def __init__(
        self,
        resume_store: ResumeStore,
        job_repository: JobPostingRepository,
        career_history_store: CareerHistoryStore,
        worker: ResumeJobMatchWorker,
        match_store: SQLiteResumeJobMatchStore,
        *,
        matcher_version: str = "resume-job-match-v2",
        career_profile_store: CareerProfileStore | None = None,
    ) -> None:
        self._resume_store = resume_store
        self._job_repository = job_repository
        self._career_history_store = career_history_store
        self._worker = worker
        self._match_store = match_store
        self._matcher_version = matcher_version
        self._career_profile_store = career_profile_store

    def match(
        self,
        *,
        user_id: str,
        resume_version_id: str,
        job_posting_id: str,
    ) -> StoredResumeJobMatch:
        if not user_id.strip() or not resume_version_id.strip() or not job_posting_id.strip():
            raise ValueError("user_id, resume_version_id, and job_posting_id are required")
        document = self._resume_store.read_version_document(
            user_id=user_id,
            resume_version_id=resume_version_id,
        )
        if document is None:
            raise ResumeJobMatchInputNotFoundError("resume_version")
        job = self._job_repository.get_job(
            user_id=user_id,
            job_posting_id=job_posting_id,
        )
        if job is None:
            raise ResumeJobMatchInputNotFoundError("job_posting")

        intent_states, transitions = self._intent_state(
            user_id=user_id,
            resume_version_id=resume_version_id,
            job_posting_id=job_posting_id,
        )
        confirmed_facts = tuple(
            ConfirmedResumeFact(
                claim=evidence.claim,
                source_locator=evidence.source_locator,
                source_quote=evidence.source_quote,
            )
            for evidence in self._career_history_store.list_evidence(
                user_id=user_id,
                verification_status="confirmed",
                source_resume_version_id=resume_version_id,
            )
            if evidence.source_locator is not None
            and evidence.source_quote is not None
        )
        evidence_fingerprint = self._evidence_fingerprint(
            confirmed_facts,
            intent_states,
            transitions,
        )
        cached = self._match_store.find(
            user_id=user_id,
            resume_version_id=resume_version_id,
            jd_snapshot_id=job.snapshot.id,
            matcher_version=self._matcher_version,
            evidence_fingerprint=evidence_fingerprint,
        )
        if cached is not None:
            return cached
        result = self._worker.match(
            document=document,
            jd_text=job.snapshot.content,
            confirmed_facts=confirmed_facts,
            intent_states=intent_states,
        )
        result = self._repair_stale_state(
            result=result,
            jd_text=job.snapshot.content,
            transitions=transitions,
        )
        return self._match_store.save(
            user_id=user_id,
            resume_version_id=resume_version_id,
            job_posting_id=job_posting_id,
            jd_snapshot_id=job.snapshot.id,
            matcher_version=self._matcher_version,
            evidence_fingerprint=evidence_fingerprint,
            result=result,
        )

    def get_match(self, *, user_id: str, match_id: str) -> StoredResumeJobMatch:
        if not user_id.strip() or not match_id.strip():
            raise ValueError("user_id and match_id are required")
        stored = self._match_store.get(user_id=user_id, match_id=match_id)
        if stored is None:
            raise ResumeJobMatchInputNotFoundError("match")
        return stored

    @staticmethod
    def _evidence_fingerprint(
        facts: tuple[ConfirmedResumeFact, ...],
        intent_states: tuple[IntentStateAnchor, ...] = (),
        transitions: tuple[IntentStateTransition, ...] = (),
    ) -> str:
        # Every component is event-driven, so a cached match is invalidated by
        # a corroboration or a revision and never by the passage of time.
        serialized = json.dumps(
            {
                "facts": [fact.model_dump(mode="json") for fact in facts],
                "intent_states": [
                    item.model_dump(mode="json") for item in intent_states
                ],
                "transitions": [
                    item.model_dump(mode="json") for item in transitions
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _intent_state(
        self,
        *,
        user_id: str,
        resume_version_id: str,
        job_posting_id: str | None = None,
    ) -> tuple[
        tuple[IntentStateAnchor, ...],
        tuple[IntentStateTransition, ...],
    ]:
        source = self._resume_store.get_version(
            user_id=user_id,
            resume_version_id=resume_version_id,
        )
        if source is None:
            return (), ()
        resume, _ = source
        target_role = self._resume_store.get_target_role(
            user_id=user_id,
            target_role_id=resume.target_role_id,
        )
        versions = list(
            self._resume_store.list_target_role_intent_versions(
                user_id=user_id,
            )
        )
        versions = [
            item
            for item in versions
            if item.scope_key.startswith(
                f"target_role_intent/{resume.target_role_id}/"
            )
            and item.admission_status == "active"
        ]
        if self._career_profile_store is not None:
            versions.extend(
                item
                for item in self._career_profile_store.list_profile_intent_versions(
                    user_id=user_id,
                )
                if item.admission_status == "active"
                and not item.pref_scope.startswith("freeform")
            )
            # Free-text preferences have a separate confirmation lifecycle.
            # Consume only the store's active view so a quarantined or
            # tombstoned statement cannot reach the matching worker.
            versions.extend(
                self._career_profile_store.list_free_text_preferences(
                    user_id=user_id,
                    statuses=("active",),
                )
            )
        effective = resolve_effective_preferences(
            versions,
            context=PreferenceResolutionContext(
                target_role_id=resume.target_role_id,
                role_domains=tuple(
                    value
                    for value in (
                        preference_scope_domain(
                            target_role.title
                            if target_role is not None
                            else None
                        ),
                    )
                    if value is not None
                ),
                job_posting_id=job_posting_id,
            ),
        )
        effective_tracks = {
            (item.scope_key, item.pref_scope) for item in effective
        }
        versions = [
            item
            for item in versions
            if (item.scope_key, item.pref_scope) in effective_tracks
        ]
        tracks: dict[tuple[str, str], list] = {}
        for version in versions:
            tracks.setdefault(
                (version.scope_key, version.pref_scope),
                [],
            ).append(version)
        anchors: list[IntentStateAnchor] = []
        transitions: list[IntentStateTransition] = []
        for (scope_key, pref_scope), track in sorted(tracks.items()):
            ordered = sorted(track, key=lambda item: item.revision)
            current = next(
                (
                    item
                    for item in reversed(ordered)
                    if item.superseded_at is None
                ),
                None,
            )
            if current is None:
                continue
            # An aged preference is reported with its confirmation date, not
            # withheld. Dropping it here also removed its transition below,
            # which is the input stale-state repair needs most for old intent.
            anchors.append(
                IntentStateAnchor(
                    scope_key=scope_key,
                    pref_scope=pref_scope,
                    value=current.value,
                    valid_from=current.valid_from,
                    last_confirmed_at=current.last_corroborated_at,
                )
            )
            predecessors = [
                item for item in ordered if item.revision < current.revision
            ]
            if predecessors:
                previous = predecessors[-1]
                transitions.append(
                    IntentStateTransition(
                        scope_key=scope_key,
                        pref_scope=pref_scope,
                        old_value=previous.value,
                        new_value=current.value,
                        old_valid_from=previous.valid_from,
                        new_valid_from=current.valid_from,
                        last_confirmed_at=current.last_corroborated_at,
                    )
                )
        return tuple(anchors), tuple(transitions)

    def _repair_stale_state(
        self,
        *,
        result: ResumeJobMatchResult,
        jd_text: str,
        transitions: tuple[IntentStateTransition, ...],
    ) -> ResumeJobMatchResult:
        audit = getattr(self._worker, "audit_state", None)
        if not transitions or not callable(audit):
            return result
        proposal = ResumeJobMatchAuditProposal.model_validate(
            audit(
                draft=result,
                jd_text=jd_text,
                transitions=transitions,
            )
        )
        valid = {
            (
                item.scope_key,
                item.pref_scope,
                item.old_value,
                item.new_value,
            ): item
            for item in transitions
        }
        stale = [
            finding
            for finding in proposal.findings
            if finding.status == "stale"
            and finding.material
            and (
                finding.scope_key,
                finding.pref_scope,
                finding.old_value,
                finding.new_value,
            )
            in valid
        ]
        if not stale:
            return result
        notes = tuple(
            f"已按当前求职状态修正：{item.old_value} → {item.new_value}。"
            for item in stale
        )
        limitations = tuple(
            dict.fromkeys((*proposal.repaired_result.limitations, *notes))
        )[:10]
        return proposal.repaired_result.model_copy(
            update={
                # Intent is not evidence of ability. A repair may change the
                # recommendation, but it cannot rewrite resume/JD grounding or
                # create personalized follow-up questions.
                "requirements": result.requirements,
                "clarification_questions": result.clarification_questions,
                "limitations": limitations,
            }
        )
