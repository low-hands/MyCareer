from __future__ import annotations

import hashlib
import json

from career_agent.agent.job_analysis_contracts import TieredRequirement
from career_agent.agent.resume_job_match_contracts import (
    ConfirmedResumeFact,
    IntentStateAnchor,
    IntentStateTransition,
    ResumeJobMatchAuditProposal,
    ResumeJobMatchResult,
    ResumeJobMatchWorker,
    IntentAlignment,
)
from career_agent.agent.openai_compatible_client import AgentWorkerError
from career_agent.agent.resume_job_fit import derive_overall_fit
from career_agent.agent.resume_tailoring_review_graph import ResumeTailoringReviewGraph
from career_agent.services.job_analysis import JOB_ANALYZER_VERSION
from career_agent.storage.context import CareerProfileStore
from career_agent.storage.career_history import CareerHistoryStore
from career_agent.storage.jobs import JobPostingRepository
from career_agent.storage.resumes import ResumeStore, StoredResumeDocument
from career_agent.storage.resume_job_matches import (
    ResumeJobMatchInputs,
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


class ResumeJobMatchAnalysisRequiredError(ValueError):
    """The exact JD snapshot has no compatible authoritative requirement set."""

    def __init__(self, *, job_posting_id: str, jd_snapshot_id: str) -> None:
        self.job_posting_id = job_posting_id
        self.jd_snapshot_id = jd_snapshot_id
        super().__init__("a compatible analysis of the exact JD snapshot is required")


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
        matcher_version: str = "resume-job-match-v4",
        career_profile_store: CareerProfileStore | None = None,
        job_analyzer_version: str = JOB_ANALYZER_VERSION,
    ) -> None:
        self._resume_store = resume_store
        self._job_repository = job_repository
        self._career_history_store = career_history_store
        self._worker = worker
        self._match_store = match_store
        self._matcher_version = matcher_version
        self._career_profile_store = career_profile_store
        self._job_analyzer_version = job_analyzer_version

    def match(
        self,
        *,
        user_id: str,
        resume_version_id: str,
        job_posting_id: str,
        jd_snapshot_id: str | None = None,
    ) -> StoredResumeJobMatch:
        if not user_id.strip() or not resume_version_id.strip() or not job_posting_id.strip():
            raise ValueError("user_id, resume_version_id, and job_posting_id are required")
        document = self._resume_store.read_version_document(
            user_id=user_id,
            resume_version_id=resume_version_id,
        )
        if document is None:
            raise ResumeJobMatchInputNotFoundError("resume_version")
        source = self._resume_store.get_version(
            user_id=user_id, resume_version_id=resume_version_id
        )
        if source is None:
            raise ResumeJobMatchInputNotFoundError("resume_version")
        resume, version = source
        job = self._job_repository.get_job(
            user_id=user_id,
            job_posting_id=job_posting_id,
        )
        if job is None:
            raise ResumeJobMatchInputNotFoundError("job_posting")
        # A pinned JD version is the text the conversation has been talking
        # about; match against it, not against a later recapture of the
        # posting. A pin that no longer resolves for this posting is refused
        # instead of quietly reading the latest.
        if jd_snapshot_id is not None and jd_snapshot_id != job.snapshot.id:
            pinned = self._job_repository.get_snapshot(
                user_id=user_id, jd_snapshot_id=jd_snapshot_id
            )
            if pinned is None or pinned.job_posting_id != job.posting.id:
                raise ResumeJobMatchInputNotFoundError("jd_snapshot")
            job = job.model_copy(update={"snapshot": pinned, "analysis": None})

        stored_analysis = self._job_repository.get_analysis_for_snapshot(
            user_id=user_id,
            jd_snapshot_id=job.snapshot.id,
            analyzer_version=self._job_analyzer_version,
        )
        analysis = (
            stored_analysis.analysis.to_result()
            if stored_analysis is not None
            else None
        )
        if (
            stored_analysis is None
            or analysis is None
            or not analysis.requirements
            or any(item.requirement_id is None for item in analysis.requirements)
            or len({item.requirement_id for item in analysis.requirements})
            != len(analysis.requirements)
        ):
            raise ResumeJobMatchAnalysisRequiredError(
                job_posting_id=job_posting_id,
                jd_snapshot_id=job.snapshot.id,
            )

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
            job_analysis_id=stored_analysis.id,
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
            tiered_requirements=analysis.requirements,
        )
        result = self._bind_and_grade_requirements(
            result=result,
            requirements=analysis.requirements,
        )
        self._verify_resume_evidence(document=document, result=result)
        result = self._repair_stale_state(
            result=result,
            jd_text=job.snapshot.content,
            transitions=transitions,
        )
        result = self._bind_and_grade_requirements(
            result=result,
            requirements=analysis.requirements,
        )
        self._verify_resume_evidence(document=document, result=result)
        return self._match_store.save(
            user_id=user_id,
            resume_version_id=resume_version_id,
            job_posting_id=job_posting_id,
            jd_snapshot_id=job.snapshot.id,
            matcher_version=self._matcher_version,
            evidence_fingerprint=evidence_fingerprint,
            result=result,
            inputs=ResumeJobMatchInputs(
                job_title=job.posting.title,
                company_name=job.posting.company_name,
                resume_id=resume.id,
                resume_name=resume.name,
                resume_version_number=version.version_number,
                resume_created_at=version.created_at,
                jd_version=job.snapshot.version,
                jd_captured_at=job.snapshot.captured_at,
            ),
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
        *,
        job_analysis_id: str = "",
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
                "job_analysis_id": job_analysis_id,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _bind_and_grade_requirements(
        *,
        result: ResumeJobMatchResult,
        requirements: tuple[TieredRequirement, ...],
    ) -> ResumeJobMatchResult:
        authoritative = {
            item.requirement_id: item
            for item in requirements
            if item.requirement_id is not None
        }
        seen: set[str] = set()
        projected = []
        for assessment in result.requirements:
            requirement_id = assessment.requirement_id
            source = authoritative.get(requirement_id)
            if source is None or requirement_id in seen:
                raise AgentWorkerError(
                    "RESUME_JOB_MATCH_REQUIREMENT_BINDING_INVALID",
                    "Requirement assessment did not bind to the supplied analysis.",
                    retryable=True,
                )
            if (
                assessment.requirement != source.text
                or assessment.jd_quote != source.jd_quote
            ):
                raise AgentWorkerError(
                    "RESUME_JOB_MATCH_REQUIREMENT_BINDING_INVALID",
                    "Requirement assessment altered authoritative requirement text.",
                    retryable=True,
                )
            seen.add(requirement_id)
            projected.append(
                assessment.model_copy(
                    update={
                        "requirement": source.text,
                        "jd_quote": source.jd_quote,
                        "tier": source.tier,
                        "kind": source.kind,
                        "tier_confidence": source.tier_confidence,
                        "classification_status": source.classification_status,
                        "tier_rationale": source.tier_rationale,
                        "tier_evidence": source.tier_evidence,
                    }
                )
            )
        if seen != set(authoritative):
            raise AgentWorkerError(
                "RESUME_JOB_MATCH_REQUIREMENT_BINDING_INVALID",
                "Requirement assessment omitted an authoritative requirement.",
                retryable=True,
            )
        projected_requirements = tuple(projected)
        fit = derive_overall_fit(tuple(requirements), projected_requirements)
        return result.model_copy(
            update={
                "overall_fit": fit,
                "requirements": projected_requirements,
                "intent_alignment": result.intent_alignment
                or IntentAlignment(
                    status="unknown",
                    rationale="当前结果未提供足够的意向约束来单独判断岗位匹配度。",
                ),
            }
        )

    @staticmethod
    def _verify_resume_evidence(
        *, document: StoredResumeDocument, result: ResumeJobMatchResult
    ) -> None:
        """Positive matches must cite the exact resume version, including PDF pages."""
        for assessment in result.requirements:
            for evidence in assessment.resume_evidence:
                check = ResumeTailoringReviewGraph._check_evidence(
                    document=document,
                    quote=ResumeTailoringReviewGraph._normalize(evidence.source_quote),
                    declared_quality="exact",
                    page=evidence.page,
                )
                if not check.matched:
                    raise AgentWorkerError(
                        "RESUME_JOB_MATCH_EVIDENCE_UNVERIFIED",
                        "A positive requirement assessment cited an unverified resume quote.",
                        retryable=True,
                    )

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
