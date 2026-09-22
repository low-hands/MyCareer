from __future__ import annotations

import hashlib
import json

from career_agent.agent.job_analysis_contracts import (
    ClassificationStatus,
    JobAnalysisResult,
    JobAnalysisWorker,
    RequirementTier,
)
from career_agent.domain.job_discovery import content_fingerprint
from career_agent.storage.jobs import (
    JDAnalysisPayload,
    JobPostingRepository,
    StoredJDAnalysis,
)

JOB_ANALYZER_VERSION = "job-analysis-v2"


class JobAnalysisInputNotFoundError(ValueError):
    """Raised when the saved job or pinned JD snapshot cannot be resolved."""

    def __init__(self, input_kind: str) -> None:
        self.input_kind = input_kind
        super().__init__(f"{input_kind} not found or does not belong to the current user")


class JobAnalysisRequirementNotFoundError(ValueError):
    """Raised when a correction does not address the selected analysis."""


class JobAnalysisService:
    """Analyzes one immutable JD snapshot and caches the result per snapshot."""

    def __init__(
        self,
        job_repository: JobPostingRepository,
        worker: JobAnalysisWorker,
        *,
        analyzer_version: str = JOB_ANALYZER_VERSION,
    ) -> None:
        self._job_repository = job_repository
        self._worker = worker
        self._analyzer_version = analyzer_version

    @property
    def analyzer_version(self) -> str:
        return self._analyzer_version

    def analyze(
        self,
        *,
        user_id: str,
        job_posting_id: str,
        jd_snapshot_id: str | None = None,
    ) -> StoredJDAnalysis:
        if not user_id.strip() or not job_posting_id.strip():
            raise ValueError("user_id and job_posting_id are required")
        job = self._job_repository.get_job(user_id=user_id, job_posting_id=job_posting_id)
        if job is None:
            raise JobAnalysisInputNotFoundError("job_posting")
        snapshot = job.snapshot
        if jd_snapshot_id is not None and jd_snapshot_id != snapshot.id:
            pinned = self._job_repository.get_snapshot(user_id=user_id, jd_snapshot_id=jd_snapshot_id)
            if pinned is None or pinned.job_posting_id != job.posting.id:
                raise JobAnalysisInputNotFoundError("jd_snapshot")
            snapshot = pinned
        fingerprint = content_fingerprint(
            job.posting.title, job.posting.company_name, snapshot.content
        )
        cached = self._job_repository.find_analysis(
            user_id=user_id,
            jd_snapshot_id=snapshot.id,
            analyzer_version=self._analyzer_version,
            content_fingerprint=fingerprint,
        )
        if cached is not None:
            latest = self._job_repository.get_analysis_for_snapshot(
                user_id=user_id,
                jd_snapshot_id=snapshot.id,
                analyzer_version=self._analyzer_version,
            )
            return latest or cached
        result = self._with_requirement_ids(
            self._worker.analyze(jd_text=snapshot.content),
            jd_snapshot_id=snapshot.id,
        )
        return self._job_repository.save_analysis(
            user_id=user_id,
            jd_snapshot_id=snapshot.id,
            analyzer_version=self._analyzer_version,
            analysis=JDAnalysisPayload.from_result(result),
            content_fingerprint=fingerprint,
        )

    def correct_requirement_tier(
        self,
        *,
        user_id: str,
        analysis_id: str,
        requirement_id: str,
        tier: RequirementTier,
        reason: str,
        classification_status: ClassificationStatus = "user_corrected",
    ) -> StoredJDAnalysis:
        """Persist a user confirmation/correction as a new analysis revision.

        The original analysis row is immutable. Matching reads the newest row
        for the snapshot, while historical reports continue to reference the
        original analysis ID.
        """
        if not reason.strip():
            raise ValueError("correction reason is required")
        stored = self._job_repository.get_analysis(user_id=user_id, analysis_id=analysis_id)
        if stored is None:
            raise JobAnalysisInputNotFoundError("job_analysis")
        result = stored.analysis.to_result()
        if result is None:
            raise ValueError("legacy analysis cannot be corrected before re-analysis")
        target = next(
            (item for item in result.requirements if item.requirement_id == requirement_id),
            None,
        )
        if target is None:
            raise JobAnalysisRequirementNotFoundError(requirement_id)
        already_corrected = target.classification_status == "user_corrected"
        if classification_status == "user_corrected" and tier == target.tier and not already_corrected:
            raise ValueError("a user correction must change the effective tier")
        if classification_status == "user_confirmed" and tier != target.tier:
            raise ValueError("user confirmation cannot change the effective tier")
        reverting_to_model = already_corrected and tier == target.model_tier
        effective_status = (
            "user_confirmed"
            if reverting_to_model
            else ("user_corrected" if already_corrected else classification_status)
        )
        effective_source = target.tier_source
        if reverting_to_model:
            effective_source = "model"
        elif effective_status == "user_corrected":
            effective_source = "user_corrected"
        corrected = target.model_copy(
            update={
                "tier": tier,
                "model_tier": target.model_tier or target.tier,
                "tier_source": effective_source,
                "tier_correction_reason": (
                    None if reverting_to_model else (
                        reason if (classification_status == "user_corrected" or already_corrected)
                        else None
                    )
                ),
                "model_tier_rationale": target.model_tier_rationale or target.tier_rationale,
                "tier_rationale": (
                    f"User {('reverted this requirement to the model tier' if reverting_to_model else ('corrected' if classification_status == 'user_corrected' else 'confirmed'))} "
                    f"this requirement as tier {tier}: {reason}"
                    if effective_status in {"user_confirmed", "user_corrected"}
                    else target.tier_rationale
                ),
                "classification_status": effective_status,
                "tier_confidence": "high" if effective_status in {"user_confirmed", "user_corrected"} else target.tier_confidence,
            }
        )
        revised = result.model_copy(
            update={
                "requirements": tuple(
                    corrected if item.requirement_id == requirement_id else item
                    for item in result.requirements
                )
            }
        )
        revision_fingerprint = hashlib.sha256(
            f"{stored.content_fingerprint}:tier:{requirement_id}:{tier}:{classification_status}:{reason}".encode()
        ).hexdigest()
        return self._job_repository.save_analysis(
            user_id=user_id,
            jd_snapshot_id=stored.jd_snapshot_id,
            analyzer_version=stored.analyzer_version,
            analysis=JDAnalysisPayload.from_result(revised),
            content_fingerprint=revision_fingerprint,
        )

    def _with_requirement_ids(
        self,
        result: JobAnalysisResult,
        *,
        jd_snapshot_id: str,
    ) -> JobAnalysisResult:
        requirements = []
        for index, requirement in enumerate(result.requirements, start=1):
            canonical = json.dumps(
                {
                    "snapshot": jd_snapshot_id,
                    "analyzer": self._analyzer_version,
                    "index": index,
                    "text": requirement.text,
                    "kind": requirement.kind,
                    "quote": requirement.jd_quote,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            requirement_id = "job_requirement_" + hashlib.sha256(
                canonical.encode("utf-8")
            ).hexdigest()[:20]
            requirements.append(
                requirement.model_copy(
                    update={
                        "requirement_id": requirement_id,
                        "model_tier": requirement.model_tier or requirement.tier,
                        "tier_source": requirement.tier_source,
                        "tier_rationale": requirement.tier_rationale
                        or "Tier classification requires review against the quoted JD language.",
                        "model_tier_rationale": requirement.model_tier_rationale
                        or requirement.tier_rationale
                        or "Tier classification requires review against the quoted JD language.",
                        "tier_evidence": requirement.tier_evidence or requirement.jd_quote,
                        "tier_confidence": requirement.tier_confidence,
                        "classification_status": requirement.classification_status,
                    }
                )
            )
        return result.model_copy(update={"requirements": tuple(requirements)})
