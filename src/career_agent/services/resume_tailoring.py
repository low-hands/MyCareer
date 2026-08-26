from __future__ import annotations

from dataclasses import dataclass

from career_agent.agent.resume_job_match_contracts import ConfirmedResumeFact
from career_agent.agent.resume_tailoring_contracts import (
    AcceptedTailoringChange,
    ResumeFinalizationWorker,
    ResumeTailoringResult,
    ResumeTailoringReviewer,
    ResumeTailoringWorker,
)
from career_agent.agent.resume_tailoring_review_graph import ResumeTailoringReviewGraph
from career_agent.domain.resume import Resume, ResumeVersion
from career_agent.services.resume_job_match import ResumeJobMatchInputNotFoundError
from career_agent.storage.career_history import CareerHistoryStore
from career_agent.storage.jobs import JobPostingRepository
from career_agent.storage.resumes import ResumeStore
from career_agent.storage.resume_job_matches import SQLiteResumeJobMatchStore
from career_agent.storage.resume_tailoring import (
    ParentTailoringDraftNotRevisableError,
    SQLiteResumeTailoringDraftStore,
    StoredResumeTailoringDraft,
)


class ResumeTailoringDraftNotFoundError(ValueError):
    """Raised when a draft is missing, expired, or belongs to another user."""


class ResumeTailoringNotReadyError(ValueError):
    """Raised when finalization lacks complete explicit review decisions."""


class ResumeTailoringAlreadyFinalizedError(ValueError):
    """Raised when attempting to mutate an already materialized draft."""


class ResumeTailoringSupersededError(ValueError):
    """Raised when a newer child draft has replaced the requested draft."""


class ResumeTailoringReviewBlockedError(ValueError):
    """Raised when automated review cannot produce a safe user-reviewable draft."""


class ResumeFinalReviewBlockedError(ValueError):
    """Raised when final QA detects unapproved or unsafe materialization."""


@dataclass(frozen=True)
class ResumeTailoringFinalization:
    draft_id: str
    resume: Resume
    resume_version: ResumeVersion
    applied_change_indices: tuple[int, ...]
    warnings: tuple[str, ...]
    created: bool


class ResumeTailoringService:
    def __init__(
        self,
        resume_store: ResumeStore,
        job_repository: JobPostingRepository,
        career_history_store: CareerHistoryStore,
        match_store: SQLiteResumeJobMatchStore,
        draft_store: SQLiteResumeTailoringDraftStore,
        worker: ResumeTailoringWorker,
        finalization_worker: ResumeFinalizationWorker,
        *,
        reviewer: ResumeTailoringReviewer | None = None,
        max_review_revisions: int = 2,
        worker_version: str = "resume-tailoring-v1",
    ) -> None:
        self._resume_store = resume_store
        self._job_repository = job_repository
        self._career_history_store = career_history_store
        self._match_store = match_store
        self._draft_store = draft_store
        self._worker = worker
        self._finalization_worker = finalization_worker
        self._reviewer = reviewer
        self._review_graph = (
            ResumeTailoringReviewGraph(
                worker,
                reviewer,
                max_revisions=max_review_revisions,
            )
            if reviewer is not None
            else None
        )
        self._worker_version = worker_version

    def create_draft(
        self,
        *,
        user_id: str,
        match_id: str,
        tailoring_goal: str | None = None,
    ) -> StoredResumeTailoringDraft:
        if not user_id.strip() or not match_id.strip():
            raise ValueError("user_id and match_id are required")
        return self._generate_draft(
            user_id=user_id,
            match_id=match_id,
            tailoring_goal=tailoring_goal,
        )

    def revise_draft(
        self,
        *,
        user_id: str,
        draft_id: str,
        feedback: str,
    ) -> StoredResumeTailoringDraft:
        if not user_id.strip() or not draft_id.strip() or not feedback.strip():
            raise ValueError("user_id, draft_id, and feedback are required")
        current = self.get_draft(user_id=user_id, draft_id=draft_id)
        if current.status == "finalized":
            raise ResumeTailoringAlreadyFinalizedError(
                "A finalized draft cannot be revised; tailor its new resume version instead"
            )
        if current.status == "superseded":
            raise ResumeTailoringSupersededError(
                "A newer tailoring draft revision already exists"
            )
        try:
            return self._generate_draft(
                user_id=user_id,
                match_id=current.match_id,
                tailoring_goal=current.tailoring_goal,
                parent_draft_id=current.id,
                revision_number=current.revision_number + 1,
                revision_feedback=feedback.strip(),
                previous_draft=current.result,
            )
        except ParentTailoringDraftNotRevisableError as error:
            raise ResumeTailoringSupersededError(
                "A newer tailoring draft revision already exists"
            ) from error

    def _generate_draft(
        self,
        *,
        user_id: str,
        match_id: str,
        tailoring_goal: str | None,
        parent_draft_id: str | None = None,
        revision_number: int = 1,
        revision_feedback: str | None = None,
        previous_draft: ResumeTailoringResult | None = None,
    ) -> StoredResumeTailoringDraft:
        stored_match = self._match_store.get(user_id=user_id, match_id=match_id)
        if stored_match is None:
            raise ResumeJobMatchInputNotFoundError("match")
        document = self._resume_store.read_version_document(
            user_id=user_id,
            resume_version_id=stored_match.resume_version_id,
        )
        if document is None:
            raise ResumeJobMatchInputNotFoundError("resume_version")
        job = self._job_repository.get_job(
            user_id=user_id,
            job_posting_id=stored_match.job_posting_id,
        )
        if job is None or job.snapshot.id != stored_match.jd_snapshot_id:
            raise ResumeJobMatchInputNotFoundError("job_posting")
        confirmed_facts = tuple(
            ConfirmedResumeFact(
                claim=evidence.claim,
                source_locator=evidence.source_locator,
                source_quote=evidence.source_quote,
            )
            for evidence in self._career_history_store.list_evidence(
                user_id=user_id,
                verification_status="confirmed",
                source_resume_version_id=stored_match.resume_version_id,
            )
            if evidence.source_locator is not None and evidence.source_quote is not None
        )
        automated_review = None
        if self._review_graph is None:
            result = self._worker.tailor(
                document=document,
                jd_text=job.snapshot.content,
                match_result=stored_match.result,
                confirmed_facts=confirmed_facts,
                tailoring_goal=tailoring_goal,
                user_feedback=revision_feedback,
                previous_draft=previous_draft,
            )
        else:
            outcome = self._review_graph.run(
                document=document,
                jd_text=job.snapshot.content,
                match_result=stored_match.result,
                confirmed_facts=confirmed_facts,
                tailoring_goal=tailoring_goal,
                previous_draft=previous_draft,
                user_feedback=revision_feedback,
            )
            if outcome.trace.status != "passed":
                raise ResumeTailoringReviewBlockedError(
                    "Automated resume review stopped before a safe draft was produced: "
                    f"{outcome.trace.stop_reason}"
                )
            result = outcome.draft
            automated_review = outcome.trace
        return self._draft_store.create(
            user_id=user_id,
            match_id=match_id,
            parent_draft_id=parent_draft_id,
            revision_number=revision_number,
            revision_feedback=revision_feedback,
            tailoring_goal=tailoring_goal,
            worker_version=self._worker_version,
            result=result,
            automated_review=automated_review,
        )

    def get_draft(
        self,
        *,
        user_id: str,
        draft_id: str,
    ) -> StoredResumeTailoringDraft:
        draft = self._draft_store.get(user_id=user_id, draft_id=draft_id)
        if draft is None:
            raise ResumeTailoringDraftNotFoundError(
                "Resume tailoring draft not found, expired, or belongs to another user"
            )
        return draft

    def review_draft(
        self,
        *,
        user_id: str,
        draft_id: str,
        accepted_change_indices: tuple[int, ...] = (),
        rejected_change_indices: tuple[int, ...] = (),
        feedback: str | None = None,
    ) -> StoredResumeTailoringDraft:
        if not user_id.strip() or not draft_id.strip():
            raise ValueError("user_id and draft_id are required")
        current = self.get_draft(user_id=user_id, draft_id=draft_id)
        if current.status == "finalized":
            raise ResumeTailoringAlreadyFinalizedError(
                "Finalized tailoring decisions cannot be changed"
            )
        if current.status == "superseded":
            raise ResumeTailoringSupersededError(
                "A superseded tailoring draft cannot be reviewed"
            )
        draft = self._draft_store.review_changes(
            user_id=user_id,
            draft_id=draft_id,
            accepted_change_indices=accepted_change_indices,
            rejected_change_indices=rejected_change_indices,
            feedback=feedback,
        )
        if draft is None:
            raise ResumeTailoringDraftNotFoundError(
                "Resume tailoring draft not found, expired, or belongs to another user"
            )
        return draft

    def finalize_draft(
        self,
        *,
        user_id: str,
        draft_id: str,
    ) -> ResumeTailoringFinalization:
        draft = self.get_draft(user_id=user_id, draft_id=draft_id)
        if draft.status == "superseded":
            raise ResumeTailoringNotReadyError(
                "A newer tailoring draft revision must be reviewed instead"
            )
        if draft.status not in {"reviewed", "finalized"}:
            raise ResumeTailoringNotReadyError(
                "Every tailoring change must be explicitly accepted or rejected"
            )
        accepted = tuple(
            AcceptedTailoringChange(
                change_index=review.change_index,
                change=draft.result.changes[review.change_index - 1],
            )
            for review in draft.change_reviews
            if review.decision == "accepted"
        )
        if not accepted:
            raise ResumeTailoringNotReadyError(
                "At least one tailoring change must be accepted"
            )
        stored_match = self._match_store.get(user_id=user_id, match_id=draft.match_id)
        if stored_match is None:
            raise ResumeJobMatchInputNotFoundError("match")
        existing = self._resume_store.get_tailored_version(
            user_id=user_id,
            tailoring_draft_id=draft.id,
        )
        expected_indices = tuple(item.change_index for item in accepted)
        if existing is not None:
            self._draft_store.mark_finalized(user_id=user_id, draft_id=draft.id)
            return ResumeTailoringFinalization(
                draft_id=draft.id,
                resume=existing[0],
                resume_version=existing[1],
                applied_change_indices=expected_indices,
                warnings=(),
                created=False,
            )
        document = self._resume_store.read_version_document(
            user_id=user_id,
            resume_version_id=stored_match.resume_version_id,
        )
        if document is None:
            raise ResumeJobMatchInputNotFoundError("resume_version")
        confirmed_facts = tuple(
            ConfirmedResumeFact(
                claim=evidence.claim,
                source_locator=evidence.source_locator,
                source_quote=evidence.source_quote,
            )
            for evidence in self._career_history_store.list_evidence(
                user_id=user_id,
                verification_status="confirmed",
                source_resume_version_id=stored_match.resume_version_id,
            )
            if evidence.source_locator is not None and evidence.source_quote is not None
        )
        finalized = self._finalization_worker.finalize(
            document=document,
            accepted_changes=accepted,
            confirmed_facts=confirmed_facts,
        )
        if (
            len(set(finalized.applied_change_indices))
            != len(finalized.applied_change_indices)
            or set(finalized.applied_change_indices) != set(expected_indices)
        ):
            raise ValueError(
                "Finalized resume did not apply exactly the accepted changes"
            )
        if self._reviewer is not None:
            final_review = self._reviewer.review_final(
                document=document,
                accepted_changes=accepted,
                finalized=finalized,
                confirmed_facts=confirmed_facts,
            )
            if final_review.verdict != "pass":
                categories = sorted(
                    {issue.category for issue in final_review.issues}
                )
                raise ResumeFinalReviewBlockedError(
                    "Final resume QA found a substantive issue that requires a new user review"
                    + (f" ({', '.join(categories)})" if categories else "")
                )
        resume, version = self._resume_store.create_tailored_version(
            user_id=user_id,
            source_resume_version_id=stored_match.resume_version_id,
            tailoring_draft_id=draft.id,
            markdown=finalized.markdown,
        )
        self._draft_store.mark_finalized(user_id=user_id, draft_id=draft.id)
        return ResumeTailoringFinalization(
            draft_id=draft.id,
            resume=resume,
            resume_version=version,
            applied_change_indices=expected_indices,
            warnings=finalized.warnings,
            created=True,
        )
