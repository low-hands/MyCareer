"""Job discovery domain contracts and invariants."""

from .models import (
    ContractModel,
    DuplicateCandidate,
    DuplicateDecisionRequired,
    IncompleteJobDetail,
    JobDetail,
    JobPosting,
    JDSnapshot,
    Provenance,
    SearchResult,
    WaitlistItem,
    WaitlistStatus,
    company_title_fingerprint,
    content_fingerprint,
    jd_content_hash,
    new_id,
    normalize_jd,
    validate_job_detail,
)

__all__ = [
    "DuplicateCandidate",
    "JobDetail",
    "JobPosting",
    "JDSnapshot",
    "Provenance",
    "SearchResult",
    "WaitlistItem",
    "WaitlistStatus",
    "content_fingerprint",
    "normalize_jd",
    "validate_job_detail",
]
