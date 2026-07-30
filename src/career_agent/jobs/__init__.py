"""Job posting domain models and persistence."""

from career_agent.jobs.agent_tools import (
    JobDiscoveryToolContext,
    build_job_discovery_tools,
)
from career_agent.jobs.discovery_service import (
    DisplayedJob,
    JobDiscoveryService,
    NoSearchResults,
    PublishedSearchResults,
)
from career_agent.jobs.grounding_guard import JobGroundingGuard
from career_agent.jobs.models import JobDetailLevel, JobPosting
from career_agent.jobs.providers import (
    BossMCPJobProvider,
    JobProviderError,
    JobSearchProvider,
    ProviderSearchResult,
    boss_provider_from_tools,
)
from career_agent.jobs.repository import JobPostingNotFoundError, JobPostingRepository
from career_agent.jobs.search_models import (
    AmbiguousSearchResult,
    ResolvedSearchResult,
    SearchCriteria,
    SearchResolution,
    SearchResultCandidate,
    SearchResultNotFound,
    SearchResultSelector,
    SearchRun,
)
from career_agent.jobs.search_repository import (
    DuplicateSearchRunError,
    SearchRunRepository,
)
from career_agent.jobs.search_service import SearchRunService
from career_agent.jobs.search_sqlite_repo import SQLiteSearchRunRepository
from career_agent.jobs.sqlite_repo import SQLiteJobPostingRepository
from career_agent.jobs.wiring import (
    build_job_discovery_registry,
    build_job_discovery_service,
)

__all__ = [
    "AmbiguousSearchResult",
    "BossMCPJobProvider",
    "DisplayedJob",
    "DuplicateSearchRunError",
    "JobDetailLevel",
    "JobDiscoveryService",
    "JobGroundingGuard",
    "JobDiscoveryToolContext",
    "JobPosting",
    "JobPostingNotFoundError",
    "JobPostingRepository",
    "JobProviderError",
    "JobSearchProvider",
    "NoSearchResults",
    "ProviderSearchResult",
    "PublishedSearchResults",
    "ResolvedSearchResult",
    "SQLiteSearchRunRepository",
    "SQLiteJobPostingRepository",
    "SearchCriteria",
    "SearchResolution",
    "SearchResultCandidate",
    "SearchResultNotFound",
    "SearchResultSelector",
    "SearchRun",
    "SearchRunRepository",
    "SearchRunService",
    "boss_provider_from_tools",
    "build_job_discovery_registry",
    "build_job_discovery_service",
    "build_job_discovery_tools",
]
