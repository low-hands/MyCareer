"""External job-source adapters."""

from career_agent.jobs.providers.base import (
    JobProviderError,
    JobSearchProvider,
    ProviderSearchResult,
)
from career_agent.jobs.providers.boss_mcp import (
    BossMCPJobProvider,
    boss_provider_from_tools,
)

__all__ = [
    "BossMCPJobProvider",
    "JobProviderError",
    "JobSearchProvider",
    "ProviderSearchResult",
    "boss_provider_from_tools",
]
