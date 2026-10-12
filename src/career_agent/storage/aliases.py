"""Compatibility imports for organization aliases now owned by the domain."""

from career_agent.domain.organization_aliases import (
    ORGANIZATION_ALIASES,
    canonical_organization_name,
    expand_organization_aliases,
)

__all__ = ["ORGANIZATION_ALIASES", "canonical_organization_name", "expand_organization_aliases"]
