from datetime import datetime, timezone

import pytest

from career_agent.agent.job_discovery_contracts import JobDiscoveryRecommendation, RecommendationItem, TargetRoleProposal
from career_agent.agent.job_discovery_promotion import JobDiscoveryPromotionFacade
from career_agent.domain.job_discovery import JobDetail, Provenance
from career_agent.services.job_discovery import JobDiscoveryService, PromotionSuccess
from career_agent.storage.memory import InMemoryJobRepository

NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)


def detail() -> JobDetail:
    return JobDetail(source_name="boss", source_job_id="job-a", security_id="security-a", title="LLM Engineer", company_name="Acme", description="Build LLM systems.", captured_at=NOW, provenance=Provenance(source_name="boss", captured_at=NOW, operation="detail", adapter_version="test"))


def test_recommendation_requires_confirmation_before_promotion() -> None:
    repository = InMemoryJobRepository()
    facade = JobDiscoveryPromotionFacade(JobDiscoveryService(repository))
    recommendation = JobDiscoveryRecommendation(summary="one", items=(RecommendationItem(result_ref="a", rank=1, rationale="relevant"),))
    target = TargetRoleProposal(id="role-ai", title="AI Engineer", source="explicit", rationale="user target")

    pending = facade.request_confirmation(user_id="u", conversation_id="c", recommendation=recommendation, details={"a": detail()}, result_ref="a", target_role=target)
    assert not repository.waitlist

    result = facade.confirm(user_id="u", conversation_id="c", confirmation_id=pending.id)
    assert isinstance(result, PromotionSuccess)
    assert len(repository.waitlist) == 1


def test_non_recommended_or_cross_user_confirmation_is_rejected() -> None:
    facade = JobDiscoveryPromotionFacade(JobDiscoveryService(InMemoryJobRepository()))
    recommendation = JobDiscoveryRecommendation(summary="one", items=())
    target = TargetRoleProposal(id="role-ai", title="AI Engineer", source="explicit", rationale="user target")
    with pytest.raises(ValueError, match="recommended"):
        facade.request_confirmation(user_id="u", conversation_id="c", recommendation=recommendation, details={"a": detail()}, result_ref="a", target_role=target)
