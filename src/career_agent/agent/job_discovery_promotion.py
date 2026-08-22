from __future__ import annotations

import hashlib

from career_agent.agent.job_discovery_contracts import JobDiscoveryRecommendation, PendingPromotionConfirmation, TargetRoleProposal
from career_agent.domain.job_discovery import JobDetail, new_id
from career_agent.services.job_discovery import JobDiscoveryService, PromotionRequest, PromotionResult, PromotionSuccess


class JobDiscoveryPromotionFacade:
    def __init__(self, promotion_service: JobDiscoveryService) -> None:
        self._promotion_service = promotion_service
        self._pending: dict[str, PendingPromotionConfirmation] = {}

    def request_confirmation(
        self,
        *,
        user_id: str,
        conversation_id: str,
        recommendation: JobDiscoveryRecommendation,
        details: dict[str, JobDetail],
        result_ref: str,
        target_role: TargetRoleProposal,
    ) -> PendingPromotionConfirmation:
        if result_ref not in {item.result_ref for item in recommendation.items}:
            raise ValueError("Only a recommended job can be promoted.")
        detail = details.get(result_ref)
        if detail is None:
            raise ValueError("Recommended job detail is unavailable.")
        pending = PendingPromotionConfirmation(
            id=new_id("confirm"),
            user_id=user_id,
            conversation_id=conversation_id,
            result_ref=result_ref,
            detail=detail,
            target_role=target_role,
        )
        self._pending[pending.id] = pending
        return pending

    def confirm(self, *, user_id: str, conversation_id: str, confirmation_id: str) -> PromotionResult:
        pending = self._pending.get(confirmation_id)
        if pending is None or pending.user_id != user_id or pending.conversation_id != conversation_id:
            raise ValueError("Promotion confirmation is not active for this user and conversation.")
        source_key = pending.detail.source_job_id or pending.detail.security_id or pending.result_ref
        idempotency_key = hashlib.sha256(f"{user_id}:{source_key}:{pending.target_role.id}".encode()).hexdigest()
        result = self._promotion_service.confirm_and_add_to_waitlist(
            PromotionRequest(
                user_id=user_id,
                detail=pending.detail,
                target_role_id=pending.target_role.id,
                target_role_title=pending.target_role.title,
                idempotency_key=idempotency_key,
            )
        )
        if isinstance(result, PromotionSuccess):
            self._pending.pop(confirmation_id, None)
        return result
