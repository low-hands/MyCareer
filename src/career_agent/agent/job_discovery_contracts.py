from __future__ import annotations

from typing import Any, Literal, Protocol, TypeVar

from pydantic import Field, model_validator

from career_agent.domain.job_discovery import ContractModel, JobDetail

T = TypeVar("T", bound=ContractModel)


class AgentWorker(Protocol):
    def decide(self, *, stage: str, input: dict[str, Any], output_type: type[T]) -> T: ...


class TargetRoleProposal(ContractModel):
    id: str
    title: str = Field(min_length=1)
    source: Literal["explicit", "agent_suggested"]
    rationale: str


class QueryProposal(ContractModel):
    query: str = Field(min_length=1)
    rationale: str


class SearchStrategy(ContractModel):
    target_role: TargetRoleProposal
    queries: tuple[QueryProposal, ...]
    city: str | None = None
    salary: str | None = None
    experience: str | None = None
    education: str | None = None


class JobDiscoveryRequest(ContractModel):
    user_id: str
    conversation_id: str
    target_role: str = Field(min_length=1)
    resume_text: str | None = None
    city: str | None = None
    salary: str | None = None
    experience: str | None = None
    education: str | None = None


class TriageSelection(ContractModel):
    result_ref: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


class CandidateTriage(ContractModel):
    selections: tuple[TriageSelection, ...]

    @model_validator(mode="after")
    def validate_selections(self) -> "CandidateTriage":
        refs = [selection.result_ref for selection in self.selections]
        if len(refs) != len(set(refs)):
            raise ValueError("triage result references must be unique")
        return self


class JDAnalysis(ContractModel):
    result_ref: str
    job_summary: str
    responsibilities: tuple[str, ...] = ()
    required_skills: tuple[str, ...] = ()
    preferred_qualifications: tuple[str, ...] = ()
    clarification_questions: tuple[str, ...] = ()


class JDComparison(ContractModel):
    common_requirements: tuple[str, ...] = ()
    key_differences: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()


class RecommendationItem(ContractModel):
    result_ref: str
    rank: int = Field(ge=1)
    rationale: str
    cautions: tuple[str, ...] = ()


class PendingPromotionConfirmation(ContractModel):
    id: str
    user_id: str
    conversation_id: str
    result_ref: str
    detail: JobDetail
    target_role: TargetRoleProposal


class JobDiscoveryRecommendation(ContractModel):
    summary: str
    items: tuple[RecommendationItem, ...] = ()
