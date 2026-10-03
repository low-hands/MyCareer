from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from career_agent.domain.job_research import JobResearchDraft, JobResearchScope


class JobResearchWorkerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    company_name: str = Field(min_length=1, max_length=500)
    role_title: str = Field(min_length=1, max_length=500)
    jd_text: str = Field(min_length=1, max_length=50_000)
    scope: JobResearchScope


class JobResearchWorker(Protocol):
    def research(
        self,
        *,
        run_id: str,
        request: JobResearchWorkerRequest,
        resume: bool = False,
    ) -> JobResearchDraft: ...

    def forget(self, run_id: str) -> None: ...
