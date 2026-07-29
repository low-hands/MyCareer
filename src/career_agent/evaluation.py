from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, Field

from career_agent.contracts import AgentRequest
from career_agent.runtime.base import AgentRuntime


class EvaluationCase(BaseModel):
    case_id: str
    prompt: str
    expected_terms: list[str] = Field(min_length=1)


class CaseResult(BaseModel):
    case_id: str
    passed: bool
    output: str
    matched_terms: list[str]


class EvaluationReport(BaseModel):
    total: int
    passed: int
    pass_rate: float
    cases: list[CaseResult]


class EvaluationRunner:
    """Deterministic first evaluator; model-judge adapters come later."""

    def __init__(self, runtime: AgentRuntime) -> None:
        self._runtime = runtime

    async def run(self, cases: Iterable[EvaluationCase]) -> EvaluationReport:
        results: list[CaseResult] = []
        for case in cases:
            response = await self._runtime.run(AgentRequest(message=case.prompt, thread_id=f"eval-{case.case_id}"))
            normalized = response.output.casefold()
            matched = [term for term in case.expected_terms if term.casefold() in normalized]
            results.append(
                CaseResult(
                    case_id=case.case_id,
                    passed=len(matched) == len(case.expected_terms),
                    output=response.output,
                    matched_terms=matched,
                )
            )

        passed = sum(result.passed for result in results)
        total = len(results)
        return EvaluationReport(
            total=total,
            passed=passed,
            pass_rate=passed / total if total else 0,
            cases=results,
        )
