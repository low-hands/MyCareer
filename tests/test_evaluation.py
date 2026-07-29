import pytest

from career_agent.contracts import AgentRequest, AgentResult
from career_agent.evaluation import EvaluationCase, EvaluationRunner


class StubRuntime:
    async def run(self, request: AgentRequest) -> AgentResult:
        return AgentResult(
            thread_id=request.thread_id,
            trace_id="trace",
            output="Python 匹配，但缺少 Kubernetes 经验。",
        )


@pytest.mark.asyncio
async def test_evaluation_runner_reports_pass_rate() -> None:
    runner = EvaluationRunner(StubRuntime())

    report = await runner.run(
        [
            EvaluationCase(
                case_id="match-1",
                prompt="分析岗位",
                expected_terms=["Python", "Kubernetes"],
            )
        ]
    )

    assert report.total == 1
    assert report.passed == 1
    assert report.pass_rate == 1
