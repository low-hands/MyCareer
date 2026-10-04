from __future__ import annotations

from career_agent.agent.execution.capability_executor import CapabilityExecutor
from career_agent.agent.contracts.observations import ToolObservation


def test_executor_uses_explicit_progress_and_runner_dependencies() -> None:
    events: list[tuple[str, ...]] = []

    class Tools:
        def invoke_atomic_tool(self, name, arguments):
            events.append(("invoke", name, arguments["query"]))
            return ToolObservation(
                tool_name=name,
                state="saved_jobs_ready",
                message="已读取岗位。",
            )

    def run_capability(pending, run):
        events.append(("run", pending["name"]))
        return run()

    executor = CapabilityExecutor(
        tools=Tools(),  # type: ignore[arg-type]
        action_execution_store=None,
        action_policy_epoch=1,
        emit_capability_started=lambda name: events.append(("started", name)),
        emit_capability_completed=lambda name, state: events.append(
            ("completed", name, state)
        ),
        run_capability=run_capability,
    )

    update = executor.act(
        {
            "pending": {
                "name": "find_saved_jobs",
                "kind": "atomic_tool",
                "effect": "READ",
                "arguments": {"query": "AI"},
            }
        }
    )

    assert events == [
        ("started", "find_saved_jobs"),
        ("run", "find_saved_jobs"),
        ("invoke", "find_saved_jobs", "AI"),
        ("completed", "find_saved_jobs", "saved_jobs_ready"),
    ]
    assert update["pending"]["result"].state == "saved_jobs_ready"
