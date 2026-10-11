"""Agent-test fixtures.

Shared doubles live in `agent_test_support`, not here: a `from conftest import`
resolves to whichever directory's conftest was collected first.
"""

from __future__ import annotations

import pytest

from career_agent.agent.capabilities.catalog import CAPABILITIES


_EXECUTION_TEST_MODULES = frozenset({
    "test_capture_source_turn", "test_job_capture_continuation",
    "test_main_agent_applications", "test_main_agent_email_tracking",
    "test_main_agent_interview_preparation", "test_main_agent_interviews",
    "test_main_agent_job_comparison", "test_main_agent_job_research",
    "test_main_agent_loop_evaluation", "test_main_agent_mock_interview",
    "test_main_agent_resume_tools", "test_main_agent_runtime",
    "test_resume_job_match", "test_tool_execution_trace",
    "test_trace_wiring_guard",
    "test_working_notes_guard",
})


@pytest.fixture(autouse=True)
def _isolate_runtime_execution_from_tool_discovery(request, monkeypatch):
    """Scripted execution tests receive their registered tools directly.

    Selection and reachability are exercised in their dedicated suites. These
    fixtures test execution, budgets, persistence, and presentation after a
    model has already chosen a tool.
    """
    if request.module.__name__.split(".")[-1] not in _EXECUTION_TEST_MODULES:
        return
    from career_agent.agent.capabilities import selection, selection_strategy
    from career_agent.agent.middleware import tool_availability

    monkeypatch.setattr(
        selection_strategy, "ALWAYS_OFFERED_TOOLS",
        tuple(name for name, descriptor in CAPABILITIES.items() if descriptor.model_callable),
    )
    if request.node.name == "test_internal_and_external_writes_draw_on_separate_budgets":
        fixture_reachable = lambda name, task: name not in {
            "execute_calendar_proposal", "get_calendar_proposal",
        }
    else:
        fixture_reachable = lambda name, task: True
    monkeypatch.setattr(selection, "reachable", fixture_reachable)
    monkeypatch.setattr(tool_availability, "reachable", fixture_reachable)
