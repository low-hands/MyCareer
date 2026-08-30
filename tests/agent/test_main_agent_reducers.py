import inspect

from career_agent.agent.main_agent_contracts import (
    CandidateContextItem,
    ConversationTaskState,
    EmailEventCandidateContextItem,
    ToolResult,
)
from career_agent.agent.main_agent_reducers import (
    ATOMIC_TASK_REDUCERS,
    reduce_task_state,
)
from career_agent.agent.main_agent_tools import MainAgentToolRegistry


def _fully_wired_registry() -> MainAgentToolRegistry:
    """Registration only checks for None, so sentinels expose every tool name."""
    sentinel = object()
    return MainAgentToolRegistry(
        **{
            name: sentinel
            for name in inspect.signature(
                MainAgentToolRegistry.__init__
            ).parameters
            if name not in {"self", "gateway"}
        },
    )


def test_every_reducer_is_keyed_to_a_real_capability() -> None:
    registry = _fully_wired_registry()
    known = set(registry.atomic_tool_names) | set(registry.workflow_names)

    # A reducer keyed to a name no tool reports would never fire, and the state
    # it guards would silently stop being maintained.
    assert set(ATOMIC_TASK_REDUCERS) <= known


def test_unregistered_tool_leaves_task_state_untouched() -> None:
    task = ConversationTaskState().enter_workflow(
        "job_discovery", run_id="run-1", phase="selection_required"
    )

    reduced = reduce_task_state(
        task, ToolResult(tool_name="get_job_analysis", state="analysis_ready", message="")
    )

    assert reduced == task


def test_reducer_skips_states_it_does_not_accept() -> None:
    task = ConversationTaskState(
        saved_job_candidates=(),
    )

    # A failed lookup must not blank the candidate list the user is choosing from.
    failed = reduce_task_state(
        task,
        ToolResult(tool_name="find_saved_jobs", state="failed", message="boom"),
    )

    assert failed.saved_job_candidates == ()
    found = reduce_task_state(
        task,
        ToolResult(
            tool_name="find_saved_jobs",
            state="saved_jobs_found",
            message="",
            payload={
                "items": [
                    {
                        "job_posting_id": "job-1",
                        "title": "AI Engineer",
                        "company_name": "Acme",
                    }
                ]
            },
        ),
    )
    assert [item.title for item in found.saved_job_candidates] == ["AI Engineer"]


def test_resolved_email_event_leaves_the_selectable_list() -> None:
    task = ConversationTaskState(
        email_event_candidates=(
            EmailEventCandidateContextItem(
                email_event_id="e1",
                event_type="interview_invite",
                status="pending_confirmation",
                summary="一面邀约",
            ),
            EmailEventCandidateContextItem(
                email_event_id="e2",
                event_type="rejection",
                status="pending_confirmation",
                summary="感谢投递",
            ),
        )
    )

    reduced = reduce_task_state(
        task,
        ToolResult(
            tool_name="resolve_email_event",
            state="email_event_resolved",
            message="",
            payload={"email_event_id": "e1", "status": "applied"},
        ),
    )

    # Leaving e1 in place would advertise a stale pending status and let index 1
    # resolve the same event twice.
    assert [item.email_event_id for item in reduced.email_event_candidates] == ["e2"]


def test_reducers_never_touch_the_workflow_slot() -> None:
    suspended = ConversationTaskState().enter_workflow(
        "job_discovery",
        run_id="run-1",
        phase="selection_required",
        candidates=(
            CandidateContextItem(
                result_ref="boss:1", title="AI Engineer", company_name="Acme"
            ),
        ),
    )

    # Atomic tools finish inside one turn, so none of them may evict a workflow
    # the user has not finished. Only enter_workflow/leave_workflow may.
    for name, entry in ATOMIC_TASK_REDUCERS.items():
        state = next(iter(entry.states), "ok")
        reduced = reduce_task_state(
            suspended, ToolResult(tool_name=name, state=state, message="")
        )
        slot = (reduced.active_workflow, reduced.run_id, reduced.phase)
        assert slot == ("job_discovery", "run-1", "selection_required"), name
        assert reduced.candidates == suspended.candidates, name
