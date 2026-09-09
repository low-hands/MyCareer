from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

from career_agent.agent.conversation_memory_contracts import ConversationSummaryContent
from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.agent.openai_compatible_client import AgentWorkerError
from career_agent.agent.main_agent_contracts import (
    AgentPreferencesContext,
    CareerProfileBudgets,
    CareerProfileContext,
    ConversationResourceReference,
    ConversationTaskState,
    HardConstraintContext,
    confirmation_recency_label,
)
from career_agent.harness.observability import (
    ACTIVE_TRACE_CONTEXT,
    InMemoryTraceRecorder,
    conversation_trace_key,
)
from career_agent.storage.context import CareerContextStore
from career_agent.storage.resumes import ResumeStore


def manager(
    tmp_path,
    *,
    limit: int = 4,
    summary_worker=None,
    summary_batch_size: int = 2,
    max_recent_context_chars: int = 16000,
    compact_occupancy_threshold: float = 0.75,
) -> ContextManager:
    return ContextManager(
        CareerContextStore(tmp_path / "context.sqlite3"),
        summary_worker=summary_worker,
        recent_message_limit=limit,
        summary_batch_size=summary_batch_size,
        max_message_chars=32,
        max_recent_context_chars=max_recent_context_chars,
        compact_occupancy_threshold=compact_occupancy_threshold,
    )


def test_configured_career_profile_budgets_reach_all_context_envelopes(
    tmp_path,
) -> None:
    budgets = CareerProfileBudgets(
        records_input_units=512,
        current_targets_input_units=256,
        hard_constraints_input_units=128,
    )
    context_manager = ContextManager(
        CareerContextStore(tmp_path / "context.sqlite3"),
        career_profile_budgets=budgets,
    )

    turn = context_manager.load_for_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="memory",
    )
    workflow = context_manager.load_for_workflow_turn(
        user_id="u1",
        conversation_id="c1",
        task=ConversationTaskState(
            active_workflow="mock_interview",
            run_id="mock-1",
        ),
    )

    assert turn.career_profile_budgets == budgets
    assert workflow.career_profile_budgets == budgets


def test_profile_current_target_block_is_rendered_from_target_role_source(
    tmp_path,
) -> None:
    context_store = CareerContextStore(tmp_path / "context.sqlite3")
    context_store.upsert_profile(
        CareerProfileContext(user_id="u1", default_city="杭州")
    )
    resumes = ResumeStore(tmp_path / "resumes.sqlite3")
    role = resumes.create_target_role(
        user_id="u1",
        title="ML Engineer",
        priority=1,
    )
    resumes.update_target_role_intent(
        user_id="u1",
        target_role_id=role.id,
        city="上海",
        salary_expectation="40-50k",
        experience="5-7 years",
        education="硕士",
    )
    context_manager = ContextManager(
        context_store,
        target_role_source=resumes,
    )

    context = context_manager.load_for_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="我当前的求职目标是什么？",
    )
    projected = context.model_context()["career_profile"]
    targets = projected["memory/current_targets.md"]

    assert '- Title: "ML Engineer"' in targets
    assert "- Priority: 1" in targets
    assert '- City: "上海"' in targets
    assert '- Salary expectation: "40-50k"' in targets
    assert '- Experience: "5-7 years"' in targets
    assert '- Education: "硕士"' in targets
    assert targets.count("- Last confirmed: 今天确认") == 4
    assert "current_targets" not in context.profile.model_dump()
    workflow_context = context_manager.load_for_workflow_turn(
        user_id="u1",
        conversation_id="c1",
        task=ConversationTaskState(
            active_workflow="mock_interview",
            run_id="mock-1",
        ),
    )
    assert workflow_context.profile.current_targets == ()


def test_projection_keeps_stale_intent_and_labels_last_confirmation(
    tmp_path,
) -> None:
    context_path = tmp_path / "context.sqlite3"
    resume_path = tmp_path / "resumes.sqlite3"
    context_store = CareerContextStore(context_path)
    constraint = HardConstraintContext(
        relation="work_arrangement",
        value="必须远程",
    )
    context_store.upsert_profile(
        CareerProfileContext(
            user_id="u1",
            default_city="杭州",
            hard_constraints=(constraint,),
        )
    )
    resumes = ResumeStore(resume_path)
    role = resumes.create_target_role(
        user_id="u1",
        title="ML Engineer",
        priority=1,
    )
    resumes.update_target_role_intent(
        user_id="u1",
        target_role_id=role.id,
        salary_expectation="40K",
    )
    stale_at = datetime.now(timezone.utc) - timedelta(days=200)
    stale = stale_at.isoformat()
    with sqlite3.connect(context_path) as connection:
        connection.execute(
            "UPDATE career_intent_versions SET last_corroborated_at = ?",
            (stale,),
        )
    with sqlite3.connect(resume_path) as connection:
        connection.execute(
            "UPDATE career_intent_versions SET last_corroborated_at = ?",
            (stale,),
        )

    projected = ContextManager(
        context_store,
        target_role_source=resumes,
    ).load_for_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="推荐岗位",
    )

    career_profile = projected.model_context()["career_profile"]
    assert projected.profile.default_city == "杭州"
    assert projected.profile.hard_constraints[0].relation == constraint.relation
    assert projected.profile.hard_constraints[0].value == constraint.value
    assert projected.profile.current_targets[0].salary_expectation == "40K"
    stale_label = confirmation_recency_label(stale_at)
    profile_file = career_profile["memory/profile.md"]
    targets_file = career_profile["memory/current_targets.md"]
    assert '- Default city: "杭州"' in profile_file
    assert f"- Last confirmed: {stale_label}" in profile_file
    assert '- Salary expectation: "40K"' in targets_file
    assert f"- Last confirmed: {stale_label}" in targets_file


class RecordingSummaryWorker:
    def __init__(self) -> None:
        self.calls = []

    def summarize(self, *, previous, messages):
        self.calls.append((previous, messages))
        decisions = previous.confirmed_decisions if previous else ()
        # Keep the newest few. A real worker consolidates rather than appends,
        # and the contract caps this tuple, so an ever-growing double would fail
        # on its own accumulation instead of on the behaviour under test.
        decisions = decisions[-4:]
        return ConversationSummaryContent(
            user_goals=("Maintain conversation continuity",),
            confirmed_decisions=(*decisions, f"covered-through-{messages[-1].sequence}"),
            unresolved_questions=(),
            active_constraints=("Do not promote this summary to career facts",),
        )


@pytest.mark.parametrize("threshold", [0.69, 0.91])
def test_compaction_occupancy_threshold_stays_in_the_measured_band(
    tmp_path, threshold
) -> None:
    with pytest.raises(ValueError, match="between 0.7 and 0.9"):
        manager(tmp_path, compact_occupancy_threshold=threshold)


def test_loads_profile_preferences_task_and_bounded_history(tmp_path) -> None:
    context_manager = manager(tmp_path)
    context_manager.upsert_profile(CareerProfileContext(user_id="u1", default_city="Shanghai"))
    context_manager.upsert_preferences(user_id="u1", preferences=AgentPreferencesContext(boss_search="allowed"))
    initial = context_manager.load_for_turn(user_id="u1", conversation_id="c1", user_message="First message")
    context_manager.commit_turn(context=initial, task=ConversationTaskState(active_workflow="job_discovery", run_id="run-1", phase="selection_required"), assistant_message="First response")

    loaded = context_manager.load_for_turn(user_id="u1", conversation_id="c1", user_message="Second message")

    assert loaded.profile.default_city == "Shanghai"
    assert loaded.preferences.boss_search == "allowed"
    assert loaded.task.run_id == "run-1"
    assert [message.content for message in loaded.recent_messages] == ["First message", "First response"]
    assert loaded.through_sequence == 0
    assert loaded.recent_from_sequence == 1
    assert "through_sequence" not in loaded.model_context()
    assert "recent_from_sequence" not in loaded.model_context()
    assert loaded.user_message == "Second message"


def test_spotlight_nonce_is_stable_for_the_durable_session(tmp_path) -> None:
    first = manager(tmp_path).load_for_turn(
        user_id="u1", conversation_id="c1", user_message="first"
    )
    rebuilt = manager(tmp_path).load_for_turn(
        user_id="u1", conversation_id="c1", user_message="second"
    )
    other = manager(tmp_path).load_for_turn(
        user_id="u1", conversation_id="c2", user_message="other"
    )

    assert first.spotlight_nonce == rebuilt.spotlight_nonce
    assert first.spotlight_nonce != other.spotlight_nonce


def test_workflow_turn_updates_routing_without_loading_or_writing_main_memory(
    tmp_path,
) -> None:
    worker = RecordingSummaryWorker()
    context_manager = manager(tmp_path, limit=2, summary_worker=worker)
    initial = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="Main Agent message"
    )
    context_manager.commit_turn(
        context=initial,
        task=ConversationTaskState(),
        assistant_message="Main Agent response",
    )
    task = ConversationTaskState(
        active_workflow="mock_interview",
        run_id="mock-session-1",
        phase="mock_interview_answer_required",
    )

    workflow_context = context_manager.load_for_workflow_turn(
        user_id="u1",
        conversation_id="c1",
        task=task,
    )
    context_manager.commit_workflow_turn(
        context=workflow_context,
        task=task.model_copy(update={"phase": "mock_interview_running"}),
    )

    assert workflow_context.recent_messages == ()
    assert workflow_context.conversation_summary is None
    assert "Private interview answer" not in workflow_context.user_message
    assert worker.calls == []
    loaded = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="Back to main"
    )
    assert [message.content for message in loaded.recent_messages] == [
        "Main Agent message",
        "Main Agent response",
    ]
    assert loaded.task.phase == "mock_interview_running"


def test_context_isolated_by_user_and_conversation(tmp_path) -> None:
    context_manager = manager(tmp_path)
    context_manager.upsert_profile(CareerProfileContext(user_id="u1"))
    first = context_manager.load_for_turn(user_id="u1", conversation_id="same", user_message="u1")
    context_manager.commit_turn(context=first, task=ConversationTaskState(active_workflow="job_discovery", run_id="run-u1"), assistant_message="done")

    other_user = context_manager.load_for_turn(user_id="u2", conversation_id="same", user_message="u2")
    other_conversation = context_manager.load_for_turn(user_id="u1", conversation_id="other", user_message="other")

    assert other_user.profile.default_city is None
    assert other_user.task.run_id is None
    assert other_user.recent_messages == ()
    assert other_conversation.task.run_id is None
    assert other_conversation.recent_messages == ()


def test_commit_trims_messages_and_survives_manager_rebuild(tmp_path) -> None:
    first = manager(tmp_path, limit=2)
    for index in range(2):
        context = first.load_for_turn(user_id="u1", conversation_id="c1", user_message=f"user-{index}")
        first.commit_turn(context=context, task=ConversationTaskState(), assistant_message=f"assistant-{index}")

    rebuilt = manager(tmp_path, limit=2)
    loaded = rebuilt.load_for_turn(user_id="u1", conversation_id="c1", user_message="next")

    assert [message.content for message in loaded.recent_messages] == ["user-1", "assistant-1"]


def test_messages_are_truncated_without_profile_mutation(tmp_path) -> None:
    context_manager = manager(tmp_path)
    context_manager.upsert_profile(CareerProfileContext(user_id="u1", default_city="Shanghai"))
    context = context_manager.load_for_turn(user_id="u1", conversation_id="c1", user_message="x" * 100)
    context_manager.commit_turn(context=context, task=ConversationTaskState(), assistant_message="y" * 100)

    loaded = context_manager.load_for_turn(user_id="u1", conversation_id="c1", user_message="next")

    assert [len(message.content) for message in loaded.recent_messages] == [32, 32]
    assert loaded.profile.default_city == "Shanghai"


def test_short_chat_below_occupancy_keeps_raw_messages_without_summary(
    tmp_path,
) -> None:
    worker = RecordingSummaryWorker()
    context_manager = manager(tmp_path, limit=4, summary_worker=worker)
    for index in range(2):
        context = context_manager.load_for_turn(
            user_id="u1",
            conversation_id="c1",
            user_message=f"user-{index}",
        )
        context_manager.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message=f"assistant-{index}",
        )

    assert worker.calls == []
    assert context_manager._store.get_conversation_summary(
        user_id="u1", conversation_id="c1"
    ) is None
    assert len(
        context_manager._store.list_messages_after(
            user_id="u1",
            conversation_id="c1",
            after_sequence=0,
            limit=10,
        )
    ) == 4


def test_rolls_old_messages_into_structured_summary_and_keeps_recent_raw_window(
    tmp_path,
) -> None:
    worker = RecordingSummaryWorker()
    context_manager = manager(
        tmp_path,
        limit=4,
        summary_worker=worker,
        max_recent_context_chars=60,
    )
    for index in range(3):
        context = context_manager.load_for_turn(
            user_id="u1",
            conversation_id="c1",
            user_message=f"user-{index}",
        )
        context_manager.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message=f"assistant-{index}",
        )

    loaded = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="next"
    )
    stored = context_manager._store.get_conversation_summary(
        user_id="u1", conversation_id="c1"
    )

    assert stored is not None
    assert stored.through_sequence == 2
    assert loaded.conversation_summary == stored.content
    assert loaded.through_sequence == 2
    assert loaded.recent_from_sequence == 3
    assert loaded.model_context()["through_sequence"] == 2
    assert loaded.model_context()["recent_from_sequence"] == 3
    assert [message.content for message in loaded.recent_messages] == [
        "user-1",
        "assistant-1",
        "user-2",
        "assistant-2",
    ]
    assert [message.sequence for message in worker.calls[0][1]] == [1, 2]
    # Summarised rows are skipped, not deleted. A summary is a model output, so
    # while it is the only thing read, it must not be the only thing kept: if it
    # loses or distorts a turn, the original is the only way to find out.
    remaining = context_manager._store.list_messages_after(
        user_id="u1",
        conversation_id="c1",
        after_sequence=0,
        limit=10,
    )
    assert [message.sequence for message in remaining] == [1, 2, 3, 4, 5, 6]


def test_occupancy_compaction_records_its_trigger_without_raw_arguments(
    tmp_path,
) -> None:
    worker = RecordingSummaryWorker()
    context_manager = manager(
        tmp_path,
        limit=4,
        summary_worker=worker,
        max_recent_context_chars=60,
    )
    recorder = InMemoryTraceRecorder()
    token = ACTIVE_TRACE_CONTEXT.set((recorder, "turn-1"))
    try:
        for index in range(2):
            context = context_manager.load_for_turn(
                user_id="u1",
                conversation_id="c1",
                user_message=f"user-{index}",
            )
            context_manager.commit_turn(
                context=context,
                task=ConversationTaskState(),
                assistant_message="a" * 20,
            )
    finally:
        ACTIVE_TRACE_CONTEXT.reset(token)

    compacted = [
        event
        for event in recorder.snapshot("turn-1").events
        if event.event_type == "context_compacted"
    ]
    assert len(compacted) == 1
    assert compacted[0].details == {
        "conversation_id": "c1",
        "conversation_key": conversation_trace_key("u1", "c1"),
        "trigger": "occupancy",
        "through_sequence": 2,
        "occupancy": 52 / 60,
        "projection_overflow": False,
        "input_occupancy_numerator": None,
        "input_occupancy_denominator": None,
        "restored_constraints": 0,
        "dropped_constraints": 0,
        "readmitted_constraints": 0,
        "dropped_user_goals": 0,
        "dropped_confirmed_decisions": 0,
        "dropped_unresolved_questions": 0,
        "omitted_active_constraint_count": 0,
        "omitted_user_goal_count": 0,
        "omitted_confirmed_decision_count": 0,
        "omitted_unresolved_question_count": 0,
        "batch_size": 2,
    }


def test_complete_request_pressure_can_exceed_one_and_trigger_compaction(
    tmp_path,
) -> None:
    worker = RecordingSummaryWorker()
    context_manager = manager(
        tmp_path,
        limit=4,
        summary_worker=worker,
        max_recent_context_chars=16000,
    )
    context_manager.configure_request_token_estimator(
        lambda context: (3000, 1000)
    )
    recorder = InMemoryTraceRecorder()
    token = ACTIVE_TRACE_CONTEXT.set((recorder, "turn-token-pressure"))
    try:
        context = context_manager.load_for_turn(
            user_id="u1", conversation_id="c1", user_message="short"
        )
        context_manager.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message="short",
        )
    finally:
        ACTIVE_TRACE_CONTEXT.reset(token)

    assert len(worker.calls) == 1
    event = next(
        event
        for event in recorder.snapshot("turn-token-pressure").events
        if event.event_type == "context_compacted"
    )
    assert event.details["occupancy"] == 3.0
    assert event.details["input_occupancy_numerator"] == 3000
    assert event.details["input_occupancy_denominator"] == 1000


def test_static_request_over_half_the_input_budget_fails_during_wiring(
    tmp_path,
) -> None:
    context_manager = manager(tmp_path)

    with pytest.raises(ValueError, match="more than 50%"):
        context_manager.configure_request_token_estimator(
            lambda context: (8000, 10000),
            static_input_tokens=5001,
            max_input_tokens=10000,
        )


def test_normal_turns_catch_up_a_preexisting_summary_backlog(tmp_path) -> None:
    path = tmp_path / "context.sqlite3"
    writer = ContextManager(CareerContextStore(path), recent_message_limit=40)
    for index in range(10):
        context = writer.load_for_turn(
            user_id="u1", conversation_id="c1", user_message=f"user-{index}"
        )
        writer.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message=f"assistant-{index}",
        )

    worker = RecordingSummaryWorker()
    context_manager = ContextManager(
        CareerContextStore(path),
        summary_worker=worker,
        recent_message_limit=4,
        summary_batch_size=4,
    )
    context_manager.configure_request_token_estimator(
        lambda context: (1000, 1000)
    )
    for index in range(3):
        context = context_manager.load_for_turn(
            user_id="u1", conversation_id="c1", user_message=f"new-{index}"
        )
        context_manager.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message=f"answer-{index}",
        )

    summary = context_manager._store.get_conversation_summary(
        user_id="u1", conversation_id="c1"
    )
    assert summary is not None
    total_messages = context_manager._store.list_messages_after(
        user_id="u1", conversation_id="c1", after_sequence=0, limit=100
    )
    assert len(worker.calls) == 6
    assert len(total_messages) - summary.through_sequence == 2


def test_one_compaction_call_advances_only_one_batch_under_large_backlog(
    tmp_path,
) -> None:
    path = tmp_path / "bounded-backlog.sqlite3"
    writer = ContextManager(CareerContextStore(path), recent_message_limit=40)
    for index in range(10):
        context = writer.load_for_turn(
            user_id="u1", conversation_id="c1", user_message=f"user-{index}"
        )
        writer.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message=f"assistant-{index}",
        )

    worker = RecordingSummaryWorker()
    context_manager = ContextManager(
        CareerContextStore(path),
        summary_worker=worker,
        recent_message_limit=4,
        summary_batch_size=4,
    )
    context_manager.configure_request_token_estimator(
        lambda context: (1000, 1000)
    )

    context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="first"
    )
    first = context_manager._store.get_conversation_summary(
        user_id="u1", conversation_id="c1"
    )
    assert first is not None
    assert first.through_sequence == 4
    assert len(worker.calls) == 1
    occupancy, projection_overflow, _, _ = context_manager._recent_pressure(
        user_id="u1",
        conversation_id="c1",
        after_sequence=first.through_sequence,
        user_message="probe",
    )
    assert occupancy == 1.0
    assert projection_overflow is True

    context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="second"
    )
    second = context_manager._store.get_conversation_summary(
        user_id="u1", conversation_id="c1"
    )
    assert second is not None
    assert second.through_sequence == 8
    assert len(worker.calls) == 2


def test_default_short_chat_compacts_when_unsummarized_rows_leave_projection(
    tmp_path,
) -> None:
    worker = RecordingSummaryWorker()
    context_manager = ContextManager(
        CareerContextStore(tmp_path / "context.sqlite3"),
        summary_worker=worker,
    )
    recorder = InMemoryTraceRecorder()
    token = ACTIVE_TRACE_CONTEXT.set((recorder, "turn-1"))
    try:
        for _ in range(6):
            context = context_manager.load_for_turn(
                user_id="u1",
                conversation_id="c1",
                user_message="ok",
            )
            context_manager.commit_turn(
                context=context,
                task=ConversationTaskState(),
                assistant_message="ok",
            )
    finally:
        ACTIVE_TRACE_CONTEXT.reset(token)

    summary = context_manager._store.get_conversation_summary(
        user_id="u1", conversation_id="c1"
    )
    assert summary is not None
    assert summary.through_sequence == 4
    compacted = [
        event
        for event in recorder.snapshot("turn-1").events
        if event.event_type == "context_compacted"
    ]
    assert compacted[-1].details["trigger"] == "projection_overflow"
    assert compacted[-1].details["projection_overflow"] is True
    assert compacted[-1].details["occupancy"] < 0.01


def test_workflow_exit_compacts_at_a_low_occupancy_seam(tmp_path) -> None:
    worker = RecordingSummaryWorker()
    context_manager = manager(tmp_path, limit=4, summary_worker=worker)
    entry_context = context_manager.load_for_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="开始模拟面试",
    )
    held = context_manager.commit_workflow_entry(
        context=entry_context,
        task=ConversationTaskState(
            active_workflow="mock_interview",
            run_id="mock-1",
            phase="mock_interview_answer_required",
        ),
    )
    workflow_context = context_manager.load_for_workflow_turn(
        user_id="u1",
        conversation_id="c1",
        task=held,
    )

    context_manager.commit_workflow_exit(
        context=workflow_context,
        task=ConversationTaskState(),
        assistant_message="模拟面试已完成。",
    )

    summary = context_manager._store.get_conversation_summary(
        user_id="u1", conversation_id="c1"
    )
    assert summary is not None
    assert summary.through_sequence == 2
    assert len(worker.calls) == 1


def test_one_seam_never_runs_a_synchronous_summary_loop(tmp_path) -> None:
    path = tmp_path / "context.sqlite3"
    writer = ContextManager(CareerContextStore(path), recent_message_limit=20)
    for index in range(5):
        context = writer.load_for_turn(
            user_id="u1", conversation_id="c1", user_message=f"user-{index}"
        )
        writer.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message=f"assistant-{index}",
        )
    worker = RecordingSummaryWorker()
    context_manager = ContextManager(
        CareerContextStore(path),
        summary_worker=worker,
        recent_message_limit=4,
        summary_batch_size=2,
    )

    context_manager._maybe_summarize(
        user_id="u1",
        conversation_id="c1",
        trigger="seam",
        user_message="continue",
    )

    assert len(worker.calls) == 1


def test_report_delivery_does_not_create_a_low_occupancy_seam(tmp_path) -> None:
    worker = RecordingSummaryWorker()
    context_manager = manager(tmp_path, limit=4, summary_worker=worker)
    context = context_manager.load_for_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="研究这个岗位",
    )

    context_manager.commit_turn(
        context=context,
        task=ConversationTaskState(),
        assistant_message="岗位调研报告已生成。",
        assistant_resource_refs=(
            ConversationResourceReference(
                kind="job_research_report",
                resource_id="report-1",
                status_at_delivery="current",
                anchored_by_other_job=False,
            ),
        ),
    )

    assert worker.calls == []
    assert context_manager._store.get_conversation_summary(
        user_id="u1", conversation_id="c1"
    ) is None


def test_a_conversation_without_a_summary_worker_still_keeps_every_message(
    tmp_path,
) -> None:
    """Retention must not depend on whether an optional worker was wired.

    Without a summary worker the read window is already limited to
    ``recent_message_limit`` messages, so pruning the table deleted only rows
    that could never be read again. It also took their resource references
    along, which is what a later turn scans to name an old report.
    """
    context_manager = manager(tmp_path, limit=4)
    window_sizes = []
    for index in range(20):
        context = context_manager.load_for_turn(
            user_id="u1", conversation_id="c1", user_message=f"user-{index}"
        )
        window_sizes.append(len(context.recent_messages))
        context_manager.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message=f"assistant-{index}",
        )

    assert max(window_sizes) <= 4
    stored = context_manager._store.list_messages_after(
        user_id="u1", conversation_id="c1", after_sequence=0, limit=1000
    )
    assert len(stored) == 40
    assert stored[0].content == "user-0"


def test_the_read_window_stays_bounded_while_the_table_keeps_growing(tmp_path) -> None:
    """What the model reads is bounded; what the file stores is not.

    These are separate properties now. Bounding the file too would mean deleting
    originals on the agent's own schedule, which is what the operator has to be
    able to decide instead.
    """
    worker = RecordingSummaryWorker()
    context_manager = manager(
        tmp_path,
        limit=4,
        summary_worker=worker,
        max_recent_context_chars=32,
    )
    window_sizes = []
    for index in range(40):
        context = context_manager.load_for_turn(
            user_id="u1", conversation_id="c1", user_message=f"user-{index}"
        )
        window_sizes.append(len(context.recent_messages))
        context_manager.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message=f"assistant-{index}",
        )

    assert max(window_sizes) <= 6
    stored = context_manager._store.list_messages_after(
        user_id="u1", conversation_id="c1", after_sequence=0, limit=1000
    )
    assert len(stored) == 80
    summary = context_manager._store.get_conversation_summary(
        user_id="u1", conversation_id="c1"
    )
    assert summary is not None
    assert summary.through_sequence >= 70
    # The recent window is still served from raw rows, not from the summary.
    loaded = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="next"
    )
    assert [message.content for message in loaded.recent_messages] == [
        "user-39",
        "assistant-39",
    ]


def test_rolling_summary_merges_previous_summary_and_is_session_scoped(tmp_path) -> None:
    worker = RecordingSummaryWorker()
    context_manager = manager(
        tmp_path,
        limit=2,
        summary_worker=worker,
        max_recent_context_chars=32,
    )
    for index in range(3):
        context = context_manager.load_for_turn(
            user_id="u1", conversation_id="c1", user_message=f"user-{index}"
        )
        context_manager.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message=f"assistant-{index}",
        )

    summary = context_manager._store.get_conversation_summary(
        user_id="u1", conversation_id="c1"
    )
    assert summary is not None
    assert summary.through_sequence == 4
    assert worker.calls[0][0] is None
    assert worker.calls[1][0].confirmed_decisions == ("covered-through-2",)
    assert summary.content.confirmed_decisions == (
        "covered-through-2",
        "covered-through-4",
    )
    assert context_manager.load_for_turn(
        user_id="u2", conversation_id="c1", user_message="other user"
    ).conversation_summary is None
    assert context_manager.load_for_turn(
        user_id="u1", conversation_id="other", user_message="other session"
    ).conversation_summary is None


class FailingSummaryWorker:
    def summarize(self, **kwargs):
        raise AgentWorkerError(
            "SUMMARY_FAILED", "temporary summary failure", retryable=True
        )


def test_summary_worker_failure_preserves_recent_conversation(tmp_path) -> None:
    context_manager = manager(
        tmp_path,
        limit=2,
        summary_worker=FailingSummaryWorker(),
        max_recent_context_chars=32,
    )
    for index in range(2):
        context = context_manager.load_for_turn(
            user_id="u1", conversation_id="c1", user_message=f"user-{index}"
        )
        context_manager.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message=f"assistant-{index}",
        )

    loaded = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="next"
    )
    assert loaded.conversation_summary is None
    assert [message.content for message in loaded.recent_messages][-2:] == [
        "user-1",
        "assistant-1",
    ]


def test_recent_message_projection_obeys_total_character_budget(tmp_path) -> None:
    context_manager = manager(
        tmp_path,
        limit=4,
        max_recent_context_chars=40,
    )
    for _ in range(2):
        context = context_manager.load_for_turn(
            user_id="u1", conversation_id="c1", user_message="x" * 32
        )
        context_manager.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message="y" * 32,
        )

    loaded = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="next"
    )
    assert sum(len(message.content) for message in loaded.recent_messages) == 40
    assert loaded.recent_messages[-1].content == "y" * 10
    assert loaded.recent_messages[-1].content_clipped is True
    assert loaded.through_sequence == 0
    assert loaded.recent_from_sequence == 1


def test_recent_window_clears_only_older_exact_large_body_duplicates(tmp_path) -> None:
    context_manager = ContextManager(
        CareerContextStore(tmp_path / "context.sqlite3"),
        recent_message_limit=4,
        max_message_chars=2000,
        max_recent_context_chars=4000,
    )
    repeated = "JD body " * 100
    for _ in range(2):
        context = context_manager.load_for_turn(
            user_id="u1", conversation_id="c1", user_message=repeated
        )
        context_manager.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message="ack",
        )

    loaded = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="continue"
    )

    assert loaded.recent_messages[0].content.startswith("[duplicate content omitted")
    assert loaded.recent_messages[2].content == repeated
    stored = context_manager._store.list_messages(
        "u1", "c1", limit=10
    )
    assert stored[0].content == stored[2].content == repeated


def test_conversation_span_is_exact_owned_bounded_and_reports_full_count(
    tmp_path,
) -> None:
    context_manager = manager(tmp_path, limit=4)
    for index in range(15):
        context = context_manager.load_for_turn(
            user_id="u1", conversation_id="c1", user_message=f"user-{index}"
        )
        context_manager.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message=f"assistant-{index}",
        )
    other = context_manager.load_for_turn(
        user_id="u1", conversation_id="c2", user_message="other-conversation"
    )
    context_manager.commit_turn(
        context=other,
        task=ConversationTaskState(),
        assistant_message="other-answer",
    )

    span = context_manager._store.read_conversation_span(
        user_id="u1",
        conversation_id="c1",
        from_sequence=1,
        through_sequence=30,
    )
    outside = context_manager._store.read_conversation_span(
        user_id="u1",
        conversation_id="c1",
        from_sequence=100,
        through_sequence=110,
    )

    assert span.returned == 8
    assert span.total == 30
    assert [message.sequence for message in span.messages] == list(range(1, 9))
    assert all("other" not in message.content for message in span.messages)
    assert outside.returned == outside.total == 0
    assert outside.messages == ()


def test_long_conversation_span_can_page_in_by_content_without_blind_scanning(
    tmp_path,
) -> None:
    context_manager = manager(tmp_path, limit=4)
    for index in range(60):
        context = context_manager.load_for_turn(
            user_id="u1", conversation_id="c1", user_message=f"ordinary-{index}"
        )
        answer = (
            "目标公司是星海科技，请记住。"
            if index == 52
            else f"assistant-{index}"
        )
        context_manager.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message=answer,
        )

    span = context_manager._store.read_conversation_span(
        user_id="u1",
        conversation_id="c1",
        from_sequence=1,
        through_sequence=120,
        query="目标公司",
    )

    assert span.returned == span.total == 1
    assert span.messages[0].sequence == 106
    assert "星海科技" in span.messages[0].content


def test_conversation_span_recovers_resource_refs_from_returned_rows(
    tmp_path,
) -> None:
    context_manager = manager(tmp_path, limit=4)
    context = context_manager.load_for_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="研究这个岗位",
    )
    reference = ConversationResourceReference(
        kind="job_research_report",
        resource_id="report-1",
        status_at_delivery="current",
        anchored_by_other_job=False,
        title="Example · AI Engineer",
    )
    context_manager.commit_turn(
        context=context,
        task=ConversationTaskState(),
        assistant_message="岗位调研报告已生成。",
        assistant_resource_refs=(reference,),
    )

    span = context_manager._store.read_conversation_span(
        user_id="u1",
        conversation_id="c1",
        from_sequence=1,
        through_sequence=2,
    )

    assert span.resource_refs == (reference,)
    result = MainAgentToolRegistry(
        conversation_store=context_manager._store
    ).invoke_atomic_tool(
        "read_conversation_span",
        {
            "user_id": "u1",
            "conversation_id": "c1",
            "from_sequence": 1,
            "through_sequence": 2,
        },
    )
    observation = MainAgentRuntime._tool_observation(
        "read_conversation_span",
        result,
    )
    projected = context.model_copy(
        update={"tool_observations": (observation,)}
    ).model_context()["tool_observations"][0]

    assert result.facts["resource_ref_count"] == 1
    assert result.facts["resource_ref_total"] == 1
    assert projected["facts"]["resource_refs"][0]["title"] == (
        "Example · AI Engineer"
    )
    handle = projected["facts"]["resource_refs"][0]["reference"]
    assert handle != reference.resource_id
    assert context.model_copy(
        update={"tool_observations": (observation,)}
    ).resolve_reference(
        reference=handle,
        kind="job_research_report",
    ) == reference.resource_id


def test_conversation_span_caps_resource_refs_and_reports_total(tmp_path) -> None:
    context_manager = manager(tmp_path, limit=4)
    context = context_manager.load_for_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="读取历史报告",
    )
    references = tuple(
        ConversationResourceReference(
            kind="mock_interview_report",
            resource_id=f"report-{index}",
            title=f"Mock report {index}",
        )
        for index in range(40)
    )
    context_manager.commit_turn(
        context=context,
        task=ConversationTaskState(),
        assistant_message="报告已生成。",
        assistant_resource_refs=references,
    )

    result = MainAgentToolRegistry(
        conversation_store=context_manager._store
    ).invoke_atomic_tool(
        "read_conversation_span",
        {
            "user_id": "u1",
            "conversation_id": "c1",
            "from_sequence": 1,
            "through_sequence": 2,
        },
    )

    assert result.facts["resource_ref_count"] == 32
    assert result.facts["resource_ref_total"] == 40
    assert len(result.resource_refs) == 32
    assert len(result.payload["resource_refs"]) == 32
    assert result.payload["resource_ref_total"] == 40


class DroppingConstraintWorker(RecordingSummaryWorker):
    def summarize(self, *, previous, messages):
        self.calls.append((previous, messages))
        return ConversationSummaryContent(
            user_goals=(f"through-{messages[-1].sequence}",),
            active_constraints=(
                ("Never send automatically",) if previous is None else ()
            ),
        )


def test_incremental_summary_cannot_silently_drop_active_constraints(tmp_path) -> None:
    worker = DroppingConstraintWorker()
    context_manager = manager(
        tmp_path,
        limit=2,
        summary_worker=worker,
        max_recent_context_chars=32,
    )
    for index in range(3):
        context = context_manager.load_for_turn(
            user_id="u1", conversation_id="c1", user_message=f"user-{index}"
        )
        context_manager.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message=f"assistant-{index}",
        )

    summary = context_manager._store.get_conversation_summary(
        user_id="u1", conversation_id="c1"
    )
    assert summary is not None
    assert summary.content.active_constraints == ("Never send automatically",)


class OverflowingConstraintWorker(RecordingSummaryWorker):
    def summarize(self, *, previous, messages):
        self.calls.append((previous, messages))
        if previous is None:
            return ConversationSummaryContent(
                active_constraints=tuple(
                    f"{index:02d}" + ("x" * 498) for index in range(12)
                )
            )
        return ConversationSummaryContent(
            active_constraints=("new-one", "new-two", "new-three")
        )


class OverflowingGoalWorker(RecordingSummaryWorker):
    def summarize(self, *, previous, messages):
        self.calls.append((previous, messages))
        if previous is None:
            return ConversationSummaryContent(
                active_constraints=tuple(
                    f"{index:02d}" + ("x" * 498) for index in range(12)
                )
            )
        return ConversationSummaryContent(
            user_goals=("Keep looking for roles in a new city",),
            confirmed_decisions=("Stay put this quarter",),
            unresolved_questions=("What salary band is acceptable?",),
            active_constraints=("new-one",),
        )


def test_constraint_truncation_is_counted_in_compaction_trace(tmp_path) -> None:
    worker = OverflowingConstraintWorker()
    context_manager = manager(
        tmp_path,
        limit=2,
        summary_worker=worker,
        max_recent_context_chars=32,
    )
    recorder = InMemoryTraceRecorder()
    token = ACTIVE_TRACE_CONTEXT.set((recorder, "turn-constraints"))
    try:
        for index in range(3):
            context = context_manager.load_for_turn(
                user_id="u1", conversation_id="c1", user_message=f"user-{index}"
            )
            context_manager.commit_turn(
                context=context,
                task=ConversationTaskState(),
                assistant_message=f"assistant-{index}",
            )
    finally:
        ACTIVE_TRACE_CONTEXT.reset(token)

    compacted = [
        event
        for event in recorder.snapshot("turn-constraints").events
        if event.event_type == "context_compacted"
    ]
    assert compacted[-1].details["dropped_constraints"] == 3
    assert compacted[-1].details["restored_constraints"] == 12
    # Absolute, not a running total: the count says how many constraints the
    # archive is currently holding back, and all of them stay retrievable.
    omitted_count = compacted[-1].details["dropped_constraints"]
    summary = context_manager._store.get_conversation_summary(
        user_id="u1", conversation_id="c1"
    )
    assert summary is not None
    assert summary.content.omitted_active_constraint_count == omitted_count
    archived = context_manager._store.list_conversation_constraints(
        user_id="u1", conversation_id="c1", statuses=("omitted",)
    )
    assert tuple(row.text for row in archived) == (
        "new-one",
        "new-two",
        "new-three",
    )
    assert (
        compacted[-1].details["omitted_active_constraint_count"]
        == omitted_count
    )
    projected = context_manager._build_context(
        user_id="u1", conversation_id="c1", user_message="inspect"
    ).model_context()
    assert (
        projected["conversation_summary"]["omitted_active_constraint_count"]
        == omitted_count
    )


class RepeatingConstraintWorker(RecordingSummaryWorker):
    """Re-extracts the same constraint from every batch.

    This is the shape that matters: the worker sees the previous summary and
    copies constraints forward, and the messages announcing a retirement are
    themselves in the batch being summarized.
    """

    def summarize(self, *, previous, messages):
        self.calls.append((previous, messages))
        return ConversationSummaryContent(
            active_constraints=("Never send automatically",)
        )


def test_a_retired_constraint_is_not_re_extracted_back_into_the_summary(
    tmp_path,
) -> None:
    context_manager = manager(
        tmp_path,
        limit=2,
        summary_worker=RepeatingConstraintWorker(),
        max_recent_context_chars=32,
    )

    def run_turns(count: int) -> None:
        for index in range(count):
            context = context_manager.load_for_turn(
                user_id="u1", conversation_id="c1", user_message=f"user-{index}"
            )
            context_manager.commit_turn(
                context=context,
                task=ConversationTaskState(),
                assistant_message=f"assistant-{index}",
            )

    run_turns(3)
    summary = context_manager._store.get_conversation_summary(
        user_id="u1", conversation_id="c1"
    )
    assert summary is not None
    assert summary.content.active_constraints == ("Never send automatically",)

    assert context_manager._store.retire_conversation_constraint(
        user_id="u1",
        conversation_id="c1",
        constraint_text="Never send automatically",
    )
    # Retirement takes effect on the next turn, not only after the next
    # compaction: the stored projection is rewritten immediately.
    assert (
        context_manager._store.get_conversation_summary(
            user_id="u1", conversation_id="c1"
        ).content.active_constraints
        == ()
    )

    run_turns(4)
    summary = context_manager._store.get_conversation_summary(
        user_id="u1", conversation_id="c1"
    )
    assert summary is not None
    assert summary.content.active_constraints == ()


class ManyShortConstraintWorker(RecordingSummaryWorker):
    def summarize(self, *, previous, messages):
        self.calls.append((previous, messages))
        # The contract caps a single worker payload, so the overflow has to
        # accumulate across batches exactly as it does in production.
        if previous is None:
            return ConversationSummaryContent(
                active_constraints=tuple(
                    f"rule-{index:02d}" for index in range(10)
                )
            )
        if len(previous.active_constraints) < 15:
            return ConversationSummaryContent(
                active_constraints=tuple(
                    f"rule-{index:02d}" for index in range(10, 18)
                )
            )
        return ConversationSummaryContent()


def test_retiring_a_constraint_readmits_one_the_cap_held_back(tmp_path) -> None:
    context_manager = manager(
        tmp_path,
        limit=2,
        summary_worker=ManyShortConstraintWorker(),
        max_recent_context_chars=32,
    )

    def run_turns(start: int, count: int) -> None:
        for index in range(start, start + count):
            context = context_manager.load_for_turn(
                user_id="u1", conversation_id="c1", user_message=f"user-{index}"
            )
            context_manager.commit_turn(
                context=context,
                task=ConversationTaskState(),
                assistant_message=f"assistant-{index}",
            )

    run_turns(0, 3)
    summary = context_manager._store.get_conversation_summary(
        user_id="u1", conversation_id="c1"
    )
    assert summary is not None
    assert len(summary.content.active_constraints) == 15
    assert summary.content.omitted_active_constraint_count == 3
    assert "rule-15" not in summary.content.active_constraints

    assert context_manager._store.retire_conversation_constraint(
        user_id="u1", conversation_id="c1", constraint_text="rule-00"
    )
    run_turns(3, 2)

    summary = context_manager._store.get_conversation_summary(
        user_id="u1", conversation_id="c1"
    )
    assert summary is not None
    assert "rule-00" not in summary.content.active_constraints
    # The freed slot goes to the oldest archived constraint, not to whatever
    # the latest batch happened to mention.
    assert "rule-15" in summary.content.active_constraints
    assert len(summary.content.active_constraints) == 15
    assert summary.content.omitted_active_constraint_count == 2


class UnevenConstraintWorker(RecordingSummaryWorker):
    """Puts an oversized constraint ahead of ones that still fit."""

    def summarize(self, *, previous, messages):
        self.calls.append((previous, messages))
        if previous is None:
            return ConversationSummaryContent(
                active_constraints=tuple(
                    f"{index:02d}" + ("x" * 498) for index in range(11)
                )
            )
        return ConversationSummaryContent(
            active_constraints=("a" * 400, "b" * 200, "c" * 50)
        )


def test_one_oversized_constraint_no_longer_discards_the_shorter_ones_after_it(
    tmp_path,
) -> None:
    context_manager = manager(
        tmp_path,
        limit=2,
        summary_worker=UnevenConstraintWorker(),
        max_recent_context_chars=32,
    )
    for index in range(3):
        context = context_manager.load_for_turn(
            user_id="u1", conversation_id="c1", user_message=f"user-{index}"
        )
        context_manager.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message=f"assistant-{index}",
        )

    summary = context_manager._store.get_conversation_summary(
        user_id="u1", conversation_id="c1"
    )
    assert summary is not None
    visible = summary.content.active_constraints
    # 11 * 500 + 400 leaves 100 characters. The 200-character constraint is
    # skipped and the 50-character one behind it is still admitted.
    assert "b" * 200 not in visible
    assert "c" * 50 in visible
    assert summary.content.omitted_active_constraint_count == 1
    archived = context_manager._store.list_conversation_constraints(
        user_id="u1", conversation_id="c1", statuses=("omitted",)
    )
    assert tuple(row.text for row in archived) == ("b" * 200,)


def test_summary_field_pops_are_counted_when_constraints_consume_the_budget(
    tmp_path,
) -> None:
    worker = OverflowingGoalWorker()
    context_manager = manager(
        tmp_path,
        limit=2,
        summary_worker=worker,
        max_recent_context_chars=32,
    )
    recorder = InMemoryTraceRecorder()
    token = ACTIVE_TRACE_CONTEXT.set((recorder, "turn-goals"))
    try:
        for index in range(3):
            context = context_manager.load_for_turn(
                user_id="u1", conversation_id="c1", user_message=f"user-{index}"
            )
            context_manager.commit_turn(
                context=context,
                task=ConversationTaskState(),
                assistant_message=f"assistant-{index}",
            )
    finally:
        ACTIVE_TRACE_CONTEXT.reset(token)

    compacted = [
        event
        for event in recorder.snapshot("turn-goals").events
        if event.event_type == "context_compacted"
    ]
    last = compacted[-1].details
    assert last["dropped_user_goals"] == 1
    assert last["dropped_confirmed_decisions"] == 1
    assert last["dropped_unresolved_questions"] == 1
    summary = context_manager._store.get_conversation_summary(
        user_id="u1", conversation_id="c1"
    )
    assert summary is not None
    assert summary.content.user_goals == ()
    assert summary.content.confirmed_decisions == ()
    assert summary.content.unresolved_questions == ()
    assert summary.content.omitted_user_goal_count == 1
    assert summary.content.omitted_confirmed_decision_count == 1
    assert summary.content.omitted_unresolved_question_count == 1
    projected = context_manager._build_context(
        user_id="u1", conversation_id="c1", user_message="inspect"
    ).model_context()
    assert projected["conversation_summary"]["omitted_user_goal_count"] == 1
    assert (
        projected["conversation_summary"]["omitted_confirmed_decision_count"]
        == 1
    )
    assert (
        projected["conversation_summary"]["omitted_unresolved_question_count"]
        == 1
    )


def test_full_stored_message_is_clipped_only_for_summary_input(tmp_path) -> None:
    worker = RecordingSummaryWorker()
    context_manager = ContextManager(
        CareerContextStore(tmp_path / "context.sqlite3"),
        summary_worker=worker,
        recent_message_limit=2,
        summary_batch_size=2,
        max_message_chars=32000,
        max_recent_context_chars=10000,
    )
    context = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="u" * 8000
    )
    context_manager.commit_turn(
        context=context,
        task=ConversationTaskState(),
        assistant_message="a" * 8000,
    )
    second = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="second"
    )
    context_manager.commit_turn(
        context=second,
        task=ConversationTaskState(),
        assistant_message="second answer",
    )

    # Triggering and retrying summary must not fail the conversation load.
    context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="next"
    )
    stored = context_manager._store.list_messages(
        "u1", "c1", limit=10
    )
    assert len(stored[0].content) == 8000
    assert [len(item.content) for item in worker.calls[0][1]] == [4000, 4000]


def test_confirmation_recency_label_uses_whole_days() -> None:
    now = datetime(2026, 9, 8, tzinfo=timezone.utc)

    assert confirmation_recency_label(now, now=now) == "今天确认"
    assert (
        confirmation_recency_label(now - timedelta(days=1), now=now)
        == "1 天前确认"
    )
    assert (
        confirmation_recency_label(now - timedelta(days=3), now=now)
        == "3 天前确认"
    )
