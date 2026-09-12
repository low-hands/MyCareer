from datetime import datetime, timedelta, timezone
import sqlite3

import pytest
from pydantic import ValidationError

from career_agent.agent.conversation_memory_contracts import (
    ConversationSummaryContent,
    DistilledFreeTextPreferenceCandidate,
)
from career_agent.agent.context_manager import ContextManager
from career_agent.agent.decision_messages import project_decision_messages
from career_agent.agent.token_budget import budget_encoding, message_token_count
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.agent.openai_compatible_client import AgentWorkerError
from career_agent.agent.main_agent_contracts import (
    AgentPreferencesContext,
    CareerProfileBudgets,
    CareerProfileContext,
    ConversationResourceReference,
    ConversationTaskState,
    FreeTextPreferenceContext,
    HardConstraintContext,
    MainAgentContext,
    _bounded_markdown,
    confirmation_recency_label,
)
from career_agent.harness.memory_telemetry import memory_context_observation
from career_agent.services.intent_capture import IntentCaptureCandidate
from career_agent.harness.observability import (
    ACTIVE_TRACE_CONTEXT,
    InMemoryTraceRecorder,
    conversation_trace_key,
)
from career_agent.storage.context import CareerContextStore
from career_agent.storage.episodes import SQLiteCareerEpisodeStore
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


class GenericPreferenceSummaryWorker:
    def summarize(self, *, previous, messages):
        source = next(message for message in messages if message.role == "user")
        avoids = "不再" in source.content
        return ConversationSummaryContent(
            long_term_memory_candidates=(
                DistilledFreeTextPreferenceCandidate(
                    topic_key="team_open_source_culture",
                    statement=(
                        "不再偏好开源社区活跃的团队"
                        if avoids
                        else "偏好开源社区活跃的团队"
                    ),
                    stance=(
                        "avoid_open_source"
                        if avoids
                        else "favor_open_source"
                        if "仍然" in source.content
                        else "prefer"
                    ),
                    source_sequence=source.sequence,
                    source_quote=source.content,
                    confidence=0.8,
                ),
            ),
        )


def test_summary_distillation_admits_generic_preferences_only_to_quarantine(
    tmp_path,
) -> None:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    context_manager = ContextManager(
        store,
        summary_worker=GenericPreferenceSummaryWorker(),
        recent_message_limit=2,
        summary_batch_size=2,
    )
    first = context_manager.load_for_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="我更喜欢开源社区活跃的团队",
    )
    context_manager.commit_turn(
        context=first,
        task=first.task,
        assistant_message="我会把这作为待确认候选。",
        compaction_trigger="seam",
    )

    quarantined = store.list_free_text_preferences(
        user_id="u1",
        statuses=("quarantined",),
    )
    assert len(quarantined) == 1
    assert quarantined[0].value == "偏好开源社区活跃的团队"
    assert quarantined[0].semantic_stance == "positive"
    assert quarantined[0].source.startswith(
        "agent_inference:conversation_distillation:"
    )
    assert store.list_free_text_preferences(
        user_id="u1",
        statuses=("active",),
    ) == ()
    stored_summary = store.get_conversation_summary(
        user_id="u1",
        conversation_id="c1",
    )
    assert stored_summary is not None
    assert stored_summary.content.long_term_memory_candidates == ()

    two_character_overlap = context_manager.load_for_turn(
        user_id="u1",
        conversation_id="c2-unrelated",
        user_message="我们聊聊开源岗位",
    )
    assert two_character_overlap.free_text_preferences == ()

    relevant = context_manager.load_for_turn(
        user_id="u1",
        conversation_id="c2",
        user_message="推荐一些开源社区活跃团队的岗位",
    )
    assert relevant.free_text_preferences[0].status == "quarantined"
    assert relevant.free_text_preferences[0].statement == "偏好开源社区活跃的团队"

    active = store.confirm_free_text_preference(
        user_id="u1",
        update_id=quarantined[0].update_id,
    )
    assert active is not None
    assert active.semantic_stance == "positive"
    projected = context_manager.load_for_turn(
        user_id="u1",
        conversation_id="c3",
        user_message="继续推荐岗位",
    )
    assert projected.free_text_preferences[0].status == "active"
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE career_intent_versions
            SET semantic_stance = 'favor_open_source'
            WHERE update_id = ?
            """,
            (active.update_id,),
        )

    reaffirmation = context_manager.load_for_turn(
        user_id="u1",
        conversation_id="c4",
        user_message="我仍然更喜欢开源社区活跃的团队",
    )
    context_manager.commit_turn(
        context=reaffirmation,
        task=reaffirmation.task,
        assistant_message="了解。",
        compaction_trigger="seam",
    )
    after_inference = store.list_free_text_preferences(user_id="u1")
    assert len(after_inference) == 1
    assert after_inference[0].update_id == active.update_id
    assert after_inference[0].last_corroborated_at == active.last_corroborated_at

    store.purge_derived_memory(
        user_id="u1",
        scope_key=active.scope_key,
    )
    assert store.list_free_text_preferences(
        user_id="u1",
        statuses=("active",),
    ) == ()


def test_summary_distillation_rejects_unverifiable_candidate_provenance(
    tmp_path,
) -> None:
    class UngroundedWorker:
        def summarize(self, *, previous, messages):
            return ConversationSummaryContent(
                long_term_memory_candidates=(
                    DistilledFreeTextPreferenceCandidate(
                        topic_key="work_style",
                        statement="偏好异步协作",
                        stance="prefer",
                        source_sequence=messages[0].sequence,
                        source_quote="这句话并不存在",
                    ),
                ),
            )

    store = CareerContextStore(tmp_path / "context.sqlite3")
    context_manager = ContextManager(
        store,
        summary_worker=UngroundedWorker(),
        recent_message_limit=2,
        summary_batch_size=2,
    )
    first = context_manager.load_for_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="聊聊工作方式",
    )
    context_manager.commit_turn(
        context=first,
        task=first.task,
        assistant_message="好的。",
        compaction_trigger="seam",
    )

    assert store.list_free_text_preferences(user_id="u1") == ()


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
        "occupancy_source": "legacy",
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
        "preference_candidates_proposed": 0,
        "preference_candidates_admitted": 0,
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
    # Measured at load and carried to the commit, where the reply is the only
    # addition: the load's estimate already holds this turn's message.
    expected = 3000 + message_token_count("short")
    assert event.details["occupancy_source"] == "carried"
    assert event.details["occupancy"] == expected / 1000
    assert event.details["input_occupancy_numerator"] == expected
    assert event.details["input_occupancy_denominator"] == 1000


def _counting_builds(monkeypatch) -> list[None]:
    """Record every full context build, so a test can pin how many a path takes."""
    builds: list[None] = []
    original = ContextManager._build_context

    def counting(self, **kwargs):
        builds.append(None)
        return original(self, **kwargs)

    monkeypatch.setattr(ContextManager, "_build_context", counting)
    return builds


def _compacted_events(recorder, run_id: str):
    return [
        event
        for event in recorder.snapshot(run_id).events
        if event.event_type == "context_compacted"
    ]


def test_an_ordinary_turn_builds_its_context_once_including_the_commit(
    tmp_path, monkeypatch
) -> None:
    worker = RecordingSummaryWorker()
    context_manager = ContextManager(
        CareerContextStore(tmp_path / "context.sqlite3"),
        summary_worker=worker,
        recent_message_limit=8,
        summary_batch_size=2,
    )
    context_manager.configure_request_token_estimator(
        lambda context: (100, 1000)
    )
    builds = _counting_builds(monkeypatch)
    recorder = InMemoryTraceRecorder()
    token = ACTIVE_TRACE_CONTEXT.set((recorder, "turn-once"))
    try:
        for index in range(3):
            builds.clear()
            context = context_manager.load_for_turn(
                user_id="u1", conversation_id="c1", user_message=f"问题{index}"
            )
            context_manager.commit_turn(
                context=context,
                task=ConversationTaskState(),
                assistant_message=f"回答{index}",
            )
            # Measuring pressure at load is the one build; the commit reuses
            # that measurement instead of building again.
            assert len(builds) == 1
    finally:
        ACTIVE_TRACE_CONTEXT.reset(token)

    assert worker.calls == []
    assert _compacted_events(recorder, "turn-once") == []


def test_commit_compacts_when_the_carried_estimate_and_reply_cross_the_threshold(
    tmp_path, monkeypatch
) -> None:
    worker = RecordingSummaryWorker()
    context_manager = ContextManager(
        CareerContextStore(tmp_path / "context.sqlite3"),
        summary_worker=worker,
        summary_batch_size=2,
    )
    context_manager.configure_request_token_estimator(
        lambda context: (700, 1000),
        static_input_tokens=400,
        max_input_tokens=1000,
    )
    reply_cap = context_manager._recent_message_tokens
    reply = EN_16K[:2000]
    assert reply_cap is not None and message_token_count(reply) > reply_cap
    builds = _counting_builds(monkeypatch)
    recorder = InMemoryTraceRecorder()
    token = ACTIVE_TRACE_CONTEXT.set((recorder, "turn-carried"))
    try:
        context = context_manager.load_for_turn(
            user_id="u1",
            conversation_id="c1",
            user_message="请帮我对比这两个岗位的要求和薪资范围",
        )
        assert worker.calls == []
        context_manager.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message=reply,
        )
    finally:
        ACTIVE_TRACE_CONTEXT.reset(token)

    assert len(worker.calls) == 1
    assert len(builds) == 1
    (event,) = _compacted_events(recorder, "turn-carried")
    # The reply, capped at its window share, is the only increment. Adding this
    # turn's message again would count it twice.
    assert event.details["occupancy_source"] == "carried"
    assert event.details["input_occupancy_numerator"] == 700 + reply_cap
    assert event.details["input_occupancy_denominator"] == 1000
    assert event.details["occupancy"] == (700 + reply_cap) / 1000


def test_commit_compacts_on_overflow_even_when_the_carried_estimate_is_low(
    tmp_path,
) -> None:
    path = tmp_path / "context.sqlite3"
    writer = ContextManager(CareerContextStore(path), recent_message_limit=40)
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
        recent_message_limit=8,
        summary_batch_size=4,
    )
    context_manager.configure_request_token_estimator(lambda context: (1, 1000))
    recorder = InMemoryTraceRecorder()
    token = ACTIVE_TRACE_CONTEXT.set((recorder, "turn-overflow"))
    try:
        # Ten unsummarised rows fit the eleven-row projection at load.
        context = context_manager.load_for_turn(
            user_id="u1", conversation_id="c1", user_message="next"
        )
        assert worker.calls == []
        # Twelve do not.
        context_manager.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message="answer",
        )
    finally:
        ACTIVE_TRACE_CONTEXT.reset(token)

    assert len(worker.calls) == 1
    (event,) = _compacted_events(recorder, "turn-overflow")
    assert event.details["trigger"] == "projection_overflow"
    assert event.details["occupancy_source"] == "carried"
    assert event.details["occupancy"] < 0.75


def test_a_commit_no_load_measured_leaves_occupancy_to_the_next_load(
    tmp_path, monkeypatch
) -> None:
    worker = RecordingSummaryWorker()
    context_manager = ContextManager(
        CareerContextStore(tmp_path / "context.sqlite3"),
        summary_worker=worker,
        summary_batch_size=2,
    )
    estimate = {"tokens": 1}
    context_manager.configure_request_token_estimator(
        lambda context: (estimate["tokens"], 1000)
    )
    context = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="第一问"
    )
    context_manager.commit_turn(
        context=context,
        task=ConversationTaskState(),
        assistant_message="第一答",
    )
    # A load that measured but never committed, as when its turn fails with
    # nothing to record. Its estimate plus a long reply would cross 0.75.
    estimate["tokens"] = 700
    context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="第二问"
    )
    workflow_context = context_manager.load_for_workflow_turn(
        user_id="u1",
        conversation_id="c1",
        task=ConversationTaskState(),
    )
    builds = _counting_builds(monkeypatch)

    context_manager.commit_turn(
        context=workflow_context,
        task=ConversationTaskState(),
        assistant_message=EN_16K[:2000],
    )

    assert worker.calls == []
    assert builds == []
    estimate["tokens"] = 3000
    context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="继续"
    )
    assert len(worker.calls) == 1


def test_a_seam_compacts_without_building_or_estimating(
    tmp_path, monkeypatch
) -> None:
    worker = RecordingSummaryWorker()
    context_manager = manager(tmp_path, limit=4, summary_worker=worker)
    estimates: list[None] = []

    def estimator(context):
        estimates.append(None)
        return (1, 1000)

    context_manager.configure_request_token_estimator(estimator)
    entry_context = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="开始模拟面试"
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
        user_id="u1", conversation_id="c1", task=held
    )
    estimates.clear()
    builds = _counting_builds(monkeypatch)
    recorder = InMemoryTraceRecorder()
    token = ACTIVE_TRACE_CONTEXT.set((recorder, "turn-seam"))
    try:
        context_manager.commit_workflow_exit(
            context=workflow_context,
            task=ConversationTaskState(),
            assistant_message="模拟面试已完成。",
        )
    finally:
        ACTIVE_TRACE_CONTEXT.reset(token)

    assert len(worker.calls) == 1
    assert (builds, estimates) == ([], [])
    (event,) = _compacted_events(recorder, "turn-seam")
    assert event.details["trigger"] == "seam"
    assert event.details["occupancy_source"] == "seam"
    assert event.details["occupancy"] is None
    assert event.details["input_occupancy_numerator"] is None


def test_a_load_that_compacted_carries_the_estimate_of_what_it_returned(
    tmp_path,
) -> None:
    path = tmp_path / "context.sqlite3"
    writer = ContextManager(CareerContextStore(path), recent_message_limit=40)
    for index in range(2):
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
        recent_message_limit=8,
        summary_batch_size=2,
    )
    # Over budget until a summary exists, nearly empty once one does.
    context_manager.configure_request_token_estimator(
        lambda context: (3000 if context.through_sequence == 0 else 1, 1000)
    )

    context = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="next"
    )

    assert len(worker.calls) == 1
    assert context.through_sequence == 2
    assert context_manager._carried_request_tokens[("u1", "c1")] == (1, 1000)
    context_manager.commit_turn(
        context=context,
        task=ConversationTaskState(),
        assistant_message="ok",
    )
    # The pre-compaction 3000 would have compacted a second batch here.
    assert len(worker.calls) == 1
    assert ("u1", "c1") not in context_manager._carried_request_tokens


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
    pressure = context_manager._recent_pressure(
        user_id="u1",
        conversation_id="c1",
        after_sequence=first.through_sequence,
        user_message="probe",
    )
    assert pressure.occupancy == 1.0
    assert pressure.projection_overflow is True

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
        measure=lambda after_sequence: context_manager._overflow_only_pressure(
            user_id="u1",
            conversation_id="c1",
            after_sequence=after_sequence,
            source="seam",
        ),
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


_ZH_LINE = (
    "负责大模型推理服务的性能优化与稳定性建设，熟悉分布式训练框架，"
    "具备鑫龘饕餮等生僻字处理经验；"
)
_EN_LINE = (
    "Design and operate the retrieval pipeline that ranks candidate job "
    "postings against a structured career profile. "
)
ZH_16K = (_ZH_LINE * 400)[:16_000]
ZH_32K = (_ZH_LINE * 800)[:32_000]
EN_16K = (_EN_LINE * 200)[:16_000]


def token_bounded_manager(tmp_path, **kwargs) -> ContextManager:
    """A manager wired like production: 32k input, 12,066 static tokens."""
    context_manager = ContextManager(
        CareerContextStore(tmp_path / "context.sqlite3"), **kwargs
    )
    context_manager.configure_request_token_estimator(
        lambda context: (1, 32_000),
        static_input_tokens=12_066,
        max_input_tokens=32_000,
    )
    return context_manager


def test_message_token_caps_are_shares_of_the_dynamic_input_budget(
    tmp_path,
) -> None:
    context_manager = token_bounded_manager(tmp_path)

    assert (
        context_manager._user_message_tokens,
        context_manager._recent_context_tokens,
        context_manager._recent_message_tokens,
    ) == (3_986, 7_973, 2_990)


def test_an_explicit_token_cap_overrides_the_derived_one(tmp_path) -> None:
    context_manager = token_bounded_manager(
        tmp_path, max_user_message_tokens=50
    )

    context = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message=ZH_32K
    )

    assert context_manager._user_message_tokens == 50
    assert context_manager._recent_context_tokens == 7_973
    assert context.user_message_clipped is True
    assert message_token_count(context.user_message) <= 50


@pytest.mark.parametrize(
    "message", ["帮我看看这个岗位", EN_16K], ids=["short", "english-16k"]
)
def test_messages_under_the_token_caps_reach_the_prompt_whole(
    tmp_path, message
) -> None:
    assert message_token_count(message) < 2_990
    context_manager = token_bounded_manager(tmp_path)

    context = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message=message
    )
    assert (
        context.user_message,
        context.user_message_clipped,
        context.user_message_source,
    ) == (message, False, None)
    context_manager.commit_turn(
        context=context,
        task=ConversationTaskState(),
        assistant_message="收到",
    )
    loaded = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="继续"
    )

    assert loaded.recent_messages[0].content == message
    assert loaded.recent_messages[0].content_clipped is False


def test_a_long_current_message_is_clipped_for_the_prompt_only(tmp_path) -> None:
    context_manager = token_bounded_manager(tmp_path)

    context = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message=ZH_32K
    )

    assert context.user_message_clipped is True
    assert context.user_message_source == ZH_32K
    assert ZH_32K.startswith(context.user_message)
    assert 0 < message_token_count(context.user_message) <= 3_986
    assert project_decision_messages(context).current_user_message.endswith(
        "content_clipped=true]"
    )

    context_manager.commit_turn(
        context=context,
        task=ConversationTaskState(),
        assistant_message="收到",
    )
    stored = context_manager._store.list_message_records(
        "u1", "c1", limit=10
    )
    assert stored[0].message.content == ZH_32K


def test_a_held_workflow_request_is_the_message_as_sent(tmp_path) -> None:
    context_manager = token_bounded_manager(tmp_path)
    context = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message=ZH_32K
    )
    assert context.user_message_clipped is True

    held = context_manager.commit_workflow_entry(
        context=context,
        task=ConversationTaskState(
            active_workflow="mock_interview",
            run_id="mock-1",
        ),
    )

    assert held.workflow_entry_message == ZH_32K


def test_workflow_exit_writes_the_held_request_not_this_turns_source(
    tmp_path,
) -> None:
    context_manager = token_bounded_manager(tmp_path)
    context = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message=ZH_32K
    )
    exiting = context.model_copy(
        update={
            "task": context.task.model_copy(
                update={"workflow_entry_message": "开始模拟面试"}
            )
        }
    )

    context_manager.commit_workflow_exit(
        context=exiting,
        task=ConversationTaskState(),
        assistant_message="面试结束",
    )

    stored = context_manager._store.list_message_records(
        "u1", "c1", limit=10
    )
    assert [record.message.content for record in stored] == [
        "开始模拟面试",
        "面试结束",
    ]


def test_a_long_message_in_the_window_is_clipped_to_its_token_share(
    tmp_path,
) -> None:
    context_manager = token_bounded_manager(tmp_path)
    context = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message=ZH_16K
    )
    context_manager.commit_turn(
        context=context,
        task=ConversationTaskState(),
        assistant_message="收到",
    )

    loaded = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="继续"
    )

    long_message, reply = loaded.recent_messages
    assert long_message.content_clipped is True
    assert ZH_16K.startswith(long_message.content)
    assert 0 < message_token_count(long_message.content) <= 2_990
    assert (reply.content, reply.content_clipped) == ("收到", False)
    assert context_manager._store.list_message_records(
        "u1", "c1", limit=10
    )[0].message.content == ZH_16K


def test_a_crowded_window_shares_the_token_cap_fairly(tmp_path) -> None:
    context_manager = token_bounded_manager(tmp_path)
    for index in range(4):
        # Distinct tails, or the window would replace older copies with the
        # duplicate marker before the budget saw them. The bodies open on a
        # character that takes more than one token, so the single token left
        # after four fair shares cannot hold it.
        context = context_manager.load_for_turn(
            user_id="u1",
            conversation_id="c1",
            user_message=ZH_16K[:15_000] + f"用户{index}",
        )
        context_manager.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message=ZH_16K[:15_000] + f"助手{index}",
        )

    loaded = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="继续"
    )

    counts = [
        message_token_count(message.content)
        for message in loaded.recent_messages
    ]
    assert sum(counts) <= 7_973
    assert all(message.content_clipped for message in loaded.recent_messages)
    assert all(0 < count <= 7_973 // 4 for count in counts)
    # One token is left for the fifth, and its first character needs more, so
    # the window ends at four rather than showing an empty clipped message.
    assert len(loaded.recent_messages) == 4
    assert loaded.recent_from_sequence == 5


def test_an_empty_stored_reply_does_not_end_the_token_bounded_window(
    tmp_path,
) -> None:
    context_manager = token_bounded_manager(tmp_path)
    for message, reply in (("第一问", "第一答"), ("第二问", "")):
        context = context_manager.load_for_turn(
            user_id="u1", conversation_id="c1", user_message=message
        )
        context_manager.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message=reply,
        )

    loaded = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="继续"
    )

    assert [
        (message.content, message.content_clipped)
        for message in loaded.recent_messages
    ] == [("第一问", False), ("第一答", False), ("第二问", False), ("", False)]


def test_a_dropped_split_character_leaves_no_scrap_of_the_window(
    tmp_path,
) -> None:
    body = _ZH_LINE * 20
    ids = budget_encoding().encode(body)
    # A fair share that cuts inside a character, so each clipped body decodes
    # to fewer tokens than the share it was given.
    share = next(
        limit
        for limit in range(8, len(ids))
        if budget_encoding().decode(ids[:limit]).endswith("\ufffd")
    )
    context_manager = token_bounded_manager(
        tmp_path,
        max_recent_context_tokens=4 * share,
        max_recent_message_tokens=4 * share,
    )
    turns = [
        ("An older English question. " * 20, "An older English answer. " * 20),
        (body + "用户1", body + "助手1"),
        (body + "用户2", body + "助手2"),
    ]
    for message, reply in turns:
        context = context_manager.load_for_turn(
            user_id="u1", conversation_id="c1", user_message=message
        )
        context_manager.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message=reply,
        )

    loaded = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="继续"
    )

    # Four clipped bodies spend the window exactly. Charged only what decoded,
    # they would leave the dropped characters' tokens for the English answer,
    # which would come back as a few-token fragment.
    assert all(
        0 < message_token_count(message.content) < share
        for message in loaded.recent_messages
    )
    assert len(loaded.recent_messages) == 4
    assert loaded.recent_from_sequence == 3


def test_retrieval_queries_read_the_prompt_copy_of_a_long_message(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "context.sqlite3"
    episode_store = SQLiteCareerEpisodeStore(path)
    context_manager = ContextManager(
        CareerContextStore(path), episode_store=episode_store
    )
    context_manager.configure_request_token_estimator(
        lambda context: (1, 32_000),
        static_input_tokens=12_066,
        max_input_tokens=32_000,
    )
    queries: list[str] = []
    search = context_manager._store.search_free_text_preference_rankings
    project = episode_store.project_relevant

    def recording_search(*, query, **kwargs):
        queries.append(query)
        return search(query=query, **kwargs)

    def recording_project(*, query, **kwargs):
        queries.append(query)
        return project(query=query, **kwargs)

    monkeypatch.setattr(
        context_manager._store,
        "search_free_text_preference_rankings",
        recording_search,
    )
    monkeypatch.setattr(episode_store, "project_relevant", recording_project)

    context = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message=ZH_32K
    )

    assert context.user_message_clipped is True
    # Two preference rankings (active, quarantined) and one episode search.
    assert len(queries) == 3
    assert set(queries) == {context.user_message}


@pytest.mark.parametrize(
    ("current", "earlier"),
    [
        ("帮我看看这个岗位", None),
        ("继续", ZH_16K),
        (ZH_32K, None),
        (EN_16K, None),
    ],
    ids=["short", "chinese-16k-in-window", "chinese-32k-current", "english-16k"],
)
def test_without_token_caps_messages_keep_the_character_bounds(
    tmp_path, current, earlier
) -> None:
    # Maintenance commands and most fixtures never install an estimator.
    context_manager = ContextManager(
        CareerContextStore(tmp_path / "context.sqlite3")
    )
    if earlier is not None:
        context = context_manager.load_for_turn(
            user_id="u1", conversation_id="c1", user_message=earlier
        )
        context_manager.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message="收到",
        )

    context = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message=current
    )

    assert (
        context.user_message,
        context.user_message_clipped,
        context.user_message_source,
    ) == (current, False, None)
    projection = project_decision_messages(context)
    assert projection.current_user_message == current
    if earlier is not None:
        assert [
            (message.content, message.content_clipped)
            for message in context.recent_messages
        ] == [(earlier, False), ("收到", False)]


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


def _seed_preferences(
    store: CareerContextStore, *, active: int = 0, quarantined: int = 0
) -> None:
    """Write preferences straight to storage, one topic each.

    One topic per preference because the projection resolves a topic's ownership
    track down to a single winner; the cap only becomes observable across topics.
    """
    now = datetime.now(timezone.utc)
    for index in range(active):
        _, version = store.capture_profile_intent(
            IntentCaptureCandidate(
                user_id="u1",
                scope_key=f"person_intent/self/active{index}_preference",
                value=f"偏好第{index}项",
                source="test",
                pref_scope="freeform.person_default",
                layer="contextual",
                semantic_stance="positive",
                observed_at=now,
            )
        )
        assert version is not None and version.admission_status == "active"
    for index in range(quarantined):
        _, version = store.capture_profile_intent(
            IntentCaptureCandidate(
                user_id="u1",
                scope_key=f"person_intent/self/held{index}_preference",
                value=f"待确认第{index}项",
                source="test",
                pref_scope="freeform.person_default",
                layer="contextual",
                ambiguous=True,
                semantic_stance="positive",
                observed_at=now,
            )
        )
        assert version is not None and version.admission_status == "quarantined"


def _preferences(context_manager: ContextManager, *, conversation_id: str = "c1"):
    # An explicit confirmation makes every quarantined candidate relevant, so
    # the cap is what decides which ones survive rather than retrieval.
    return context_manager.load_for_turn(
        user_id="u1",
        conversation_id=conversation_id,
        user_message="确认",
    )


def test_a_capped_preference_projection_says_how_many_it_held_back(
    tmp_path,
) -> None:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    _seed_preferences(store, active=10, quarantined=4)
    context = _preferences(ContextManager(store))

    statuses = [item.status for item in context.free_text_preferences]
    assert statuses.count("active") == 5
    assert statuses.count("quarantined") == 3
    assert context.free_text_preferences_active_total == 10
    assert context.free_text_preferences_quarantined_total == 4

    projected = context.model_context()["free_text_preferences"]
    assert "（另有 5 条已确认偏好未列出）" in projected
    assert "（另有 1 条待确认偏好未列出）" in projected


def test_an_uncapped_preference_projection_stays_silent(tmp_path) -> None:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    _seed_preferences(store, active=8)
    context = _preferences(ContextManager(store))

    assert len(context.free_text_preferences) == 8
    assert context.free_text_preferences_quarantined_total == 0
    assert "未列出" not in context.model_context()["free_text_preferences"]


def test_quarantined_candidates_take_only_their_reserved_share(tmp_path) -> None:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    _seed_preferences(store, active=2, quarantined=5)
    context = _preferences(ContextManager(store))

    statuses = [item.status for item in context.free_text_preferences]
    # Three is what the projection renders, so a fourth candidate would be
    # carried and never shown. The two spare slots go unused rather than to
    # candidates that could not appear.
    assert statuses.count("active") == 2
    assert statuses.count("quarantined") == 3

    projected = context.model_context()["free_text_preferences"]
    assert "（另有 2 条待确认偏好未列出）" in projected
    assert "已确认偏好未列出" not in projected


def test_confirmed_preferences_keep_a_slot_a_candidate_cannot_use(tmp_path) -> None:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    _seed_preferences(store, active=10, quarantined=1)
    context = _preferences(ContextManager(store))

    statuses = [item.status for item in context.free_text_preferences]
    assert statuses.count("active") == 7
    assert statuses.count("quarantined") == 1


def test_a_total_below_what_was_projected_is_rejected() -> None:
    values = {
        "conversation_id": "c1",
        "profile": CareerProfileContext(user_id="u1"),
        "user_message": "帮我看看",
        "free_text_preferences": (
            FreeTextPreferenceContext(
                scope_key="person_intent/self/held_preference",
                topic_key="held",
                statement="待确认偏好",
                status="quarantined",
                observed_at=datetime.now(timezone.utc),
                update_id="intent_update_" + "a" * 32,
            ),
        ),
    }

    with pytest.raises(ValidationError):
        MainAgentContext(**values, free_text_preferences_quarantined_total=0)
    assert (
        MainAgentContext(
            **values, free_text_preferences_quarantined_total=4
        ).free_text_preferences_quarantined_total
        == 4
    )


def test_memory_telemetry_reports_hidden_preferences_only_when_some_are(
    tmp_path,
) -> None:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    _seed_preferences(store, active=2)
    context_manager = ContextManager(store)

    visible = memory_context_observation(
        _preferences(context_manager),
        career_memory_enabled=False,
    )
    assert "free_text_preferences_hidden" not in visible

    _seed_preferences(store, quarantined=4)
    truncated = memory_context_observation(
        _preferences(context_manager, conversation_id="c2"),
        career_memory_enabled=False,
    )
    assert truncated["free_text_preferences_hidden"] == 1


def test_a_full_budget_leaves_no_truncation_notice() -> None:
    lines = ["## 标题", "- 第一条", "- 第二条"]
    exact = len("\n".join(lines))

    assert _bounded_markdown(lines, budget=exact, line_limit=220) == "\n".join(
        lines
    )


def test_dropped_lines_are_counted_in_the_markdown() -> None:
    lines = ["## 标题", "- 第一条", "- 第二条", "- 第三条", "- 第四条"]
    # Two lines fit alongside the notice; the remaining three are counted.
    budget = len("\n".join(lines[:2])) + len("\n- （另有 3 行未显示）")

    bounded = _bounded_markdown(lines, budget=budget, line_limit=220)

    assert bounded.splitlines()[:2] == lines[:2]
    assert bounded.splitlines()[-1] == "- （另有 3 行未显示）"


def test_the_truncation_notice_is_paid_for_out_of_the_budget() -> None:
    lines = ["## 标题", "- 第一条", "- 第二条", "- 第三条"]
    # Room for three lines, but not for three plus the notice, nor two plus it:
    # lines are handed back until the notice fits, and the count grows with each
    # one given up, so it ends up reporting all three that are not shown.
    budget = 20

    bounded = _bounded_markdown(lines, budget=budget, line_limit=220)

    assert len(bounded) <= budget
    assert bounded == "## 标题\n- （另有 3 行未显示）"


def test_a_budget_too_small_for_any_line_yields_nothing() -> None:
    assert _bounded_markdown(
        ["## 一个很长的标题占满整个预算", "- 第一条"],
        budget=8,
        line_limit=220,
    ) == ""


def test_confirmed_preferences_are_capped_by_relevance_to_this_message(
    tmp_path,
) -> None:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    now = datetime.now(timezone.utc)
    for index in range(9):
        store.capture_profile_intent(
            IntentCaptureCandidate(
                user_id="u1",
                scope_key=f"person_intent/self/filler{index}_preference",
                value=f"无关偏好第{index}项",
                source="test",
                pref_scope="freeform.person_default",
                layer="contextual",
                semantic_stance="positive",
                observed_at=now,
            )
        )
    store.capture_profile_intent(
        IntentCaptureCandidate(
            user_id="u1",
            scope_key="person_intent/self/remote_preference",
            value="希望远程办公",
            source="test",
            pref_scope="freeform.person_default",
            layer="contextual",
            semantic_stance="positive",
            observed_at=now,
        )
    )

    context = ContextManager(store).load_for_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="远程办公的岗位有哪些？",
    )

    statements = [item.statement for item in context.free_text_preferences]
    # Ten confirmed preferences, eight slots. The one this message is about was
    # captured last, so resolver order alone would have dropped it.
    assert statements[0] == "希望远程办公"
    assert context.free_text_preferences_active_total == 10
    # Eight statements this long exceed the character cap, so here the second cut
    # is what bites and it takes the slot-count line with it. It no longer does so
    # in silence, which is the whole reason the notice lives inside the bounding.
    projected = context.model_context()["free_text_preferences"]
    assert projected.splitlines()[-1].endswith("行未显示）")


def test_confirmed_hits_cannot_crowd_a_candidate_out_of_retrieval(tmp_path) -> None:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    now = datetime.now(timezone.utc)
    for index in range(40):
        store.capture_profile_intent(
            IntentCaptureCandidate(
                user_id="u1",
                scope_key=f"person_intent/self/remote{index}_preference",
                value=f"远程办公的岗位第{index}项",
                source="test",
                pref_scope="freeform.person_default",
                layer="contextual",
                semantic_stance="positive",
                observed_at=now,
            )
        )
    _, held = store.capture_profile_intent(
        IntentCaptureCandidate(
            user_id="u1",
            scope_key="person_intent/self/held_preference",
            value="如果薪资合适也可以接受远程办公",
            source="test",
            pref_scope="freeform.person_default",
            layer="contextual",
            ambiguous=True,
            semantic_stance="positive",
            observed_at=now,
        )
    )
    assert held is not None and held.admission_status == "quarantined"

    context = ContextManager(store).load_for_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="远程办公的岗位有哪些？",
    )

    # Every confirmed statement matches more of the message than the candidate
    # does, so one ranking shared across both statuses fills its whole limit
    # with confirmed hits. The message is no confirmation, so retrieval is the
    # candidate's only way through the gate.
    held_back = [
        item.statement
        for item in context.free_text_preferences
        if item.status == "quarantined"
    ]
    assert held_back == ["如果薪资合适也可以接受远程办公"]
