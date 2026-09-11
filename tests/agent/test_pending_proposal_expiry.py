from datetime import datetime, timedelta, timezone

import pytest

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import (
    PENDING_PROPOSAL_SLOTS,
    PENDING_PROPOSAL_TTL,
    CareerFactProposal,
    CareerProfileContext,
    ConstraintRetirementProposal,
    ConversationTaskState,
    FreeTextPreferenceConfirmationProposal,
    JobIntentUpdate,
    MainAgentContext,
    MemoryAmendmentProposal,
    MemoryTombstoneProposal,
    ToolResult,
    project_job_intent_arguments,
)
from career_agent.agent.main_agent_reducers import (
    ATOMIC_TASK_REDUCERS,
    reduce_task_state,
)
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.harness.observability import (
    ACTIVE_TRACE_CONTEXT,
    InMemoryTraceRecorder,
    conversation_trace_key,
)
from career_agent.storage.context import CareerContextStore

_NOW = datetime(2026, 9, 11, 12, tzinfo=timezone.utc)
_RETIREMENT = ConstraintRetirementProposal(
    target_kind="conversation_constraint",
    constraint="只看上海岗位",
    reason="用户说现在也考虑杭州。",
)


def _proposed(proposal: ConstraintRetirementProposal) -> ToolResult:
    return ToolResult(
        tool_name="propose_constraint_retirement",
        state="constraint_retirement_proposed",
        message="拟停止应用这条约束。",
        payload={"proposal": proposal.model_dump(mode="json")},
    )


def test_propose_reducer_stamps_and_a_reshown_proposal_restarts_the_clock() -> None:
    first = reduce_task_state(
        ConversationTaskState(), _proposed(_RETIREMENT), now=_NOW
    )
    assert first.pending_proposed_at == {"pending_constraint_retirement": _NOW}

    later = _NOW + timedelta(days=3)
    reshown = reduce_task_state(first, _proposed(_RETIREMENT), now=later)
    # Equal proposal, but shown again: the user saw a fresh readback.
    assert reshown.pending_constraint_retirement == first.pending_constraint_retirement
    assert reshown.pending_proposed_at == {"pending_constraint_retirement": later}


def test_unrelated_reducer_keeps_stamps_and_confirmation_drops_them() -> None:
    task = reduce_task_state(
        ConversationTaskState(), _proposed(_RETIREMENT), now=_NOW
    )
    unrelated = reduce_task_state(
        task,
        ToolResult(
            tool_name="list_email_events",
            state="no_email_events_found",
            message="没有邮件事件。",
            payload={"items": []},
        ),
        now=_NOW + timedelta(days=5),
    )
    assert unrelated.pending_proposed_at == {"pending_constraint_retirement": _NOW}

    confirmed = reduce_task_state(
        unrelated,
        ToolResult(
            tool_name="confirm_constraint_retirement",
            state="constraint_retired",
            message="已停止应用这条约束。",
        ),
        now=_NOW + timedelta(days=5),
    )
    assert confirmed.pending_constraint_retirement is None
    assert confirmed.pending_proposed_at == {}


def test_structured_follow_up_starts_its_own_clock() -> None:
    task = ConversationTaskState(
        pending_job_intent_update=JobIntentUpdate(city="上海"),
        pending_proposed_at={"pending_job_intent_update": _NOW},
    )
    kept = reduce_task_state(
        task,
        ToolResult(
            tool_name="confirm_free_text_preference",
            state="free_text_preference_confirmed",
            message="已确认。",
        ),
        now=_NOW + timedelta(days=6),
    )
    # The reducer carried the older job intent forward untouched.
    assert kept.pending_proposed_at == {"pending_job_intent_update": _NOW}

    replaced = reduce_task_state(
        task,
        ToolResult(
            tool_name="confirm_free_text_preference",
            state="free_text_preference_confirmed_structured_proposed",
            message="已确认，并识别到结构化版本。",
            payload={"structured_proposal": {"city": "杭州"}},
        ),
        now=_NOW + timedelta(days=6),
    )
    assert replaced.pending_proposed_at == {
        "pending_job_intent_update": _NOW + timedelta(days=6)
    }


def _all_slots_task(stamp: datetime) -> ConversationTaskState:
    return ConversationTaskState(
        pending_job_intent_update=JobIntentUpdate(city="上海"),
        pending_free_text_preference=FreeTextPreferenceConfirmationProposal(
            update_id="intent_update_" + "a" * 32,
            topic_key="company_scale",
            statement="不去大厂",
        ),
        pending_memory_amendment=MemoryAmendmentProposal(
            target_kind="career_evidence",
            detail_ref="detail_" + "a" * 24,
            new_claim="了解 Rust",
            reason="用户更正熟练度。",
        ),
        pending_memory_tombstone=MemoryTombstoneProposal(
            target_kind="career_evidence",
            detail_ref="detail_" + "b" * 24,
            reason="用户要求删除。",
        ),
        pending_career_fact=CareerFactProposal(
            career_evidence_id="career_evidence_" + "a" * 32,
            career_record_id="career_record_" + "a" * 32,
            claim="曾带领 5 人团队。",
            reason="用户补充。",
        ),
        pending_constraint_retirement=_RETIREMENT,
        pending_proposed_at={slot: stamp for slot in PENDING_PROPOSAL_SLOTS},
    )


def _stamp_bytes(task: ConversationTaskState) -> dict[str, str]:
    return {
        slot: task.model_dump_json(include={"pending_proposed_at": {slot}})
        for slot in task.pending_proposed_at
    }


_REDUCER_CASES = [
    (name, state)
    for name, entry in sorted(ATOMIC_TASK_REDUCERS.items())
    for state in (sorted(entry.states) or ["any_state"])
]


@pytest.mark.parametrize(("tool_name", "state"), _REDUCER_CASES)
def test_no_reducer_renews_a_slot_it_did_not_fill(tool_name, state) -> None:
    # Stamping trusts that reducers carry untouched slots by reference. A
    # reducer rebuilt as model_validate(task.model_dump() | {...}) would hand
    # every slot a new object and silently renew all of them. With no payload
    # no reducer has a proposal to put anywhere, so any surviving slot whose
    # stamp moved was renewed by a rebuild.
    old = _NOW - timedelta(days=6)
    task = _all_slots_task(old)
    reduced = reduce_task_state(
        task,
        ToolResult(tool_name=tool_name, state=state, message="结果。"),
        now=_NOW,
    )

    before = _stamp_bytes(task)
    after = _stamp_bytes(reduced)
    for slot in PENDING_PROPOSAL_SLOTS:
        if getattr(reduced, slot) is None:
            assert slot not in reduced.pending_proposed_at
        else:
            assert after[slot] == before[slot], slot


def test_filling_one_slot_leaves_every_other_stamp_byte_identical() -> None:
    old = _NOW - timedelta(days=6)
    task = _all_slots_task(old)
    replacement = _RETIREMENT.model_copy(update={"reason": "用户换了说法。"})

    reduced = reduce_task_state(task, _proposed(replacement), now=_NOW)

    before = _stamp_bytes(task)
    after = _stamp_bytes(reduced)
    assert reduced.pending_proposed_at["pending_constraint_retirement"] == _NOW
    for slot in PENDING_PROPOSAL_SLOTS:
        if slot != "pending_constraint_retirement":
            assert after[slot] == before[slot], slot


def test_a_rebuilt_task_is_indistinguishable_from_reshowing_every_proposal() -> None:
    # Pins the premise the guard above depends on: if this ever stops renewing,
    # the guard can no longer detect a rebuilding reducer and must be revisited.
    old = _NOW - timedelta(days=6)
    task = _all_slots_task(old)
    rebuilt = ConversationTaskState.model_validate(task.model_dump())

    stamped = rebuilt.stamp_new_proposals(task, _NOW)

    assert stamped.pending_proposed_at == {
        slot: _NOW for slot in PENDING_PROPOSAL_SLOTS
    }


def test_load_for_turn_expires_stale_and_unstamped_proposals_only(tmp_path) -> None:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    now = _NOW
    fresh = JobIntentUpdate(city="上海")
    store.upsert_task(
        user_id="u1",
        conversation_id="c1",
        task=ConversationTaskState(
            pending_job_intent_update=fresh,
            pending_constraint_retirement=_RETIREMENT,
            pending_career_fact=None,
            pending_proposed_at={
                "pending_job_intent_update": now - timedelta(days=1),
                "pending_constraint_retirement": (
                    now - PENDING_PROPOSAL_TTL - timedelta(minutes=1)
                ),
            },
        ),
    )
    store.upsert_task(
        user_id="u1",
        conversation_id="legacy",
        task=ConversationTaskState(
            pending_job_intent_update=fresh,
            bare_confirmation_target="job_intent",
        ),
    )
    manager = ContextManager(store, clock=lambda: now)
    recorder = InMemoryTraceRecorder()
    token = ACTIVE_TRACE_CONTEXT.set((recorder, "turn-1"))
    try:
        context = manager.load_for_turn(
            user_id="u1", conversation_id="c1", user_message="确认那条"
        )
        legacy = manager.load_for_turn(
            user_id="u1", conversation_id="legacy", user_message="确认"
        )
    finally:
        ACTIVE_TRACE_CONTEXT.reset(token)

    assert context.task.pending_job_intent_update == fresh
    assert context.task.pending_constraint_retirement is None
    assert set(context.task.pending_proposed_at) == {"pending_job_intent_update"}
    assert store.get_task("u1", "c1") == context.task
    # No stamp means the proposal's age is unknown; it does not get a new week.
    assert legacy.task.pending_job_intent_update is None
    assert legacy.task.bare_confirmation_target is None

    expired = [
        event.details
        for event in recorder.snapshot("turn-1").events
        if event.event_type == "memory_proposal_expired"
    ]
    assert expired == [
        {
            "conversation_key": conversation_trace_key("u1", "c1"),
            "slots": ["pending_constraint_retirement"],
        },
        {
            "conversation_key": conversation_trace_key("u1", "legacy"),
            "slots": ["pending_job_intent_update"],
        },
    ]


def test_confirm_handler_refuses_an_expired_stored_proposal(
    tmp_path, monkeypatch
) -> None:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    store.upsert_task(
        user_id="u1",
        conversation_id="c1",
        task=ConversationTaskState(
            pending_constraint_retirement=_RETIREMENT,
            pending_proposed_at={
                "pending_constraint_retirement": datetime.now(timezone.utc)
                - timedelta(days=8)
            },
        ),
    )

    def must_not_retire(**_kwargs):
        raise AssertionError("an expired proposal reached the write")

    monkeypatch.setattr(store, "retire_conversation_constraint", must_not_retire)
    registry = MainAgentToolRegistry(conversation_store=store)

    result = registry.invoke_atomic_tool(
        "confirm_constraint_retirement",
        {"user_id": "u1", "conversation_id": "c1", "proposal": _RETIREMENT},
    )

    assert result.state == "constraint_retirement_confirmation_missing"
    assert "7 天" in result.message
    assert result.execution_outcome == "not_committed"


def test_confirm_projection_refuses_an_expired_proposal() -> None:
    def context(proposed_at: datetime) -> MainAgentContext:
        return MainAgentContext(
            conversation_id="c1",
            profile=CareerProfileContext(user_id="u1"),
            task=ConversationTaskState(
                pending_job_intent_update=JobIntentUpdate(city="上海"),
                pending_proposed_at={"pending_job_intent_update": proposed_at},
            ),
            user_message="确认",
        )

    live = project_job_intent_arguments(
        context(datetime.now(timezone.utc)), "confirm_job_intent", {}
    )
    assert live["update"] == JobIntentUpdate(city="上海")

    with pytest.raises(ValueError, match="expired"):
        project_job_intent_arguments(
            context(datetime.now(timezone.utc) - timedelta(days=8)),
            "confirm_job_intent",
            {},
        )
