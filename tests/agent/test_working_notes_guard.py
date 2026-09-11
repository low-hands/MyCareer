from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from time import perf_counter

import pytest

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.conversation_memory_contracts import (
    ConversationSummaryContent,
)
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    CareerProfileContext,
    ConversationMessageContext,
    ConversationTaskState,
    DecisionObservation,
    FreeTextPreferenceContext,
    MainAgentContext,
    SavedJobCandidateContextItem,
    ToolCall,
    ToolObservation,
    WorkingNotesContext,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.agent.tool_effects import is_notes_guarded
from career_agent.agent.working_notes_guard import working_notes_only_tokens
from career_agent.harness.observability import InMemoryTraceRecorder
from career_agent.storage.context import CareerContextStore
from career_agent.storage.working_notes import WorkingNotesStore


_NOW = datetime(2026, 9, 11, 12, tzinfo=timezone.utc)


def _context(**updates) -> MainAgentContext:
    values = {
        "conversation_id": "c1",
        "profile": CareerProfileContext(user_id="u1"),
        "working_notes": WorkingNotesContext(
            markdown="- 未确认观察：用户可能偏好 Rust；不想通勤",
            revision="aaaaaaaaaaaa",
        ),
        "user_message": "帮我看看",
    }
    values.update(updates)
    return MainAgentContext(**values)


def test_note_only_argument_token_is_detected_but_user_text_authorizes_it() -> None:
    assert working_notes_only_tokens(
        arguments={"query": "Rust"}, context=_context()
    ) == ("rust",)
    assert working_notes_only_tokens(
        arguments={"query": "Rust"},
        context=_context(user_message="帮我找 Rust 岗位"),
    ) == ()


@pytest.mark.parametrize("authority", ("recent", "preference", "observation"))
def test_turn_authority_sources_remove_a_note_only_hit(authority: str) -> None:
    updates = {}
    if authority == "recent":
        updates["recent_messages"] = (
            ConversationMessageContext(
                role="user", content="我想看 Rust", created_at=_NOW
            ),
        )
    elif authority == "preference":
        updates["free_text_preferences"] = (
            FreeTextPreferenceContext(
                scope_key="freeform.person_default/rust",
                topic_key="language",
                statement="偏好 Rust",
                status="active",
                observed_at=_NOW,
                confirmed_at=_NOW,
                update_id="intent_update_" + "a" * 32,
            ),
        )
    else:
        updates["tool_observations"] = (
            DecisionObservation(
                tool_name="search_career_memory",
                state="career_memory_search_found",
                message="已找到 Rust 相关的确认记录。",
            ),
        )

    assert working_notes_only_tokens(
        arguments={"query": "Rust"}, context=_context(**updates)
    ) == ()


@pytest.mark.parametrize("authority", ("summary", "profile", "candidate", "body"))
def test_projected_authority_sources_remove_a_note_only_hit(authority: str) -> None:
    updates = {}
    if authority == "summary":
        updates["conversation_summary"] = ConversationSummaryContent(
            active_constraints=("只看 Rust 岗位",)
        )
    elif authority == "profile":
        updates["profile"] = CareerProfileContext(
            user_id="u1", default_city="Rust"
        )
    elif authority == "candidate":
        updates["task"] = ConversationTaskState(
            saved_job_candidates=(
                SavedJobCandidateContextItem(
                    job_posting_id="job-1",
                    title="Rust Engineer",
                    company_name="Example",
                ),
            )
        )
    else:
        updates["tool_observations"] = (
            DecisionObservation(
                tool_name="search_career_memory",
                state="career_memory_search_found",
                message="已完成检索。",
                body="确认记录提到 Rust。",
            ),
        )

    assert working_notes_only_tokens(
        arguments={"query": "Rust"}, context=_context(**updates)
    ) == ()


def test_guard_refusal_does_not_make_its_own_token_authoritative() -> None:
    context = _context(
        tool_observations=(
            DecisionObservation(
                tool_name="find_saved_jobs",
                state="working_notes_derived_argument",
                message="只有工作笔记提到 rust；请确认。",
            ),
        )
    )

    assert working_notes_only_tokens(
        arguments={"query": "Rust"}, context=context
    ) == ("rust",)


def test_a_compacted_unresolved_question_is_not_authority() -> None:
    context = _context(
        conversation_summary=ConversationSummaryContent(
            unresolved_questions=("用户是否偏好 Rust？",)
        )
    )

    assert working_notes_only_tokens(
        arguments={"query": "Rust"}, context=context
    ) == ("rust",)


def test_a_superseded_history_claim_is_not_authority() -> None:
    context = _context(
        working_notes=WorkingNotesContext(
            markdown="- 精通 Rust", revision="aaaaaaaaaaaa"
        ),
        tool_observations=(
            DecisionObservation(
                tool_name="search_career_history",
                state="career_history_found",
                message="找到 1/1 条匹配的历史职业声明。",
                body="以下为已被更正的历史声明，不能作为当前值使用：\n- 精通 Rust",
            ),
        ),
    )

    assert set(
        working_notes_only_tokens(
            arguments={"tailoring_goal": "突出精通 Rust"}, context=context
        )
    ) >= {"rust", "精通"}


def test_an_assistant_echo_of_a_note_is_not_user_authority() -> None:
    context = _context(
        recent_messages=(
            ConversationMessageContext(
                role="assistant", content="你确认偏好 Rust 吗？", created_at=_NOW
            ),
        )
    )

    assert working_notes_only_tokens(
        arguments={"query": "Rust"}, context=context
    ) == ("rust",)


def test_opaque_ids_digests_numbers_and_selection_indexes_are_ignored() -> None:
    opaque = "career_fact_" + "a" * 16
    digest = "sha256:" + "b" * 64
    context = _context(
        working_notes=WorkingNotesContext(
            markdown=f"{opaque} {digest} 123456",
            revision="aaaaaaaaaaaa",
        )
    )

    assert working_notes_only_tokens(
        arguments={
            "ref": opaque,
            "digest": digest,
            "count": 123456,
            "page": "123456",
            "selection_index": "123456",
        },
        context=context,
    ) == ()


def test_cjk_ngrams_detect_literal_note_use_and_accept_user_grounding() -> None:
    assert "通勤" in working_notes_only_tokens(
        arguments={"query": "通勤"}, context=_context()
    )
    assert working_notes_only_tokens(
        arguments={"query": "通勤"},
        context=_context(user_message="通勤太远"),
    ) == ()


def test_update_working_notes_is_the_only_write_exempt_from_the_guard() -> None:
    assert not is_notes_guarded("update_working_notes")
    assert is_notes_guarded("create_application")
    assert is_notes_guarded("find_saved_jobs")
    assert not is_notes_guarded("get_saved_job")
    # Memory searches are the authoritative check the refusal points to.
    assert not is_notes_guarded("search_career_memory")
    assert not is_notes_guarded("search_career_episodes")
    assert not is_notes_guarded("search_career_history")
    assert not is_notes_guarded("read_conversation_span")


class _NeverDecisionMaker:
    def decide(self, context, tool_specs):  # pragma: no cover - direct node test
        raise AssertionError("decision maker should not run")


class _Registry(MainAgentToolRegistry):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, dict]] = []
        self._atomic_handlers["find_saved_jobs"] = self._invoke
        self._atomic_handlers["get_saved_job"] = self._invoke

    def _invoke(self, arguments):
        self.calls.append(("handler", dict(arguments)))
        return ToolObservation(
            tool_name="get_saved_job",
            state="saved_job_not_found",
            message="没有找到。",
        )


class _IdentityProjectionRuntime(MainAgentRuntime):
    @staticmethod
    def _project_atomic_tool_arguments(context, name, arguments):
        return dict(arguments)


def test_runtime_blocks_guarded_handler_but_executes_an_unguarded_read(
    tmp_path,
) -> None:
    manager = ContextManager(CareerContextStore(tmp_path / "context.sqlite3"))
    registry = _Registry()
    runtime = _IdentityProjectionRuntime(
        context_manager=manager,
        decision_maker=_NeverDecisionMaker(),
        tools=registry,
    )
    context = _context()
    blocked_state = {
        "context": context,
        "decision": AgentDecision(
            action="tool_call",
            tool_call=ToolCall(
                name="find_saved_jobs", arguments={"query": "Rust"}
            ),
        ),
        "control": {},
        "pending": {},
    }

    blocked = runtime._authorize(blocked_state)
    assert blocked["authorization_route"] == "observe"
    assert blocked["pending"]["result"].state == "working_notes_derived_argument"
    assert blocked["pending"]["result"].execution_outcome == "not_committed"
    assert blocked["pending"]["result"].payload == {
        "tokens": ["rust"],
        "tool_name": "find_saved_jobs",
    }
    assert registry.calls == []

    allowed_state = {
        **blocked_state,
        "decision": AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name="get_saved_job", arguments={"query": "Rust"}),
        ),
    }
    allowed = runtime._authorize(allowed_state)
    assert allowed["authorization_route"] == "act"
    runtime._act({**allowed_state, **allowed})
    assert len(registry.calls) == 1


def test_owner_confirmed_seal_is_not_rejudged_by_the_notes_guard(
    tmp_path,
) -> None:
    runtime = _IdentityProjectionRuntime(
        context_manager=ContextManager(
            CareerContextStore(tmp_path / "context.sqlite3")
        ),
        decision_maker=_NeverDecisionMaker(),
        tools=_Registry(),
    )
    # The resumed turn's context lacks the observations that grounded the
    # sealed arguments; the owner's approval must still execute.
    state = {
        "context": _context(user_message="确认"),
        "decision": AgentDecision(
            action="tool_call",
            tool_call=ToolCall(name="find_saved_jobs", arguments={}),
        ),
        "control": {},
        "pending": {
            "name": "find_saved_jobs",
            "arguments": {"query": "Rust"},
            "owner_confirmed": True,
            "confirmation_id": "confirmation-1",
        },
    }

    assert runtime._authorize(state)["authorization_route"] == "act"


class _Decisions:
    def __init__(self, *decisions: AgentDecision) -> None:
        self.decisions = list(decisions)

    def decide(self, context, tool_specs):
        return self.decisions.pop(0)


def test_runtime_memory_trace_reports_note_guard_and_omits_counts_without_notes(
    tmp_path,
) -> None:
    notes = WorkingNotesStore(tmp_path / "notes")
    notes.replace(
        user_id="u1",
        markdown="用户可能偏好 Rust",
        expected_revision="empty",
    )
    manager = ContextManager(
        CareerContextStore(tmp_path / "context.sqlite3"),
        working_notes_store=notes,
    )
    manager.upsert_profile(CareerProfileContext(user_id="u1"))
    recorder = InMemoryTraceRecorder()
    registry = _Registry()
    runtime = MainAgentRuntime(
        context_manager=manager,
        decision_maker=_Decisions(
            AgentDecision(
                action="tool_call",
                tool_call=ToolCall(
                    name="find_saved_jobs", arguments={"query": "Rust"}
                ),
            ),
            AgentDecision(action="ask_user", message="你确认偏好 Rust 吗？"),
        ),
        tools=registry,
        trace_recorder=recorder,
    )

    result = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="帮我找岗位"
    )

    memory_events = [
        event
        for events in recorder._events.values()
        for event in events
        if event.event_type == "memory_context_observed"
    ]
    assert memory_events[0].details["working_notes_chars"] > 0
    assert memory_events[0].details["working_notes_only_tokens"] == 1
    assert memory_events[0].details["working_notes_only_argument"] == 1
    assert result.context.tool_observations[-1].state == (
        "working_notes_derived_argument"
    )
    assert registry.calls == []

    empty_recorder = InMemoryTraceRecorder()
    empty_manager = ContextManager(
        CareerContextStore(tmp_path / "empty-context.sqlite3")
    )
    empty_manager.upsert_profile(CareerProfileContext(user_id="u2"))
    MainAgentRuntime(
        context_manager=empty_manager,
        decision_maker=_Decisions(
            AgentDecision(action="final", message="完成。")
        ),
        tools=MainAgentToolRegistry(),
        trace_recorder=empty_recorder,
    ).run_turn(user_id="u2", conversation_id="c2", user_message="继续")
    empty_event = next(
        event
        for events in empty_recorder._events.values()
        for event in events
        if event.event_type == "memory_context_observed"
    )
    assert empty_event.details["working_notes_chars"] == 0
    assert "working_notes_only_tokens" not in empty_event.details
    assert "working_notes_only_argument" not in empty_event.details


@pytest.mark.parametrize(("age_days", "expected"), ((15, 15), (1, None)))
def test_stale_days_projects_only_after_the_threshold(
    tmp_path, age_days: int, expected: int | None
) -> None:
    notes = WorkingNotesStore(tmp_path / f"notes-{age_days}")
    notes.replace(user_id="u1", markdown="保留", expected_revision="empty")
    timestamp = (_NOW - timedelta(days=age_days)).timestamp()
    os.utime(notes._path("u1"), (timestamp, timestamp))
    manager = ContextManager(
        CareerContextStore(tmp_path / f"context-{age_days}.sqlite3"),
        working_notes_store=notes,
        clock=lambda: _NOW,
    )
    projected = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="继续"
    ).model_context()["working_notes"]

    if expected is None:
        assert "stale_days" not in projected
    else:
        assert projected["stale_days"] == expected


def test_guard_stays_under_one_millisecond_for_bounded_inputs() -> None:
    context = _context(
        working_notes=WorkingNotesContext(
            markdown=("Rust 后端岗位，不想通勤。" * 100)[:2000],
            revision="aaaaaaaaaaaa",
        )
    )
    arguments = {"query": "Rust 通勤", "filters": ["backend", "remote"]}
    working_notes_only_tokens(arguments=arguments, context=context)
    started = perf_counter()
    for _ in range(200):
        working_notes_only_tokens(arguments=arguments, context=context)
    average_seconds = (perf_counter() - started) / 200

    assert average_seconds < 0.001
