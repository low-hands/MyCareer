from career_agent.agent.context_manager import ContextManager
from career_agent.agent.conversation_memory_contracts import (
    ConversationSummaryContent,
)
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    ConstraintRetirementProposal,
    ToolCall,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.harness.streaming import InteractionResponse
from career_agent.storage.capability_confirmations import SQLiteCapabilityConfirmationStore
from career_agent.storage.context import CareerContextStore
from conftest import enter_tool_profile


class SequenceDecisionMaker:
    def __init__(self, *decisions: AgentDecision) -> None:
        self.decisions = list(decisions)

    def decide(self, context, tool_specs):
        return self.decisions.pop(0)


def _tool(name: str, arguments: dict | None = None) -> AgentDecision:
    return AgentDecision(
        action="tool_call",
        tool_call=ToolCall(name=name, arguments=arguments or {}),
    )


def _final() -> AgentDecision:
    return AgentDecision(action="final", message="")


def _runtime(context: CareerContextStore, *decisions, profile=None):
    if profile is not None:
        enter_tool_profile(context, profile)
    return MainAgentRuntime(
        context_manager=ContextManager(context),
        decision_maker=SequenceDecisionMaker(*decisions),
        tools=MainAgentToolRegistry(conversation_store=context),
        capability_confirmation_store=SQLiteCapabilityConfirmationStore(
            context.path.parent / "confirmations.sqlite3"
        ),
    )


def _seed(context: CareerContextStore, *constraints: str, omitted=()) -> None:
    context.compact_conversation_summary(
        user_id="u1",
        conversation_id="c1",
        expected_previous_through_sequence=0,
        content=ConversationSummaryContent(active_constraints=constraints),
        through_sequence=1,
        omitted_constraints=omitted,
    )


def test_retirement_cannot_execute_in_the_turn_that_prepared_it(
    tmp_path,
) -> None:
    context = CareerContextStore(tmp_path / "context.sqlite3")
    _seed(context, "不接受 996")
    runtime = _runtime(
        context,
        _tool(
            "propose_constraint_retirement",
            {"constraint": "不接受 996", "reason": "换了岗位。"},
        ),
        _final(),
        profile="memory",
    )

    premature = runtime._tools.invoke_atomic_tool(
        "confirm_constraint_retirement",
        {
            "user_id": "u1",
            "conversation_id": "c1",
            "proposal": ConstraintRetirementProposal(
                target_kind="conversation_constraint",
                constraint="不接受 996",
                reason="换了岗位。",
            ),
        },
    )
    assert premature.state == "constraint_retirement_confirmation_missing"

    result = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="这条约束不用管了。",
    )

    assert [item.state for item in result.tool_results] == [
        "constraint_retirement_proposed"
    ]
    live = context.list_conversation_constraints(
        user_id="u1", conversation_id="c1", statuses=("active",)
    )
    assert tuple(row.text for row in live) == ("不接受 996",)


def test_a_confirmed_retirement_stops_the_constraint_applying(tmp_path) -> None:
    context = CareerContextStore(tmp_path / "context.sqlite3")
    _seed(context, "不接受 996")
    runtime = _runtime(
        context,
        _tool(
            "propose_constraint_retirement",
            {"constraint": "不接受 996", "reason": "换了岗位。"},
        ),
        _final(),
        profile="memory",
    )

    proposed = runtime.run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="这条约束不用管了。",
    )
    gate = MainAgentRuntime._interaction_event(result=proposed, conversation_id="c1")
    assert gate is not None and gate.scope == "capability_confirmation"
    assert (
        context.get_task("u1", "c1").pending_constraint_retirement is not None
    )

    second = _runtime(context, profile="memory").run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="确认退役",
        interaction_response=InteractionResponse(
            interaction_id=gate.interaction_id,
            scope="capability_confirmation",
            action="confirm",
        ),
    )

    assert [item.state for item in second.tool_results] == ["constraint_retired"]
    assert context.get_task("u1", "c1").pending_constraint_retirement is None
    retired = context.list_conversation_constraints(
        user_id="u1", conversation_id="c1", statuses=("retired",)
    )
    assert tuple(row.text for row in retired) == ("不接受 996",)
    summary = context.get_conversation_summary(
        user_id="u1", conversation_id="c1"
    )
    assert summary is not None
    assert summary.content.active_constraints == ()
    repeated = _runtime(context, profile="memory").run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="再次确认",
        interaction_response=InteractionResponse(
            interaction_id=gate.interaction_id,
            scope="capability_confirmation",
            action="confirm",
        ),
    )
    assert repeated.tool_result.state == "capability_confirmation_expired"


def test_a_second_click_on_the_same_seal_does_not_retire_twice(tmp_path) -> None:
    """A second click on a settled seal is answered, not executed again.

    This pins the runtime-visible half of exactly-once: after the first
    confirmation retires the constraint, the same interaction id comes back
    (resent request, second tab, impatient double click) and must produce the
    settled answer with no second write.

    What stops it here is the cleared proposal slot, not the store: by the time
    the second click arrives the seal is no longer PENDING, so the lookup finds
    nothing to claim. The store's conditional PENDING -> APPLYING transition is
    the guard for the racing case, and it is pinned separately by
    ``tests/storage/test_capability_confirmations.py``. Do not read this test
    as covering that race; a sequential double click cannot reach it.
    """

    context = CareerContextStore(tmp_path / "context.sqlite3")
    _seed(context, "不接受 996")
    proposed = _runtime(
        context,
        _tool(
            "propose_constraint_retirement",
            {"constraint": "不接受 996", "reason": "换了岗位。"},
        ),
        _final(),
        profile="memory",
    ).run_turn(
        user_id="u1", conversation_id="c1", user_message="这条约束不用管了。",
    )
    gate = MainAgentRuntime._interaction_event(result=proposed, conversation_id="c1")
    assert gate is not None

    def click():
        return _runtime(context, profile="memory").run_turn(
            user_id="u1",
            conversation_id="c1",
            user_message="确认退役",
            interaction_response=InteractionResponse(
                interaction_id=gate.interaction_id,
                scope="capability_confirmation",
                action="confirm",
            ),
        )

    first = click()
    assert [item.state for item in first.tool_results] == ["constraint_retired"]

    second = click()
    assert [item.state for item in second.tool_results] == [
        "capability_confirmation_expired"
    ]
    # Sequential clicks are stopped by the cleared proposal slot, before the
    # store is consulted. The concurrent case below is what pins the store's
    # conditional transition; keep both, they fail for different reasons.

    # The durable effect happened once: one retired row, and the proposal slot
    # stays cleared rather than being re-armed by the second click.
    retired = context.list_conversation_constraints(
        user_id="u1", conversation_id="c1", statuses=("retired",)
    )
    assert tuple(row.text for row in retired) == ("不接受 996",)
    assert context.get_task("u1", "c1").pending_constraint_retirement is None


def test_text_confirmation_only_reoffers_the_sealed_retirement(tmp_path) -> None:
    context = CareerContextStore(tmp_path / "context.sqlite3")
    _seed(context, "不接受 996")
    proposed = _runtime(
        context,
        _tool(
            "propose_constraint_retirement",
            {"constraint": "不接受 996", "reason": "用户改变要求。"},
        ),
        _final(),
        profile="memory",
    ).run_turn(user_id="u1", conversation_id="c1", user_message="退掉这条约束")
    original_gate = MainAgentRuntime._interaction_event(
        result=proposed, conversation_id="c1"
    )
    assert original_gate is not None

    text_turn = _runtime(
        context, _tool("confirm_constraint_retirement"), _final(), profile="memory"
    ).run_turn(user_id="u1", conversation_id="c1", user_message="确认")
    assert text_turn.tool_result.state == "capability_confirmation_required"
    assert tuple(
        row.text for row in context.list_conversation_constraints(
            user_id="u1", conversation_id="c1", statuses=("active",)
        )
    ) == ("不接受 996",)
    repeated_gate = MainAgentRuntime._interaction_event(
        result=text_turn, conversation_id="c1"
    )
    assert repeated_gate is not None
    assert repeated_gate.interaction_id == original_gate.interaction_id

    cancelled = _runtime(context, profile="memory").run_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="取消",
        interaction_response=InteractionResponse(
            interaction_id=original_gate.interaction_id,
            scope="capability_confirmation",
            action="cancel",
        ),
    )
    assert cancelled.tool_result.state == "capability_confirmation_cancelled"
    assert tuple(
        row.text for row in context.list_conversation_constraints(
            user_id="u1", conversation_id="c1", statuses=("active",)
        )
    ) == ("不接受 996",)


def test_missing_seal_store_refuses_retirement_even_after_text_consent(tmp_path) -> None:
    context = CareerContextStore(tmp_path / "context.sqlite3")
    _seed(context, "不接受 996")
    _runtime(
        context,
        _tool(
            "propose_constraint_retirement",
            {"constraint": "不接受 996", "reason": "用户改变要求。"},
        ),
        _final(),
        profile="memory",
    ).run_turn(user_id="u1", conversation_id="c1", user_message="退掉约束")
    runtime = MainAgentRuntime(
        context_manager=ContextManager(context),
        decision_maker=SequenceDecisionMaker(
            _tool("confirm_constraint_retirement"), _final()
        ),
        tools=MainAgentToolRegistry(conversation_store=context),
    )
    refused = runtime.run_turn(
        user_id="u1", conversation_id="c1", user_message="确认"
    )
    assert all(item.state != "constraint_retired" for item in refused.tool_results)
    assert tuple(
        row.text for row in context.list_conversation_constraints(
            user_id="u1", conversation_id="c1", statuses=("active",)
        )
    ) == ("不接受 996",)


def test_a_constraint_the_conversation_never_recorded_cannot_be_retired(
    tmp_path,
) -> None:
    """The readback has to name something the user actually saw."""

    context = CareerContextStore(tmp_path / "context.sqlite3")
    _seed(context, "不接受 996")
    runtime = _runtime(context)

    refused = runtime._tools.invoke_atomic_tool(
        "propose_constraint_retirement",
        {
            "user_id": "u1",
            "conversation_id": "c1",
            "proposal": ConstraintRetirementProposal(
                target_kind="conversation_constraint",
                constraint="不接受出差",
                reason="模型自己想出来的。",
            ),
        },
    )

    assert refused.state == "constraint_not_found"
    assert refused.execution_outcome == "not_committed"


def test_an_archived_constraint_can_be_retired_without_being_visible(
    tmp_path,
) -> None:
    context = CareerContextStore(tmp_path / "context.sqlite3")
    _seed(context, "不接受 996", omitted=("不接受出差",))
    runtime = _runtime(context)

    proposed = runtime._tools.invoke_atomic_tool(
        "propose_constraint_retirement",
        {
            "user_id": "u1",
            "conversation_id": "c1",
            "proposal": ConstraintRetirementProposal(
                target_kind="conversation_constraint",
                constraint="不接受出差",
                reason="用户说不再适用。",
            ),
        },
    )

    assert proposed.state == "constraint_retirement_proposed"


def test_the_archive_fetch_reports_the_constraints_the_cap_held_back(
    tmp_path,
) -> None:
    context = CareerContextStore(tmp_path / "context.sqlite3")
    _seed(context, "不接受 996", omitted=("不接受出差", "不接受降薪"))
    runtime = _runtime(context)

    fetched = runtime._tools.invoke_atomic_tool(
        "fetch_archived_constraints",
        {"user_id": "u1", "conversation_id": "c1"},
    )

    assert fetched.state == "archived_constraints_ready"
    # The texts have to be in the message: that is the part of an observation
    # the decision model reads.
    assert "不接受出差" in fetched.message
    assert "不接受降薪" in fetched.message
    assert "不接受 996" not in fetched.message


def test_the_archive_fetch_says_so_when_nothing_is_held_back(tmp_path) -> None:
    context = CareerContextStore(tmp_path / "context.sqlite3")
    _seed(context, "不接受 996")
    runtime = _runtime(context)

    fetched = runtime._tools.invoke_atomic_tool(
        "fetch_archived_constraints",
        {"user_id": "u1", "conversation_id": "c1"},
    )

    assert fetched.state == "no_archived_constraints"


def test_retirement_survives_a_compaction_that_re_extracts_the_same_text(
    tmp_path,
) -> None:
    """Retirement beats a stale read of the ledger.

    The caller decides the visible set outside the compaction transaction, so
    a retirement landing in between must still win.
    """

    context = CareerContextStore(tmp_path / "context.sqlite3")
    _seed(context, "不接受 996")
    assert context.retire_conversation_constraint(
        user_id="u1", conversation_id="c1", constraint_text="不接受 996"
    )

    assert context.compact_conversation_summary(
        user_id="u1",
        conversation_id="c1",
        expected_previous_through_sequence=1,
        content=ConversationSummaryContent(
            active_constraints=("不接受 996", "不接受降薪")
        ),
        through_sequence=3,
    )

    summary = context.get_conversation_summary(
        user_id="u1", conversation_id="c1"
    )
    assert summary is not None
    assert summary.content.active_constraints == ("不接受降薪",)
