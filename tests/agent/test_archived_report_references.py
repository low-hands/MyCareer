"""A report stays nameable after its turn scrolls out of the recent window.

``resource_ref`` lives on the message a turn wrote, and the recent window is
loaded from ``through_sequence`` onward. Once summarisation passed a report's
turn, the reference left the agent's context for good while the transcript
endpoint — which reads every message — kept showing the card. The user could
open a report the model could no longer name, so asking about it a week later
got a fresh paid search or an admission of ignorance.

The catalogue is read back from those same messages rather than copied into the
summary: a second record of what a turn delivered is a second thing that can
drift from the first.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.conversation_memory_contracts import (
    ConversationSummaryContent,
)
from career_agent.agent.main_agent_contracts import (
    CareerProfileContext,
    ConversationResourceReference,
    ConversationTaskState,
    MainAgentContext,
)
from career_agent.storage.context import CareerContextStore

_NOW = datetime(2026, 8, 31, tzinfo=timezone.utc)


class _Worker:
    """A summariser that always succeeds, so compaction actually advances."""

    def summarize(self, *, previous, messages):
        return ConversationSummaryContent(user_goals=("找算法岗",))


def _research_ref(resource_id: str) -> ConversationResourceReference:
    return ConversationResourceReference(
        kind="job_research_report",
        resource_id=resource_id,
        status_at_delivery="current",
        anchored_by_other_job=False,
    )


def _manager(tmp_path) -> tuple[ContextManager, CareerContextStore]:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    return (
        ContextManager(
            store,
            summary_worker=_Worker(),
            recent_message_limit=2,
            summary_batch_size=2,
        ),
        store,
    )


def _turn(manager, *, index: int, ref=None) -> None:
    context = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message=f"第 {index} 轮"
    )
    manager.commit_turn(
        context=context,
        task=ConversationTaskState(),
        assistant_message=f"回复 {index}",
        assistant_resource_ref=ref,
    )


def test_a_report_stays_nameable_after_its_turn_is_summarised(tmp_path) -> None:
    manager, _ = _manager(tmp_path)
    _turn(manager, index=1, ref=_research_ref("report-1"))
    for index in range(2, 7):
        _turn(manager, index=index)

    context = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="那份调研提到的竞品是谁"
    )

    # The delivering turn is long gone from the window.
    assert all(
        message.resource_ref is None for message in context.recent_messages
    )
    # It is still reachable, and resolvable back to the internal id.
    assert [ref.resource_id for ref in context.referenced_resources()] == ["report-1"]
    assert (
        context.resolve_reference_index(
            reference_index=1, kind="job_research_report"
        )
        == "report-1"
    )


def test_the_catalogue_reaches_the_model_with_a_bounded_label(tmp_path) -> None:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    manager = ContextManager(
        store, summary_worker=_Worker(), recent_message_limit=2, summary_batch_size=2
    )
    context = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="帮我调研这家公司"
    )
    manager.commit_turn(
        context=context,
        task=ConversationTaskState(),
        assistant_message="岗位研究已完成。这家公司近年主要投入企业级搜索。\n正文" * 200,
        assistant_resource_ref=_research_ref("report-1"),
    )
    for index in range(2, 7):
        _turn(manager, index=index)

    projected = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="继续"
    ).model_context()

    entry = projected["archived_reports"][0]
    assert entry["kind"] == "job_research_report"
    assert entry["reference_index"] == 1
    # A catalogue carried every turn for the life of a conversation cannot grow
    # with the reports it lists.
    assert len(entry["summary"]) <= 220
    assert "resource_id" not in entry
    assert "report-1" not in str(projected)


def test_numbering_is_continuous_across_the_catalogue_and_the_window(tmp_path) -> None:
    """The two loops that number references must not drift.

    ``model_context`` shows an index and ``resolve_reference_index`` resolves it.
    They are produced by different code walking the same two lists, so an
    off-by-one hands the model report 3 when it asked for report 2 — silent,
    and wrong in the worst possible way.
    """
    manager, _ = _manager(tmp_path)
    _turn(manager, index=1, ref=_research_ref("old-report"))
    for index in range(2, 7):
        _turn(manager, index=index)
    context = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="再调研一次"
    )
    manager.commit_turn(
        context=context,
        task=ConversationTaskState(),
        assistant_message="岗位研究已完成。",
        assistant_resource_ref=_research_ref("new-report"),
    )

    context = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="比较一下两份"
    )
    projected = context.model_context()
    shown = [
        entry["reference_index"] for entry in projected["archived_reports"]
    ] + [
        message["resource"]["reference_index"]
        for message in projected["recent_messages"]
        if "resource" in message
    ]

    assert shown == list(range(1, len(context.referenced_resources()) + 1))
    resolved = [
        context.resolve_reference_index(
            reference_index=index, kind="job_research_report"
        )
        for index in shown
    ]
    assert resolved == [ref.resource_id for ref in context.referenced_resources()]
    # Oldest first, so an index does not mean a different report depending on
    # which side of the window boundary it fell.
    assert resolved == ["old-report", "new-report"]


def test_pruning_summarised_originals_keeps_the_reports_reachable(tmp_path) -> None:
    """Reclaiming disk must not quietly cost the agent its handles.

    Pruning deletes messages a summary already covers. A delivering turn's row
    is one of those, and it is also the only path back to the report — while its
    content is a bounded line, so keeping it reclaims almost nothing.
    """
    manager, store = _manager(tmp_path)
    _turn(manager, index=1, ref=_research_ref("report-1"))
    for index in range(2, 7):
        _turn(manager, index=index)

    deleted = store.prune_compacted_messages(user_id="u1", conversation_id="c1")

    assert deleted > 0
    context = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="那份调研呢"
    )
    assert [ref.resource_id for ref in context.referenced_resources()] == ["report-1"]


def test_the_catalogue_is_empty_before_anything_has_scrolled_away(tmp_path) -> None:
    """No summary means the window already holds every reference."""
    manager, _ = _manager(tmp_path)
    _turn(manager, index=1, ref=_research_ref("report-1"))

    context = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="继续"
    )

    assert context.archived_resources == ()
    assert len(context.referenced_resources()) == 1


def test_the_catalogue_is_bounded_however_many_reports_a_conversation_holds(
    tmp_path,
) -> None:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    manager = ContextManager(
        store, summary_worker=_Worker(), recent_message_limit=2, summary_batch_size=2
    )
    for index in range(1, 21):
        _turn(manager, index=index, ref=_research_ref(f"report-{index}"))

    context = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="继续"
    )

    assert len(context.archived_resources) == 12
    # The newest survive a trim: an older report is likelier to have been
    # superseded, and the recent window covers the very newest anyway.
    assert context.archived_resources[-1].resource_ref.resource_id == "report-19"


def test_the_limit_is_validated_rather_than_silently_clamped(tmp_path) -> None:
    with pytest.raises(ValueError, match="archived resource limit"):
        ContextManager(
            CareerContextStore(tmp_path / "context.sqlite3"),
            archived_resource_limit=99,
        )


def test_the_reclaim_notice_counts_only_what_a_prune_would_delete(tmp_path) -> None:
    """The invariant that keeping delivering rows quietly broke.

    Delivering turns are excluded from the prune so a report stays reachable.
    They were still counted as reclaimable, so once enough reports accumulated
    the operator was told there was disk to reclaim, ran the command, and was
    told the same thing again — permanently, with nothing to do about it.
    """
    manager, store = _manager(tmp_path)
    _turn(manager, index=1, ref=_research_ref("report-1"))
    for index in range(2, 7):
        _turn(manager, index=index)

    before, _ = store.count_compacted_messages(user_id="u1", conversation_id="c1")
    deleted = store.prune_compacted_messages(user_id="u1", conversation_id="c1")
    after, byte_size = store.count_compacted_messages(
        user_id="u1", conversation_id="c1"
    )

    assert deleted == before
    # Nothing left to offer, even though the delivering row is still stored.
    assert (after, byte_size) == (0, 0)
    assert [
        ref.resource_id
        for ref in manager.load_for_turn(
            user_id="u1", conversation_id="c1", user_message="那份调研呢"
        ).referenced_resources()
    ] == ["report-1"]
