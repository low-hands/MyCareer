"""A report stays nameable after its turn scrolls out of the recent window.

``resource_refs`` lives on the message a turn wrote, and the recent window is
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
            max_recent_context_chars=16,
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
        assistant_resource_refs=(ref,) if ref is not None else (),
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
        not message.resource_refs for message in context.recent_messages
    )
    # It is still reachable, and resolvable back to the internal id.
    assert [ref.resource_id for ref in context.referenced_resources()] == ["report-1"]
    handle = context.reference_handle(context.referenced_resources()[0])
    assert (
        context.resolve_reference(reference=handle, kind="job_research_report")
        == "report-1"
    )


def test_default_budget_compacts_before_a_short_report_turn_leaves_projection(
    tmp_path,
) -> None:
    """The count cap cannot hide originals that have no watermark yet."""

    store = CareerContextStore(tmp_path / "context.sqlite3")
    manager = ContextManager(store, summary_worker=_Worker())
    _turn(manager, index=1, ref=_research_ref("report-1"))
    for index in range(2, 21):
        _turn(manager, index=index)

    context = manager.load_for_turn(
        user_id="u1",
        conversation_id="c1",
        user_message="第一轮那份报告怎么说",
    )

    assert context.through_sequence >= 2
    assert len(context.recent_messages) <= 11
    assert all(
        ref.resource_id != "report-1"
        for message in context.recent_messages
        for ref in message.resource_refs
    )
    assert [ref.resource_id for ref in context.referenced_resources()] == [
        "report-1"
    ]


def test_the_catalogue_uses_resource_metadata_not_delivery_prose(tmp_path) -> None:
    store = CareerContextStore(tmp_path / "context.sqlite3")
    manager = ContextManager(
        store,
        summary_worker=_Worker(),
        recent_message_limit=2,
        summary_batch_size=2,
        max_recent_context_chars=16,
    )
    context = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="帮我调研这家公司"
    )
    manager.commit_turn(
        context=context,
        task=ConversationTaskState(),
        assistant_message="岗位研究已完成。这家公司近年主要投入企业级搜索。\n正文" * 200,
        assistant_resource_refs=(_research_ref("report-1"),),
    )
    for index in range(2, 7):
        _turn(manager, index=index)

    projected = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="继续"
    ).model_context()

    entry = projected["archived_reports"]["items"][0]
    assert entry["kind"] == "job_research_report"
    assert entry["reference"].startswith("report_")
    # Delivery prose belongs to the conversation, not to the resource index.
    # A producer-owned description is the only supported preview.
    assert "summary" not in entry
    assert "resource_id" not in entry
    assert "report-1" not in str(projected)


def test_the_catalogue_and_the_window_name_reports_the_same_way(tmp_path) -> None:
    """A report keeps its name when it crosses the window boundary.

    Under ordinals this test guarded against two loops drifting, because a
    report's number depended on where it sat: summarising a turn could renumber
    everything after it. A derived handle has no such dependence, so what is
    worth checking now is the property that replaced it — the same report is
    named the same way whether it reaches the model through the catalogue or
    through the window, and every name shown resolves to what it was shown for.
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
        assistant_resource_refs=(_research_ref("new-report"),),
    )

    context = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="比较一下两份"
    )
    projected = context.model_context()
    shown = [
        entry["reference"] for entry in projected["archived_reports"]["items"]
    ] + [
        entry["reference"]
        for message in projected["recent_messages"]
        for entry in message.get("resources", ())
    ]

    resolved = [
        context.resolve_reference(reference=handle, kind="job_research_report")
        for handle in shown
    ]
    # One report reached through the catalogue, one through the window, and the
    # catalogue one is the older — the reading order is kept even though nothing
    # depends on it any more.
    assert resolved == ["old-report", "new-report"]
    # The names are the resources', not the positions'. Rebuilding the same
    # references in a context where they sit somewhere else yields the same
    # handles.
    assert shown == [
        context.reference_handle(reference)
        for reference in context.referenced_resources()
    ]


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
        store,
        summary_worker=_Worker(),
        recent_message_limit=2,
        summary_batch_size=2,
        max_recent_context_chars=16,
    )
    for index in range(1, 21):
        _turn(manager, index=index, ref=_research_ref(f"report-{index}"))

    context = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="继续"
    )

    assert len(context.archived_resources) == 12
    # The newest survive a trim: an older report is likelier to have been
    # superseded, and the recent window covers the very newest anyway.
    assert context.archived_resources[-1].resource_refs[0].resource_id == "report-19"


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


def test_a_capped_catalogue_says_how_much_it_is_not_showing(tmp_path) -> None:
    """Twelve entries with nothing else said read as everything there is.

    ``archived_resource_limit`` is 12 and capped at 12, so the thirteenth-oldest
    report leaves the projection entirely — no handle, no line. The user still
    asks about it: it was delivered to them and their transcript still shows it.

    Measured behaviour is that the model reaches for the nearest plausible entry
    when it believes the thing it wants is on screen, so the projection must
    stop implying that. A count rather than a flag, for the reason that made
    ``next_action`` prose: "there are more" leaves the model guessing how many,
    while 12 of 15 says how far short the list falls.
    """
    manager, _ = _manager(tmp_path)
    for number in range(1, 16):
        _turn(manager, index=number, ref=_research_ref(f"report-{number}"))
    for number in range(16, 22):
        _turn(manager, index=number)

    context = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="上个月那份调研呢"
    )
    projected = context.model_context()

    catalogue = projected["archived_reports"]
    assert len(catalogue["items"]) == 12
    assert catalogue["unlisted"] == "另有 3 份更早的调研未列出，无法按引用取回。"
    # The three it cannot show are genuinely unreachable, not merely unlisted:
    # nothing else in the projection names them either.
    named = {entry["reference"] for entry in catalogue["items"]}
    assert set(context.reference_handles()) >= named
    assert len(context.reference_handles()) < 15


def test_a_complete_catalogue_says_so_by_agreeing_with_itself(tmp_path) -> None:
    """The signal has to distinguish, or it is noise on every turn."""
    manager, _ = _manager(tmp_path)
    for number in range(1, 4):
        _turn(manager, index=number, ref=_research_ref(f"report-{number}"))
    for number in range(4, 10):
        _turn(manager, index=number)

    projected = manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="继续"
    ).model_context()

    catalogue = projected["archived_reports"]
    assert len(catalogue["items"]) == 3
    assert catalogue["unlisted"] is None
