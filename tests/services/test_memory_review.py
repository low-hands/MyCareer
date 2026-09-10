from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pytest

from career_agent.agent.main_agent_contracts import (
    ConversationMessageContext,
    ConversationTaskState,
    MemoryTombstoneProposal,
)
from career_agent.agent.main_agent_tools import MainAgentToolRegistry
from career_agent.services.memory_review import MemoryReviewService
from career_agent.services.intent_capture import IntentCaptureCandidate
from career_agent.storage.career_history import CareerHistoryStore
from career_agent.storage.context import CareerContextStore
from career_agent.storage.intent_versions import intent_entry_id
from career_agent.storage.working_notes import WorkingNotesStore


def _stores(tmp_path):
    context = CareerContextStore(tmp_path / "context.sqlite3")
    history = CareerHistoryStore(tmp_path / "career.sqlite3")
    service = MemoryReviewService(
        context_store=context,
        career_history_store=history,
    )
    record = history.create_record(
        user_id="u1",
        record_type="work",
        title="AI Engineer",
        organization="Example",
    )
    items = []
    for claim in ("Led the retrieval project.", "Owned the evaluation suite."):
        items.append(
            history.confirm_evidence(
                user_id="u1",
                career_evidence_id=history.create_evidence(
                    user_id="u1",
                    career_record_id=record.id,
                    claim=claim,
                    origin="user_input",
                ).id,
            )
        )
    return context, history, service, record, items


def _remove_block(markdown: str, update_id: str) -> str:
    return re.sub(
        rf'\n<!-- memory:start[^\n]+update_id="{update_id}" -->\n.*?'
        rf'\n<!-- memory:end[^\n]+update_id="{update_id}" -->',
        "",
        markdown,
        count=1,
        flags=re.DOTALL,
    )


def test_memory_review_files_are_not_exposed_as_model_tools(tmp_path) -> None:
    context, history, _, _, _ = _stores(tmp_path)
    registry = MainAgentToolRegistry(
        career_history_store=history,
        conversation_store=context,
        career_profile_store=context,
    )

    assert "export_memory_review" not in registry.names
    assert "review_memory_file" not in registry.names
    assert {
        schema["function"]["name"] for schema in registry.schemas()
    }.isdisjoint({"export_memory_review", "review_memory_file"})


def test_export_edit_delete_add_confirm_and_export_round_trip(tmp_path) -> None:
    context, history, service, record, items = _stores(tmp_path)
    markdown, _, count = service.export(user_id="u1")

    assert count == 2
    for item in items:
        assert (
            f'scope_key="{item.scope_key}" update_id="{item.update_id}"'
            in markdown
        )
    edited = markdown.replace(
        "- Led the retrieval project.",
        "- Led the retrieval and ranking project.",
    )
    edited = _remove_block(edited, items[1].update_id)
    record_marker = f'<!-- memory:record record_id="{record.id}" -->'
    edited = edited.replace(
        record_marker,
        record_marker + "\n- Mentored two engineers.",
        1,
    )

    prepared = service.prepare(user_id="u1", markdown=edited)
    assert [
        len(getattr(prepared.analysis, field))
        for field in ("amendments", "tombstones", "new_facts")
    ] == [1, 1, 1]
    applied = service.apply(
        user_id="u1",
        markdown=edited,
        confirmation_digest=prepared.confirmation_digest,
    )
    assert applied.amendments == 1
    assert applied.tombstones == 1
    assert applied.new_facts == 1

    claims = {
        item.claim
        for item in history.list_evidence(
            user_id="u1",
            verification_status="confirmed",
        )
    }
    assert claims == {
        "Led the retrieval and ranking project.",
        "Mentored two engineers.",
    }
    exported_again, _, _ = service.export(user_id="u1")
    check = service.analyze(user_id="u1", markdown=exported_again)
    assert check.proposal_count == 0
    assert not check.conflicts


def test_a_modified_anchor_comment_ignores_the_whole_block(tmp_path) -> None:
    _, _, service, _, items = _stores(tmp_path)
    markdown, _, _ = service.export(user_id="u1")
    broken = markdown.replace(
        f'update_id="{items[0].update_id}" -->',
        f'update_id="{items[0].update_id}" unexpected="1" -->',
        1,
    ).replace(
        "- Led the retrieval project.",
        "- This edit must not become a proposal.",
    )

    analysis = service.analyze(user_id="u1", markdown=broken)

    assert analysis.warnings
    assert all(
        proposal.new_claim != "This edit must not become a proposal."
        for proposal in analysis.amendments
    )
    assert all(
        proposal.detail_ref != items[0].detail_ref
        for proposal in analysis.tombstones
    )


def test_repeating_a_review_is_read_only_and_creates_no_pending_fact(tmp_path) -> None:
    _, history, service, record, _ = _stores(tmp_path)
    markdown, _, _ = service.export(user_id="u1")
    record_marker = f'<!-- memory:record record_id="{record.id}" -->'
    edited = markdown.replace(
        record_marker,
        record_marker + "\n- Mentored two engineers.",
        1,
    )

    first = service.analyze(user_id="u1", markdown=edited)
    second = service.analyze(user_id="u1", markdown=edited)

    assert len(first.new_facts) == len(second.new_facts) == 1
    assert first.new_facts == second.new_facts
    pending = history.list_evidence(
        user_id="u1",
        career_record_id=record.id,
        verification_status="pending",
        include_historical=False,
    )
    assert pending == ()


def test_a_partially_applied_batch_can_be_reviewed_and_resumed(
    tmp_path,
    monkeypatch,
) -> None:
    _, history, service, _, _ = _stores(tmp_path)
    markdown, _, _ = service.export(user_id="u1")
    edited = markdown.replace(
        "- Led the retrieval project.",
        "- Led retrieval and ranking.",
    ).replace(
        "- Owned the evaluation suite.",
        "- Owned retrieval evaluation.",
    )
    prepared = service.prepare(user_id="u1", markdown=edited)
    original_correct = history.correct_evidence
    calls = 0

    def fail_second_correction(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected second-write failure")
        return original_correct(**kwargs)

    monkeypatch.setattr(history, "correct_evidence", fail_second_correction)
    with pytest.raises(OSError, match="second-write failure"):
        service.apply(
            user_id="u1",
            markdown=edited,
            confirmation_digest=prepared.confirmation_digest,
        )
    monkeypatch.setattr(history, "correct_evidence", original_correct)

    resumed = service.prepare(user_id="u1", markdown=edited)
    assert not resumed.analysis.conflicts
    assert len(resumed.analysis.amendments) == 1
    service.apply(
        user_id="u1",
        markdown=edited,
        confirmation_digest=resumed.confirmation_digest,
    )
    assert {
        item.claim
        for item in history.list_evidence(
            user_id="u1",
            verification_status="confirmed",
        )
    } == {"Led retrieval and ranking.", "Owned retrieval evaluation."}


def test_an_applied_preference_amendment_is_a_noop_when_batch_resumes(
    tmp_path,
    monkeypatch,
) -> None:
    context, history, service, record, _ = _stores(tmp_path)
    _active_preference(
        context,
        value="默认优先小团队",
        pref_scope="freeform.person_default",
        layer="contextual",
    )
    markdown, _, _ = service.export(user_id="u1")
    marker = f'<!-- memory:record record_id="{record.id}" -->'
    edited = markdown.replace(
        "- 默认优先小团队",
        "- 默认优先中小团队",
    ).replace(
        marker,
        marker + "\n- Mentored two engineers.",
        1,
    )
    prepared = service.prepare(user_id="u1", markdown=edited)
    original_confirm = history.confirm_evidence

    def fail_new_fact_confirmation(**_kwargs):
        raise OSError("injected post-preference failure")

    monkeypatch.setattr(history, "confirm_evidence", fail_new_fact_confirmation)
    with pytest.raises(OSError, match="post-preference failure"):
        service.apply(
            user_id="u1",
            markdown=edited,
            confirmation_digest=prepared.confirmation_digest,
        )
    monkeypatch.setattr(history, "confirm_evidence", original_confirm)

    resumed = service.prepare(user_id="u1", markdown=edited)
    assert not resumed.analysis.conflicts
    assert resumed.analysis.preference_amendments == ()
    assert len(resumed.analysis.new_facts) == 1
    service.apply(
        user_id="u1",
        markdown=edited,
        confirmation_digest=resumed.confirmation_digest,
    )

    assert {
        item.value
        for item in context.list_free_text_preferences(
            user_id="u1",
            statuses=("active",),
        )
    } == {"默认优先中小团队"}
    assert "Mentored two engineers." in {
        item.claim
        for item in history.list_evidence(
            user_id="u1",
            verification_status="confirmed",
        )
    }


def test_batch_preflight_rejects_a_changed_tombstone_before_any_write(
    tmp_path,
    monkeypatch,
) -> None:
    _, history, service, _, items = _stores(tmp_path)
    markdown, _, _ = service.export(user_id="u1")
    edited = markdown.replace(
        "- Led the retrieval project.",
        "- Led retrieval and ranking.",
    )
    edited = _remove_block(edited, items[1].update_id)
    prepared = service.prepare(user_id="u1", markdown=edited)
    history.correct_evidence(
        user_id="u1",
        career_evidence_id=items[1].id,
        new_claim="Concurrent evaluation edit.",
        reason="Concurrent update",
    )
    monkeypatch.setattr(service, "prepare", lambda **_kwargs: prepared)

    with pytest.raises(ValueError, match="deletion target changed"):
        service.apply(
            user_id="u1",
            markdown=edited,
            confirmation_digest=prepared.confirmation_digest,
        )

    current = history.get_evidence_by_detail_ref(
        user_id="u1",
        detail_ref=items[0].detail_ref,
    )
    assert current is not None
    assert current.claim == "Led the retrieval project."


def test_a_new_database_revision_rejects_the_stale_review(tmp_path) -> None:
    _, history, service, _, items = _stores(tmp_path)
    markdown, _, _ = service.export(user_id="u1")
    history.correct_evidence(
        user_id="u1",
        career_evidence_id=items[0].id,
        new_claim="Database won the race.",
        reason="Concurrent update",
    )
    edited = markdown.replace(
        "- Led the retrieval project.",
        "- Stale file edit.",
    )

    analysis = service.analyze(user_id="u1", markdown=edited)

    assert analysis.conflicts
    assert not analysis.amendments


def test_an_export_snapshot_cannot_cross_users(tmp_path) -> None:
    _, _, service, _, _ = _stores(tmp_path)
    markdown, _, _ = service.export(user_id="u1")

    try:
        service.analyze(user_id="u2", markdown=markdown)
    except ValueError as error:
        assert "不属于当前用户" in str(error)
    else:
        raise AssertionError("cross-user review snapshot was accepted")


def _active_preference(
    context: CareerContextStore,
    *,
    value: str,
    pref_scope: str,
    layer: str,
    timescale: str = "permanent",
):
    now = datetime.now(timezone.utc)
    _, version = context.capture_profile_intent(
        IntentCaptureCandidate(
            user_id="u1",
            scope_key="person_intent/self/company_scale",
            value=value,
            source="test",
            pref_scope=pref_scope,
            layer=layer,
            timescale=timescale,
            valid_until=(
                now + timedelta(days=7)
                if timescale == "situational"
                else None
            ),
            semantic_stance=(
                "negative" if "不" in value else "positive"
            ),
            observed_at=now,
        )
    )
    assert version is not None
    return version


def test_preferences_render_in_four_layers_and_amend_through_confirmation(tmp_path) -> None:
    context, history, service, _, _ = _stores(tmp_path)
    preferences = (
        _active_preference(
            context,
            value="一直不去大厂",
            pref_scope="freeform.person_stable",
            layer="stable",
        ),
        _active_preference(
            context,
            value="默认优先小团队",
            pref_scope="freeform.person_default",
            layer="contextual",
        ),
        _active_preference(
            context,
            value="后端岗位可以看大厂",
            pref_scope="freeform.role.backend",
            layer="contextual",
        ),
        _active_preference(
            context,
            value="这次阿里是例外",
            pref_scope="freeform.job.alibaba",
            layer="transient",
            timescale="situational",
        ),
    )
    markdown, _, _ = service.export(user_id="u1")
    for title in (
        "① 人级恒定层",
        "② 人级默认层",
        "③ 岗位类型层",
        "④ 单次情境层",
    ):
        assert title in markdown
    for item in preferences:
        assert item.value in markdown

    edited = markdown.replace(
        "- 默认优先小团队",
        "- 默认优先中小团队",
    )
    prepared = service.prepare(user_id="u1", markdown=edited)
    assert len(prepared.analysis.preference_amendments) == 1
    proposal = prepared.analysis.preference_amendments[0]
    assert proposal.base_update_id == preferences[1].update_id
    applied = service.apply(
        user_id="u1",
        markdown=edited,
        confirmation_digest=prepared.confirmation_digest,
    )
    assert applied.preference_amendments == 1
    active = context.list_free_text_preferences(
        user_id="u1", statuses=("active",)
    )
    assert "默认优先中小团队" in {item.value for item in active}
    assert "默认优先小团队" not in {item.value for item in active}


def test_a_deleted_preference_uses_the_tombstone_confirmation_gate(tmp_path) -> None:
    context, history, service, _, _ = _stores(tmp_path)
    preference = _active_preference(
        context,
        value="默认优先小团队",
        pref_scope="freeform.person_default",
        layer="contextual",
    )
    markdown, _, _ = service.export(user_id="u1")
    _, second_export_id, _ = service.export(user_id="u1")
    edited = _remove_block(markdown, preference.update_id)
    notes = WorkingNotesStore(tmp_path / "working-notes")
    notes.replace(user_id="u1", markdown="- 默认优先小团队")
    prepared = service.prepare(user_id="u1", markdown=edited)
    proposal = prepared.analysis.tombstones[0]
    assert proposal.target_kind == "intent_preference"

    applied = service.apply(
        user_id="u1",
        markdown=edited,
        confirmation_digest=prepared.confirmation_digest,
        working_notes=notes,
    )

    assert applied.tombstones == 1
    assert applied.working_notes_cleared
    assert notes.read(user_id="u1") == ""
    assert context.list_free_text_preferences(
        user_id="u1", statuses=("active",)
    ) == ()
    for export_id in (prepared.analysis.export_id, second_export_id):
        retained = context.get_memory_review_export(
            user_id="u1",
            export_id=export_id,
        )
        assert retained is not None
        assert all(item["update_id"] != preference.update_id for item in retained)
        assert all(item["value"] != preference.value for item in retained)


def test_notes_are_cleared_even_if_derived_cleanup_fails(
    tmp_path,
    monkeypatch,
) -> None:
    context, _, service, _, _ = _stores(tmp_path)
    preference = _active_preference(
        context,
        value="默认优先小团队",
        pref_scope="freeform.person_default",
        layer="contextual",
    )
    markdown, export_id, _ = service.export(user_id="u1")
    edited = _remove_block(markdown, preference.update_id)
    prepared = service.prepare(user_id="u1", markdown=edited)
    notes = WorkingNotesStore(tmp_path / "working-notes")
    notes.replace(user_id="u1", markdown="- 默认优先小团队")

    def fail_cleanup(**_kwargs):
        raise OSError("injected cleanup failure")

    monkeypatch.setattr(context, "purge_derived_memory", fail_cleanup)
    applied = service.apply(
        user_id="u1",
        markdown=edited,
        confirmation_digest=prepared.confirmation_digest,
        working_notes=notes,
    )

    assert applied.cleanup_incomplete == (
        f"derived_memory:{preference.update_id}",
    )
    assert notes.read(user_id="u1") == ""
    assert context.list_free_text_preferences(
        user_id="u1",
        statuses=("active",),
    ) == ()
    retained = context.get_memory_review_export(
        user_id="u1",
        export_id=export_id,
    )
    assert retained is not None
    assert all(item["update_id"] != preference.update_id for item in retained)


def test_cli_batch_reports_working_notes_cleanup_failure_without_raising(
    tmp_path,
) -> None:
    context, _, service, _, items = _stores(tmp_path)
    markdown, _, _ = service.export(user_id="u1")
    edited = _remove_block(markdown, items[0].update_id)
    prepared = service.prepare(user_id="u1", markdown=edited)

    class FailingNotes:
        def clear(self, *, user_id: str) -> bool:
            raise OSError(f"cannot clear notes for {user_id}")

    applied = service.apply(
        user_id="u1",
        markdown=edited,
        confirmation_digest=prepared.confirmation_digest,
        working_notes=FailingNotes(),
    )

    assert applied.tombstones == 1
    assert applied.working_notes_cleared is False
    assert applied.cleanup_incomplete == ("working_notes",)


def test_preference_cleanup_failure_returns_state_and_can_be_retried(
    tmp_path,
    monkeypatch,
) -> None:
    context, history, _, _, _ = _stores(tmp_path)
    preference = _active_preference(
        context,
        value="默认优先小团队",
        pref_scope="freeform.person_default",
        layer="contextual",
    )
    proposal = MemoryTombstoneProposal(
        target_kind="intent_preference",
        scope_key=preference.scope_key,
        pref_scope=preference.pref_scope,
        update_id=preference.update_id,
        expected_content_sha256=preference.content_digest,
        reason="用户删除该偏好。",
    )
    context.upsert_task(
        user_id="u1",
        conversation_id="c1",
        task=ConversationTaskState(pending_memory_tombstone=proposal),
    )
    notes = WorkingNotesStore(tmp_path / "working-notes")
    notes.replace(user_id="u1", markdown="- 默认优先小团队")
    registry = MainAgentToolRegistry(
        career_history_store=history,
        conversation_store=context,
        working_notes_store=notes,
    )
    original_clear = notes.clear

    def fail_clear(**_kwargs):
        raise OSError("unlink failed")

    monkeypatch.setattr(notes, "clear", fail_clear)

    failed = registry.invoke_atomic_tool(
        "confirm_memory_tombstone",
        {
            "user_id": "u1",
            "conversation_id": "c1",
            "proposal": proposal,
        },
    )
    assert failed.state == "memory_tombstone_cleanup_incomplete"
    assert failed.payload["working_notes_cleared"] is False
    assert failed.payload["cleanup_incomplete"] == "working_notes"

    monkeypatch.setattr(notes, "clear", original_clear)
    retried = registry.invoke_atomic_tool(
        "confirm_memory_tombstone",
        {
            "user_id": "u1",
            "conversation_id": "c1",
            "proposal": proposal,
        },
    )
    assert retried.state == "memory_tombstoned"
    assert notes.read(user_id="u1") == ""


def test_deleting_an_amended_preference_scrubs_every_exported_revision(
    tmp_path,
) -> None:
    context, _, service, _, _ = _stores(tmp_path)
    original = _active_preference(
        context,
        value="默认优先小团队",
        pref_scope="freeform.person_default",
        layer="contextual",
    )
    original_markdown, original_export_id, _ = service.export(user_id="u1")
    amended_markdown = original_markdown.replace(
        "- 默认优先小团队",
        "- 默认优先中小团队",
    )
    amendment = service.prepare(user_id="u1", markdown=amended_markdown)
    service.apply(
        user_id="u1",
        markdown=amended_markdown,
        confirmation_digest=amendment.confirmation_digest,
    )
    current = context.list_free_text_preferences(
        user_id="u1",
        statuses=("active",),
    )[0]
    current_markdown, current_export_id, _ = service.export(user_id="u1")
    deletion_markdown = _remove_block(current_markdown, current.update_id)
    deletion = service.prepare(user_id="u1", markdown=deletion_markdown)

    service.apply(
        user_id="u1",
        markdown=deletion_markdown,
        confirmation_digest=deletion.confirmation_digest,
    )

    for export_id in (original_export_id, current_export_id):
        retained = context.get_memory_review_export(
            user_id="u1",
            export_id=export_id,
        )
        assert retained is not None
        assert all(
            item["update_id"] not in {original.update_id, current.update_id}
            for item in retained
        )
        assert all(
            item["value"] not in {"默认优先小团队", "默认优先中小团队"}
            for item in retained
        )


def test_deleting_one_preference_layer_does_not_tombstone_its_siblings(
    tmp_path,
) -> None:
    context, history, service, _, _ = _stores(tmp_path)
    default = _active_preference(
        context,
        value="我不考虑大厂",
        pref_scope="freeform.person_default",
        layer="contextual",
    )
    exception = _active_preference(
        context,
        value="这家例外，我愿意去大厂",
        pref_scope="freeform.conversation.c1",
        layer="transient",
        timescale="situational",
    )
    now = datetime.now(timezone.utc)
    for conversation_id, preference in (
        ("c1", default),
        ("c2", exception),
    ):
        context.commit_turn(
            user_id="u1",
            conversation_id=conversation_id,
            task=ConversationTaskState(),
            user_message=ConversationMessageContext(
                role="user",
                content=preference.value,
                created_at=now,
            ),
            assistant_message=ConversationMessageContext(
                role="assistant",
                content="已记录。",
                created_at=now,
            ),
            memory_scope_keys=(
                intent_entry_id(
                    preference.scope_key,
                    preference.pref_scope,
                ),
            ),
        )
    markdown, _, _ = service.export(user_id="u1")
    edited = _remove_block(markdown, exception.update_id)
    prepared = service.prepare(user_id="u1", markdown=edited)
    proposal = prepared.analysis.tombstones[0]
    assert proposal.update_id == exception.update_id

    applied = service.apply(
        user_id="u1",
        markdown=edited,
        confirmation_digest=prepared.confirmation_digest,
    )

    assert applied.tombstones == 1
    active = context.list_free_text_preferences(
        user_id="u1",
        statuses=("active",),
    )
    assert [item.update_id for item in active] == [default.update_id]
    assert context.list_messages_after(
        user_id="u1",
        conversation_id="c1",
        after_sequence=0,
        limit=10,
    )
    assert context.list_messages_after(
        user_id="u1",
        conversation_id="c2",
        after_sequence=0,
        limit=10,
    ) == ()
