"""The v1→v2 rewrite of conversation message payloads.

``_upgrade_to_v2`` is one-way, runs on a user's real data, and is now the only
code that knows the singular ``resource_ref`` shape ever existed — the contract
carries no compatibility shim on purpose. That combination is exactly what has
to be covered: if the rewrite is wrong, the reference is gone and the report it
pointed at becomes unreachable from the transcript, silently.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from career_agent.storage.context import CareerContextStore


def _v1_file(path: Path, rows: list[dict]) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE conversation_messages (
            user_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            payload TEXT NOT NULL,
            PRIMARY KEY(user_id, conversation_id, sequence)
        );
        CREATE TABLE schema_versions (
            component TEXT PRIMARY KEY,
            version INTEGER NOT NULL,
            updated_at TEXT NOT NULL
        );
        INSERT INTO schema_versions VALUES ('agent_context', 1, '2026-01-01');
        """
    )
    for index, payload in enumerate(rows, start=1):
        connection.execute(
            "INSERT INTO conversation_messages VALUES ('u1','c1',?,?)",
            (index, json.dumps(payload)),
        )
    connection.commit()
    connection.close()


def _message(content: str, *, reference: dict | None) -> dict:
    return {
        "role": "assistant",
        "content": content,
        "created_at": "2026-09-01T00:00:00+00:00",
        "resource_ref": reference,
    }


def test_a_v1_row_keeps_its_reference_through_the_upgrade(tmp_path) -> None:
    path = tmp_path / "context.sqlite3"
    _v1_file(
        path,
        [
            _message(
                "岗位研究已完成。",
                reference={
                    "kind": "job_research_report",
                    "resource_id": "report-legacy",
                    "status_at_delivery": "current",
                    "anchored_by_other_job": False,
                    "label": "Example Corp",
                },
            ),
            _message("普通回复。", reference=None),
        ],
    )

    store = CareerContextStore(path)
    messages = store.list_messages("u1", "c1", limit=10)

    assert [
        [reference.resource_id for reference in message.resource_refs]
        for message in messages
    ] == [["report-legacy"], []]
    # Delivery-time render metadata survives the rewrite, not just the id.
    reference = messages[0].resource_refs[0]
    assert reference.status_at_delivery == "current"
    assert reference.anchored_by_other_job is False
    assert reference.title == "Example Corp"
    assert reference.description is None


def test_the_upgraded_rows_answer_the_predicates_that_index_them(tmp_path) -> None:
    """The archived-report lookup filters on the new shape, not the old one."""
    path = tmp_path / "context.sqlite3"
    _v1_file(
        path,
        [
            _message(
                "报告一。",
                reference={
                    "kind": "interview_preparation",
                    "resource_id": "prep-1",
                    "status_at_delivery": None,
                    "anchored_by_other_job": None,
                },
            ),
            _message("普通回复。", reference=None),
        ],
    )

    store = CareerContextStore(path)
    store.compact_conversation_summary(
        user_id="u1",
        conversation_id="c1",
        expected_previous_through_sequence=0,
        content=_summary(),
        through_sequence=2,
    )
    archived = store.list_archived_resource_messages(
        user_id="u1", conversation_id="c1", through_sequence=2, limit=10
    )

    assert [
        reference.resource_id
        for message in archived
        for reference in message.resource_refs
    ] == ["prep-1"]


def _summary():
    from career_agent.agent.conversation_memory_contracts import (
        ConversationSummaryContent,
    )

    return ConversationSummaryContent(
        user_goals=("回顾旧对话。",),
        confirmed_decisions=(),
        unresolved_questions=(),
        active_constraints=(),
    )
