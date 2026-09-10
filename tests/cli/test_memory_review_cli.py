from __future__ import annotations

from io import StringIO
import json
import re

from career_agent.cli import EXIT_ARGUMENT_ERROR, main
from career_agent.storage.career_history import CareerHistoryStore
from career_agent.storage.context import CareerContextStore
from career_agent.storage.working_notes import WorkingNotesStore


def _remove_block(markdown: str, update_id: str) -> str:
    return re.sub(
        rf'\n<!-- memory:start[^\n]+update_id="{update_id}" -->\n.*?'
        rf'\n<!-- memory:end[^\n]+update_id="{update_id}" -->',
        "",
        markdown,
        count=1,
        flags=re.DOTALL,
    )


def test_cli_reviews_full_diff_and_applies_it_with_one_confirmation(tmp_path) -> None:
    context_path = tmp_path / "context.sqlite3"
    history_path = tmp_path / "career.sqlite3"
    notes_path = tmp_path / "notes"
    memory_path = tmp_path / "MEMORY.md"
    CareerContextStore(context_path)
    history = CareerHistoryStore(history_path)
    record = history.create_record(
        user_id="u1",
        record_type="work",
        title="Engineer",
        organization="Example",
    )
    evidence = tuple(
        history.confirm_evidence(
            user_id="u1",
            career_evidence_id=history.create_evidence(
                user_id="u1",
                career_record_id=record.id,
                claim=claim,
                origin="user_input",
            ).id,
        )
        for claim in ("Led retrieval.", "Owned evaluation.")
    )
    notes = WorkingNotesStore(notes_path)
    notes.replace(user_id="u1", markdown="- Owned evaluation.")

    export_output = StringIO()
    assert main(
        [
            "memory",
            "export",
            "--user-id",
            "u1",
            "--context-store",
            str(context_path),
            "--resume-store",
            str(history_path),
            "--output",
            str(memory_path),
        ],
        stdout=export_output,
        stderr=StringIO(),
    ) == 0
    exported = json.loads(export_output.getvalue())
    assert "Led retrieval." not in export_output.getvalue()
    markdown = memory_path.read_text(encoding="utf-8")
    markdown = markdown.replace("- Led retrieval.", "- Led retrieval and ranking.")
    markdown = _remove_block(markdown, evidence[1].update_id)
    memory_path.write_text(markdown, encoding="utf-8")
    evidence_before_review = history.list_evidence(
        user_id="u1",
        include_historical=True,
    )

    review_output = StringIO()
    assert main(
        [
            "memory",
            "review",
            "--user-id",
            "u1",
            "--context-store",
            str(context_path),
            "--resume-store",
            str(history_path),
            "--file",
            str(memory_path),
        ],
        stdout=review_output,
        stderr=StringIO(),
    ) == 0
    assert history.list_evidence(
        user_id="u1",
        include_historical=True,
    ) == evidence_before_review
    review = json.loads(review_output.getvalue())
    assert review["proposal_count"] == 2
    assert review["changes"] == [
        {
            "action": "amend",
            "target_kind": "career_evidence",
            "identity": evidence[0].detail_ref,
            "before": "Led retrieval.",
            "after": "Led retrieval and ranking.",
        },
        {
            "action": "delete",
            "target_kind": "career_evidence",
            "identity": evidence[1].detail_ref,
            "before": "Owned evaluation.",
            "after": None,
        },
    ]

    rejected_output = StringIO()
    assert main(
        [
            "memory",
            "apply",
            "--user-id",
            "u1",
            "--context-store",
            str(context_path),
            "--resume-store",
            str(history_path),
            "--file",
            str(memory_path),
            "--confirmation-digest",
            "sha256:" + "0" * 64,
            "--confirm",
            "--working-notes-dir",
            str(notes_path),
        ],
        stdout=rejected_output,
        stderr=StringIO(),
    ) == EXIT_ARGUMENT_ERROR
    assert {
        item.claim
        for item in history.list_evidence(
            user_id="u1",
            verification_status="confirmed",
        )
    } == {"Led retrieval.", "Owned evaluation."}

    apply_output = StringIO()
    assert main(
        [
            "memory",
            "apply",
            "--user-id",
            "u1",
            "--context-store",
            str(context_path),
            "--resume-store",
            str(history_path),
            "--file",
            str(memory_path),
            "--confirmation-digest",
            review["confirmation_digest"],
            "--confirm",
            "--working-notes-dir",
            str(notes_path),
        ],
        stdout=apply_output,
        stderr=StringIO(),
    ) == 0
    applied = json.loads(apply_output.getvalue())
    assert applied["state"] == "memory_review_applied"
    assert applied["amendments"] == 1
    assert applied["tombstones"] == 1
    assert applied["working_notes_cleared"] is True
    assert notes.read(user_id="u1") == ""
    assert {
        item.claim
        for item in history.list_evidence(
            user_id="u1",
            verification_status="confirmed",
        )
    } == {"Led retrieval and ranking."}
    retained = CareerContextStore(context_path).get_memory_review_export(
        user_id="u1",
        export_id=exported["export_id"],
    )
    assert retained is not None
    assert all(item["update_id"] != evidence[1].update_id for item in retained)
    assert all(item["value"] != "Owned evaluation." for item in retained)


def test_cli_batch_apply_requires_explicit_confirm_flag(tmp_path) -> None:
    context_path = tmp_path / "context.sqlite3"
    history_path = tmp_path / "career.sqlite3"
    memory_path = tmp_path / "MEMORY.md"
    CareerContextStore(context_path)
    CareerHistoryStore(history_path)
    assert main(
        [
            "memory",
            "export",
            "--user-id",
            "u1",
            "--context-store",
            str(context_path),
            "--resume-store",
            str(history_path),
            "--output",
            str(memory_path),
        ],
        stdout=StringIO(),
        stderr=StringIO(),
    ) == 0
    review_output = StringIO()
    assert main(
        [
            "memory",
            "review",
            "--user-id",
            "u1",
            "--context-store",
            str(context_path),
            "--resume-store",
            str(history_path),
            "--file",
            str(memory_path),
        ],
        stdout=review_output,
        stderr=StringIO(),
    ) == 0
    digest = json.loads(review_output.getvalue())["confirmation_digest"]
    apply_output = StringIO()
    assert main(
        [
            "memory",
            "apply",
            "--user-id",
            "u1",
            "--context-store",
            str(context_path),
            "--resume-store",
            str(history_path),
            "--file",
            str(memory_path),
            "--confirmation-digest",
            digest,
        ],
        stdout=apply_output,
        stderr=StringIO(),
    ) == EXIT_ARGUMENT_ERROR
    assert "requires --confirm" in json.loads(apply_output.getvalue())["error_detail"]
