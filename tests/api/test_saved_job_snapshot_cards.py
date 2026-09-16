"""A ``saved_job`` card is pinned to one JD version and survives what happens
to the posting afterwards: re-capture, deletion, another user's key."""

from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import (
    ConversationMessageContext,
    ConversationResourceReference,
    ConversationTaskState,
)
from career_agent.api.app import create_app
from career_agent.api.reads import WorkspaceReader
from career_agent.domain.job_discovery import JobDetail, Provenance
from career_agent.storage.context import CareerContextStore
from career_agent.storage.jobs import SQLiteJobPostingRepository


def _args(path: Path) -> Namespace:
    return Namespace(
        **{
            name: str(path / f"{name}.sqlite3")
            for name in (
                "context_store",
                "resume_store",
                "application_store",
                "job_store",
                "interview_store",
                "email_store",
                "action_center_store",
                "calendar_store",
                "mock_interview_store",
                "job_research_store",
            )
        }
    )


class _Runtime:
    def close(self) -> None:
        pass


def _job(path: Path, description: str = "第一版岗位描述", *, user_id: str = "u1"):
    now = datetime.now(timezone.utc)
    return SQLiteJobPostingRepository(Path(_args(path).job_store)).save_detail(
        user_id=user_id,
        run_id="discovery-1",
        result_ref="result-1",
        selection_index=1,
        detail=JobDetail(
            source_name="BOSS直聘",
            source_job_id="job-1",
            source_url="https://example.test/jobs/1",
            title="AI Agent 实习生",
            company_name="量霸科技",
            description=description,
            captured_at=now,
            provenance=Provenance(
                source_name="BOSS直聘",
                source_job_id="job-1",
                source_url="https://example.test/jobs/1",
                captured_at=now,
                operation="detail",
                adapter_version="test-v1",
            ),
        ),
    )


def _store_turn(
    path: Path, *refs: ConversationResourceReference, reply: str = "已保存岗位。"
) -> None:
    store = CareerContextStore(Path(_args(path).context_store))
    ContextManager(store).load_for_turn(
        user_id="u1", conversation_id="c1", user_message="保存这个岗位"
    )
    now = datetime.now(timezone.utc)
    store.commit_turn(
        user_id="u1",
        conversation_id="c1",
        task=ConversationTaskState(),
        user_message=ConversationMessageContext(
            role="user", content="保存这个岗位", created_at=now
        ),
        assistant_message=ConversationMessageContext(
            role="assistant", content=reply, created_at=now, resource_refs=refs
        ),
        assistant_bodies=(),
        memory_scope_keys=(),
        turn_id="t1",
    )


def _card(saved) -> ConversationResourceReference:
    return ConversationResourceReference(
        kind="saved_job",
        resource_id=saved.snapshot.id,
        job_posting_id=saved.posting.id,
        title="量霸科技｜AI Agent 实习生",
        description=f"JD 第 {saved.snapshot.version} 版 · BOSS直聘",
    )


def _app(path: Path, api_keys):
    reader = WorkspaceReader(_args(path))
    return (
        create_app(
            runtime_factory=_Runtime,
            workspace_reader_factory=lambda: reader,
            api_key_store_factory=lambda: api_keys,
            action_center_factory=lambda: None,
        ),
        reader,
    )


def test_the_card_survives_a_reload_and_stays_on_the_version_it_read(
    tmp_path, api_keys, auth
):
    first = _job(tmp_path)
    _store_turn(tmp_path, _card(first))
    second = _job(tmp_path, "第二版岗位描述")
    assert second.posting.id == first.posting.id
    assert second.snapshot.id != first.snapshot.id and second.snapshot.version == 2

    app, _ = _app(tmp_path, api_keys)
    with TestClient(app) as client:
        transcript = client.get("/v1/conversations/c1/messages", headers=auth).json()
        [card] = transcript["messages"][-1]["resources"]
        assert card["kind"] == "saved_job"
        assert card["resource_id"] == first.snapshot.id
        assert card["title"] == "量霸科技｜AI Agent 实习生"
        assert card["description"] == "JD 第 1 版 · BOSS直聘"
        assert card["available"] is True
        # The reply names the job; the JD text lives behind the card only.
        assert "岗位描述" not in transcript["messages"][-1]["content"]

        snapshot = client.get(f"/v1/jd-snapshots/{card['resource_id']}", headers=auth)
        assert snapshot.status_code == 200
        body = snapshot.json()
        assert body["jd_text"] == "第一版岗位描述"
        assert body["jd_version"] == 1 and body["latest_jd_version"] == 2
        assert body["job_posting_id"] == first.posting.id
        assert body["company_name"] == "量霸科技" and body["title"] == "AI Agent 实习生"

        current = client.get(f"/v1/jobs/{first.posting.id}", headers=auth).json()
        assert current["jd_text"] == "第二版岗位描述" and current["jd_version"] == 2


def test_a_deleted_job_keeps_its_name_on_the_card_but_not_its_text(
    tmp_path, api_keys, auth
):
    saved = _job(tmp_path)
    _store_turn(tmp_path, _card(saved))
    app, reader = _app(tmp_path, api_keys)
    assert reader.delete_job(user_id="u1", job_posting_id=saved.posting.id) == "deleted"
    with TestClient(app) as client:
        transcript = client.get("/v1/conversations/c1/messages", headers=auth).json()
        [card] = transcript["messages"][-1]["resources"]
        assert card["title"] == "量霸科技｜AI Agent 实习生"
        assert card["description"] == "JD 第 1 版 · BOSS直聘"
        assert card["available"] is False
        gone = client.get(f"/v1/jd-snapshots/{saved.snapshot.id}", headers=auth)
        assert gone.status_code == 404
        assert gone.json()["detail"]["code"] == "JD_SNAPSHOT_NOT_FOUND"


def test_another_users_key_cannot_read_the_snapshot(tmp_path, api_keys, auth, issue_key):
    saved = _job(tmp_path)
    app, _ = _app(tmp_path, api_keys)
    with TestClient(app) as client:
        assert client.get(f"/v1/jd-snapshots/{saved.snapshot.id}", headers=auth).status_code == 200
        assert (
            client.get(
                f"/v1/jd-snapshots/{saved.snapshot.id}", headers=issue_key("u2")
            ).status_code
            == 404
        )
        assert client.get(f"/v1/jd-snapshots/{saved.snapshot.id}").status_code in (401, 403)


def test_a_jd_card_and_a_report_card_share_one_message(tmp_path, api_keys, auth):
    saved = _job(tmp_path)
    _store_turn(
        tmp_path,
        _card(saved),
        ConversationResourceReference(
            kind="job_research_report",
            resource_id="report-1",
            status_at_delivery="current",
            anchored_by_other_job=False,
        ),
        reply="已保存岗位，公司调研见下方。",
    )
    app, _ = _app(tmp_path, api_keys)
    with TestClient(app) as client:
        transcript = client.get("/v1/conversations/c1/messages", headers=auth).json()
        cards = transcript["messages"][-1]["resources"]
    assert [(card["kind"], card["resource_id"]) for card in cards] == [
        ("saved_job", saved.snapshot.id),
        ("job_research_report", "report-1"),
    ]
    assert cards[0]["available"] is True and cards[0]["title"] is not None
    assert cards[1]["status_at_delivery"] == "current"
