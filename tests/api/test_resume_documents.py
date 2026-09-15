"""The resume library end of the chat attachment flow, over the real stores.

Import is idempotent per user and key; the original bytes are served back only
to their owner, under headers a resume cannot tamper with; and a transcript
keeps naming a version the user attached after the resume itself is gone.
"""

from __future__ import annotations

from argparse import Namespace
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import sqlite3

from fastapi.testclient import TestClient
from pypdf import PdfWriter

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.main_agent_contracts import (
    ConversationMessageContext,
    ConversationResourceReference,
    ConversationTaskState,
)
from career_agent.api.app import create_app
from career_agent.api.reads import (
    WorkspaceReader,
    content_disposition,
    resume_document_filename,
)
from career_agent.storage.api_keys import WORKSPACE_READ, WORKSPACE_WRITE
from career_agent.storage.context import CareerContextStore
from career_agent.storage.resumes import ResumeStore


def _pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


PDF = _pdf()


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


def _client(tmp_path, api_keys) -> tuple[TestClient, WorkspaceReader]:
    reader = WorkspaceReader(_args(tmp_path))
    app = create_app(
        runtime_factory=_Runtime,
        workspace_reader_factory=lambda: reader,
        api_key_store_factory=lambda: api_keys,
        action_center_factory=lambda: None,
    )
    return TestClient(app), reader


def _role(client: TestClient, headers) -> str:
    return client.post("/v1/target-roles", headers=headers, json={"title": "后端工程师"}).json()["id"]


def test_import_replays_on_the_same_key_and_refuses_a_changed_request(tmp_path, api_keys, issue_key):
    client, _ = _client(tmp_path, api_keys)
    writer = issue_key("u1", WORKSPACE_WRITE, WORKSPACE_READ)
    with client:
        role_id = _role(client, writer)
        form = {"name": "主简历", "target_role_id": role_id}
        first = client.post(
            "/v1/resumes/import",
            headers={**writer, "Idempotency-Key": "upload-1"},
            data=form,
            files={"file": ("resume.md", b"# Resume", "text/markdown")},
        )
        retried = client.post(
            "/v1/resumes/import",
            headers=writer,
            data={**form, "client_upload_id": "upload-1"},
            files={"file": ("resume.md", b"# Resume", "text/markdown")},
        )
        changed = client.post(
            "/v1/resumes/import",
            headers={**writer, "Idempotency-Key": "upload-1"},
            data={**form, "name": "改名"},
            files={"file": ("resume.md", b"# Resume", "text/markdown")},
        )
        disagree = client.post(
            "/v1/resumes/import",
            headers={**writer, "Idempotency-Key": "a"},
            data={**form, "client_upload_id": "b"},
            files={"file": ("resume.md", b"# Resume", "text/markdown")},
        )
        library = client.get("/v1/resumes", headers=writer).json()

    assert first.status_code == 200 and retried.status_code == 200
    assert retried.json() == first.json()
    assert changed.status_code == 409
    assert changed.json()["detail"]["code"] == "RESUME_IMPORT_CONFLICT"
    assert disagree.status_code == 400
    [resume] = library
    assert [v["version_number"] for v in resume["versions"]] == [1]
    assert resume["versions"][0]["id"] == first.json()["resume_version_id"]


def test_the_document_route_serves_the_owner_the_original_bytes(tmp_path, api_keys, issue_key):
    client, _ = _client(tmp_path, api_keys)
    owner = issue_key("u1", WORKSPACE_WRITE, WORKSPACE_READ)
    with client:
        role_id = _role(client, owner)
        pdf = client.post(
            "/v1/resumes/import",
            headers=owner,
            data={"name": '张三 简历/"final"\r\nX-Injected: yes', "target_role_id": role_id},
            files={"file": ("resume.pdf", PDF, "application/pdf")},
        ).json()
        markdown = client.post(
            "/v1/resumes/import",
            headers=owner,
            data={"name": "Markdown 版", "target_role_id": role_id},
            files={"file": ("resume.md", b"<script>alert(1)</script>", "text/markdown")},
        ).json()
        other_pdf = client.post(
            "/v1/resumes/import",
            headers=owner,
            data={"name": "另一份", "target_role_id": role_id},
            files={"file": ("other.pdf", PDF, "application/pdf")},
        ).json()

        route = f"/v1/resumes/{pdf['resume_id']}/versions/{pdf['resume_version_id']}/document"
        inline = client.get(route, headers=owner)
        download = client.get(route + "?download=true", headers=owner)
        preview = client.get(
            f"/v1/resumes/{markdown['resume_id']}/versions/{markdown['resume_version_id']}/document",
            headers=owner,
        )
        markdown_download = client.get(
            f"/v1/resumes/{markdown['resume_id']}/versions/{markdown['resume_version_id']}/document?download=1",
            headers=owner,
        )
        stranger = client.get(route, headers=issue_key("u2", WORKSPACE_READ))
        mismatched = client.get(
            f"/v1/resumes/{other_pdf['resume_id']}/versions/{pdf['resume_version_id']}/document",
            headers=owner,
        )
        unscoped = client.get(route, headers=issue_key("u1", WORKSPACE_WRITE))
        anonymous = client.get(route)

    assert inline.status_code == 200
    assert inline.content == PDF
    assert inline.headers["content-type"] == "application/pdf"
    assert inline.headers["content-disposition"].startswith('inline; filename="')
    assert inline.headers["x-content-type-options"] == "nosniff"
    assert inline.headers["cache-control"] == "private, no-store"
    for header in (inline.headers["content-disposition"], download.headers["content-disposition"]):
        assert "\r" not in header and "\n" not in header and ":" not in header
        ascii_part = header.split("filename*=")[0]
        assert "/" not in ascii_part and ascii_part.count('"') == 2
    assert download.headers["content-disposition"].startswith("attachment; ")
    assert "filename*=UTF-8''" in download.headers["content-disposition"]

    assert preview.status_code == 200
    assert preview.headers["content-type"].startswith("text/plain")
    assert preview.content == b"<script>alert(1)</script>"
    assert markdown_download.headers["content-type"].startswith("text/markdown")
    assert markdown_download.headers["content-disposition"].startswith("attachment; ")
    assert markdown_download.headers["content-disposition"].endswith(".md")

    assert stranger.status_code == 404
    assert mismatched.status_code == 404
    assert unscoped.status_code == 403
    assert anonymous.status_code == 401


def test_filename_helpers_strip_header_breaking_characters():
    name = resume_document_filename('a\r\nb: "c"/d\\e%f', version_number=3, extension="pdf")
    assert name == "ab cdef-v3.pdf"
    assert resume_document_filename("   ", version_number=1, extension="txt") == "resume-v1.txt"
    header = content_disposition("inline", "简历 final.pdf")
    assert header == "inline; filename=\"final.pdf\"; filename*=UTF-8''%E7%AE%80%E5%8E%86%20final.pdf"
    assert "\n" not in content_disposition("attachment", "x\ny.pdf")


def test_transcript_keeps_the_snapshot_and_reports_a_deleted_resume(tmp_path, api_keys, auth):
    client, reader = _client(tmp_path, api_keys)
    args = _args(tmp_path)
    resumes = ResumeStore(Path(args.resume_store))
    role = resumes.create_target_role(user_id="u1", title="后端", priority=1)
    resume, version = resumes.import_document(
        user_id="u1", target_role_id=role.id, name="主简历", content=b"PRIVATE", document_format="text"
    )
    context = CareerContextStore(Path(args.context_store))
    ContextManager(context).load_for_turn(user_id="u1", conversation_id="c1", user_message="分析")
    now = datetime.now(timezone.utc)
    context.commit_turn(
        user_id="u1",
        conversation_id="c1",
        task=ConversationTaskState(),
        user_message=ConversationMessageContext(
            role="user",
            content="帮我分析这份简历",
            created_at=now,
            resource_refs=(
                ConversationResourceReference(
                    kind="resume_version",
                    resource_id=version.id,
                    title="主简历 v1",
                    description="text · 7 bytes",
                ),
            ),
        ),
        assistant_message=ConversationMessageContext(role="assistant", content="偏后端。", created_at=now),
    )

    with client:
        before = client.get("/v1/conversations/c1/messages", headers=auth).json()
        with sqlite3.connect(resumes.path) as connection:
            connection.execute("DELETE FROM resume_version_documents")
            connection.execute("DELETE FROM resume_versions")
            connection.execute("DELETE FROM resumes")
            connection.commit()
        after = client.get("/v1/conversations/c1/messages", headers=auth).json()

    [live] = before["messages"][0]["resources"]
    assert live["kind"] == "resume_version" and live["resource_id"] == version.id
    assert live["title"] == "主简历 v1" and live["available"] is True
    assert live["resume_id"] == resume.id
    [gone] = after["messages"][0]["resources"]
    assert gone["title"] == "主简历 v1" and gone["available"] is False
    assert gone["resume_id"] is None
    assert "PRIVATE" not in str(after)


def test_deleting_the_conversation_leaves_the_resume_in_the_library(tmp_path, api_keys, issue_key):
    auth = issue_key("u1", WORKSPACE_READ, WORKSPACE_WRITE)
    client, _ = _client(tmp_path, api_keys)
    args = _args(tmp_path)
    resumes = ResumeStore(Path(args.resume_store))
    role = resumes.create_target_role(user_id="u1", title="后端", priority=1)
    _, version = resumes.import_document(
        user_id="u1", target_role_id=role.id, name="主简历", content=b"x", document_format="text"
    )
    context = CareerContextStore(Path(args.context_store))
    ContextManager(context).load_for_turn(user_id="u1", conversation_id="c1", user_message="分析")
    now = datetime.now(timezone.utc)
    context.commit_turn(
        user_id="u1",
        conversation_id="c1",
        task=ConversationTaskState(),
        user_message=ConversationMessageContext(
            role="user",
            content="分析",
            created_at=now,
            resource_refs=(
                ConversationResourceReference(kind="resume_version", resource_id=version.id, title="主简历 v1"),
            ),
        ),
        assistant_message=ConversationMessageContext(role="assistant", content="好。", created_at=now),
    )

    with client:
        deleted = client.delete("/v1/conversations/c1", headers=auth)
        library = client.get("/v1/resumes", headers=auth).json()

    assert deleted.status_code == 200
    assert context.get_session("u1", "c1") is None
    assert resumes.get_version(user_id="u1", resume_version_id=version.id) is not None
    assert [item["versions"][0]["id"] for item in library] == [version.id]
