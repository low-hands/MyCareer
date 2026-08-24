import hashlib
import sqlite3

import pytest

from career_agent.storage.resumes import ResumeStore


def create_role(store: ResumeStore, user_id: str = "u1", title: str = "AI Engineer"):
    return store.create_target_role(user_id=user_id, title=title, priority=1)


def test_role_classifies_multiple_resume_families_with_immutable_versions(tmp_path) -> None:
    store = ResumeStore(tmp_path / "resumes.sqlite3")
    role = create_role(store)
    first_content = b"First resume"
    resume, v1 = store.import_document(user_id="u1", target_role_id=role.id, name="Base", content=first_content, document_format="text")
    updated, v2 = store.import_document(user_id="u1", resume_id=resume.id, content=b"Second resume", document_format="markdown")
    other_resume, other_v1 = store.import_document(user_id="u1", target_role_id=role.id, name="Tailored", content=b"Tailored resume", document_format="text")

    assert v1.version_number == 1
    assert v2.version_number == 2
    assert updated.latest_version_id == v2.id
    assert other_v1.version_number == 1
    assert {item.id for item in store.list_resumes(user_id="u1", target_role_id=role.id)} == {resume.id, other_resume.id}
    assert [version.version_number for version in store.list_versions(user_id="u1", resume_id=resume.id)] == [2, 1]
    assert v1.content_sha256 == hashlib.sha256(first_content).hexdigest()
    assert store.get_resume(user_id="u2", resume_id=resume.id) is None
    assert store.list_versions(user_id="u2", resume_id=resume.id) == ()


def test_new_resume_requires_same_user_target_role_and_append_cannot_change_it(tmp_path) -> None:
    store = ResumeStore(tmp_path / "resumes.sqlite3")
    role = create_role(store)

    try:
        store.import_document(user_id="u1", name="Missing role", content=b"resume", document_format="text")
        assert False, "Expected target-role validation"
    except ValueError as error:
        assert "target_role_id" in str(error)
    try:
        store.import_document(user_id="u2", target_role_id=role.id, name="Foreign role", content=b"resume", document_format="text")
        assert False, "Expected owner validation"
    except ValueError as error:
        assert "Target role not found" in str(error)
    resume, _ = store.import_document(user_id="u1", target_role_id=role.id, name="Base", content=b"resume", document_format="text")
    try:
        store.import_document(user_id="u1", resume_id=resume.id, target_role_id=role.id, content=b"v2", document_format="text")
        assert False, "Expected immutable role validation"
    except ValueError as error:
        assert "cannot change" in str(error)


def test_store_does_not_expose_document_content_in_metadata(tmp_path) -> None:
    path = tmp_path / "resumes.sqlite3"
    store = ResumeStore(path)
    role = create_role(store)
    resume, version = store.import_document(user_id="u1", target_role_id=role.id, name="Safe", content=b"private resume body", document_format="text")

    assert "content" not in version.model_dump()
    raw = sqlite3.connect(path).execute("SELECT content FROM resume_version_documents WHERE resume_version_id = ?", (version.id,)).fetchone()[0]
    assert raw == b"private resume body"


@pytest.mark.parametrize(
    ("document_format", "raw_bytes"),
    [
        ("pdf", b"%PDF-1.7\nresume artifact"),
        ("text", "简历文本".encode("utf-8")),
        ("markdown", "# 简历\n\n工作经历".encode("utf-8")),
    ],
)
def test_read_version_document_returns_exact_original_bytes(
    tmp_path, document_format: str, raw_bytes: bytes
) -> None:
    store = ResumeStore(tmp_path / "resumes.sqlite3")
    role = create_role(store)
    _, version = store.import_document(
        user_id="u1",
        target_role_id=role.id,
        name=f"{document_format} resume",
        content=raw_bytes,
        document_format=document_format,
    )

    document = store.read_version_document(
        user_id="u1", resume_version_id=version.id
    )

    assert document is not None
    assert document.resume_version_id == version.id
    assert document.document_format == document_format
    assert document.raw_bytes == raw_bytes
    assert hashlib.sha256(document.raw_bytes).hexdigest() == version.content_sha256


def test_read_version_document_is_user_scoped(tmp_path) -> None:
    store = ResumeStore(tmp_path / "resumes.sqlite3")
    role = create_role(store, user_id="u1")
    _, version = store.import_document(
        user_id="u1",
        target_role_id=role.id,
        name="Private",
        content=b"PRIVATE RESUME CONTENT",
        document_format="text",
    )

    assert store.read_version_document(
        user_id="u2", resume_version_id=version.id
    ) is None
    assert store.read_version_document(
        user_id="u1", resume_version_id="missing-version"
    ) is None


def test_migrates_legacy_v1_resumes_to_unassigned_target_role(tmp_path) -> None:
    path = tmp_path / "resumes.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE resumes (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, name TEXT NOT NULL, status TEXT NOT NULL, latest_version_id TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")
    connection.execute("CREATE TABLE resume_versions (id TEXT PRIMARY KEY, resume_id TEXT NOT NULL, version_number INTEGER NOT NULL, source_type TEXT NOT NULL, document_format TEXT NOT NULL, content_sha256 TEXT NOT NULL, byte_size INTEGER NOT NULL, created_at TEXT NOT NULL, UNIQUE(resume_id, version_number))")
    connection.execute("CREATE TABLE resume_version_documents (resume_version_id TEXT PRIMARY KEY, content BLOB NOT NULL)")
    connection.execute("INSERT INTO resumes VALUES (?, ?, ?, ?, ?, ?, ?)", ("resume_legacy", "u1", "Legacy", "active", "version_legacy", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"))
    connection.execute("INSERT INTO resume_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?)", ("version_legacy", "resume_legacy", 1, "user_import", "text", "a" * 64, 7, "2026-01-01T00:00:00+00:00"))
    connection.execute("INSERT INTO resume_version_documents VALUES (?, ?)", ("version_legacy", b"legacy!"))
    connection.commit()
    connection.close()

    store = ResumeStore(path)
    roles = store.list_target_roles(user_id="u1")
    resumes = store.list_resumes(user_id="u1")

    assert len(roles) == 1
    assert roles[0].title == "Unassigned"
    assert resumes[0].target_role_id == roles[0].id
    assert store.list_versions(user_id="u1", resume_id="resume_legacy")[0].content_sha256 == "a" * 64
    assert sqlite3.connect(path).execute("SELECT content FROM resume_version_documents WHERE resume_version_id = 'version_legacy'").fetchone()[0] == b"legacy!"
