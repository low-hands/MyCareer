"""A retried import stores one version, and a reused key cannot mean two things.

The key is the client's, scoped to the user. It is deliberately not the file
hash: the same file imported for two target roles is two intended versions,
and only the caller knows that.
"""

import sqlite3

import pytest

from career_agent.storage.resumes import ResumeImportConflictError, ResumeStore
from career_agent.storage.schema import check_schema_version


def _store(tmp_path) -> tuple[ResumeStore, str]:
    store = ResumeStore(tmp_path / "resumes.sqlite3")
    role = store.create_target_role(user_id="u1", title="AI Engineer", priority=1)
    return store, role.id


def test_the_same_key_replays_the_first_version_without_storing_another(tmp_path):
    store, role_id = _store(tmp_path)
    request = dict(
        user_id="u1",
        target_role_id=role_id,
        name="主简历",
        content=b"# Resume",
        document_format="markdown",
        idempotency_key="upload-1",
    )

    resume, version = store.import_document(**request)
    again, replayed = store.import_document(**request)

    assert (again.id, replayed.id) == (resume.id, version.id)
    assert [v.version_number for v in store.list_versions(user_id="u1", resume_id=resume.id)] == [1]
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM resume_version_documents").fetchone() == (1,)


def test_the_same_key_with_a_different_request_is_refused(tmp_path):
    store, role_id = _store(tmp_path)
    base = dict(user_id="u1", target_role_id=role_id, content=b"# Resume", document_format="markdown", idempotency_key="upload-1")
    store.import_document(name="主简历", **base)

    with pytest.raises(ResumeImportConflictError):
        store.import_document(name="另一个名字", **base)
    with pytest.raises(ResumeImportConflictError):
        store.import_document(name="主简历", **{**base, "content": b"# Changed"})


def test_keys_are_scoped_to_the_user(tmp_path):
    store, role_id = _store(tmp_path)
    other_role = store.create_target_role(user_id="u2", title="PM", priority=1)
    store.import_document(user_id="u1", target_role_id=role_id, name="A", content=b"x", document_format="text", idempotency_key="k")

    resume, version = store.import_document(
        user_id="u2", target_role_id=other_role.id, name="B", content=b"y", document_format="text", idempotency_key="k"
    )

    assert resume.user_id == "u2" and version.version_number == 1


def test_identical_bytes_under_different_keys_are_distinct_imports(tmp_path):
    """Dedup by file hash would silently drop the second role's copy."""
    store, role_id = _store(tmp_path)
    second_role = store.create_target_role(user_id="u1", title="Architect", priority=2)
    content = b"same file"

    first, _ = store.import_document(user_id="u1", target_role_id=role_id, name="A", content=content, document_format="text", idempotency_key="k1")
    second, _ = store.import_document(user_id="u1", target_role_id=second_role.id, name="A", content=content, document_format="text", idempotency_key="k2")
    same, appended = store.import_document(user_id="u1", resume_id=first.id, content=content, document_format="text", idempotency_key="k3")

    assert first.id != second.id
    assert same.id == first.id and appended.version_number == 2


def test_a_missing_key_never_replays(tmp_path):
    store, role_id = _store(tmp_path)
    request = dict(user_id="u1", target_role_id=role_id, name="A", content=b"x", document_format="text")
    first, _ = store.import_document(**request)
    second, _ = store.import_document(**request)
    assert first.id != second.id


def test_a_version_8_file_gains_the_receipt_table_on_open(tmp_path):
    path = tmp_path / "resumes.sqlite3"
    ResumeStore(path)
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE resume_import_receipts")
        connection.execute("UPDATE schema_versions SET version = 8 WHERE component = 'resumes'")
        connection.commit()

    store = ResumeStore(path)

    with sqlite3.connect(path) as connection:
        assert check_schema_version(connection, "resumes", 9) == 9
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE name = 'resume_import_receipts'"
        ).fetchone() is not None
    role = store.create_target_role(user_id="u1", title="AI", priority=1)
    a, va = store.import_document(user_id="u1", target_role_id=role.id, name="A", content=b"x", document_format="text", idempotency_key="k")
    b, vb = store.import_document(user_id="u1", target_role_id=role.id, name="A", content=b"x", document_format="text", idempotency_key="k")
    assert (a.id, va.id) == (b.id, vb.id)
