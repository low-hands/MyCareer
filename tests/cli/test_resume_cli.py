import json
from io import StringIO

from career_agent.cli import EXIT_ARGUMENT_ERROR, main


def run(args):
    output = StringIO()
    code = main(args, stdout=output, stderr=StringIO())
    return code, json.loads(output.getvalue())


def test_target_role_classifies_resume_families_and_versions(tmp_path) -> None:
    store = tmp_path / "resumes.sqlite3"
    code, role_payload = run(["target-role", "create", "--user-id", "u1", "--title", "AI Engineer", "--priority", "1", "--resume-store", str(store)])
    assert code == 0
    role = role_payload["target_role"]

    source = tmp_path / "resume.md"
    content = "# Resume\n\nPrivate career history"
    source.write_text(content, encoding="utf-8")
    code, imported = run(["resume", "import", "--user-id", "u1", "--target-role-id", role["id"], "--name", "Base", "--file", str(source), "--resume-store", str(store)])
    assert code == 0
    assert imported["resume"]["target_role_id"] == role["id"]
    assert content not in json.dumps(imported)
    assert str(source) not in json.dumps(imported)

    source_v2 = tmp_path / "resume-v2.md"
    source_v2.write_text("# Resume v2", encoding="utf-8")
    code, appended = run(["resume", "import", "--user-id", "u1", "--resume-id", imported["resume"]["id"], "--file", str(source_v2), "--resume-store", str(store)])
    assert code == 0
    assert appended["versions"][0]["version_number"] == 2

    source_other = tmp_path / "other.md"
    source_other.write_text("# Other", encoding="utf-8")
    run(["resume", "import", "--user-id", "u1", "--target-role-id", role["id"], "--name", "Tailored", "--file", str(source_other), "--resume-store", str(store)])
    code, listed = run(["resume", "list", "--user-id", "u1", "--resume-store", str(store)])
    assert code == 0
    assert listed["target_roles"][0]["id"] == role["id"]
    assert len(listed["target_roles"][0]["resumes"]) == 2

    code, shown = run(["resume", "show", "--user-id", "u1", "--resume-id", imported["resume"]["id"], "--resume-store", str(store)])
    assert code == 0
    assert [version["version_number"] for version in shown["versions"]] == [2, 1]


def test_resume_import_requires_target_role(tmp_path) -> None:
    source = tmp_path / "resume.md"
    source.write_text("# Resume", encoding="utf-8")
    code, payload = run(["resume", "import", "--user-id", "u1", "--name", "Base", "--file", str(source), "--resume-store", str(tmp_path / "resumes.sqlite3")])

    assert code == EXIT_ARGUMENT_ERROR
    assert "target_role_id" in payload["error_detail"]
