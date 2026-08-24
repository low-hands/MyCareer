from datetime import datetime, timezone
from io import StringIO
import json

from career_agent.cli import EXIT_ARGUMENT_ERROR, main
from career_agent.domain.job_discovery import JobDetail, Provenance
from career_agent.storage.jobs import JDAnalysisPayload, SQLiteJobPostingRepository


NOW = datetime(2026, 8, 23, 8, 0, tzinfo=timezone.utc)


def seed(path) -> str:
    repository = SQLiteJobPostingRepository(path)
    stored = repository.save_detail(
        user_id="u1",
        run_id="run-1",
        result_ref="ref-1",
        selection_index=1,
        detail=JobDetail(
            source_name="boss",
            source_job_id="boss-1",
            title="AI Engineer",
            company_name="Acme",
            description="Build reliable RAG and agent systems.",
            city="Shanghai",
            captured_at=NOW,
            provenance=Provenance(source_name="boss", source_job_id="boss-1", captured_at=NOW, operation="detail", adapter_version="test-v1"),
        ),
    )
    repository.save_analysis(
        user_id="u1",
        jd_snapshot_id=stored.snapshot.id,
        analyzer_version="jd-analysis-v1",
        analysis=JDAnalysisPayload(job_summary="构建可靠的 RAG 与 Agent 系统。", required_skills=("Python",)),
    )
    return stored.posting.id


def run_cli(arguments):
    output = StringIO()
    code = main(arguments, stdout=output, stderr=StringIO())
    return code, json.loads(output.getvalue())


def test_job_list_and_find_return_safe_metadata(tmp_path) -> None:
    path = tmp_path / "jobs.sqlite3"
    job_id = seed(path)

    list_code, listed = run_cli(["job", "list", "--user-id", "u1", "--job-store", str(path)])
    find_code, found = run_cli(["job", "find", "--user-id", "u1", "--query", "RAG", "--job-store", str(path)])

    assert list_code == find_code == 0
    assert listed["jobs"][0]["job_posting_id"] == job_id
    assert found["jobs"][0]["title"] == "AI Engineer"
    assert "Build reliable" not in json.dumps(listed)


def test_job_show_supports_run_selection_and_enforces_user_scope(tmp_path) -> None:
    path = tmp_path / "jobs.sqlite3"
    seed(path)

    code, shown = run_cli(["job", "show", "--user-id", "u1", "--run-id", "run-1", "--selection-index", "1", "--job-store", str(path)])
    rejected_code, rejected = run_cli(["job", "show", "--user-id", "other", "--run-id", "run-1", "--selection-index", "1", "--job-store", str(path)])

    assert code == 0
    assert shown["jd_snapshot"]["content"] == "Build reliable RAG and agent systems."
    assert shown["analysis"]["required_skills"] == ["Python"]
    assert rejected_code == EXIT_ARGUMENT_ERROR
    assert rejected["error_code"] == "JOB_STORE_INPUT_ERROR"
