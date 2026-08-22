from pathlib import Path


ROOT = Path(__file__).parents[2]
SKILL = ROOT / "skills" / "job-discovery" / "SKILL.md"
REFERENCES = ROOT / "skills" / "job-discovery" / "references"


def test_job_discovery_skill_has_valid_entrypoint_and_references() -> None:
    content = SKILL.read_text(encoding="utf-8")

    assert content.startswith("---\nname: job-discovery\n")
    assert "description: Use when the user explicitly asks to search for jobs" in content
    assert "The only model-visible tool is `job_discovery`" in content
    assert "job_discovery.research(request)" not in content
    assert "job_discovery.select(run_id, result_ref, user_id)" not in content
    assert "job_discovery.analyze_provided_jd(run_id, result_ref, jd_text, user_id)" not in content
    assert "selection_required" in content
    assert "detail_unavailable" in content
    assert "Never invent `result_ref`" in content
    assert (REFERENCES / "tool-contracts.md").is_file()
    assert (REFERENCES / "failure-and-recovery.md").is_file()
