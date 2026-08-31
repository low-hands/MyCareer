from pathlib import Path


ROOT = Path(__file__).parents[2]
SKILL = ROOT / "skills" / "job-research" / "SKILL.md"


def test_job_research_skill_has_a_valid_entrypoint() -> None:
    content = SKILL.read_text(encoding="utf-8")

    assert content.startswith("---\nname: job-research\n")
    assert "description: Research public company" in content


def test_the_skill_points_at_where_chinese_primary_material_actually_is() -> None:
    """A broad web query underserves Chinese employers badly enough to name the
    places the primary material lives."""
    content = SKILL.read_text(encoding="utf-8")

    for source in ("招股书", "天眼查", "36氪", "晚点LatePost"):
        assert source in content


def test_self_reported_posts_can_never_support_a_fact() -> None:
    """The platforms worth searching are also the ones full of anonymous
    recollections; the skill has to say which side of the evidence line they
    fall on, or they will be cited as facts."""
    content = SKILL.read_text(encoding="utf-8")
    section = content.split("### Self-reported content is not a primary source")[1]

    assert "脉脉" in section and "牛客网" in section
    assert "may not\nsupport a `fact` under any circumstances" in section
    assert "`inference`" in section and "`low` confidence" in section


def test_republished_wire_copy_is_not_independent_corroboration() -> None:
    content = SKILL.read_text(encoding="utf-8")

    assert "three outlets carrying one story are one source, not three" in content


def test_compensation_and_interview_loops_stay_out_of_company_research() -> None:
    """Those posts are easy to find, which is exactly why the boundary has to be
    written down rather than left to judgement."""
    content = SKILL.read_text(encoding="utf-8")

    assert "What belongs to interview preparation instead" in content
    assert "Compensation bands, interview loops" in content
