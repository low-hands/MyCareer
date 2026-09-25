"""Free practice loads exactly what was pinned at start: resume, JD, company, research."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from career_agent.agent.mock_interview_graph import StoredMockInterviewSourceProvider
from career_agent.agent.mock_interview_presenter import practice_basis_line
from career_agent.domain.job_discovery import JobDetail, Provenance
from career_agent.services.interview_context import InterviewPreparationContextFactory
from career_agent.storage.jobs import SQLiteJobPostingRepository
from career_agent.storage.mock_interviews import SQLiteMockInterviewStore
from career_agent.storage.resumes import ResumeStore

NOW = datetime(2026, 9, 20, tzinfo=timezone.utc)


class Research:
    def __init__(self, created_at: datetime) -> None:
        self.created_at = created_at
        self.cutoffs: list[datetime] = []

    def get_report(self, *, user_id, report_id, outdated_before=None):
        if report_id != "research-1":
            return None
        self.cutoffs.append(outdated_before)
        return SimpleNamespace(
            created_at=self.created_at,
            status="current" if self.created_at >= outdated_before else "outdated",
            summary="字节跳动的业务概览。",
            findings=(
                SimpleNamespace(
                    topic="产品", statement="旗下有抖音。", evidence_type="fact",
                    confidence="high",
                ),
            ),
        )


def _provider(tmp_path, research):
    resumes = ResumeStore(tmp_path / "resumes.sqlite3")
    jobs = SQLiteJobPostingRepository(tmp_path / "jobs.sqlite3")
    provider = StoredMockInterviewSourceProvider(
        context_factory=InterviewPreparationContextFactory(
            interviews=None, applications=None, resumes=resumes, career_history=None,
        ),
        jobs=jobs,
        research=research,
    )
    return provider, jobs, SQLiteMockInterviewStore(tmp_path / "mock.sqlite3")


def test_a_chosen_job_brings_its_pinned_jd_company_and_title(tmp_path) -> None:
    provider, jobs, store = _provider(tmp_path, Research(datetime.now(timezone.utc)))
    job = jobs.save_detail(
        user_id="u1", run_id="run-1", result_ref="ref-1", selection_index=1,
        detail=JobDetail(
            source_name="test", source_job_id="bd-1", title="AI 产品经理",
            company_name="字节跳动", description="负责大模型产品。", captured_at=NOW,
            provenance=Provenance(
                source_name="test", source_job_id="bd-1", captured_at=NOW,
                operation="detail", adapter_version="test-v1",
            ),
        ),
    )
    session = store.create_session(
        user_id="u1", interview_type="behavioral",
        job_posting_id=job.posting.id, jd_snapshot_id=job.snapshot.id,
        company_research_report_id="research-1",
    )

    sources = provider.load(session=session)

    assert sources.document is None
    assert (sources.context.company_name, sources.context.role_title) == ("字节跳动", "AI 产品经理")
    assert sources.context.jd_text == job.snapshot.content
    research = sources.context.company_research
    assert research is not None and not research.outdated
    assert [finding.statement for finding in research.findings] == ["旗下有抖音。"]


def test_company_only_practice_has_the_name_but_no_jd_or_research(tmp_path) -> None:
    provider, _, store = _provider(tmp_path, Research(NOW))
    session = store.create_session(
        user_id="u1", interview_type="behavioral", target_company="字节", target_role="PM",
    )

    context = provider.load(session=session).context

    assert (context.company_name, context.role_title, context.jd_text) == ("字节", "PM", "")
    assert context.company_research is None


def test_old_research_is_marked_outdated(tmp_path) -> None:
    old = datetime.now(timezone.utc) - timedelta(days=30)
    provider, _, store = _provider(tmp_path, Research(old))
    session = store.create_session(
        user_id="u1", interview_type="behavioral", target_company="字节跳动",
        company_research_report_id="research-1",
    )

    assert provider.load(session=session).context.company_research.outdated is True


def test_the_basis_line_names_every_source_and_says_what_is_missing() -> None:
    assert practice_basis_line(resume=None) == "本场不参考简历，只考察通用题和专业基础。"
    assert practice_basis_line(resume=("正式简历", 2)) == "本场基于简历《正式简历》v2 出题。"
    assert practice_basis_line(
        resume=("正式简历", 2), job="字节跳动 · AI 产品经理", research_at=NOW,
    ) == "本场基于简历《正式简历》v2、岗位《字节跳动 · AI 产品经理》的 JD 出题；业务背景参考你 2026-09-20 的公司研究。"
    assert practice_basis_line(resume=None, company="字节") == (
        "本场不参考简历，基于公司「字节」（没有 JD）出题。"
    )


def test_the_basis_line_says_whose_style_and_whether_it_was_inferred() -> None:
    assert practice_basis_line(resume=None, company="字节", style="字节跳动") == (
        "本场不参考简历，基于公司「字节」（没有 JD）出题；面试风格参考字节跳动。"
    )
    assert practice_basis_line(
        resume=("正式简历", 1), job="北京字节跳动科技有限公司 · 产品经理",
        style="字节跳动", style_inferred=True, research_at=NOW,
    ) == (
        "本场基于简历《正式简历》v1、岗位《北京字节跳动科技有限公司 · 产品经理》的 JD 出题；"
        "面试风格参考字节跳动（由公司名推断）；业务背景参考你 2026-09-20 的公司研究。"
    )


def test_every_style_profile_has_its_section_in_the_skill_and_back() -> None:
    import re
    from pathlib import Path

    from career_agent.agent.mock_interview_company_styles import COMPANY_STYLE_PROFILES

    text = Path("skills/mock-interview/references/company.md").read_text(encoding="utf-8")
    section = text.split("## Preparation profiles", 1)[1].split("\n## ", 1)[0]
    headings = re.findall(r"^### (.+)$", section, flags=re.MULTILINE)
    assert sorted(headings) == sorted(profile.heading for profile in COMPANY_STYLE_PROFILES)
