"""Tests for the resume parsing module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from career_agent.prompts import RESUME_STRUCTURE_PROMPT
from career_agent.resume import (
    Award,
    Education,
    ProjectExperience,
    Publication,
    ResumeData,
    WorkExperience,
    extract_text_from_file,
    parse_resume_with_llm,
)

# ---------------------------------------------------------------------------
# ResumeData.to_facts()
# ---------------------------------------------------------------------------


class TestResumeDataToFacts:
    def test_empty_resume_produces_no_facts(self) -> None:
        data = ResumeData()
        assert data.to_facts() == []

    def test_name_appears_in_facts(self) -> None:
        data = ResumeData(name="张三")
        facts = data.to_facts()
        assert any("张三" in f for f in facts)

    def test_technologies_and_skills_separated(self) -> None:
        data = ResumeData(
            skills=["沟通能力", "团队协作"],
            technologies=["Python", "PyTorch"],
        )
        facts = data.to_facts()
        skills_fact = next(f for f in facts if "核心技能" in f)
        tech_fact = next(f for f in facts if "技术与工具" in f)
        assert "沟通能力" in skills_fact
        assert "Python" in tech_fact
        assert "PyTorch" not in skills_fact

    def test_work_experience_format(self) -> None:
        data = ResumeData(
            work_experience=[
                WorkExperience(
                    company="字节跳动",
                    title="算法实习生",
                    duration="3个月",
                    highlights=["推荐系统优化", "离线指标提升10%"],
                ),
            ]
        )
        facts = data.to_facts()
        work_fact = next(f for f in facts if "工作/实习" in f)
        assert "字节跳动 | 算法实习生 | 3个月" in work_fact
        assert "推荐系统优化" in work_fact
        assert "离线指标提升10%" in work_fact

    def test_project_experience_keeps_all_highlights(self) -> None:
        data = ResumeData(
            project_experience=[
                ProjectExperience(
                    name="推荐系统",
                    role="负责人",
                    duration="6个月",
                    highlights=["构建召回模块", "实现重排策略"],
                ),
            ]
        )
        facts = data.to_facts()
        proj_fact = next(f for f in facts if "项目经历" in f)
        assert "构建召回模块" in proj_fact
        assert "实现重排策略" in proj_fact

    def test_publication_does_not_infer_missing_venue(self) -> None:
        data = ResumeData(
            publications=[Publication(title="My Paper", venue="", year="2024")]
        )
        facts = data.to_facts()
        pub_fact = next(f for f in facts if "论文" in f)
        assert "My Paper | 2024" in pub_fact
        assert "在投" not in pub_fact

    def test_award_uses_minimal_fields(self) -> None:
        data = ResumeData(
            awards=[Award(name="一等奖学金", organization="北京大学")]
        )
        award_fact = next(f for f in data.to_facts() if "奖项" in f)
        assert "一等奖学金 | 北京大学" in award_fact

    def test_all_education_entries_appear_in_facts(self) -> None:
        data = ResumeData(
            education=[
                Education(
                    school="北京大学",
                    degree="硕士",
                    major="计算机",
                    duration="2022-2024",
                    gpa="3.9/4.0",
                ),
                Education(
                    school="北京航空航天大学",
                    degree="本科",
                    major="软件工程",
                    duration="2018-2022",
                ),
            ]
        )
        facts = data.to_facts()
        edu_fact = next(f for f in facts if "教育经历" in f)
        assert "北京大学" in edu_fact
        assert "硕士" in edu_fact
        assert "GPA 3.9/4.0" in edu_fact
        assert "北京航空航天大学" in edu_fact

    def test_all_skills_and_technologies_appear_in_facts(self) -> None:
        technologies = [f"技术{i}" for i in range(15)]
        data = ResumeData(
            skills=[f"技能{i}" for i in range(12)],
            technologies=technologies,
        )
        facts = data.to_facts()
        skills_fact = next(f for f in facts if "核心技能" in f)
        tech_fact = next(f for f in facts if "技术与工具" in f)
        assert "技能11" in skills_fact
        assert "技术14" in tech_fact

    def test_stated_targets_default_to_empty(self) -> None:
        data = ResumeData()
        assert data.stated_target_roles == []
        assert data.stated_target_cities == []
        assert data.stated_expected_salary == ""

    def test_wechat_is_preserved(self) -> None:
        data = ResumeData(wechat="example_wechat")
        assert data.wechat == "example_wechat"


# ---------------------------------------------------------------------------
# extract_text_from_file
# ---------------------------------------------------------------------------


class TestExtractTextFromFile:
    def test_reads_plain_text_file(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "resume.txt"
        txt_file.write_text("张三\n北京大学", encoding="utf-8")
        result = extract_text_from_file(txt_file)
        assert "张三" in result
        assert "北京大学" in result

    def test_strips_whitespace(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "resume.txt"
        txt_file.write_text("  \n  内容  \n  ", encoding="utf-8")
        result = extract_text_from_file(txt_file)
        assert result == "内容"


# ---------------------------------------------------------------------------
# RESUME_STRUCTURE_PROMPT template
# ---------------------------------------------------------------------------


class TestResumeStructurePrompt:
    def test_resume_text_injected_into_prompt(self) -> None:
        prompt = RESUME_STRUCTURE_PROMPT.format(text="测试文本")
        assert "测试文本" in prompt

    def test_prompt_forbids_inference(self) -> None:
        prompt = RESUME_STRUCTURE_PROMPT.format(text="")
        assert "不得根据经历、技能或常识进行推断" in prompt
        assert "只有原文明示求职意向时才能填写" in prompt

    def test_prompt_restricts_skills_and_technologies(self) -> None:
        prompt = RESUME_STRUCTURE_PROMPT.format(text="")
        assert "skills 只能来自明确的技能栏目" in prompt
        assert "不要把冒号后的整句话作为一项" in prompt
        assert "不得包含 API 或在线服务、数据集、评测基准" in prompt
        assert "SFT、DAPO、LoRA、ReAct" in prompt

    def test_prompt_separates_gpa_and_awards(self) -> None:
        prompt = RESUME_STRUCTURE_PROMPT.format(text="")
        assert "education.gpa 只保存 GPA 数值或等级本身" in prompt
        assert "奖学金或荣誉单独填入 awards" in prompt

    def test_prompt_does_not_request_resume_label(self) -> None:
        prompt = RESUME_STRUCTURE_PROMPT.format(text="")
        assert "简历方向标签" not in prompt


# ---------------------------------------------------------------------------
# parse_resume_with_llm (mock)
# ---------------------------------------------------------------------------


class TestParseResumeWithLLM:
    @pytest.mark.asyncio
    async def test_returns_resume_data(self) -> None:
        expected = ResumeData(
            name="张三",
            education=[
                Education(
                    school="北京大学",
                    degree="硕士",
                    major="CS",
                    duration="2022-2024",
                )
            ],
        )
        mock_model = MagicMock()
        mock_model.with_structured_output.return_value = mock_model
        mock_model.ainvoke = AsyncMock(return_value=expected)

        result = await parse_resume_with_llm("张三 北京大学 硕士", mock_model)

        assert result.name == "张三"
        assert result.raw_text == "张三 北京大学 硕士"
        mock_model.with_structured_output.assert_called_once_with(ResumeData)

    @pytest.mark.asyncio
    async def test_truncates_long_text(self) -> None:
        long_text = "x" * 20000
        mock_model = MagicMock()
        mock_model.with_structured_output.return_value = mock_model
        mock_model.ainvoke = AsyncMock(return_value=ResumeData())

        await parse_resume_with_llm(long_text, mock_model)

        call_args = mock_model.ainvoke.call_args
        messages = call_args[0][0]
        actual_text = messages[0].content
        assert len(actual_text) < len(long_text)
