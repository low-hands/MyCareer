"""Resume parsing and structured data extraction.

This module handles reading resume files (PDF/text) and converting them
into structured ResumeData. The parsing logic is decoupled from CLI/API
so it can be reused by any interface.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field


class ResumeLabel(str, Enum):
    """Resume category targeting different job directions."""

    BACKEND = "backend"               # 后端开发
    FRONTEND = "frontend"             # 前端开发
    FULLSTACK = "fullstack"           # 全栈开发
    ML = "ml"                         # 机器学习/深度学习
    LLM = "llm"                       # 大模型/NLP
    CV = "cv"                         # 计算机视觉
    MULTIMODAL = "multimodal"         # 多模态
    AGENT = "agent"                   # Agent/智能体
    RECOMMENDATION = "recommendation" # 推荐系统/搜索
    DATA = "data"                     # 数据工程/数据科学
    DEVOPS = "devops"                 # DevOps/SRE
    QA = "qa"                         # 测试/质量
    SECURITY = "security"             # 安全
    EMBEDDED = "embedded"             # 嵌入式/硬件
    PRODUCT = "product"               # 产品经理
    GENERAL = "general"               # 通用/未分类


class WorkExperience(BaseModel):
    """Work or internship experience."""

    company: str = ""
    title: str = ""
    duration: str = ""
    description: str = ""


class ProjectExperience(BaseModel):
    """A project entry (academic, personal, or open-source)."""

    name: str = ""
    role: str = ""
    duration: str = ""
    description: str = ""


class Publication(BaseModel):
    """A paper, article, or conference publication."""

    title: str = ""
    venue: str = ""
    year: str = ""
    authors: str = ""


class Award(BaseModel):
    """A competition, award, or certification."""

    name: str = ""
    organization: str = ""
    year: str = ""
    rank: str = ""


class Education(BaseModel):
    """A single education entry."""

    school: str = ""
    degree: str = ""
    major: str = ""
    graduation_year: str = ""


class ResumeData(BaseModel):
    """Structured resume data extracted from a file.

    Supports multiple resumes per user via the ``label`` field.
    Each resume targets different roles/industries.
    """

    label: ResumeLabel = ResumeLabel.GENERAL
    name: str = ""
    email: str = ""
    phone: str = ""
    summary: str = ""
    skills: list[str] = Field(default_factory=list)
    tech_stack: list[str] = Field(default_factory=list)
    work_experience: list[WorkExperience] = Field(default_factory=list)
    project_experience: list[ProjectExperience] = Field(default_factory=list)
    publications: list[Publication] = Field(default_factory=list)
    awards: list[Award] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    target_roles: list[str] = Field(default_factory=list)
    target_cities: list[str] = Field(default_factory=list)
    expected_salary: str = ""
    raw_text: str = Field(default="", exclude=True)

    def to_facts(self) -> list[str]:
        """Convert resume data into concise facts for prompt injection."""
        facts: list[str] = []
        if self.name:
            facts.append(f"用户姓名: {self.name}")
        if self.skills:
            facts.append(f"核心技能: {', '.join(self.skills[:10])}")
        if self.tech_stack:
            facts.append(f"技术栈: {', '.join(self.tech_stack[:10])}")
        if self.target_roles:
            facts.append(f"目标岗位: {', '.join(self.target_roles)}")
        if self.target_cities:
            facts.append(f"目标城市: {', '.join(self.target_cities)}")
        if self.expected_salary:
            facts.append(f"期望薪资: {self.expected_salary}")
        if self.education:
            edu = self.education[0]
            facts.append(f"学历: {edu.school} {edu.degree} {edu.major}")
        if self.work_experience:
            entries = [f"{w.company}: {w.title}" for w in self.work_experience[:5]]
            facts.append(f"工作/实习经历: {'; '.join(entries)}")
        if self.project_experience:
            entries = [f"{p.name}: {p.description[:30]}" for p in self.project_experience[:5]]
            facts.append(f"项目经历: {'; '.join(entries)}")
        if self.publications:
            entries = [f"{pub.title}: {pub.venue or '在投'}" for pub in self.publications[:5]]
            facts.append(f"论文/发表: {'; '.join(entries)}")
        if self.awards:
            entries = [f"{a.name}: {a.rank or a.organization}" for a in self.awards[:5]]
            facts.append(f"竞赛/奖项: {'; '.join(entries)}")
        return facts


def extract_text_from_pdf(file_path: Path) -> str:
    """Extract raw text from a PDF file."""
    from pypdf import PdfReader

    reader = PdfReader(str(file_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def extract_text_from_file(file_path: Path) -> str:
    """Extract text from a file, supporting PDF and plain text."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    # Treat everything else as plain text
    return file_path.read_text(encoding="utf-8").strip()


RESUME_STRUCTURE_PROMPT = """\
你是一个简历解析助手。请从以下简历文本中提取结构化信息。

## 提取字段

- label: 简历方向标签，必须是以下之一：
  backend, frontend, fullstack, ml, llm, cv, multimodal, agent,
  recommendation, data, devops, qa, security, embedded, product, general
  根据简历内容判断最匹配的方向，无法确定则用 "general"
- name: 姓名
- email: 邮箱
- phone: 手机号
- summary: 一句话个人简介
- skills: 核心能力列表（软技能、领域知识等，最多 10 个）
- tech_stack: 技术栈列表（编程语言、框架、工具、平台等，最多 15 个）
- work_experience: 工作/实习经历列表（company, title, duration, description）
- project_experience: 项目经历列表（name, role, duration, description）
- publications: 论文/发表列表（title, venue, year, authors）
- awards: 竞赛/奖项列表（name, organization, year, rank）
- education: 教育经历列表（school, degree, major, graduation_year）
- target_roles: 目标岗位/求职方向（从文本中推断，2-3 个；找不到则留空）
- target_cities: 目标工作城市（从文本中推断，1-3 个；找不到则留空）
- expected_salary: 期望薪资（如有提及；否则留空）

## 规则

1. 只提取文本中明确提到的信息，不要编造。
2. 如果某个字段在文本中找不到，留空字符串或空列表。
3. skills 放软技能和领域知识，tech_stack 放具体技术工具。
4. 区分工作经历（公司实习/全职）和项目经历（学术项目、个人项目、开源项目）。
5. 论文包括会议论文、期刊论文、预印本等。
6. 奖项包括竞赛获奖、奖学金、认证等。

## 简历文本

{text}
"""


async def parse_resume_with_llm(
    raw_text: str,
    model: BaseChatModel,
) -> ResumeData:
    """Use LLM to structure raw resume text into ResumeData."""
    from langchain_core.messages import HumanMessage, SystemMessage

    prompt = RESUME_STRUCTURE_PROMPT.format(text=raw_text[:8000])  # Limit to ~8k chars
    response = await model.with_structured_output(ResumeData).ainvoke(
        [SystemMessage(content="你是一个精确的简历解析助手。"), HumanMessage(content=prompt)]
    )
    response.raw_text = raw_text
    return response


async def load_resume(file_path: str | Path, model: BaseChatModel) -> ResumeData:
    """Load a resume file and return structured ResumeData.

    This is the main entry point for resume handling.
    CLI and API both call this function.
    """
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Resume file not found: {path}")

    raw_text = extract_text_from_file(path)
    if not raw_text:
        raise ValueError(f"Resume file is empty: {path}")

    return await parse_resume_with_llm(raw_text, model)
