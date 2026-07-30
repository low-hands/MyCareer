"""Domain models for parsed resumes and tenant-aware role pools."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class WorkExperience(BaseModel):
    """Work or internship experience."""

    company: str = ""
    title: str = ""
    duration: str = ""
    highlights: list[str] = Field(default_factory=list)


class ProjectExperience(BaseModel):
    """A project entry (academic, personal, or open-source)."""

    name: str = ""
    role: str = ""
    duration: str = ""
    highlights: list[str] = Field(default_factory=list)


class Publication(BaseModel):
    """A paper, article, or conference publication."""

    title: str = ""
    venue: str = ""
    year: str = ""


class Award(BaseModel):
    """A competition, award, or certification."""

    name: str = ""
    organization: str = ""


class Education(BaseModel):
    """A single education entry."""

    school: str = ""
    degree: str = ""
    major: str = ""
    duration: str = ""
    gpa: str = ""


class ResumeData(BaseModel):
    """Structured facts extracted from a resume source document."""

    name: str = ""
    email: str = ""
    phone: str = ""
    wechat: str = ""
    summary: str = ""
    skills: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    work_experience: list[WorkExperience] = Field(default_factory=list)
    project_experience: list[ProjectExperience] = Field(default_factory=list)
    publications: list[Publication] = Field(default_factory=list)
    awards: list[Award] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    stated_target_roles: list[str] = Field(default_factory=list)
    stated_target_cities: list[str] = Field(default_factory=list)
    stated_expected_salary: str = ""
    raw_text: str = Field(default="", exclude=True)

    def to_facts(self) -> list[str]:
        """Convert resume data into facts for agent context injection."""

        facts: list[str] = []
        if self.name:
            facts.append(f"用户姓名: {self.name}")
        if self.skills:
            facts.append(f"核心技能: {', '.join(self.skills)}")
        if self.technologies:
            facts.append(f"技术与工具: {', '.join(self.technologies)}")
        if self.stated_target_roles:
            facts.append(f"简历明确目标岗位: {', '.join(self.stated_target_roles)}")
        if self.stated_target_cities:
            facts.append(f"简历明确目标城市: {', '.join(self.stated_target_cities)}")
        if self.stated_expected_salary:
            facts.append(f"简历明确期望薪资: {self.stated_expected_salary}")
        if self.education:
            entries = [
                " ".join(
                    part
                    for part in (
                        education.school,
                        education.degree,
                        education.major,
                        education.duration,
                        f"GPA {education.gpa}" if education.gpa else "",
                    )
                    if part
                )
                for education in self.education
            ]
            facts.append(f"教育经历: {'; '.join(entries)}")
        if self.work_experience:
            entries = [
                _format_experience(
                    experience.company,
                    experience.title,
                    experience.duration,
                    experience.highlights,
                )
                for experience in self.work_experience
            ]
            facts.append(f"工作/实习经历: {'; '.join(entries)}")
        if self.project_experience:
            entries = [
                _format_experience(
                    project.name,
                    project.role,
                    project.duration,
                    project.highlights,
                )
                for project in self.project_experience
            ]
            facts.append(f"项目经历: {'; '.join(entries)}")
        if self.publications:
            entries = [
                " | ".join(part for part in (publication.title, publication.venue, publication.year) if part)
                for publication in self.publications
            ]
            facts.append(f"论文/发表: {'; '.join(entries)}")
        if self.awards:
            entries = [" | ".join(part for part in (award.name, award.organization) if part) for award in self.awards]
            facts.append(f"竞赛/奖项: {'; '.join(entries)}")
        return facts


def _format_experience(
    organization: str,
    role: str,
    duration: str,
    highlights: list[str],
) -> str:
    """Render one structured experience without dropping extracted details."""

    heading = " | ".join(part for part in (organization, role, duration) if part)
    detail = "；".join(highlight for highlight in highlights if highlight)
    if heading and detail:
        return f"{heading}: {detail}"
    return heading or detail


class ResumeVersion(BaseModel):
    """One immutable resume file stored inside a role-oriented pool."""

    version_id: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    role_type: str = Field(min_length=1)
    version_number: int = Field(ge=1)
    note: str = ""
    original_filename: str
    stored_file_path: str
    file_hash: str = Field(min_length=1)
    parsed_data: ResumeData
    created_at: datetime

    @property
    def display_name(self) -> str:
        """Return a stable user-facing name without using it as identity."""

        suffix = f" · {self.note}" if self.note else ""
        return f"{self.role_type} · v{self.version_number}{suffix}"
