"""Which interview-style profile in the mock-interview skill a company gets.

The profiles themselves live in ``skills/mock-interview/references/company.md``;
this table only maps what users and job boards call an employer to one of its
``###`` headings, so matching does not depend on the model translating "字节"
into "ByteDance". Only exact names match (after folding case and spaces):
prefix matching would send 京东方 to JD.com. A name not listed here is left to
the model, which must then say which profile it used, if any.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompanyStyleProfile:
    heading: str
    """The profile's exact ``###`` heading in company.md."""

    display_name: str
    aliases: tuple[str, ...]


COMPANY_STYLE_PROFILES: tuple[CompanyStyleProfile, ...] = (
    CompanyStyleProfile(
        "Alibaba and related businesses", "阿里巴巴",
        ("阿里", "阿里巴巴", "阿里巴巴集团", "Alibaba", "淘天", "淘天集团", "淘宝", "天猫",
         "阿里云", "蚂蚁", "蚂蚁集团", "菜鸟", "钉钉", "高德", "饿了么"),
    ),
    CompanyStyleProfile(
        "Tencent and related businesses", "腾讯",
        ("腾讯", "腾讯科技", "Tencent", "鹅厂", "微信", "腾讯云", "腾讯游戏"),
    ),
    CompanyStyleProfile(
        "ByteDance and related businesses", "字节跳动",
        ("字节", "字节跳动", "ByteDance", "抖音", "抖音集团", "TikTok", "飞书", "剪映",
         "火山引擎", "今日头条"),
    ),
    CompanyStyleProfile("Baidu", "百度", ("百度", "Baidu", "百度智能云")),
    CompanyStyleProfile("Meituan", "美团", ("美团", "Meituan", "大众点评")),
    CompanyStyleProfile("Huawei", "华为", ("华为", "Huawei", "华为技术")),
    CompanyStyleProfile(
        "JD.com", "京东", ("京东", "JD.com", "京东集团", "京东物流", "京东科技"),
    ),
    CompanyStyleProfile(
        "PDD Holdings and related businesses", "拼多多", ("拼多多", "PDD", "Temu"),
    ),
    CompanyStyleProfile("Google", "Google", ("Google", "谷歌")),
    CompanyStyleProfile("Meta", "Meta", ("Meta", "Facebook", "Instagram", "WhatsApp")),
    CompanyStyleProfile(
        "Amazon and AWS", "Amazon", ("Amazon", "亚马逊", "AWS", "亚马逊云科技"),
    ),
    CompanyStyleProfile("Microsoft", "Microsoft", ("Microsoft", "微软")),
)

_BY_HEADING = {profile.heading: profile for profile in COMPANY_STYLE_PROFILES}


def _fold(name: str) -> str:
    return " ".join(name.split()).casefold()


_BY_ALIAS = {
    _fold(alias): profile
    for profile in COMPANY_STYLE_PROFILES
    for alias in profile.aliases
}


def match_company_style(company_name: str | None) -> CompanyStyleProfile | None:
    """The profile an employer name maps to exactly, or ``None``."""
    if not company_name or not company_name.strip():
        return None
    return _BY_ALIAS.get(_fold(company_name))


def company_style_by_heading(heading: str | None) -> CompanyStyleProfile | None:
    return _BY_HEADING.get(heading) if heading else None
