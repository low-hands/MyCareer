"""Adapter from boss-agent-cli MCP envelopes to Career Agent job models."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote
from uuid import uuid4

from career_agent.jobs.models import JobPosting
from career_agent.jobs.providers.base import (
    AsyncTool,
    JobProviderError,
    ProviderSearchResult,
)
from career_agent.jobs.search_models import SearchCriteria

_SALARY_BUCKETS = (
    ("3K以下", 0, 3),
    ("3-5K", 3, 5),
    ("5-10K", 5, 10),
    ("10-15K", 10, 15),
    ("10-20K", 10, 20),
    ("20-50K", 20, 50),
    ("50K以上", 50, float("inf")),
)


class BossMCPJobProvider:
    """Use only the read-only Boss search/detail MCP tools."""

    def __init__(
        self,
        *,
        search_tool: AsyncTool,
        detail_tool: AsyncTool,
        fetch_count_per_city: int = 50,
    ) -> None:
        if fetch_count_per_city < 1:
            raise ValueError("fetch_count_per_city must be positive")
        self._search_tool = search_tool
        self._detail_tool = detail_tool
        self._fetch_count_per_city = fetch_count_per_city

    async def search(
        self,
        *,
        tenant_id: str,
        criteria: SearchCriteria,
        page: int = 1,
    ) -> ProviderSearchResult:
        if not criteria.query:
            raise ValueError("Boss search requires a non-empty query")
        if page != 1:
            raise ValueError("Boss inline export adapter does not expose pages")

        cities: list[str | None] = list(criteria.cities) or [None]
        raw_items: list[dict[str, Any]] = []
        provider_total = 0
        has_more = False
        for city in cities:
            arguments = _search_arguments(
                criteria,
                city=city,
                count=self._fetch_count_per_city,
            )
            envelope = _parse_envelope(await self._search_tool.ainvoke(arguments))
            data = envelope.get("data")
            if not isinstance(data, dict) or not isinstance(data.get("jobs"), list):
                raise _payload_error("boss_export data.jobs must be a list")
            jobs = data["jobs"]
            raw_items.extend(item for item in jobs if isinstance(item, dict))
            count_value = data.get("count")
            if isinstance(count_value, int):
                provider_total += max(0, count_value)
            pagination = envelope.get("pagination")
            if isinstance(pagination, dict):
                total = pagination.get("total")
                if isinstance(total, int) and not isinstance(count_value, int):
                    provider_total += max(0, total)
                has_more = has_more or bool(pagination.get("has_more"))

        postings: list[JobPosting] = []
        warnings: list[str] = []
        seen: set[str] = set()
        now = datetime.now(UTC)
        for item in raw_items:
            if not _matches_local_criteria(item, criteria):
                continue
            external_id = _text(item.get("job_id")) or _text(item.get("security_id"))
            identity = external_id.casefold()
            if not identity or identity in seen:
                if not identity:
                    warnings.append("Skipped Boss result without job/security ID")
                continue
            try:
                posting = _posting_from_search_item(
                    item,
                    tenant_id=tenant_id,
                    external_id=external_id,
                    now=now,
                )
            except ValueError as exc:
                warnings.append(str(exc))
                continue
            seen.add(identity)
            postings.append(posting)

        return ProviderSearchResult(
            postings=tuple(postings),
            candidate_count=len(postings),
            provider_total_count=provider_total or len(raw_items),
            has_more=has_more,
            warnings=tuple(warnings),
        )

    async def get_detail(self, posting: JobPosting) -> JobPosting:
        if posting.source != "boss":
            raise ValueError("Boss provider can only load Boss postings")
        if not posting.detail_locator:
            raise ValueError("Boss posting is missing its security_id locator")

        envelope = _parse_envelope(
            await self._detail_tool.ainvoke(
                {
                    "security_id": posting.detail_locator,
                    "job_id": posting.external_id,
                }
            )
        )
        data = envelope.get("data")
        if not isinstance(data, dict):
            raise _payload_error("boss_detail data must be an object")
        description = _text(data.get("description"))
        if not description:
            raise _payload_error("boss_detail returned no job description")

        company_info = data.get("company_info")
        if not isinstance(company_info, dict):
            company_info = {}
        now = datetime.now(UTC)
        security_id = _text(data.get("security_id")) or posting.detail_locator
        external_id = _text(data.get("job_id")) or posting.external_id
        return posting.model_copy(
            update={
                "external_id": external_id,
                "detail_locator": security_id,
                "source_url": _boss_job_url(security_id),
                "title": _text(data.get("title")) or posting.title,
                "company_name": _text(data.get("company")) or posting.company_name,
                "location": _text(data.get("city")) or posting.location,
                "employment_type": _text(data.get("employment_type")) or posting.employment_type,
                "salary": _text(data.get("salary")) or posting.salary,
                "experience_required": _text(data.get("experience")) or posting.experience_required,
                "education_required": _text(data.get("education")) or posting.education_required,
                "company_industry": _text(company_info.get("industry")) or posting.company_industry,
                "company_size": _text(company_info.get("scale")) or posting.company_size,
                "company_stage": _text(company_info.get("stage")) or posting.company_stage,
                "description": description,
                "detail_level": "full",
                "fetched_at": now,
                "updated_at": now,
            }
        )


def boss_provider_from_tools(
    tools: Iterable[AsyncTool],
    *,
    fetch_count_per_city: int = 50,
) -> BossMCPJobProvider:
    """Select the two read-only Boss MCP tools used by the owned adapter."""

    by_name = {name: tool for tool in tools if (name := getattr(tool, "name", None)) in {"boss_export", "boss_detail"}}
    missing = {"boss_export", "boss_detail"} - set(by_name)
    if missing:
        raise ValueError(f"Missing Boss MCP tools: {sorted(missing)}")
    return BossMCPJobProvider(
        search_tool=by_name["boss_export"],
        detail_tool=by_name["boss_detail"],
        fetch_count_per_city=fetch_count_per_city,
    )


def _search_arguments(
    criteria: SearchCriteria,
    *,
    city: str | None,
    count: int,
) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "query": criteria.query,
        "count": count,
        "format": "json",
        "include_private": True,
    }
    if city:
        arguments["city"] = city
    if salary := _boss_salary_filter(
        criteria.salary_min_k,
        criteria.salary_max_k,
    ):
        arguments["salary"] = salary
    if criteria.experience:
        arguments["experience"] = criteria.experience
    if criteria.education:
        arguments["education"] = criteria.education
    if criteria.company_industries:
        arguments["industry"] = ",".join(criteria.company_industries)
    if criteria.company_sizes:
        arguments["scale"] = ",".join(criteria.company_sizes)
    if criteria.company_stages:
        arguments["stage"] = ",".join(criteria.company_stages)
    if criteria.employment_types:
        arguments["job_type"] = ",".join(criteria.employment_types)
    return arguments


def _boss_salary_filter(
    minimum: int | None,
    maximum: int | None,
) -> str:
    if minimum is None and maximum is None:
        return ""
    for label, bucket_low, bucket_high in _SALARY_BUCKETS:
        exact_minimum = 0 if minimum is None else minimum
        exact_maximum = float("inf") if maximum is None else maximum
        if exact_minimum == bucket_low and exact_maximum == bucket_high:
            return label
    low = minimum if minimum is not None else 0
    high = maximum if maximum is not None else float("inf")
    labels = [label for label, bucket_low, bucket_high in _SALARY_BUCKETS if bucket_high >= low and bucket_low <= high]
    return ",".join(labels)


def _matches_local_criteria(
    item: dict[str, Any],
    criteria: SearchCriteria,
) -> bool:
    filters = (
        (criteria.company_industries, _text(item.get("industry"))),
        (criteria.company_sizes, _text(item.get("scale"))),
        (criteria.company_stages, _text(item.get("stage"))),
    )
    if any(requested and not _matches_any(requested, actual) for requested, actual in filters):
        return False

    requested_types = criteria.employment_types
    if requested_types:
        actual_type = _text(item.get("employment_type"))
        if actual_type:
            return _matches_any(requested_types, actual_type)
        if {_normalize(value) for value in requested_types} == {"实习"}:
            return False
    return True


def _matches_any(expected_values: list[str], actual: str) -> bool:
    normalized_actual = _normalize(actual)
    if not normalized_actual:
        return False
    return any(
        _normalize(expected) in normalized_actual or normalized_actual in _normalize(expected)
        for expected in expected_values
    )


def _posting_from_search_item(
    item: dict[str, Any],
    *,
    tenant_id: str,
    external_id: str,
    now: datetime,
) -> JobPosting:
    title = _text(item.get("title"))
    company = _text(item.get("company"))
    if not title or not company:
        raise ValueError("Skipped Boss result without title/company")
    security_id = _text(item.get("security_id"))
    city = _text(item.get("city"))
    district = _text(item.get("district"))
    return JobPosting(
        job_id=uuid4().hex,
        tenant_id=tenant_id,
        source="boss",
        external_id=external_id,
        detail_locator=security_id,
        source_url=_boss_job_url(security_id),
        title=title,
        company_name=company,
        location=" ".join(part for part in (city, district) if part),
        employment_type=_text(item.get("employment_type")),
        salary=_text(item.get("salary")),
        experience_required=_text(item.get("experience")),
        education_required=_text(item.get("education")),
        company_industry=_text(item.get("industry")),
        company_size=_text(item.get("scale")),
        company_stage=_text(item.get("stage")),
        detail_level="summary",
        fetched_at=now,
        created_at=now,
        updated_at=now,
    )


def _parse_envelope(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise _payload_error("Boss tool returned invalid JSON") from exc
    elif isinstance(raw, list) and raw:
        block = raw[0]
        if isinstance(block, dict):
            return _parse_envelope(block.get("text"))
        return _parse_envelope(getattr(block, "text", None))

    if not isinstance(raw, dict):
        raise _payload_error("Boss tool returned an unsupported payload")
    if raw.get("ok") is not True:
        error = raw.get("error")
        if not isinstance(error, dict):
            error = {}
        raise JobProviderError(
            code=_text(error.get("code")) or "PROVIDER_ERROR",
            message=_text(error.get("message")) or "Boss tool call failed",
            recoverable=bool(error.get("recoverable")),
            recovery_action=_text(error.get("recovery_action")) or None,
        )
    return raw


def _payload_error(message: str) -> JobProviderError:
    return JobProviderError(code="INVALID_PAYLOAD", message=message)


def _boss_job_url(security_id: str) -> str:
    if not security_id:
        return ""
    return f"https://www.zhipin.com/job_detail/{quote(security_id, safe='')}.html"


def _normalize(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""
