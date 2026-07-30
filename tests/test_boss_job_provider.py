"""Contract tests for adapting boss-agent-cli MCP envelopes."""

from __future__ import annotations

import json
from typing import Any

import pytest

from career_agent.jobs.providers.base import JobProviderError
from career_agent.jobs.providers.boss_mcp import BossMCPJobProvider
from career_agent.jobs.search_models import SearchCriteria


class FakeAsyncTool:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, arguments: dict[str, Any]) -> Any:
        self.calls.append(arguments)
        return self.responses.pop(0)


def _success(
    data: Any,
    *,
    pagination: dict[str, Any] | None = None,
) -> str:
    return json.dumps(
        {
            "ok": True,
            "data": data,
            "pagination": pagination,
            "error": None,
            "hints": None,
        },
        ensure_ascii=False,
    )


def _search_item(
    *,
    job_id: str = "enc-job-1",
    security_id: str = "security-1",
    city: str = "深圳",
    scale: str = "100-499人",
    employment_type: str = "",
) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "security_id": security_id,
        "title": "Agent 开发工程师",
        "company": "示例科技",
        "salary": "25-45K",
        "city": city,
        "district": "南山区",
        "experience": "3-5年",
        "education": "本科",
        "industry": "人工智能",
        "scale": scale,
        "stage": "B轮",
        "employment_type": employment_type,
    }


@pytest.mark.asyncio
async def test_search_maps_multiple_city_envelopes_to_job_postings() -> None:
    search_tool = FakeAsyncTool(
        [
            _success({"count": 1, "jobs": [_search_item()]}),
            _success(
                {
                    "count": 1,
                    "jobs": [
                        _search_item(
                            job_id="enc-job-2",
                            security_id="security-2",
                            city="北京",
                        )
                    ],
                }
            ),
        ]
    )
    provider = BossMCPJobProvider(
        search_tool=search_tool,
        detail_tool=FakeAsyncTool([]),
    )

    result = await provider.search(
        tenant_id="tenant-a",
        criteria=SearchCriteria(
            query="Agent 开发",
            cities=["深圳", "北京"],
            salary_min_k=20,
            salary_max_k=50,
            company_sizes=["100-499人"],
        ),
    )

    assert [posting.external_id for posting in result.postings] == [
        "enc-job-1",
        "enc-job-2",
    ]
    assert result.postings[0].detail_locator == "security-1"
    assert result.postings[0].location == "深圳 南山区"
    assert result.candidate_count == 2
    assert result.provider_total_count == 2
    assert result.has_more is False
    assert [call["city"] for call in search_tool.calls] == ["深圳", "北京"]
    assert all(call["salary"] == "20-50K" for call in search_tool.calls)
    assert all(call["scale"] == "100-499人" for call in search_tool.calls)
    assert all(call["include_private"] is True for call in search_tool.calls)


@pytest.mark.asyncio
async def test_search_locally_filters_criteria_missing_from_mcp_schema() -> None:
    search_tool = FakeAsyncTool(
        [
            _success(
                {
                    "count": 2,
                    "jobs": [
                        _search_item(job_id="keep", security_id="keep-security"),
                        _search_item(
                            job_id="drop",
                            security_id="drop-security",
                            scale="10000人以上",
                        ),
                    ],
                }
            )
        ]
    )
    provider = BossMCPJobProvider(
        search_tool=search_tool,
        detail_tool=FakeAsyncTool([]),
    )

    result = await provider.search(
        tenant_id="tenant-a",
        criteria=SearchCriteria(
            query="Agent",
            company_sizes=["100-499人"],
        ),
    )

    assert [posting.external_id for posting in result.postings] == ["keep"]
    assert search_tool.calls[0]["scale"] == "100-499人"


@pytest.mark.asyncio
async def test_search_rejects_error_envelope() -> None:
    search_tool = FakeAsyncTool(
        [
            json.dumps(
                {
                    "ok": False,
                    "data": None,
                    "error": {
                        "code": "AUTH_REQUIRED",
                        "message": "请先登录",
                        "recoverable": True,
                        "recovery_action": "boss login",
                    },
                },
                ensure_ascii=False,
            )
        ]
    )
    provider = BossMCPJobProvider(
        search_tool=search_tool,
        detail_tool=FakeAsyncTool([]),
    )

    with pytest.raises(JobProviderError) as captured:
        await provider.search(
            tenant_id="tenant-a",
            criteria=SearchCriteria(query="Agent"),
        )

    assert captured.value.code == "AUTH_REQUIRED"
    assert captured.value.recovery_action == "boss login"


@pytest.mark.asyncio
async def test_detail_merges_full_jd_into_existing_posting() -> None:
    search_tool = FakeAsyncTool([_success({"count": 1, "jobs": [_search_item()]})])
    detail_tool = FakeAsyncTool(
        [
            _success(
                {
                    "job_id": "enc-job-1",
                    "security_id": "security-1",
                    "title": "高级 Agent 开发工程师",
                    "company": "示例科技",
                    "salary": "30-50K",
                    "city": "深圳",
                    "experience": "3-5年",
                    "education": "本科",
                    "description": "负责 Agent 平台设计与开发。",
                    "employment_type": "全职",
                    "company_info": {
                        "industry": "人工智能",
                        "scale": "100-499人",
                        "stage": "B轮",
                    },
                }
            )
        ]
    )
    provider = BossMCPJobProvider(
        search_tool=search_tool,
        detail_tool=detail_tool,
    )
    posting = (
        await provider.search(
            tenant_id="tenant-a",
            criteria=SearchCriteria(query="Agent"),
        )
    ).postings[0]

    detailed = await provider.get_detail(posting)

    assert detailed.job_id == posting.job_id
    assert detailed.detail_level == "full"
    assert detailed.description == "负责 Agent 平台设计与开发。"
    assert detailed.title == "高级 Agent 开发工程师"
    assert detail_tool.calls == [{"security_id": "security-1", "job_id": "enc-job-1"}]
