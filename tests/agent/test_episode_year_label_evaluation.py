import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location("q42_year_labels", Path(__file__).parents[2] / "evals/episode_retrieval/temporal_v1/run_q42.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("answer,expected", [
    ("没有找到去年（2025 年）的字节投递记录。", "passed"),
    ("找到去年在字节的投递记录，2026 年 3 月已投递。", "failed"),
    ("没有去年字节的投递记录，但腾讯有。", "failed"),
    ("去年字节的投递记录没有找到；今年 2026 年有一条。", "human_review"),
    ("如果没有去年字节的记录，我就重新投。", "human_review"),
    ("不是没有去年字节的记录。", "human_review"),
    ("没有找到字节的记录。", "human_review"),
    ("关于你去年投递字节跳动的情况如下：投递职位：推荐算法工程师。当前状态：已投递。第一轮面试已完成，第二轮结果未通过。目前没有更多关于该职位的其他轮次或后续进展的记录。", "failed"),
    ("关于去年字节的投递记录如下，已投递，没有更多记录。", "failed"),
    ("去年字节第一轮已完成。没有找到更多记录。", "failed"),
    ("没有找到去年字节的投递记录。第二轮已结束。", "human_review"),
    ("去年字节的投递记录没有找到。", "passed"),
    ("", "failed"),
])
def test_q42_grading_requires_explicit_absence_and_reviews_ambiguity(answer, expected):
    assert MODULE.grade(answer, [{"title": "腾讯 · 工程师"}]) == expected


def test_frozen_fixture_builds_real_projection_before_sampling(tmp_path):
    captured = []
    class Client:
        def create(self, **request):
            captured.append(request)
            return SimpleNamespace(model_dump=lambda **kw: {"choices": [{"message": {
                "content": "没有找到去年（2025 年）的字节投递记录。",
            }}]})
    client = SimpleNamespace(chat=SimpleNamespace(completions=Client()))
    data = json.loads(MODULE.FIXTURE.read_text(encoding="utf-8"))
    assert {i["kind"] for i in data["corpus"]} <= {"mock_interview", "job_research", "application", "interview_round"}
    result = MODULE.run(SimpleNamespace(root=tmp_path, record=True, model="qwen3.7-plus"), data, 0, client)
    assert result["status"] == "valid"
    assert "今年" in captured[0]["messages"][1]["content"]
    assert len(captured[0]["tools"]) == 1
