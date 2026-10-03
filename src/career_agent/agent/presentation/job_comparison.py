from __future__ import annotations

from career_agent.domain.job_comparison import DIMENSION_ORDER, JobComparison


_DIMENSION_LABELS = {
    "resume_fit": "简历匹配",
    "requirement_coverage": "要求覆盖",
    "salary_disclosure": "薪资披露",
    "location": "城市",
    "availability": "在招状态",
}

_VALUE_LABELS = {
    "strong": "强",
    "moderate": "中等",
    "weak": "弱",
    "insufficient_evidence": "证据不足",
    "full": "几乎全覆盖",
    "most": "多数覆盖",
    "partial": "部分覆盖",
    "little": "覆盖很少",
    "disclosed": "已披露",
    "undisclosed": "未披露",
    "matches_preference": "与默认城市一致",
    "differs": "与默认城市不同",
    "active": "在招",
    "closed": "已关闭",
    "unknown": "未知",
}


def render_job_comparison(comparison: JobComparison) -> str:
    """Render the matrix as a matrix, with no total and no implied ordering."""
    header = "| 岗位 | " + " | ".join(
        _DIMENSION_LABELS[dimension] for dimension in DIMENSION_ORDER
    ) + " |"
    separator = "|---" * (len(DIMENSION_ORDER) + 1) + "|"
    lines = [header, separator]
    for row in comparison.rows:
        cells = " | ".join(_VALUE_LABELS[cell.value] for cell in row.cells)
        lines.append(f"| {row.title} — {row.company_name} | {cells} |")

    blocks = ["# 岗位横向对比", "\n".join(lines)]
    blocks.append(
        "> 这是一张对照表，不是排名。各维度没有权重也没有总分，"
        "因为把不可通约的判断平均成一个数字只会掩盖它们的差别。"
    )

    details = []
    for row in comparison.rows:
        reasons = "\n".join(
            f"- {_DIMENSION_LABELS[cell.dimension]}："
            f"{_VALUE_LABELS[cell.value]} — {cell.basis}"
            for cell in row.cells
        )
        details.append(f"### {row.title} — {row.company_name}\n\n{reasons}")
    blocks.append("## 每格的依据\n\n" + "\n\n".join(details))

    if comparison.uninformative_dimensions:
        blank = "、".join(
            _DIMENSION_LABELS[dimension]
            for dimension in comparison.uninformative_dimensions
        )
        blocks.append(
            "## 这些维度分不出差别\n\n"
            f"{blank}：所有岗位在这些维度上都是未知。"
            "空白列的意思是缺少数据，不是这些岗位在这里没有区别。"
        )
    if comparison.notes:
        blocks.append("## 说明\n\n" + "\n".join(f"- {note}" for note in comparison.notes))
    return "\n\n".join(blocks)
