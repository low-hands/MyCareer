from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


_SECTION_LABELS = {
    "overdue": "已逾期",
    "due_today": "今天到期",
    "upcoming": "即将到期",
    "no_due_date": "未设日期",
}


def render_daily_brief(payload: Mapping[str, Any]) -> str:
    """Render the complete, ephemeral brief directly into the message."""
    blocks = ["# 今日职业简报"]
    total = 0
    for key, label in _SECTION_LABELS.items():
        raw_items = payload.get(key)
        if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes)):
            continue
        lines = []
        for raw in raw_items:
            if not isinstance(raw, Mapping):
                continue
            total += 1
            title = str(raw.get("title") or "未命名待办")
            summary = str(raw.get("summary") or "").strip()
            due_at = raw.get("due_at")
            suffix = f"（{due_at}）" if due_at else ""
            lines.append(f"- {title}{suffix}" + (f"：{summary}" if summary else ""))
        if lines:
            blocks.append(f"## {label}\n\n" + "\n".join(lines))
    if total == 0:
        blocks.append("今天没有待办事项。")
    generated_at = payload.get("generated_at")
    timezone = payload.get("timezone")
    if generated_at:
        blocks.append(f"> 生成时间：{generated_at}" + (f"（{timezone}）" if timezone else ""))
    return "\n\n".join(blocks)
