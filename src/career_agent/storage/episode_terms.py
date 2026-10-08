"""Shared lexical terms for episode search and automatic projection."""

from __future__ import annotations

import re
from collections.abc import Iterable

_CJK = re.compile(r"[\u4e00-\u9fff]")
_LATIN = re.compile(r"[a-z0-9][a-z0-9+#.×-]*", re.IGNORECASE)
MAX_SEARCH_TERMS = 32


def search_terms(text: str, entities: Iterable[str]) -> tuple[str, ...]:
    """Keep known names whole, then split other Chinese into bigrams/trigrams."""
    normalized = text.casefold()
    names = sorted(
        {name.strip().casefold() for name in entities if name.strip()},
        key=len,
        reverse=True,
    )
    anchors: list[str] = []
    fragments: list[str] = []
    index = 0
    while index < len(normalized):
        name = next((value for value in names if normalized.startswith(value, index)), None)
        if name is not None:
            anchors.append(name)
            index += len(name)
            continue
        latin = _LATIN.match(normalized, index)
        if latin is not None:
            anchors.append(latin.group())
            index = latin.end()
            continue
        if _CJK.match(normalized, index):
            end = index + 1
            while end < len(normalized) and _CJK.match(normalized, end) and not any(
                normalized.startswith(value, end) for value in names
            ):
                end += 1
            run = normalized[index:end]
            if len(run) == 1:
                fragments.append(run)
            for width in (2, 3):
                fragments.extend(run[pos : pos + width] for pos in range(len(run) - width + 1))
            index = end
            continue
        index += 1
    return tuple(dict.fromkeys((*anchors, *fragments)))[:MAX_SEARCH_TERMS]


def title_entities(titles: Iterable[str]) -> tuple[str, ...]:
    """Extract company and role labels from structured episode titles."""
    entities: set[str] = set()
    fixed = {"公司调研", "简历定制已完成", "面试", "岗位研究", "投递记录", "求职意图已确认"}
    for title in titles:
        for part in title.split("·"):
            label = part.strip().removeprefix("模拟面试：").strip()
            if not label or label in fixed or re.fullmatch(r"第\s*\d+\s*轮面试", label):
                continue
            entities.add(label.casefold())
    return tuple(sorted(entities, key=len, reverse=True))
