"""One program-maintained organization alias table shared by all retrieval paths.

Use conservative identity aliases, not business subsidiaries or fuzzy spelling.
Seeds come from company publications listed in docs/iteration/117-episode-retrieval-fixes.md.
Only unambiguous full English names are included; unknown nicknames stay unchanged.
"""
from __future__ import annotations

import re

ORGANIZATION_ALIASES: dict[str, tuple[str, ...]] = {
    "字节跳动": ("ByteDance",),
    "拼多多": ("Pinduoduo",),
    "腾讯": ("Tencent",),
    "阿里巴巴": ("Alibaba",),
    "美团": ("Meituan",),
    "网易": ("NetEase",),
    "快手": ("Kuaishou",),
}

# Include canonical spellings and match longest first so 字节跳动 is not split.
_NAMES = {name.casefold(): canonical for canonical, aliases in ORGANIZATION_ALIASES.items()
          for name in (canonical, *aliases)}
_PATTERN = re.compile("|".join(
    r"(?<![a-z0-9_])" + re.escape(name) + r"(?![a-z0-9_])"
    if name.isascii() else re.escape(name)
    for name in sorted(_NAMES, key=lambda name: (-len(name), name))
), re.IGNORECASE)


def expand_organization_aliases(query: str) -> str:
    """Keep the surface term and add canonical identity for each known alias."""
    present = {match.group().casefold() for match in _PATTERN.finditer(query)}
    canonicals = tuple(dict.fromkeys(
        _NAMES[match.group().casefold()] for match in _PATTERN.finditer(query)
        if _NAMES[match.group().casefold()].casefold() not in present
    ))
    return " ".join((query, *canonicals)) if canonicals else query
