"""Organization identity aliases and target-company-only short names.

Retrieval uses conservative full-name aliases. Short names are limited to
target-company normalization and evidence checks because they can be ambiguous
in general search text.
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

# Names users commonly use for a target company. These do not expand search
# queries: a short surface form such as "字节" is too ambiguous for retrieval.
TARGET_COMPANY_SHORT_NAMES: dict[str, tuple[str, ...]] = {
    "字节跳动": ("字节",),
    "阿里巴巴": ("阿里",),
    "拼多多": ("PDD",),
}

# Include canonical spellings and match longest first so 字节跳动 is not split.
_NAMES = {name.casefold(): canonical for canonical, aliases in ORGANIZATION_ALIASES.items()
          for name in (canonical, *aliases)}
_TARGET_COMPANY_NAMES = {
    name.casefold(): canonical
    for canonical, aliases in TARGET_COMPANY_SHORT_NAMES.items()
    for name in aliases
}


def canonical_organization_name(name: str) -> str:
    """Resolve a target company's known full or short name."""
    stripped = name.strip()
    return _NAMES.get(stripped.casefold(), _TARGET_COMPANY_NAMES.get(stripped.casefold(), stripped))


def target_company_scope_key(name: str) -> str:
    """Use one case-insensitive version chain per target company."""
    canonical = canonical_organization_name(name)
    return "person_intent/self/target_company/" + canonical.casefold().replace("%", "%25").replace("/", "%2F")
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


def organization_mentioned(name: str, message: str) -> bool:
    """Require a proposed company or a known alias in this user's own message."""
    canonical = canonical_organization_name(name)
    aliases = (
        canonical, *ORGANIZATION_ALIASES.get(canonical, ()),
        *TARGET_COMPANY_SHORT_NAMES.get(canonical, ()),
    )
    return any(
        re.search(
            (r"(?<![a-z0-9_])" + re.escape(alias) + r"(?![a-z0-9_])")
            if alias.isascii() else re.escape(alias),
            message,
            flags=re.IGNORECASE,
        )
        for alias in aliases
    )
