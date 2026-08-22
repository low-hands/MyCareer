from __future__ import annotations

import re
from typing import Any


_SENSITIVE_KEY = re.compile(r"(cookie|token|authorization|password|secret|stoken)", re.IGNORECASE)
_SENSITIVE_VALUE = re.compile(r"(?:__zp_stoken__|Bearer\s+|wt2=)[^\s,;]+", re.IGNORECASE)


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: "<redacted>" if _SENSITIVE_KEY.search(str(key)) else redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return _SENSITIVE_VALUE.sub("<redacted>", value)
    return value
