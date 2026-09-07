"""Credential scrubbing for untrusted external text and structured payloads.

Two entry points with different jobs:

- ``redact`` walks structured payloads (connector envelopes, hint dicts, log
  records) and blanks values under credential-shaped keys.
- ``redact_text`` scrubs free text that a model or a store is about to receive.

``redact_text`` deliberately preserves URLs while stripping their credential
parameters. Recruiting mail carries both meeting links the classifier must read
and magic-link tokens it must never see, so blanking whole URLs would trade a
leak for a capability loss.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit


REDACTED = "<redacted>"

_SENSITIVE_KEY = re.compile(
    r"(cookie|token|authorization|password|passwd|secret|stoken|credential"
    r"|api[_-]?key|access[_-]?key|private[_-]?key|session[_-]?id|otp|verification[_-]?code)",
    re.IGNORECASE,
)
_SAFE_NUMERIC_TOKEN_METRICS = frozenset(
    {
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "uncached_input_tokens",
    }
)

# Bare credential values that carry their own marker, so they are recognisable
# without a surrounding key.
_SENSITIVE_VALUE = re.compile(
    r"(?:__zp_stoken__|Bearer\s+|wt2=)[^\s,;]+",
    re.IGNORECASE,
)

# Parameter names that authenticate rather than address. Matched on a normalised
# name (separators removed, case folded) because the same credential arrives as
# `access_token`, `accessToken`, `X-Amz-Signature` and `magic-token`; an exact
# allowlist has to enumerate every spelling and silently passes the ones it missed.
_SENSITIVE_FRAGMENTS = (
    "token", "signature", "credential", "password", "passwd", "secret",
    "apikey", "accesskey", "privatekey", "sessionid", "securityid",
    "authorization", "jwt", "assertion", "nonce", "verificationcode",
)
# Fragments too short or too common to match as substrings: `key` would blank the
# `keyword` a JD search needs, `code` would blank `postcode`, `sig` would blank
# `signup`. These have to equal the whole normalised name.
_SENSITIVE_EXACT = frozenset({
    "auth", "code", "key", "sig", "session", "sid", "otp", "passcode",
    "ticket", "verify", "verification", "pwd", "state", "hmac", "mac", "salt",
})


def _is_sensitive_param(name: str) -> bool:
    normalised = re.sub(r"[^a-z0-9]", "", name.casefold())
    if normalised in _SENSITIVE_EXACT:
        return True
    return any(fragment in normalised for fragment in _SENSITIVE_FRAGMENTS)

# One-time codes announced in prose. Anchored on the label so ordinary numbers
# (salary, headcount, dates) survive: the classifier still needs those.
_CODE_PHRASE = re.compile(
    r"((?:验证码|校验码|动态码|激活码|verification\s+code|security\s+code|access\s+code"
    r"|one[-\s]?time\s+(?:code|password)|otp|passcode)"
    r"\s*(?:is|为|是|:|：)?\s*)([A-Za-z0-9]{4,12})\b",
    re.IGNORECASE,
)

_URL = re.compile(r"https?://[^\s<>\"'）)]+", re.IGNORECASE)
_MAX_NESTED_URL_DEPTH = 4
_MAX_PERCENT_DECODE_LAYERS = 3


def _redact_url(match: re.Match[str]) -> str:
    """Keep the address, drop the parameters that authenticate."""
    return _redact_url_value(match.group(0), depth=0)


def _redact_url_value(raw: str, *, depth: int) -> str:
    """Scrub one URL with an explicit bound for nested redirect URLs."""
    # Trailing punctuation belongs to the sentence, not the URL.
    trailing = ""
    while raw and raw[-1] in ".,;:!?。，、；：！？":
        trailing = raw[-1] + trailing
        raw = raw[:-1]
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return REDACTED + trailing
    # A fragment carrying key=value pairs is where OAuth-style magic links hide
    # their token, so it gets the same treatment as the query. A bare fragment
    # has no key to inspect and cannot be judged safe.
    fragment = parsed.fragment
    if fragment:
        fragment = _redact_pairs(fragment, depth=depth) if "=" in fragment else REDACTED
    # `https://user:secret@host/path` puts the credential in the authority, where
    # no parameter rule would ever see it. Keep the host, drop the userinfo.
    netloc = parsed.netloc
    if "@" in netloc:
        netloc = netloc.rsplit("@", 1)[1]
    cleaned = urlunsplit(
        (
            parsed.scheme,
            netloc,
            parsed.path,
            _redact_pairs(parsed.query, depth=depth),
            fragment,
        )
    )
    return cleaned + trailing


def _redact_pairs(raw: str, *, depth: int) -> str:
    """Blank authenticating parameters, leaving the marker readable.

    ``urlencode`` would percent-encode the marker into ``%3Credacted%3E``, which
    still leaks nothing but makes a reviewer squint to confirm it.
    """
    if not raw:
        return ""
    return "&".join(
        f"{key}={REDACTED}"
        if _is_sensitive_param(key)
        else urlencode([(key, _redact_nested(value, depth=depth))])
        for key, value in parse_qsl(raw, keep_blank_values=True)
    )


def _redact_nested(value: str, *, depth: int) -> str:
    """Scrub a parameter whose value is itself a URL.

    A `?next=`/`?redirect=` parameter is address-shaped, so blanking it would
    destroy information the reader needs — but percent-encoded inside it can sit a
    second query string with its own token. ``parse_qsl`` has already decoded one
    layer, so the inner URL is plain text by the time we see it.
    """
    leading = value[: len(value) - len(value.lstrip())]
    candidate = value.lstrip()
    for _ in range(_MAX_PERCENT_DECODE_LAYERS + 1):
        if candidate.lower().startswith(("http://", "https://")):
            if depth >= _MAX_NESTED_URL_DEPTH:
                return REDACTED
            scrubbed = _URL.sub(
                lambda match: _redact_url_value(match.group(0), depth=depth + 1),
                candidate,
            )
            return leading + scrubbed
        decoded = unquote(candidate)
        if decoded == candidate:
            return value
        candidate = decoded
    # A repeatedly encoded URL-like value beyond the inspection budget is
    # untrusted, not safe merely because we stopped decoding it.
    if candidate.casefold().startswith(("http%", "https%")):
        return REDACTED
    return value


def redact_text(value: str) -> str:
    """Scrub credentials from free text bound for a model or a store."""
    if not isinstance(value, str):
        return value
    scrubbed = _SENSITIVE_VALUE.sub(REDACTED, value)
    scrubbed = _URL.sub(_redact_url, scrubbed)
    return _CODE_PHRASE.sub(lambda match: match.group(1) + REDACTED, scrubbed)


def redact(value: Any) -> Any:
    """Blank credential-shaped keys and scrub the strings underneath."""
    if isinstance(value, dict):
        return {
            key: (
                item
                if str(key) in _SAFE_NUMERIC_TOKEN_METRICS
                and type(item) is int
                else REDACTED
                if _SENSITIVE_KEY.search(str(key))
                else redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        redacted = [redact(item) for item in value]
        return tuple(redacted) if isinstance(value, tuple) else redacted
    if isinstance(value, str):
        return redact_text(value)
    return value
