"""cl100k_base token counts for packing and request occupancy.

The main agent talks to an externally aliased OpenAI-compatible model, so
this does not sniff a hostname or model name for an encoding. cl100k_base is
the budget encoding. On Chinese-dominant payloads it overcounts versus o200k
(measured 57–69% on prose); on ASCII it tracks 1.00, and on some shell-script
text it undercounts (~0.96). Conservative for this repository's mix, not a
universal upper bound.

The 1.1 input-token safety factor is gone because it never covered the old
character-ratio estimator's actual error (40–80% on this corpus). A scalar
margin cannot stand in for an exact count.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

import tiktoken

from career_agent.agent.tiktoken_assets import (
    bundled_cl100k_vocab_path,
    bundled_tiktoken_cache_dir,
)

BUDGET_ENCODING = "cl100k_base"


@lru_cache(maxsize=1)
def budget_encoding() -> tiktoken.Encoding:
    should_revert = False
    if "TIKTOKEN_CACHE_DIR" not in os.environ:
        vocab = bundled_cl100k_vocab_path()
        if not vocab.is_file():
            raise RuntimeError(
                "bundled cl100k_base vocab is missing at "
                f"{vocab}. Fill it at install or build with "
                "`python -m career_agent.agent.tiktoken_assets`."
            )
        should_revert = True
        os.environ["TIKTOKEN_CACHE_DIR"] = str(bundled_tiktoken_cache_dir())
    try:
        return tiktoken.get_encoding(BUDGET_ENCODING)
    finally:
        if should_revert:
            del os.environ["TIKTOKEN_CACHE_DIR"]


def count_tokens(text: str) -> int:
    return len(budget_encoding().encode(text))


@lru_cache(maxsize=128)
def message_token_count(text: str) -> int:
    """``count_tokens`` for one message body, memoised.

    A turn projects the same messages more than once (pressure measurement, the
    load, a reload after a memory write), and each projection measures them.
    Kept apart from ``count_tokens`` because the whole serialised request goes
    through that one: every request is distinct, so its entries would never hit
    and would each be the size of the prompt. Message bodies are capped at
    ``max_message_chars``, which is what bounds this cache.
    """
    return count_tokens(text)


def clip_to_tokens(text: str, limit: int) -> tuple[str, bool]:
    """Return ``text`` cut at a token boundary, and whether anything was cut."""
    if limit < 1:
        raise ValueError("token limit must be at least 1")
    if message_token_count(text) <= limit:
        return text, False
    encoding = budget_encoding()
    ids = encoding.encode(text)
    end = limit
    while True:
        # A multi-byte character cut between two tokens decodes to U+FFFD.
        # Drop it rather than show the model a character nobody typed.
        clipped = encoding.decode(ids[:end]).rstrip("\ufffd")
        # Re-encoding a decoded prefix need not give back the same ids, so the
        # bound is checked on the text rather than assumed from the slice.
        if len(encoding.encode(clipped)) <= limit:
            return clipped, True
        end -= 1


def serialized_token_count(value: Any) -> int:
    return count_tokens(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    )
