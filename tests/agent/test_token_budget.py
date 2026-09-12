import os
from pathlib import Path

import pytest
import tiktoken.load

from career_agent.agent.tiktoken_assets import (
    CL100K_BLOB_URL,
    bundled_cl100k_vocab_path,
    populate_bundled_tiktoken_cache,
    tiktoken_cache_key,
)
from career_agent.agent.token_budget import (
    BUDGET_ENCODING,
    budget_encoding,
    clip_to_tokens,
    count_tokens,
    serialized_token_count,
)

# Rare characters on purpose: several of them take more than one token, so some
# cuts land inside a character.
CHINESE_JD_LINE = (
    "负责大模型推理服务的性能优化与稳定性建设，熟悉分布式训练框架，"
    "具备鑫龘饕餮等生僻字处理经验；"
)


def test_budget_encoding_is_cl100k_base() -> None:
    assert budget_encoding().name == BUDGET_ENCODING == "cl100k_base"


def test_cjk_costs_more_than_ascii_of_equal_length() -> None:
    assert count_tokens("中" * 400) > count_tokens("a" * 400)


def test_serialized_token_count_matches_dumped_json() -> None:
    payload = {"default_city": None, "hard_constraints": []}
    dumped = '{"default_city": null, "hard_constraints": []}'
    assert serialized_token_count(payload) == count_tokens(dumped)


def test_tiktoken_cache_key_is_sha1_of_the_blob_url() -> None:
    assert tiktoken_cache_key(CL100K_BLOB_URL) == (
        "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"
    )
    assert tiktoken_cache_key(CL100K_BLOB_URL) != "cl100k_base.tiktoken"


def test_bundled_vocab_uses_url_sha1_filename() -> None:
    populate_bundled_tiktoken_cache()
    vocab = bundled_cl100k_vocab_path()
    assert vocab.is_file()
    assert vocab.name == tiktoken_cache_key(CL100K_BLOB_URL)


def test_budget_encoding_restores_unset_cache_dir(monkeypatch) -> None:
    budget_encoding.cache_clear()
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)
    populate_bundled_tiktoken_cache()
    budget_encoding()
    assert "TIKTOKEN_CACHE_DIR" not in os.environ


def test_budget_encoding_does_not_override_user_cache_dir(
    monkeypatch, tmp_path: Path
) -> None:
    populate_bundled_tiktoken_cache()
    shutil_copy = tmp_path / bundled_cl100k_vocab_path().name
    shutil_copy.write_bytes(bundled_cl100k_vocab_path().read_bytes())
    budget_encoding.cache_clear()
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))
    encoding = budget_encoding()
    assert encoding.name == BUDGET_ENCODING
    assert os.environ["TIKTOKEN_CACHE_DIR"] == str(tmp_path)


def test_budget_encoding_does_not_fetch_when_bundled_vocab_is_present(
    monkeypatch,
) -> None:
    populate_bundled_tiktoken_cache()
    budget_encoding.cache_clear()
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)

    def fail_fetch(*_args, **_kwargs):
        raise AssertionError("tiktoken tried to fetch vocab")

    monkeypatch.setattr(tiktoken.load, "read_file", fail_fetch)
    assert budget_encoding().name == BUDGET_ENCODING


def test_missing_bundled_vocab_fails_without_fetching(
    monkeypatch, tmp_path: Path
) -> None:
    budget_encoding.cache_clear()
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)
    monkeypatch.setattr(
        "career_agent.agent.token_budget.bundled_cl100k_vocab_path",
        lambda: tmp_path / "missing",
    )

    def fail_fetch(*_args, **_kwargs):
        raise AssertionError("tiktoken tried to fetch vocab")

    monkeypatch.setattr(tiktoken.load, "read_file", fail_fetch)
    with pytest.raises(RuntimeError, match="bundled cl100k_base vocab is missing"):
        budget_encoding()


def test_clip_to_tokens_returns_text_within_the_limit_unchanged() -> None:
    text = "Ship the retrieval pipeline."
    assert clip_to_tokens(text, count_tokens(text)) == (text, False)
    assert clip_to_tokens(CHINESE_JD_LINE, 10_000) == (CHINESE_JD_LINE, False)


def test_clip_to_tokens_stays_in_the_limit_and_drops_a_split_character() -> None:
    populate_bundled_tiktoken_cache()
    encoding = budget_encoding()
    ids = encoding.encode(CHINESE_JD_LINE)
    # Without a cut that lands inside a character, the replacement-character
    # assertion below would pass for any implementation.
    assert any(
        encoding.decode(ids[:limit]).endswith("\ufffd")
        for limit in range(1, len(ids))
    )

    for limit in range(1, len(ids)):
        clipped, was_clipped = clip_to_tokens(CHINESE_JD_LINE, limit)
        assert was_clipped is True
        assert count_tokens(clipped) <= limit
        assert not clipped.endswith("\ufffd")
        assert CHINESE_JD_LINE.startswith(clipped)


@pytest.mark.parametrize("limit", [0, -1])
def test_clip_to_tokens_rejects_a_limit_below_one(limit: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        clip_to_tokens("text", limit)
