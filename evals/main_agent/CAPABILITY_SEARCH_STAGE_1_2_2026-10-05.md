# Capability search, stages 1–2 — 2026-10-05

This stage registers `search_capabilities` and persists names it loads. It does
not offer the tool in any legacy profile or change the `decide` selection rule.

## Visibility and fingerprints

| Surface | Result |
| --- | --- |
| Registered schemas | 72; `search_capabilities` is last |
| Registered schema SHA-256 | `47d2012a1dc66fb7dd6355cd3cd0c55186ebc0e2506ba0be7a832a976ca62a78` |
| Legacy-visible schemas | Original 71, unchanged in every profile |
| Legacy-visible schema SHA-256 | `f801c8fd86e99a8704f1d5f1189099598f37134703dd285078bec64b013ffdef` |
| Legacy-visible prompt fingerprint | `077b9db3d56cdd8b7f07e8b364b875e5a7db6dbcbc3f448b5500d4f1edf2fe40` |

The existing cassette staleness test still reports its two previously stale
recordings; this stage adds no new staleness. `loaded_capabilities` is durable
task state and is absent from `model_context()`.

## Search recall

The test uses the **original first user message** of each development case as
the query. A hit means that at least one accepted first-step business tool is
in the first five search results. Two development controls have no positive
first-step demand and are excluded, leaving 43 cases. Later steps are not
scored from the first message because they may depend on a tool observation.

| Configuration | First-step top-5 recall |
| --- | ---: |
| Weighted BM25 only | **30/43 (69.8%)** |
| BM25 + semantic RRF | Not measured: no `CAREER_EMBEDDING_*` provider is configured locally |

The CJK tokenizer now uses adjacent two-character terms; it retains a single
character only when the whole Chinese segment has one character. The lexical
first-step result fell from 31/43 (72.1%) to 30/43 (69.8%). Irrelevant
queries such as `天气怎么样` and `的` now produce zero results. An exact
`names=["memory.proposals"]` request loads all 10 members even with `limit=1`;
`limit` applies to ranked query results only.

All **141 Chinese aliases** of the 70 searchable tools retrieve their owner in
the top five with lexical search. This is a metadata sanity check, not a
conversational recall measure. The semantic index and RRF path are tested with
an injected deterministic embedding client, including catalog-hash caching,
minimum similarity 0.55, and at most 10 semantic candidates. Its catalogue
vectors are prepared during runtime composition. A search embeds only its
query with a 3-second timeout and falls back to BM25 on failure;
that synthetic client cannot estimate production semantic recall. Semantic
retrieval should remain opt-in until a configured provider is evaluated on
the same development queries and then on the frozen holdouts.

## Test status

The focused catalog, search, and offline-selection tests pass. The complete
suite was run with `PYTHONUTF8=1` to avoid Windows GBK decode errors and a
writable pytest temp directory. The JUnit run reported **2,494 passed, 56
failed, 6 errors, and 4 expected failures**. A previous full-suite log lists
160 failures. Of the current 56 failure identifiers, 54 also appear in that
older log. The two new-only identifiers are outside this change: one timed out
waiting for an API worker thread (it passed when rerun alone) and one asserts
Unix `0600` permissions on a Windows file (it fails when rerun alone). The six
setup errors include very long path test parameters.
The aggregate is not a clean pass; the focused search and visibility tests
pass, and the failure-list comparison found no new failure in changed areas.
