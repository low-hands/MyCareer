# Interaction-exit measurement — 2026-10-05

## Decision

Production keeps the current exit: first request with `tool_choice="auto"`; a text-only reply is rejected and the model is re-asked with `respond_to_user` forced. Neither `required` nor direction B is adopted. Revisit `required` only after the tool-selection working set offers just the tools usable now (see "When to revisit").

All runs used `qwen3.7-plus`, non-thinking, recorded to separate directories under `output/` (git-ignored, local only). The baseline cassettes in this directory were not overwritten. The numbers below are copied here because the raw recordings are not committed.

## Why the retry exists

On 10 interaction-heavy scenarios × 3 samples with the new retry telemetry, 19/28 interaction decisions (67.9%) needed a second request. All 19 retries were **text output rejected**; invalid interaction arguments, unoffered tools, argument pre-checks and transport retries were each 0. All 12 final answers were retried; 7 of 16 questions or questionnaires were retried. Each extra request costs about 3 s (median 3.2 s with one request, 6.2 s with two, in the earlier cassette statistics).

The cause is Qwen's habit of answering in prose first, not confusion between the four interaction functions. Merging the interaction exits into one would not remove the retry.

## Candidates

| Candidate | Change | Result |
| --- | --- | --- |
| `required` | First request with `tool_choice="required"` | Retries removed, latency halved, but the model is pushed toward acting instead of asking |
| B | Keep `auto`; on rejected text, keep the text and force `respond_to_user` only to classify it | No action bias, but loses the chance to change course on the second request; single-question `questions` violated the questionnaire contract |
| B2 (offline only) | B, with one returned question treated as `ask_user` on the preserved text | Counterfactual lower bound still above `auto` |

Qwen accepted and obeyed `required` in a minimal probe and in every measured decision, contrary to the Model Studio documentation's general statement that Qwen models cannot guarantee it. Re-probe on any model change.

## Full-catalogue comparison

46 scenarios × 3 samples (144) per candidate. Comparisons use the 44 scenarios common to all runs, weighting each scenario equally.

| Metric | `auto` (current) | `required` | B |
| --- | ---: | ---: | ---: |
| Mean scenario failure rate | 16.1% | 24.1% | 26.7% |
| Mean scenario business-tool call rate | 50.8% (52.8% on B's matched set) | 60.2% (59.3% on B's matched set) | 52.0% |
| Decision latency, median | 3.80 s | 1.85 s | 2.39 s |

`required`, whole catalogue: 41/144 samples failed across 17 scenarios, including **14 terminations from argument pre-check** (the model, forced to call something, invented arguments and was rejected twice) and **6 business-tool calls where a direct answer was expected**.

B: the preserved text was never rewritten (45 classification injections with matching hashes; 28 successful interactions with identical final bodies). The new failures were 17 classifications that returned exactly one question, which the questionnaire contract (2–8 questions) rejects. Classification output had a median of 111 tokens, not the few tokens expected, because the model still wrote questions.

Earlier 10-scenario run of `required` (for the failure shape): failures moved rather than simply grew. `an_invalid_selection_is_not_reconstructed` improved (auto 2/3 failed, required 0/3); `a_compacted_fact_without_page_in_is_not_invented` (an ablation that offers the span tool with no valid range) and `a_note_derived_filter_is_confirmed_with_the_user` (a tool still offered after a "confirm with the user" observation) newly failed 2/3 each.

## B2 offline replay

Classification question counts in B: 0 questions 21, 1 question 17, 2 questions 7.

Treating the 17 single-question classifications as `ask_user`: 6 still fail (2 should have ended directly, 4 should first read history); 11 cannot be verified because the recording kept only a hash of the original text. Even if all 11 passed, the scenario-weighted failure rate is at least **18.6%**, above `auto`'s 16.1%. This is a counterfactual lower bound, not a measured rate.

The 7 two-question questionnaires: 4 reasonable (city and role direction both missing), 1 ambiguous (a second question beyond the preference confirmation), 2 off-request (an empty conversation read followed by a question about JD analysis).

B's limitation is structural: the second request may only classify the text, so a model that should have called a business tool (for example, reading history) cannot.

## Limits of this evidence

- Three samples per scenario cannot establish "not worse" at a 2–3 point difference. The 16.1% vs 18.6% gap may be noise; the 8-point gap to `required`, together with its failure types, is the firmer signal.
- Future comparisons should fix a non-inferiority margin in advance (for example, at most +3 points), use paired per-scenario samples with a confidence interval, and record the original text, not only its hash.

## When to revisit

Part of `required`'s regression came from tools that should not have been offered: a tool without a usable prerequisite, and a tool awaiting user confirmation. Two selection rules follow and belong to the tool-selection work:

1. Do not offer a tool whose prerequisite is unmet.
2. After a tool returns a "needs user confirmation" state in this turn, do not offer it again until the user responds.

Re-measure `required` against `auto` once the working-set selector offers only tools usable now, using the protocol above. If it then meets the margin, `respond_to_user` remains only as a fallback and its removal can be evaluated.
