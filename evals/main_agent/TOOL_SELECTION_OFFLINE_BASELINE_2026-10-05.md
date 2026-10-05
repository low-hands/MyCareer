# Offline tool-selection baseline — 2026-10-05

This report evaluates **which schemas were offered at each decision snapshot**, without calling a model, changing a prompt, or re-recording a cassette. It uses the existing 46 trajectory scenarios plus five new offline-only follow-on scenarios. Run `pytest tests/agent/test_tool_selection_evaluation.py` to reproduce the locked counts and missing-step list.

## Measurement contract

- `LegacyProfileSelector` reproduces the current `decide` offer: `tool_profile` selects a registered schema subset, and schema-gated prerequisites are filtered. A test compares its schema tuple with `DecisionEngine.tool_schemas` at **every snapshot**. Existing trajectory contexts are already *post-ingress* decision snapshots. `select_ingress` separately reproduces `keyword_tool_profile` promotion for raw turns whose stored profile is `core`; applying ingress a second time to recorded snapshots would change what those cassettes tested.
- Each offered tool records its source as `core` or `profile:<name>`. This is provenance of the **current mechanism**, not a recommendation to retain profiles.
- A step with `expect_tool` or `expect_tools` is a business-tool demand. A step expecting `route_to_capability` is charged one extra decision round-trip and evaluated against the **following business tool** at the pre-route snapshot. The post-route snapshot is marked `legacy_only` and excluded from comparative recall and size figures: it contains a profile switch and observation that only the old mechanism would produce. Each business demand therefore enters the denominator **once, when it first appears**. Steps without an explicit positive tool expectation are excluded from the recall denominator; negative assertions still belong to the existing trajectory tests.
- A missing required tool is labelled `not_selected` if absent from the selector's candidate set, or `unreachable` if selected but filtered by a prerequisite. The selector interface passes an opaque state value from one step to the next and resets it between scenarios, allowing future working-set or search-loading selectors to retain their own state. The current selector ignores that value.
- Schema-token numbers use the repository's bundled `cl100k_base` budget encoding on compact JSON schemas. They are a **proxy**, not Qwen's tokenizer count or complete request tokens.

| Set | All snapshots / comparable | First-demand coverage | First-offer recall | Expected route steps | Mean offered tools (range) | Mean schema-token proxy (range) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Existing 46 scenarios | 54 / 51 | 20 / 23 | 87.0% | 3 | 23.7 (17–31) | 4,612 (3,018–7,201) |
| Five offline-only scenarios | 10 / 8 | 6 / 8 | 75.0% | 2 | 26.1 (23–30) | 5,011 (4,114–7,111) |
| Combined | 64 / 59 | 26 / 31 | **83.9%** | 5 | 24.1 (17–31) | 4,666 (3,018–7,201) |

The expected-route count is **five in one pass over all declared scenarios**, not five Qwen calls in production. The preserved Qwen cassettes contain four actual `route_to_capability` calls across 96 recordings; they are a separate stochastic observation.

The 31 first-demand snapshots require **22 distinct model-callable tools out of 71 (31.0%)**. Positive demands touch all existing tool groups (`core`, `job`, `resume`, `application`, `interview`, `memory`), but most tools have no positive selection test. The catalogue was originally designed mainly for policy and negative-behaviour checks; this sample is too narrow to tune a selector to the reported recall alone. Expand representative positive tasks before choosing selector thresholds.

## Missing business tools at the pre-route snapshot

| Scenario | Step | Required business tool absent before route | Reason |
| --- | ---: | --- | --- |
| `a_core_request_routes_before_job_analysis` | 0 | `analyze_job` | `not_selected` |
| `tool_selection_switches_from_resume_to_job_research` | 0 | `research_job` | `not_selected` |
| `tool_selection_combines_job_analysis_and_resume_match` | 1 | `match_resume_to_job` | `not_selected` |
| `offline_resume_to_interview_preparation` | 0 | `prepare_interview` | `not_selected` |
| `offline_job_to_application_creation` | 0 | `create_application` | `not_selected` |

The other three new probes — application → email events, interview → calendar proposal, and match → resume tailoring — have no tool-group switch in the current catalog and show no missing tool. They are useful controls for follow-on selection after a tool result.

## Classified Qwen behaviour failures

After the 2026-10-05 re-record (see [QWEN_BASELINE_2026-10-04.md](QWEN_BASELINE_2026-10-04.md#current-state--re-record-of-2026-10-05)), 10 fresh cassettes fail a behaviour assertion and 2 are stale. Stale cassettes are excluded from classification and are **not** passes. The original 2026-10-04 recording had 13 failures (2 / 7 / 4 in the categories below).

The categories are **not mutually exclusive causes**. In two recorded scenarios, a business tool is absent before routing *and* the model makes a wrong decision when a path is available. They must not be reported as pure selector failures. The evaluation derives the two tool-related classifications from the recorded calls and the actual offer at each failed step; only the remaining behavior types need human interpretation. The separate quality-floor failure is outside these hard-behaviour counts.

| Classification | Scenarios | Count |
| --- | --- | ---: |
| Pre-route selection gap **and** model decision error | `tool_selection_switches_from_resume_to_job_research`; `tool_selection_combines_job_analysis_and_resume_match` | 2 |
| Expected tool was offered, or a forbidden tool was called | `research_is_not_started_as_part_of_matching`; `a_note_derived_filter_is_confirmed_with_the_user`; `a_report_made_this_turn_without_an_index_cannot_be_named`; `a_compacted_fact_is_paged_in_rather_than_guessed`; `a_long_compacted_history_is_searched_in_one_page_in_call` | 5 |
| Other behavior: user confirmation, reference grounding, or response delivery | `working_notes_never_choose_or_rank_a_job`; `a_report_that_scrolled_out_of_the_catalogue_is_not_faked`; `an_empty_conversation_span_is_not_filled_from_the_window` | 3 |
| Stale, not classified | `stated_intent_is_proposed_before_it_is_recorded`; `a_questionnaire_answer_is_proposed_with_user_input_provenance` | 2 |

Both cross-group scenarios now pass 0/3. Job analysis → resume match repeats the earlier pattern: after the job step every sample calls `list_resumes`, both before the route and in the synthetic post-route snapshot where `match_resume_to_job` is offered. Resume → job research, which passed 2/3 on 2026-10-04, no longer calls the route in any sample (`get_saved_job → research_job`, `list_resumes → ask_user`, `get_saved_job → get_saved_job`). The selection gap is real, but these results cannot be attributed solely to selector recall.

## Scope of the five new probes

These are declared standard trajectories using fixed task state and synthetic observations. They are deliberately outside `SCENARIOS`, so they require no cassette and do not change existing prompt or schema fingerprints. Two cross tool groups (`resume` → interview and `job` → application); three test a follow-on tool within the same group (application → emails, interview → calendar, match → tailoring). They measure tool availability, **not** whether Qwen would complete those workflows.
