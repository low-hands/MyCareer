# Offline tool-selection baseline — 2026-10-05

This report measures which schemas the current mechanism offers at each decision snapshot. It calls no model and changes no prompt, schema, or cassette. Run `pytest tests/agent/test_tool_selection_evaluation.py` to reproduce the development baseline.

## 2B revised selection baseline

There are **65 selection cases: 45 development and 20 original holdout**, plus **12 separately authored independent holdout cases** reserved for final evaluation. The 46 existing trajectory scenarios remain separate. The 65 cases comprise 34 single, 14 cross, 9 chain, and 8 control cases. Development has 27 single, 9 cross, 6 chain, and 3 control cases; the original holdout has 7, 5, 3, and 5 respectively. Thus both splits exercise cross-tool-group requests and multistep continuation.

New first-turn cases start with `tool_profile=core`; the legacy baseline applies the real `keyword_tool_profile` ingress before the first decision. Cross-group cases start in the prior group. The five older offline probes are already post-ingress snapshots and are not routed again. After a new user message, ingress is applied again; within a turn the projected task state persists. This corrects the earlier preselected-profile baseline.

The original holdout's fixture manifest SHA-256 is `c2f8477110cf2591f8ccbe9d707071ef4262667fd438ff9607939d44705ef209`. **Change log:** 2026-10-05, revised the split, fixed the clock, corrected scenario state and gold expectations, and added `raw_turn` to distinguish ingress snapshots; the prior `e073edab2d2d0c5fb83e1e9e9872f22226f54ca54ce4690f09709c3787963183` freeze is superseded. The original holdout was authored by the same implementer who saw catalog metadata, so it is a frozen regression set rather than a blind sample.

A separate agent wrote the 12 independent prompts and next-action expectations from the case-writing rules without seeing catalog aliases, successor edges, or existing fixtures. They are stored in `independent_tool_selection_holdout.py` and have a separate manifest SHA-256 of `f7fcfef7322a566a80e739fd5cb1690626adf2b59123a1e53d5d9e5c70b23804`. The implementation bound their references to runtime-shaped task states and checked that every expected tool is reachable. They have **not been used for selector tuning or scored in this report**. Their wording and gold choices were independently authored; implementation and state validation were performed in this repository.

### LegacyProfileSelector by split and case type

Recall is first-offer coverage of business-tool demand snapshots. Size and token ranges exclude the two synthetic post-route snapshots. The last three metrics count (snapshot, tool) exposures, not unique tools. Schema tokens use the repository's proxy tokenizer.

| Split | Type | Cases | Snapshots / comparable | Recall | Route steps | Offered tools, mean (range) | Schema tokens, mean (range) | Unreachable offered | Waiting reoffered | Unrequested writes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dev | All | 45 | 66 / 64 | **46 / 62 (74.2%)** | 2 | 24.6 (17–30) | 4,827 (3,018–7,111) | 339 | 1 | 469 |
| Dev | Single | 27 | 27 / 27 | 21 / 27 | 0 | 23.4 (17–30) | 4,470 (3,018–7,111) | 150 | 0 | 204 |
| Dev | Cross | 9 | 20 / 18 | 10 / 18 | 2 | 24.9 (23–27) | 4,772 (4,114–5,389) | 88 | 0 | 132 |
| Dev | Chain | 6 | 16 / 16 | 14 / 16 | 0 | 26.1 (23–30) | 5,311 (4,114–7,111) | 85 | 0 | 107 |
| Dev | Control | 3 | 3 / 3 | 1 / 1 | 0 | 26.7 (25–30) | 5,788 (5,127–7,111) | 16 | 1 | 26 |
| Original holdout | All | 20 | 32 / 32 | **26 / 29 (89.7%)** | 0 | 26.2 (17–31) | 5,327 (3,018–7,201) | 173 | 2 | 249 |
| Original holdout | Single | 7 | 7 / 7 | 6 / 7 | 0 | 23.3 (17–30) | 4,517 (3,018–7,111) | 50 | 0 | 56 |
| Original holdout | Cross | 5 | 12 / 12 | 10 / 12 | 0 | 26.9 (17–30) | 5,494 (3,018–7,111) | 63 | 0 | 84 |
| Original holdout | Chain | 3 | 7 / 7 | 7 / 7 | 0 | 28.3 (25–31) | 6,095 (5,127–7,201) | 33 | 1 | 66 |
| Original holdout | Control | 5 | 6 / 6 | 3 / 3 | 0 | 25.5 (17–30) | 5,043 (3,018–7,111) | 27 | 1 | 43 |

The 46 cassette scenarios separately remain **20/23** first-offer coverage. The 65 selection cases demand **55 of 71** model-callable tools. Development demands 39 distinct tools; the original holdout adds 16. Each of the 17 business namespaces has at least two development cases. The `control` namespace also has three development controls; its migration tool `route_to_capability` is intentionally not a positive business demand.

| Namespace | Dev | Original holdout | Namespace | Dev | Original holdout |
| --- | ---: | ---: | --- | ---: | ---: |
| `context` | 3 | 1 | `actions` | 2 | 2 |
| `job.library` | 6 | 2 | `job.analysis` | 4 | 0 |
| `job.research` | 2 | 1 | `job.intent` | 2 | 1 |
| `resume.library` | 3 | 1 | `resume.match` | 4 | 1 |
| `resume.tailoring` | 5 | 1 | `application.tracking` | 7 | 2 |
| `application.email` | 5 | 0 | `interview.schedule` | 5 | 4 |
| `interview.prep` | 4 | 1 | `interview.calendar` | 3 | 3 |
| `interview.mock` | 2 | 0 | `memory.search` | 4 | 0 |
| `memory.proposals` | 2 | 3 | `control` | 3 | 5 |

The comparison metrics are:

1. **Unreachable offered:** an offered tool does not meet its state prerequisite.
2. **Waiting reoffered:** a tool whose observation says user confirmation is needed is offered again in the same turn. The check includes observations already present in the initial snapshot and `working_notes_derived_argument`; a new user message resets it. The development control now exposes one such reoffer.
3. **Unrequested writes:** an offered write tool is neither an accepted next action nor in a namespace named by the case's user intent. This is a coarse trend metric, not an authorization verdict.

### Missing business tools at first offer

All 19 misses are `not_selected`; none is `unreachable`.

| Split | Case | Step | Missing business tool |
| --- | --- | ---: | --- |
| Dev | `offline_resume_to_interview_preparation` | 0 | `prepare_interview` |
| Dev | `offline_job_to_application_creation` | 0 | `create_application` |
| Dev | `selection_dev_job_analysis_first` | 0 | `analyze_job` |
| Dev | `selection_dev_job_research_first` | 0 | `research_job` |
| Dev | `selection_dev_job_intent_first` | 0 | `propose_job_intent` |
| Dev | `selection_dev_interview_prep_first` | 0 | `prepare_interview` |
| Dev | `selection_dev_memory_search_first` | 0 | `search_career_history` |
| Dev | `selection_dev_memory_proposals_first` | 0 | `propose_career_fact` |
| Dev | `selection_dev_cross_compare_then_match` | 2 | `match_resume_to_job` |
| Dev | `selection_dev_cross_export_then_prepare` | 1 | `prepare_interview` |
| Dev | `selection_dev_cross_read_job_then_track` | 1 | `create_application` |
| Dev | `selection_dev_cross_memory_then_job` | 1 | `search_career_episodes` |
| Dev | `selection_dev_cross_application_then_interview` | 1 | `create_interview` |
| Dev | `selection_dev_cross_source_then_tailor` | 1 | `draft_resume_tailoring` |
| Dev | `selection_dev_chain_job_resume` | 2 | `match_resume_to_job` |
| Dev | `selection_dev_chain_job_resume` | 3 | `draft_resume_tailoring` |
| Original holdout | `selection_holdout_interview_calendar_first` | 0 | `list_calendar_accounts` |
| Original holdout | `selection_holdout_cross_research_retry_then_application` | 1 | `create_application` |
| Original holdout | `selection_holdout_cross_retro_then_memory` | 1 | `propose_career_fact` |

The development counts and miss list are locked in pytest. The original and independent holdouts lock their fixtures and evaluability, but no recall score. During selector tuning, use development only. Score both holdouts once after the selector is fixed and report them separately.

## Earlier 2A snapshot (historical)

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
