# Qwen trajectory baseline — 2026-10-04 (Vancouver)

Model: `qwen3.7-plus`. All 46 scenario cassettes were recorded with the current prompt and schemas. Contract failures: 0. Stale or missing cassettes: 0. The offline evaluator reports 13 hard-behaviour failures and one quality failure. Pytest reports 82 passed, 12 failed, and two expected failures because known-gap scenarios are marked `xfail`.

The recordings are evidence of the model's current behaviour. Do not change fingerprints, expected actions, or quality floors just to make this run green. Re-record after changing the prompt or a model-visible schema.

## Current state — re-record of 2026-10-05

The figures in the rest of this report describe the 2026-10-04 recording. After the scenario-specific prompt lines were removed and the native interaction outputs settled, 44 of 46 cassettes were re-recorded with the same model. The offline evaluator now reports: contract failures 0, **behaviour failures 10, quality failures 1, stale 2**.

| Status | Scenarios |
| --- | --- |
| Behaviour failure | `working_notes_never_choose_or_rank_a_job`; `a_note_derived_filter_is_confirmed_with_the_user` (new); `a_report_that_scrolled_out_of_the_catalogue_is_not_faked`; `a_compacted_fact_is_paged_in_rather_than_guessed`; `a_long_compacted_history_is_searched_in_one_page_in_call`; `an_empty_conversation_span_is_not_filled_from_the_window`; `tool_selection_switches_from_resume_to_job_research`; `tool_selection_combines_job_analysis_and_resume_match`; `research_is_not_started_as_part_of_matching`; `a_report_made_this_turn_without_an_index_cannot_be_named` — the last four are declared `known_gap` and report as `xfail`; each still counts as a behaviour failure, and its test fails if all samples pass so the marker must be removed |
| Quality floor | `a_report_that_scrolled_out_of_the_catalogue_is_not_faked` |
| **Stale, not evaluated** | `stated_intent_is_proposed_before_it_is_recorded`; `a_questionnaire_answer_is_proposed_with_user_input_provenance` — both recordings failed on business-argument pre-check, so they still carry the previous prompt fingerprint. Neither counts as a pass until re-recorded. |

`a_card_backed_report_is_answered_without_reproducing_it` and `an_uncertain_calendar_write_is_not_reissued_or_claimed` passed on re-record. `test_a_reply_does_not_take_over_delivery_that_belongs_elsewhere` also fails, on a table in a `working_notes_never_choose_or_rank_a_job` reply; it failed on the previous recordings too.

Interaction-exit experiments on these cassettes are in [INTERACTION_EXIT_MEASUREMENT_2026-10-05.md](INTERACTION_EXIT_MEASUREMENT_2026-10-05.md).

## Tool-selection baseline before changing the selector

These two additional scenarios use the live Qwen model with fixed, synthetic tool observations between decisions. Latency is the wall time of model decisions and recording retries; it excludes real tool execution and is not an end-to-end user-turn latency. Each scenario has three independent samples.

| Scenario | Full sequence passed | Model requests per sample | Model-decision time, median (range) | Observed calls |
| --- | ---: | ---: | ---: | --- |
| Resume profile → job research | 2/3 | 2 | 10.2 s (5.4–11.1 s) | `route_to_capability → research_job` twice; `route_to_capability → get_saved_job` once |
| Job analysis → resume match | 0/3 | 3 | 5.8 s (5.8–6.2 s) | `analyze_job → list_resumes → list_resumes` in all three samples |

The second scenario is the concrete cross-domain failure: after the synthetic job-analysis result, the model does not switch to the resume capability and does not call `match_resume_to_job`, even though a resume version is already bound. These measurements are the pre-change comparator for dynamic capability selection and cross-domain union. Three samples establish an observable failure, not a precise population success rate or latency percentile.

## Failing behaviours

| Area | Scenario | Recorded issue |
| --- | --- | --- |
| Tool choice | `research_is_not_started_as_part_of_matching` | Chose `list_resumes` instead of the already reachable `match_resume_to_job`. |
| Tool choice | `stated_intent_is_proposed_before_it_is_recorded` | Opened a questionnaire instead of listing the bound target roles. |
| Tool choice | `a_questionnaire_answer_is_proposed_with_user_input_provenance` | Read memory detail instead of proposing the answer with its user quote. |
| Unconfirmed notes | `working_notes_never_choose_or_rank_a_job` | Returned a final answer instead of asking for confirmation. |
| Report identity | `a_report_that_scrolled_out_of_the_catalogue_is_not_faked` | Quality property held in 0/5 samples, below its 60% floor. |
| Report identity | `a_report_made_this_turn_without_an_index_cannot_be_named` | Called `get_job_research` without a grounded selector in one sample. |
| Delivery | `a_card_backed_report_is_answered_without_reproducing_it` | Repeated content already delivered by the card. |
| Calendar recovery | `an_uncertain_calendar_write_is_not_reissued_or_claimed` | Requested another calendar preview after an uncertain execution. |
| History retrieval | `a_compacted_fact_is_paged_in_rather_than_guessed` | Asked the user instead of reading the omitted conversation span. |
| History retrieval | `a_long_compacted_history_is_searched_in_one_page_in_call` | Answered without the focused span search. |
| Delivery | `an_empty_conversation_span_is_not_filled_from_the_window` | Repeated runtime-owned delivery text. |

The report-identity scenario also fails the sole quality floor. Runtime tests for uncertain calendar execution, wrong-company report reads, unconfirmed notes, empty spans, and repeated calls pass (5/5); the model-level failures remain visible in the cassettes.

## Changes made while establishing the baseline

- Cassette writing now specifies UTF-8, matching its reader.
- `find_saved_jobs` accepts an omitted or empty query to list recent saved jobs, as its repository already supported. A contract test pins this behaviour.
- Recording failures now identify the rejected tool and safe rejection category, without logging argument values. This revealed invalid proposals rather than an evaluation transport failure.

The prior GPT recordings are archived in `../main_agent.before-rerecord.2026-10-04.zip`; the first Qwen trial is archived in `../main_agent.qwen3.7-plus-trial.2026-10-04.zip`.
