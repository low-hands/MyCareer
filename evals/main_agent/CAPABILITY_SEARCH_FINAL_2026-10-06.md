# Capability lexical search finalization — 2026-10-06

The query path ranks catalogue matches by relevance. It contains no phrase-based
intent lists or separate thresholds for writes and reads. A non-exact query must
match at least two uncommon indexed terms across the name, aliases, summary,
parameters, and example queries. Exact tool names, namespaces, and aliases
remain directly searchable. `names` expands any searchable capability, including
writes; execution authorization remains a separate boundary.

The development set fixed this rule and its weights. The original and
independently authored holdouts were each evaluated **once** after the rule was
frozen. No search rule was changed after those evaluations.

| Split | Positive first-step demands | Top 1 | Top 3 | Top 5 |
| --- | ---: | ---: | ---: | ---: |
| Development | 43 | 17 (39.5%) | 28 (65.1%) | 34 (79.1%) |
| Original holdout | 18 | 9 (50.0%) | 12 (66.7%) | 13 (72.2%) |
| Independent holdout | 12 | 3 (25.0%) | 7 (58.3%) | 10 (83.3%) |

The score uses each case's original user message as the query and accepts any
listed first-step business tool as a hit. It measures catalogue search, not the
Agent's actual tool choice or execution. Semantic search was not included.

Original holdout top-5 misses: `selection_holdout_cross_interview_then_links`
(`get_interview`), `selection_holdout_chain_intent_confirmation`
(`propose_job_intent`), `selection_holdout_chain_memory_amendment`
(`propose_memory_amendment`), `selection_holdout_control_no_calendar`
(`get_interview`), and `selection_holdout_control_waiting_for_confirmation`
(`propose_memory_amendment`). Independent holdout misses:
`independent_holdout_saved_job_requirements` (`get_saved_job`) and
`independent_holdout_email_to_interview` (`sync_application_emails`).

All **141/141** Chinese aliases retrieve their tool in the top five. `天气怎么样`,
`的`, `这个怎么弄`, `帮我看一下`, and an unknown English word return no result.
`今天吃什么` retrieves only `get_daily_brief`; it is recorded as an ambiguous
colloquial query, not counted as an empty-result regression.

Read-intent probes are observational only. Top-five action-capability results
(WRITE or `propose_*`) are: `帮我看看我的经历` →
`propose_memory_amendment`; `看看我的面试安排` → `prepare_interview`,
`complete_interview`; `我的投递记录有哪些` → `create_interview`,
`update_application_status`; `这个岗位不错` → `analyze_job`,
`match_resume_to_job`, `correct_job_requirement_tier`. These illustrate why
search relevance must not authorize execution. The catalogue entries are still
subject to reachability, argument validation, authorization, owner rules,
approval, and journal checks at the execution boundary.

The development control cases received eight WRITE results in total. This is
an exposure metric, not an execution count. Registered schemas, legacy-visible
schemas, prompt fingerprints, and cassette contents are unchanged by the
retrieval update.
