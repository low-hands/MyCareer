# Tool-selection preparation — 2026-10-05

## Existing comparator

Keep [QWEN_BASELINE_2026-10-04.md](QWEN_BASELINE_2026-10-04.md) and its Qwen cassettes as the pre-selection comparator. The recorded cross-domain job-analysis → resume-match sequence passed 0/3 samples; the resume → job-research switch passed 2/3. Do not re-record these cassettes merely to make a candidate selector pass. Compare later candidates on task completion, missing required tools, model requests, and decision latency; preserve separate execution-safety assertions.

These two cassettes were re-recorded on 2026-10-05 after prompt and interaction-schema changes, not to favour a selector. Both now pass 0/3; the 2026-10-04 figures above remain the historical comparator, and the current recordings are the comparator for the next selector.

The model-free offer baseline and its explicit failure attribution are in [TOOL_SELECTION_OFFLINE_BASELINE_2026-10-05.md](TOOL_SELECTION_OFFLINE_BASELINE_2026-10-05.md).

The interaction-exit experiments (`tool_choice="required"` and text-classification retry), and the two selection rules they motivate, are in [INTERACTION_EXIT_MEASUREMENT_2026-10-05.md](INTERACTION_EXIT_MEASUREMENT_2026-10-05.md).

## Audit of the old `core` profile

`core` currently contains 17 model-callable tools. It is a shared **profile**, not an always-offered set:

| Tools | Why they are not automatically permanent |
| --- | --- |
| `route_to_capability` | Compatibility route for the current profile mechanism. It should disappear only after a replacement discovery/recovery path is tested. |
| `load_skill`, `read_conversation_span`, `fetch_archived_constraints`, `search_career_memory`, `update_working_notes` | Useful for particular context or memory conditions; they need not occupy every model call. |
| `get_daily_brief`, `list_action_items`, `find_saved_jobs`, `list_resumes`, `list_applications`, `list_interviews` | Business discovery reads; relevant to different user objectives, not every objective. |
| `update_owner_settings`, `complete_action_item`, `dismiss_action_item`, `snooze_action_item`, `open_job_search` | Writes or state changes; they require explicit relevance and remain subject to execution authorization. |

No existing `core` tool qualifies as a permanent tool for the *new* selection path yet. Accordingly, `ALWAYS_OFFERED_TOOLS` is empty. This does **not** change the live profile behavior. A future `search_capabilities` could become permanent if a measured recovery need warrants it; the legacy route remains available in the current runtime during migration.

## Contract before wiring a selector

`prepare_capability_selection` takes capability names from any domains, the current task, and the registered schemas. It produces one frozen selection record with selected names, actually offered names/schemas, and unmet prerequisites. Schema order follows the catalog, regardless of the selector's output order. Unknown, runtime-owned, or unregistered names fail closed. An unmet precondition prevents that schema being offered and records the requirement.

When wired into `decide`, the **same snapshot** must drive provider schemas, `available_now`, and the stored `offered_tool_names` used by the execution-side visibility check. Selection is relevance plus current prerequisites; authorization, approval, and idempotency remain checks at tool execution. The current profile-based runtime does not use this contract yet, so the existing Qwen prompt/schema fingerprints remain unchanged.
