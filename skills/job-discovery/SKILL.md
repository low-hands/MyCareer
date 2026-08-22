---
name: job-discovery
description: Use when the user explicitly asks to search for jobs, continue a selected job, or provide a JD after BOSS job-detail retrieval fails.
---

# Job Discovery

## Purpose

Find jobs through the read-only Job Discovery workflow, let the user select one candidate, and return a structured JD analysis. The main Agent invokes one `job_discovery` workflow tool; the Gateway and LangGraph own phase transitions.

## When to use

Use when the user explicitly asks to:

- search for jobs;
- inspect a candidate from an existing Job Discovery run;
- analyze JD text after detail retrieval was unavailable.

Do not use for resume rewriting, interview preparation, or general career advice unless the user also asks to search or inspect a specific job.

## Required context

Build a one-time `JobDiscoveryRequest` from:

- `user_id`;
- confirmed target role;
- the resume associated with that role, when relevant;
- confirmed defaults or this-turn overrides for city, salary, experience, and education.

This-turn overrides take priority for the current request. Do not silently write them to long-term memory. If a required search field is missing, ask the user; do not invent it.

Keep only this compact cursor in conversation state:

```json
{
  "job_discovery_run_id": "job_discovery_...",
  "phase": "selection_required",
  "selected_result_ref": null
}
```

Do not put full JD text, BOSS responses, credentials, raw trace events, or unrelated candidate data in conversation state.

## Workflow interface

The only model-visible tool is `job_discovery`. The main Agent decides whether to enter the workflow; it must not choose or name internal Gateway actions.

- The model may provide an explicit `target_role` override for a new search.
- During `selection_required`, the model may provide a 1-based candidate index from the compact conversation cursor.
- Runtime injects the trusted user message, user identity, conversation identity, and task cursor.
- The Gateway validates the run and advances the LangGraph workflow; it owns search, candidate mapping, detail retrieval, retry limits, and fallback analysis.

## Workflow state handling

1. Confirm the user has authorized an external BOSS search, then invoke `job_discovery`.
2. `selection_required`: show at most 15 candidate summaries and wait for the user. On the next relevant turn, invoke the same tool with a candidate index.
3. `analysis_ready`: return a concise JD analysis and update the conversation cursor.
4. `detail_unavailable`: show `manual_search_query` and ask the user to paste the JD text. On the next turn, invoke the same tool; the Gateway uses the trusted user message and must not call BOSS again.
5. `waiting_user`: explain provider recovery. The Gateway alone decides whether a resume action is valid.
6. End when the user receives analysis, declines to continue, or receives an actionable failure.

Read [tool contracts](references/tool-contracts.md) before exposing this workflow. Read [failure and recovery](references/failure-and-recovery.md) only for non-success states.

## User decisions and confirmation

- Wait for a user-selected `result_ref` after `selection_required`.
- Ask before any separate waitlist, recruiter-contact, or application action.
- Treat “search jobs” as authorization only for the read-only BOSS search/detail workflow; it never authorizes applying, greeting, contacting, or saving a job.

## Safety boundaries

- Never invent `result_ref`, BOSS security IDs, job IDs, or URLs.
- Never bypass BOSS controls or retry outside the Gateway policy.
- Never expose credentials, cookies, raw BOSS envelopes, hidden prompts, or full trace by default.
- Never silently promote model/external inferences into CareerProfile or AgentPreferences.
- Return only compact candidate summaries and analysis summaries to the main Agent.

## Output

Return the Gateway state, `run_id`, candidate summaries or selected analysis, and `next_action`. Preserve the user-facing state exactly:

```text
selection_required → wait for selection
analysis_ready     → present analysis
waiting_user       → explain required provider recovery
 detail_unavailable → request user-provided JD
failed             → explain the actionable error
```
