# Job Discovery Skill Example

Use this example as a reference when adding `skills/job-discovery/` after the main Agent tool registry exists.

```markdown
---
name: job-discovery
description: Use when the user explicitly asks to search for jobs, inspect a selected job, or analyze a JD they provide after job-detail retrieval fails.
---

# Job Discovery

## When to use

Use only when the user explicitly authorizes a job search or asks to continue an existing Job Discovery run.

## Required context

- `user_id`
- task-relevant CareerProfile projection
- current conversation cursor, if one exists
- temporary search overrides from the latest user message

## Workflow

1. Build a `JobDiscoveryRequest` from confirmed CareerProfile defaults plus this-turn overrides.
2. Call `job_discovery.research`.
3. Return at most 15 summaries and save only `run_id`, state, and candidate references in the conversation cursor.
4. Wait for the user to choose a result.
5. Call `job_discovery.select(run_id, result_ref)`.
6. If state is `analysis_ready`, return the compact JD analysis and offer the next explicitly requested capability.
7. If state is `detail_unavailable`, show `manual_search_query` and request pasted JD text.
8. Call `job_discovery.analyze_provided_jd` only after the user provides text.

## Tool contracts

- `job_discovery.research(request)`
- `job_discovery.select(run_id, result_ref)`
- `job_discovery.analyze_provided_jd(run_id, result_ref, jd_text)`

## Rules

- Never search BOSS without explicit user authorization.
- Never invent a `result_ref` or BOSS identifier.
- Never apply, contact a recruiter, or add a job to a waitlist without a separate confirmation.
- Do not retry BOSS detail beyond the Gateway policy.
- Do not place raw JD text, provider payloads, credentials, or traces in conversation state.

## Completion

Complete when the user has either received a JD analysis, declined to continue, or the workflow returned an actionable failure.
```
