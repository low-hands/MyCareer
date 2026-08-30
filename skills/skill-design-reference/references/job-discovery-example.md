# Job Search Navigation Skill Example

```markdown
---
name: job-discovery
description: Use when the user explicitly asks to open a recruitment-platform job search.
---

# Job Discovery

## When to use

Use only when the user explicitly asks to search for new jobs. Use saved-job tools for jobs the user previously saved.

## Required context

- task-relevant CareerProfile projection;
- explicit keyword and optional city from the current request.

## Workflow

1. Build `keyword` and optional `city` from explicit user input and confirmed defaults.
2. Call `open_job_search` once.
3. Deliver the typed `open_url` client action.
4. Tell the user that browsing and saving remain under their control, then end the turn.
5. A browser extension may preview the current detail page, but persists it only after the user clicks save.

## Tool contracts

- `open_job_search(keyword, city?)`;
- `find_saved_jobs(query)` for historical recall only;
- `get_saved_job(selection_index)` after a saved-job search.

## Rules

- Never automate scrolling, opening details, result extraction, applying, or messaging.
- Never call recruitment-platform internal APIs or read credentials, cookies, tokens, or browser storage.
- Never claim results were found merely because a search page opened.
- Never save a JD without an explicit browser-side user action.

## Completion

Complete when the search page action has been delivered or a safe navigation failure has been reported.
```
