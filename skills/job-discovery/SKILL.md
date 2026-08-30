---
name: job-discovery
description: Use when the user explicitly asks to find new jobs or open a recruitment-platform search page.
---

# Job Discovery

## Purpose

Open a recruitment search page for the user's requested role and city without automating the platform. The Main Agent invokes `open_job_search`; the user browses normally and explicitly saves only jobs they care about. The Main Agent does not receive search results or JD content merely because it opened the page.

## When to use

Use when the user explicitly asks to:

- find new jobs;
- open BOSS with a specific keyword and city.

Use `find_saved_jobs` and `get_saved_job`, not this skill, when the user wants to recall a previously saved job. Do not use it for resume rewriting, interview preparation, or general career advice.

## Required context

Use the explicit keyword and city from the current request. If city is omitted, a confirmed profile default may be used. Do not add salary, experience, education, company, or other filters the user did not request. Opening a page creates no workflow cursor.

## Workflow interface

The model-visible new-job tool is `open_job_search`.

- The model supplies `keyword`, optional `city`, and `platform="boss"`.
- Runtime may fill a missing city from the confirmed profile default.
- The tool returns a bounded HTTPS client action for an allowlisted BOSS search URL.
- The browser client attempts to open the URL and always renders a clickable fallback.
- The tool never reads page content, results, cookies, or login state.

## Workflow state handling

1. Extract the requested role keyword and optional city.
2. Invoke `open_job_search` once.
3. Tell the user that the page was opened and that browsing and saving remain under their control.
4. End the turn. Do not wait for search results or claim that jobs were found.

Read [tool contracts](references/tool-contracts.md) before exposing this action.

## User decisions and confirmation

- Treat “search jobs” as authorization only to open the search page.
- Saving a JD must remain an explicit browser-side user action.
- Searching never authorizes applying, greeting, contacting, or messaging a recruiter.

## Safety boundaries

- Construct URLs only through the allowlisted tool; never invent or follow a model-provided BOSS internal API URL.
- Never automate scrolling, opening details, reading results, applying, or messaging.
- Never expose or collect credentials, cookies, tokens, browser storage, or raw BOSS envelopes.
- Never silently promote model/external inferences into CareerProfile or AgentPreferences.
- Never claim a job or JD was captured until a separate user-confirmed save succeeds.

## Output

Return `job_search_page_ready` with the requested platform, keyword, city, and an HTTPS `open_url` client action. The public client action is delivered outside the model-visible observation:

```text
job_search_page_ready → browser attempts to open; user browses and saves explicitly
failed                → provide a clickable or manual-search fallback
```
