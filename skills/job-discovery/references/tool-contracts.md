# Job Search Navigation Tool Contract

The production Main Agent exposes one model-visible tool for finding new jobs:

```text
open_job_search
```

It prepares a client-side browser navigation. It is an atomic tool, not a
stateful Job Discovery workflow, and it never reads BOSS results or JD content.

## Model-visible arguments

```json
{
  "platform": "boss",
  "keyword": "AI 产品经理",
  "city": "上海"
}
```

- `platform` currently accepts only `boss`.
- `keyword` is required and must describe the role the user requested.
- `city` is optional. Runtime may use the user's confirmed default city.
- Internal identifiers, raw URLs, cookies, tokens, JD text, and BOSS request
  parameters are not accepted.

## Result

```text
state       = job_search_page_ready
next_action = browse_and_save_job
```

The complete tool payload contains a bounded client action:

```json
{
  "client_action": {
    "type": "open_url",
    "url": "https://www.zhipin.com/web/geek/job?...",
    "label": "在 BOSS 搜索 AI 产品经理"
  }
}
```

The decision model sees only the safe observation state and next action. The
runtime emits the URL as a typed SSE `client_action`; the React client attempts
to open it and retains a clickable fallback in case the browser blocks the
popup.

## Boundary

Opening the page does not mean jobs were found, viewed, captured, or saved. The
user operates BOSS normally. A separate browser-side, explicit save action
imports a selected JD into `JobPosting + JDSnapshot`. There is no legacy online
discovery workflow or BOSS connector behind this tool.
