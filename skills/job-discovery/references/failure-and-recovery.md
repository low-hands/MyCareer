# Job Discovery Failure and Recovery

Read this reference only when a Gateway result is not `selection_required` or `analysis_ready`.

## `waiting_user`

BOSS needs a user action such as restoring login or resolving a provider-side condition.

- Explain the Gateway `recovery_action`.
- Do not retry in a loop.
- Resume only after the user confirms the required action is complete.

## `detail_unavailable`

The selected job detail could not be retrieved within the Gateway's bounded policy. The candidate run remains valid.

- Do not call `select` again for the same failed selection.
- If `fallback_url` exists, show it only as a user-facing link; do not open or fetch it automatically.
- If no URL exists, show `manual_search_query` so the user can locate the job manually in BOSS.
- Ask the user to paste the JD or safely provide it through a file/stdin path.
- Then call `job_discovery.analyze_provided_jd`; it does not call BOSS again.

## `failed`

Return the safe `error_code`, `error_stage`, and `error_detail` in a concise explanation. Do not expose raw provider payloads or internal trace unless the user explicitly requests diagnostics.

## Detail retry boundary

The Gateway owns BOSS detail retries. Transient detail failures have a maximum of three total attempts. Authentication, account-risk, rate-limit, malformed-response, and policy errors stop earlier. The Skill must not add a fourth attempt.

## Start a new search

If the user changes target role or search criteria, create a new `research` run. Do not reuse a previous run ID for different search intent.
