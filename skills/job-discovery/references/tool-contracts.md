# Job Discovery Workflow Tool Contract

The Main Agent exposes one model-visible tool:

```text
job_discovery
```

It advances a stateful Job Discovery workflow. The model must not call BOSS, LangGraph nodes, or internal Gateway methods.

## Model-visible arguments

```json
{
  "target_role": "optional target-role override for a new search",
  "selection_index": 1
}
```

- `target_role` is optional. When omitted, Runtime uses the only confirmed target role from the Profile; otherwise the Gateway asks the user to choose one.
- `selection_index` is a 1-based index and is meaningful only when state is `selection_required`.

The model must not provide:

```text
user_id
conversation_id
run_id
result_ref
security_id
job_id
jd_text
```

Runtime injects the user identity, session, task cursor, and current user message. Gateway maps a valid selection index to an opaque result reference and validates ownership and current workflow phase.

## Gateway state results

```text
selection_required
→ return at most 15 candidate summaries; wait for a user selection

analysis_ready
→ return selected job summary and compact JD analysis

detail_unavailable
→ return manual_search_query; the next user JD text continues the same workflow without BOSS detail retrieval

waiting_user
→ return provider recovery instructions

failed
→ return safe error information
```

## Internal implementation

`JobDiscoveryGateway.advance()` is the Main Agent entry point. It may invoke existing internal Gateway methods (`research`, `select`, or `analyze_provided_jd`) only after validating the current run state. Those methods remain available to the CLI but are not model-visible tools.
