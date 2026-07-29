# Career Agent

An intentionally small agent foundation for a career assistant product. It owns
its runtime contracts and keeps framework-specific code behind adapters.

## Architecture

```text
Application / domain agents
        |
        v
Career Agent contracts + policies + tracing
        |
        +-- AgentLoopRuntime (default, owned function-calling loop)
        |
        +-- LangGraphRuntime (comparison adapter)
        |
        `-- DeepAgentsRuntime (planned adapter)
```

The first vertical slice provides:

- provider-independent run contracts;
- a permission-aware tool registry;
- a replaceable context manager and isolated prompt catalog;
- run tracing;
- an owned function-calling loop and conversation state;
- a LangGraph comparison adapter with thread checkpoints;
- an evaluation runner;
- a CLI for smoke testing an OpenAI-compatible model.

Deep Agents is deliberately optional. It can be evaluated later without making
domain code depend directly on it.

## Setup

```bash
uv sync
```

Configure an OpenAI-compatible endpoint:

```bash
export CAREER_AGENT_API_KEY="..."
export CAREER_AGENT_BASE_URL="https://your-provider.example/v1"
export CAREER_AGENT_MODEL="your-model-id"
```

Run:

```bash
uv run career-agent
uv run pytest
```

## Design rules

1. Domain code depends on `AgentRuntime`, not LangGraph or Deep Agents.
2. Tools declare permissions before they can be exposed to a model.
3. Every run has a stable thread ID and trace ID.
4. New framework adapters must pass the same contract and evaluation suites.
5. External writes require explicit policy approval; model output is never authority.

## Workflow versus Agent Loop

A resume upload is a deterministic ingestion workflow:

```text
upload -> validate -> parse -> normalize -> persist -> index
```

It should not consume an Agent Loop. The loop starts when there is an open goal
whose next action depends on observations:

```text
goal/query
  -> model decides which allowed tool to call
  -> runtime validates and executes the function
  -> tool result becomes an observation
  -> model decides again
  -> final answer, clarification, or approval request
```

For example, “Use my latest backend resume to find suitable jobs and explain
the top three” may require `get_resume`, `search_jobs`, and several
`compare_resume_to_job` calls. The sequence and stopping point are not fixed in
advance, so it belongs in the Agent Loop. A scheduled event may generate a
goal, but a loop still needs a goal even when no user typed a query.

## File map

```text
career-agent/
├── pyproject.toml
├── README.md
├── src/career_agent/
│   ├── __init__.py
│   ├── contracts.py
│   ├── prompts.py
│   ├── context.py
│   ├── state.py
│   ├── tools.py
│   ├── tracing.py
│   ├── models.py
│   ├── evaluation.py
│   ├── cli.py
│   └── runtime/
│       ├── __init__.py
│       ├── base.py
│       ├── agent_loop_runtime.py
│       └── langgraph_runtime.py
└── tests/
```

- `contracts.py` defines framework-neutral request, result, context, and
  permission schemas. Application code should exchange these types.
- `prompts.py` is the only initial home for model-facing instructions and
  prompt rendering. Runtime code must not own prompt text.
- `context.py` decides which facts enter a run and asks `prompts.py` to render
  them. Later retrieval and memory policies belong here.
- `state.py` owns framework-neutral conversation persistence contracts.
- `tools.py` registers tools and filters them by side-effect permission before
  exposing them to a model; it also enforces timeouts during execution.
- `tracing.py` defines run/event telemetry independently of a vendor. A
  Langfuse or OpenTelemetry sink can replace the in-memory implementation.
- `models.py` creates model clients. Provider-specific compatibility belongs in
  separate adapters here, not in agent or domain code.
- `evaluation.py` runs framework-neutral cases against any `AgentRuntime`.
- `cli.py` is a thin local entry point and contains no agent logic.
- `runtime/base.py` defines the stable `AgentRuntime` protocol.
- `runtime/agent_loop_runtime.py` is the default observe-decide-act loop and
  directly executes model function calls.
- `runtime/langgraph_runtime.py` is retained as a comparison adapter for
  graph-shaped orchestration and checkpoint experiments.
- `tests/` protects contracts, policies, prompt rendering, state continuation,
  and evaluation behavior without requiring a live API.
