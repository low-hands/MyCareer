# Career Agent

> [中文](README.md) | English

An intelligent job-seeking assistant powered by LLM + MCP multi-agent architecture. Search jobs, analyze JDs, and prepare for interviews efficiently.

## Features

- **MCP Tool Integration** — 50+ tools via [boss-agent-cli](https://github.com/can4hou6joeng4/boss-agent-cli) (search, details, AI analysis, etc.)
- **Agent Loop Runtime** — Custom observe-decide-act loop with multi-step tool calling
- **Anti-Hallucination Prompt** — Strict grounding rules, no fabricated job listings
- **Permission Control** — Tools classified by READ / WRITE / EXTERNAL levels
- **Trace Visualization** — `--trace` flag shows tool call trajectory per turn
- **Auto Schema Fix** — Handles non-standard MCP tool schemas (e.g. `int` → `integer`)
- **Pluggable Architecture** — Runtime / Context / TraceSink are all Protocols

## Architecture

```
User Request
  ↓
AgentLoopRuntime (observe → decide → act loop)
  ↓
ToolRegistry (permission filter + timeout)
  ↓
MCP Server (boss-agent-cli, 50+ tools)
  ↓
Formatted Output (tables / analysis / suggestions)
```

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Boss Zhipin account (for MCP tool authentication)

### Install

```bash
git clone git@github.com:low-hands/MyCareer.git
cd MyCareer
uv sync
```

### Configure

Create a `.env` file (or export environment variables):

```bash
ORCHESTRATOR_MODEL=gpt-4o
ORCHESTRATOR_API_KEY=sk-your-key-here
ORCHESTRATOR_BASE_URL=https://api.openai.com/v1
```

### Boss Zhipin Login

MCP tools require a one-time Boss Zhipin login:

```bash
uv run boss login
uv run boss status   # verify login status
```

### Run

```bash
uv run career-agent            # start with MCP tools
uv run career-agent --trace    # start with tool call tracing
uv run career-agent --no-mcp   # bare chat mode (no MCP)
```

### Test

```bash
uv run pytest
uv run ruff check src/
```

## Project Structure

```
career-agent/
├── src/career_agent/
│   ├── cli.py                          # CLI entry point
│   ├── contracts.py                    # AgentRequest / AgentResult / ToolPermission
│   ├── prompts.py                      # Prompt templates (strategy layer, tools injected dynamically)
│   ├── context.py                      # ContextManager (facts injection)
│   ├── tools.py                        # ToolRegistry (permissions + timeouts)
│   ├── tracing.py                      # TraceSink Protocol + InMemoryTraceSink
│   ├── models.py                       # Model client factory
│   ├── mcp_client.py                   # MCP client wrapper
│   ├── state.py                        # Conversation state persistence
│   ├── evaluation.py                   # Evaluation framework
│   └── runtime/
│       ├── base.py                     # AgentRuntime Protocol
│       ├── agent_loop_runtime.py       # Default Agent Loop
│       └── langgraph_runtime.py        # LangGraph adapter
└── tests/
```

## Design Principles

1. **Runtime-First** — Domain code depends on `AgentRuntime` Protocol, not specific frameworks
2. **Permission Declaration** — Tools must declare permission level before exposure to the model
3. **Layered Prompts** — Static strategy layer + dynamic injection layer (tool schemas / state / facts)
4. **Traceable** — Every run has stable thread_id and trace_id
5. **Safety Boundary** — Model output is never authority; external operations require explicit approval

## Roadmap

- [x] Single Agent + MCP tool integration
- [x] Prompt engineering (anti-hallucination + Chinese output)
- [x] Schema compatibility + Trace visualization
- [ ] LangGraph hybrid architecture (outer orchestration + inner Agent Loop)
- [x] Resume parsing (PDF/text + structured fact extraction)
- [ ] Job matching
- [ ] Multiple Specialist nodes (JD analysis, interview prep, etc.)
- [ ] Evaluation system + Golden Cases

## License

MIT
