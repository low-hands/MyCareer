# Changelog

> [中文](CHANGELOG.md) | English

All notable changes to this project.

## [Unreleased]

### Added

- **Resume parsing module** (`src/career_agent/resume.py`)
  - PDF text extraction via `pypdf` + plain text support
  - LLM-based structured parsing using `with_structured_output(ResumeData)`
  - `ResumeData` Pydantic model with categorized experience:
    - `WorkExperience` — work/internship
    - `ProjectExperience` — academic/personal/open-source projects
    - `Publication` — papers, articles, conference publications
    - `Award` — competitions, awards, certifications
  - Separate `skills` (soft skills) and `tech_stack` (languages/frameworks/tools)
  - `ResumeLabel` enum with 16 CS-related directions (backend, ml, llm, cv, agent, etc.)
  - `to_facts()` generates structured facts (company:role, project:summary, paper:venue)
  - Multi-resume support with label-based management

- **Resume CLI integration**
  - `--resume <path>` flag for loading resume at startup
  - `/resume [label] <path>` command for loading during conversation
  - `/resume list` to view all loaded resumes
  - `/resume use <label>` to switch active resume
  - Resume facts auto-injected into system prompt via `metadata["facts"]`

- **Trace visualization** (`--trace` flag)
  - Shows tool call trajectory per turn (tool name + step number)

- **Schema auto-fix** (`_sanitize_tools`)
  - Recursively fixes non-standard JSON Schema types from MCP servers
  - Handles `int` → `integer`, `float` → `number`, `bool` → `boolean`

### Changed

- **Prompt engineering** — Rewrote system prompt with:
  - Anti-hallucination grounding rules (no fabricated job listings)
  - Chinese output formatting (table layout for job listings)
  - Safety boundaries (no auto-applying, no decision-making for user)

- **MCP integration** — Switched from `mcp-jobs` to `boss-agent-cli` (50 tools)

### Dependencies

- Added `pypdf` for PDF resume parsing

## [0.1.0] — 2025-07-29

### Initial Release

- `AgentLoopRuntime` — custom observe-decide-act function-calling loop
- `LangGraphRuntime` — LangGraph-based comparison adapter with checkpoints
- `ToolRegistry` — permission-aware tool registration (READ/WRITE/EXTERNAL)
- `TraceSink` Protocol + `InMemoryTraceSink` — pluggable run tracing
- `ContextManager` Protocol + `BasicContextManager` — system prompt + facts injection
- `ConversationStore` Protocol + `InMemoryConversationStore` — thread-based state
- MCP client integration via `langchain-mcp-adapters`
- CLI entry point with `--no-mcp` bare chat mode
- Role-based environment variables (`ORCHESTRATOR_*`)
- Evaluation framework skeleton
- Full test suite (ruff + pytest)
