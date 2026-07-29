# Changelog

> [中文](CHANGELOG.md) | English

All notable changes to this project.

## [Unreleased]

### Added

- **Resume parsing module** (`src/career_agent/resume.py`)
  - PDF text extraction via PyMuPDF + plain text support
  - LLM-based structured parsing using `with_structured_output(ResumeData)`
  - `ResumeData` Pydantic model with categorized experience:
    - `WorkExperience` — work/internship
    - `ProjectExperience` — academic/personal/open-source projects
    - `Publication` — papers, articles, conference publications
    - `Award` — competitions, awards, certifications
  - Work and project details preserved as individual `highlights`
  - Lightweight `list[str]` fields for `skills` and `tech_stack`
  - Explicit resume objectives stored in `stated_target_*` fields, separate from inferred role recommendations
  - `to_facts()` retains all extracted skills, technologies, education, and experience details
  - CLI resume labels come from the user or filename rather than the parsing model

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

- **Resume fact boundaries**
  - Removed `ResumeLabel` from the parsing schema
  - Stopped inferring target roles, cities, salary, and summary during factual extraction
  - Allowed every field to remain empty instead of pressuring the model to satisfy required fields
  - Replaced `description` with `highlights` and removed unused publication-author and award year/rank fields

- **PDF parser**
  - Switched from `pypdf` to PyMuPDF (`fitz`) for better readable text and layout extraction on Chinese resumes
  - A real Chinese technical-resume regression retained 2,324 characters plus education, work, and project sections
  - Documented that scanned PDFs still need a future OCR fallback

- **Prompt engineering** — Rewrote system prompt with:
  - Anti-hallucination grounding rules (no fabricated job listings)
  - Chinese output formatting (table layout for job listings)
  - Safety boundaries (no auto-applying, no decision-making for user)

- **MCP integration** — Switched from `mcp-jobs` to `boss-agent-cli` (50 tools)

### Dependencies

- Added `pymupdf` for Chinese PDF resume text extraction
- Removed unused `pypdf`

### Known Limitations

- `skills` may still absorb capability phrases from project prose and should be restricted to explicit skill sections
- The current schema does not retain messaging contact IDs or education GPA
- `tech_stack` also contains models, APIs, datasets, and benchmarks; its final name needs validation on more resumes

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
