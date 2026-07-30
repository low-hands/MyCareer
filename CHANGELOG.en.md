# Changelog

> [中文](CHANGELOG.md) | English

All notable changes to this project.

## [Unreleased]

### Added

- **Source-grounded job posting storage** (`src/career_agent/jobs/`)
  - `JobPosting` stores traceable summary or full-JD facts without mixing in match results
  - SQLite repository scopes every read and update by `tenant_id`
  - Deduplicates by source posting ID, normalized URL, then content fingerprint
  - Summary refreshes preserve previously stored full JD and other non-empty details
  - `SearchCriteria` preserves original user intent and `SearchRun` stores immutable displayed-result snapshots
  - SearchRun tables keep only ordered `job_id` references rather than duplicating full JD content
  - Resolves follow-up references by position or company/title/location/salary with resolved, ambiguous, and not-found outcomes
  - Boss MCP provider uses read-only inline `boss_export` for rich filtered retrieval and `boss_detail` for full JD content
  - Normalizes upstream JSON envelopes into `JobPosting` while retaining a non-secret detail locator
  - Builds tenant/thread-scoped search and detail tools so the LLM supplies only `SearchCriteria`/`SearchResultSelector`
  - Wires the search application path into the CLI without exposing raw Boss MCP tools to the LLM
  - Reuses conversation state across turns while rebuilding tools with trusted tenant/thread/user-query context
  - Adds a terminal response guard inside the Agent loop, requiring successful tool evidence for explicit job search/detail requests
  - Ungrounded answers receive bounded retry feedback and fail closed; tool errors do not count as evidence

- **Resume domain models and parsing modules** (`src/career_agent/resumes/models.py`, `src/career_agent/resumes/parser.py`)
  - PDF text extraction via PyMuPDF + plain text support
  - LLM-based structured parsing using `with_structured_output(ResumeData)`
  - `ResumeData` Pydantic model with categorized experience:
    - `WorkExperience` — work/internship
    - `ProjectExperience` — academic/personal/open-source projects
    - `Publication` — papers, articles, conference publications
    - `Award` — competitions, awards, certifications
  - Work and project details preserved as individual `highlights`
  - Lightweight `list[str]` fields for `skills` and `technologies`
  - Explicit resume objectives stored in `stated_target_*` fields, separate from inferred role recommendations
  - `to_facts()` retains all extracted skills, technologies, education, and experience details
  - Role-pool metadata is separated from parsed facts rather than inferred by the parsing model

- **Role-oriented resume pools** (`src/career_agent/resumes/`)
  - Groups versions by a free-form `role_type`
  - Allocates version numbers independently per pool and keeps immutable file/parse snapshots
  - Uses one SQLite database with mandatory `tenant_id` scoping
  - Prevents duplicate files within a tenant's role pool via SHA-256
  - Leaves no database or file record when parsing fails
  - Persists the active version across process restarts

- **Resume CLI integration**
  - `--resume <path> --resume-role <role>` adds a startup resume to a pool
  - `/resume add <role> <path> [note]` adds a version
  - `/resume list [role]` lists versions by role pool
  - `/resume show <version-id>` shows a parsed version
  - `/resume use <version-id>` selects the active version
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
  - Restricted `skills` to explicit skill sections instead of deriving them from project prose
  - Restricted `technologies` to languages, frameworks, libraries, models, development/deployment tools, and software platforms
  - Added factual extraction for messaging contact IDs and education GPA

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
