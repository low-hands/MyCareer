---
name: skill-design-reference
description: Use when creating or reviewing a Skill package for this Career Agent, including its SKILL.md, progressive disclosure, tool workflow, MCP coordination, subagent boundaries, validation, and iteration rules.
---

# Skill Design Reference

## Purpose

Use this reference when designing a new Skill. A Skill is a reusable workflow package for the main Agent. It explains when the capability applies, which tools it may use, how steps depend on one another, where user confirmation is required, and what structured result returns to the main Agent.

A Skill is not a database, not a replacement for a deterministic service, and not a replacement for a stateful workflow engine.

## Required package shape

Every Skill package must contain a root `SKILL.md`:

```text
skills/<skill-name>/
├── SKILL.md                 # required entry point
├── scripts/                 # optional deterministic executable helpers
├── references/              # optional detailed documentation
└── assets/                  # optional templates and static resources
```

Keep the root package small. Load detailed material from `references/` only when the workflow needs it.

## SKILL.md frontmatter

The frontmatter must contain:

```yaml
---
name: skill-name
description: One sentence explaining what the Skill does and when to use it.
---
```

Rules:

- Use lowercase kebab-case for `name`.
- Make `description` specific enough for the Agent to decide whether to load the Skill.
- Mention the user request or capability that triggers the Skill.
- Do not put implementation details, credentials, or long workflow text in frontmatter.

## Recommended body sections

Use only sections that improve execution. Complex Skills normally include:

```markdown
# Purpose
# When to use
# When not to use
# Required context
# Workflow
# Tool usage
# User confirmation points
# Failure and recovery
# Output contract
# Security boundaries
# Examples
```

The body is progressively disclosed: the Agent reads the frontmatter first, loads the Skill when relevant, and reads referenced documents only when required.

## Workflow design

Describe steps in execution order. For every step specify:

- input and output;
- dependency on earlier steps;
- how success is verified;
- what happens on failure;
- whether the user must make a decision;
- whether the step can be retried safely.

Example:

```markdown
1. Read the task-relevant CareerProfile projection.
2. Build and validate the request with the request schema.
3. Call `job_discovery.research`.
4. Display at most 15 candidate summaries and wait for a user selection.
5. Call `job_discovery.select` with the existing `run_id` and `result_ref`.
6. If detail is unavailable, show the manual search query and request pasted JD text.
7. Call `job_discovery.analyze_provided_jd` without requesting BOSS again.
```

Do not describe a workflow without stating its stop conditions. A Skill must say when it is complete, when it waits for the user, and when it returns a failure.

## Tools and MCP coordination

A Tool is one structured action. A Skill may coordinate multiple tools or MCP servers.

```text
Skill
├── profile.get_projection
├── job_discovery.research
├── job_discovery.select
├── resume.match
└── resume.save_after_confirmation
```

Pass data between tools through explicit schemas. Prefer identifiers and compact summaries over raw responses:

```text
research → run_id + candidate summaries
user selection → run_id + result_ref
select → JDAnalysis summary + selected job reference
resume.match → resume_id + JDAnalysis
```

For multi-tool workflows document:

- which steps are sequential;
- which independent steps may run in parallel;
- how errors from one server affect later steps;
- what data may cross each boundary;
- the final stop/confirmation point.

Do not let a Skill invent opaque external IDs. The receiving Gateway or Tool must validate ownership and resource membership.

## Scripts

Use `scripts/` for deterministic, bounded helpers such as:

- schema validation;
- file type and size checks;
- normalization;
- format conversion;
- report rendering;
- quality checks.

Scripts must be non-interactive by default, accept structured flags or stdin, write machine-readable results when requested, and return non-zero exit codes on failure. Do not hide the primary state machine in a script when the project already has a Gateway or workflow implementation.

## References

Use `references/` for long material that should not be loaded on every Skill invocation:

- tool contracts;
- domain rules;
- error code tables;
- quality criteria;
- provider-specific notes;
- extended examples.

The root `SKILL.md` should link to the reference file and state when to read it.

## Assets

Use `assets/` for static resources:

- output templates;
- JSON schema examples;
- report layouts;
- prompt fragments that are intentionally versioned with the Skill.

Never store secrets, user credentials, cookies, or unredacted private user data in a Skill package.

## Main Agent, Skill, Tool, workflow, and subagent

Use these boundaries:

```text
Main Agent
├── reads CareerProfile and conversation state
├── selects a Skill or a simple Tool
├── handles ambiguity and user confirmation
└── stores a compact cross-step task cursor

Skill
├── describes a reusable multi-step capability
├── coordinates tools and optional subagents
├── defines stop conditions and recovery
└── returns a compact structured result

Tool / Gateway
├── performs one bounded action
├── validates schema and authorization
└── owns deterministic side effects and external boundaries

Workflow engine (for example LangGraph)
├── enforces state transitions
├── pauses and resumes
├── bounds retries and timeouts
└── preserves execution trace

Subagent
├── handles an isolated, potentially long analysis
├── receives a minimal task package
└── returns a structured result, not hidden global state
```

A Skill does not automatically require a subagent. A deterministic BOSS search should remain a Tool/connector. A long JD analysis, resume tailoring, OCR pass, or independent review may use a subagent.

## Context boundaries

Share only the minimum task projection:

```text
CareerProfile projection + current user request
→ Skill input
→ Tool-specific validated input
→ structured result
→ compact conversation state
```

Do not pass the complete conversation, all resumes, unrelated job runs, credentials, raw BOSS responses, model prompts, or full traces unless the workflow explicitly requires them.

Long-term confirmed facts and operating preferences belong in the structured memory store. Temporary run IDs, user selections, pending actions, and retry state belong in conversation/task state or the workflow run store. Inferences from a model or external source must not silently become long-term memory.

## Iterative workflows

For generation-and-review workflows, define a bounded loop:

```text
initial draft
→ deterministic validation
→ independent quality review
→ revise affected sections
→ validate again
→ stop when threshold is met or iteration limit is reached
```

The Skill must define:

- quality criteria;
- validation command or Tool;
- maximum iterations;
- what is regenerated versus preserved;
- the stop condition;
- what is returned when the threshold is not met.

Never use an unbounded "repeat until good" instruction.

## Safety and confirmation

Every Skill must state:

- which operations are read-only;
- which operations create or update durable state;
- which actions require explicit user confirmation;
- what happens after provider failure or rate limiting;
- what data is redacted from output and trace.

For external job discovery:

- search only after the user authorizes it;
- never invent BOSS IDs or bypass provider controls;
- never auto-apply or add to a waitlist;
- preserve candidates when detail fails;
- request user-provided JD text as a fallback;
- retry only within a bounded policy.

## Output contract

Return a compact structured result containing only what the main Agent needs:

```json
{
  "state": "selection_required",
  "run_id": "run_123",
  "items": [],
  "next_action": "select_result"
}
```

Do not return raw provider envelopes, credentials, cookies, hidden prompts, full trace events, or large source documents by default. Provide an explicit diagnostic mode when detailed trace is needed.

## Review checklist

Before adding or changing a Skill, verify:

- [ ] Root `SKILL.md` exists and frontmatter is valid.
- [ ] Description states when the Skill should load.
- [ ] Workflow has explicit inputs, outputs, dependencies, and stop conditions.
- [ ] Tools and MCP calls use structured contracts.
- [ ] User confirmation points are explicit.
- [ ] Retries and iterations are bounded.
- [ ] Scripts are deterministic and non-interactive.
- [ ] Long references are progressively disclosed.
- [ ] No secret or unredacted private data is packaged.
- [ ] Main Agent context and isolated subagent context are clearly separated.
- [ ] Output is compact, parseable, and actionable.
