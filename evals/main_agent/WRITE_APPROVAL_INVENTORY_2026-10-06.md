# Write capability approval inventory — 2026-10-06

`approval_policy` is the execution-side review floor. All model-callable writes
use `owner_rule` (default permit, but an owner's `confirm_before` rule can
require review, including the application confirmation setting) or `always`
(review required for the exact call, regardless of owner settings). `never` is
reserved for non-writes and runtime-owned writes, which do not go through the
model's owner-rule approval boundary. This inventory does not change visibility
or grant execution rights.

All entries below have effect `WRITE`. “Undo” describes an available practical
recovery path, not an external exactly-once guarantee. Unless stated otherwise,
the repo does not expose a direct inverse operation for the recorded write.

| Capability | Before → after | External write | Impact; undo/recovery | Policy reason |
| --- | --- | --- | --- | --- |
| `update_working_notes` | owner_rule → owner_rule | No | Provisional working notes; replaceable | Agent scratch state; optional owner review remains available |
| `update_owner_settings` | always → always | No | Durable owner rules; editable but self-governing | Model cannot approve a change to its own rules |
| `complete_action_item` | owner_rule → owner_rule | No | Action status; can be changed through later action handling | Represents user's decision to finish a task |
| `dismiss_action_item` | owner_rule → owner_rule | No | Action status; “今日待办 → 已忽略”可恢复 | Represents user's decision to discard a task |
| `snooze_action_item` | owner_rule → owner_rule | No | Action reminder time; can be changed later | Represents user's timing decision |
| `open_job_search` | owner_rule → owner_rule | No | Search/navigation state; can reopen | No external write; optional owner review remains available |
| `analyze_job` | owner_rule → owner_rule | No | Generated job analysis; can regenerate | Reproducible result; optional owner review remains available |
| `correct_job_requirement_tier` | owner_rule → owner_rule | No | Stored requirement tier; can correct again | Changes a user-facing interpretation of a requirement |
| `research_job` | owner_rule → owner_rule | No | Generated job research; can rerun | Reproducible result; optional owner review remains available |
| `retry_job_research` | owner_rule → owner_rule | No | Replaces failed research result; can retry | Retry of derived result; optional owner review |
| `confirm_job_intent` | owner_rule → owner_rule | No | Saved job intent; can later revise | Commits the user's stated intent |
| `match_resume_to_job` | owner_rule → owner_rule | No | Generated match; can recompute | Reproducible result; optional owner review |
| `draft_resume_tailoring` | owner_rule → owner_rule | No | Draft version; can regenerate | Provisional draft; optional owner review |
| `review_resume_tailoring` | owner_rule → owner_rule | No | Generated draft review; can rerun | Reproducible critique; optional owner review |
| `revise_resume_tailoring` | owner_rule → owner_rule | No | New draft revision; can revise again | Provisional draft; optional owner review |
| `finalize_resume_tailoring` | owner_rule → owner_rule | No | Final resume version; later version can supersede it | Marks a draft as the user's final version |
| `export_resume_artifact` | owner_rule → owner_rule | No | Generated local artifact; can re-export | No external publication; optional owner review |
| `create_application` | owner_rule → owner_rule | No | New application record; “投递记录”可按条删除，无关联项时直接撤销 | Records that the user has chosen to apply |
| `update_application_status` | owner_rule → owner_rule | No | Application status; can update again | Changes tracked progress on the user's behalf |
| `sync_application_emails` | owner_rule → owner_rule | No | Imported mail events; can resync, no direct bulk undo | Imports external facts into durable tracking |
| `resolve_email_event` | owner_rule → owner_rule | No | Email/application link or resolution; can revisit | Accepts an interpretation of an external event |
| `create_interview` | owner_rule → owner_rule | No | New interview round; “面试中心”可按轮次删除，无日历或练习关联时直接撤销 | Creates a durable user schedule record |
| `update_interview` | owner_rule → owner_rule | No | Interview details; can update again | Changes a user schedule record |
| `complete_interview` | owner_rule → owner_rule | No | Interview status; no direct undo exposed | Closes a user schedule record |
| `record_interview_retro` | owner_rule → owner_rule | No | Durable interview retrospective; can add later notes | Stores the user's account of an event |
| `prepare_interview` | owner_rule → owner_rule | No | Generated preparation; can regenerate | Reproducible result; optional owner review |
| `prepare_interview_calendar_sync` | owner_rule → owner_rule | No | Internal calendar proposal; can prepare again | Preview only; external execution has mandatory review |
| `execute_calendar_proposal` | always → always | **Yes** | External calendar event/link; reconciliation required if result unknown | External write requires exact-call approval |
| `start_mock_interview` | owner_rule → owner_rule | No | Practice session; can start another | Ephemeral practice; optional owner review |
| `restart_mock_interview` | owner_rule → owner_rule | No | Practice progress reset; cannot restore discarded answers automatically | May discard in-progress work |
| `confirm_free_text_preference` | owner_rule → owner_rule | No | Saved preference; can revise later | Commits an owner preference after proposal |
| `confirm_memory_amendment` | owner_rule → owner_rule | No | Changed career fact; can amend again | Commits a factual correction after proposal |
| `confirm_memory_tombstone` | always → always | No | Retired career fact; no direct restoration exposed | Destructive factual deletion |
| `propose_career_fact` | owner_rule → owner_rule | No | Pending fact proposal; can leave unconfirmed | Provisional proposal; optional owner review |
| `confirm_career_fact` | owner_rule → owner_rule | No | Accepted career fact; can amend/tombstone later | Commits a fact after proposal |
| `confirm_constraint_retirement` | always → always | No | Retired conversation constraint; no direct restore exposed | Removes an active safety or preference constraint |
| `handle_mock_interview_input` | never → never | No | Runtime-owned practice turn; controlled by active workflow | Not model callable; workflow validates input |
| `retry_mock_interview` | never → never | No | Runtime-owned practice retry; controlled by active workflow | Not model callable; workflow validates retry |

There are 38 writes: two runtime-owned `never`, 32 model-callable `owner_rule`,
and four `always`. The only declared external write is
`execute_calendar_proposal`, and it remains `always`. No execution behavior
changes from the earlier inventory: the 14 proposed `never` labels for
model-callable writes were removed because they behaved exactly like
`owner_rule`. The catalogue now rejects any future model-callable WRITE with
`never`.
