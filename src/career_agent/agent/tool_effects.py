from __future__ import annotations

from types import MappingProxyType
from typing import Literal, Mapping


ToolEffect = Literal["READ", "WRITE", "CONTROL"]


_CONTROL_CAPABILITIES = frozenset(
    {
        "route_to_capability",
    }
)
"""Calls that steer the decision loop itself and touch no domain store.

A CONTROL capability changes which tools the next decision is made against.
It has no business effect to record: nothing to seal for the owner, nothing to
replay after a crash, and no read or write budget to draw on. It still leaves
a step in the trace, because a route is part of the trajectory an evaluation
replays, and the same route twice in one turn is still a repeated call.
"""


_WRITE_CAPABILITIES = frozenset(
    {
        "analyze_resume",
        "complete_action_item",
        "complete_interview",
        "confirm_constraint_retirement",
        "confirm_job_intent",
        "confirm_free_text_preference",
        "confirm_career_fact",
        "confirm_memory_amendment",
        "confirm_memory_tombstone",
        "create_application",
        "create_interview",
        "dismiss_action_item",
        "draft_resume_tailoring",
        "execute_calendar_proposal",
        "export_resume_artifact",
        "finalize_resume_tailoring",
        "handle_mock_interview_input",
        "match_resume_to_job",
        "open_job_search",
        "prepare_interview",
        "prepare_interview_calendar_sync",
        "record_interview_retro",
        "research_job",
        "resolve_email_event",
        "retry_job_research",
        "review_resume_tailoring",
        "revise_resume_tailoring",
        "snooze_action_item",
        "start_mock_interview",
        "sync_application_emails",
        "update_application_status",
        "update_interview",
        "update_owner_settings",
        "update_working_notes",
        "propose_career_fact",
        "restart_mock_interview",
        "retry_mock_interview",
    }
)

_READ_CAPABILITIES = frozenset(
    {
        "compare_saved_jobs",
        "find_saved_jobs",
        "get_application",
        "get_career_memory_detail",
        "get_calendar_proposal",
        "get_daily_brief",
        "get_interview",
        "get_interview_preparation",
        "get_job_research",
        "get_mock_interview_result",
        "get_resume_analysis",
        "get_resume_job_match",
        "get_resume_metadata",
        "get_resume_tailoring_draft",
        "get_saved_job",
        "list_action_items",
        "list_applications",
        "list_calendar_accounts",
        "list_calendar_links",
        "list_email_events",
        "list_interviews",
        "fetch_archived_constraints",
        "list_resumes",
        "list_target_roles",
        "propose_constraint_retirement",
        "propose_job_intent",
        "propose_free_text_preference_confirmation",
        "propose_memory_amendment",
        "propose_memory_tombstone",
        "read_conversation_span",
        "resolve_claim_source",
        "search_career_episodes",
        "search_career_memory",
        "search_career_history",
    }
)

# Reads that filter or rank what the user is shown are guarded.  Searches over
# confirmed memory are not: they are how a note-derived hunch gets checked
# against an authoritative source, which is what the guard refusal asks for.
_NOTES_GUARDED_CAPABILITIES = (
    (_WRITE_CAPABILITIES - {"update_working_notes"})
    | {
        "find_saved_jobs",
        "compare_saved_jobs",
        "open_job_search",
    }
)

# Capabilities whose result is a choice among the user's options — a ranking,
# a fit verdict, an application.  When the user asks for one "by the preference
# you remember" and nothing confirmed is remembered, the choice would rest on
# the scratchpad alone.
_PREFERENCE_BOUND_CAPABILITIES = frozenset(
    {
        "compare_saved_jobs",
        "match_resume_to_job",
        "create_application",
    }
)

_EXTERNAL_WRITE_CAPABILITIES = frozenset(
    {
        "execute_calendar_proposal",
    }
)
"""Writes whose effect lands outside this deployment's own stores.

The axis is where the effect lives, not how risky it feels. A calendar event
exists on Google's servers and is visible to anyone the calendar is shared with;
nothing here can roll it back, only issue a second write. An application row,
a memory tombstone or an exported file lives in a store this process owns and
can be amended, retired or deleted by a later turn.

Reading an external system is not an external write: ``sync_application_emails``
fetches mail and writes local events, ``research_job`` searches the web and
writes a local report. ``open_job_search`` only asks the client to open a URL
and never touches the platform itself. All three stay internal.

External writes carry a system-level Review verdict
(``system_capability_verdict``) and their own turn budget. New writes that
send, post, book or pay on the user's behalf belong in this set on the day
they are added; a WRITE that is not listed here is claiming to be reversible
from within this codebase.
"""

_RUNTIME_OWNED_CAPABILITIES = frozenset(
    {
        "handle_mock_interview_input",
        "retry_mock_interview",
    }
)
"""Writes the runtime invokes on its own ownership rule, never on a model call.

They continue a workflow the user already entered (an answer inside a running
mock interview) and ``_authorize`` permits them without consulting owner rules,
so a ``confirm_before`` entry naming one would never fire. Must mirror the
registry's ``runtime_workflow_names``.
"""

if _READ_CAPABILITIES & _WRITE_CAPABILITIES:
    raise RuntimeError("a Main Agent capability cannot be both READ and WRITE")
if _CONTROL_CAPABILITIES & (_READ_CAPABILITIES | _WRITE_CAPABILITIES):
    raise RuntimeError("a CONTROL capability cannot also be READ or WRITE")
if _EXTERNAL_WRITE_CAPABILITIES - _WRITE_CAPABILITIES:
    raise RuntimeError("an external write must be declared as a WRITE capability")
if _RUNTIME_OWNED_CAPABILITIES - _WRITE_CAPABILITIES:
    raise RuntimeError("a runtime-owned workflow must be declared as a WRITE capability")
if _RUNTIME_OWNED_CAPABILITIES & _EXTERNAL_WRITE_CAPABILITIES:
    raise RuntimeError("an external write cannot bypass owner rules as runtime-owned")
if _PREFERENCE_BOUND_CAPABILITIES - (_READ_CAPABILITIES | _WRITE_CAPABILITIES):
    raise RuntimeError("a preference-bound capability must be a declared capability")

TOOL_EFFECTS: Mapping[str, ToolEffect] = MappingProxyType(
    {
        **dict.fromkeys(_READ_CAPABILITIES, "READ"),
        **dict.fromkeys(_WRITE_CAPABILITIES, "WRITE"),
        **dict.fromkeys(_CONTROL_CAPABILITIES, "CONTROL"),
    }
)


_REPLAY_SAFE_CAPABILITIES = frozenset(
    {
        # UNIQUE(user_id, job_posting_id): a second call for the same posting
        # returns the existing application rather than creating another.
        "create_application",
        # Carries its own idempotency key and a reconciliation path; the service
        # treats an unsettled attempt as recovery rather than as a new write.
        "execute_calendar_proposal",
        # The handler keys the applied change by the bound confirmation id;
        # a retry returns the audit event's recorded after-state.
        "update_owner_settings",
    }
)
"""Writes that may be invoked again for a slot whose outcome is unknown.

Declared, never inferred. All writes record intent, but re-running one requires
something downstream that collapses a second call into the original effect.

``research_job`` is deliberately absent although it looks eligible. Its unique
index is ``WHERE status = 'running'`` — a guard against two concurrent runs, not
a deduplication of results. What actually collapses a repeat is the freshness
cache (``find_completed(created_after=now - freshness)``), which holds for a
crash-retry seconds later and stops holding once the window passes. Safety that
depends on how long the operator took to retry is not the kind that belongs in
this set.

"""


def replay_safe(name: str) -> bool:
    """Whether re-invoking this capability cannot produce a second effect."""

    return name in _REPLAY_SAFE_CAPABILITIES


def is_notes_guarded(name: str) -> bool:
    """Whether note-only argument content must be reviewed before execution."""

    return name in _NOTES_GUARDED_CAPABILITIES


def is_preference_bound(name: str) -> bool:
    """Whether a remembered-preference request must have a confirmed source."""

    return name in _PREFERENCE_BOUND_CAPABILITIES


def effect_for(name: str) -> ToolEffect:
    try:
        return TOOL_EFFECTS[name]
    except KeyError as error:
        raise ValueError(f"Main Agent capability has no declared effect: {name}") from error


def is_external_write(name: str) -> bool:
    """Whether this capability's effect lands outside the deployment's stores."""

    return name in _EXTERNAL_WRITE_CAPABILITIES


def declared_write_capabilities() -> frozenset[str]:
    """Every capability whose declared effect is WRITE."""

    return _WRITE_CAPABILITIES


def is_runtime_owned(name: str) -> bool:
    """Whether the runtime, not the model, decides to invoke this capability."""

    return name in _RUNTIME_OWNED_CAPABILITIES


def owner_rule_capabilities() -> frozenset[str]:
    """Names an owner rule such as ``confirm_before`` may refer to.

    A write the model can propose is stopped by ``_authorize`` when an owner
    rule says ``review``; a runtime-owned write is not, so a rule naming it
    would be accepted and never enforced.
    """

    return _WRITE_CAPABILITIES - _RUNTIME_OWNED_CAPABILITIES
