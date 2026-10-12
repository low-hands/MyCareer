"""Deterministic, turn-local policy fragments for the runtime control message."""

from __future__ import annotations

import re

from career_agent.agent.contracts.context import MainAgentContext


# ASCII boundaries avoid matching offered/reapplication while still matching
# mixed Chinese/English phrases such as "拿到offer了".
_INTERVIEW_TERMS = re.compile(
    r"面试|投递|应聘|招聘|约面|[一二三四终复初]面|HR面|技术面|笔试|"
    r"(?<![A-Za-z0-9_])(?:interview(?:s|ing|er|ers)?|application(?:s)?|"
    r"appl(?:ied|ying)|offers?|onsite|screening)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_NOTES_TERMS = re.compile(r"工作笔记|工作便笺|working[ _-]notes|scratchpad", re.IGNORECASE)


def _offered(context: MainAgentContext) -> frozenset[str]:
    selection = context.capability_selection
    return frozenset(getattr(selection, "offered_names", ()))


def conditional_policy(context: MainAgentContext) -> str:
    """Return only policies whose authoritative runtime signals are active."""

    sections: list[str] = []
    offered = _offered(context)
    if context.has_compressed_history:
        sections.append(
            "Compressed history: through_sequence > 0 and recent_from_sequence > 1 "
            "omit earlier turns. The summary is a lossy digest; zero "
            "omitted_*_count values do not prove completeness. archived_fact_count "
            "marks hidden facts. Read back missing earlier chat messages, not the "
            "active constraint ledger; follow tool schema for bounds."
        )
    if (
        context.free_text_preferences_active_total
        or context.free_text_preferences_quarantined_total
    ):
        sections.append(
            "Free-text preferences: the confirmed section is the runtime-resolved "
            "effective view for the current job context; do not recompute how "
            "sources were combined. Only confirmed preferences may guide relevant "
            "recommendations; never filter, rank, or recommend from quarantined "
            "candidates. When relevant, ask the user to confirm the exact statement "
            "with propose_free_text_preference_confirmation. Call "
            "confirm_free_text_preference only on a later turn after explicit agreement."
        )
    notes = context.working_notes
    if (
        (notes is not None and bool(notes.markdown.strip()))
        or _NOTES_TERMS.search(context.user_message)
        or any(
            observation.state == "working_notes_derived_argument"
            for observation in context.tool_observations
        )
    ):
        sections.append(
            "Working notes: working_notes is an unconfirmed agent scratchpad. "
            "It may guide clarifying questions and response style only. Never "
            "use it to filter, rank, recommend, apply, schedule, or mutate "
            "authoritative career state. If a choice rests on a note-only "
            "preference, even when the user asks you to go by what you remember, "
            "read facts if useful but do not choose; identify the note as "
            "unconfirmed and ask for confirmation or the user's preference. "
            "Use update_working_notes to replace stale observations or unfinished "
            "threads, keeping the complete note under 2000 characters. When "
            "stale_days appears, review each note before carrying it forward "
            "and remove outdated material. After working_notes_derived_argument, "
            "do not retry note-derived content with different wording; ask for "
            "confirmation or use an authoritative source. Pass the current "
            "working_notes revision; after working_notes_stale, merge the current "
            "note before retrying and never overwrite it directly."
        )
    if context.career_episodes:
        sections.append(
            "Career episodes are a bounded cross-conversation event catalogue, "
            "not factual evidence. Use titles and synopses only to locate events. "
            "When details matter, call search_career_episodes with the projected "
            "detail_ref and dereference returned resource_refs."
        )
    interview_tools = {
        name for name in offered
        if "interview" in name or "application" in name
    }
    # The final two native messages cover the adjacent exchange that can make
    # a short confirmation refer back to an interview. Longer messages are
    # treated as new requests unless their own text names an interview topic.
    recent_text = " ".join(message.content for message in context.recent_messages[-2:])
    is_short_followup = len(context.user_message) <= 60
    if (
        _INTERVIEW_TERMS.search(context.user_message)
        or (
            interview_tools
            and (context.task.active_interview_round_id or context.task.active_application_id)
        )
        or (_INTERVIEW_TERMS.search(recent_text) and is_short_followup)
    ):
        sections.append(
            "Interview preparation and tracking: a user may ask for preparation "
            "before tracking an application, JD, resume, or interview. Do not "
            "force-link to an unrelated saved application or require those records. "
            "Use known company, role, schedule, timezone, format, preparation "
            "focus, runtime clock and active entity; ask only for missing inputs "
            "for the next action. Give an explicitly provisional preparation plan "
            "from those facts; JD-specific and resume-specific advice requires "
            "those optional sources. Offer tracked applications only when the "
            "user asks to link or persist the interview. A bare report such as "
            "'I have an interview' does not authorize a write; ask one concise "
            "confirmation about tracking and moving the application to "
            "interviewing. An immediately adjacent affirmative or imperative "
            "response is authority for that exact write; do not ask again. Link "
            "one uniquely compatible active application or saved job directly "
            "without repeating company or role. With no compatible focus, do "
            "not attach an unrelated historical JD; ask for missing company/role "
            "or let the user select or update an application. A uniquely focused "
            "saved job may create the minimal tracking application and move it "
            "to interviewing; a resume is optional."
        )
    return "\n\n".join(sections)
