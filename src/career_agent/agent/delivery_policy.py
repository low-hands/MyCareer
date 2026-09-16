from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping


@dataclass(frozen=True)
class DeliveryPolicy:
    """How one tool-result state is delivered, declared in one place.

    These properties used to be four independent tables inside
    ``MainAgentRuntime``. Nothing held them to the same set of states, so a
    state could be added to some and not the others, and one was.

    The two questions below are separate and were briefly conflated into one
    flag, with the consequence that a state whose transcript row has to be
    compressed also stopped delivering its body — the mock interview single
    question readback renders four thousand characters the candidate explicitly
    asked for, has no card to hold them, and was reduced to a one-line receipt.
    """

    waiting: bool = False
    """The result is waiting on the user, so no further tool call may run."""

    outcome: Literal["completed", "failed"] = "completed"
    """Whether the result represents a capability failure.

    This is explicit because spelling conventions are not semantics: checkpoint
    loss and graph incompatibility are failures without a ``_failed`` suffix.
    """

    durable_message: Literal["full", "summary"] = "full"
    """Whether the transcript row keeps the delivered prose or a bounded line.

    ``summary`` means ``message`` is a short statement of outcome written for
    this purpose, and the row keeps that instead of what was displayed. It is
    about the row's size in every later turn's window — nothing else.
    """

    body_delivery: Literal["message", "resource_card"] = "message"
    """Where the full content reaches the reader.

    ``resource_card`` means the turn attaches a ``resource_ref`` and the UI
    renders the body from the stored entity, so the message itself carries only
    prose about it. ``message`` means the message *is* the delivery and nothing
    else will show it — compressing what is streamed would simply lose it.

    This is what decides the writer's ceiling and what gets streamed. Not
    ``durable_message``: a verbatim readback needs a bounded row *and* a full
    delivery, which is only contradictory if the two are one flag.
    """

    body_title: str | None = None
    """The restored card's heading, independent of the live delivery."""

    body_retention: Literal["none", "snapshot", "source"] = "none"

    def __post_init__(self) -> None:
        if self.body_delivery == "resource_card" and self.durable_message != "summary":
            raise ValueError(
                "a card-delivered body must not also be kept in the row"
            )
        keeps_body = self.body_retention != "none"
        if keeps_body != (self.body_title is not None):
            raise ValueError("a retained body requires a card title")
        if keeps_body and (
            self.durable_message != "summary" or self.body_delivery != "message"
        ):
            raise ValueError("only condensed message bodies need separate retention")
        if self.waiting and self.outcome == "failed":
            raise ValueError("a failed result must return to orchestration, not wait")

    @property
    def condensed_message(self) -> bool:
        return self.durable_message == "summary"

    @property
    def delivers_body_elsewhere(self) -> bool:
        return self.body_delivery == "resource_card"

_WAITING = DeliveryPolicy(waiting=True)
_FAILED = DeliveryPolicy(outcome="failed")
_PLAIN = DeliveryPolicy()


def _card() -> DeliveryPolicy:
    """A report the UI renders from its entity; the message is prose about it."""
    return DeliveryPolicy(
        durable_message="summary",
        body_delivery="resource_card",
    )


def _summarised(
    title: str | None = None,
    retention: Literal["none", "snapshot", "source"] = "none",
) -> DeliveryPolicy:
    return DeliveryPolicy(
        durable_message="summary", body_title=title, body_retention=retention
    )


_POLICIES: dict[str, DeliveryPolicy] = {
    # Waiting on the user. Reaching ``decide`` with one of these ends the turn.
    "calendar_approval_required": _WAITING,
    # An owner rule stopped this action and sealed it. The turn ends here by
    # the same logic as every other gate: nothing further can be decided until
    # the person answers, and the answer arrives as its own ingress.
    "capability_confirmation_required": _WAITING,
    "email_events_pending": _WAITING,
    "mock_interview_answer_required": _WAITING,
    "mock_interview_running": _WAITING,
    "memory_amendment_proposed": _WAITING,
    "memory_tombstone_proposed": _WAITING,
    "free_text_preference_confirmation_proposed": _WAITING,
    "free_text_preference_confirmed_structured_proposed": _WAITING,
    "career_fact_proposed": _WAITING,
    "constraint_retirement_proposed": _WAITING,
    "resume_final_review_blocked": _WAITING,
    "resume_tailoring_review_blocked": _WAITING,
    "resume_tailoring_superseded": _WAITING,
    # Failures return to the decision model with a bounded receipt and, when
    # known, an explicit retryability fact. Names are intentionally irrelevant.
    # Nothing was written this turn, and retrying cannot change that: the slot
    # holds an earlier action whose outcome only reconciliation can establish.
    # Classified as failed so the loop returns to the model to explain, and
    # without a ``retryable`` fact so the retry budget is not spent on it.
    "action_reconciliation_required": _FAILED,
    # A seal that resolved without running: expired, already settled, or
    # cancelled by the owner. Plain rather than failed — nothing went wrong, and
    # the turn has no model step to return to.
    "capability_confirmation_cancelled": _PLAIN,
    "capability_confirmation_expired": _PLAIN,
    "capability_confirmation_in_progress": _PLAIN,
    "owner_settings_updated": _PLAIN,
    "owner_settings_conflict": _PLAIN,
    "tool_profile_switched": _PLAIN,
    "tool_profile_unchanged": _PLAIN,
    "calendar_sync_not_available": _FAILED,
    "calendar_write_failed": _FAILED,
    "failed": _FAILED,
    "job_research_failed": _FAILED,
    "mock_interview_checkpoint_missing": _FAILED,
    "mock_interview_graph_incompatible": _FAILED,
    "mock_interview_restart_failed": _FAILED,
    "resume_tailoring_not_ready": _FAILED,
    # Report-shaped. Presenter renders the screen, ``message`` is the row.
    "daily_brief_ready": _summarised("每日简报", "snapshot"),
    "interview_preparation_ready": _card(),
    "interview_retro_recorded": _card(),
    "job_analysis_ready": _card(),
    "job_research_ready": _card(),
    "mock_interview_completed": _card(),
    "mock_interview_result_found": _card(),
    "resume_analysis_ready": _summarised("简历分析", "source"),
    "resume_job_match_ready": _card(),
    "resume_tailoring_draft_ready": _card(),
    # Reading a saved job hands the model the JD as its observation body; the
    # reader gets a card pinned to the immutable snapshot, and the row keeps
    # the bounded receipt. The JD text itself never enters the message.
    "saved_job_ready": _card(),
    # The comparison table is rendered prose, far richer than its receipt.
    # Registered plain, the model could not see it and the reader would lose
    # it once the model — not the presenter — writes the message.
    "saved_jobs_compared": _summarised("岗位对比", "snapshot"),
    # Split screen and row without the writer: one mock interview exchange is
    # read back verbatim, so restating it would only cost fidelity, but the
    # answer it quotes is up to 20k characters and cannot enter the row.
    # Bounded row, full delivery: the exchange is quoted verbatim because
    # that is what was asked for, and the answer it quotes runs to twenty
    # thousand characters, which the row cannot carry.
    "mock_interview_question_found": _summarised("模拟面试问答回看", "source"),
    # Historical rows are useful as a turn-local observation, but storing the
    # rendered span would immediately put the recalled text back into every
    # later recent window. Keep only the bounded receipt in conversation.
    "conversation_span_found": _summarised(),
    # Resume quotations are provenance readbacks, not durable conversation
    # memory. Keep the quote in the bounded turn-local observation only.
    "claim_source_found": _summarised(),
    "career_memory_detail_found": _summarised(),
    "career_memory_search_found": _summarised(),
    "career_episode_search_found": _summarised(),
    "career_history_found": _summarised(),
}

_POLICIES.update(
    dict.fromkeys(
        (
            "action_item_not_found",
            "action_item_resolved",
            "action_item_snoozed",
            "action_items_found",
            "action_execution_replayed",
            "application_input_not_found",
            "application_not_found",
            "application_ready",
            "application_update_conflict",
            "applications_found",
            "authorization_refused",
            "calendar_account_required",
            "calendar_accounts_found",
            "calendar_approval_invalid",
            "calendar_links_found",
            "calendar_proposal_not_found",
            "calendar_proposal_ready",
            "calendar_sync_complete",
            "claim_source_not_found",
            "archived_constraints_ready",
            "no_archived_constraints",
            "constraint_not_found",
            "constraint_retired",
            "constraint_retirement_confirmation_missing",
            "career_memory_detail_not_found",
            "career_memory_search_empty",
            "career_episode_search_empty",
            "career_history_empty",
            "conversation_span_empty",
            "compare_input_not_found",
            "email_account_not_found",
            "email_event_not_found",
            "email_event_resolution_conflict",
            "email_event_resolved",
            "email_events_found",
            "free_text_preference_confirmation_missing",
            "free_text_preference_confirmed",
            "interview_application_conflict",
            "interview_completion_conflict",
            "interview_not_found",
            "interview_preparation_input_not_found",
            "interview_preparation_not_available",
            "interview_preparation_not_found",
            "interview_ready",
            "interview_retro_conflict",
            "interview_update_conflict",
            "interviews_found",
            "invalid_action_transition",
            "invalid_application_transition",
            "invalid_timezone",
            "job_intent_proposed",
            "job_intent_recorded",
            "career_memory_amended",
            "career_fact_candidate_missing",
            "career_fact_confirmation_missing",
            "career_fact_confirmed",
            "memory_amendment_confirmation_missing",
            "memory_amendment_target_not_found",
            "memory_tombstone_cleanup_incomplete",
            "memory_tombstone_confirmation_missing",
            "memory_tombstone_target_changed",
            "memory_tombstone_target_not_found",
            "memory_tombstoned",
            "working_notes_stale",
            "working_notes_updated",
            "working_notes_derived_argument",
            "job_research_not_found",
            "job_research_not_retryable",
            "job_search_page_ready",
            "match_input_not_found",
            "mock_interview_cancelled",
            "mock_interview_input_retry_required",
            "invalid_input",
            "no_mock_interview_result_found",
            "no_mock_interview_to_restart",
            "resume_analysis_confirmed",
            "resume_analysis_decision_expired",
            "resume_analysis_not_found",
            "resume_analysis_rejected",
            "resume_artifact_ready",
            "resume_job_match_not_found",
            "resume_metadata_ready",
            "resume_not_found",
            "resume_tailoring_already_finalized",
            "resume_tailoring_draft_not_found",
            "resume_tailoring_finalized",
            "resume_version_not_found",
            "resumes_found",
            "saved_job_not_found",
            "saved_jobs_found",
            "target_role_not_found",
            "target_roles_found",
        ),
        _PLAIN,
    )
)

DELIVERY_POLICIES: Mapping[str, DeliveryPolicy] = MappingProxyType(_POLICIES)


def policy_for(state: str) -> DeliveryPolicy:
    """The policy for a state, defaulting to plain delivery.

    An unregistered state is delivered as-is rather than raising: a tool result
    is already in hand by the time this is asked, and refusing to present it
    would lose work that may have had durable effects. The registry is held
    complete by test instead, which fails the build rather than the turn.
    """
    return DELIVERY_POLICIES.get(state, _PLAIN)


def is_waiting(state: str) -> bool:
    return policy_for(state).waiting


def is_failed(state: str) -> bool:
    return policy_for(state).outcome == "failed"


def condenses_message(state: str) -> bool:
    return policy_for(state).condensed_message


def delivers_body_elsewhere(state: str) -> bool:
    return policy_for(state).delivers_body_elsewhere
