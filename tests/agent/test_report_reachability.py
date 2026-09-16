"""Every report kind has a way back that does not depend on the recent window.

The archived catalogue is bounded at twelve, so a direct reference expires. That
is a deliberate bound — the catalogue rides in every turn's context — and it is
only acceptable because a report is still reachable through the entity it
belongs to. Job research is keyed to a saved job and mock interview results to
an application; interview preparation had no such route, which made the bound a
real loss for that kind alone rather than a loss of convenience.

These tests hold the routes uniform, so trimming the catalogue never costs
reachability again.
"""

from __future__ import annotations

import inspect

import pytest

from career_agent.agent import main_agent_contracts as contracts
from career_agent.agent.main_agent_contracts import (
    CareerProfileContext,
    ConversationTaskState,
    GetInterviewPreparationToolArguments,
    InterviewCandidateContextItem,
    MainAgentContext,
)

_ENTITY_SELECTORS = {
    "GetJobResearchToolArguments": "selection_index",
    "GetMockInterviewResultToolArguments": "application_selection_index",
    "GetInterviewPreparationToolArguments": "interview_selection_index",
}


@pytest.mark.parametrize(
    ("model_name", "selector"), sorted(_ENTITY_SELECTORS.items())
)
def test_each_readback_can_be_reached_from_its_owning_entity(
    model_name, selector
) -> None:
    """A route that survives the catalogue being trimmed."""
    model = getattr(contracts, model_name)
    assert selector in model.model_fields, (
        f"{model_name} can only be reached by a resource handle, so its report "
        "becomes unreachable once the archived catalogue trims it"
    )


def test_the_resource_contract_covers_every_card_reader_kind() -> None:
    """The closed kind set is the dispatch surface used by the generic API."""
    kinds = set(
        contracts.ConversationResourceReference.model_fields["kind"].annotation.__args__
    )
    assert kinds == {
        "job_research_report",
        "mock_interview_report",
        "interview_preparation",
        "interview_retro_report",
        "job_analysis",
        "resume_job_match",
        "resume_tailoring_draft",
        # The one user-supplied kind: an exact resume version attached to a
        # user message. It is opened through the resume document route rather
        # than a card reader, and never appears on an assistant message.
        "resume_version",
        # An immutable JD snapshot of a saved job, attached to the assistant
        # message that read it. Opened through the jd-snapshot route by
        # ``resource_id``, so the card keeps showing the version it analysed.
        "saved_job",
    }
    # The original three also have entity-keyed model routes after their
    # archived reference leaves the bounded prompt catalogue. The newer three
    # are historical UI resources first; adding model-side content reasoning
    # for them belongs to the controlled-observation work rather than faking a
    # CRUD route whose contents the decision model still cannot see.
    assert set(_ENTITY_SELECTORS) == {
        "GetJobResearchToolArguments",
        "GetMockInterviewResultToolArguments",
        "GetInterviewPreparationToolArguments",
    }


def _context() -> MainAgentContext:
    return MainAgentContext(
        conversation_id="c1",
        profile=CareerProfileContext(user_id="u1"),
        task=ConversationTaskState(
            interview_candidates=(
                InterviewCandidateContextItem(
                    interview_round_id="round-7",
                    application_id="app-1",
                    sequence_number=1,
                    employer_label=None,
                    status="scheduled",
                    scheduled_start=None,
                ),
            )
        ),
        user_message="那场面试的准备材料呢",
    )


def test_an_interview_selection_resolves_to_the_interview_not_a_preparation() -> None:
    """Which preparation belongs to an interview is the store's answer.

    Resolving to a preparation id here would need the projection to know what
    the store knows, and would pin the answer to whichever preparation was
    current when the candidate list was built.
    """
    projected = contracts.project_interview_preparation_arguments(
        _context(), "get_interview_preparation", {"interview_selection_index": 1}
    )

    assert projected["interview_round_id"] == "round-7"
    assert projected["preparation_id"] is None
    assert projected["user_id"] == "u1"


def test_the_selection_index_is_bounded_like_every_other_selector() -> None:
    with pytest.raises(ValueError, match="out of range"):
        contracts.project_interview_preparation_arguments(
            _context(), "get_interview_preparation", {"interview_selection_index": 9}
        )
    with pytest.raises(ValueError):
        GetInterviewPreparationToolArguments.model_validate(
            {"interview_selection_index": 0}
        )


def test_the_model_never_supplies_the_resolved_round_id() -> None:
    """It is projection output, so it must not be in the model-facing schema."""
    assert (
        "interview_round_id" not in GetInterviewPreparationToolArguments.model_fields
    )
    source = inspect.getsource(contracts.project_interview_preparation_arguments)
    assert "_reject_internal_identifiers" in source
