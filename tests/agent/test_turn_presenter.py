from career_agent.agent.contracts.main_agent import (
    ConversationResourceReference,
    ToolObservation,
)
from career_agent.agent.presentation.presenter import TurnPresenter


def _presenter() -> TurnPresenter:
    return TurnPresenter(
        render_result=lambda result: str(result.payload.get("body") or result.message)
    )


def test_turn_presenter_fails_open_when_a_card_reference_is_missing() -> None:
    result = ToolObservation(
        tool_name="get_job_research",
        state="job_research_ready",
        message="调研已完成。",
        payload={"body": "完整调研正文"},
    )

    assert TurnPresenter.conversation_content(
        result,
        screen="完整调研正文",
        composed=False,
    ) == "完整调研正文"
    assert _presenter()._undelivered_bodies((result,)) == "完整调研正文"


def test_turn_presenter_deduplicates_resource_cards_across_results_and_inputs() -> None:
    report = ConversationResourceReference(
        kind="job_research_report",
        resource_id="report-1",
        status_at_delivery="current",
        anchored_by_other_job=False,
    )
    saved_job = ConversationResourceReference(
        kind="saved_job",
        resource_id="snapshot-1",
        job_posting_id="job-1",
    )
    results = (
        ToolObservation(
            tool_name="research_job",
            state="job_research_ready",
            message="调研已完成。",
            resource_ref=report,
        ),
        ToolObservation(
            tool_name="get_job_research",
            state="job_research_ready",
            message="已读取调研。",
            resource_ref=report,
        ),
    )

    assert TurnPresenter.turn_resource_refs(
        results,
        input_refs=(saved_job, saved_job),
    ) == (report, saved_job)
