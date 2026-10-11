from career_agent.agent.contracts.observations import (
    DecisionObservation, append_decision_observation, decision_observation_projection,
)
from career_agent.agent.contracts.resources import ConversationResourceReference


def test_success_receipt_does_not_claim_report_contents_or_readback():
    observation = DecisionObservation(tool_name="research_job", state="ready", message="Completed")
    projected = decision_observation_projection((observation,))[0]
    assert projected["evidence"] == {"body_status": "receipt_only", "readback_status": "unavailable"}


def test_readback_requires_a_resolvable_handle_and_keeps_its_subject():
    reference = ConversationResourceReference(
        kind="job_research_report", resource_id="private-report-id", title="Company A report",
        status_at_delivery="current", anchored_by_other_job=False,
    )
    observation = DecisionObservation(
        tool_name="research_job", state="ready", message="Completed", body="An excerpt",
        resource_ref=reference,
    )
    unresolved = decision_observation_projection((observation,))[0]
    assert unresolved["evidence"]["readback_status"] == "unavailable"
    resolved = decision_observation_projection((observation,), reference_handles={"private-report-id": "report-1"})[0]
    assert resolved["evidence"] == {"body_status": "excerpt", "readback_status": "available"}
    assert resolved["title"] == "Company A report"
    assert resolved["reference"] == "report-1"
    assert "private-report-id" not in str(resolved)


def test_body_eviction_updates_evidence_without_losing_readback():
    reference = ConversationResourceReference(kind="job_research_report", resource_id="stored-report", status_at_delivery="current", anchored_by_other_job=False)
    old = DecisionObservation(tool_name="research_job", state="ready", message="Done", body="Report excerpt", resource_ref=reference)
    new = DecisionObservation(tool_name="list_resumes", state="ready", message="Listed")
    retained = append_decision_observation((old,), new)
    projected = decision_observation_projection(retained, reference_handles={"stored-report": "report-1"})
    assert projected[0]["evidence"] == {"body_status": "receipt_only", "readback_status": "available"}

