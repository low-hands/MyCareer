from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from career_agent.connectors.calendar import (
    CalendarConnectorError,
    CalendarReconciliationResult,
    CalendarWriteResult,
)
from career_agent.domain.interviews import InterviewRound
from career_agent.services.calendar import (
    CalendarProposalConflictError,
    CalendarService,
    CalendarSyncNotAvailableError,
)
from career_agent.services.interviews import InterviewDetail
from career_agent.storage.calendar import SQLiteCalendarStore


NOW = datetime(2026, 8, 26, tzinfo=timezone.utc)


class Interviews:
    def __init__(self):
        self.interview = InterviewRound(
            id="interview-1", user_id="u1", application_id="application-1",
            sequence_number=1, employer_label=None, status="scheduled",
            scheduled_start=NOW + timedelta(days=2),
            scheduled_end=NOW + timedelta(days=2, hours=1),
            timezone="Asia/Shanghai", interview_format="video",
            meeting_url="https://meet.example/interview",
            created_at=NOW, updated_at=NOW,
        )

    def get_interview(self, **kwargs):
        if kwargs["user_id"] != "u1" or kwargs["interview_round_id"] != self.interview.id:
            raise AssertionError("unexpected interview lookup")
        return InterviewDetail(interview=self.interview, events=())


class Applications:
    def get_application(self, **kwargs):
        assert kwargs == {"user_id": "u1", "application_id": "application-1"}
        posting = SimpleNamespace(company_name="Acme", title="AI Engineer")
        return SimpleNamespace(job=SimpleNamespace(posting=posting))


class Connector:
    def __init__(self):
        self.calls = []
        self.reconcile_calls = []
        self.remote = None

    def apply(self, **kwargs):
        self.calls.append(kwargs)
        self.remote = kwargs
        return CalendarWriteResult(
            external_event_id=kwargs["external_event_id"],
            etag=f'etag-{len(self.calls)}',
            html_link="https://calendar.example/event",
        )

    def reconcile(self, **kwargs):
        self.reconcile_calls.append(kwargs)
        if self.remote is None:
            return CalendarReconciliationResult(outcome="not_applied")
        if self.remote["payload_hash"] == kwargs["payload_hash"]:
            return CalendarReconciliationResult(
                outcome="applied",
                write_result=CalendarWriteResult(
                    external_event_id=kwargs["external_event_id"],
                    etag="reconciled-etag",
                    html_link="https://calendar.example/event",
                ),
            )
        if (
            kwargs["operation"] == "update"
            and self.remote["payload_hash"] == kwargs["prior_payload_hash"]
        ):
            return CalendarReconciliationResult(outcome="not_applied")
        return CalendarReconciliationResult(outcome="conflict")


class Resolver:
    def __init__(self, connector):
        self.connector = connector

    def resolve(self, account):
        assert account.user_id == "u1"
        return self.connector


def build_service(tmp_path):
    store = SQLiteCalendarStore(tmp_path / "calendar.sqlite3")
    account = store.add_account(
        user_id="u1", email_address="user@example.com", calendar_id="primary",
        credential_ref="env:GOOGLE_CALENDAR_CREDENTIAL", now=NOW,
    )
    interviews = Interviews()
    connector = Connector()
    service = CalendarService(
        store, interviews, Applications(), Resolver(connector)
    )
    return service, store, account, interviews, connector


def test_calendar_requires_proposal_then_reuses_event_for_updates_and_cancel(tmp_path) -> None:
    service, store, account, interviews, connector = build_service(tmp_path)

    create = service.prepare_interview_sync(
        user_id="u1", interview_round_id="interview-1", now=NOW,
    )
    assert create.operation == "create"
    assert connector.calls == []

    created = service.execute_proposal(
        user_id="u1", proposal_id=create.id, now=NOW + timedelta(minutes=1),
    )
    external_event_id = created.link.external_event_id
    assert created.link.status == "active"
    assert connector.calls[0]["operation"] == "create"

    with pytest.raises(CalendarSyncNotAvailableError, match="already synchronized"):
        service.prepare_interview_sync(
            user_id="u1", interview_round_id="interview-1",
            calendar_account_id=account.id, now=NOW + timedelta(minutes=2),
        )

    interviews.interview = interviews.interview.model_copy(
        update={
            "scheduled_start": NOW + timedelta(days=3),
            "scheduled_end": NOW + timedelta(days=3, hours=1),
            "updated_at": NOW + timedelta(hours=1),
        }
    )
    update = service.prepare_interview_sync(
        user_id="u1", interview_round_id="interview-1", now=NOW + timedelta(hours=1),
    )
    assert update.operation == "update"
    assert update.external_event_id == external_event_id
    service.execute_proposal(
        user_id="u1", proposal_id=update.id, now=NOW + timedelta(hours=1, minutes=1),
    )
    assert connector.calls[1]["operation"] == "update"

    interviews.interview = interviews.interview.model_copy(
        update={"status": "cancelled", "updated_at": NOW + timedelta(hours=2)}
    )
    cancel = service.prepare_interview_sync(
        user_id="u1", interview_round_id="interview-1", now=NOW + timedelta(hours=2),
    )
    assert cancel.operation == "cancel"
    assert cancel.external_event_id == external_event_id
    cancelled = service.execute_proposal(
        user_id="u1", proposal_id=cancel.id,
        now=NOW + timedelta(hours=2, minutes=1),
    )
    assert cancelled.link.status == "cancelled"
    assert connector.calls[2]["operation"] == "cancel"
    assert [event.event_type for event in store.list_events(
        user_id="u1", proposal_id=cancel.id
    )] == ["proposed", "execution_started", "executed"]


def test_ambiguous_write_is_reconciled_without_applying_twice(tmp_path) -> None:
    service, store, _, _, connector = build_service(tmp_path)
    proposal = service.prepare_interview_sync(
        user_id="u1", interview_round_id="interview-1", now=NOW,
    )
    original_apply = connector.apply

    def ambiguous_apply(**kwargs):
        result = original_apply(**kwargs)
        raise CalendarConnectorError(
            "GOOGLE_CALENDAR_TRANSPORT_ERROR",
            "response timed out",
            outcome_unknown=True,
        )

    connector.apply = ambiguous_apply
    with pytest.raises(CalendarConnectorError) as raised:
        service.execute_proposal(
            user_id="u1", proposal_id=proposal.id,
            now=NOW + timedelta(minutes=1),
        )
    assert raised.value.outcome_unknown is True
    assert store.get_proposal(
        user_id="u1", proposal_id=proposal.id
    ).status == "reconciliation_required"
    assert store.get_execution(
        user_id="u1", proposal_id=proposal.id
    ).status == "reconciliation_required"

    connector.apply = original_apply
    recovered = service.execute_proposal(
        user_id="u1", proposal_id=proposal.id,
        now=NOW + timedelta(minutes=2),
    )

    assert recovered.proposal.status == "executed"
    assert len(connector.calls) == 1
    assert len(connector.reconcile_calls) == 1
    assert store.get_execution(
        user_id="u1", proposal_id=proposal.id
    ).status == "succeeded"


def test_process_crash_leaves_intent_and_reconciles_after_lease(tmp_path) -> None:
    service, store, _, _, connector = build_service(tmp_path)
    proposal = service.prepare_interview_sync(
        user_id="u1", interview_round_id="interview-1", now=NOW,
    )
    original_apply = connector.apply

    class SimulatedProcessCrash(BaseException):
        pass

    def crash_after_remote_write(**kwargs):
        original_apply(**kwargs)
        raise SimulatedProcessCrash()

    connector.apply = crash_after_remote_write
    with pytest.raises(SimulatedProcessCrash):
        service.execute_proposal(
            user_id="u1", proposal_id=proposal.id,
            now=NOW + timedelta(minutes=1),
        )

    assert store.get_proposal(
        user_id="u1", proposal_id=proposal.id
    ).status == "executing"
    assert store.get_execution(
        user_id="u1", proposal_id=proposal.id
    ).status == "applying"

    connector.apply = original_apply
    recovered = service.execute_proposal(
        user_id="u1", proposal_id=proposal.id,
        now=NOW + timedelta(minutes=2, seconds=1),
    )

    assert recovered.proposal.status == "executed"
    assert len(connector.calls) == 1
    assert len(connector.reconcile_calls) == 1


def test_active_execution_lease_blocks_a_concurrent_duplicate(tmp_path) -> None:
    service, _, _, _, connector = build_service(tmp_path)
    proposal = service.prepare_interview_sync(
        user_id="u1", interview_round_id="interview-1", now=NOW,
    )
    original_apply = connector.apply

    class SimulatedProcessCrash(BaseException):
        pass

    def crash_after_remote_write(**kwargs):
        original_apply(**kwargs)
        raise SimulatedProcessCrash()

    connector.apply = crash_after_remote_write
    with pytest.raises(SimulatedProcessCrash):
        service.execute_proposal(
            user_id="u1", proposal_id=proposal.id,
            now=NOW + timedelta(minutes=1),
        )

    connector.apply = original_apply
    with pytest.raises(CalendarConnectorError) as raised:
        service.execute_proposal(
            user_id="u1", proposal_id=proposal.id,
            now=NOW + timedelta(minutes=1, seconds=30),
        )
    assert raised.value.code == "CALENDAR_EXECUTION_IN_PROGRESS"
    assert len(connector.calls) == 1


def test_expired_execution_owner_cannot_overwrite_a_new_claim(tmp_path) -> None:
    service, store, _, _, _ = build_service(tmp_path)
    proposal = service.prepare_interview_sync(
        user_id="u1", interview_round_id="interview-1", now=NOW,
    )
    first_proposal, first, _ = store.claim_execution(
        proposal=proposal,
        now=NOW + timedelta(minutes=1),
        lease_duration=timedelta(minutes=1),
    )
    second_proposal, second, must_reconcile = store.claim_execution(
        proposal=first_proposal,
        now=NOW + timedelta(minutes=2, seconds=1),
        lease_duration=timedelta(minutes=1),
    )

    assert must_reconcile is True
    assert second.attempt_count == 2
    store.fail_execution(
        proposal=first_proposal,
        execution=first,
        now=NOW + timedelta(minutes=2, seconds=2),
        error_code="STALE_OWNER",
        error_detail="the first process returned late",
    )

    assert store.get_proposal(
        user_id="u1", proposal_id=proposal.id
    ).status == "executing"
    current = store.get_execution(user_id="u1", proposal_id=proposal.id)
    assert current.status == "applying"
    assert current.attempt_count == second.attempt_count
    assert second_proposal.status == "executing"


def test_local_commit_failure_recovers_from_external_marker(tmp_path, monkeypatch) -> None:
    service, store, _, _, connector = build_service(tmp_path)
    proposal = service.prepare_interview_sync(
        user_id="u1", interview_round_id="interview-1", now=NOW,
    )
    complete = store.complete_execution
    calls = 0

    def fail_once(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("database temporarily unavailable")
        return complete(**kwargs)

    monkeypatch.setattr(store, "complete_execution", fail_once)
    with pytest.raises(CalendarConnectorError) as raised:
        service.execute_proposal(
            user_id="u1", proposal_id=proposal.id,
            now=NOW + timedelta(minutes=1),
        )
    assert raised.value.code == "CALENDAR_LOCAL_COMMIT_FAILED"

    recovered = service.execute_proposal(
        user_id="u1", proposal_id=proposal.id,
        now=NOW + timedelta(minutes=2),
    )

    assert recovered.proposal.status == "executed"
    assert len(connector.calls) == 1
    assert len(connector.reconcile_calls) == 1


def test_ambiguous_update_reapplies_only_after_old_payload_is_verified(tmp_path) -> None:
    service, _, account, interviews, connector = build_service(tmp_path)
    create = service.prepare_interview_sync(
        user_id="u1", interview_round_id="interview-1", now=NOW,
    )
    service.execute_proposal(
        user_id="u1", proposal_id=create.id, now=NOW + timedelta(minutes=1),
    )
    old_hash = create.payload_hash
    interviews.interview = interviews.interview.model_copy(
        update={
            "scheduled_start": NOW + timedelta(days=3),
            "scheduled_end": NOW + timedelta(days=3, hours=1),
        }
    )
    update = service.prepare_interview_sync(
        user_id="u1", interview_round_id="interview-1",
        calendar_account_id=account.id, now=NOW + timedelta(hours=1),
    )
    original_apply = connector.apply

    def timeout_before_remote_write(**kwargs):
        raise CalendarConnectorError(
            "GOOGLE_CALENDAR_TRANSPORT_ERROR",
            "request outcome unavailable",
            outcome_unknown=True,
        )

    connector.apply = timeout_before_remote_write
    with pytest.raises(CalendarConnectorError):
        service.execute_proposal(
            user_id="u1", proposal_id=update.id,
            now=NOW + timedelta(hours=1, minutes=1),
        )

    connector.apply = original_apply
    recovered = service.execute_proposal(
        user_id="u1", proposal_id=update.id,
        now=NOW + timedelta(hours=1, minutes=2),
    )

    assert recovered.proposal.status == "executed"
    assert connector.reconcile_calls[-1]["prior_payload_hash"] == old_hash
    assert [call["operation"] for call in connector.calls] == ["create", "update"]


def test_definitive_connector_failure_closes_the_execution(tmp_path) -> None:
    service, store, _, _, connector = build_service(tmp_path)
    proposal = service.prepare_interview_sync(
        user_id="u1", interview_round_id="interview-1", now=NOW,
    )

    def rejected(**kwargs):
        raise CalendarConnectorError("GOOGLE_CALENDAR_HTTP_400", "bad request")

    connector.apply = rejected
    with pytest.raises(CalendarConnectorError) as raised:
        service.execute_proposal(
            user_id="u1", proposal_id=proposal.id,
            now=NOW + timedelta(minutes=1),
        )

    assert raised.value.outcome_unknown is False
    assert store.get_proposal(
        user_id="u1", proposal_id=proposal.id
    ).status == "failed"
    assert store.get_execution(
        user_id="u1", proposal_id=proposal.id
    ).status == "failed"


def test_new_preview_reconciles_a_prior_not_applied_execution_first(tmp_path) -> None:
    service, store, _, _, connector = build_service(tmp_path)
    proposal = service.prepare_interview_sync(
        user_id="u1", interview_round_id="interview-1", now=NOW,
    )

    def ambiguous(**kwargs):
        raise CalendarConnectorError(
            "GOOGLE_CALENDAR_TRANSPORT_ERROR",
            "timeout",
            outcome_unknown=True,
        )

    connector.apply = ambiguous
    with pytest.raises(CalendarConnectorError):
        service.execute_proposal(
            user_id="u1", proposal_id=proposal.id,
            now=NOW + timedelta(minutes=1),
        )

    replacement = service.prepare_interview_sync(
        user_id="u1", interview_round_id="interview-1",
        now=NOW + timedelta(minutes=2),
    )

    assert replacement.id != proposal.id
    assert replacement.status == "pending"
    assert store.get_proposal(
        user_id="u1", proposal_id=proposal.id
    ).status == "failed"
    assert connector.reconcile_calls[-1]["operation"] == "create"


def test_new_preview_adopts_a_prior_applied_execution_instead_of_rewriting(tmp_path) -> None:
    service, store, _, _, connector = build_service(tmp_path)
    proposal = service.prepare_interview_sync(
        user_id="u1", interview_round_id="interview-1", now=NOW,
    )
    original_apply = connector.apply

    def applied_but_timed_out(**kwargs):
        original_apply(**kwargs)
        raise CalendarConnectorError(
            "GOOGLE_CALENDAR_TRANSPORT_ERROR",
            "timeout",
            outcome_unknown=True,
        )

    connector.apply = applied_but_timed_out
    with pytest.raises(CalendarConnectorError):
        service.execute_proposal(
            user_id="u1", proposal_id=proposal.id,
            now=NOW + timedelta(minutes=1),
        )

    with pytest.raises(CalendarSyncNotAvailableError, match="already synchronized"):
        service.prepare_interview_sync(
            user_id="u1", interview_round_id="interview-1",
            now=NOW + timedelta(minutes=2),
        )

    assert store.get_proposal(
        user_id="u1", proposal_id=proposal.id
    ).status == "executed"
    assert len(connector.calls) == 1


def test_changed_interview_invalidates_fixed_payload_approval(tmp_path) -> None:
    service, store, _, interviews, connector = build_service(tmp_path)
    proposal = service.prepare_interview_sync(
        user_id="u1", interview_round_id="interview-1", now=NOW,
    )
    interviews.interview = interviews.interview.model_copy(
        update={
            "scheduled_start": NOW + timedelta(days=4),
            "scheduled_end": NOW + timedelta(days=4, hours=1),
            "updated_at": NOW + timedelta(minutes=2),
        }
    )

    with pytest.raises(CalendarProposalConflictError, match="changed after approval"):
        service.execute_proposal(
            user_id="u1", proposal_id=proposal.id, now=NOW + timedelta(minutes=2),
        )

    assert connector.calls == []
    assert store.get_proposal(user_id="u1", proposal_id=proposal.id).status == "superseded"


def test_changed_execution_policy_invalidates_an_older_approval(tmp_path) -> None:
    service, store, _, interviews, connector = build_service(tmp_path)
    proposal = service.prepare_interview_sync(
        user_id="u1", interview_round_id="interview-1", now=NOW,
    )
    assert proposal.policy_epoch == 1
    upgraded = CalendarService(
        store,
        interviews,
        Applications(),
        Resolver(connector),
        policy_epoch=2,
    )

    with pytest.raises(CalendarProposalConflictError, match="policy changed"):
        upgraded.execute_proposal(
            user_id="u1", proposal_id=proposal.id,
            now=NOW + timedelta(minutes=1),
        )

    assert connector.calls == []
    assert store.get_proposal(
        user_id="u1", proposal_id=proposal.id
    ).status == "superseded"


def test_changed_policy_reconciles_a_prepared_action_but_never_reapplies_it(
    tmp_path,
) -> None:
    service, store, _, interviews, connector = build_service(tmp_path)
    proposal = service.prepare_interview_sync(
        user_id="u1", interview_round_id="interview-1", now=NOW,
    )

    def timeout_before_remote_write(**kwargs):
        raise CalendarConnectorError(
            "GOOGLE_CALENDAR_TRANSPORT_ERROR",
            "request outcome unavailable",
            outcome_unknown=True,
        )

    connector.apply = timeout_before_remote_write
    with pytest.raises(CalendarConnectorError):
        service.execute_proposal(
            user_id="u1",
            proposal_id=proposal.id,
            now=NOW + timedelta(minutes=1),
        )
    upgraded = CalendarService(
        store,
        interviews,
        Applications(),
        Resolver(connector),
        policy_epoch=2,
    )

    with pytest.raises(CalendarProposalConflictError, match="policy changed"):
        upgraded.execute_proposal(
            user_id="u1",
            proposal_id=proposal.id,
            now=NOW + timedelta(minutes=2),
        )

    assert connector.reconcile_calls[-1]["idempotency_key"] == proposal.id
    assert store.get_execution(
        user_id="u1", proposal_id=proposal.id
    ).status == "failed"


def test_expired_proposal_never_calls_external_calendar(tmp_path) -> None:
    service, store, _, _, connector = build_service(tmp_path)
    proposal = service.prepare_interview_sync(
        user_id="u1", interview_round_id="interview-1", now=NOW,
    )

    with pytest.raises(CalendarProposalConflictError, match="expired"):
        service.execute_proposal(
            user_id="u1", proposal_id=proposal.id, now=NOW + timedelta(minutes=16),
        )

    assert connector.calls == []
    assert store.get_proposal(user_id="u1", proposal_id=proposal.id).status == "expired"
