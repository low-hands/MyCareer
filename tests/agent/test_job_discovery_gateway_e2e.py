import pytest
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from career_agent.agent.main_agent_contracts import ConversationTaskState
from career_agent.agent.job_discovery_contracts import JobDiscoveryRequest
from career_agent.agent.job_discovery_gateway import JobDiscoveryGateway
from career_agent.agent.openai_compatible_agent_worker import OpenAICompatibleAgentWorker
from career_agent.agent.openai_compatible_client import OpenAICompatibleAgentConfig
from career_agent.connectors.boss_readonly import BossReadOnlyAdapter
from career_agent.services.job_discovery import JobDiscoveryService
from career_agent.storage.memory import InMemoryJobRepository
from career_agent.storage.jobs import SQLiteJobPostingRepository
from career_agent.storage.runs import JobDiscoveryRunStore

NOW = datetime(2026, 8, 18, tzinfo=timezone.utc)


def envelope(data, command):
    return json.dumps({"ok": True, "schema_version": "1.0", "command": command, "data": data, "pagination": None, "error": None, "hints": {}})


class Transport:
    def __init__(self): self.calls = []
    def __call__(self, args):
        self.calls.append(tuple(args))
        if args[0] == "search":
            return envelope([{ "security_id": "security-1", "job_id": "job-1", "title": "AI Engineer", "company": "Acme", "city": "Shanghai", "salary": "30-50K", "skills": ["Python", "LLM"], "url": "https://www.zhipin.com/job_detail/abc?token=secret"}], "search")
        return envelope({"security_id": "security-1", "job_id": "job-1", "title": "AI Engineer", "company": "Acme", "description": "Build reliable LLM systems.", "city": "Shanghai", "salary": "30-50K", "skills": ["Python", "LLM"]}, "detail")


class FailingDetailTransport(Transport):
    def __call__(self, args):
        self.calls.append(tuple(args))
        if args[0] == "detail":
            return json.dumps({"ok": False, "schema_version": "1.0", "command": "detail", "data": None, "pagination": None, "error": {"code": "UNKNOWN", "message": "detail unavailable"}, "hints": {}})
        return super().__call__(args)


class EmptySearchTransport(Transport):
    def __call__(self, args):
        self.calls.append(tuple(args))
        if args[0] == "search":
            return envelope([], "search")
        return super().__call__(args)


class Completions:
    def __init__(self): self.requests = []
    def create(self, **kwargs):
        self.requests.append(kwargs)
        stage = kwargs["messages"][0]["content"]
        input_data = json.loads(kwargs["messages"][1]["content"])["input"]
        if "search_strategy" in stage:
            response = {"target_role": input_data["target_role"], "queries": [{"query": "AI Engineer", "rationale": "Matches the requested role."}]}
        else:
            response = {"result_ref": input_data["result_ref"], "job_summary": "Build reliable LLM systems.", "responsibilities": ["Build LLM systems"], "required_skills": ["Python"], "preferred_qualifications": ["LLM experience"], "clarification_questions": ["What experience level is required?"]}
        message = type("Message", (), {"content": json.dumps(response)})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class Client:
    def __init__(self):
        self.completions = Completions()
        self.chat = type("Chat", (), {"completions": self.completions})()


def test_gateway_returns_results_then_researches_user_selected_job():
    transport = Transport()
    client = Client()
    repository = InMemoryJobRepository()
    gateway = JobDiscoveryGateway(
        BossReadOnlyAdapter(transport, clock=lambda: NOW),
        OpenAICompatibleAgentWorker(OpenAICompatibleAgentConfig(endpoint="https://example.test/v1/chat/completions", api_key="test", model="test-model"), client=client),
        JobDiscoveryService(repository),
    )

    started = gateway.research(JobDiscoveryRequest(user_id="user-1", conversation_id="conversation-1", target_role="AI Engineer"))

    assert started.state == "selection_required"
    assert started.next_action == "select_result"
    assert started.items[0].result_ref == "boss:security-1"
    assert transport.calls == [("search", "AI Engineer", "--page", "1")]
    assert len(client.completions.requests) == 1

    selected = gateway.select(run_id=started.run_id, result_ref="boss:security-1")

    assert selected.state == "analysis_ready"
    assert selected.selected_result_ref == "boss:security-1"
    assert selected.detail.description == "Build reliable LLM systems."
    assert selected.analysis.job_summary == "Build reliable LLM systems."
    assert "Build reliable LLM systems." not in selected.trace.model_dump_json()
    assert transport.calls == [("search", "AI Engineer", "--page", "1"), ("detail", "security-1", "--job-id", "job-1")]
    assert len(client.completions.requests) == 2

    confirmation = gateway.request_waitlist_confirmation(run_id=selected.run_id, result_ref="boss:security-1")
    completed = gateway.confirm_waitlist(run_id=selected.run_id, confirmation_id=confirmation.confirmation_id)

    assert confirmation.state == "confirmation_required"
    assert completed.state == "waitlisted"
    assert len(repository.waitlist) == 1


def test_gateway_rejects_unknown_selection_without_detail_call():
    transport = Transport()
    client = Client()
    gateway = JobDiscoveryGateway(
        BossReadOnlyAdapter(transport, clock=lambda: NOW),
        OpenAICompatibleAgentWorker(OpenAICompatibleAgentConfig(endpoint="https://example.test/v1/chat/completions", api_key="test", model="test-model"), client=client),
        JobDiscoveryService(InMemoryJobRepository()),
    )

    started = gateway.research(JobDiscoveryRequest(user_id="user-1", conversation_id="conversation-2", target_role="AI Engineer"))
    rejected = gateway.select(run_id=started.run_id, result_ref="unknown")

    assert rejected.state == "selection_required"
    assert rejected.error_code == "INVALID_RESULT_REF"
    assert transport.calls == [("search", "AI Engineer", "--page", "1")]
    assert len(client.completions.requests) == 1


def test_gateway_select_survives_new_gateway_instance(tmp_path):
    transport = Transport()
    store_path = tmp_path / "runs.sqlite3"
    gateway_a = JobDiscoveryGateway(
        BossReadOnlyAdapter(transport, clock=lambda: NOW),
        OpenAICompatibleAgentWorker(OpenAICompatibleAgentConfig(endpoint="https://example.test/v1/chat/completions", api_key="test", model="test-model"), client=Client()),
        JobDiscoveryService(InMemoryJobRepository()),
        run_store=JobDiscoveryRunStore(store_path),
    )
    started = gateway_a.research(JobDiscoveryRequest(user_id="user-1", conversation_id="conversation-3", target_role="AI Engineer", resume_text="private resume"))

    client_b = Client()
    gateway_b = JobDiscoveryGateway(
        BossReadOnlyAdapter(transport, clock=lambda: NOW),
        OpenAICompatibleAgentWorker(OpenAICompatibleAgentConfig(endpoint="https://example.test/v1/chat/completions", api_key="test", model="test-model"), client=client_b),
        JobDiscoveryService(InMemoryJobRepository()),
        run_store=JobDiscoveryRunStore(store_path),
    )
    selected = gateway_b.select(run_id=started.run_id, result_ref="boss:security-1", user_id="user-1")

    assert selected.state == "analysis_ready"
    assert selected.detail.description == "Build reliable LLM systems."
    assert transport.calls == [("search", "AI Engineer", "--page", "1"), ("detail", "security-1", "--job-id", "job-1")]
    assert len(client_b.completions.requests) == 1


def test_gateway_persists_complete_jd_and_analysis_for_later_user_scoped_retrieval(tmp_path):
    transport = Transport()
    jobs_path = tmp_path / "jobs.sqlite3"
    jobs = SQLiteJobPostingRepository(jobs_path)
    gateway = JobDiscoveryGateway(
        BossReadOnlyAdapter(transport, clock=lambda: NOW),
        OpenAICompatibleAgentWorker(OpenAICompatibleAgentConfig(endpoint="https://example.test/v1/chat/completions", api_key="test", model="test-model"), client=Client()),
        JobDiscoveryService(InMemoryJobRepository()),
        job_repository=jobs,
    )

    started = gateway.research(JobDiscoveryRequest(user_id="user-1", conversation_id="conversation-persist-jd", target_role="AI Engineer"))
    gateway.select(run_id=started.run_id, result_ref="boss:security-1", user_id="user-1")

    rebuilt = SQLiteJobPostingRepository(jobs_path)
    stored = rebuilt.get_for_run(user_id="user-1", run_id=started.run_id, selection_index=1)
    assert stored is not None
    assert stored.snapshot.content == "Build reliable LLM systems."
    assert stored.analysis is not None
    assert stored.analysis.analyzer_version == "jd-analysis-v1"
    assert stored.analysis.analysis.job_summary == "Build reliable LLM systems."
    assert stored.analysis.analysis.required_skills == ("Python",)
    assert "boss:security-1" not in stored.analysis.model_dump_json()
    assert rebuilt.get_for_run(user_id="other", run_id=started.run_id, selection_index=1) is None


def test_gateway_advance_starts_then_selects_by_index():
    transport = Transport()
    gateway = JobDiscoveryGateway(
        BossReadOnlyAdapter(transport, clock=lambda: NOW),
        OpenAICompatibleAgentWorker(OpenAICompatibleAgentConfig(endpoint="https://example.test/v1/chat/completions", api_key="test", model="test-model"), client=Client()),
        JobDiscoveryService(InMemoryJobRepository()),
    )
    request = JobDiscoveryRequest(user_id="user-1", conversation_id="conversation-advance", target_role="AI Engineer")

    started = gateway.advance(user_id="user-1", conversation_id="conversation-advance", task=ConversationTaskState(), user_message="Find jobs", research_request=request)
    selected = gateway.advance(
        user_id="user-1",
        conversation_id="conversation-advance",
        task=ConversationTaskState(active_workflow="job_discovery", run_id=started.run_id, phase="selection_required"),
        user_message="Show the first one",
        selection_index=1,
    )

    assert started.state == "selection_required"
    assert selected.state == "analysis_ready"
    assert transport.calls == [("search", "AI Engineer", "--page", "1"), ("detail", "security-1", "--job-id", "job-1")]


def test_gateway_advance_selects_a_restored_run_without_checkpoint(tmp_path):
    transport = Transport()
    store_path = tmp_path / "runs.sqlite3"
    gateway_a = JobDiscoveryGateway(
        BossReadOnlyAdapter(transport, clock=lambda: NOW),
        OpenAICompatibleAgentWorker(OpenAICompatibleAgentConfig(endpoint="https://example.test/v1/chat/completions", api_key="test", model="test-model"), client=Client()),
        JobDiscoveryService(InMemoryJobRepository()),
        run_store=JobDiscoveryRunStore(store_path),
    )
    started = gateway_a.research(JobDiscoveryRequest(user_id="user-1", conversation_id="conversation-restored-advance", target_role="AI Engineer"))
    gateway_b = JobDiscoveryGateway(
        BossReadOnlyAdapter(transport, clock=lambda: NOW),
        OpenAICompatibleAgentWorker(OpenAICompatibleAgentConfig(endpoint="https://example.test/v1/chat/completions", api_key="test", model="test-model"), client=Client()),
        JobDiscoveryService(InMemoryJobRepository()),
        run_store=JobDiscoveryRunStore(store_path),
    )

    selected = gateway_b.advance(
        user_id="user-1",
        conversation_id="conversation-restored-advance",
        task=ConversationTaskState(active_workflow="job_discovery", run_id=started.run_id, phase="selection_required"),
        user_message="Show the first one.",
        selection_index=1,
    )

    assert selected.state == "analysis_ready"
    assert transport.calls == [("search", "AI Engineer", "--page", "1"), ("detail", "security-1", "--job-id", "job-1")]

    transport = Transport()
    gateway = JobDiscoveryGateway(
        BossReadOnlyAdapter(transport, clock=lambda: NOW),
        OpenAICompatibleAgentWorker(OpenAICompatibleAgentConfig(endpoint="https://example.test/v1/chat/completions", api_key="test", model="test-model"), client=Client()),
        JobDiscoveryService(InMemoryJobRepository()),
    )
    request = JobDiscoveryRequest(user_id="user-1", conversation_id="conversation-index", target_role="AI Engineer")
    started = gateway.advance(user_id="user-1", conversation_id="conversation-index", task=ConversationTaskState(), user_message="Find jobs", research_request=request)

    result = gateway.advance(
        user_id="user-1",
        conversation_id="conversation-index",
        task=ConversationTaskState(active_workflow="job_discovery", run_id=started.run_id, phase="selection_required"),
        user_message="Show the second one",
        selection_index=2,
    )

    assert result.state == "selection_required"
    assert transport.calls == [("search", "AI Engineer", "--page", "1")]

    transport = FailingDetailTransport()
    client = Client()
    gateway = JobDiscoveryGateway(
        BossReadOnlyAdapter(transport, clock=lambda: NOW),
        OpenAICompatibleAgentWorker(OpenAICompatibleAgentConfig(endpoint="https://example.test/v1/chat/completions", api_key="test", model="test-model"), client=client),
        JobDiscoveryService(InMemoryJobRepository()),
    )

    started = gateway.research(JobDiscoveryRequest(user_id="user-1", conversation_id="conversation-4", target_role="AI Engineer"))
    failed = gateway.select(run_id=started.run_id, result_ref="boss:security-1", user_id="user-1")

    assert failed.state == "detail_unavailable"
    assert failed.next_action == "provide_jd"
    assert failed.fallback_url == "https://www.zhipin.com/job_detail/abc"
    assert failed.manual_search_query == "Acme · AI Engineer · Shanghai"
    assert failed.error_code == "UNKNOWN"
    assert len(failed.items) == 1
    assert transport.calls.count(("detail", "security-1", "--job-id", "job-1")) == 3
    assert len(client.completions.requests) == 1

    repeated = gateway.select(run_id=started.run_id, result_ref="boss:security-1", user_id="user-1")

    assert repeated.error_code == "INVALID_RESULT_REF"
    assert transport.calls.count(("detail", "security-1", "--job-id", "job-1")) == 3

    analyzed = gateway.analyze_provided_jd(run_id=started.run_id, result_ref="boss:security-1", jd_text="User copied this JD text.", user_id="user-1")

    assert analyzed.state == "analysis_ready"
    assert analyzed.detail.content_origin == "user_provided"
    assert analyzed.analysis.job_summary == "Build reliable LLM systems."
    assert transport.calls.count(("detail", "security-1", "--job-id", "job-1")) == 3


def _worker():
    return OpenAICompatibleAgentWorker(OpenAICompatibleAgentConfig(endpoint="https://example.test/v1/chat/completions", api_key="test", model="test-model"), client=Client())


def test_evicted_run_is_restored_from_the_run_store_on_next_touch(tmp_path):
    """The cap bounds memory without ending a run the user can still resume."""
    gateway = JobDiscoveryGateway(
        BossReadOnlyAdapter(Transport(), clock=lambda: NOW),
        _worker(),
        JobDiscoveryService(InMemoryJobRepository()),
        run_store=JobDiscoveryRunStore(tmp_path / "runs.sqlite3"),
        max_cached_runs=2,
    )
    first = gateway.research(JobDiscoveryRequest(user_id="user-1", conversation_id="conversation-evict-1", target_role="AI Engineer"))
    for index in range(2, 5):
        gateway.research(JobDiscoveryRequest(user_id="user-1", conversation_id=f"conversation-evict-{index}", target_role="AI Engineer"))

    assert len(gateway._states) == 2
    assert first.run_id not in gateway._states

    revived = gateway.select(run_id=first.run_id, result_ref="boss:security-1", user_id="user-1")
    assert revived.state == "analysis_ready"
    assert revived.detail.description == "Build reliable LLM systems."


def test_an_analyzed_run_is_never_evicted_because_the_store_cannot_rebuild_it(tmp_path):
    """The record holds results and phase, not details or analyses.

    Evicting here would restore a run still claiming ``analysis_ready`` with the
    payload behind it empty, and a later waitlist confirmation would fail on
    state the caller was told existed.
    """
    gateway = JobDiscoveryGateway(
        BossReadOnlyAdapter(Transport(), clock=lambda: NOW),
        _worker(),
        JobDiscoveryService(InMemoryJobRepository()),
        run_store=JobDiscoveryRunStore(tmp_path / "runs.sqlite3"),
        max_cached_runs=1,
    )
    analyzed = gateway.research(JobDiscoveryRequest(user_id="user-1", conversation_id="conversation-analyzed", target_role="AI Engineer"))
    gateway.select(run_id=analyzed.run_id, result_ref="boss:security-1", user_id="user-1")
    assert gateway._states[analyzed.run_id]["phase"] == "analysis_ready"

    for index in range(3):
        gateway.research(JobDiscoveryRequest(user_id="user-1", conversation_id=f"conversation-pressure-{index}", target_role="AI Engineer"))

    assert analyzed.run_id in gateway._states
    still_there = gateway.request_waitlist_confirmation(run_id=analyzed.run_id, result_ref="boss:security-1")
    assert still_there.state == "confirmation_required"


def test_a_run_holding_a_pending_confirmation_is_never_evicted(tmp_path):
    """The pending record lives in the promotion facade, not the run store.

    Dropping our side of the mapping would leave an approval the user can still
    answer and we can no longer attribute to a run.
    """
    gateway = JobDiscoveryGateway(
        BossReadOnlyAdapter(Transport(), clock=lambda: NOW),
        _worker(),
        JobDiscoveryService(InMemoryJobRepository()),
        run_store=JobDiscoveryRunStore(tmp_path / "runs.sqlite3"),
        max_cached_runs=1,
    )
    owner = gateway.research(JobDiscoveryRequest(user_id="user-1", conversation_id="conversation-pending", target_role="AI Engineer"))
    gateway.select(run_id=owner.run_id, result_ref="boss:security-1", user_id="user-1")
    pending = gateway.request_waitlist_confirmation(run_id=owner.run_id, result_ref="boss:security-1")
    confirmation_id = pending.confirmation_id

    for index in range(3):
        gateway.research(JobDiscoveryRequest(user_id="user-1", conversation_id=f"conversation-noise-{index}", target_role="AI Engineer"))

    assert gateway._confirmation_runs.get(confirmation_id) == owner.run_id


def test_a_gateway_without_a_run_store_never_evicts():
    """With no durable record the cache *is* the run, so the cap must not apply."""
    gateway = JobDiscoveryGateway(
        BossReadOnlyAdapter(Transport(), clock=lambda: NOW),
        _worker(),
        JobDiscoveryService(InMemoryJobRepository()),
        max_cached_runs=2,
    )
    run_ids = [
        gateway.research(JobDiscoveryRequest(user_id="user-1", conversation_id=f"conversation-keep-{index}", target_role="AI Engineer")).run_id
        for index in range(4)
    ]

    assert len(gateway._states) == 4
    assert all(run_id in gateway._states for run_id in run_ids)


def test_eviction_drops_the_request_and_the_state_together(tmp_path):
    """A run_id left in one cache but not the other resumes down the wrong path."""
    gateway = JobDiscoveryGateway(
        BossReadOnlyAdapter(Transport(), clock=lambda: NOW),
        _worker(),
        JobDiscoveryService(InMemoryJobRepository()),
        run_store=JobDiscoveryRunStore(tmp_path / "runs.sqlite3"),
        max_cached_runs=1,
    )
    first = gateway.research(JobDiscoveryRequest(user_id="user-1", conversation_id="conversation-pair-1", target_role="AI Engineer"))
    gateway.research(JobDiscoveryRequest(user_id="user-1", conversation_id="conversation-pair-2", target_role="AI Engineer"))

    assert first.run_id not in gateway._states
    assert first.run_id not in gateway._requests
    assert first.run_id not in gateway._restored_without_checkpoint
    # The state dicts are not the only per-run memory the cap has to release.
    assert gateway._trace.snapshot(first.run_id).events == ()


def test_per_run_memory_stays_flat_as_runs_accumulate(tmp_path):
    """The cap has to bound every per-run store, not just the two dicts.

    ``InMemorySaver`` keeps three keyed collections and ``InMemoryTraceRecorder``
    a fourth. Asserting a run_id is absent proves only that one key went; it says
    nothing about growth. Measure the totals instead.
    """
    gateway = JobDiscoveryGateway(
        BossReadOnlyAdapter(Transport(), clock=lambda: NOW),
        _worker(),
        JobDiscoveryService(InMemoryJobRepository()),
        run_store=JobDiscoveryRunStore(tmp_path / "runs.sqlite3"),
        max_cached_runs=2,
    )
    checkpointer = gateway._graph.checkpointer

    def footprint() -> tuple[int, ...]:
        return (
            len(gateway._states),
            len(gateway._requests),
            len(gateway._persisted_runs),
            len(gateway._trace._events),
            len(checkpointer.storage),
            len(checkpointer.blobs),
            len(checkpointer.writes),
        )

    measurements = {}
    for index in range(12):
        gateway.research(
            JobDiscoveryRequest(
                user_id="user-1",
                conversation_id=f"conversation-flat-{index}",
                target_role="AI Engineer",
            )
        )
        measurements[index + 1] = footprint()

    assert measurements[12] == measurements[4], measurements
    # Nothing survives that does not belong to a still-cached run.
    cached = set(gateway._states)
    assert {key[0] for key in checkpointer.blobs} <= cached
    assert {key[0] for key in checkpointer.writes} <= cached
    assert set(checkpointer.storage) <= cached


def test_the_cap_must_leave_room_for_the_run_being_served():
    with pytest.raises(ValueError, match="at least 1"):
        JobDiscoveryGateway(
            BossReadOnlyAdapter(Transport(), clock=lambda: NOW),
            _worker(),
            JobDiscoveryService(InMemoryJobRepository()),
            max_cached_runs=0,
        )


def test_an_empty_search_becomes_a_persisted_failure_before_eviction(tmp_path):
    store = JobDiscoveryRunStore(tmp_path / "runs.sqlite3")
    gateway = JobDiscoveryGateway(
        BossReadOnlyAdapter(EmptySearchTransport(), clock=lambda: NOW),
        _worker(),
        JobDiscoveryService(InMemoryJobRepository()),
        run_store=store,
        max_cached_runs=1,
    )
    first = gateway.research(
        JobDiscoveryRequest(
            user_id="user-1",
            conversation_id="conversation-empty-1",
            target_role="AI Engineer",
        )
    )
    gateway.research(
        JobDiscoveryRequest(
            user_id="user-1",
            conversation_id="conversation-empty-2",
            target_role="AI Engineer",
        )
    )

    assert first.run_id not in gateway._states
    stored = store.get(first.run_id)
    assert stored is not None
    assert stored.phase == "failed"
    assert stored.error_code == "NO_RESULTS"
    restored = gateway.status(run_id=first.run_id)
    assert restored.state == "failed"
    assert restored.error_code == "NO_RESULTS"
    assert restored.items == ()


def test_the_evictable_phase_list_stays_a_whitelist_of_known_safe_phases():
    """A phase added later must default to un-evictable, not to evictable.

    The graph emits `partial_analysis_ready` too, and it holds the same
    unpersisted `details`/`analyses` that make `analysis_ready` unsafe to drop.
    Naming the safe phases rather than the unsafe ones is what makes forgetting to
    update this list a memory cost instead of a correctness bug.
    """
    source = Path("src/career_agent/agent/job_discovery_graph.py").read_text()
    # Two spellings in the source: the literal in a returned dict, and the local
    # `phase = "..."` that a conditional assigns before returning it.
    emitted = set(re.findall(r'"phase": "([a-z_]+)"', source)) | set(
        re.findall(r'phase = "([a-z_]+)"', source)
    )
    assert {"analysis_ready", "partial_analysis_ready", "detail_unavailable"} <= emitted
    assert JobDiscoveryGateway._REBUILDABLE_PHASES <= emitted
    # Anything holding post-fetch payload the run store does not persist is out.
    assert JobDiscoveryGateway._REBUILDABLE_PHASES == {"selection_required", "failed"}
