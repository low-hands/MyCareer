import json
from datetime import datetime, timezone

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
