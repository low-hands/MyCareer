from datetime import datetime, timezone

from career_agent.agent.job_discovery_graph import LangGraphJobDiscovery
from career_agent.agent.job_discovery_policy import JobDiscoveryPolicy
from career_agent.agent.job_discovery_contracts import JDAnalysis, JobDiscoveryRequest, QueryProposal, SearchStrategy, TargetRoleProposal
from career_agent.connectors.boss_readonly import AuthStatus, BossAdapterError, SearchQuery
from career_agent.domain.job_discovery import JobDetail, Provenance, SearchResult

NOW = datetime(2026, 8, 17, tzinfo=timezone.utc)


def result(ref: str, index: int = 0) -> SearchResult:
    return SearchResult(result_ref=ref, source_name="boss", source_job_id=f"j{index}", security_id=f"s{index}", title="LLM Engineer", company_name="Acme", captured_at=NOW, provenance=Provenance(source_name="boss", captured_at=NOW, operation="search", adapter_version="test"))


class Adapter:
    def status(self): return AuthStatus(state="active")
    def search(self, query: SearchQuery): return (result("a"),)
    def detail(self, security_id, job_id=None): return JobDetail(source_name="boss", source_job_id=job_id, security_id=security_id, title="LLM Engineer", company_name="Acme", description="Build LLM systems.", captured_at=NOW, provenance=Provenance(source_name="boss", captured_at=NOW, operation="detail", adapter_version="test"))


class Worker:
    def decide(self, *, stage, input, output_type):
        if stage == "search_strategy":
            return SearchStrategy(target_role=input["target_role"], queries=(QueryProposal(query="AI Engineer", rationale="target"),))
        return JDAnalysis(result_ref=input["result_ref"], job_summary="Build LLM systems.")


class RecordingWorker(Worker):
    def __init__(self):
        self.inputs = []

    def decide(self, *, stage, input, output_type):
        self.inputs.append((stage, input))
        return super().decide(stage=stage, input=input, output_type=output_type)


class MultiSelectAdapter(Adapter):
    def __init__(self):
        self.detail_calls = []

    def search(self, query: SearchQuery):
        return result("a"), result("b", 1), result("c", 2)

    def detail(self, security_id, job_id=None):
        self.detail_calls.append(security_id)
        return super().detail(security_id, job_id)


class ManyResultAdapter(Adapter):
    def search(self, query: SearchQuery):
        return tuple(result(f"r{i}", i) for i in range(16))


class CityMixAdapter(Adapter):
    def search(self, query: SearchQuery):
        shanghai = result("shanghai").model_copy(update={"city": "上海"})
        shenzhen = result("shenzhen", 1).model_copy(update={"city": "深圳"})
        return shanghai, shenzhen


class RecoveringAdapter(Adapter):
    def __init__(self): self.calls = 0
    def search(self, query: SearchQuery):
        self.calls += 1
        if self.calls == 1:
            raise BossAdapterError("AUTH_EXPIRED", "expired", recoverable=True, recovery_action="Log in again.")
        return super().search(query)


class RetryingDetailAdapter(Adapter):
    def __init__(self, code: str, failures: int):
        self.code = code
        self.failures = failures
        self.detail_calls = 0

    def detail(self, security_id, job_id=None):
        self.detail_calls += 1
        if self.detail_calls <= self.failures:
            raise BossAdapterError(self.code, "temporary detail failure", recoverable=True)
        return super().detail(security_id, job_id)


def test_graph_returns_search_results_and_waits_for_selection():
    graph = LangGraphJobDiscovery(Adapter(), Worker())

    state = graph.invoke(JobDiscoveryRequest(user_id="u", conversation_id="select", target_role="AI Engineer"))

    assert "__interrupt__" in state
    assert state["phase"] == "selection_required"
    assert len(state["results"]) == 1
    assert "detail" not in state
    assert not any(event.stage == "jd_analysis" for event in graph.trace("select").events)


def test_graph_resumes_selected_result_and_runs_one_detail_analysis():
    graph = LangGraphJobDiscovery(Adapter(), Worker())
    graph.invoke(JobDiscoveryRequest(user_id="u", conversation_id="research", target_role="AI Engineer"))

    state = graph.resume(conversation_id="research", value={"result_refs": ["a"]})

    assert state["phase"] == "analysis_ready"
    assert state["selected_result_refs"] == ("a",)
    assert state["details"]["a"].title == "LLM Engineer"
    assert state["analyses"]["a"].job_summary == "Build LLM systems."


def test_graph_sends_only_jd_text_and_result_ref_to_analysis_worker():
    worker = RecordingWorker()
    graph = LangGraphJobDiscovery(Adapter(), worker)
    graph.invoke(JobDiscoveryRequest(user_id="u", conversation_id="jd-input", target_role="AI Engineer"))

    graph.resume(conversation_id="jd-input", value={"result_refs": ["a"]})

    jd_inputs = [input for stage, input in worker.inputs if stage == "jd_analysis"]
    assert jd_inputs == [{"result_ref": "a", "jd_text": "Build LLM systems."}]


def test_graph_analyzes_up_to_three_selected_results_in_order():
    adapter = MultiSelectAdapter()
    graph = LangGraphJobDiscovery(adapter, Worker())
    graph.invoke(JobDiscoveryRequest(user_id="u", conversation_id="multi", target_role="AI Engineer"))

    state = graph.resume(conversation_id="multi", value={"result_refs": ["c", "a", "b"]})

    assert state["phase"] == "analysis_ready"
    assert state["selected_result_refs"] == ("c", "a", "b")
    assert list(state["analyses"]) == ["c", "a", "b"]
    assert adapter.detail_calls == ["s2", "s0", "s1"]
    assert "comparison" not in state


def test_graph_rejects_unknown_selection_without_detail_fetch():
    adapter = Adapter()
    graph = LangGraphJobDiscovery(adapter, Worker())
    graph.invoke(JobDiscoveryRequest(user_id="u", conversation_id="invalid-select", target_role="AI Engineer"))

    state = graph.resume(conversation_id="invalid-select", value={"result_refs": ["unknown"]})

    assert state["error_code"] == "INVALID_RESULT_SELECTION"
    assert state["error_stage"] == "selection"
    assert state["phase"] == "selection_required"
    assert not any(event.stage == "boss_detail" for event in graph.trace("invalid-select").events)


def test_graph_caps_search_results_at_fifteen():
    graph = LangGraphJobDiscovery(ManyResultAdapter(), Worker(), policy=JobDiscoveryPolicy(max_search_results=20))

    state = graph.invoke(JobDiscoveryRequest(user_id="u", conversation_id="bounded", target_role="AI Engineer"))

    assert len(state["results"]) == 15
    assert state["phase"] == "selection_required"


def test_graph_filters_out_results_outside_requested_city():
    graph = LangGraphJobDiscovery(CityMixAdapter(), Worker())

    state = graph.invoke(JobDiscoveryRequest(user_id="u", conversation_id="city", target_role="AI Engineer", city="上海"))

    assert [candidate.result_ref for candidate in state["results"]] == ["shanghai"]


def test_graph_interrupts_and_resumes_recoverable_search():
    graph = LangGraphJobDiscovery(RecoveringAdapter(), Worker())
    request = JobDiscoveryRequest(user_id="u", conversation_id="recover", target_role="AI Engineer")

    interrupted = graph.invoke(request)
    resumed = graph.resume(conversation_id="recover", value={"retry": True})

    assert "__interrupt__" in interrupted
    assert resumed["phase"] == "selection_required"
    assert resumed["results"][0].result_ref == "a"
    assert any(event.event_type == "run_interrupted" for event in graph.trace("recover").events)


def test_graph_retries_transient_detail_failures_up_to_three_total_attempts():
    adapter = RetryingDetailAdapter("UNKNOWN", failures=2)
    graph = LangGraphJobDiscovery(adapter, Worker())
    graph.invoke(JobDiscoveryRequest(user_id="u", conversation_id="detail-retry-success", target_role="AI Engineer"))

    state = graph.resume(conversation_id="detail-retry-success", value={"result_refs": ["a"]})

    assert adapter.detail_calls == 3
    assert state["phase"] == "analysis_ready"
    assert state["analyses"]["a"].job_summary == "Build LLM systems."


def test_graph_stops_after_three_transient_detail_failures():
    adapter = RetryingDetailAdapter("UNKNOWN", failures=3)
    graph = LangGraphJobDiscovery(adapter, Worker())
    graph.invoke(JobDiscoveryRequest(user_id="u", conversation_id="detail-retry-fail", target_role="AI Engineer"))

    state = graph.resume(conversation_id="detail-retry-fail", value={"result_refs": ["a"]})

    assert adapter.detail_calls == 3
    assert state["phase"] == "detail_unavailable"
    assert state["error_code"] == "UNKNOWN"
    assert state["recoverable"] is False
    assert "analysis" not in state


def test_graph_does_not_retry_non_transient_detail_failures():
    adapter = RetryingDetailAdapter("ACCOUNT_RISK", failures=3)
    graph = LangGraphJobDiscovery(adapter, Worker())
    graph.invoke(JobDiscoveryRequest(user_id="u", conversation_id="detail-no-retry", target_role="AI Engineer"))

    state = graph.resume(conversation_id="detail-no-retry", value={"result_refs": ["a"]})

    assert adapter.detail_calls == 1
    assert state["phase"] == "detail_unavailable"
    assert state["error_code"] == "ACCOUNT_RISK"
