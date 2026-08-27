from __future__ import annotations

import hashlib
import operator
from time import perf_counter_ns
from typing import Annotated, Any, Callable, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

from career_agent.agent.job_discovery_contracts import AgentWorker, JDAnalysis, JDComparison, JobDiscoveryRequest, SearchStrategy, TargetRoleProposal
from career_agent.connectors.boss_readonly import BossAdapterError, BossReadOnlyAdapter, SearchQuery
from career_agent.domain.job_discovery import JobDetail, Provenance, SearchResult, normalize_jd
from career_agent.agent.openai_compatible_client import AgentWorkerError
from career_agent.agent.job_discovery_policy import JobDiscoveryPolicy
from career_agent.harness.observability import InMemoryTraceRecorder, RunTrace, TraceRecorder


class JobDiscoveryState(TypedDict, total=False):
    request: JobDiscoveryRequest
    run_id: str
    trace_id: str
    target: TargetRoleProposal
    strategy: SearchStrategy
    phase: str
    results: tuple[SearchResult, ...]
    selected_result_refs: tuple[str, ...]
    details: dict[str, JobDetail]
    analyses: dict[str, JDAnalysis]
    detail_errors: dict[str, str]
    comparison: JDComparison
    error_code: str
    error_stage: str
    error_detail: str
    recovery_action: str
    recoverable: bool


class LangGraphJobDiscovery:
    def __init__(self, adapter: BossReadOnlyAdapter, worker: AgentWorker, *, checkpointer: Any | None = None, policy: JobDiscoveryPolicy | None = None, trace_recorder: TraceRecorder | None = None) -> None:
        self._adapter = adapter
        self._worker = worker
        self._policy = policy or JobDiscoveryPolicy.from_env()
        self._trace = trace_recorder or InMemoryTraceRecorder()
        graph = StateGraph(JobDiscoveryState)
        graph.add_node("search_strategy", lambda state: self._observe_node("search_strategy", self._strategy, state))
        graph.add_node("boss_search", lambda state: self._observe_node("boss_search", self._search, state))
        graph.add_node("await_selection", lambda state: self._observe_node("await_selection", self._await_selection, state))
        graph.add_node("await_recovery", lambda state: self._observe_node("await_recovery", self._await_recovery, state))
        graph.add_node("detail_and_analysis", lambda state: self._observe_node("detail_and_analysis", self._detail_and_analysis, state))
        graph.add_node("failed", lambda state: {})
        graph.add_edge(START, "search_strategy")
        graph.add_conditional_edges("search_strategy", self._after_strategy, {"boss_search": "boss_search", "failed": "failed"})
        graph.add_conditional_edges("boss_search", self._after_search, {"selection": "await_selection", "recovery": "await_recovery", "failed": "failed"})
        graph.add_conditional_edges("await_selection", self._after_selection, {"detail": "detail_and_analysis", "failed": "failed"})
        graph.add_edge("await_recovery", "boss_search")
        graph.add_edge("detail_and_analysis", END)
        graph.add_edge("failed", END)
        self._checkpointer = checkpointer or InMemorySaver()
        self._graph = graph.compile(checkpointer=self._checkpointer)

    @property
    def checkpointer(self) -> Any:
        """Exposed so a caller bounding its own caches can drop a run's steps too."""
        return self._checkpointer

    @staticmethod
    def _duration_ms(start_ns: int) -> int:
        return max(0, (perf_counter_ns() - start_ns) // 1_000_000)

    @staticmethod
    def _safe_ref(result_ref: str) -> str:
        return hashlib.sha256(result_ref.encode("utf-8")).hexdigest()[:12]

    def _observe_node(self, stage: str, handler: Callable[[JobDiscoveryState], dict[str, Any]], state: JobDiscoveryState) -> dict[str, Any]:
        run_id = state.get("run_id") or state["request"].conversation_id
        started_ns = perf_counter_ns()
        self._trace.record(run_id, "node_started", stage, outcome="started", details={"state_keys": sorted(state.keys())})
        try:
            result = handler(state)
        except Exception as error:
            event_type = "node_interrupted" if type(error).__name__ == "GraphInterrupt" else "node_failed"
            self._trace.record(run_id, event_type, stage, duration_ms=self._duration_ms(started_ns), outcome="interrupted" if event_type == "node_interrupted" else "failed", details={"exception_type": type(error).__name__})
            raise
        error_code = result.get("error_code")
        details = {"output_keys": sorted(result.keys())}
        for key, label in (("results", "result_count"), ("detail", "detail_count"), ("analysis", "analysis_count")):
            if key in result and result[key] is not None:
                details[label] = 1
        if "results" in result:
            details["result_count"] = len(result["results"])
        event_type = "node_failed" if error_code else "node_completed"
        self._trace.record(run_id, event_type, stage, duration_ms=self._duration_ms(started_ns), outcome="failed" if error_code else "succeeded", details=details, error_code=error_code, error_detail=result.get("error_detail"), recoverable=result.get("recoverable"))
        return result

    def _decide(self, *, run_id: str, stage: str, input: dict[str, Any], output_type: type[Any]) -> Any:
        for attempt in range(1, self._policy.retry_limit + 2):
            started_ns = perf_counter_ns()
            details = {"schema": output_type.__name__, "input_keys": sorted(input.keys())}
            if "candidates" in input:
                details["candidate_count"] = len(input["candidates"])
            if "allowed_result_refs" in input:
                details["allowed_ref_count"] = len(input["allowed_result_refs"])
            if isinstance(input.get("result_ref"), str):
                details["result_ref_hash"] = self._safe_ref(input["result_ref"])
            self._trace.record(run_id, "model_attempt", stage, attempt=attempt, outcome="started", details=details)
            try:
                worker = self._worker
                result = worker.decide(stage=stage, input=input, output_type=output_type)
            except AgentWorkerError as error:
                self._trace.record(run_id, "model_failed", stage, attempt=attempt, duration_ms=self._duration_ms(started_ns), outcome="failed", details=details, error_code=error.code, error_detail=error.detail, recoverable=error.retryable)
                if error.code in {"AGENT_WORKER_EMPTY_RESPONSE", "AGENT_WORKER_INVALID_RESPONSE"} and attempt <= self._policy.retry_limit:
                    continue
                raise
            output_details = {"schema": output_type.__name__}
            for field, label in (("queries", "query_count"), ("selections", "selection_count"), ("items", "item_count")):
                value = getattr(result, field, None)
                if value is not None:
                    output_details[label] = len(value)
            self._trace.record(run_id, "model_succeeded", stage, attempt=attempt, duration_ms=self._duration_ms(started_ns), outcome="succeeded", details=output_details)
            return result
        raise RuntimeError("Agent decision retry loop exited unexpectedly")

    def _execute(self, run_id: str, operation: Callable[[], JobDiscoveryState], *, resumed: bool = False) -> JobDiscoveryState:
        self._trace.record(run_id, "run_resumed" if resumed else "run_started", "job_discovery", outcome="started", details={"resumed": resumed})
        try:
            state = operation()
        except Exception as error:
            self._trace.record(run_id, "run_completed", "job_discovery", outcome="failed", details={"exception_type": type(error).__name__})
            raise
        if state.get("__interrupt__"):
            self._trace.record(run_id, "run_interrupted", "job_discovery", outcome="interrupted", details={"last_stage": state.get("error_stage")})
        else:
            self._trace.record(run_id, "run_completed", "job_discovery", outcome="failed" if state.get("error_code") else "succeeded", details={"phase": state.get("phase"), "result_count": len(state.get("results", ())), "detail_count": 1 if state.get("detail") else 0, "analysis_count": 1 if state.get("analysis") else 0}, error_code=state.get("error_code"), error_detail=state.get("error_detail"), recoverable=state.get("recoverable"))
        return state

    def trace(self, run_id: str) -> RunTrace:
        return self._trace.snapshot(run_id)

    def invoke(self, request: JobDiscoveryRequest, *, run_id: str | None = None) -> JobDiscoveryState:
        run_id = run_id or request.conversation_id
        return self._execute(run_id, lambda: self._graph.invoke({"request": request, "run_id": run_id, "trace_id": run_id, "phase": "searching"}, config={"configurable": {"thread_id": run_id}}))

    def resume(self, *, conversation_id: str, value: object = None) -> JobDiscoveryState:
        return self._execute(conversation_id, lambda: self._graph.invoke(Command(resume=value or {"retry": True}), config={"configurable": {"thread_id": conversation_id}}), resumed=True)

    def research_selected_many(self, *, request: JobDiscoveryRequest, results: tuple[SearchResult, ...], result_refs: tuple[str, ...], run_id: str) -> JobDiscoveryState:
        target = TargetRoleProposal(id=f"role_{request.target_role.casefold().replace(' ', '_')}", title=request.target_role, source="explicit", rationale="The user explicitly stated this target role.")
        initial: JobDiscoveryState = {"request": request, "run_id": run_id, "trace_id": run_id, "phase": "researching", "target": target, "results": results, "selected_result_refs": result_refs}
        return self._execute(run_id, lambda: {**initial, **self._observe_node("detail_and_analysis", self._detail_and_analysis, initial)}, resumed=True)

    def research_selected(self, *, request: JobDiscoveryRequest, results: tuple[SearchResult, ...], result_ref: str, run_id: str) -> JobDiscoveryState:
        return self.research_selected_many(request=request, results=results, result_refs=(result_ref,), run_id=run_id)

    def analyze_user_provided_jd(self, *, request: JobDiscoveryRequest, result: SearchResult, jd_text: str, run_id: str) -> JobDiscoveryState:
        description = normalize_jd(jd_text)
        if not description:
            raise ValueError("JD text must contain non-whitespace content.")
        target = TargetRoleProposal(id=f"role_{request.target_role.casefold().replace(' ', '_')}", title=request.target_role, source="explicit", rationale="The user explicitly stated this target role.")
        detail = JobDetail(
            source_name=result.source_name,
            source_job_id=result.source_job_id,
            security_id=result.security_id,
            source_url=result.source_url,
            title=result.title,
            company_name=result.company_name,
            description=description,
            captured_at=result.captured_at,
            provenance=Provenance(source_name="user", captured_at=result.captured_at, operation="provided_jd", adapter_version="user-input-v1", source_job_id=result.source_job_id),
            city=result.city,
            salary=result.salary,
            experience=result.experience,
            education=result.education,
            labels=result.labels,
            content_origin="user_provided",
        )
        initial: JobDiscoveryState = {"request": request, "run_id": run_id, "trace_id": run_id, "phase": "researching", "target": target, "results": (result,), "selected_result_refs": (result.result_ref,)}
        analysis = self._analyze_detail(initial, result, detail)
        if isinstance(analysis, JDAnalysis):
            return self._execute(run_id, lambda: {**initial, "details": {result.result_ref: detail}, "analyses": {result.result_ref: analysis}, "detail_errors": {}, "phase": "analysis_ready"}, resumed=True)
        return self._execute(run_id, lambda: {**initial, "details": {}, "analyses": {}, "detail_errors": {result.result_ref: analysis.get("error_code", "JD_ANALYSIS_FAILED")}, "phase": "detail_unavailable"}, resumed=True)

    def state(self, run_id: str) -> JobDiscoveryState:
        return self._graph.get_state({"configurable": {"thread_id": run_id}}).values

    def _strategy(self, state: JobDiscoveryState) -> dict[str, Any]:
        request = state["request"]
        target = TargetRoleProposal(id=f"role_{request.target_role.casefold().replace(' ', '_')}", title=request.target_role, source="explicit", rationale="The user explicitly stated this target role.")
        try:
            strategy = self._decide(run_id=state.get("run_id") or state["request"].conversation_id, stage="search_strategy", input={"target_role": target, "resume_text": request.resume_text, "city": request.city, "salary": request.salary, "experience": request.experience, "education": request.education}, output_type=SearchStrategy)
        except AgentWorkerError as error:
            return {"target": target, "error_code": error.code, "error_stage": "search_strategy", "error_detail": error.detail, "recoverable": error.retryable}
        return {"target": target, "strategy": strategy}

    @staticmethod
    def _after_strategy(state: JobDiscoveryState) -> str:
        return "failed" if state.get("error_code") else "boss_search"

    @staticmethod
    def _matches_requested_city(result: SearchResult, city: str | None) -> bool:
        if not city:
            return True
        normalized_city = city.strip().removesuffix("市").casefold()
        return result.city is not None and result.city.strip().removesuffix("市").casefold() == normalized_city

    def _search(self, state: JobDiscoveryState) -> dict[str, Any]:
        strategy = state["strategy"]
        try:
            results = self._adapter.search(SearchQuery(query=strategy.queries[0].query, city=strategy.city, salary=strategy.salary, experience=strategy.experience, education=strategy.education))
        except BossAdapterError as error:
            return {"error_code": error.code, "error_stage": "boss_search", "recovery_action": error.recovery_action, "recoverable": error.recoverable}
        max_results = min(self._policy.max_search_results, 15)
        requested_city = state["request"].city
        filtered_results = tuple(result for result in results if self._matches_requested_city(result, requested_city))
        if not filtered_results:
            return {
                "results": (),
                "phase": "failed",
                "error_code": "NO_RESULTS",
                "error_stage": "boss_search",
                "error_detail": "No jobs matched the current search filters.",
                "recoverable": False,
            }
        return {"results": filtered_results[:max_results], "phase": "selection_required"}

    @staticmethod
    def _after_search(state: JobDiscoveryState) -> str:
        if not state.get("error_code"):
            return "selection"
        return "recovery" if state.get("recoverable") else "failed"

    def _await_recovery(self, state: JobDiscoveryState) -> dict[str, Any]:
        interrupt({"code": state.get("error_code"), "recovery_action": state.get("recovery_action")})
        return {"error_code": "", "recovery_action": "", "recoverable": False, "phase": "searching"}

    def _await_selection(self, state: JobDiscoveryState) -> dict[str, Any]:
        value = interrupt({"code": "SELECTION_REQUIRED", "selection_limit": 3})
        refs = value.get("result_refs") if isinstance(value, dict) else None
        valid_refs = {result.result_ref for result in state.get("results", ())}
        if not isinstance(refs, (list, tuple)) or not 1 <= len(refs) <= 3 or len(set(refs)) != len(refs) or any(not isinstance(ref, str) or ref not in valid_refs for ref in refs):
            return {"error_code": "INVALID_RESULT_SELECTION", "error_stage": "selection", "error_detail": "Select one to three unique results returned by this search.", "recoverable": True, "phase": "selection_required"}
        return {"selected_result_refs": tuple(refs), "phase": "researching"}

    @staticmethod
    def _after_selection(state: JobDiscoveryState) -> str:
        return "failed" if state.get("error_code") else "detail"

    def _detail_and_analysis(self, state: JobDiscoveryState) -> dict[str, Any]:
        selected = {result.result_ref: result for result in state["results"]}
        details: dict[str, JobDetail] = {}
        analyses: dict[str, JDAnalysis] = {}
        errors: dict[str, str] = {}
        for result_ref in state["selected_result_refs"]:
            result = selected[result_ref]
            if not result.security_id:
                errors[result_ref] = "MISSING_SECURITY_ID"
                continue
            detail: JobDetail | None = None
            for attempt in range(1, self._policy.max_detail_attempts + 1):
                try:
                    detail = self._adapter.detail(result.security_id, result.source_job_id)
                    break
                except BossAdapterError as error:
                    transient = error.code in {"TIMEOUT", "NETWORK_ERROR", "UNKNOWN", "CLI_ERROR"}
                    if not transient or attempt == self._policy.max_detail_attempts:
                        errors[result_ref] = error.code
                        break
            if detail is None:
                continue
            details[result_ref] = detail
            analysis = self._analyze_detail(state, result, detail)
            if isinstance(analysis, JDAnalysis):
                analyses[result_ref] = analysis
            else:
                errors[result_ref] = analysis.get("error_code", "JD_ANALYSIS_FAILED")
        if not analyses:
            phase = "detail_unavailable" if errors else "failed"
            return {"details": details, "analyses": analyses, "detail_errors": errors, "phase": phase, "error_code": next(iter(errors.values()), None), "recoverable": False}
        phase = "partial_analysis_ready" if errors else "analysis_ready"
        return {"details": details, "analyses": analyses, "detail_errors": errors, "phase": phase}

    def _analyze_detail(self, state: JobDiscoveryState, result: SearchResult, detail: JobDetail) -> JDAnalysis | dict[str, Any]:
        try:
            analysis = self._decide(run_id=state.get("run_id") or state["request"].conversation_id, stage="jd_analysis", input={"result_ref": result.result_ref, "jd_text": detail.description}, output_type=JDAnalysis)
        except AgentWorkerError as error:
            return {"error_code": error.code}
        if analysis.result_ref != result.result_ref:
            return {"error_code": "ANALYSIS_RESULT_REF_MISMATCH"}
        return analysis

    @staticmethod
    def _compare_analyses(selected_refs: tuple[str, ...], analyses: dict[str, JDAnalysis]) -> JDComparison:
        ordered_analyses = tuple(analyses[ref] for ref in selected_refs if ref in analyses)
        skill_sets = [set(analysis.required_skills) for analysis in ordered_analyses]
        common = tuple(sorted(set.intersection(*skill_sets))) if skill_sets else ()
        differences = tuple(
            f"岗位 {index}: {', '.join(analysis.required_skills) or '暂无明确必备技能'}"
            for index, analysis in enumerate(ordered_analyses, start=1)
        )
        return JDComparison(common_requirements=common, key_differences=differences)
