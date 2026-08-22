from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, ConfigDict, Field

from career_agent.agent.main_agent_contracts import ConversationTaskState
from career_agent.agent.job_discovery_contracts import JDAnalysis, JDComparison, JobDiscoveryRecommendation, JobDiscoveryRequest, PendingPromotionConfirmation, RecommendationItem, TargetRoleProposal
from career_agent.agent.job_discovery_graph import JobDiscoveryState, LangGraphJobDiscovery
from career_agent.agent.job_discovery_promotion import JobDiscoveryPromotionFacade
from career_agent.connectors.boss_readonly import BossReadOnlyAdapter
from career_agent.domain.job_discovery import JobDetail, SearchResult, new_id
from career_agent.harness.observability import InMemoryTraceRecorder, RunTrace, TraceRecorder
from career_agent.services.job_discovery import JobDiscoveryService, PromotionResult, PromotionSuccess
from career_agent.storage.runs import JobDiscoveryRunStore


class GatewayJobItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    result_ref: str
    title: str
    company_name: str
    city: str | None = None
    salary: str | None = None
    source_url: str | None = None
    rationale: str | None = None
    cautions: tuple[str, ...] = ()


class GatewayJobFallback(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    selection_index: int
    title: str
    company_name: str
    city: str | None = None
    error_code: str
    fallback_url: str | None = None
    manual_search_query: str | None = None


class JobDiscoveryGatewayResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    state: str
    message: str
    items: tuple[GatewayJobItem, ...] = ()
    next_action: str | None = None
    confirmation_id: str | None = None
    waitlist_item_id: str | None = None
    recovery_action: str | None = None
    error_code: str | None = None
    error_stage: str | None = None
    error_detail: str | None = None
    selected_result_ref: str | None = None
    selected_selection_indices: tuple[int, ...] = ()
    detail: JobDetail | None = None
    analysis: JDAnalysis | None = None
    analysis_items: tuple[JDAnalysis, ...] = ()
    analysis_selection_indices: tuple[int, ...] = ()
    comparison: JDComparison | None = None
    fallback_result_refs: tuple[str, ...] = ()
    fallbacks: tuple[GatewayJobFallback, ...] = ()
    fallback_url: str | None = None
    manual_search_query: str | None = None
    trace: RunTrace | None = None


class JobDiscoveryGateway:
    """Business-facing capability boundary for the main Career Agent."""

    def __init__(self, adapter: BossReadOnlyAdapter, worker: Any, promotion_service: JobDiscoveryService, *, checkpointer: Any | None = None, trace_recorder: TraceRecorder | None = None, run_store: JobDiscoveryRunStore | None = None) -> None:
        self._trace = trace_recorder or InMemoryTraceRecorder()
        self._run_store = run_store
        self._graph = LangGraphJobDiscovery(adapter, worker, checkpointer=checkpointer or InMemorySaver(), trace_recorder=self._trace)
        self._promotion = JobDiscoveryPromotionFacade(promotion_service)
        self._requests: dict[str, JobDiscoveryRequest] = {}
        self._states: dict[str, JobDiscoveryState] = {}
        self._restored_without_checkpoint: set[str] = set()
        self._confirmation_runs: dict[str, str] = {}

    def research(self, request: JobDiscoveryRequest) -> JobDiscoveryGatewayResult:
        run_id = new_id("job_discovery")
        state = self._graph.invoke(request, run_id=run_id)
        self._requests[run_id] = request
        self._states[run_id] = state
        self._restored_without_checkpoint.discard(run_id)
        self._persist(run_id, state)
        return self._project(run_id, state)

    def start(self, request: JobDiscoveryRequest) -> JobDiscoveryGatewayResult:
        return self.research(request)

    def advance(
        self,
        *,
        user_id: str,
        conversation_id: str,
        task: ConversationTaskState,
        user_message: str,
        selection_indices: tuple[int, ...] = (),
        selection_index: int | None = None,
        jd_selection_index: int | None = None,
        research_request: JobDiscoveryRequest | None = None,
    ) -> JobDiscoveryGatewayResult:
        if task.active_workflow != "job_discovery" or task.run_id is None:
            if research_request is None:
                return JobDiscoveryGatewayResult(run_id="", state="waiting_user", message="Choose a target role before starting job discovery.", next_action="provide_target_role")
            return self.research(research_request)
        if task.run_id not in self._requests:
            self._restore(task.run_id)
        request = self._request(task.run_id)
        if request.user_id != user_id or request.conversation_id != conversation_id:
            return JobDiscoveryGatewayResult(run_id=task.run_id, state="failed", message="This Job Discovery run does not belong to the active session.", error_code="RUN_CONTEXT_MISMATCH", error_stage="advance", trace=self._graph.trace(task.run_id))
        state = self._states[task.run_id]
        phase = state.get("phase")
        if task.phase != phase:
            return self._project(task.run_id, state)
        if phase == "selection_required":
            selection_indices = selection_indices or ((selection_index,) if selection_index else ())
            if not 1 <= len(selection_indices) <= 3 or len(set(selection_indices)) != len(selection_indices) or any(index > len(state.get("results", ())) for index in selection_indices):
                return self._project(task.run_id, state)
            result_refs = tuple(state["results"][index - 1].result_ref for index in selection_indices)
            if task.run_id in self._restored_without_checkpoint:
                state = self._graph.research_selected_many(request=request, results=tuple(state["results"]), result_refs=result_refs, run_id=task.run_id)
            else:
                state = self._graph.resume(conversation_id=task.run_id, value={"result_refs": result_refs})
            self._states[task.run_id] = state
            self._persist(task.run_id, state)
            return self._project(task.run_id, state)
        if phase == "detail_unavailable":
            if len(user_message.strip()) < 40:
                return self._project(task.run_id, state)
            selected = state.get("selected_result_ref")
            if not isinstance(selected, str):
                return self._project(task.run_id, state)
            return self.analyze_provided_jd(run_id=task.run_id, result_ref=selected, jd_text=user_message, user_id=user_id)
        if phase == "waiting_user":
            return self.resume(run_id=task.run_id)
        if phase in {"analysis_ready", "failed"} and research_request is not None:
            return self.research(research_request)
        return self._project(task.run_id, state)

    def select(self, *, run_id: str, result_ref: str, user_id: str | None = None) -> JobDiscoveryGatewayResult:
        was_in_memory = run_id in self._states and run_id not in self._restored_without_checkpoint
        if run_id not in self._states:
            self._restore(run_id)
        request = self._request(run_id)
        if user_id is not None and user_id != request.user_id:
            return JobDiscoveryGatewayResult(run_id=run_id, state="failed", message="The run does not belong to this user.", error_code="RUN_OWNERSHIP_MISMATCH", error_stage="selection", error_detail="Use the user_id that created this run.", trace=self._graph.trace(run_id))
        state = self._states.get(run_id) or self._graph.state(run_id)
        results = {result.result_ref: result for result in state.get("results", ())}
        if state.get("phase") != "selection_required" or result_ref not in results:
            return JobDiscoveryGatewayResult(
                run_id=run_id,
                state="selection_required",
                message="Select one result_ref returned by this search.",
                items=tuple(self._item(result) for result in results.values()),
                next_action="select_result",
                error_code="INVALID_RESULT_REF",
                error_stage="selection",
                error_detail="The result_ref must belong to the current search run.",
                trace=self._graph.trace(run_id),
            )
        if was_in_memory and state.get("phase") == "selection_required":
            state = self._graph.resume(conversation_id=run_id, value={"result_refs": [result_ref]})
        else:
            state = self._graph.research_selected(request=request, results=tuple(results.values()), result_ref=result_ref, run_id=run_id)
        self._states[run_id] = state
        self._restored_without_checkpoint.discard(run_id)
        self._persist(run_id, state)
        return self._project(run_id, state)

    def analyze_provided_jd(self, *, run_id: str, result_ref: str, jd_text: str, user_id: str | None = None) -> JobDiscoveryGatewayResult:
        if run_id not in self._states:
            self._restore(run_id)
        request = self._request(run_id)
        if user_id is not None and user_id != request.user_id:
            return JobDiscoveryGatewayResult(run_id=run_id, state="failed", message="The run does not belong to this user.", error_code="RUN_OWNERSHIP_MISMATCH", error_stage="provided_jd", error_detail="Use the user_id that created this run.", trace=self._graph.trace(run_id))
        state = self._states[run_id]
        selected_refs = state.get("selected_result_refs", ())
        if state.get("phase") != "detail_unavailable" or result_ref not in selected_refs:
            return JobDiscoveryGatewayResult(run_id=run_id, state="failed", message="Provide JD text only for the selected candidate whose BOSS detail is unavailable.", error_code="PROVIDED_JD_NOT_ALLOWED", error_stage="provided_jd", trace=self._graph.trace(run_id))
        result = next((item for item in state.get("results", ()) if item.result_ref == result_ref), None)
        if result is None:
            return JobDiscoveryGatewayResult(run_id=run_id, state="failed", message="The selected candidate is unavailable.", error_code="INVALID_RESULT_REF", error_stage="provided_jd", trace=self._graph.trace(run_id))
        updated = self._graph.analyze_user_provided_jd(request=request, result=result, jd_text=jd_text, run_id=run_id)
        self._states[run_id] = updated
        self._persist(run_id, updated)
        return self._project(run_id, updated)

    def resume(self, *, run_id: str, value: object = None) -> JobDiscoveryGatewayResult:
        request = self._request(run_id)
        state = self._graph.resume(conversation_id=run_id, value=value)
        self._states[run_id] = state
        self._persist(run_id, state)
        return self._project(run_id, state)

    def status(self, *, run_id: str) -> JobDiscoveryGatewayResult:
        if run_id not in self._requests:
            self._restore(run_id)
        state = self._states.get(run_id) or self._graph.state(run_id)
        self._states[run_id] = state
        return self._project(run_id, state)

    def request_waitlist_confirmation(self, *, run_id: str, result_ref: str, target_role_title: str | None = None) -> JobDiscoveryGatewayResult:
        request = self._request(run_id)
        state = self._states.get(run_id) or self._graph.state(run_id)
        details = state.get("details", {})
        if state.get("phase") != "analysis_ready" or result_ref not in state.get("selected_result_refs", ()) or result_ref not in details:
            raise ValueError("Select and analyze this result before requesting waitlist confirmation.")
        target = state.get("target")
        if not isinstance(target, TargetRoleProposal):
            raise ValueError("Job discovery has no target role.")
        if target_role_title:
            target = TargetRoleProposal(id=f"role_{target_role_title.casefold().replace(' ', '_')}", title=target_role_title, source="explicit", rationale="User edited the target for this confirmation.")
        recommendation = JobDiscoveryRecommendation(summary="User-selected job", items=(RecommendationItem(result_ref=result_ref, rank=1, rationale="Selected by the user."),))
        pending = self._promotion.request_confirmation(user_id=request.user_id, conversation_id=request.conversation_id, recommendation=recommendation, details={result_ref: details[result_ref]}, result_ref=result_ref, target_role=target)
        self._confirmation_runs[pending.id] = run_id
        detail = details[result_ref]
        return JobDiscoveryGatewayResult(
            run_id=run_id,
            state="confirmation_required",
            message=f"Add {detail.title} at {detail.company_name} to the {target.title} waitlist?",
            confirmation_id=pending.id,
            items=(GatewayJobItem(result_ref=result_ref, title=detail.title, company_name=detail.company_name),),
            next_action="confirm_waitlist",
            trace=self._graph.trace(run_id),
        )

    def confirm_waitlist(self, *, run_id: str, confirmation_id: str) -> JobDiscoveryGatewayResult:
        request = self._request(run_id)
        if self._confirmation_runs.get(confirmation_id) != run_id:
            raise ValueError("Confirmation does not belong to this Job Discovery run.")
        result = self._promotion.confirm(user_id=request.user_id, conversation_id=request.conversation_id, confirmation_id=confirmation_id)
        if not isinstance(result, PromotionSuccess):
            return JobDiscoveryGatewayResult(run_id=run_id, state="waiting_user" if result.recoverable else "failed", message=result.message, next_action="retry_confirmation" if result.recoverable else None)
        self._confirmation_runs.pop(confirmation_id, None)
        return JobDiscoveryGatewayResult(run_id=run_id, state="waitlisted", message="Job added to waitlist.", waitlist_item_id=result.records.waitlist.id)

    @staticmethod
    def _fallbacks(selected_refs: tuple[str, ...], results: dict[str, SearchResult], errors: dict[str, str]) -> tuple[GatewayJobFallback, ...]:
        fallbacks = []
        for selection_index, result_ref in enumerate(selected_refs, start=1):
            if result_ref not in errors or (result := results.get(result_ref)) is None:
                continue
            manual_search_query = " · ".join(part for part in (result.company_name, result.title, result.city) if part)
            fallbacks.append(GatewayJobFallback(
                selection_index=selection_index,
                title=result.title,
                company_name=result.company_name,
                city=result.city,
                error_code=errors[result_ref],
                fallback_url=result.source_url,
                manual_search_query=manual_search_query or None,
            ))
        return tuple(fallbacks)

    @staticmethod
    def _item(result: SearchResult) -> GatewayJobItem:
        return GatewayJobItem(result_ref=result.result_ref, title=result.title, company_name=result.company_name, city=result.city, salary=result.salary, source_url=result.source_url)

    def _project(self, run_id: str, state: JobDiscoveryState) -> JobDiscoveryGatewayResult:
        trace = self._graph.trace(run_id)
        results = {result.result_ref: result for result in state.get("results", ())}
        phase = state.get("phase")
        if phase == "selection_required":
            return JobDiscoveryGatewayResult(
                run_id=run_id,
                state="selection_required",
                message="Select one search result to fetch its JD and run analysis, or start a new search.",
                items=tuple(self._item(result) for result in results.values()),
                next_action="select_result",
                trace=trace,
            )
        if state.get("__interrupt__"):
            return JobDiscoveryGatewayResult(run_id=run_id, state="waiting_user", message="BOSS needs user action before the search can continue.", next_action="resume_job_discovery", recovery_action=state.get("recovery_action"), trace=trace)
        if phase == "detail_unavailable":
            selected_refs = tuple(state.get("selected_result_refs", ()))
            errors = state.get("detail_errors", {})
            fallbacks = self._fallbacks(selected_refs, results, errors)
            first = fallbacks[0] if fallbacks else None
            return JobDiscoveryGatewayResult(
                run_id=run_id,
                state="detail_unavailable",
                message="The selected job details are unavailable. Search BOSS manually using the provided company, title, and city, then provide the JD text for analysis.",
                items=tuple(self._item(result) for result in results.values()),
                selected_result_ref=selected_refs[0] if selected_refs else None,
                selected_selection_indices=tuple(index for index, ref in enumerate(state.get("results", ()), start=1) if ref.result_ref in selected_refs),
                fallback_result_refs=tuple(ref for ref in selected_refs if ref in errors),
                fallbacks=fallbacks,
                fallback_url=first.fallback_url if first else None,
                manual_search_query=first.manual_search_query if first else None,
                next_action="provide_jd",
                recovery_action=state.get("recovery_action"),
                error_code=next(iter(errors.values()), None),
                error_stage=state.get("error_stage"),
                error_detail=state.get("error_detail"),
                trace=trace,
            )
        if state.get("error_code"):
            return JobDiscoveryGatewayResult(run_id=run_id, state="failed", message="Job Discovery could not complete.", error_code=state["error_code"], error_stage=state.get("error_stage"), error_detail=state.get("error_detail"), trace=trace)
        if phase in {"analysis_ready", "partial_analysis_ready"}:
            selected_refs = tuple(state.get("selected_result_refs", ()))
            selected = selected_refs[0] if selected_refs else None
            selected_result = results.get(selected) if selected else None
            details = state.get("details", {})
            analyses = state.get("analyses", {})
            errors = state.get("detail_errors", {})
            analysis_refs = tuple(ref for ref in selected_refs if ref in analyses)
            return JobDiscoveryGatewayResult(
                run_id=run_id,
                state=phase,
                message="Selected job details and analyses are ready." if phase == "analysis_ready" else "Available job analyses are ready; some selected jobs need manual JD input.",
                items=tuple(self._item(results[ref]) for ref in selected_refs if ref in results),
                selected_result_ref=selected,
                selected_selection_indices=tuple(index for index, ref in enumerate(state.get("results", ()), start=1) if ref.result_ref in selected_refs),
                detail=details.get(selected) if selected else None,
                analysis=analyses.get(selected) if selected else None,
                analysis_items=tuple(analyses[ref] for ref in analysis_refs),
                analysis_selection_indices=tuple(selected_refs.index(ref) + 1 for ref in analysis_refs),
                comparison=state.get("comparison"),
                fallback_result_refs=tuple(ref for ref in selected_refs if ref in errors),
                fallbacks=self._fallbacks(selected_refs, results, errors),
                next_action="provide_jd" if errors else "request_waitlist_confirmation",
                trace=trace,
            )
        return JobDiscoveryGatewayResult(run_id=run_id, state="running", message="Job Discovery is running.", trace=trace)

    def _persist(self, run_id: str, state: JobDiscoveryState) -> None:
        if self._run_store is None or not state.get("results"):
            return
        self._run_store.save(
            run_id=run_id,
            request=state["request"],
            results=tuple(state["results"]),
            phase=state.get("phase", "unknown"),
            trace=self._graph.trace(run_id),
            selected_result_ref=state.get("selected_result_refs", (None,))[0],
            selected_result_refs=tuple(state.get("selected_result_refs", ())),
            error_code=state.get("error_code"),
            error_stage=state.get("error_stage"),
            error_detail=state.get("error_detail"),
            recoverable=state.get("recoverable"),
        )

    def _restore(self, run_id: str) -> JobDiscoveryState:
        if self._run_store is None:
            raise ValueError("Job discovery run does not exist in this process.")
        record = self._run_store.get(run_id)
        if record is None:
            raise ValueError("Job discovery run does not exist.")
        request = record.request
        target = TargetRoleProposal(id=f"role_{request.target_role.casefold().replace(' ', '_')}", title=request.target_role, source="explicit", rationale="The user explicitly stated this target role.")
        state: JobDiscoveryState = {
            "request": request,
            "run_id": record.run_id,
            "trace_id": record.run_id,
            "phase": record.phase,
            "target": target,
            "results": record.results,
        }
        if record.selected_result_refs:
            state["selected_result_refs"] = record.selected_result_refs
        elif record.selected_result_ref:
            state["selected_result_refs"] = (record.selected_result_ref,)
        if record.error_code:
            state["error_code"] = record.error_code
            state["error_stage"] = record.error_stage or ""
            state["error_detail"] = record.error_detail or ""
            state["recoverable"] = bool(record.recoverable)
        self._requests[run_id] = request
        self._states[run_id] = state
        self._restored_without_checkpoint.add(run_id)
        restore = getattr(self._trace, "restore", None)
        if restore is not None:
            restore(record.trace)
        return state

    def _request(self, run_id: str) -> JobDiscoveryRequest:
        request = self._requests.get(run_id)
        if request is None:
            raise ValueError("Job Discovery run does not exist.")
        return request
