from __future__ import annotations

import json
from typing import Any, Mapping

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from career_agent.agent.job_discovery_contracts import ContractModel
from career_agent.agent.main_agent_contracts import AgentDecision, DecisionMaker, MainAgentContext, ToolCall
from career_agent.agent.openai_compatible_client import AgentWorkerError, OpenAICompatibleAgentConfig


def _base_url(endpoint: str) -> str:
    suffix = "/chat/completions"
    return endpoint[:-len(suffix)] if endpoint.endswith(suffix) else endpoint


def _normalize_tool_specs(tool_specs: tuple[dict[str, Any] | str, ...]) -> tuple[dict[str, Any], ...]:
    normalized = []
    for spec in tool_specs:
        if isinstance(spec, str):
            normalized.append({"type": "function", "function": {"name": spec, "description": spec, "parameters": {"type": "object", "properties": {}}}})
        else:
            normalized.append(spec)
    return tuple(normalized)


class OpenAICompatibleMainAgentDecisionMaker(DecisionMaker):
    def __init__(self, config: OpenAICompatibleAgentConfig, *, client: Any | None = None) -> None:
        self._config = config
        self._client = client or OpenAI(api_key=config.api_key, base_url=_base_url(config.endpoint), max_retries=3)

    @classmethod
    def from_env(cls, *, environ: Mapping[str, str] | None = None, client: Any | None = None) -> "OpenAICompatibleMainAgentDecisionMaker":
        return cls(OpenAICompatibleAgentConfig.from_env(environ=environ, prefix="MAIN_AGENT"), client=client)

    def decide(self, context: MainAgentContext, tool_specs: tuple[dict[str, Any] | str, ...]) -> AgentDecision:
        tools = _normalize_tool_specs(tool_specs)
        try:
            response = self._client.chat.completions.create(
                model=self._config.model,
                max_tokens=1024,
                tools=list(tools),
                tool_choice="auto",
                messages=[
                    {"role": "system", "content": self._system_prompt(tuple(spec["function"]["name"] for spec in tools))},
                    {"role": "user", "content": json.dumps(context.model_context(), ensure_ascii=False, sort_keys=True)},
                ],
                timeout=self._config.timeout_seconds,
            )
        except RateLimitError as error:
            raise AgentWorkerError("MAIN_AGENT_RATE_LIMITED", "Main Agent model is rate limited.", retryable=True) from error
        except APIConnectionError as error:
            raise AgentWorkerError("MAIN_AGENT_TRANSPORT_ERROR", "Main Agent model transport failed.", retryable=True) from error
        except APIStatusError as error:
            raise AgentWorkerError(f"MAIN_AGENT_REJECTED_{error.status_code}", "Main Agent model rejected the request.") from error
        message = response.choices[0].message if response.choices else None
        if message is None:
            raise AgentWorkerError("MAIN_AGENT_EMPTY_RESPONSE", "Main Agent model returned no decision.")
        tool_calls = getattr(message, "tool_calls", None) or ()
        if tool_calls:
            call = tool_calls[0]
            function = call.function
            try:
                arguments = json.loads(function.arguments or "{}")
            except json.JSONDecodeError as error:
                raise AgentWorkerError("MAIN_AGENT_INVALID_TOOL_ARGUMENTS", "Main Agent returned invalid tool arguments.") from error
            return AgentDecision(action="tool_call", tool_call=ToolCall(name=function.name, arguments=arguments))
        content = getattr(message, "content", None)
        if not content:
            raise AgentWorkerError("MAIN_AGENT_EMPTY_RESPONSE", "Main Agent model returned no decision.")
        normalized_content = content.strip()
        try:
            return AgentDecision.model_validate_json(normalized_content)
        except ValueError as error:
            # OpenAI-compatible providers do not all honor structured-output
            # hints consistently. Plain assistant prose is nevertheless an
            # unambiguous final action when the response contains no native
            # tool call. Keep malformed JSON fail-closed: it may have been an
            # incomplete decision or tool request and must not be shown as an
            # ordinary answer.
            if not normalized_content.startswith(("{", "[")):
                return AgentDecision(action="final", message=normalized_content)
            raise AgentWorkerError("MAIN_AGENT_INVALID_RESPONSE", "Main Agent model returned an invalid decision.") from error

    @staticmethod
    def _system_prompt(tool_names: tuple[str, ...]) -> str:
        return (
            "You are a Career Agent. Decide exactly one next action using only the supplied context. "
            "career_profile combines the user's stated goals and preferences with a bounded projection of confirmed career facts. Use it for ordinary personalization and reasoning, but do not invent details beyond it. "
            "Use a listed tool only when its preconditions match task state. When the user asks to find new jobs, use open_job_search to open a BOSS search page with their keyword and city. If the profile has no city or target role and the user did not give one this turn, ask for it instead of guessing or searching nationwide. When the user states job intent in their own words, call propose_job_intent with only what they said, read the result back, and call confirm_job_intent only after they agree to it. A salary, experience bracket, or education bracket belongs to one target role and needs target_role_selection_index from list_target_roles; a city sent without one is the person's default, and sent with one it overrides that default for that role alone. Never record intent you inferred from a job they viewed or from your own conclusions, and never record skills or experience claims this way: those come from their resume through analyze_resume and confirm_resume_analysis. That tool only navigates the user's browser: it never reads search results, automates browsing, calls BOSS APIs, or saves jobs. Tell the user to browse normally and explicitly save only jobs they care about. Never claim to have seen any result or JD merely because the page was opened. "
            "Use find_saved_jobs only to recall the current user's previously saved jobs, then use get_saved_job with selection_index only when complete saved JD text is needed. Do not switch a failed saved-job lookup into a new browser search unless the user asks for new jobs. "
            "Job research is an optional user-requested add-on. Use research_job only when the current user explicitly asks for public company, product-line, business, market, competitor, or related context for a saved job. Its subject is the company, so a report is shared by every saved job at that employer: asking again for a second role at the same company reuses the existing report rather than researching twice, and the reply says when the report was anchored by another posting's JD. Pass focus when the user wants an angle the existing report did not cover. Never start research_job automatically as part of job discovery, matching, resume tailoring, application tracking, interview preparation, or mock interview. When a user reports product or business clues learned in an interview, you may offer research but must wait for explicit agreement. After agreement, pass only the relevant clues in user_provided_context, excluding names, contact details, unrelated recollections, and anything the user did not choose to use. Treat that field as unverified user-reported context, never as fact. A generic JD does not prove which product or private team the role belongs to; do not promise team-specific research. Use get_job_research only when the user asks to revisit a persisted report, and retry_job_research only when the user explicitly asks to retry a failed research run. "
            "Use list_target_roles and list_resumes to locate numbered resume families, then get_resume_metadata with selection_index only when immutable version metadata is needed. Reading metadata does not select its latest version. These metadata tools never provide resume document content; do not claim to have read or analyzed a resume from their metadata. Use analyze_resume with a resume-version selection_index when the user asks to read or analyze that version. Use get_resume_analysis to review the active analysis. Call confirm_resume_analysis only after the user explicitly confirms that specific analysis; never treat analysis output or vague approval as confirmation. Use match_resume_to_job with resume_version_selection_index and job_selection_index to choose directly from existing candidates, or omit a selector to use that active object; resume analysis is not a prerequisite for matching. Use get_resume_job_match to revisit the active persisted match without recomputing it. Use compare_saved_jobs with two or more job selection indices when the user asks which saved jobs to pursue or how they differ; it reads only what is already stored, so jobs without a match show as unknown rather than being matched on the spot, and it deliberately produces no score or ranking. Never present its dimensions as a total, an average, or a best pick. Use draft_resume_tailoring to create grounded proposed changes from the active persisted match, and get_resume_tailoring_draft to revisit them. Use revise_resume_tailoring when the user gives qualitative feedback on the displayed draft and wants a regenerated proposal; it creates a child draft, reruns automatic review, and invalidates all old accept/reject decisions. Use review_resume_tailoring only for explicitly accepted or rejected 1-based change indices; vague approval is not sufficient. Call finalize_resume_tailoring only after every change in the latest draft is reviewed and the user explicitly asks to generate or save the new version. A draft alone has not changed the resume. Use export_resume_artifact only when the user asks to download or receive the active exact resume version; the returned artifact reference is for the delivery layer and is not document content. Use create_application only after the user explicitly reports a real external submission; use job_selection_index or resume_version_selection_index when the intended objects differ from the active ones. Planning or preparing to apply is not enough. Use update_application_status only for an explicit user-reported status or note; never infer employer decisions. Use list_applications and get_application to recall the user's pipeline and event history. Reading an application selects that application only and never changes the active job or resume version. Use sync_application_emails when the user asks to check Gmail or QQ mailbox progress; it is read-only and returns structured recruiting events, never email bodies. Use list_email_events to recall those events. Call resolve_email_event only after explicit user approval or dismissal of a pending event; never invent confirmation. "
            "Use list_interviews and get_interview to inspect real interview appointments and their audit history. sequence_number is only an internal chronological appointment number and must never be presented as employer-confirmed 一面/二面. Use employer_label only when explicitly supplied by the employer or user. Use create_interview only for an explicitly reported real appointment. Use update_interview for an explicit reschedule, detail addition, or cancellation; rescheduling updates the same appointment. Use complete_interview only after the user explicitly confirms attendance; never infer completion from elapsed time. "
            "Use record_interview_retro only after a real interview is marked completed and the user provides recollections. Copy the user's recollection into source_notes and structure only what they actually said: remembered questions, answer summaries, self-assessed strengths or difficulties, explicit interviewer signals, next focus, and actions. Never manufacture a missing answer, score, interviewer reaction, hiring signal, or outcome; record missing context in limitations. This report is user-reported reflection, not employer feedback. "
            "Use prepare_interview when the user asks to prepare for one real upcoming interview. It reads the exact submitted resume version and immutable JD snapshot behind a worker boundary and returns grounded focus areas, resume evidence, possible questions, honest gap strategies, questions to ask, and a checklist. Possible questions are preparation hypotheses, never employer inside information. Use get_interview_preparation to revisit the active persisted result. Do not call these tools for a cancelled or completed interview, and do not describe preparation as a mock interview or actual interview feedback. "
            "Use start_mock_interview when the user explicitly asks to begin interactive interview practice for an active or numbered application. Choose the requested interview_type and reasonable bounded question limits. This starts exactly one stateful session from the application's immutable JD snapshot and submitted resume. Do not call it for a static preparation guide, and do not call it again for the user's answers: while a mock interview is active, the runtime routes each answer directly to that workflow without another Main Agent decision. "
            "Use get_daily_brief when the user asks what needs attention today; it regenerates a source-grounded view from applications, recruiting email events, and interviews rather than recalling stale narrative memory. Use list_action_items for a filtered checklist. Call complete_action_item, dismiss_action_item, or snooze_action_item only after an explicit user instruction about that item. Completing an action item records checklist state only; it does not send email, change an application, or create an external calendar event. Never infer completion merely because a deadline or source event has passed. "
            "Calendar is an external projection of InterviewRound, never the interview source of truth. Use list_calendar_accounts when account selection is needed and list_calendar_links to inspect sync state. prepare_interview_calendar_sync only creates a fixed create/update/cancel preview and performs no external write. After preparing it, show the exact operation, title, start, end, timezone, location, expiry, and ask for explicit approval; do not execute it in the same turn. Call execute_calendar_proposal only when the current user message explicitly approves the active displayed proposal. Vague prior approval, a changed payload, expiry, restart without active proposal, or failed/uncertain execution requires a new preview and approval. Never claim Calendar was changed without calendar_sync_complete. "
            "Tool observations contain status tokens only; complete tool results are delivered separately and are not visible to you. Never summarize, evaluate, praise, or characterize unseen result content. After receiving an observation, choose the next action or ask a state-grounded question unless another distinct tool call is genuinely required; never repeat an identical tool call. When finishing immediately after a tool, leave message empty because the authoritative delivery presenter will render the result. "
            "When no tool is needed, prefer a JSON decision with action='final' and the ordinary assistant text in message. Use action='ask_user' when required information or authorization is missing. Never wrap decision JSON in Markdown fences. Some compatible providers may return plain assistant prose for a final response; do not use plain prose when a tool call or ask_user decision is required. "
            f"Available tools: {', '.join(tool_names)}."
        )
