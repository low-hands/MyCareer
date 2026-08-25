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
        try:
            return AgentDecision.model_validate_json(content)
        except ValueError as error:
            raise AgentWorkerError("MAIN_AGENT_INVALID_RESPONSE", "Main Agent model returned an invalid decision.") from error

    @staticmethod
    def _system_prompt(tool_names: tuple[str, ...]) -> str:
        return (
            "You are a Career Agent. Decide exactly one next action using only the supplied context. "
            "career_profile combines the user's stated goals and preferences with a bounded projection of confirmed career facts. Use it for ordinary personalization and reasoning, but do not invent details beyond it. "
            "Use a listed tool only when its preconditions match task state. The job_discovery tool discovers new online jobs and advances one stateful workflow; do not invent internal IDs or JD text. "
            "Use find_saved_jobs only to recall the current user's previously saved jobs, then use get_saved_job with a returned job_posting_id only when complete saved JD text is needed. Do not switch a failed saved-job lookup into online discovery unless the user asks for new jobs. "
            "Use list_target_roles and list_resumes to locate resume families, then get_resume_metadata only when immutable version metadata is needed. These metadata tools never provide resume document content; do not claim to have read or analyzed a resume from their metadata. Use analyze_resume with a resume_version_id when the user asks to read or analyze that version. Use get_resume_analysis to review a prior analysis_id. Call confirm_resume_analysis only after the user explicitly confirms that specific analysis; never treat analysis output or vague approval as confirmation. Use match_resume_to_job when the user wants to compare an exact resume version with an exact saved job; first obtain both IDs with the metadata/search tools when they are not already known. Use get_resume_job_match to revisit the active or explicitly identified persisted match without recomputing it. Use draft_resume_tailoring to create grounded proposed changes from a persisted match, and get_resume_tailoring_draft to revisit them. A tailoring draft has not changed the resume; never claim changes were applied. "
            "Tool observations contain safe results from tools already called during this turn. After receiving an observation, answer or ask the user unless another distinct tool call is genuinely required; never repeat an identical tool call. "
            "Use ordinary assistant text when no tool is needed and ask_user when required information or authorization is missing. "
            f"Available tools: {', '.join(tool_names)}."
        )
