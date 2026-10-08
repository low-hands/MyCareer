"""Single source of truth for Main Agent capability metadata.

Handlers remain dependency-bound by :class:`MainAgentToolRegistry`, but every
static fact used to expose, authorize, gate, and replay a capability lives in
this catalogue.  The small compatibility modules ``tool_effects``,
``tool_reachability`` derives its public view from it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Mapping

from career_agent.agent.capabilities.example_queries import EXAMPLE_QUERIES

if TYPE_CHECKING:
    from career_agent.agent.contracts.task_state import ConversationTaskState
else:
    ConversationTaskState = Any

ToolEffect = Literal["READ", "WRITE", "CONTROL"]
ExecutionKind = Literal["atomic_tool", "workflow", "runtime_workflow"]
# `never` is reserved for non-WRITE and runtime-owned capabilities. Every
# model-callable WRITE is owner-rule confirmable or unconditionally reviewed.
ApprovalPolicy = Literal["never", "owner_rule", "always"]
ReplayPolicy = Literal["not_applicable", "never", "idempotent"]
RecoveryPolicy = Literal["not_applicable", "retry", "reconcile"]
Precondition = Callable[[ConversationTaskState], bool]


@dataclass(frozen=True)
class CapabilityDescriptor:
    name: str
    description: str | None
    arguments_model: str | None
    effect: ToolEffect
    namespace: str | None = None
    summary: str | None = None
    aliases_zh: tuple[str, ...] = ()
    example_queries: tuple[str, ...] = ()
    successors: tuple[str, ...] = ()
    execution_kind: ExecutionKind = "atomic_tool"
    approval_policy: ApprovalPolicy = "never"
    replay_policy: ReplayPolicy = "not_applicable"
    recovery_policy: RecoveryPolicy = "not_applicable"
    output_model: str = "ToolObservation"
    external_write: bool = False
    runtime_owned: bool = False
    notes_guarded: bool = False
    preference_bound: bool = False
    reference_readback: bool = False
    schema_gated: bool = False
    precondition: Precondition | None = None
    requirement: str | None = None

    def validate_arguments(self, context: Any, arguments: dict[str, Any]) -> None:
        """Run the same pure input binding used before execution."""
        from career_agent.agent.contracts import main_agent as contracts
        from career_agent.agent.contracts.decisions import _reject_internal_identifiers
        from career_agent.agent.middleware.argument_projection import project_atomic_arguments, project_workflow_arguments
        _reject_internal_identifiers(self.name, arguments)
        if self.arguments_model is not None:
            getattr(contracts, self.arguments_model).model_validate(arguments)
        elif arguments:
            raise ValueError("this capability accepts no model arguments")
        if self.execution_kind == "workflow":
            project_workflow_arguments(context, self.name, arguments)
        elif self.execution_kind == "atomic_tool":
            project_atomic_arguments(context, self.name, arguments, source_turn_id=None)

    @property
    def model_callable(self) -> bool:
        return self.execution_kind != "runtime_workflow"

    @property
    def handler_name(self) -> str:
        """Registry method bound for this capability by naming convention."""

        return f"_{self.name}"

    @property
    def replay_safe(self) -> bool:
        """Whether a durable receipt may safely return the prior result."""

        return self.replay_policy == "idempotent"

    def tool_schema(self) -> dict[str, object]:
        """Build the provider-facing function schema from the declared contract."""

        if not self.model_callable or self.description is None:
            raise ValueError(f"runtime-only capability has no tool schema: {self.name}")
        if self.arguments_model is None:
            parameters: dict[str, object] = {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            }
        else:
            # Delayed to keep the catalogue importable while main_agent_contracts
            # itself imports effect/profile views derived from this module.
            from career_agent.agent.contracts import main_agent as main_agent_contracts

            model = getattr(main_agent_contracts, self.arguments_model)
            parameters = model.model_json_schema()
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

    def output_schema(self) -> dict[str, object]:
        """Return the internal result-envelope schema for adapters and docs.

        Function-calling providers currently receive only ``tool_schema``.
        Keeping the output contract separate avoids claiming that providers
        enforce a response schema which is actually enforced by our registry.
        """

        from career_agent.agent.contracts import main_agent as main_agent_contracts

        model = getattr(main_agent_contracts, self.output_model)
        return model.model_json_schema()

    def validate_output(self, value: object) -> Any:
        """Validate a handler result and bind it to the invoked capability."""

        from career_agent.agent.contracts import main_agent as main_agent_contracts

        model = getattr(main_agent_contracts, self.output_model)
        result = model.model_validate(value)
        if result.tool_name != self.name:
            raise ValueError(
                f"capability {self.name!r} returned result for {result.tool_name!r}"
            )
        return result


def _reachable_via_job(task: ConversationTaskState) -> bool:
    return bool(task.active_job_posting_id or task.saved_job_candidates)


def _reachable_via_resume_version(task: ConversationTaskState) -> bool:
    return bool(task.active_resume_version_id or task.resume_version_candidates)


def _reachable_via_application(task: ConversationTaskState) -> bool:
    return bool(task.active_application_id or task.application_candidates)


def _reachable_via_interview(task: ConversationTaskState) -> bool:
    return bool(task.active_interview_round_id or task.interview_candidates)


def _reachable_via_action_item(task: ConversationTaskState) -> bool:
    return bool(task.active_action_item_id or task.action_candidates)


def _has_current_job_analysis(task: ConversationTaskState) -> bool:
    return bool(
        task.active_job_analysis_id
        and task.job_analysis_status == "ready"
        and task.active_job_analysis_jd_snapshot_id
        and task.active_job_analysis_jd_snapshot_id == task.active_jd_snapshot_id
    )


_NEEDS_JOB = "先用 find_saved_jobs 列出或选定一个已收藏岗位"
_NEEDS_APPLICATION = "先用 list_applications 列出或选定一条投递记录"
_NEEDS_INTERVIEW = "先用 list_interviews 列出或选定一轮面试"
_NEEDS_ACTION_ITEM = "先用 list_action_items 列出待办事项"
_NEEDS_TAILORING_DRAFT = "先用 draft_resume_tailoring 生成定制草稿"
_NEEDS_PROPOSAL = "先调用对应的 propose_* 工具向用户展示提案"

_ALWAYS_CONFIRM = frozenset({
    "update_owner_settings",
    "execute_calendar_proposal",
    "confirm_memory_tombstone",
    "confirm_constraint_retirement",
})

# Model-facing schema metadata. The insertion order is part of the stable
# tool-prefix contract sent to providers; append deliberately and do not sort.
_SCHEMA_SPECS: Mapping[str, tuple[str | None, str]] = MappingProxyType({
    'read_conversation_span': (
        'ReadConversationSpanToolArguments',
        (
            "The projection's through_sequence is the last message covered by conversation_summary, a"
            'nd recent_from_sequence is the first raw recent message. When the user needs an earlier '
            'fact absent from the summary and recent messages, read the compressed history even when '
            'the two sequence boundaries are adjacent. Omit the sequence bounds to let the runtime '
            'select 1 through the message before recent_from_sequence, including any omitted rows '
            'after the summary watermark; provide both only when the user names exact numbers. For '
            'long gaps, pass focused query terms to search message content instead of walking spans e'
            'ight rows at a time. Without query it returns the oldest rows in the exact span. Returns'
            ' at most 8 matching messages, clips each at 4000 characters, and reports returned/total '
            'plus clipping honestly. It never searches another conversation or substitutes nearby row'
            's when the requested span is empty. This reads conversation messages, not the active '
            'constraint ledger; use fetch_archived_constraints for omitted active constraints.'
        ),
    ),
    'fetch_archived_constraints': (
        'FetchArchivedConstraintsToolArguments',
        (
            'Read the constraints this conversation recorded but conversation_summary is not showing.'
            ' Call this when omitted_active_constraint_count is above zero and the reply depends on w'
            'hich constraints apply; when the count is zero it returns not applicable. An archived '
            'constraint still applies. It does not recover '
            'arbitrary earlier chat facts; use read_conversation_span for those. Read-only.'
        ),
    ),
    'propose_constraint_retirement': (
        'ProposeConstraintRetirementToolArguments',
        (
            'Prepare to stop applying one recorded constraint, passing its exact text from conversati'
            'on_summary.active_constraints or from fetch_archived_constraints. Use this only when the'
            ' user says a constraint no longer holds; never to make room for a new one, and never bec'
            'ause a constraint looks stale. This only reads the target and shows a bounded proposal.'
        ),
    ),
    'confirm_constraint_retirement': (
        None,
        (
            'Execute the exact constraint retirement already shown to the user. Call only after expli'
            'cit agreement. The constraint stops applying and will not return even if a later summary'
            ' rewrite re-extracts the same text.'
        ),
    ),
    'search_career_episodes': (
        'SearchCareerEpisodesToolArguments',
        (
            'Search completed applications, job research, interviews, and mock interviews across '
            'conversations, or expand one projected detail_ref. In query, pass company names, role '
            'names, and key nouns separated by spaces. Include a known alias and full name together, '
            'for example 腾讯 鹅厂 or 拼多多 PDD. Put time and episode type in start_datetime, '
            'end_datetime, and kinds, not in query. Feedback is scoped to those filters: '
            'matched_terms means the full query word occurs in an episode; '
            'partially_matched_terms means only fragments occur and is not evidence that the '
            'named object exists; unmatched_terms means no fragment occurs. Long unspaced '
            'sentences have no term feedback. If no spelling of an object, including aliases, '
            'is in matched_terms, do not substitute another object. Separate term matches do '
            'not prove that the terms occur together in one episode. '
            'Results are compact pointers and synopses; dereference resource_refs before using an '
            'episode as factual evidence.'
        ),
    ),
    'update_working_notes': (
        'UpdateWorkingNotesToolArguments',
        (
            'Replace the complete per-user agent scratchpad. It is unconfirmed and may guide question'
            's or response style only; never use it as a basis for filtering, ranking, applications, '
            'scheduling, or authoritative state.'
        ),
    ),
    'resolve_claim_source': (
        'ResolveClaimSourceToolArguments',
        (
            'Read the original resume evidence for a confirmed career claim only when its projected s'
            "ource_ref is relevant to the user's request. The quotation is returned as a bounded turn"
            '-local result.'
        ),
    ),
    'get_career_memory_detail': (
        'GetCareerMemoryDetailToolArguments',
        (
            'Expand one current career claim only when its projected detail_ref is relevant. Returns '
            'the current revision, direct support, and correction lineage without treating an old quo'
            'tation as support for the corrected claim.'
        ),
    ),
    'search_career_memory': (
        'SearchCareerMemoryToolArguments',
        (
            'Search active confirmed career claims omitted from the bounded Tier-1 window. Use this l'
            'ayered archive fetch when career_profile.memory_overflow names this tool; paginate with '
            'the returned cursor.'
            ' This store contains confirmed career claims only: it does not contain conversation '
            'transcripts or generated research reports. Do not use it to reconstruct a missing '
            'report or the exact words of an old message.'
        ),
    ),
    'search_career_history': (
        'SearchCareerHistoryToolArguments',
        (
            'Search superseded or rolled-back career claims for historical questions. Pass focused te'
            'rms expected inside the earlier claim; this is the query-indexed history path, not sourc'
            'e_ref lookup. When next_cursor is returned, pass it back with the identical query to rea'
            'd the next page.'
            ' Historical means changed career claims, not conversation history. This tool cannot '
            'retrieve old messages, company research reports or their competitor analysis. For '
            'a missing message use read_conversation_span if available; otherwise explain the '
            'missing context. For an inaccessible report explain that it cannot be retrieved.'
        ),
    ),
    'propose_career_fact': (
        'ProposeCareerFactToolArguments',
        (
            'Create a quarantined career-fact candidate for one projected career record and read the '
            'exact claim back to the user. Use only for an explicit user statement; it is not active '
            "until confirmed. Supply user_quote as an exact excerpt from the user's message or questi"
            'onnaire answer to mark user_input provenance; omit it for inference.'
        ),
    ),
    'confirm_career_fact': (
        'ConfirmCareerFactToolArguments',
        (
            'Confirm the exact quarantined career-fact proposal shown on the preceding turn. Takes no'
            ' arguments.'
        ),
    ),
    'propose_memory_amendment': (
        'ProposeMemoryAmendmentToolArguments',
        (
            'Prepare a field-level correction for one current career claim identified by detail_ref. '
            'This only shows the exact replacement claim and reason; it does not write a revision.'
        ),
    ),
    'confirm_memory_amendment': (
        None,
        (
            'Write the exact correction proposal already shown to the user as a new revision. Call on'
            'ly after explicit agreement.'
        ),
    ),
    'propose_memory_tombstone': (
        'ProposeMemoryTombstoneToolArguments',
        (
            'Prepare an irreversible field-level deletion for one current career claim identified by '
            'detail_ref. This only reads the target and shows a bounded proposal; it never deletes an'
            'ything.'
        ),
    ),
    'confirm_memory_tombstone': (
        None,
        (
            'Execute the exact memory deletion proposal already shown to the user. Call only after ex'
            'plicit agreement; the write redacts the complete claim lineage and is not reversible.'
        ),
    ),
    'open_job_search': (
        'OpenJobSearchToolArguments',
        (
            'Open a BOSS recruitment search page when the user asks to find new jobs. If neither the '
            'profile nor a target role supplies a city and the user did not give one this turn, ask f'
            'or the city instead of guessing or searching nationwide. This only constructs a safe sea'
            'rch URL for the client; it never reads results, automates browsing, calls BOSS APIs, or '
            'saves a job. The user browses normally and explicitly chooses which JD to save. Use job_'
            'type for 实习, 全职, or 兼职 instead of putting that word in keyword. After opening it, tell t'
            'he user that browsing and saving are theirs, and never claim that jobs were found or sav'
            'ed.'
        ),
    ),
    'propose_free_text_preference_confirmation': (
        'ProposeFreeTextPreferenceConfirmationToolArguments',
        (
            'Show one numbered quarantined free-text preference from the free_text_preferences Markdo'
            'wn block back to the user and ask whether it should become a lasting active preference. '
            "This never activates it and ends the turn waiting for the user's answer."
        ),
    ),
    'confirm_free_text_preference': (
        'ConfirmFreeTextPreferenceToolArguments',
        (
            'Promote the exact free-text preference previously shown by propose_free_text_preference_'
            'confirmation. Call only on a later turn after explicit agreement.'
        ),
    ),
    'propose_job_intent': (
        'ProposeJobIntentToolArguments',
        (
            'Show the user what would be recorded as their stated job intent, without saving anything'
            '. Send only the fields they just stated in their own words; never infer a city, salary, '
            'or bracket from a job they looked at or from anything you concluded. Salary, experience,'
            ' and education belong to one target role and require target_role_selection_index from li'
            'st_target_roles, because a candidate pursuing two tracks wants different numbers for eac'
            "h. A city sent without a selection index is the person's default; sent with one it overr"
            'ides that default for that role alone. Person-level hard constraints may use only the de'
            'clared work_arrangement, work_schedule, or company_scale relations and must preserve the'
            " user's own wording (for example, 必须远程 or 不接受996); never infer one. This tool records in"
            'tent only: skill and experience claims come from the resume, never from being told.'
        ),
    ),
    'confirm_job_intent': (
        'ConfirmJobIntentToolArguments',
        (
            'Save the update the user was just shown. Call it only after they explicitly agree to tha'
            't specific readback; continuing the conversation is not agreement.'
        ),
    ),
    'compare_saved_jobs': (
        'CompareSavedJobsToolArguments',
        (
            'Lay two or more saved jobs out side by side on fixed dimensions, using only what is alre'
            'ady on record. It never runs a new resume match and never scores, weights, or ranks the '
            'jobs: a dimension the data does not answer is reported as unknown rather than filled in.'
            ' Use it when the user asks which saved jobs to pursue or how they differ.'
        ),
    ),
    'find_saved_jobs': (
        'FindSavedJobsToolArguments',
        (
            "Search only the current user's previously saved or viewed jobs. Use this for historical "
            'recall, not for discovering new online jobs. Omit query to list recent saved jobs. Results become numbered saved-job candidate'
            's; complete JD text stays outside the decision context.'
        ),
    ),
    'get_saved_job': (
        'GetSavedJobToolArguments',
        (
            'Read the selected or active saved job. Pass selection_index after find_saved_jobs, or om'
            'it it to use the active job. The complete JD is delivered outside the decision context.'
        ),
    ),
    'research_job': (
        'ResearchJobToolArguments',
        (
            'Optional current public-web research for the selected or active saved job. Call only whe'
            'n the user explicitly asks to research company, product-line, business, market, competit'
            'or, or related public context; never start it automatically during matching, tailoring, '
            'application, or interview workflows. A generic JD cannot establish what a specific priva'
            'te team works on. Results are source-grounded and persisted. If the user explicitly agre'
            'es to investigate business clues they reported after an interview, pass only those relev'
            'ant clues as user_provided_context; they remain unverified until supported by public sou'
            'rces.'
        ),
    ),
    'retry_job_research': (
        'RetryJobResearchToolArguments',
        (
            'Resume the active failed job-research run from its checkpoint. Use only after a retryabl'
            'e research failure and an explicit user request to retry.'
        ),
    ),
    'get_job_research': (
        'GetJobResearchToolArguments',
        (
            'Read a persisted job-research report without running web research again. When a complete'
            'd tool result carries a title and reference, match the requested company to that same re'
            'sult and pass its exact reference; never borrow a reference from an older chat resource '
            "or a differently titled result. A grounded saved-job selection_index reads that company'"
            's latest available report, not a specific historical version. Looking up a saved job can'
            'not recover the identity of a missing historical report. If no selector identifies the r'
            'equested report, explain that it cannot be read and ask the user to supply it. Omit both'
            ' only when the user actually means the active one.'
        ),
    ),
    'list_target_roles': (
        'ListTargetRolesToolArguments',
        (
            "List the current user's resume target-role categories as numbered candidates. Never retu"
            'rns resume document content.'
        ),
    ),
    'list_resumes': (
        'ListResumesToolArguments',
        (
            "List the current user's resume families. If the user named a resume, direction, or tag, "
            'pass it as query so only matching candidates are returned; otherwise returns a small rec'
            'ent candidate set. Returns numbered safe metadata only; never returns resume document co'
            'ntent.'
        ),
    ),
    'get_resume_metadata': (
        'GetResumeMetadataToolArguments',
        (
            'Read a resume family selected by selection_index and list its immutable versions as numb'
            'ered metadata. Never returns PDF, text, Markdown, extracted content, or file paths.'
        ),
    ),
    'load_skill': (
        'LoadSkillToolArguments',
        (
            'Load the working instructions (a skill) for a task you do yourself, then follow them in '
            'your reply. The skill text is for you only; the user does not see it. Available skills: '
            'resume-critique, for reviewing, critiquing or improving a resume as a document when the '
            'user names no job (a resume compared with a job is match_resume_to_job). It works from t'
            'he resume attached to the message, whose text is in attached_resumes; if none is attache'
            'd, ask the user to attach the resume first instead of loading the skill.'
        ),
    ),
    'analyze_job': (
        'AnalyzeJobToolArguments',
        (
            "Analyze one saved job's complete JD on its own: core objective, inferred seniority, S/A/"
            'B/C tiered requirements with JD quotes, core competencies, implicit requirements, ATS ke'
            'ywords, HR / hiring-manager focus, likely interview topics, red flags, and information g'
            'aps. Pass selection_index after find_saved_jobs, or omit it to use the active job. Reads'
            ' only the JD text: no resume, no preferences, no online research. Use match_resume_to_jo'
            'b instead when the user wants a comparison against a resume.'
        ),
    ),
    'correct_job_requirement_tier': (
        'CorrectJobRequirementTierToolArguments',
        (
            "After the user explicitly confirms or corrects one requirement's classification, create "
            'a new immutable job-analysis revision. Never infer user confirmation.'
        ),
    ),
    'match_resume_to_job': (
        'MatchResumeToJobToolArguments',
        (
            "Compare an exact current-user resume version with one saved job's complete JD. Requires "
            'both objects AND a ready analysis of the current JD; call analyze_job first when that '
            'analysis is missing. Reads their source documents itself. Use resume_version_selection_index'
            ' and job_selection_index to choose directly from existing candidates, or omit either sel'
            'ector to use its active object. Returns a grounded assessment outside the decision conte'
            'xt; does not search online and never returns either original document.'
        ),
    ),
    'get_resume_job_match': (
        'GetResumeJobMatchToolArguments',
        (
            "Retrieve the current conversation's active persisted resume-job match. Returns only the "
            'structured assessment, never the original resume or complete JD.'
        ),
    ),
    'draft_resume_tailoring': (
        'DraftResumeTailoringToolArguments',
        (
            'Create a reviewable tailoring draft from the active persisted resume-job match. May acce'
            'pt a user tailoring goal. Grounded proposed changes are delivered outside decision conte'
            'xt; this does not alter or create a resume version.'
        ),
    ),
    'get_resume_tailoring_draft': (
        'GetResumeTailoringDraftToolArguments',
        (
            'Retrieve the active unexpired tailoring draft. Returns proposed changes for review outsi'
            'de decision context; it does not apply them.'
        ),
    ),
    'review_resume_tailoring': (
        'ReviewResumeTailoringToolArguments',
        (
            'Accept or reject specific 1-based change indices in an active tailoring draft. Use only '
            'decisions the user explicitly made; never infer acceptance from vague approval. Decision'
            's are persisted and may be completed across turns. This does not create a new resume ver'
            'sion.'
        ),
    ),
    'revise_resume_tailoring': (
        'ReviseResumeTailoringToolArguments',
        (
            'Create a new child draft from the active tailoring draft using explicit user feedback. T'
            'he new draft discards all prior accept/reject decisions, reruns the bounded Writer/Revie'
            'wer loop, and must be reviewed again. It never overwrites the parent draft or creates a '
            'ResumeVersion.'
        ),
    ),
    'finalize_resume_tailoring': (
        'FinalizeResumeTailoringToolArguments',
        (
            'Create one new immutable Markdown ResumeVersion from the explicitly accepted changes in '
            'a fully reviewed tailoring draft. Call only when the user explicitly asks to generate/sa'
            've the new version after reviewing every change. Repeated calls are idempotent. Never us'
            'e vague approval as authorization.'
        ),
    ),
    'export_resume_artifact': (
        'ExportResumeArtifactToolArguments',
        (
            'Prepare the active owned immutable resume version for download and return only an opaque'
            ' artifact reference plus safe file metadata. Call only when the user asks to download, e'
            'xport, or receive the resume file. Never place file content or a local path in the conve'
            'rsation.'
        ),
    ),
    'create_application': (
        'CreateApplicationToolArguments',
        (
            "Track a real externally submitted application against the saved job's current immutable "
            'JD snapshot. Attach the exact owned resume version when known; it may be omitted for a r'
            'eferral or an already-progressed process whose resume is unknown. Use selection indexes '
            'to override active objects. Call only after the user explicitly reports that they actual'
            'ly applied; planning or preparing is not sufficient, but do not lecture about this rule '
            'unless the user says they have not applied yet. Identify the job from what the user name'
            'd: call find_saved_jobs with that company or title, use the job if exactly one matches, '
            'and ask the user to choose only among the matches if several do. Never offer the whole s'
            'aved-job library as options; if the user named no job, ask which company or role in plai'
            'n text. Repeated calls for the same job return the original application.'
        ),
    ),
    'update_application_status': (
        'UpdateApplicationStatusToolArguments',
        (
            'Update the active or specified application to a valid pipeline status, or add a note by '
            'supplying the unchanged status with a note. Use only status changes or facts explicitly '
            'supplied by the user.'
        ),
    ),
    'list_applications': (
        'ListApplicationsToolArguments',
        (
            "List the current user's tracked applications, optionally filtered by pipeline statuses. "
            'Returns safe job and application metadata, not resume or JD contents.'
        ),
    ),
    'get_application': (
        'GetApplicationToolArguments',
        (
            'Read a tracked application selected by selection_index, or use the active application, i'
            'ncluding its append-only event timeline.'
        ),
    ),
    'update_owner_settings': (
        'UpdateOwnerSettingsToolArguments',
        (
            'Propose a persistent owner setting change. The runtime always stops this call and shows '
            'the exact change to the owner; it takes effect only after the owner confirms the bound i'
            'nteraction. Never claim it changed before confirmation.'
        ),
    ),
    'sync_application_emails': (
        'SyncApplicationEmailsToolArguments',
        (
            'Run the read-only Gmail/QQ recruiting-email synchronization workflow. It fetches metadat'
            'a first, reads only candidate bodies in an isolated worker, links events to tracked appl'
            'ications, and returns safe summaries. Use when the user asks to check or refresh employe'
            'r email progress.'
        ),
    ),
    'list_email_events': (
        'ListEmailEventsToolArguments',
        (
            'List safe structured recruiting-email events, optionally only those awaiting confirmatio'
            'n. Never returns email bodies or credentials.'
        ),
    ),
    'resolve_email_event': (
        'ResolveEmailEventToolArguments',
        (
            'Approve or dismiss one pending email event. Approval may explicitly correct its applicat'
            'ion link and updates the application only when the transition is valid. Use only after c'
            'lear user confirmation.'
        ),
    ),
    'list_interviews': (
        'ListInterviewsToolArguments',
        (
            'List real interview appointments, optionally for one application or by status. sequence_'
            "number is only the system's chronological appointment number, not an employer-confirmed "
            'round label.'
        ),
    ),
    'get_interview': (
        'GetInterviewToolArguments',
        (
            'Read one interview appointment and its append-only invitation, reschedule, detail-update'
            ', cancellation, and completion history.'
        ),
    ),
    'create_interview': (
        'CreateInterviewToolArguments',
        (
            'Create a user-reported real interview appointment after explicit tracking authority. It '
            'uses the uniquely focused active application when present; otherwise the runtime may cre'
            'ate a minimal tracked application from the uniquely focused active saved job, without in'
            'venting a resume, and move it to interviewing. Never attach it to an unrelated historica'
            'l JD. employer_label may be provided only when the employer explicitly used that label; '
            'never infer 一面/二面 from sequence.'
        ),
    ),
    'update_interview': (
        'UpdateInterviewToolArguments',
        (
            'Apply an explicitly user-reported reschedule, added detail, or cancellation to one exist'
            'ing interview. A reschedule updates the same appointment rather than creating another on'
            'e.'
        ),
    ),
    'complete_interview': (
        'CompleteInterviewToolArguments',
        (
            'Mark one real interview completed only after the user explicitly confirms they attended '
            'it. Time passing alone is never confirmation.'
        ),
    ),
    'record_interview_retro': (
        'RecordInterviewRetroToolArguments',
        (
            'Create a versioned post-interview report for one user-confirmed completed real interview'
            '. Use only facts the user just provided: preserve their source notes, structure remember'
            'ed questions and answers, and put missing information in limitations. Never invent inter'
            "viewer feedback or present self-assessment as the employer's decision."
        ),
    ),
    'prepare_interview': (
        'PrepareInterviewToolArguments',
        (
            'Generate or reuse a grounded preparation guide for one real upcoming interview from its '
            'exact JD snapshot, submitted resume version, confirmed evidence, and logistics. This is '
            'preparation, not a mock interview and not employer inside information.'
        ),
    ),
    'get_interview_preparation': (
        'GetInterviewPreparationToolArguments',
        (
            'Read one persisted interview preparation result without re-reading full source documents'
            '. Pass reference to read the material a specific earlier message produced, or omit it to'
            ' use the active preparation.'
        ),
    ),
    'start_mock_interview': (
        'StartMockInterviewToolArguments',
        (
            'Start one stateful mock interview for an application/interview or as explicit free pract'
            'ice (practice_scope=free). Application runs must be selected from the visible applicatio'
            'n_candidates or interview_candidates when no active application is set: use the matching'
            ' selection index. Those candidates are scoped to this conversation; do not list or infer'
            ' a global recent application just because its role/company sounds similar. If no current'
            ' application or visible selection exists, ask the user which one to use. Use free practi'
            'ce only when the user explicitly says it is not for a tracked application. Never omit th'
            'e scope/selection and let the runtime guess between these meanings. use the exact submit'
            'ted resume and immutable JD. Free practice runs only on a resume the user chose: a singl'
            'e resume attached to this message, resume_version_selection_index from the resume choice'
            ' this tool offered, or without_resume=true when the user said not to use one. With none '
            'of these the tool starts nothing and shows the user a resume choice; never pick one your'
            'self. For a specific job, pass job_selection_index from a saved-job list or the job choi'
            'ce this tool offered; a single job attached to this message is used as is. When the user'
            ' names only a company, pass company_name exactly as they said it: the tool offers that c'
            "ompany's saved jobs first, and without_job=true is the user's answer that they want comp"
            'any-only practice. Optionally bind a numbered real interview appointment for context. Af'
            'ter the first question, user answers are routed directly to the active mock-interview wo'
            'rkflow; do not call this tool again to answer.'
        ),
    ),
    'restart_mock_interview': (
        'RestartMockInterviewToolArguments',
        (
            'Retire a mock interview that reported mock_interview_checkpoint_missing or mock_intervie'
            'w_graph_incompatible and start a replacement against the same application, resume versio'
            'n, and JD. Call this only after the user agrees to abandon the stuck run: its answers st'
            'ay readable but it can never be finished. Takes no arguments.'
        ),
    ),
    'get_mock_interview_result': (
        'GetMockInterviewResultToolArguments',
        (
            'Read back a finished mock interview. Pass reference to read the exact run a specific ear'
            'lier message reported, application_selection_index for the latest run on a numbered appl'
            'ication, or omit both for the latest run on the active application. Without a question n'
            'umber this lists the questions with their ratings; with one it returns that question, th'
            "e user's full answer, its evaluation, and any follow-ups. Use this whenever the user ask"
            's about a past mock interview, since the conversation only keeps a reference to the repo'
            'rt.'
        ),
    ),
    'get_daily_brief': (
        'GetDailyBriefToolArguments',
        (
            "Generate the current user's source-grounded daily career brief from applications, recrui"
            'ting email events, real interviews, and unresolved resume-tailoring gaps. The report is '
            'computed on demand and is not stored as stale narrative memory.'
        ),
    ),
    'list_action_items': (
        'ListActionItemsToolArguments',
        (
            'Refresh and list persisted career action items such as follow-ups, pending email confirm'
            'ations, interview preparation, reminders, material requests, and retrospectives.'
        ),
    ),
    'complete_action_item': (
        'ResolveActionItemToolArguments',
        (
            'Mark one action item completed only after the user explicitly reports completing it.'
        ),
    ),
    'dismiss_action_item': (
        'ResolveActionItemToolArguments',
        (
            'Dismiss one action item only after the user explicitly says it is not applicable or shou'
            'ld be ignored.'
        ),
    ),
    'snooze_action_item': (
        'SnoozeActionItemToolArguments',
        (
            'Snooze one action item until an explicit future timestamp requested by the user.'
        ),
    ),
    'list_calendar_accounts': (
        'ListCalendarAccountsToolArguments',
        (
            'List safe Google Calendar account metadata. Credentials are never returned.'
        ),
    ),
    'list_calendar_links': (
        'ListCalendarLinksToolArguments',
        (
            "List the user's interview-to-calendar synchronization links and current sync status."
        ),
    ),
    'prepare_interview_calendar_sync': (
        'PrepareInterviewCalendarSyncToolArguments',
        (
            'Prepare a fixed create, update, or cancel preview for one real InterviewRound. This does'
            ' not write to an external calendar and must be shown to the user for approval.'
        ),
    ),
    'get_calendar_proposal': (
        'GetCalendarProposalToolArguments',
        (
            'Read one pending or historical fixed calendar-change proposal without executing it.'
            ' When the user confirms a bound proposal, read this proposal or request its execution; '
            'do not list unrelated action items. An expired proposal can still be read to explain '
            'its status, but cannot be executed.'
        ),
    ),
    'execute_calendar_proposal': (
        'ExecuteCalendarProposalToolArguments',
        (
            'Execute exactly one unchanged, unexpired calendar proposal only after the user explicitl'
            'y approves that displayed proposal. This is an external write. A failed or outcome-unkno'
            'wn execution must not be repeated or claimed successful; reconcile it, then prepare and '
            'approve a new preview.'
        ),
    ),
    'search_capabilities': (
        'SearchCapabilitiesToolArguments',
        'Search and load capabilities. Use names for known tool or namespace names, or query for one natural-language need. Names may load writes. Query ranks by relevance. Loaded tools become callable from the next decision; search itself does not execute them.',
    ),
})
MODEL_SCHEMA_ORDER: tuple[str, ...] = tuple(_SCHEMA_SPECS)


def _capability(
    name: str,
    effect: ToolEffect,
    execution_kind: ExecutionKind = "atomic_tool",
    external_write: bool = False,
    replay_safe: bool = False,
    runtime_owned: bool = False,
    notes_guarded: bool | None = None,
    preference_bound: bool = False,
    reference_readback: bool = False,
    schema_gated: bool = False,
    precondition: Precondition | None = None,
    requirement: str | None = None,
) -> CapabilityDescriptor:
    schema_spec = _SCHEMA_SPECS.get(name)
    approval_policy: ApprovalPolicy
    if runtime_owned or effect != "WRITE":
        approval_policy = "never"
    elif name in _ALWAYS_CONFIRM:
        approval_policy = "always"
    else:
        approval_policy = "owner_rule"
    replay_policy: ReplayPolicy = (
        "not_applicable"
        if effect != "WRITE"
        else ("idempotent" if replay_safe else "never")
    )
    recovery_policy: RecoveryPolicy = (
        "not_applicable"
        if effect != "WRITE"
        else ("reconcile" if external_write or not replay_safe else "retry")
    )
    return CapabilityDescriptor(
        name=name,
        arguments_model=schema_spec[0] if schema_spec is not None else None,
        description=schema_spec[1] if schema_spec is not None else None,
        effect=effect,
        execution_kind=execution_kind,
        approval_policy=approval_policy,
        replay_policy=replay_policy,
        recovery_policy=recovery_policy,
        external_write=external_write,
        runtime_owned=runtime_owned,
        notes_guarded=(effect == "WRITE" if notes_guarded is None else notes_guarded),
        preference_bound=preference_bound,
        reference_readback=reference_readback,
        schema_gated=schema_gated,
        precondition=precondition,
        requirement=requirement,
    )


def _declared_descriptors() -> Iterable[CapabilityDescriptor]:
    yield _capability("load_skill", "READ")
    yield _capability("read_conversation_span", "READ")
    yield _capability("update_working_notes", "WRITE", notes_guarded=False)
    yield _capability("fetch_archived_constraints", "READ")
    yield _capability("search_career_memory", "READ")
    yield _capability("update_owner_settings", "WRITE", replay_safe=True)
    yield _capability("get_daily_brief", "READ")
    yield _capability("list_action_items", "READ")
    yield _capability("complete_action_item", "WRITE", precondition=_reachable_via_action_item, requirement=_NEEDS_ACTION_ITEM)
    yield _capability("dismiss_action_item", "WRITE", precondition=_reachable_via_action_item, requirement=_NEEDS_ACTION_ITEM)
    yield _capability("snooze_action_item", "WRITE", precondition=_reachable_via_action_item, requirement=_NEEDS_ACTION_ITEM)
    yield _capability("open_job_search", "WRITE")
    yield _capability("find_saved_jobs", "READ", notes_guarded=True)
    yield _capability("list_resumes", "READ")
    yield _capability("list_applications", "READ")
    yield _capability("list_interviews", "READ")

    yield _capability("get_saved_job", "READ", precondition=_reachable_via_job, requirement=_NEEDS_JOB)
    yield _capability("analyze_job", "WRITE", precondition=_reachable_via_job, requirement=_NEEDS_JOB)
    yield _capability("correct_job_requirement_tier", "WRITE")
    yield _capability("compare_saved_jobs", "READ", notes_guarded=True, preference_bound=True, precondition=lambda task: bool(task.saved_job_candidates), requirement="先用 find_saved_jobs 列出可比较的岗位")
    yield _capability("research_job", "WRITE", execution_kind="workflow", precondition=_reachable_via_job, requirement=_NEEDS_JOB)
    yield _capability("retry_job_research", "WRITE", execution_kind="workflow", schema_gated=True, precondition=lambda task: bool(task.active_job_research_run_id), requirement="只能重试当前会话里已发起的公司调研")
    yield _capability("get_job_research", "READ", reference_readback=True)
    yield _capability("list_target_roles", "READ")
    yield _capability("propose_job_intent", "READ")
    yield _capability("confirm_job_intent", "WRITE", schema_gated=True, precondition=lambda task: task.pending_job_intent_update is not None, requirement="先用 propose_job_intent 展示意图变更")

    yield _capability("get_resume_metadata", "READ", precondition=lambda task: bool(task.resume_candidates), requirement="先用 list_resumes 列出简历")
    yield _capability("match_resume_to_job", "WRITE", preference_bound=True, precondition=lambda task: _reachable_via_job(task) and _reachable_via_resume_version(task) and _has_current_job_analysis(task), requirement="先用 analyze_job 分析当前 JD，并同时选定一个岗位和一个简历版本")
    yield _capability("get_resume_job_match", "READ", precondition=lambda task: bool(task.active_resume_job_match_id), requirement="先用 match_resume_to_job 完成岗位匹配")
    yield _capability("draft_resume_tailoring", "WRITE", precondition=lambda task: bool(task.active_resume_job_match_id), requirement="定制前需先用 match_resume_to_job 完成岗位匹配")
    yield _capability("get_resume_tailoring_draft", "READ", precondition=lambda task: bool(task.active_resume_tailoring_draft_id), requirement=_NEEDS_TAILORING_DRAFT)
    yield _capability("review_resume_tailoring", "WRITE", schema_gated=True, precondition=lambda task: bool(task.active_resume_tailoring_draft_id), requirement=_NEEDS_TAILORING_DRAFT)
    yield _capability("revise_resume_tailoring", "WRITE", schema_gated=True, precondition=lambda task: bool(task.active_resume_tailoring_draft_id), requirement=_NEEDS_TAILORING_DRAFT)
    yield _capability("finalize_resume_tailoring", "WRITE", schema_gated=True, precondition=lambda task: bool(task.active_resume_tailoring_draft_id), requirement=_NEEDS_TAILORING_DRAFT)
    yield _capability("export_resume_artifact", "WRITE", precondition=lambda task: bool(task.active_resume_version_id), requirement="先选定一个简历版本（定制完成后自动选定）")

    yield _capability("get_application", "READ", precondition=_reachable_via_application, requirement=_NEEDS_APPLICATION)
    yield _capability("create_application", "WRITE", replay_safe=True, preference_bound=True, precondition=_reachable_via_job, requirement=_NEEDS_JOB)
    yield _capability("update_application_status", "WRITE", precondition=_reachable_via_application, requirement=_NEEDS_APPLICATION)
    yield _capability("sync_application_emails", "WRITE", execution_kind="workflow")
    yield _capability("list_email_events", "READ")
    yield _capability("resolve_email_event", "WRITE", precondition=lambda task: bool(task.email_event_candidates), requirement="先用 list_email_events 列出邮件事件")

    yield _capability("get_interview", "READ", precondition=_reachable_via_interview, requirement=_NEEDS_INTERVIEW)
    yield _capability("create_interview", "WRITE", precondition=lambda task: _reachable_via_application(task) or _reachable_via_job(task), requirement="需要上下文唯一指向一条投递记录或一个已保存岗位；若都没有，先询问是否纳入跟踪，并请用户提供或选择公司与岗位，不能关联无关 JD")
    yield _capability("update_interview", "WRITE", precondition=_reachable_via_interview, requirement=_NEEDS_INTERVIEW)
    yield _capability("complete_interview", "WRITE", precondition=_reachable_via_interview, requirement=_NEEDS_INTERVIEW)
    yield _capability("record_interview_retro", "WRITE", precondition=_reachable_via_interview, requirement=_NEEDS_INTERVIEW)
    yield _capability("prepare_interview", "WRITE", precondition=lambda task: _reachable_via_interview(task) or bool(task.action_candidates), requirement="先选定一轮面试或一条待办事项")
    yield _capability("get_interview_preparation", "READ", reference_readback=True)
    yield _capability("list_calendar_accounts", "READ")
    yield _capability("list_calendar_links", "READ")
    yield _capability("prepare_interview_calendar_sync", "WRITE", precondition=_reachable_via_interview, requirement=_NEEDS_INTERVIEW)
    yield _capability("get_calendar_proposal", "READ", precondition=lambda task: bool(task.active_calendar_proposal_id), requirement="先用 prepare_interview_calendar_sync 生成日历预览")
    yield _capability("execute_calendar_proposal", "WRITE", external_write=True, replay_safe=True, schema_gated=True, precondition=lambda task: bool(task.active_calendar_proposal_id), requirement="先用 prepare_interview_calendar_sync 生成日历预览")
    yield _capability("start_mock_interview", "WRITE", execution_kind="workflow", precondition=lambda task: True, requirement="可直接自由练习，也可选择一条投递或面试")
    yield _capability("restart_mock_interview", "WRITE", execution_kind="workflow", schema_gated=True, precondition=lambda task: task.active_workflow == "mock_interview" and task.phase in {"mock_interview_checkpoint_missing", "mock_interview_graph_incompatible"}, requirement="只有模拟面试检查点丢失或不兼容时才能重启")
    yield _capability("get_mock_interview_result", "READ", reference_readback=True)

    yield _capability("search_career_history", "READ")
    yield _capability("search_career_episodes", "READ")
    yield _capability("get_career_memory_detail", "READ")
    yield _capability("resolve_claim_source", "READ")
    yield _capability("propose_free_text_preference_confirmation", "READ")
    yield _capability("confirm_free_text_preference", "WRITE", schema_gated=True, precondition=lambda task: task.pending_free_text_preference is not None, requirement=_NEEDS_PROPOSAL)
    yield _capability("propose_memory_amendment", "READ")
    yield _capability("confirm_memory_amendment", "WRITE", schema_gated=True, precondition=lambda task: task.pending_memory_amendment is not None, requirement=_NEEDS_PROPOSAL)
    yield _capability("propose_memory_tombstone", "READ")
    yield _capability("confirm_memory_tombstone", "WRITE", schema_gated=True, precondition=lambda task: task.pending_memory_tombstone is not None, requirement=_NEEDS_PROPOSAL)
    yield _capability("propose_career_fact", "WRITE")
    yield _capability("confirm_career_fact", "WRITE", schema_gated=True, precondition=lambda task: task.pending_career_fact is not None, requirement=_NEEDS_PROPOSAL)
    yield _capability("propose_constraint_retirement", "READ")
    yield _capability("confirm_constraint_retirement", "WRITE", schema_gated=True, precondition=lambda task: task.pending_constraint_retirement is not None, requirement=_NEEDS_PROPOSAL)

    yield _capability("handle_mock_interview_input", "WRITE", execution_kind="runtime_workflow", runtime_owned=True)
    yield _capability("retry_mock_interview", "WRITE", execution_kind="runtime_workflow", runtime_owned=True)
    yield _capability("search_capabilities", "CONTROL")


def _descriptors() -> Iterable[CapabilityDescriptor]:
    """Attach discovery metadata without changing provider-facing schemas.

    Namespaces are loading units, not authorization grants. Successors only
    suggest likely next tools after a completed call; execution checks still
    apply to each call independently.

    A read never suggests an ungated write: having looked at a job does not
    make creating an application a likely next step, and offering it invites
    acting instead of asking. A read may lead to a write only when that write
    is schema-gated, i.e. offered only while a pending proposal or draft exists.
    """
    groups = {
        "context": (
            ("load_skill", "读取当前任务所需的操作说明。", ("加载技能", "读取操作指南"), ()),
            ("read_conversation_span", "读取当前对话中摘要遗漏的消息片段。", ("读取对话片段", "找回聊天原文"), ()),
            ("update_working_notes", "更新未确认的工作便笺，仅供提问和表达参考。", ("更新工作便笺", "记录临时笔记"), ()),
            ("fetch_archived_constraints", "取回摘要未展示但仍有效的对话约束。", ("取回归档约束", "查看旧约束"), ()),
            ("search_career_memory", "分页查找当前窗口外的已确认职业事实。", ("搜索已确认经历", "查找职业记忆"), ()),
            ("update_owner_settings", "按用户要求更新助手的持久设置。", ("修改助手设置", "更新主人偏好设置"), ()),
        ),
        "actions": (
            ("get_daily_brief", "读取今天的行动摘要。", ("查看今日简报", "今天做什么"), ()),
            ("list_action_items", "列出待办事项并取得后续操作所需的选择项。", ("列出待办", "查看行动事项"), ()),
            ("complete_action_item", "把已选待办标记为完成。", ("完成待办", "标记行动完成"), ()),
            ("dismiss_action_item", "撤销已选的不再需要的待办。", ("忽略待办", "移除行动事项"), ()),
            ("snooze_action_item", "推迟已选待办的提醒时间。", ("暂缓待办", "待办稍后提醒"), ()),
        ),
        "job.library": (
            ("open_job_search", "按用户给定的条件打开招聘网站搜索页，结果由用户浏览。", ("打开岗位搜索", "去招聘网站找职位"), ()),
            ("find_saved_jobs", "查找已收藏岗位；用于定位或选择岗位。", ("查找收藏岗位", "列出保存的职位"), ("get_saved_job", "compare_saved_jobs")),
            ("get_saved_job", "读取已选岗位的职位信息。", ("查看收藏岗位详情", "读取职位描述"), ()),
            ("compare_saved_jobs", "比较多条已收藏岗位，不创建投递。", ("比较收藏岗位", "多个职位对比"), ()),
            ("list_target_roles", "列出用户已记录的目标岗位方向和选择项。", ("查看目标职位", "列出求职方向"), ()),
        ),
        "job.analysis": (
            ("analyze_job", "分析已选岗位的职位要求；已有分析时先检查是否仍对应当前 JD。", ("分析岗位要求", "解析职位描述"), ("match_resume_to_job",)),
            ("correct_job_requirement_tier", "修正岗位分析中某项要求的优先级。", ("修正岗位要求等级", "调整职位要求层级"), ()),
        ),
        "job.research": (
            ("research_job", "用户需要公司或岗位调研时启动调研。", ("调研公司", "研究岗位背景"), ("get_job_research",)),
            ("retry_job_research", "重试当前会话中失败的岗位调研。", ("重试公司调研", "重新运行岗位研究"), ("get_job_research",)),
            ("get_job_research", "按可用引用读取已生成的岗位调研报告。", ("读取公司调研报告", "查看岗位研究结果"), ()),
        ),
        "job.intent": (
            ("propose_job_intent", "先展示用户明确表达的求职意向变更，等待确认。", ("提出求职意向", "预览目标岗位变更"), ("confirm_job_intent",)),
            ("confirm_job_intent", "用户明确同意后保存已展示的求职意向。", ("确认求职意向", "保存目标岗位意向"), ()),
        ),
        "resume.library": (
            ("list_resumes", "列出简历供选择；已有绑定简历时无需再次列出。", ("列出简历", "选择简历版本"), ("get_resume_metadata",)),
            ("get_resume_metadata", "读取已列出简历的版本和元数据。", ("查看简历信息", "读取简历元数据"), ()),
            ("export_resume_artifact", "导出已选或已完成定制的简历文件。", ("导出简历", "下载简历文件"), ()),
        ),
        "resume.match": (
            ("match_resume_to_job", "已有当前岗位分析和简历版本时，直接计算匹配。", ("简历匹配", "对比简历和岗位", "匹配度"), ("get_resume_job_match", "draft_resume_tailoring")),
            ("get_resume_job_match", "读取已生成的简历与岗位匹配结果。", ("查看简历匹配结果", "读取岗位匹配报告"), ()),
        ),
        "resume.tailoring": (
            ("draft_resume_tailoring", "根据已有匹配结果生成定制简历草稿。", ("起草定制简历", "生成针对岗位的简历"), ("get_resume_tailoring_draft", "review_resume_tailoring")),
            ("get_resume_tailoring_draft", "读取当前定制简历草稿。", ("查看定制简历草稿", "读取简历修改稿"), ("review_resume_tailoring",)),
            ("review_resume_tailoring", "检查定制草稿的证据和质量。", ("审查定制简历", "检查简历草稿"), ("revise_resume_tailoring", "finalize_resume_tailoring")),
            ("revise_resume_tailoring", "根据反馈修改当前定制简历草稿。", ("修改定制简历", "调整简历草稿"), ("review_resume_tailoring", "finalize_resume_tailoring")),
            ("finalize_resume_tailoring", "确认并生成定制简历的最终版本。", ("完成定制简历", "定稿岗位简历"), ("export_resume_artifact",)),
        ),
        "application.tracking": (
            ("list_applications", "列出投递记录并取得选择项。", ("列出投递记录", "查看已投岗位"), ("get_application",)),
            ("get_application", "读取已选投递记录的当前状态。", ("查看投递详情", "读取申请记录"), ("list_email_events",)),
            ("create_application", "经用户授权后为已选岗位创建投递记录。", ("创建投递", "登记求职申请"), ("get_application",)),
            ("update_application_status", "更新已选投递记录的状态。", ("更新投递状态", "修改申请进度"), ()),
        ),
        "application.email": (
            ("sync_application_emails", "同步与投递相关的邮件事件。", ("同步投递邮件", "抓取申请邮件"), ("list_email_events",)),
            ("list_email_events", "列出可关联的投递邮件事件。", ("列出邮件事件", "查看投递邮件"), ()),
            ("resolve_email_event", "处理已选邮件事件与投递的关联。", ("处理邮件事件", "确认投递邮件关联"), ()),
        ),
        "interview.schedule": (
            ("list_interviews", "列出面试轮次并取得选择项。", ("列出面试", "查看面试安排"), ("get_interview",)),
            ("get_interview", "读取已选面试轮次的信息。", ("查看面试详情", "读取面试轮次"), ()),
            ("create_interview", "为有明确关联的岗位或投递创建面试轮次。", ("创建面试", "记录新面试"), ("get_interview",)),
            ("update_interview", "修改已选面试轮次的信息。", ("更新面试", "调整面试记录"), ()),
            ("complete_interview", "将已选面试轮次标记为完成。", ("完成面试", "标记面试结束"), ("record_interview_retro",)),
            ("record_interview_retro", "记录已选面试轮次的复盘。", ("记录面试复盘", "写面试回顾"), ()),
        ),
        "interview.prep": (
            ("prepare_interview", "为已选面试或待办生成面试准备材料。", ("准备面试", "生成面试准备"), ("get_interview_preparation",)),
            ("get_interview_preparation", "读取已生成的面试准备材料。", ("查看面试准备", "读取面试准备报告"), ()),
        ),
        "interview.calendar": (
            ("list_calendar_accounts", "列出可用的日历账户。", ("列出日历账户", "查看可用日历"), ()),
            ("list_calendar_links", "查看面试与日历事件的现有关联。", ("查看日历关联", "列出面试日历链接"), ()),
            ("prepare_interview_calendar_sync", "为已选面试生成日历变更预览，不执行写入。", ("预览面试日历同步", "准备日历变更"), ("get_calendar_proposal", "execute_calendar_proposal")),
            ("get_calendar_proposal", "读取指定的日历变更提案及其有效状态。", ("查看日历提案", "读取日历预览"), ("execute_calendar_proposal",)),
            ("execute_calendar_proposal", "用户批准具体提案后执行日历变更。", ("执行日历同步", "确认写入日历"), ()),
        ),
        "interview.mock": (
            ("start_mock_interview", "启动一场模拟面试练习。", ("开始模拟面试", "练习面试问答"), ("get_mock_interview_result",)),
            ("restart_mock_interview", "仅在模拟面试检查点丢失或不兼容时重启。", ("重启模拟面试", "恢复面试练习"), ()),
            ("get_mock_interview_result", "读取已完成的模拟面试结果。", ("查看模拟面试结果", "读取练习反馈"), ()),
        ),
        "memory.search": (
            ("search_career_history", "查找被修订或撤销的历史职业事实。", ("搜索职业事实历史", "查找旧版经历"), ()),
            ("search_career_episodes", "查找跨对话的已完成求职事件摘要。", ("搜索求职事件", "查找历史投递面试"), ()),
            ("get_career_memory_detail", "读取当前职业事实的细节和修订链。", ("查看职业事实详情", "读取经历修订记录"), ("resolve_claim_source",)),
            ("resolve_claim_source", "读取已确认职业事实所引用的原始简历证据。", ("追溯经历来源", "查看事实原文"), ()),
        ),
        "memory.proposals": (
            ("propose_free_text_preference_confirmation", "展示待确认的自由文本偏好，不立即启用。", ("提出偏好确认", "预览长期偏好"), ("confirm_free_text_preference",)),
            ("confirm_free_text_preference", "用户同意后启用已展示的自由文本偏好。", ("确认自由文本偏好", "保存长期偏好"), ()),
            ("propose_memory_amendment", "展示对当前职业事实的字段级修订提案。", ("提出记忆修订", "预览经历更正"), ("confirm_memory_amendment",)),
            ("confirm_memory_amendment", "用户同意后写入已展示的职业事实修订。", ("确认记忆修订", "保存经历更正"), ()),
            ("propose_memory_tombstone", "展示职业事实的删除提案，不立即删除。", ("提出记忆删除", "预览经历删除"), ("confirm_memory_tombstone",)),
            ("confirm_memory_tombstone", "用户同意后执行已展示的职业事实删除。", ("确认记忆删除", "删除职业事实"), ()),
            ("propose_career_fact", "把用户明确陈述的职业事实作为待确认提案展示。", ("提出职业事实", "预览新增经历"), ("confirm_career_fact",)),
            ("confirm_career_fact", "用户同意后保存已展示的职业事实。", ("确认职业事实", "保存新增经历"), ()),
            ("propose_constraint_retirement", "展示停用现有对话约束的提案。", ("提出约束退役", "预览取消限制"), ("confirm_constraint_retirement",)),
            ("confirm_constraint_retirement", "用户同意后停用已展示的对话约束。", ("确认约束退役", "取消旧限制"), ()),
        ),
        "control": (
            ("search_capabilities", "按描述或名称搜索可用能力，不执行搜索结果。", ("查找工具", "搜索能力"), ()),
        ),
    }
    metadata = {
        name: (namespace, summary, aliases, successors)
        for namespace, rows in groups.items()
        for name, summary, aliases, successors in rows
    }
    if len(metadata) != sum(len(rows) for rows in groups.values()):
        raise RuntimeError("duplicate capability discovery metadata")
    declared_names: set[str] = set()
    for descriptor in _declared_descriptors():
        if not descriptor.model_callable:
            yield descriptor
            continue
        declared_names.add(descriptor.name)
        if descriptor.name not in metadata:
            raise RuntimeError(f"capability lacks discovery metadata: {descriptor.name}")
        namespace, summary, aliases, successors = metadata[descriptor.name]
        yield replace(
            descriptor,
            namespace=namespace,
            summary=summary,
            aliases_zh=aliases,
            example_queries=EXAMPLE_QUERIES.get(descriptor.name, ()),
            successors=successors,
        )
    if extra := metadata.keys() - declared_names:
        raise RuntimeError(f"discovery metadata names unknown capabilities: {sorted(extra)}")
    if extra := EXAMPLE_QUERIES.keys() - declared_names:
        raise RuntimeError(f"example queries name unknown capabilities: {sorted(extra)}")


def _build_catalog() -> Mapping[str, CapabilityDescriptor]:
    result: dict[str, CapabilityDescriptor] = {}
    namespace_sizes: dict[str, int] = {}
    alias_owners: dict[str, str] = {}
    for descriptor in _descriptors():
        if descriptor.name in result:
            raise RuntimeError(f"duplicate Main Agent capability: {descriptor.name}")
        if descriptor.external_write and descriptor.effect != "WRITE":
            raise RuntimeError(f"external capability must be WRITE: {descriptor.name}")
        if descriptor.replay_policy == "idempotent" and descriptor.effect != "WRITE":
            raise RuntimeError(f"replay-safe capability must be WRITE: {descriptor.name}")
        if descriptor.effect != "WRITE" and descriptor.replay_policy != "not_applicable":
            raise RuntimeError(f"non-WRITE capability cannot declare replay: {descriptor.name}")
        if descriptor.effect == "WRITE" and descriptor.replay_policy == "not_applicable":
            raise RuntimeError(f"WRITE capability needs a replay policy: {descriptor.name}")
        if descriptor.effect != "WRITE" and descriptor.recovery_policy != "not_applicable":
            raise RuntimeError(f"non-WRITE capability cannot declare recovery: {descriptor.name}")
        if descriptor.effect == "WRITE" and descriptor.recovery_policy == "not_applicable":
            raise RuntimeError(f"WRITE capability needs a recovery policy: {descriptor.name}")
        if descriptor.external_write and descriptor.recovery_policy != "reconcile":
            raise RuntimeError(f"external write must reconcile: {descriptor.name}")
        if descriptor.approval_policy != "never" and descriptor.effect != "WRITE":
            raise RuntimeError(f"only WRITE capabilities may require approval: {descriptor.name}")
        if descriptor.runtime_owned and descriptor.approval_policy != "never":
            raise RuntimeError(f"runtime-owned capability cannot await owner approval: {descriptor.name}")
        if descriptor.model_callable and descriptor.effect == "WRITE" and descriptor.approval_policy == "never":
            raise RuntimeError(f"model-callable WRITE must honor owner approval rules: {descriptor.name}")
        if descriptor.external_write and descriptor.approval_policy != "always":
            raise RuntimeError(f"external write must always require approval: {descriptor.name}")
        if descriptor.runtime_owned != (descriptor.execution_kind == "runtime_workflow"):
            raise RuntimeError(f"runtime ownership and execution kind disagree: {descriptor.name}")
        if descriptor.model_callable:
            if not descriptor.namespace or not descriptor.namespace.strip():
                raise RuntimeError(f"model-callable capability needs a namespace: {descriptor.name}")
            if not descriptor.summary or not descriptor.summary.strip():
                raise RuntimeError(f"model-callable capability needs a summary: {descriptor.name}")
            if not 2 <= len(descriptor.aliases_zh) <= 5:
                raise RuntimeError(f"model-callable capability needs 2-5 Chinese aliases: {descriptor.name}")
            searchable = descriptor.name != "search_capabilities"
            if searchable and not 5 <= len(descriptor.example_queries) <= 10:
                raise RuntimeError(f"searchable capability needs 5-10 example queries: {descriptor.name}")
            if not searchable and descriptor.example_queries:
                raise RuntimeError(f"excluded capability has example queries: {descriptor.name}")
            for example in descriptor.example_queries:
                normalized = example.strip()
                if not 4 <= len(normalized) <= 40:
                    raise RuntimeError(f"example query must have 4-40 characters: {descriptor.name}")
                if normalized == descriptor.name or normalized in descriptor.aliases_zh:
                    raise RuntimeError(f"example query duplicates name or alias: {descriptor.name}")
            namespace_sizes[descriptor.namespace] = namespace_sizes.get(descriptor.namespace, 0) + 1
            if namespace_sizes[descriptor.namespace] > 10:
                raise RuntimeError(f"capability namespace exceeds ten tools: {descriptor.namespace}")
            for alias in descriptor.aliases_zh:
                normalized = alias.strip()
                if not normalized:
                    raise RuntimeError(f"empty Chinese alias: {descriptor.name}")
                owner = alias_owners.setdefault(normalized, descriptor.name)
                if owner != descriptor.name:
                    raise RuntimeError(
                        f"Chinese alias {normalized!r} belongs to both {owner} and {descriptor.name}"
                    )
        elif descriptor.namespace or descriptor.summary or descriptor.aliases_zh or descriptor.example_queries or descriptor.successors:
            raise RuntimeError(f"runtime-only capability has discovery metadata: {descriptor.name}")
        if descriptor.model_callable and descriptor.name not in _SCHEMA_SPECS:
            raise RuntimeError(f"model-callable capability needs a tool schema: {descriptor.name}")
        if not descriptor.model_callable and descriptor.name in _SCHEMA_SPECS:
            raise RuntimeError(f"runtime-only capability cannot expose a tool schema: {descriptor.name}")
        if (descriptor.precondition is None) != (descriptor.requirement is None):
            raise RuntimeError(f"precondition and requirement must be declared together: {descriptor.name}")
        if descriptor.schema_gated and descriptor.precondition is None:
            raise RuntimeError(f"schema-gated capability needs a precondition: {descriptor.name}")
        result[descriptor.name] = descriptor
    for descriptor in result.values():
        for successor in descriptor.successors:
            target = result.get(successor)
            if target is None or not target.model_callable or successor == descriptor.name:
                raise RuntimeError(
                    f"invalid capability successor {descriptor.name} -> {successor}"
                )
            if (
                descriptor.effect != "WRITE"
                and target.effect == "WRITE"
                and not target.schema_gated
            ):
                raise RuntimeError(
                    f"a read cannot suggest an ungated write: {descriptor.name} -> {successor}"
                )
    return MappingProxyType(result)


CAPABILITIES: Mapping[str, CapabilityDescriptor] = _build_catalog()


def capability(name: str) -> CapabilityDescriptor:
    try:
        return CAPABILITIES[name]
    except KeyError as error:
        raise ValueError(f"Unknown Main Agent capability: {name}") from error
