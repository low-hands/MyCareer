from __future__ import annotations

from collections.abc import Mapping
import hashlib
import hmac
from datetime import datetime, timedelta, timezone
import json
import re
from typing import Annotated, Any, Literal, NamedTuple, Protocol, get_args

from pydantic import AliasChoices, Field, field_validator, model_validator

from career_agent.agent.support.summary_text import DELIVERY_SUMMARY_LIMIT, clamp
from career_agent.agent.presentation.delivery_policy import is_failed, is_waiting
from career_agent.agent.presentation.body_contracts import (
    BodyDependency,
    DeliveredBodySource,
)
from career_agent.agent.contracts.memory import (
    SUMMARY_ITEM_MAX_CHARS,
    SUMMARY_SOURCE_MAX_CHARS,
    ConversationSummaryContent,
)
from career_agent.agent.providers.token_budget import serialized_token_count
from career_agent.agent.capabilities.effects import approval_policy, owner_rule_capabilities
from career_agent.agent.contracts.questionnaire import PendingQuestionnaire, UserQuestion
from career_agent.domain.applications import ApplicationStatus
from career_agent.domain.action_center import ActionSourceType, ActionStatus, ActionType
from career_agent.domain.email_tracking import EmailEventStatus
from career_agent.domain.interviews import (
    InterviewDetails,
    InterviewRetroQuestion,
    InterviewSelfAssessment,
    InterviewStatus,
)
from career_agent.domain.job_discovery import ContractModel
from career_agent.domain.job_research.models import company_key
from career_agent.domain.mock_interviews import MockInterviewType

from career_agent.agent.contracts.candidates import SavedJobCandidateContextItem
from career_agent.agent.contracts.resources import ConversationResourceReference
from career_agent.agent.contracts.context import MainAgentContext
from career_agent.agent.contracts.tools.core_memory import OpenJobSearchToolArguments
from career_agent.agent.contracts.tools.job import *
from career_agent.agent.contracts.decisions import _reject_internal_identifiers
from career_agent.agent.contracts.projections.common import _pinned_jd_snapshot_id

def project_open_job_search_arguments(
    context: MainAgentContext, arguments: dict[str, Any]
) -> dict[str, Any]:
    _reject_internal_identifiers("open_job_search", arguments)
    model_arguments = OpenJobSearchToolArguments.model_validate(arguments)
    # Falls back to the person-level default only. A target role's city override
    # is deliberately not consulted here: task state has no notion of which role
    # the conversation is working in, so the only available rule would be "any
    # role that happens to have a city", which would silently search the wrong
    # place. Resolving it properly needs an active target role first.
    return {
        **model_arguments.model_copy(
            update={"city": model_arguments.city or context.profile.default_city}
        ).model_dump(),
        # The search is opened on behalf of this conversation; the capture
        # intent it creates has to remember which one, or the job the user
        # saves from it cannot find its way back.
        "user_id": context.profile.user_id,
        "conversation_id": context.conversation_id,
    }


def _report_is_about(
    held: ConversationResourceReference, candidate: SavedJobCandidateContextItem
) -> bool:
    """Whether a held report and a saved job name the same employer.

    Decided on identity when the reference carries it: the anchoring job or the
    company key the report is stored under, which the saved job's formal name
    folds to the same way. Missing identity is not reconstructed from a title.
    """
    if held.job_posting_id is not None or held.company_key is not None:
        if held.job_posting_id == candidate.job_posting_id:
            return True
        return (
            held.company_key is not None
            and bool(candidate.company_name.strip())
            and company_key(candidate.company_name) == held.company_key
        )
    return False


_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")


def _company_mention(company_name: str, user_message: str) -> str | None:
    """The longest leading part of ``company_name`` the message writes out.

    Users shorten employers from the front — "字节" for "字节跳动", "ByteDance"
    for "ByteDance Ltd" — so a leading part of at least two CJK characters, or
    three otherwise, is taken as naming the company. This only decides what
    the request is *about*, never which report answers it.
    """
    name = company_key(company_name) if company_name.strip() else ""
    message = " ".join(user_message.split()).casefold()
    for end in range(len(name), 1, -1):
        prefix = name[:end].rstrip()
        minimum = 2 if _CJK.search(prefix) else 3
        if len(prefix) >= minimum and prefix in message:
            return prefix
    return None


class _AskedCompanies(NamedTuple):
    jobs: tuple[tuple[int, SavedJobCandidateContextItem], ...]
    ambiguous_mention: str | None


def _companies_asked_for(
    context: MainAgentContext,
    held: ConversationResourceReference,
) -> _AskedCompanies:
    """The saved jobs this turn's request is about, as (selection_index, job).

    A search the model ran this turn already resolved the user's wording —
    short name, alias, or otherwise — to job entities, so when every job it
    returned belongs to one employer, that employer is what the user asked
    about. Absent such a search, a company counts when the message writes out
    its name or a leading part of it; a mixed result list says nothing about
    which company was meant. A leading part that heads more than one company
    — "中国" for both 中国移动 and 中国银行, or a saved job and the held report
    alike — names none of them, and is reported as ambiguous instead.
    """
    numbered = tuple(enumerate(context.task.saved_job_candidates, start=1))
    searched_this_turn = any(
        observation.tool_name == "find_saved_jobs"
        and observation.state == "saved_jobs_found"
        for observation in context.tool_observations
    )
    if searched_this_turn and len(
        {
            company_key(candidate.company_name)
            for _, candidate in numbered
            if candidate.company_name.strip()
        }
    ) == 1:
        return _AskedCompanies(numbered, None)
    held_company = held.company_key or ""
    asked: list[tuple[int, SavedJobCandidateContextItem]] = []
    companies_by_mention: dict[str, set[str]] = {}
    for index, candidate in numbered:
        mention = _company_mention(candidate.company_name, context.user_message)
        if mention is None:
            continue
        asked.append((index, candidate))
        named = companies_by_mention.setdefault(mention, set())
        named.add(company_key(candidate.company_name))
        if held_company.startswith(mention):
            named.add(held_company)
    ambiguous = next(
        (mention for mention, named in companies_by_mention.items() if len(named) > 1),
        None,
    )
    return _AskedCompanies(tuple(asked), ambiguous)


def _reject_borrowed_report_reference(
    context: MainAgentContext, *, reference: str, report_id: str
) -> None:
    """A report handle is bound to a company; the request must be about it.

    ``resolve_reference`` proves the handle was issued, not that it is the one
    the user meant. When the request is about a saved-job company and the held
    report is about a different one, the read would answer about the wrong
    company while looking grounded, so it is refused in favour of the company's
    own selection index. A request that also covers the report's company, or
    is about no saved-job company at all, is left to the model; one whose short
    name fits several companies is refused until the user says which.
    """
    held = next(
        item for item in context.referenced_resources()
        if item.resource_id == report_id
    )
    asked, ambiguous = _companies_asked_for(context, held)
    if not asked:
        return
    selectors = "、".join(
        f"{index}（{candidate.company_name}）" for index, candidate in asked
    )
    if ambiguous is not None:
        raise ValueError(
            f"'{ambiguous}' names more than one saved company, so resource "
            f"reference '{reference}' cannot be read as the one the user meant; "
            f"ask which company is meant (saved: selection_index {selectors}) "
            "instead of guessing"
        )
    if any(_report_is_about(held, candidate) for _, candidate in asked):
        return
    about = "bound to another company"
    raise ValueError(
        f"resource reference '{reference}' is {about}, not the company "
        f"the user asked about; use selection_index {selectors} or say that "
        "report is not reachable. A company named by a short name or alias "
        "must be resolved with find_saved_jobs first, never by guessing which "
        "title it means"
    )


def project_job_research_arguments(
    context: MainAgentContext,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    _reject_internal_identifiers(name, arguments)
    if name == "research_job":
        model_arguments = ResearchJobToolArguments.model_validate(arguments)
        payload = model_arguments.model_dump()
        selection_index = payload.pop("selection_index", None)
        job_posting_id = context.task.active_job_posting_id
        if selection_index is not None:
            if not 1 <= selection_index <= len(context.task.saved_job_candidates):
                raise ValueError("saved-job selection index is out of range")
            job_posting_id = context.task.saved_job_candidates[
                selection_index - 1
            ].job_posting_id
        if job_posting_id is None:
            raise ValueError("research_job requires a selected or active saved job")
        payload["job_posting_id"] = job_posting_id
        payload["jd_snapshot_id"] = _pinned_jd_snapshot_id(
            context, job_posting_id=job_posting_id, explicit_selection=selection_index is not None
        )
    elif name == "retry_job_research":
        RetryJobResearchToolArguments.model_validate(arguments)
        if context.task.active_job_research_run_id is None:
            raise ValueError("retry_job_research requires an active failed run")
        payload = {"run_id": context.task.active_job_research_run_id}
    elif name == "get_job_research":
        model_arguments = GetJobResearchToolArguments.model_validate(arguments)
        selection_index = model_arguments.selection_index
        if model_arguments.reference is not None:
            report_id = context.resolve_reference(
                reference=model_arguments.reference,
                kind="job_research_report",
            )
            _reject_borrowed_report_reference(
                context, reference=model_arguments.reference, report_id=report_id
            )
            payload = {"report_id": report_id}
        elif selection_index is not None:
            if not 1 <= selection_index <= len(context.task.saved_job_candidates):
                raise ValueError("saved-job selection index is out of range")
            payload = {
                "job_posting_id": context.task.saved_job_candidates[
                    selection_index - 1
                ].job_posting_id
            }
        elif context.task.active_job_research_report_id is not None:
            # No selector: the active report stands in for "the report". The
            # request travels with it so the read can be refused when the
            # user was asking about a different company.
            payload = {
                "report_id": context.task.active_job_research_report_id,
                IMPLICIT_REQUEST_KEY: context.user_message,
            }
        elif context.task.active_job_posting_id is not None:
            payload = {
                "job_posting_id": context.task.active_job_posting_id,
                IMPLICIT_REQUEST_KEY: context.user_message,
            }
        else:
            raise ValueError("get_job_research requires an active research report or job")
    else:
        raise ValueError(f"Unknown job research tool: {name}")
    return {"user_id": context.profile.user_id, **payload}

