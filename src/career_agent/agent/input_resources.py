"""Resolve the structured inputs a chat turn names into verified context.

A request may attach ``{"kind": "resume_version", "id": ...}``,
``{"kind": "job_posting", "id": ...}`` or ``{"kind": "jd_snapshot", "id": ...}``
(one exact JD version of a saved job). The id is a claim, not a fact: before
any of the asset reaches the model this module looks it up under the
authenticated user, so a version or posting belonging to someone else, or one
that no longer exists, is refused as unknown rather than silently dropped.
What comes back is the controlled shape the model may see — metadata and a
bounded excerpt for a resume, the library row for a job — never the file. The bound
is per turn, not per attachment: ``ATTACHED_RESUME_EXCERPT_CHARS`` is shared
across everything attached, so eight resumes cannot project eight budgets.
"""

from __future__ import annotations

from career_agent.agent.main_agent_contracts import (
    ATTACHED_RESUME_EXCERPT_CHARS,
    AttachedResumeContext,
    SavedJobCandidateContextItem,
)
from career_agent.agent.summary_text import condense
from career_agent.harness.streaming import TurnInputResource
from career_agent.services.resume_import import resume_document_text
from career_agent.services.resume_text import ResumeTextService
from career_agent.storage.jobs import JobPostingRepository
from career_agent.storage.resumes import ResumeStore
from career_agent.services.applications import ApplicationService


def saved_job_title(title: str, company_name: str) -> str:
    return condense(f"{company_name.strip()}｜{title.strip()}", limit=80)


def saved_job_description(version: int, source_name: str) -> str:
    return condense(f"JD 第 {version} 版 · {source_name.strip()}", limit=200)


class InputResourceNotFoundError(LookupError):
    """A named input does not exist for the authenticated user."""

    def __init__(self, resource: TurnInputResource) -> None:
        super().__init__(
            f"{resource.kind} '{resource.id}' was not found or does not belong "
            "to the current user"
        )
        self.resource = resource


class InputResourceUnavailableError(RuntimeError):
    """Inputs were attached but nothing here can resolve them."""


class InputResourceRejectedError(ValueError):
    """Inputs were attached to a turn that cannot take them.

    A workflow that owns the conversation (a running mock interview) reads the
    message itself and never sees the attached context, so accepting the
    attachment would mean quietly ignoring it. Refusing is the honest answer.
    """


def resolve_application_input_resource(
    service: ApplicationService | None,
    *,
    user_id: str,
    resources: tuple[TurnInputResource, ...],
) -> tuple[str | None, str | None]:
    """Verify the attached application belongs to the authenticated user."""
    selected = tuple(resource for resource in resources if resource.kind == "application")
    if not selected:
        return (None, None)
    if len(selected) > 1:
        raise ValueError("only one application may be attached to a turn")
    if service is None:
        raise InputResourceUnavailableError("application inputs cannot be resolved")
    resource = selected[0]
    try:
        detail = service.get_application(user_id=user_id, application_id=resource.id)
    except Exception as error:
        raise InputResourceNotFoundError(resource) from error
    return detail.application.id, detail.application.status


def excerpt_budgets(lengths: tuple[int, ...], total: int) -> tuple[int, ...]:
    """Split ``total`` characters across texts of the given ``lengths``.

    Water-filling: every text gets an equal share, and a text shorter than its
    share hands the remainder to the longer ones. Two short resumes and one
    long one therefore still give the long one most of the budget, while the
    sum never exceeds ``total``.
    """
    budgets = [0] * len(lengths)
    remaining = total
    pending = sorted(range(len(lengths)), key=lambda index: lengths[index])
    while pending:
        share = remaining // len(pending)
        index = pending.pop(0)
        grant = min(lengths[index], share)
        budgets[index] = grant
        remaining -= grant
    return tuple(budgets)


def resolve_job_input_resources(
    repository: JobPostingRepository | None,
    *,
    user_id: str,
    resources: tuple[TurnInputResource, ...],
) -> tuple[SavedJobCandidateContextItem, ...]:
    """Verify each named posting belongs to ``user_id`` and return its display row."""
    jobs = tuple(
        resource
        for resource in resources
        if resource.kind in ("job_posting", "jd_snapshot")
    )
    if not jobs:
        return ()
    if repository is None:
        raise InputResourceUnavailableError(
            "job inputs cannot be resolved without a job repository"
        )
    attached: list[SavedJobCandidateContextItem] = []
    for resource in jobs:
        if resource.kind == "jd_snapshot":
            snapshot = repository.get_snapshot(
                user_id=user_id, jd_snapshot_id=resource.id
            )
            if snapshot is None:
                raise InputResourceNotFoundError(resource)
            record = repository.get_job(
                user_id=user_id, job_posting_id=snapshot.job_posting_id
            )
        else:
            record = repository.get_job(user_id=user_id, job_posting_id=resource.id)
            snapshot = record.snapshot if record is not None else None
        if record is None or snapshot is None:
            raise InputResourceNotFoundError(resource)
        attached.append(
            SavedJobCandidateContextItem(
                job_posting_id=record.posting.id,
                title=record.posting.title[:200],
                company_name=record.posting.company_name[:200],
                city=record.city,
                salary=record.salary,
                jd_snapshot_id=snapshot.id,
                jd_version=snapshot.version,
            )
        )
    return tuple(attached)


def resolve_input_resources(
    store: ResumeStore | None,
    *,
    user_id: str,
    resources: tuple[TurnInputResource, ...],
    text_service: ResumeTextService | None = None,
) -> tuple[AttachedResumeContext, ...]:
    resources = tuple(
        resource for resource in resources if resource.kind == "resume_version"
    )
    if not resources:
        return ()
    if store is None:
        raise InputResourceUnavailableError(
            "resume inputs cannot be resolved without a resume store"
        )
    located_versions = []
    texts: list[str | None] = []
    for resource in resources:
        located = store.get_version(user_id=user_id, resume_version_id=resource.id)
        if located is None:
            raise InputResourceNotFoundError(resource)
        resume, version = located
        document = (text_service.ensure if text_service is not None else store.read_version_document)(
            user_id=user_id, resume_version_id=version.id
        )
        located_versions.append((resume, version))
        texts.append(
            resume_document_text(document)
            if document is not None
            else None
        )
    budgets = excerpt_budgets(
        tuple(len(text) if text is not None else 0 for text in texts),
        ATTACHED_RESUME_EXCERPT_CHARS,
    )
    attached: list[AttachedResumeContext] = []
    for (resume, version), text, budget in zip(located_versions, texts, budgets):
        role = store.get_target_role(
            user_id=user_id, target_role_id=resume.target_role_id
        )
        excerpt = text
        truncated = False
        if text is not None and len(text) > budget:
            excerpt = text[:budget].rstrip() or None
            truncated = True
        attached.append(
            AttachedResumeContext(
                resume_version_id=version.id,
                resume_id=resume.id,
                resume_name=resume.name[:200],
                version_number=version.version_number,
                is_latest_version=resume.latest_version_id == version.id,
                target_role=role.title[:200] if role is not None else None,
                document_format=version.document_format,
                byte_size=version.byte_size,
                uploaded_at=version.created_at,
                excerpt=excerpt,
                excerpt_truncated=truncated,
                text_unavailable=text is None,
            )
        )
    return tuple(attached)
