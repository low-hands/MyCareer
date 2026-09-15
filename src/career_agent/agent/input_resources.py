"""Resolve the structured inputs a chat turn names into verified context.

A request may attach ``{"kind": "resume_version", "id": ...}``. The id is a
claim, not a fact: before any of the resume reaches the model this module
looks the version up under the authenticated user, so a version belonging to
someone else, or to a resume that no longer exists, is refused as unknown
rather than silently dropped. What comes back is the controlled shape the
model may see — metadata and a bounded excerpt — never the file. The bound
is per turn, not per attachment: ``ATTACHED_RESUME_EXCERPT_CHARS`` is shared
across everything attached, so eight resumes cannot project eight budgets.
"""

from __future__ import annotations

from career_agent.agent.main_agent_contracts import (
    ATTACHED_RESUME_EXCERPT_CHARS,
    AttachedResumeContext,
)
from career_agent.harness.streaming import TurnInputResource
from career_agent.services.resume_import import extract_resume_text
from career_agent.storage.resumes import ResumeStore


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


def resolve_input_resources(
    store: ResumeStore | None,
    *,
    user_id: str,
    resources: tuple[TurnInputResource, ...],
) -> tuple[AttachedResumeContext, ...]:
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
        document = store.read_version_document(
            user_id=user_id, resume_version_id=version.id
        )
        located_versions.append((resume, version))
        texts.append(
            extract_resume_text(document.document_format, document.raw_bytes)
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
