"""A resume version's text, read once and kept.

Every reader of a resume (analysis, review, matching, tailoring, chat
attachments) works from its text, so the text is read once per immutable
version and stored beside the file, as resume parsers and document-chat
products do, rather than parsed again, or the whole PDF resent, on each use.

The local PDF parser goes first. When it cannot give the page's own text (a
scan, a font without a Unicode map), the model reads the PDF and transcribes
it. A failure to read is not stored, so a later attempt can still succeed.
"""

from __future__ import annotations

from threading import Lock
from typing import Protocol

from career_agent.agent.local_resume_extraction import (
    cached_pdf_read,
    is_transient_pdf_failure,
)
from career_agent.agent.openai_compatible_client import AgentWorkerError
from career_agent.storage.resumes import (
    ResumeStore,
    StoredResumeDocument,
    StoredResumeText,
)


class ResumeTranscriber(Protocol):
    def transcribe(self, *, pdf: bytes, page_count: int | None) -> tuple[str, ...]: ...


_VERSION_LOCKS: dict[str, Lock] = {}
_VERSION_LOCKS_GUARD = Lock()


def _version_lock(resume_version_id: str) -> Lock:
    # Two turns attaching the same new resume must not transcribe it twice.
    with _VERSION_LOCKS_GUARD:
        return _VERSION_LOCKS.setdefault(resume_version_id, Lock())


class ResumeTextService:
    def __init__(
        self,
        store: ResumeStore,
        transcriber: ResumeTranscriber | None = None,
    ) -> None:
        self._store = store
        self._transcriber = transcriber

    def ensure(
        self, *, user_id: str, resume_version_id: str
    ) -> StoredResumeDocument | None:
        """The version's document with its text, reading and storing it if new.

        ``None`` when the version does not exist for this user. The document's
        ``text`` stays ``None`` when neither the parser nor the model could
        read it this time.
        """

        document = self._store.read_version_document(
            user_id=user_id, resume_version_id=resume_version_id
        )
        if document is None or document.text is not None:
            return document
        with _version_lock(resume_version_id):
            stored = self._store.get_version_text(
                user_id=user_id, resume_version_id=resume_version_id
            )
            if stored is not None:
                return document.model_copy(update={"text": stored})
            text = self.read(document)
            if text is None:
                return document
            self._store.save_version_text(
                user_id=user_id, resume_version_id=resume_version_id, text=text
            )
        return document.model_copy(update={"text": text})

    def read(self, document: StoredResumeDocument) -> StoredResumeText | None:
        if document.document_format != "pdf":
            try:
                decoded = document.raw_bytes.decode("utf-8-sig")
            except UnicodeDecodeError:
                return None
            return StoredResumeText(pages=(decoded,), method="plain") if decoded.strip() else None
        try:
            read = cached_pdf_read(document.raw_bytes)
        except AgentWorkerError as error:
            if is_transient_pdf_failure(error):
                # The parser did not run; the file may be fine. Try again later.
                return None
            read = None
        if read is not None and not read.text_unreliable and all(page.strip() for page in read.pages):
            return StoredResumeText(
                pages=read.pages, method="local", has_visual_content=read.has_visual_content
            )
        if self._transcriber is None:
            return None
        try:
            pages = self._transcriber.transcribe(
                pdf=document.raw_bytes,
                page_count=len(read.pages) if read is not None else None,
            )
        except AgentWorkerError:
            return None
        return StoredResumeText(
            pages=pages,
            method="model",
            has_visual_content=read.has_visual_content if read is not None else True,
        )
