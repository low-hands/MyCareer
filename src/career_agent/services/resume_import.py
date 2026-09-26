from __future__ import annotations

from pathlib import Path

from career_agent.agent.local_resume_extraction import (
    cached_pdf_pages,
    cached_pdf_read,
    is_transient_pdf_failure,
)
from career_agent.agent.openai_compatible_client import AgentWorkerError
from career_agent.storage.resumes import StoredResumeDocument


MAX_RESUME_IMPORT_BYTES = 5 * 1_048_576


def resume_document_format(filename: str) -> str:
    suffix = Path(filename).suffix.casefold()
    document_format = {
        ".pdf": "pdf",
        ".txt": "text",
        ".md": "markdown",
        ".markdown": "markdown",
    }.get(suffix)
    if document_format is None:
        raise ValueError(
            "Resume import supports only .pdf, .txt, .md, and .markdown files."
        )
    return document_format


def validate_resume_document(filename: str, content: bytes) -> tuple[bytes, str]:
    document_format = resume_document_format(filename)
    if len(content) > MAX_RESUME_IMPORT_BYTES:
        raise ValueError("Resume file exceeds the 5 MiB import limit.")
    if not content:
        raise ValueError("Resume file must not be empty.")
    if document_format == "pdf":
        if not content.startswith(b"%PDF-"):
            raise ValueError("Resume PDF has an invalid header.")
        try:
            cached_pdf_read(content)
        except AgentWorkerError as error:
            if is_transient_pdf_failure(error):
                # The parser could not run; that says nothing against the file.
                return content, document_format
            code = error.code.removeprefix("RESUME_ANALYSIS_")
            if code == "PDF_ENCRYPTED":
                raise ValueError("Encrypted resume PDFs are not supported.") from None
            if code == "EMPTY_DOCUMENT":
                raise ValueError("Resume PDF contains no pages.") from None
            raise ValueError("Resume PDF is malformed or unreadable.") from None
        return content, document_format
    if b"\x00" in content:
        raise ValueError("Resume text must not contain NUL bytes.")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Resume text must be UTF-8 encoded.") from error
    if not text.strip():
        raise ValueError("Resume text must contain non-whitespace content.")
    return content, document_format


def extract_resume_text(document_format: str, content: bytes) -> str | None:
    """Plain text of a stored resume, or ``None`` when none can be read.

    A scanned PDF yields no text and an undecodable upload yields nothing at
    all; both come back as ``None`` so callers never mistake an empty
    extraction for an empty resume. Whitespace is collapsed per line so the
    layout noise of a PDF does not consume the caller's budget.
    """
    if document_format == "pdf":
        read = cached_pdf_pages(content)
        # Text PDFium could not map is not the resume's text; no text is better.
        if read is None or read.text_unreliable:
            return None
        raw = "\n".join(read.pages)
    else:
        try:
            raw = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            return None
    return _collapsed(raw)


def resume_document_text(document: StoredResumeDocument) -> str | None:
    """``extract_resume_text`` for a stored version: its stored text once read."""

    if document.text is not None:
        return _collapsed("\n".join(document.text.pages))
    return extract_resume_text(document.document_format, document.raw_bytes)


def _collapsed(raw: str) -> str | None:
    lines = [" ".join(line.split()) for line in raw.splitlines()]
    text = "\n".join(line for line in lines if line)
    return text or None
