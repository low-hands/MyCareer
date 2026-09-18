"""Bounded local extraction for formal resume analysis; never uploads a PDF.

PDF parsing runs in a short-lived, resource-limited subprocess. Its captured IPC
contains source text, but parser diagnostics are discarded, never logged. OCR is
not configured here: a page without a text layer fails closed with an explicit
OCR-required result rather than silently analysing only the remaining pages.
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from career_agent.agent.openai_compatible_client import AgentWorkerError

if TYPE_CHECKING:
    from career_agent.storage.resumes import StoredResumeDocument


@dataclass(frozen=True)
class ResumeExtractionLimits:
    max_bytes: int = 5 * 1_048_576
    max_pages: int = 20
    max_characters: int = 60_000
    # One token per UTF-8 byte is deliberately conservative for an endpoint
    # with an unknown tokenizer. The complete request is checked separately.
    max_text_tokens: int = 24_000
    pdf_timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        if min(
            self.max_bytes,
            self.max_pages,
            self.max_characters,
            self.max_text_tokens,
            self.pdf_timeout_seconds,
        ) <= 0 or not math.isfinite(self.pdf_timeout_seconds):
            raise ValueError("Resume extraction limits must be positive.")


@dataclass(frozen=True)
class ResumeSourceParagraph:
    page: int
    paragraph: int
    text: str

    @property
    def locator(self) -> str:
        return f"page {self.page}, paragraph {self.paragraph}"


@dataclass(frozen=True)
class ExtractedResumeSource:
    paragraphs: tuple[ResumeSourceParagraph, ...]
    page_count: int

    @property
    def quotes_by_locator(self) -> dict[str, str]:
        return {item.locator: item.text for item in self.paragraphs}

    def as_prompt_data(self) -> str:
        # JSON framing prevents text that resembles a locator/delimiter from
        # being confused with a locally issued locator. It grants no authority.
        return json.dumps(
            {
                "source_paragraphs": [
                    {"source_locator": item.locator, "text": item.text}
                    for item in self.paragraphs
                ]
            },
            ensure_ascii=False,
        )


_MESSAGES = {
    "EMPTY_DOCUMENT": "Resume document contains no text. Select a non-empty document.",
    "INVALID_TEXT_ENCODING": "Text resume must be UTF-8 encoded.",
    "INVALID_TEXT_CONTENT": "Resume text contains invalid control characters. Export readable UTF-8 text.",
    "DOCUMENT_TOO_LARGE": "Resume exceeds the local analysis file-size limit. Select a smaller document.",
    "PDF_ENCRYPTED": "Encrypted PDFs cannot be analysed. Export an unencrypted copy locally.",
    "PDF_DAMAGED": "Resume PDF is damaged or unreadable. Export a valid PDF or UTF-8 text locally.",
    "PDF_TOO_MANY_PAGES": "Resume exceeds the analysis page limit. Select a shorter document.",
    "CHARACTER_BUDGET_EXCEEDED": "Resume exceeds the analysis character budget. Select a shorter document.",
    "TOKEN_BUDGET_EXCEEDED": "Resume exceeds the analysis token budget. Select a shorter document or check the input budget configuration.",
    "OCR_REQUIRED": "At least one PDF page has no readable text layer. Local OCR is required but not configured; run OCR locally or select a text-based PDF/UTF-8 document.",
    "PDF_EXTRACTION_LIMIT": "Local PDF extraction exceeded its time or memory limit. Export a simpler text-based PDF or UTF-8 document.",
    "PDF_SANDBOX_UNAVAILABLE": "The local PDF parser sandbox could not apply its resource limits on this system. Use a UTF-8 text resume or report this platform issue.",
    "PDF_EXTRACTION_FAILED": "The local PDF parser exited unexpectedly. Export a simpler text-based PDF or UTF-8 document.",
}

# Child exit status reserved for "resource limits could not be applied", so the
# parent never reports a sandbox setup failure as a parser limit breach.
_SANDBOX_UNAVAILABLE_EXIT = 3


def _failure(code: str) -> AgentWorkerError:
    return AgentWorkerError(f"RESUME_ANALYSIS_{code}", _MESSAGES[code])


def _check_text(text: str, limits: ResumeExtractionLimits) -> None:
    if len(text) > limits.max_characters:
        raise _failure("CHARACTER_BUDGET_EXCEEDED")
    if len(text.encode("utf-8")) > limits.max_text_tokens:
        raise _failure("TOKEN_BUDGET_EXCEEDED")
    if any(ord(char) < 32 and char not in "\n\r\t" for char in text):
        raise _failure("INVALID_TEXT_CONTENT")


def _paragraphs(text: str, page: int) -> tuple[ResumeSourceParagraph, ...]:
    # A non-empty extracted line is the deterministic paragraph unit. Never
    # collapse internal spaces, normalize Unicode, or paraphrase source quotes.
    lines = (line.strip() for line in text.splitlines())
    return tuple(
        ResumeSourceParagraph(page=page, paragraph=index, text=line)
        for index, line in enumerate((line for line in lines if line), start=1)
    )


def extract_resume_source(
    document: StoredResumeDocument,
    *,
    limits: ResumeExtractionLimits = ResumeExtractionLimits(),
) -> ExtractedResumeSource:
    if not document.raw_bytes:
        raise _failure("EMPTY_DOCUMENT")
    if len(document.raw_bytes) > limits.max_bytes:
        raise _failure("DOCUMENT_TOO_LARGE")
    if document.document_format == "pdf":
        return _extract_pdf_isolated(document.raw_bytes, limits)
    try:
        text = document.raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise _failure("INVALID_TEXT_ENCODING") from None
    _check_text(text, limits)
    paragraphs = _paragraphs(text, page=1)
    if not paragraphs:
        raise _failure("EMPTY_DOCUMENT")
    return ExtractedResumeSource(paragraphs=paragraphs, page_count=1)


def _extract_pdf(raw: bytes, limits: ResumeExtractionLimits) -> ExtractedResumeSource:
    from pypdf import PdfReader

    if not raw.startswith(b"%PDF-"):
        raise _failure("PDF_DAMAGED")
    try:
        reader = PdfReader(BytesIO(raw), strict=True)
        if reader.is_encrypted:
            raise _failure("PDF_ENCRYPTED")
        page_count = len(reader.pages)
        if page_count > limits.max_pages:
            raise _failure("PDF_TOO_MANY_PAGES")
        if page_count == 0:
            raise _failure("EMPTY_DOCUMENT")
        paragraphs: list[ResumeSourceParagraph] = []
        characters = 0
        text_tokens = 0
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            _check_text(text, limits)
            characters += len(text)
            text_tokens += len(text.encode("utf-8"))
            if characters > limits.max_characters:
                raise _failure("CHARACTER_BUDGET_EXCEEDED")
            if text_tokens > limits.max_text_tokens:
                raise _failure("TOKEN_BUDGET_EXCEEDED")
            page_paragraphs = _paragraphs(text, page=page_number)
            if not page_paragraphs:
                raise _failure("OCR_REQUIRED")
            paragraphs.extend(page_paragraphs)
        return ExtractedResumeSource(tuple(paragraphs), page_count)
    except AgentWorkerError:
        raise
    except MemoryError:
        raise _failure("PDF_EXTRACTION_LIMIT") from None
    except Exception:
        # pypdf failures can embed document bytes in the exception message.
        raise _failure("PDF_DAMAGED") from None


def _extract_pdf_isolated(
    raw: bytes, limits: ResumeExtractionLimits
) -> ExtractedResumeSource:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            (str(Path(__file__).resolve().parents[2]), environment.get("PYTHONPATH")),
        )
    )
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "career_agent.agent.local_resume_extraction",
                str(limits.max_pages),
                str(limits.max_characters),
                str(limits.max_text_tokens),
                str(limits.max_bytes),
            ],
            input=raw,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=limits.pdf_timeout_seconds,
            check=False,
            env=environment,
        )
    except (subprocess.TimeoutExpired, OSError):
        raise _failure("PDF_EXTRACTION_LIMIT") from None
    if completed.returncode == _SANDBOX_UNAVAILABLE_EXIT:
        raise _failure("PDF_SANDBOX_UNAVAILABLE")
    if completed.returncode < 0:
        # Killed by a signal: SIGXCPU/SIGKILL from RLIMIT_CPU is the expected
        # path; any other signal is still an abnormal bounded-parser stop.
        raise _failure("PDF_EXTRACTION_LIMIT")
    if completed.returncode != 0:
        raise _failure("PDF_EXTRACTION_FAILED")
    try:
        payload = json.loads(completed.stdout)
        if not isinstance(payload, dict):
            raise ValueError("Invalid parser response")
        code = payload.get("error")
        if code in _MESSAGES:
            raise _failure(code)
        paragraphs = tuple(
            ResumeSourceParagraph(**item) for item in payload["paragraphs"]
        )
        return ExtractedResumeSource(paragraphs, payload["page_count"])
    except (ValueError, TypeError, KeyError):
        raise _failure("PDF_DAMAGED") from None


_PDF_CHILD_MEMORY_BYTES = 512 * 1_048_576
_PDF_CHILD_CPU_SECONDS = 8


def _apply_child_limits() -> None:
    # Limits apply only to this parser subprocess, not the API process or any
    # existing shell/server. Unix is the supported deployment boundary.
    import resource

    # macOS rejects any finite RLIMIT_AS ("current limit exceeds maximum
    # limit") and does not enforce RLIMIT_DATA for mmap-backed allocations, so
    # there the parser is bounded by CPU time plus the parent's wall-clock
    # timeout only. Elsewhere the address-space cap is mandatory.
    if sys.platform != "darwin":
        resource.setrlimit(
            resource.RLIMIT_AS, (_PDF_CHILD_MEMORY_BYTES, _PDF_CHILD_MEMORY_BYTES)
        )
    resource.setrlimit(
        resource.RLIMIT_CPU, (_PDF_CHILD_CPU_SECONDS, _PDF_CHILD_CPU_SECONDS)
    )


def _pdf_child() -> None:
    try:
        _apply_child_limits()
    except (ValueError, OSError):
        # Fail closed: never parse an untrusted PDF without the limits.
        sys.exit(_SANDBOX_UNAVAILABLE_EXIT)
    logging.disable(logging.CRITICAL)
    warnings.simplefilter("ignore")
    limits = ResumeExtractionLimits(
        max_pages=int(sys.argv[1]),
        max_characters=int(sys.argv[2]),
        max_text_tokens=int(sys.argv[3]),
        max_bytes=int(sys.argv[4]),
    )
    raw = sys.stdin.buffer.read(limits.max_bytes + 1)
    try:
        if len(raw) > limits.max_bytes:
            raise _failure("DOCUMENT_TOO_LARGE")
        source = _extract_pdf(raw, limits)
        payload = {
            "page_count": source.page_count,
            "paragraphs": [
                {"page": item.page, "paragraph": item.paragraph, "text": item.text}
                for item in source.paragraphs
            ],
        }
    except AgentWorkerError as error:
        payload = {"error": error.code.removeprefix("RESUME_ANALYSIS_")}
    except MemoryError:
        # Raised under RLIMIT_AS outside _extract_pdf (e.g. importing pypdf).
        payload = {"error": "PDF_EXTRACTION_LIMIT"}
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    _pdf_child()
