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
import re
import subprocess
import sys
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import Lock
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
    # Conservative routing hint only; never claims OCR or visual verification.
    has_visual_content: bool = False
    # The extracted characters cannot be trusted to be the ones on the page:
    # a composite font without a ToUnicode map yields glyph ids that decode to
    # unrelated code points (a LaTeX Fandol resume reads as Gurmukhi/Tibetan).
    # Unlike visual content this is never safe to ignore.
    text_unreliable: bool = False

    @property
    def quotes_by_locator(self) -> dict[str, str]:
        return {item.locator: item.text for item in self.paragraphs}

    @property
    def locator_by_number(self) -> dict[int, str]:
        """One-based request-local identifiers for the model-facing payload."""
        return {
            number: item.locator
            for number, item in enumerate(self.paragraphs, start=1)
        }

    @property
    def paragraph_by_number(self) -> dict[int, str]:
        return {
            number: item.text
            for number, item in enumerate(self.paragraphs, start=1)
        }

    def as_numbered_prompt_data(self) -> str:
        # JSON framing prevents text that resembles a locator/delimiter from
        # being confused with a locally issued locator. It grants no authority.
        return json.dumps(
            {
                "source_paragraphs": [
                    {
                        "paragraph_number": number,
                        "source_text": item.text,
                    }
                    for number, item in enumerate(self.paragraphs, start=1)
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
    if document.text is not None:
        # Read once when the version was imported; locally or, where the
        # parser could not, transcribed by the model. Either way it is the
        # page's own text.
        return _source_from_pages(
            document.text.pages,
            limits,
            has_visual_content=document.text.has_visual_content,
            text_unreliable=False,
        )
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


@dataclass(frozen=True)
class ExtractedPdfPages:
    """What the sandboxed parser read: each page's text, in page order."""

    pages: tuple[str, ...]
    has_visual_content: bool
    text_unreliable: bool


# Characters a parser emits for a glyph it could not map to Unicode.
_UNMAPPED_CHARACTERS = re.compile("[\x00\ufffd\ufffe]")


def _has_visual_content(page, pdfium_c) -> bool:
    # Images anywhere on the page, including inside form XObjects. A link
    # annotation (email, portfolio) carries no picture; any other kind may.
    for _ in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE], max_depth=8):
        return True
    for index in range(pdfium_c.FPDFPage_GetAnnotCount(page)):
        annotation = pdfium_c.FPDFPage_GetAnnot(page, index)
        try:
            subtype = pdfium_c.FPDFAnnot_GetSubtype(annotation)
        finally:
            pdfium_c.FPDFPage_CloseAnnot(annotation)
        if subtype != pdfium_c.FPDF_ANNOT_LINK:
            return True
    return False


def _has_unmapped_characters(textpage, pdfium_c) -> bool:
    # PDFium recovers Unicode from the embedded font when a composite font has
    # no ToUnicode map (LaTeX CJK fonts); this flags the glyphs it could not.
    return any(
        pdfium_c.FPDFText_HasUnicodeMapError(textpage, index) == 1
        for index in range(pdfium_c.FPDFText_CountChars(textpage))
    )


def _read_pdf(raw: bytes, *, max_pages: int, max_characters: int) -> ExtractedPdfPages:
    """Parse ``raw`` with PDFium. Called only inside the limited child process:
    PDFium is native code reading an untrusted file, and it is not thread-safe."""

    import pypdfium2 as pdfium
    import pypdfium2.raw as pdfium_c

    if not raw.startswith(b"%PDF-"):
        raise _failure("PDF_DAMAGED")
    try:
        try:
            pdf = pdfium.PdfDocument(raw)
        except pdfium.PdfiumError as error:
            encrypted = "password" in str(error).casefold()
            raise _failure("PDF_ENCRYPTED" if encrypted else "PDF_DAMAGED") from None
        try:
            page_count = len(pdf)
            if page_count > max_pages:
                raise _failure("PDF_TOO_MANY_PAGES")
            if page_count == 0:
                raise _failure("EMPTY_DOCUMENT")
            pages: list[str] = []
            characters = 0
            has_visual_content = False
            text_unreliable = False
            for index in range(page_count):
                page = pdf[index]
                try:
                    has_visual_content |= _has_visual_content(page, pdfium_c)
                    textpage = page.get_textpage()
                    try:
                        text = textpage.get_text_range()
                        text_unreliable |= (
                            _has_unmapped_characters(textpage, pdfium_c)
                            or _UNMAPPED_CHARACTERS.search(text) is not None
                        )
                    finally:
                        textpage.close()
                finally:
                    page.close()
                characters += len(text)
                if characters > max_characters:
                    raise _failure("CHARACTER_BUDGET_EXCEEDED")
                pages.append(text)
            return ExtractedPdfPages(tuple(pages), has_visual_content, text_unreliable)
        finally:
            pdf.close()
    except AgentWorkerError:
        raise
    except MemoryError:
        raise _failure("PDF_EXTRACTION_LIMIT") from None
    except Exception:
        # Parser failures can embed document bytes in the exception message.
        raise _failure("PDF_DAMAGED") from None


def read_pdf_pages(
    raw: bytes, limits: ResumeExtractionLimits = ResumeExtractionLimits()
) -> ExtractedPdfPages:
    """Each page's text of ``raw``, read by PDFium in a resource-limited child.

    Every PDF text read goes through here, so no untrusted PDF is parsed by
    native code inside the API process.
    """

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
        pages = payload["pages"]
        if not isinstance(pages, list) or not all(isinstance(page, str) for page in pages):
            raise ValueError("Invalid parser pages")
        return ExtractedPdfPages(
            tuple(pages),
            bool(payload.get("has_visual_content", True)),
            bool(payload.get("text_unreliable", True)),
        )
    except (ValueError, TypeError, KeyError):
        raise _failure("PDF_DAMAGED") from None


def _extract_pdf_isolated(
    raw: bytes, limits: ResumeExtractionLimits
) -> ExtractedResumeSource:
    read = read_pdf_pages(raw, limits)
    return _source_from_pages(
        read.pages,
        limits,
        has_visual_content=read.has_visual_content,
        text_unreliable=read.text_unreliable,
    )


def _source_from_pages(
    pages: tuple[str, ...],
    limits: ResumeExtractionLimits,
    *,
    has_visual_content: bool,
    text_unreliable: bool,
) -> ExtractedResumeSource:
    paragraphs: list[ResumeSourceParagraph] = []
    characters = 0
    text_tokens = 0
    for page_number, text in enumerate(pages, start=1):
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
    return ExtractedResumeSource(
        tuple(paragraphs), len(pages), has_visual_content, text_unreliable
    )


_PAGE_CACHE: OrderedDict[str, ExtractedPdfPages | str] = OrderedDict()
_PAGE_CACHE_LOCK = Lock()
_PAGE_CACHE_SIZE = 64
# Reading a stored resume for validation, display or excerpts: no analysis
# budgets, only the parser's own safety bounds.
_READ_LIMITS = ResumeExtractionLimits(
    max_pages=100, max_characters=400_000, max_text_tokens=1_600_000
)
# Failures of the parser run, not of the file: never remembered.
_TRANSIENT_CODES = frozenset(
    {"PDF_EXTRACTION_LIMIT", "PDF_SANDBOX_UNAVAILABLE", "PDF_EXTRACTION_FAILED"}
)


def cached_pdf_read(raw: bytes) -> ExtractedPdfPages:
    """``read_pdf_pages`` for callers that only read, cached by content.

    Pages read the same file on every visit, so the child process runs once
    per distinct document. Raises the parser's ``AgentWorkerError``; one that
    comes from the file itself (damaged, encrypted) is cached like a result.
    """

    digest = sha256(raw).hexdigest()
    with _PAGE_CACHE_LOCK:
        cached = _PAGE_CACHE.get(digest)
        if cached is not None:
            _PAGE_CACHE.move_to_end(digest)
    if isinstance(cached, str):
        raise _failure(cached)
    if cached is not None:
        return cached
    try:
        result: ExtractedPdfPages | str = read_pdf_pages(raw, _READ_LIMITS)
    except AgentWorkerError as error:
        code = error.code.removeprefix("RESUME_ANALYSIS_")
        if code in _TRANSIENT_CODES or code not in _MESSAGES:
            raise
        result = code
    with _PAGE_CACHE_LOCK:
        _PAGE_CACHE[digest] = result
        _PAGE_CACHE.move_to_end(digest)
        while len(_PAGE_CACHE) > _PAGE_CACHE_SIZE:
            _PAGE_CACHE.popitem(last=False)
    if isinstance(result, str):
        raise _failure(result)
    return result


def cached_pdf_pages(raw: bytes) -> ExtractedPdfPages | None:
    """``cached_pdf_read``, with ``None`` when no text could be read at all."""

    try:
        return cached_pdf_read(raw)
    except AgentWorkerError:
        return None


def is_transient_pdf_failure(error: AgentWorkerError) -> bool:
    """The parser could not run; that says nothing about the file."""

    return error.code.removeprefix("RESUME_ANALYSIS_") in _TRANSIENT_CODES


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
    max_pages, max_characters, max_bytes = (int(value) for value in sys.argv[1:4])
    raw = sys.stdin.buffer.read(max_bytes + 1)
    try:
        if len(raw) > max_bytes:
            raise _failure("DOCUMENT_TOO_LARGE")
        read = _read_pdf(raw, max_pages=max_pages, max_characters=max_characters)
        payload = {
            "pages": list(read.pages),
            "has_visual_content": read.has_visual_content,
            "text_unreliable": read.text_unreliable,
        }
    except AgentWorkerError as error:
        payload = {"error": error.code.removeprefix("RESUME_ANALYSIS_")}
    except MemoryError:
        # Raised under RLIMIT_AS outside _read_pdf (e.g. importing pypdfium2).
        payload = {"error": "PDF_EXTRACTION_LIMIT"}
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    _pdf_child()
