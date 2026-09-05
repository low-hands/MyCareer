from __future__ import annotations

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


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
            reader = PdfReader(BytesIO(content), strict=False)
            if reader.is_encrypted:
                raise ValueError("Encrypted resume PDFs are not supported.")
            if not reader.pages:
                raise ValueError("Resume PDF contains no pages.")
        except ValueError:
            raise
        except Exception as error:
            raise ValueError("Resume PDF is malformed or unreadable.") from error
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
