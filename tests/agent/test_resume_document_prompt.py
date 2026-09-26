from io import BytesIO
import json

import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject,
    DictionaryObject,
    NameObject,
    NumberObject,
    TextStringObject,
)

from career_agent.agent import resume_document_prompt as prompts
from career_agent.agent.deepagent_resume_tailoring_worker import (
    DeepAgentResumeTailoringWorker, DeepAgentResumeFinalizationWorker,
)
from career_agent.agent.openai_resume_job_match_worker import OpenAIResumeJobMatchWorker
from career_agent.agent.openai_resume_tailoring_reviewer import OpenAIResumeTailoringReviewer
from career_agent.agent.openai_compatible_client import AgentWorkerError
from career_agent.agent.resume_job_match_contracts import ResumeJobMatchResult
from career_agent.storage.resumes import StoredResumeDocument
from tests.agent.test_resume_093_extraction import synthetic_pdf


@pytest.fixture(autouse=True)
def clear_cache():
    prompts._CACHE.clear()
    yield
    prompts._CACHE.clear()


def document(raw, version="same-version"):
    return StoredResumeDocument(resume_version_id=version, document_format="pdf", raw_bytes=raw)


def test_extracted_pdf_preserves_unicode_quotes_and_original_pages():
    result = prompts.pdf_text_prompt(document(synthetic_pdf(("张三 Python 开发", "Built Go services"))))
    payload = json.loads(result.split("\n", 1)[1])
    assert payload["page_count"] == 2
    assert payload["source_paragraphs"] == [
        {"page": 1, "paragraph": 1, "text": "张三 Python 开发"},
        {"page": 2, "paragraph": 1, "text": "Built Go services"},
    ]


@pytest.mark.parametrize("pages", [(None,), ("Text page", None)])
def test_scanned_or_mixed_pdf_never_silently_loses_pages(pages):
    assert prompts.pdf_text_prompt(document(synthetic_pdf(pages))) is None


def _text_and_image_page() -> bytes:
    """One page with a text layer and an image drawn on it."""
    reader = PdfReader(BytesIO(synthetic_pdf(("Small text layer", None))))
    first, image_page = reader.pages
    first["/Resources"][NameObject("/XObject")] = image_page["/Resources"]["/XObject"]
    writer = PdfWriter()
    page = writer.add_page(first)
    # Drawn, not only listed: an unused resource puts nothing on the page.
    page.merge_page(image_page)
    out = BytesIO()
    writer.write(out)
    return out.getvalue()


def test_image_plus_text_on_same_page_retains_original_pdf():
    assert prompts.pdf_text_prompt(document(_text_and_image_page())) is None


def test_cache_uses_content_not_version_id(monkeypatch):
    original = prompts.extract_resume_source
    calls = []
    def extract(doc, **kwargs):
        calls.append(doc.raw_bytes)
        return original(doc, **kwargs)
    monkeypatch.setattr(prompts, "extract_resume_source", extract)
    first = synthetic_pdf(("First version",))
    second = synthetic_pdf(("Changed version",))
    one = prompts.pdf_text_prompt(document(first))
    assert prompts.pdf_text_prompt(document(first, "different-id")) == one
    assert prompts.pdf_text_prompt(document(second)) != one
    assert len(calls) == 2


def test_parser_failure_is_cached_and_falls_back(monkeypatch):
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise AgentWorkerError("RESUME_ANALYSIS_PDF_EXTRACTION_LIMIT", "Timed out")
    monkeypatch.setattr(prompts, "extract_resume_source", fail)
    doc = document(b"%PDF-test")
    assert prompts.pdf_text_prompt(doc) is None
    assert prompts.pdf_text_prompt(doc) is None
    assert len(calls) == 1


@pytest.mark.parametrize("text_pdf", [True, False])
def test_all_pipeline_stages_share_text_or_original_pdf_route(text_pdf):
    doc = document(synthetic_pdf(("Built production APIs",) if text_pdf else (None,)))
    match = ResumeJobMatchResult(overall_fit="moderate", summary="Summary", requirements=())
    contents = [
        OpenAIResumeJobMatchWorker._document_content(doc, "JD", ()),
        DeepAgentResumeTailoringWorker._document_content(
            doc, jd_text="JD", match_result=match, confirmed_facts=(), tailoring_goal=None,
            user_feedback=None, review_feedback=(), previous_draft=None,
        ),
        OpenAIResumeTailoringReviewer._document_content(doc, "Review context"),
        DeepAgentResumeFinalizationWorker._document_content(doc, accepted_changes=(), confirmed_facts=()),
    ]
    for content in contents:
        assert content[0]["type"] in (("text", "input_text") if text_pdf else ("file", "input_file"))
        if text_pdf:
            assert '"page": 1' in content[0]["text"]
            assert "Built production APIs" in content[0]["text"]


def test_prompt_cache_is_bounded(monkeypatch):
    from career_agent.agent.local_resume_extraction import ExtractedResumeSource, ResumeSourceParagraph
    calls = []
    def extract(doc, **kwargs):
        calls.append(doc.raw_bytes)
        return ExtractedResumeSource((ResumeSourceParagraph(1, 1, "Synthetic"),), 1)
    monkeypatch.setattr(prompts, "extract_resume_source", extract)
    for index in range(prompts._CACHE_SIZE + 1):
        prompts.pdf_text_prompt(document(str(index).encode()))
    assert len(prompts._CACHE) == prompts._CACHE_SIZE
    prompts.pdf_text_prompt(document(b"0"))
    assert len(calls) == prompts._CACHE_SIZE + 2


def test_route_telemetry_contains_no_resume_content():
    from career_agent.harness.observability import ACTIVE_TRACE_CONTEXT, InMemoryTraceRecorder
    recorder = InMemoryTraceRecorder()
    token = ACTIVE_TRACE_CONTEXT.set((recorder, "test"))
    try:
        doc = document(synthetic_pdf(("Private candidate text",)))
        prompts.pdf_text_prompt(doc)
        prompts.pdf_text_prompt(doc)
    finally:
        ACTIVE_TRACE_CONTEXT.reset(token)
    events = recorder.snapshot("test").events
    assert [event.details for event in events] == [
        {"input_mode": "extracted_text", "cache_hit": False},
        {"input_mode": "extracted_text", "cache_hit": True},
    ]
    assert "Private candidate" not in recorder.snapshot("test").model_dump_json()
    assert all(event.duration_ms is not None for event in events)


def _rewritten(raw: bytes, edit) -> bytes:
    reader = PdfReader(BytesIO(raw))
    writer = PdfWriter()
    for page in reader.pages:
        edit(page, writer)
        writer.add_page(page)
    out = BytesIO()
    writer.write(out)
    return out.getvalue()


def _without_to_unicode(page, writer):
    for font in page["/Resources"]["/Font"].values():
        font.get_object().pop("/ToUnicode", None)


def _with_mailto_link(page, writer):
    link = DictionaryObject({
        NameObject("/Type"): NameObject("/Annot"),
        NameObject("/Subtype"): NameObject("/Link"),
        NameObject("/Rect"): ArrayObject([NumberObject(0)] * 4),
        NameObject("/A"): DictionaryObject({
            NameObject("/S"): NameObject("/URI"),
            NameObject("/URI"): TextStringObject("mailto:someone@example.invalid"),
        }),
    })
    page[NameObject("/Annots")] = ArrayObject([writer._add_object(link)])


def test_a_link_annotation_is_not_visual_content():
    """An email or portfolio link is on most resumes and carries no picture."""
    raw = _rewritten(synthetic_pdf(("张三 Python 开发",)), _with_mailto_link)
    result = prompts.pdf_text_prompt(document(raw))
    assert result is not None and "张三 Python 开发" in result


def test_a_composite_font_without_to_unicode_always_sends_the_pdf():
    """Its glyph ids decode to unrelated characters; no caller may take them."""
    raw = _rewritten(synthetic_pdf(("张三 Python 开发",)), _without_to_unicode)
    assert prompts.pdf_text_prompt(document(raw)) is None
    assert prompts.pdf_text_prompt(document(raw), ignore_visual_content=True) is None


def test_visual_content_can_be_ignored_only_by_the_caller_that_asks():
    doc = document(_text_and_image_page())

    # Either order: the flag is part of the cache key, so one caller's route
    # never leaks into the other's.
    assert prompts.pdf_text_prompt(doc, ignore_visual_content=True) is not None
    assert prompts.pdf_text_prompt(doc) is None
    prompts._CACHE.clear()
    assert prompts.pdf_text_prompt(doc) is None
    assert prompts.pdf_text_prompt(doc, ignore_visual_content=True) is not None
