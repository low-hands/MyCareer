"""A resume version's text is read once, and the model reads what the parser cannot."""

from __future__ import annotations

from io import BytesIO
import sqlite3

from pypdf import PdfReader, PdfWriter

from career_agent.agent.local_resume_extraction import extract_resume_source
from career_agent.agent.openai_compatible_client import AgentWorkerError
from career_agent.services.resume_text import ResumeTextService
from career_agent.storage.resumes import ResumeStore
from tests.agent.test_resume_093_extraction import synthetic_pdf


def _without_to_unicode(raw: bytes) -> bytes:
    """A composite font with no Unicode map: its glyphs cannot be decoded."""
    reader = PdfReader(BytesIO(raw))
    writer = PdfWriter()
    for page in reader.pages:
        for font in page["/Resources"]["/Font"].values():
            font.get_object().pop("/ToUnicode", None)
        writer.add_page(page)
    out = BytesIO()
    writer.write(out)
    return out.getvalue()


class Transcriber:
    def __init__(self, pages: tuple[str, ...] = ("张三\n北京大学 计算机科学",), *, fail: bool = False) -> None:
        self.pages = pages
        self.fail = fail
        self.calls: list[int | None] = []

    def transcribe(self, *, pdf: bytes, page_count: int | None) -> tuple[str, ...]:
        self.calls.append(page_count)
        if self.fail:
            raise AgentWorkerError("RESUME_TRANSCRIPTION_TRANSPORT_ERROR", "offline", retryable=True)
        return self.pages


def _version(tmp_path, content: bytes, document_format: str = "pdf"):
    store = ResumeStore(tmp_path / "resumes.sqlite3")
    role = store.create_target_role(user_id="u1", title="AI", priority=1)
    _, version = store.import_document(
        user_id="u1", target_role_id=role.id, name="A", content=content, document_format=document_format
    )
    return store, version.id


def test_a_readable_pdf_is_read_locally_once_and_kept(tmp_path) -> None:
    store, version_id = _version(tmp_path, synthetic_pdf(("张三 Python 开发",)))
    transcriber = Transcriber()
    service = ResumeTextService(store, transcriber)

    document = service.ensure(user_id="u1", resume_version_id=version_id)

    assert document is not None and document.text is not None
    assert document.text.method == "local"
    assert document.text.pages == ("张三 Python 开发",)
    assert transcriber.calls == []
    # Every later reader gets it from the store, not from parsing again.
    stored = store.read_version_document(user_id="u1", resume_version_id=version_id)
    assert stored is not None and stored.text == document.text


def test_text_the_parser_cannot_decode_is_transcribed_by_the_model(tmp_path) -> None:
    store, version_id = _version(tmp_path, _without_to_unicode(synthetic_pdf(("张三 Python 开发",))))
    transcriber = Transcriber()
    service = ResumeTextService(store, transcriber)

    document = service.ensure(user_id="u1", resume_version_id=version_id)

    assert transcriber.calls == [1]
    assert document is not None and document.text is not None
    assert document.text.method == "model"
    # Analysis quotes the transcription, not the undecodable text layer.
    source = extract_resume_source(document)
    assert source.text_unreliable is False
    assert source.quotes_by_locator == {
        "page 1, paragraph 1": "张三",
        "page 1, paragraph 2": "北京大学 计算机科学",
    }
    service.ensure(user_id="u1", resume_version_id=version_id)
    assert transcriber.calls == [1]


def test_a_failed_transcription_is_not_kept_and_is_tried_again(tmp_path) -> None:
    store, version_id = _version(tmp_path, _without_to_unicode(synthetic_pdf(("张三",))))
    failing = Transcriber(fail=True)

    document = ResumeTextService(store, failing).ensure(user_id="u1", resume_version_id=version_id)

    assert document is not None and document.text is None
    assert store.get_version_text(user_id="u1", resume_version_id=version_id) is None
    later = ResumeTextService(store, Transcriber()).ensure(user_id="u1", resume_version_id=version_id)
    assert later is not None and later.text is not None and later.text.method == "model"


def test_a_scanned_page_is_transcribed_with_its_page_count(tmp_path) -> None:
    store, version_id = _version(tmp_path, synthetic_pdf(("第一页文字", None)))
    transcriber = Transcriber(pages=("第一页文字", "第二页文字"))

    document = ResumeTextService(store, transcriber).ensure(user_id="u1", resume_version_id=version_id)

    assert transcriber.calls == [2]
    assert document is not None and document.text is not None
    assert document.text.pages == ("第一页文字", "第二页文字")


def test_without_a_model_an_unreadable_pdf_stays_without_text(tmp_path) -> None:
    store, version_id = _version(tmp_path, synthetic_pdf((None,)))

    document = ResumeTextService(store).ensure(user_id="u1", resume_version_id=version_id)

    assert document is not None and document.text is None


def test_a_text_resume_is_kept_as_one_page(tmp_path) -> None:
    store, version_id = _version(tmp_path, "张三\n后端工程师".encode(), "text")

    document = ResumeTextService(store).ensure(user_id="u1", resume_version_id=version_id)

    assert document is not None and document.text is not None
    assert (document.text.method, document.text.pages) == ("plain", ("张三\n后端工程师",))


def test_another_users_version_is_neither_read_nor_written(tmp_path) -> None:
    store, version_id = _version(tmp_path, "张三".encode(), "text")

    assert ResumeTextService(store).ensure(user_id="u2", resume_version_id=version_id) is None
    assert store.get_version_text(user_id="u1", resume_version_id=version_id) is None


def test_deleting_a_resume_deletes_its_text(tmp_path) -> None:
    store, version_id = _version(tmp_path, "张三".encode(), "text")
    ResumeTextService(store).ensure(user_id="u1", resume_version_id=version_id)
    resume_id = store.get_version(user_id="u1", resume_version_id=version_id)[0].id

    assert store.delete_resume(user_id="u1", resume_id=resume_id)

    with sqlite3.connect(tmp_path / "resumes.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM resume_version_texts").fetchone() == (0,)


def test_a_version_9_file_gains_the_text_table_on_open(tmp_path) -> None:
    path = tmp_path / "resumes.sqlite3"
    ResumeStore(path)
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE resume_version_texts")
        connection.execute("UPDATE schema_versions SET version = 9 WHERE component = 'resumes'")
        connection.commit()

    store = ResumeStore(path)
    role = store.create_target_role(user_id="u1", title="AI", priority=1)
    _, version = store.import_document(
        user_id="u1", target_role_id=role.id, name="A", content=b"x", document_format="text"
    )

    assert ResumeTextService(store).ensure(user_id="u1", resume_version_id=version.id).text is not None
