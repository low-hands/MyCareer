"""Opt-in live resume analysis smoke using synthetic PDF and Chinese text."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from career_agent.agent.local_resume_extraction import extract_resume_source
from career_agent.agent.openai_resume_analysis_worker import OpenAIResumeAnalysisWorker
from career_agent.agent.resume_analysis_contracts import ResumeAnalysisResult
from career_agent.services.resume_analysis import ResumeAnalysisService
from career_agent.storage.career_history import CareerHistoryStore
from career_agent.storage.resume_analysis import SQLiteResumeAnalysisDraftStore
from career_agent.storage.resumes import ResumeStore, StoredResumeDocument


_CHINESE_TEXT = (
    "测试候选人\n工作经历：示例科技，后端工程师，2022年3月至2024年5月。\n"
    "负责检索服务开发。\n使用 Python 开发接口。\n项目经历：内部搜索平台。\n"
)


def _validate_result(document: StoredResumeDocument, result: ResumeAnalysisResult) -> tuple[int, int]:
    source = extract_resume_source(document)
    records = result.records
    if not records:
        raise ValueError("no_records")
    result.validate_source_quotes(source.quotes_by_locator)
    evidence = sum(len(record.evidence) for record in records)
    if not evidence:
        raise ValueError("no_evidence")
    for record in records:
        for item in (record, *record.evidence):
            if item.source_quote != source.quotes_by_locator[item.source_locator]:
                raise ValueError("quote_not_full_paragraph")
    return len(records), evidence


def run_smoke(*, pdf_path: Path, work_root: Path, timeout_seconds: float) -> list[dict[str, object]]:
    root = work_root / f"resume093-smoke-{uuid4().hex}"
    root.mkdir(mode=0o700)
    user_id = "synthetic-smoke"
    resumes = ResumeStore(root / "resumes.sqlite3")
    role = resumes.create_target_role(user_id=user_id, title="Synthetic engineer", priority=1)
    drafts = SQLiteResumeAnalysisDraftStore(root / "analysis.sqlite3")
    worker = OpenAIResumeAnalysisWorker.from_env(timeout_seconds=timeout_seconds)
    service = ResumeAnalysisService(
        resumes, worker, drafts, CareerHistoryStore(root / "career.sqlite3")
    )
    cases = (
        ("english_pdf", "pdf", pdf_path.read_bytes()),
        ("chinese_text", "text", _CHINESE_TEXT.encode("utf-8")),
    )
    results: list[dict[str, object]] = []
    try:
        for name, document_format, content in cases:
            started = perf_counter()
            row: dict[str, object] = {"case": name, "passed": False}
            try:
                _, version = resumes.import_document(
                    user_id=user_id, content=content, document_format=document_format,
                    name=f"synthetic-{name}", target_role_id=role.id,
                )
                draft = service.analyze_version(user_id=user_id, resume_version_id=version.id)
                document = resumes.read_version_document(
                    user_id=user_id, resume_version_id=version.id
                )
                if document is None:
                    raise ValueError("document_readback")
                record_count, evidence_count = _validate_result(document, draft.result)
                reopened = SQLiteResumeAnalysisDraftStore(root / "analysis.sqlite3")
                saved = reopened.get(user_id=user_id, analysis_id=draft.id)
                if saved != draft or saved.status != "pending":
                    raise ValueError("draft_readback")
                if reopened.get(user_id="another-owner", analysis_id=draft.id) is not None:
                    raise ValueError("owner_isolation")
                row.update(passed=True, records=record_count, evidence_items=evidence_count,
                           draft_persisted=True, quotes_full_local_paragraphs=True)
            except Exception as error:
                # Provider bodies and resume text can occur in exception messages.
                row["error_type"] = type(error).__name__
                code = getattr(error, "code", None)
                if isinstance(code, str) and code.startswith("RESUME_ANALYSIS_"):
                    row["capability_error_code"] = code
            row["elapsed_seconds"] = round(perf_counter() - started, 4)
            results.append(row)
            print(json.dumps(row, ensure_ascii=True), flush=True)
    finally:
        worker._client.close()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True, help="Synthetic text-layer English PDF")
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 300:
        parser.error("--timeout must be between 1 and 300 seconds")
    if not args.pdf.is_file() or not args.work_root.is_dir() or not args.report.parent.is_dir():
        parser.error("PDF and artifact directories must already exist")
    if args.report.exists():
        parser.error("--report must be a new file")
    logging.disable(logging.CRITICAL)
    try:
        results = run_smoke(pdf_path=args.pdf, work_root=args.work_root,
                            timeout_seconds=args.timeout)
    except Exception as error:
        print(json.dumps({"configuration_available": False,
                          "error_type": type(error).__name__}), flush=True)
        return 2
    args.report.write_text(json.dumps({"synthetic": True, "results": results,
                                       "passed": all(row["passed"] for row in results)},
                                      ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return 0 if all(row["passed"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
