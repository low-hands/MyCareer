from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

from career_agent.agent.job_research_contracts import (
    JobResearchWorker,
    JobResearchWorkerRequest,
)
from career_agent.domain.job_research import (
    company_key,
    JobResearchDraft,
    JobResearchFinding,
    JobResearchReport,
    JobResearchRun,
    JobResearchScope,
    JobResearchSource,
)
from career_agent.security.redaction import redact_text
from career_agent.storage.job_research import SQLiteJobResearchStore
from career_agent.storage.jobs import JobPostingRepository


class JobResearchInputNotFoundError(ValueError):
    pass


class JobResearchValidationError(ValueError):
    pass


class JobResearchRunNotRetryableError(ValueError):
    pass


class JobResearchExecutionError(RuntimeError):
    def __init__(
        self,
        *,
        run_id: str,
        code: str,
        retryable: bool,
        detail: str,
    ) -> None:
        super().__init__(detail)
        self.run_id = run_id
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True)
class JobResearchResult:
    run: JobResearchRun
    report: JobResearchReport
    sources: tuple[JobResearchSource, ...]
    cached: bool


class JobResearchService:
    def __init__(
        self,
        *,
        jobs: JobPostingRepository,
        store: SQLiteJobResearchStore,
        worker: JobResearchWorker,
        worker_version: str = "job-research-v1",
        freshness: timedelta = timedelta(days=7),
    ) -> None:
        if freshness <= timedelta(0):
            raise ValueError("job research freshness must be positive")
        self._jobs = jobs
        self._store = store
        self._worker = worker
        self._worker_version = worker_version
        self._freshness = freshness

    def research(
        self,
        *,
        user_id: str,
        job_posting_id: str,
        focus: str | None = None,
        user_provided_context: str | None = None,
        max_sources: int = 8,
    ) -> JobResearchResult:
        job = self._jobs.get_job(user_id=user_id, job_posting_id=job_posting_id)
        if job is None:
            raise JobResearchInputNotFoundError("job_posting")
        scope = JobResearchScope(
            focus=focus,
            user_provided_context=(
                redact_text(user_provided_context)
                if user_provided_context is not None
                else None
            ),
            max_sources=max_sources,
        )
        key = company_key(job.posting.company_name)
        fingerprint = self._fingerprint(company=key, scope=scope)
        now = datetime.now(timezone.utc)
        cached = self._store.find_completed(
            user_id=user_id,
            company_key=key,
            input_fingerprint=fingerprint,
            worker_version=self._worker_version,
            created_after=now - self._freshness,
        )
        if cached is not None:
            run = self._store.get_run(user_id=user_id, run_id=cached.run_id)
            if run is None:
                raise RuntimeError("cached job research report has no run")
            return JobResearchResult(
                run=run,
                report=cached,
                sources=self._store.list_sources(
                    user_id=user_id,
                    report_id=cached.id,
                ),
                cached=True,
            )
        run = JobResearchRun(
            id=f"job_research_run_{uuid4().hex}",
            user_id=user_id,
            company_key=key,
            job_posting_id=job.posting.id,
            jd_snapshot_id=job.snapshot.id,
            scope=scope,
            status="running",
            input_fingerprint=fingerprint,
            worker_version=self._worker_version,
            started_at=now,
            updated_at=now,
        )
        self._store.create_run(run)
        request = JobResearchWorkerRequest(
            company_name=job.posting.company_name,
            role_title=job.posting.title,
            jd_text=job.snapshot.content,
            scope=scope,
        )
        return self._execute(run=run, request=request, resume=False)

    def retry(self, *, user_id: str, run_id: str) -> JobResearchResult:
        run = self._store.get_run(user_id=user_id, run_id=run_id)
        if run is None:
            raise JobResearchInputNotFoundError("job_research_run")
        if run.status != "failed":
            raise JobResearchRunNotRetryableError(
                "only failed job research runs can be retried"
            )
        job = self._jobs.get_job(
            user_id=user_id,
            job_posting_id=run.job_posting_id,
        )
        snapshot = self._jobs.get_snapshot(
            user_id=user_id,
            jd_snapshot_id=run.jd_snapshot_id,
        )
        if job is None or snapshot is None or snapshot.job_posting_id != run.job_posting_id:
            raise JobResearchInputNotFoundError("job_research_sources")
        running = self._store.mark_running(
            run=run,
            updated_at=datetime.now(timezone.utc),
        )
        request = JobResearchWorkerRequest(
            company_name=job.posting.company_name,
            role_title=job.posting.title,
            jd_text=snapshot.content,
            scope=run.scope,
        )
        return self._execute(run=running, request=request, resume=True)

    def get_report(
        self,
        *,
        user_id: str,
        report_id: str | None = None,
        job_posting_id: str | None = None,
    ) -> JobResearchResult:
        cutoff = datetime.now(timezone.utc) - self._freshness
        if report_id is not None:
            report = self._store.get_report(
                user_id=user_id,
                report_id=report_id,
                outdated_before=cutoff,
            )
        elif job_posting_id is not None:
            job = self._jobs.get_job(
                user_id=user_id,
                job_posting_id=job_posting_id,
            )
            if job is None:
                raise JobResearchInputNotFoundError("job_posting")
            report = self._store.latest_company_report(
                user_id=user_id,
                company_key=company_key(job.posting.company_name),
                outdated_before=cutoff,
            )
        else:
            raise ValueError("report_id or job_posting_id is required")
        if report is None:
            raise JobResearchInputNotFoundError("job_research_report")
        run = self._store.get_run(user_id=user_id, run_id=report.run_id)
        if run is None:
            raise RuntimeError("job research report has no run")
        return JobResearchResult(
            run=run,
            report=report,
            sources=self._store.list_sources(user_id=user_id, report_id=report.id),
            cached=True,
        )

    def _execute(
        self,
        *,
        run: JobResearchRun,
        request: JobResearchWorkerRequest,
        resume: bool,
    ) -> JobResearchResult:
        try:
            draft = self._worker.research(
                run_id=run.id,
                request=request,
                resume=resume,
            )
            sources, report = self._validate_and_materialize(
                run=run,
                draft=draft,
                retrieved_at=datetime.now(timezone.utc),
            )
            stored = self._store.complete(run=run, sources=sources, report=report)
        except Exception as error:
            code = getattr(error, "code", "JOB_RESEARCH_FAILED")
            detail = str(error) or type(error).__name__
            self._store.fail(
                run=run,
                error_code=str(code)[:200],
                error_detail=redact_text(detail)[:2000],
                updated_at=datetime.now(timezone.utc),
            )
            raise JobResearchExecutionError(
                run_id=run.id,
                code=str(code)[:200],
                retryable=bool(getattr(error, "retryable", False)),
                detail=redact_text(detail)[:2000],
            ) from error
        try:
            self._worker.forget(run.id)
        except Exception:
            # The formal report is durable. A retention sweep can remove an
            # orphaned terminal checkpoint without changing the user result.
            pass
        completed = self._store.get_run(user_id=run.user_id, run_id=run.id)
        if completed is None:
            raise RuntimeError("completed job research run disappeared")
        return JobResearchResult(
            run=completed,
            report=stored,
            sources=sources,
            cached=False,
        )

    def _validate_and_materialize(
        self,
        *,
        run: JobResearchRun,
        draft: JobResearchDraft,
        retrieved_at: datetime,
    ) -> tuple[tuple[JobResearchSource, ...], JobResearchReport]:
        if len(draft.sources) > run.scope.max_sources:
            raise JobResearchValidationError("research exceeded max_sources")
        normalized: dict[str, str] = {}
        safe_urls: dict[str, str] = {}
        for source in draft.sources:
            safe_url = redact_text(source.url)
            normalized_url = self._normalize_url(safe_url)
            if normalized_url in normalized.values():
                raise JobResearchValidationError("research contains duplicate source URLs")
            normalized[source.source_key] = normalized_url
            safe_urls[source.source_key] = safe_url
        known = set(normalized)
        referenced: set[str] = set()
        for finding in draft.findings:
            missing = set(finding.source_keys) - known
            if missing:
                raise JobResearchValidationError(
                    f"finding cites unknown source keys: {sorted(missing)}"
                )
            referenced.update(finding.source_keys)
        unused = known - referenced
        if unused:
            raise JobResearchValidationError(
                f"research includes uncited sources: {sorted(unused)}"
            )

        sources = tuple(
            JobResearchSource(
                id=f"job_research_source_{uuid4().hex}",
                run_id=run.id,
                source_key=source.source_key,
                url=safe_urls[source.source_key],
                normalized_url=normalized[source.source_key],
                title=source.title,
                publisher=source.publisher,
                published_at=source.published_at,
                retrieved_at=retrieved_at,
                relevant_excerpt=source.relevant_excerpt,
                content_sha256=hashlib.sha256(
                    source.relevant_excerpt.encode("utf-8")
                ).hexdigest(),
            )
            for source in draft.sources
        )
        report = JobResearchReport(
            id=f"job_research_report_{uuid4().hex}",
            run_id=run.id,
            user_id=run.user_id,
            company_key=run.company_key,
            job_posting_id=run.job_posting_id,
            jd_snapshot_id=run.jd_snapshot_id,
            status="current",
            scope=run.scope,
            summary=draft.summary,
            findings=tuple(
                JobResearchFinding.model_validate(finding.model_dump())
                for finding in draft.findings
            ),
            open_questions=draft.open_questions,
            limitations=draft.limitations,
            created_at=retrieved_at,
        )
        return sources, report

    def _fingerprint(self, *, company: str, scope: JobResearchScope) -> str:
        """Identify a research run by its subject, which is the company.

        The JD is deliberately absent. It supplies search anchors, but business
        lines, market position, and competitors do not change because a second
        role at the same employer was saved, or because the posting's text was
        edited. Keying on the JD meant three saved jobs at one company produced
        three duplicate runs, each with its own freshness window.

        The scope stays in the key: a different focus is a different question,
        and must not be answered from a report that never asked it.
        """
        canonical = json.dumps(
            {
                "company_key": company,
                "scope": scope.model_dump(mode="json"),
                "worker_version": self._worker_version,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_url(value: str) -> str:
        try:
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError as error:
            raise JobResearchValidationError("research source URL is invalid") from error
        scheme = parsed.scheme.casefold()
        host = (parsed.hostname or "").casefold()
        if scheme not in {"http", "https"} or not host:
            raise JobResearchValidationError("research source URL must be HTTP(S)")
        if parsed.username or parsed.password:
            raise JobResearchValidationError("research source URL contains userinfo")
        if port not in {None, 80, 443}:
            raise JobResearchValidationError("research source URL uses an unsupported port")
        netloc = host
        if ":" in host and not host.startswith("["):
            netloc = f"[{host}]"
        if port is not None and not (
            (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
        ):
            netloc = f"{netloc}:{port}"
        query = urlencode(
            sorted(
                (key, item)
                for key, item in parse_qsl(parsed.query, keep_blank_values=True)
                if not key.casefold().startswith(("utm_", "ref_"))
                and key.casefold() not in {"gclid", "fbclid"}
            )
        )
        return urlunsplit((scheme, netloc, parsed.path or "/", query, ""))
