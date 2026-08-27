from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field

from career_agent.domain.job_discovery import JobDetail, Provenance, SearchResult
from career_agent.security.redaction import redact_text


class BossAdapterError(Exception):
    def __init__(self, code: str, message: str, *, recoverable: bool, recovery_action: str | None = None, operation: str | None = None, hints: dict[str, str] | None = None, transport_returncode: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.recoverable = recoverable
        self.recovery_action = recovery_action
        self.operation = operation
        self.hints = hints or {}
        self.transport_returncode = transport_returncode


class BossErrorPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    code: str | None = None
    message: str | None = None


class BossEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    schema_version: Literal["1.0"]
    command: str
    data: Any = None
    pagination: Any = None
    error: BossErrorPayload | None = None
    hints: dict[str, Any] | None = None


class SearchQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1)
    city: str | None = None
    salary: str | None = None
    experience: str | None = None
    education: str | None = None
    welfare: str | None = None
    page: int = Field(default=1, ge=1)


class AuthStatus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state: str
    message: str | None = None


class BossTransport(Protocol):
    def __call__(self, args: Sequence[str]) -> str: ...


_ALLOWED_COMMANDS = frozenset({"status", "search", "detail"})
_RECOVERABLE = {
    "AUTH_REQUIRED": "Complete BOSS login outside the Agent, then resume the task.",
    "AUTH_EXPIRED": "Restore the BOSS login outside the Agent, then resume the task.",
    "TOKEN_REFRESH_FAILED": "Restore the BOSS login outside the Agent, then resume the task.",
    "RATE_LIMITED": "Wait for the platform limit to clear, then explicitly resume.",
    "NETWORK_ERROR": "Check the network and explicitly retry the task.",
    "UNKNOWN": "BOSS returned an unknown error; wait and explicitly retry this detail request.",
    "CLI_ERROR": "BOSS CLI failed; inspect the local BOSS session and explicitly retry.",
}

_HINT_KEYS = frozenset({"retry_after", "retry_after_seconds", "auth_state", "operation", "message"})


def _safe_hints(hints: Mapping[str, Any] | None) -> dict[str, str]:
    if not isinstance(hints, Mapping):
        return {}
    return {key: str(hints[key])[:160] for key in _HINT_KEYS if key in hints and hints[key] is not None}

def parse_envelope(raw: str | Mapping[str, Any]) -> BossEnvelope:
    try:
        envelope = BossEnvelope.model_validate_json(raw) if isinstance(raw, str) else BossEnvelope.model_validate(raw)
    except (ValueError, TypeError) as error:
        raise BossAdapterError("MALFORMED_RESPONSE", "BOSS returned an unsupported response envelope.", recoverable=False) from error
    if not envelope.ok:
        code = envelope.error.code if envelope.error and envelope.error.code else "CLI_ERROR"
        message = envelope.error.message if envelope.error and envelope.error.message else "BOSS operation failed."
        recovery_action = _RECOVERABLE.get(code)
        return_error = BossAdapterError(code, message, recoverable=recovery_action is not None, recovery_action=recovery_action, operation=envelope.command, hints=_safe_hints(envelope.hints))
        raise return_error
    return envelope


class SubprocessBossTransport:
    """Fixed-argv transport; the host owns the executable and credential directory."""

    def __init__(self, data_dir: Path, executable: str = "boss", *, timeout_seconds: float = 30.0, max_output_bytes: int = 2_000_000) -> None:
        self.data_dir = data_dir
        self.executable = executable
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes

    def __call__(self, args: Sequence[str]) -> str:
        if not args or args[0] not in _ALLOWED_COMMANDS:
            raise BossAdapterError("OPERATION_NOT_ALLOWED", "Only BOSS status, search, and detail are allowed.", recoverable=False)
        if any("\x00" in argument for argument in args):
            raise BossAdapterError("INVALID_PARAM", "BOSS arguments cannot contain NUL bytes.", recoverable=False)
        try:
            completed = subprocess.run(
                [self.executable, "--data-dir", str(self.data_dir), *args],
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                shell=False,
            )
        except subprocess.TimeoutExpired as error:
            raise BossAdapterError(
                "TIMEOUT",
                "BOSS command timed out.",
                recoverable=True,
                recovery_action="Wait for the BOSS request to finish, then explicitly retry the task.",
                operation=args[0],
            ) from error
        output = completed.stdout
        if len(output.encode("utf-8")) > self.max_output_bytes:
            raise BossAdapterError("OUTPUT_TOO_LARGE", "BOSS response exceeded the adapter output limit.", recoverable=False)
        if completed.returncode != 0 and not output.strip():
            raise BossAdapterError("CLI_ERROR", "BOSS command failed without a structured response.", recoverable=True, recovery_action=_RECOVERABLE["CLI_ERROR"], operation=args[0], transport_returncode=completed.returncode)
        return output


class BossReadOnlyAdapter:
    def __init__(self, transport: BossTransport, *, adapter_version: str = "boss-readonly-v1", clock: Callable[[], datetime] | None = None) -> None:
        self._transport = transport
        self.adapter_version = adapter_version
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def status(self) -> AuthStatus:
        envelope = parse_envelope(self._transport(("status",)))
        data = envelope.data if isinstance(envelope.data, Mapping) else {}
        logged_in = data.get("logged_in")
        if logged_in is True:
            state = "authenticated"
        elif logged_in is False:
            state = "auth_required"
        else:
            state = str(data.get("auth_state") or data.get("state") or data.get("status") or "unknown")
        return AuthStatus(state=state, message=str(data.get("auth_summary") or data.get("message")) if data.get("auth_summary") or data.get("message") else None)

    def search(self, query: SearchQuery) -> tuple[SearchResult, ...]:
        args = ["search", query.query, "--page", str(query.page)]
        for name, value in (("city", query.city), ("salary", query.salary), ("experience", query.experience), ("education", query.education), ("welfare", query.welfare)):
            if value:
                args.extend([f"--{name}", value])
        envelope = parse_envelope(self._transport(tuple(args)))
        if not isinstance(envelope.data, list):
            raise BossAdapterError("INVALID_SCHEMA", "BOSS search data is not a list.", recoverable=False)
        captured_at = self._clock()
        return tuple(self._search_result(item, captured_at) for item in envelope.data if isinstance(item, Mapping))

    def detail(self, security_id: str, job_id: str | None = None) -> JobDetail:
        try:
            return self._detail(security_id, job_id)
        except BossAdapterError as error:
            if error.code != "INVALID_PARAM" or not job_id:
                raise
            return self._detail(security_id, None)

    def _detail(self, security_id: str, job_id: str | None) -> JobDetail:
        args = ["detail", security_id]
        if job_id:
            args.extend(["--job-id", job_id])
        envelope = parse_envelope(self._transport(tuple(args)))
        if not isinstance(envelope.data, Mapping):
            raise BossAdapterError("INVALID_SCHEMA", "BOSS detail data is not an object.", recoverable=False)
        data = envelope.data
        captured_at = self._clock()
        source_job_id = str(data.get("job_id") or data.get("encrypt_job_id") or job_id or "") or None
        provenance = Provenance(source_name="boss", captured_at=captured_at, operation="detail", adapter_version=self.adapter_version, source_job_id=source_job_id)
        return JobDetail(
            source_name="boss",
            source_job_id=source_job_id,
            security_id=security_id,
            title=str(data.get("title") or ""),
            company_name=str(data.get("company") or ""),
            # The JD body is free text from the platform and is the one field
            # large enough to carry a tracking token or session URL into both the
            # snapshot store and the analysis prompt.
            description=redact_text(str(data.get("description") or "")),
            captured_at=captured_at,
            provenance=provenance,
            city=str(data.get("city") or "") or None,
            salary=str(data.get("salary") or "") or None,
            experience=str(data.get("experience") or "") or None,
            education=str(data.get("education") or "") or None,
            labels=tuple(str(label) for label in data.get("skills", ()) if label),
            source_url=None,
        )

    def _search_result(self, item: Mapping[str, Any], captured_at: datetime) -> SearchResult:
        source_job_id = str(item.get("job_id") or item.get("encrypt_job_id") or "") or None
        security_id = str(item.get("security_id") or "") or None
        source_url = self._source_url(item.get("url"))
        provenance = Provenance(source_name="boss", captured_at=captured_at, operation="search", adapter_version=self.adapter_version, source_job_id=source_job_id, source_url=source_url)
        return SearchResult(
            result_ref=f"boss:{security_id or source_job_id or captured_at.isoformat()}",
            source_name="boss",
            source_job_id=source_job_id,
            source_url=source_url,
            security_id=security_id,
            title=str(item.get("title") or item.get("job_name") or ""),
            company_name=str(item.get("company") or item.get("brand_name") or ""),
            captured_at=captured_at,
            provenance=provenance,
            city=str(item.get("city") or "") or None,
            salary=str(item.get("salary") or "") or None,
            experience=str(item.get("experience") or "") or None,
            education=str(item.get("education") or "") or None,
            labels=tuple(str(label) for label in item.get("skills", ()) if label),
        )

    @staticmethod
    def _source_url(value: Any) -> str | None:
        if not isinstance(value, str) or len(value) > 2_000:
            return None
        try:
            parsed = urlsplit(value)
        except ValueError:
            return None
        host = (parsed.hostname or "").casefold()
        if parsed.scheme != "https" or (host != "zhipin.com" and not host.endswith(".zhipin.com")):
            return None
        if parsed.username or parsed.password or parsed.port not in {None, 443}:
            return None
        return urlunsplit(("https", host, parsed.path, "", ""))

