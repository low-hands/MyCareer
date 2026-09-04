from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from typing import Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from career_agent.connectors.email_accounts import (
    EmailCredentialError,
    GoogleOAuthTokenProvider,
)
from career_agent.domain.calendar import (
    CalendarAccount,
    CalendarEventPayload,
    CalendarOperation,
)


class CalendarConnectorError(RuntimeError):
    def __init__(
        self, code: str, detail: str, *, outcome_unknown: bool = False
    ) -> None:
        super().__init__(detail)
        self.code = code
        self.outcome_unknown = outcome_unknown


@dataclass(frozen=True)
class CalendarWriteResult:
    external_event_id: str
    etag: str | None = None
    html_link: str | None = None


@dataclass(frozen=True)
class CalendarReconciliationResult:
    outcome: Literal["applied", "not_applied", "conflict"]
    write_result: CalendarWriteResult | None = None


class CalendarConnector(Protocol):
    def apply(
        self,
        *,
        operation: CalendarOperation,
        calendar_id: str,
        external_event_id: str,
        payload: CalendarEventPayload | None,
        idempotency_key: str,
        payload_hash: str,
    ) -> CalendarWriteResult: ...

    def reconcile(
        self,
        *,
        operation: CalendarOperation,
        calendar_id: str,
        external_event_id: str,
        payload_hash: str,
        prior_payload_hash: str | None = None,
    ) -> CalendarReconciliationResult: ...


class GoogleCalendarConnector:
    """Minimal Google Calendar v3 writer; it never adds attendees or emails guests."""

    def __init__(self, token_provider, *, timeout: float = 30.0) -> None:
        self._token_provider = token_provider
        self._timeout = timeout

    def apply(
        self,
        *,
        operation: CalendarOperation,
        calendar_id: str,
        external_event_id: str,
        payload: CalendarEventPayload | None,
        idempotency_key: str,
        payload_hash: str,
    ) -> CalendarWriteResult:
        encoded_calendar = quote(calendar_id, safe="")
        encoded_event = quote(external_event_id, safe="")
        base = f"https://www.googleapis.com/calendar/v3/calendars/{encoded_calendar}/events"
        query = urlencode({"sendUpdates": "none"})
        body = None
        if operation == "create":
            if payload is None:
                raise ValueError("calendar create requires payload")
            method = "POST"
            url = f"{base}?{query}"
            body_payload = self._event_body(
                payload,
                idempotency_key=idempotency_key,
                payload_hash=payload_hash,
            )
            body_payload["id"] = external_event_id
            body = json.dumps(body_payload).encode()
        elif operation == "update":
            if payload is None:
                raise ValueError("calendar update requires payload")
            method = "PATCH"
            url = f"{base}/{encoded_event}?{query}"
            body = json.dumps(
                self._event_body(
                    payload,
                    idempotency_key=idempotency_key,
                    payload_hash=payload_hash,
                )
            ).encode()
        else:
            method = "DELETE"
            url = f"{base}/{encoded_event}?{query}"
        try:
            token = self._token_provider()
            request = Request(
                url,
                data=body,
                method=method,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )
            with urlopen(request, timeout=self._timeout) as response:
                raw = response.read()
        except EmailCredentialError as error:
            raise CalendarConnectorError(
                "CALENDAR_CREDENTIAL_ERROR", str(error)
            ) from error
        except HTTPError as error:
            if operation == "cancel" and error.code == 404:
                return CalendarWriteResult(external_event_id=external_event_id)
            detail = error.read().decode("utf-8", errors="replace")[:2000]
            raise CalendarConnectorError(
                f"GOOGLE_CALENDAR_HTTP_{error.code}",
                detail or str(error),
                outcome_unknown=(error.code in {408, 409} or error.code >= 500),
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise CalendarConnectorError(
                "GOOGLE_CALENDAR_TRANSPORT_ERROR",
                str(error),
                outcome_unknown=True,
            ) from error
        if operation == "cancel":
            return CalendarWriteResult(external_event_id=external_event_id)
        try:
            result = json.loads(raw or b"{}")
        except json.JSONDecodeError as error:
            raise CalendarConnectorError(
                "GOOGLE_CALENDAR_INVALID_RESPONSE",
                "Calendar response was not JSON",
                outcome_unknown=True,
            ) from error
        returned_id = result.get("id")
        if not isinstance(returned_id, str) or returned_id != external_event_id:
            raise CalendarConnectorError(
                "GOOGLE_CALENDAR_ID_MISMATCH",
                "Calendar response did not confirm the fixed external event ID",
                outcome_unknown=True,
            )
        return CalendarWriteResult(
            external_event_id=returned_id,
            etag=result.get("etag") if isinstance(result.get("etag"), str) else None,
            html_link=(
                result.get("htmlLink")
                if isinstance(result.get("htmlLink"), str)
                else None
            ),
        )

    def reconcile(
        self,
        *,
        operation: CalendarOperation,
        calendar_id: str,
        external_event_id: str,
        payload_hash: str,
        prior_payload_hash: str | None = None,
    ) -> CalendarReconciliationResult:
        encoded_calendar = quote(calendar_id, safe="")
        encoded_event = quote(external_event_id, safe="")
        url = (
            "https://www.googleapis.com/calendar/v3/calendars/"
            f"{encoded_calendar}/events/{encoded_event}"
        )
        try:
            token = self._token_provider()
            request = Request(
                url,
                method="GET",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                },
            )
            with urlopen(request, timeout=self._timeout) as response:
                raw = response.read()
        except EmailCredentialError as error:
            raise CalendarConnectorError(
                "CALENDAR_CREDENTIAL_ERROR", str(error)
            ) from error
        except HTTPError as error:
            if error.code in {404, 410}:
                return CalendarReconciliationResult(
                    outcome="applied" if operation == "cancel" else "not_applied",
                    write_result=(
                        CalendarWriteResult(external_event_id=external_event_id)
                        if operation == "cancel"
                        else None
                    ),
                )
            detail = error.read().decode("utf-8", errors="replace")[:2000]
            raise CalendarConnectorError(
                f"GOOGLE_CALENDAR_HTTP_{error.code}", detail or str(error)
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise CalendarConnectorError(
                "GOOGLE_CALENDAR_RECONCILIATION_UNAVAILABLE", str(error)
            ) from error

        if operation == "cancel":
            return CalendarReconciliationResult(outcome="not_applied")
        try:
            result = json.loads(raw or b"{}")
        except json.JSONDecodeError as error:
            raise CalendarConnectorError(
                "GOOGLE_CALENDAR_INVALID_RESPONSE",
                "Calendar reconciliation response was not JSON",
            ) from error
        returned_id = result.get("id")
        if not isinstance(returned_id, str) or returned_id != external_event_id:
            raise CalendarConnectorError(
                "GOOGLE_CALENDAR_ID_MISMATCH",
                "Calendar reconciliation did not return the fixed external event ID",
            )
        extended = result.get("extendedProperties")
        private = extended.get("private") if isinstance(extended, dict) else None
        remote_hash = (
            private.get("careerAgentPayloadHash")
            if isinstance(private, dict)
            else None
        )
        if remote_hash == payload_hash:
            return CalendarReconciliationResult(
                outcome="applied",
                write_result=CalendarWriteResult(
                    external_event_id=returned_id,
                    etag=result.get("etag") if isinstance(result.get("etag"), str) else None,
                    html_link=(
                        result.get("htmlLink")
                        if isinstance(result.get("htmlLink"), str)
                        else None
                    ),
                ),
            )
        if operation == "update" and remote_hash == prior_payload_hash:
            return CalendarReconciliationResult(outcome="not_applied")
        if remote_hash != payload_hash:
            return CalendarReconciliationResult(outcome="conflict")
        raise AssertionError("unreachable calendar reconciliation outcome")

    @staticmethod
    def _event_body(
        payload: CalendarEventPayload,
        *,
        idempotency_key: str,
        payload_hash: str,
    ) -> dict[str, object]:
        body: dict[str, object] = {
            "summary": payload.title,
            "description": payload.description,
            "start": {
                "dateTime": payload.start_at.isoformat(),
                "timeZone": payload.timezone,
            },
            "end": {
                "dateTime": payload.end_at.isoformat(),
                "timeZone": payload.timezone,
            },
            "extendedProperties": {
                "private": {
                    "careerAgentExecutionKey": idempotency_key,
                    "careerAgentPayloadHash": payload_hash,
                }
            },
        }
        if payload.location is not None:
            body["location"] = payload.location
        return body


class EnvironmentCalendarConnectorResolver:
    def __init__(self, environ: Mapping[str, str] | None = None) -> None:
        self._environ = environ if environ is not None else os.environ
        self._providers: dict[str, GoogleOAuthTokenProvider] = {}

    def resolve(self, account: CalendarAccount) -> CalendarConnector:
        if account.provider != "google":
            raise CalendarConnectorError(
                "UNSUPPORTED_CALENDAR_PROVIDER", account.provider
            )
        prefix = "env:"
        if not account.credential_ref.startswith(prefix):
            raise CalendarConnectorError(
                "INVALID_CALENDAR_CREDENTIAL_REF",
                "Only env: credential references are supported",
            )
        name = account.credential_ref[len(prefix):]
        if not name or not name.replace("_", "").isalnum():
            raise CalendarConnectorError(
                "INVALID_CALENDAR_CREDENTIAL_REF", "Invalid environment variable name"
            )
        secret = self._environ.get(name, "")
        if not secret:
            raise CalendarConnectorError(
                "CALENDAR_CREDENTIAL_UNAVAILABLE",
                f"Credential environment variable is unavailable: {name}",
            )
        provider = self._providers.get(account.credential_ref)
        if provider is None:
            try:
                provider = GoogleOAuthTokenProvider(secret)
            except EmailCredentialError as error:
                raise CalendarConnectorError(
                    "CALENDAR_CREDENTIAL_ERROR", str(error)
                ) from error
            self._providers[account.credential_ref] = provider
        return GoogleCalendarConnector(provider)
