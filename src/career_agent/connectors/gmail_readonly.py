from __future__ import annotations

import base64
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from email.header import decode_header, make_header
from html.parser import HTMLParser
import json
from typing import Any, Protocol
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from career_agent.domain.email_tracking import (
    EmailSyncBatch,
    EmailSyncCursor,
    RemoteEmailContent,
    RemoteEmailMetadata,
)


class GmailTransport(Protocol):
    def get(self, path: str, *, params: Mapping[str, object] | None = None) -> dict[str, Any]: ...


class GmailAPIError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class HTTPSGmailTransport:
    def __init__(self, access_token: str | Callable[[], str], *, timeout: float = 30.0) -> None:
        self._access_token = access_token
        self._timeout = timeout

    def get(self, path: str, *, params: Mapping[str, object] | None = None) -> dict[str, Any]:
        query = urlencode(params or {}, doseq=True)
        url = f"https://gmail.googleapis.com/gmail/v1/{path}"
        if query:
            url = f"{url}?{query}"
        token = self._access_token() if callable(self._access_token) else self._access_token
        request = Request(url, headers={"Authorization": f"Bearer {token}"})
        try:
            with urlopen(request, timeout=self._timeout) as response:
                return json.loads(response.read())
        except HTTPError as error:
            raise GmailAPIError(error.code, "Gmail API request failed") from error


class _PlainTextHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())


class GmailReadOnlyConnector:
    provider = "gmail"

    def __init__(self, transport: GmailTransport) -> None:
        self._transport = transport

    def test_connection(self) -> None:
        self._transport.get("users/me/profile")

    def sync_metadata(
        self,
        *,
        cursor: EmailSyncCursor | None,
        since: datetime,
    ) -> EmailSyncBatch:
        if cursor is not None and cursor.cursor_type != "gmail_history_id":
            raise ValueError("Gmail connector requires a Gmail history cursor")
        message_ids: list[str] = []
        next_history_id: str | None = None
        if cursor is None:
            page_token: str | None = None
            while True:
                params: dict[str, object] = {
                    "maxResults": 100,
                    "q": f"after:{int(since.timestamp())}",
                }
                if page_token:
                    params["pageToken"] = page_token
                response = self._transport.get("users/me/messages", params=params)
                message_ids.extend(item["id"] for item in response.get("messages", ()))
                page_token = response.get("nextPageToken")
                if not page_token:
                    break
            profile = self._transport.get("users/me/profile")
            next_history_id = str(profile["historyId"])
        else:
            page_token = None
            while True:
                params = {
                    "startHistoryId": cursor.value,
                    "historyTypes": ["messageAdded"],
                    "maxResults": 500,
                }
                if page_token:
                    params["pageToken"] = page_token
                response = self._transport.get("users/me/history", params=params)
                for history in response.get("history", ()):
                    message_ids.extend(
                        added["message"]["id"]
                        for added in history.get("messagesAdded", ())
                    )
                next_history_id = str(response.get("historyId") or cursor.value)
                page_token = response.get("nextPageToken")
                if not page_token:
                    break

        seen: set[str] = set()
        messages = []
        for message_id in message_ids:
            if message_id in seen:
                continue
            seen.add(message_id)
            response = self._transport.get(
                f"users/me/messages/{message_id}",
                params={
                    "format": "metadata",
                    "metadataHeaders": ["From", "Subject", "Date", "Message-ID"],
                },
            )
            messages.append(self._metadata(response))
        if next_history_id is None:
            raise GmailAPIError(500, "Gmail API did not return a history cursor")
        return EmailSyncBatch(
            messages=tuple(messages),
            next_cursor_value=next_history_id,
        )

    def get_content(self, *, external_message_id: str) -> RemoteEmailContent:
        response = self._transport.get(
            f"users/me/messages/{external_message_id}",
            params={"format": "full"},
        )
        text = self._payload_text(response.get("payload", {})).strip()
        if not text:
            text = response.get("snippet", "").strip()
        if not text:
            raise ValueError("Gmail message has no readable text body")
        return RemoteEmailContent(external_message_id=external_message_id, text=text)

    @staticmethod
    def _metadata(response: dict[str, Any]) -> RemoteEmailMetadata:
        headers = {
            header["name"].casefold(): str(make_header(decode_header(header["value"])))
            for header in response.get("payload", {}).get("headers", ())
        }
        received_at = datetime.fromtimestamp(
            int(response["internalDate"]) / 1000,
            tz=timezone.utc,
        )
        return RemoteEmailMetadata(
            external_message_id=response["id"],
            external_thread_id=response.get("threadId"),
            sender=headers.get("from", "unknown sender"),
            subject=headers.get("subject", "(no subject)"),
            received_at=received_at,
        )

    @classmethod
    def _payload_text(cls, payload: dict[str, Any]) -> str:
        plain: list[str] = []
        html: list[str] = []

        def visit(part: dict[str, Any]) -> None:
            mime_type = part.get("mimeType", "")
            data = part.get("body", {}).get("data")
            if data and mime_type in {"text/plain", "text/html"}:
                decoded = base64.urlsafe_b64decode(data + "=" * (-len(data) % 4)).decode(
                    "utf-8", errors="replace"
                )
                (plain if mime_type == "text/plain" else html).append(decoded)
            for child in part.get("parts", ()):
                visit(child)

        visit(payload)
        if plain:
            return "\n".join(plain)
        parser = _PlainTextHTMLParser()
        for value in html:
            parser.feed(value)
        return "\n".join(parser.parts)
