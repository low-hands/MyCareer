from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from email import policy
from email.header import decode_header, make_header
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
import imaplib
from typing import Any

from career_agent.domain.email_tracking import (
    EmailSyncBatch,
    EmailSyncCursor,
    RemoteEmailContent,
    RemoteEmailMetadata,
)


def _decode_header(value: str | None, fallback: str) -> str:
    return str(make_header(decode_header(value))) if value else fallback


def _message_text(message: Any) -> str:
    if message.is_multipart():
        plain: list[str] = []
        html: list[str] = []
        for part in message.walk():
            if part.get_content_disposition() == "attachment":
                continue
            if part.get_content_type() == "text/plain":
                plain.append(part.get_content())
            elif part.get_content_type() == "text/html":
                html.append(part.get_content())
        if plain:
            return "\n".join(plain).strip()
        if html:
            from career_agent.connectors.gmail_readonly import _PlainTextHTMLParser

            parser = _PlainTextHTMLParser()
            for value in html:
                parser.feed(value)
            return "\n".join(parser.parts).strip()
        return ""
    return message.get_content().strip()


class QQEmailReadOnlyConnector:
    provider = "qq"

    def __init__(
        self,
        email_address: str,
        authorization_code: str,
        *,
        client_factory: Callable[..., Any] = imaplib.IMAP4_SSL,
        host: str = "imap.qq.com",
        port: int = 993,
    ) -> None:
        self._email_address = email_address
        self._authorization_code = authorization_code
        self._client_factory = client_factory
        self._host = host
        self._port = port

    def test_connection(self) -> None:
        client = self._connect()
        try:
            status, _ = client.select("INBOX", readonly=True)
            if status != "OK":
                raise RuntimeError("QQ IMAP could not open INBOX read-only")
        finally:
            client.logout()

    def sync_metadata(
        self,
        *,
        cursor: EmailSyncCursor | None,
        since: datetime,
    ) -> EmailSyncBatch:
        if cursor is not None and cursor.cursor_type != "imap_uid":
            raise ValueError("QQ connector requires an IMAP UID cursor")
        client = self._connect()
        try:
            status, _ = client.select("INBOX", readonly=True)
            if status != "OK":
                raise RuntimeError("QQ IMAP could not open INBOX read-only")
            uid_validity = self._uid_validity(client)
            effective_cursor = cursor
            if cursor is not None and cursor.uid_validity != uid_validity:
                effective_cursor = None
            if effective_cursor is None:
                search_args = (None, "SINCE", since.strftime("%d-%b-%Y"))
            else:
                search_args = (None, "UID", f"{int(effective_cursor.value) + 1}:*")
            search_status, search_data = client.uid("search", *search_args)
            if search_status != "OK":
                raise RuntimeError("QQ IMAP UID search failed")
            uids = [value for value in search_data[0].split() if value]
            if effective_cursor is not None:
                uids = [uid for uid in uids if int(uid) > int(effective_cursor.value)]
            messages = tuple(self._fetch_metadata(client, uid) for uid in uids)
            last_uid = max(
                [int(uid) for uid in uids]
                + ([int(effective_cursor.value)] if effective_cursor else [0])
            )
            return EmailSyncBatch(
                messages=messages,
                next_cursor_value=str(last_uid),
                uid_validity=uid_validity,
            )
        finally:
            client.logout()

    def get_content(self, *, external_message_id: str) -> RemoteEmailContent:
        client = self._connect()
        try:
            status, _ = client.select("INBOX", readonly=True)
            if status != "OK":
                raise RuntimeError("QQ IMAP could not open INBOX read-only")
            fetch_status, data = client.uid(
                "fetch", external_message_id, "(BODY.PEEK[])"
            )
            if fetch_status != "OK":
                raise RuntimeError("QQ IMAP message fetch failed")
            raw = self._response_bytes(data)
            text = _message_text(BytesParser(policy=policy.default).parsebytes(raw))
            if not text:
                raise ValueError("QQ email has no readable text body")
            return RemoteEmailContent(external_message_id=external_message_id, text=text)
        finally:
            client.logout()

    def _connect(self) -> Any:
        client = self._client_factory(self._host, self._port)
        status, _ = client.login(self._email_address, self._authorization_code)
        if status != "OK":
            client.logout()
            raise RuntimeError("QQ IMAP authorization failed")
        return client

    @staticmethod
    def _uid_validity(client: Any) -> str:
        _, values = client.response("UIDVALIDITY")
        if not values or values[0] is None:
            raise RuntimeError("QQ IMAP did not return UIDVALIDITY")
        value = values[0]
        return value.decode() if isinstance(value, bytes) else str(value)

    @classmethod
    def _fetch_metadata(cls, client: Any, uid: bytes) -> RemoteEmailMetadata:
        status, data = client.uid(
            "fetch",
            uid,
            "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE MESSAGE-ID REFERENCES)])",
        )
        if status != "OK":
            raise RuntimeError("QQ IMAP metadata fetch failed")
        message = BytesParser(policy=policy.default).parsebytes(cls._response_bytes(data))
        received_at = parsedate_to_datetime(message.get("Date")) if message.get("Date") else datetime.now(timezone.utc)
        if received_at.tzinfo is None:
            received_at = received_at.replace(tzinfo=timezone.utc)
        references = message.get("References")
        return RemoteEmailMetadata(
            external_message_id=uid.decode(),
            external_thread_id=(references.split()[-1] if references else message.get("Message-ID")),
            sender=_decode_header(message.get("From"), "unknown sender"),
            subject=_decode_header(message.get("Subject"), "(no subject)"),
            received_at=received_at,
        )

    @staticmethod
    def _response_bytes(data: list[Any]) -> bytes:
        for item in data:
            if isinstance(item, tuple) and isinstance(item[1], bytes):
                return item[1]
        raise RuntimeError("IMAP response did not contain a message")
