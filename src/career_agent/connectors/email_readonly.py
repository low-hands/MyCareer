from __future__ import annotations

from datetime import datetime
from typing import Protocol

from career_agent.domain.email_tracking import (
    EmailProvider,
    EmailSyncBatch,
    EmailSyncCursor,
    RemoteEmailContent,
)


class ReadOnlyEmailConnector(Protocol):
    """Provider boundary. Implementations must never mutate mailbox state."""

    provider: EmailProvider

    def test_connection(self) -> None: ...

    def sync_metadata(
        self,
        *,
        cursor: EmailSyncCursor | None,
        since: datetime,
    ) -> EmailSyncBatch: ...

    def get_content(self, *, external_message_id: str) -> RemoteEmailContent: ...


class EmailConnectorResolver(Protocol):
    def resolve(
        self,
        *,
        provider: EmailProvider,
        email_address: str,
        credential_ref: str,
    ) -> ReadOnlyEmailConnector: ...
