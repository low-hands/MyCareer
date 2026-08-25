import base64
from datetime import datetime, timezone

from career_agent.connectors.gmail_readonly import GmailReadOnlyConnector
from career_agent.domain.email_tracking import EmailSyncCursor


class Transport:
    def __init__(self):
        self.calls = []

    def get(self, path, *, params=None):
        self.calls.append((path, params))
        if path == "users/me/history":
            return {
                "historyId": "12",
                "history": [{"messagesAdded": [{"message": {"id": "m1"}}]}],
            }
        if path == "users/me/messages/m1" and params["format"] == "metadata":
            return {
                "id": "m1", "threadId": "t1", "internalDate": "1787184000000",
                "payload": {"headers": [
                    {"name": "From", "value": "Acme <jobs@acme.example>"},
                    {"name": "Subject", "value": "Interview"},
                ]},
            }
        if path == "users/me/messages/m1":
            encoded = base64.urlsafe_b64encode("面试邀请".encode()).decode().rstrip("=")
            return {"payload": {"mimeType": "text/plain", "body": {"data": encoded}}}
        raise AssertionError(path)


def test_gmail_incremental_sync_and_body_decode() -> None:
    transport = Transport()
    connector = GmailReadOnlyConnector(transport)
    cursor = EmailSyncCursor(
        account_id="a1", cursor_type="gmail_history_id", value="10",
        updated_at=datetime.now(timezone.utc),
    )

    batch = connector.sync_metadata(
        cursor=cursor, since=datetime(2026, 8, 1, tzinfo=timezone.utc)
    )
    content = connector.get_content(external_message_id="m1")

    assert batch.next_cursor_value == "12"
    assert batch.messages[0].external_thread_id == "t1"
    assert content.text == "面试邀请"
    assert transport.calls[0][0] == "users/me/history"
