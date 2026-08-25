from datetime import datetime, timezone

from career_agent.connectors.qq_email_readonly import QQEmailReadOnlyConnector
from career_agent.domain.email_tracking import EmailSyncCursor


HEADER = (
    b"From: Acme Recruiting <jobs@acme.example>\r\n"
    b"Subject: Interview invitation\r\n"
    b"Date: Thu, 20 Aug 2026 10:00:00 +0800\r\n"
    b"Message-ID: <message-8@acme.example>\r\n\r\n"
)
BODY = (
    b"From: Acme Recruiting <jobs@acme.example>\r\n"
    b"Subject: Interview invitation\r\n"
    b"Date: Thu, 20 Aug 2026 10:00:00 +0800\r\n"
    b"Message-ID: <message-8@acme.example>\r\n"
    b"Content-Type: text/plain; charset=utf-8\r\n\r\n"
    b"Interview invitation"
)


class IMAPClient:
    def __init__(self, host, port):
        self.calls = []

    def login(self, address, authorization_code):
        self.calls.append(("login", address, authorization_code))
        return "OK", []

    def select(self, mailbox, readonly=False):
        self.calls.append(("select", mailbox, readonly))
        return "OK", [b"1"]

    def response(self, name):
        return "UIDVALIDITY", [b"99"]

    def uid(self, command, *args):
        self.calls.append(("uid", command, *args))
        if command == "search":
            return "OK", [b"8"]
        if "HEADER.FIELDS" in args[-1]:
            return "OK", [(b"8 FETCH", HEADER), b")"]
        return "OK", [(b"8 FETCH", BODY), b")"]

    def logout(self):
        self.calls.append(("logout",))
        return "BYE", []


def test_qq_connector_uses_readonly_uid_and_body_peek() -> None:
    clients = []

    def factory(host, port):
        client = IMAPClient(host, port)
        clients.append(client)
        return client

    connector = QQEmailReadOnlyConnector(
        "123@qq.com", "authorization-code", client_factory=factory
    )
    cursor = EmailSyncCursor(
        account_id="a1", cursor_type="imap_uid", value="7", uid_validity="99",
        updated_at=datetime.now(timezone.utc),
    )

    batch = connector.sync_metadata(
        cursor=cursor, since=datetime(2026, 8, 1, tzinfo=timezone.utc)
    )
    content = connector.get_content(external_message_id="8")

    assert batch.next_cursor_value == "8"
    assert batch.uid_validity == "99"
    assert batch.messages[0].subject == "Interview invitation"
    assert content.text == "Interview invitation"
    assert all(
        call != ("select", "INBOX", False)
        for client in clients for call in client.calls
    )
    fetch_calls = [
        call for client in clients for call in client.calls
        if call[:2] == ("uid", "fetch")
    ]
    assert all("BODY.PEEK" in call[-1] for call in fetch_calls)
