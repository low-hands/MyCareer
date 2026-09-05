import pytest

from career_agent.storage.connector_secrets import (
    ConnectorSecretError,
    KeyringConnectorSecretStore,
)


class FakeKeyring:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    def set_password(self, service: str, name: str, value: str) -> None:
        self.values[(service, name)] = value

    def get_password(self, service: str, name: str) -> str | None:
        return self.values.get((service, name))

    def delete_password(self, service: str, name: str) -> None:
        del self.values[(service, name)]


def test_keyring_store_returns_opaque_reference_and_deletes_secret() -> None:
    backend = FakeKeyring()
    store = KeyringConnectorSecretStore(backend)

    reference = store.put("refresh-token")

    assert reference.startswith("keyring:")
    assert "refresh-token" not in reference
    assert store.get(reference) == "refresh-token"
    store.delete(reference)
    with pytest.raises(ConnectorSecretError, match="unavailable"):
        store.get(reference)


def test_keyring_store_rejects_non_keyring_reference() -> None:
    with pytest.raises(ConnectorSecretError, match="Invalid"):
        KeyringConnectorSecretStore(FakeKeyring()).get("env:GOOGLE_TOKEN")
