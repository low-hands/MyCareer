from __future__ import annotations

from typing import Protocol
from uuid import uuid4


class ConnectorSecretError(RuntimeError):
    pass


class ConnectorSecretStore(Protocol):
    def put(self, secret: str) -> str: ...
    def get(self, reference: str) -> str: ...
    def delete(self, reference: str) -> None: ...


class KeyringConnectorSecretStore:
    """Keep connector credentials in the operating-system credential vault."""

    SERVICE = "career-agent-connectors"
    PREFIX = "keyring:"

    def __init__(self, backend=None) -> None:
        if backend is None:
            try:
                import keyring
            except ImportError as error:
                raise ConnectorSecretError("System keyring support is unavailable") from error
            backend = keyring
        self._backend = backend

    def put(self, secret: str) -> str:
        if not secret:
            raise ConnectorSecretError("Connector secret must not be empty")
        secret_id = uuid4().hex
        try:
            self._backend.set_password(self.SERVICE, secret_id, secret)
        except Exception as error:
            raise ConnectorSecretError("Could not write to the system keyring") from error
        return f"{self.PREFIX}{secret_id}"

    def get(self, reference: str) -> str:
        secret_id = self._secret_id(reference)
        try:
            secret = self._backend.get_password(self.SERVICE, secret_id)
        except Exception as error:
            raise ConnectorSecretError("Could not read from the system keyring") from error
        if not secret:
            raise ConnectorSecretError("Connector credential is unavailable")
        return secret

    def delete(self, reference: str) -> None:
        secret_id = self._secret_id(reference)
        try:
            if self._backend.get_password(self.SERVICE, secret_id) is not None:
                self._backend.delete_password(self.SERVICE, secret_id)
        except Exception as error:
            raise ConnectorSecretError("Could not delete from the system keyring") from error

    def _secret_id(self, reference: str) -> str:
        if not reference.startswith(self.PREFIX):
            raise ConnectorSecretError("Invalid keyring credential reference")
        secret_id = reference[len(self.PREFIX):]
        if not secret_id or not secret_id.isalnum():
            raise ConnectorSecretError("Invalid keyring credential reference")
        return secret_id
