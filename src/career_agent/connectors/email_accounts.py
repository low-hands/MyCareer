from __future__ import annotations

from collections.abc import Mapping
import json
import os
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from career_agent.connectors.email_readonly import ReadOnlyEmailConnector
from career_agent.connectors.gmail_readonly import GmailReadOnlyConnector, HTTPSGmailTransport
from career_agent.connectors.qq_email_readonly import QQEmailReadOnlyConnector
from career_agent.domain.email_tracking import EmailProvider
from career_agent.storage.connector_secrets import (
    ConnectorSecretError,
    ConnectorSecretStore,
)


class EmailCredentialError(ValueError):
    pass


class GoogleOAuthTokenProvider:
    """Refreshes a Gmail OAuth access token without persisting it locally."""

    def __init__(
        self,
        credential_json: str,
        *,
        environ: Mapping[str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        try:
            credential = json.loads(credential_json)
        except json.JSONDecodeError as error:
            raise EmailCredentialError("Gmail credential must be JSON") from error
        if (
            not isinstance(credential, dict)
            or not isinstance(credential.get("refresh_token"), str)
            or not credential["refresh_token"]
        ):
            raise EmailCredentialError("Gmail credential requires refresh_token")
        self._credential: dict[str, str] = credential
        self._environ = environ if environ is not None else os.environ
        self._timeout = timeout
        self._access_token: str | None = None
        self._expires_at = 0.0

    def __call__(self) -> str:
        if self._access_token is not None and time.time() < self._expires_at - 60:
            return self._access_token
        client_id = self._environ.get("GOOGLE_OAUTH_CLIENT_ID", "")
        client_secret = self._environ.get("GOOGLE_OAUTH_CLIENT_SECRET", "")
        if not client_id:
            # Older credentials may carry the client ID to distinguish OAuth
            # clients. The app secret is intentionally never read from them.
            client_id = self._credential.get("client_id", "")
        if not client_id or not client_secret:
            raise EmailCredentialError(
                "Google OAuth client credentials are unavailable in the environment"
            )
        body = urlencode(
            {
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": self._credential["refresh_token"],
                "grant_type": "refresh_token",
            }
        ).encode()
        request = Request(
            "https://oauth2.googleapis.com/token",
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urlopen(request, timeout=self._timeout) as response:
            payload: dict[str, Any] = json.loads(response.read())
        token = payload.get("access_token")
        if not isinstance(token, str) or not token:
            raise EmailCredentialError("Google OAuth response has no access token")
        self._access_token = token
        self._expires_at = time.time() + int(payload.get("expires_in", 3600))
        return token


class EnvironmentEmailConnectorResolver:
    """Resolves secret refs such as env:CAREER_GMAIL_ACCOUNT_1 at the boundary."""

    def __init__(
        self,
        environ: Mapping[str, str] | None = None,
        secret_store: ConnectorSecretStore | None = None,
    ) -> None:
        self._environ = environ if environ is not None else os.environ
        self._secret_store = secret_store
        self._gmail_token_providers: dict[str, GoogleOAuthTokenProvider] = {}

    def resolve(
        self,
        *,
        provider: EmailProvider,
        email_address: str,
        credential_ref: str,
    ) -> ReadOnlyEmailConnector:
        secret = self._read_secret(credential_ref)
        if provider == "gmail":
            token_provider = self._gmail_token_providers.get(credential_ref)
            if token_provider is None:
                token_provider = GoogleOAuthTokenProvider(secret, environ=self._environ)
                self._gmail_token_providers[credential_ref] = token_provider
            return GmailReadOnlyConnector(HTTPSGmailTransport(token_provider))
        if provider == "qq":
            return QQEmailReadOnlyConnector(email_address, secret)
        raise ValueError(f"Unsupported email provider: {provider}")

    def _read_secret(self, credential_ref: str) -> str:
        if credential_ref.startswith("keyring:"):
            if self._secret_store is None:
                raise EmailCredentialError("System keyring resolver is unavailable")
            try:
                return self._secret_store.get(credential_ref)
            except ConnectorSecretError as error:
                raise EmailCredentialError(str(error)) from error
        prefix = "env:"
        if not credential_ref.startswith(prefix):
            raise EmailCredentialError("Unsupported credential reference")
        name = credential_ref[len(prefix):]
        if not name or not name.replace("_", "").isalnum():
            raise EmailCredentialError("Invalid credential environment variable name")
        secret = self._environ.get(name, "")
        if not secret:
            raise EmailCredentialError(f"Credential environment variable is unavailable: {name}")
        return secret
