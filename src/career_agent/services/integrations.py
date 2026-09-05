from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
import secrets
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from career_agent.connectors.qq_email_readonly import QQEmailReadOnlyConnector
from career_agent.storage.calendar import SQLiteCalendarStore
from career_agent.storage.connector_secrets import ConnectorSecretStore
from career_agent.storage.email_tracking import SQLiteEmailTrackingStore
from career_agent.storage.oauth_flows import OAuthFlow, SQLiteOAuthFlowStore


GOOGLE_SCOPES = {
    "gmail": (
        "openid",
        "email",
        "https://www.googleapis.com/auth/gmail.readonly",
    ),
    "calendar": (
        "openid",
        "email",
        "https://www.googleapis.com/auth/calendar.events",
    ),
}


class IntegrationConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ConnectedGoogleAccount:
    kind: str
    account_id: str
    email_address: str


class IntegrationConnectionService:
    def __init__(
        self,
        *,
        email_store: SQLiteEmailTrackingStore,
        calendar_store: SQLiteCalendarStore,
        flow_store: SQLiteOAuthFlowStore,
        secret_store: ConnectorSecretStore,
        client_id: str,
        client_secret: str,
        callback_url: str,
        frontend_url: str,
        opener: Callable[..., Any] = urlopen,
        qq_connector_factory: Callable[[str, str], Any] = QQEmailReadOnlyConnector,
    ) -> None:
        self._email_store = email_store
        self._calendar_store = calendar_store
        self._flows = flow_store
        self._secrets = secret_store
        self._client_id = client_id
        self._client_secret = client_secret
        self.callback_url = callback_url
        self.frontend_url = frontend_url.rstrip("/")
        self._opener = opener
        self._qq_connector_factory = qq_connector_factory

    @classmethod
    def from_env(
        cls,
        *,
        email_store: SQLiteEmailTrackingStore,
        calendar_store: SQLiteCalendarStore,
        flow_store: SQLiteOAuthFlowStore,
        secret_store: ConnectorSecretStore,
    ) -> "IntegrationConnectionService":
        load_dotenv()
        return cls(
            email_store=email_store,
            calendar_store=calendar_store,
            flow_store=flow_store,
            secret_store=secret_store,
            client_id=os.environ.get("GOOGLE_OAUTH_CLIENT_ID", ""),
            client_secret=os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET", ""),
            callback_url=os.environ.get(
                "GOOGLE_OAUTH_CALLBACK_URL",
                "http://127.0.0.1:8000/v1/connections/google/callback",
            ),
            frontend_url=os.environ.get(
                "CAREER_AGENT_WEB_URL", "http://127.0.0.1:5173"
            ),
        )

    def google_authorization_url(self, *, user_id: str, kind: str) -> str:
        self._require_google_config()
        if kind not in GOOGLE_SCOPES:
            raise ValueError("Unknown Google connection kind")
        state = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(64)
        challenge = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode()).digest()
        ).rstrip(b"=").decode()
        self._flows.create(
            OAuthFlow(
                state=state,
                user_id=user_id,
                code_verifier=verifier,
                expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
                connection_kind=kind,
            )
        )
        return "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(
            {
                "client_id": self._client_id,
                "redirect_uri": self.callback_url,
                "response_type": "code",
                "scope": " ".join(GOOGLE_SCOPES[kind]),
                "access_type": "offline",
                "prompt": "consent",
                "state": state,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            }
        )

    def complete_google(self, *, state: str, code: str) -> ConnectedGoogleAccount:
        self._require_google_config()
        flow = self._flows.consume(state)
        if flow is None:
            raise ValueError("OAuth state is invalid, expired, or already used")
        token = self._post_form(
            "https://oauth2.googleapis.com/token",
            {
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "code": code,
                "code_verifier": flow.code_verifier,
                "grant_type": "authorization_code",
                "redirect_uri": self.callback_url,
            },
        )
        access_token = token.get("access_token")
        refresh_token = token.get("refresh_token")
        if not isinstance(access_token, str) or not isinstance(refresh_token, str):
            raise ValueError("Google did not return the required offline credentials")
        profile = self._get_json(
            "https://openidconnect.googleapis.com/v1/userinfo",
            access_token=access_token,
        )
        email_address = profile.get("email")
        if not isinstance(email_address, str) or not email_address:
            raise ValueError("Google account email was not returned")
        if profile.get("email_verified") is not True:
            raise ValueError("Google account email is not verified")
        reference = self._secrets.put(
            json.dumps(
                {
                    "refresh_token": refresh_token,
                }
            )
        )
        try:
            if flow.connection_kind == "gmail":
                account = self._email_store.add_account(
                    user_id=flow.user_id,
                    provider="gmail",
                    email_address=email_address,
                    credential_ref=reference,
                )
            elif flow.connection_kind == "calendar":
                account = self._calendar_store.add_account(
                    user_id=flow.user_id,
                    email_address=email_address,
                    calendar_id="primary",
                    credential_ref=reference,
                )
            else:
                raise ValueError("OAuth flow has an invalid connection kind")
        except Exception:
            self._secrets.delete(reference)
            raise
        return ConnectedGoogleAccount(flow.connection_kind, account.id, email_address)

    def google_connection_kind(self, *, state: str, consume: bool = False) -> str | None:
        """Recover callback routing from opaque state, consuming cancelled flows."""

        flow = self._flows.consume(state) if consume else self._flows.peek(state)
        if flow is None or flow.connection_kind not in GOOGLE_SCOPES:
            return None
        return flow.connection_kind

    def connect_qq(
        self, *, user_id: str, email_address: str, authorization_code: str
    ) -> str:
        connector = self._qq_connector_factory(email_address, authorization_code)
        connector.test_connection()
        reference = self._secrets.put(authorization_code)
        try:
            account = self._email_store.add_account(
                user_id=user_id,
                provider="qq",
                email_address=email_address,
                credential_ref=reference,
            )
        except Exception:
            self._secrets.delete(reference)
            raise
        return account.id

    def disconnect(self, *, user_id: str, kind: str, account_id: str) -> bool:
        if kind == "email":
            account = self._email_store.get_account(
                user_id=user_id, account_id=account_id
            )
            if account is None:
                return False
            self._email_store.disable_account(user_id=user_id, account_id=account_id)
            reference = account.credential_ref
        elif kind == "calendar":
            account = self._calendar_store.get_account(
                user_id=user_id, calendar_account_id=account_id
            )
            if account is None:
                return False
            self._calendar_store.disable_account(
                user_id=user_id, calendar_account_id=account_id
            )
            reference = account.credential_ref
        else:
            raise ValueError("Unknown integration kind")
        if (
            reference.startswith("keyring:")
            and self._email_store.active_credential_ref_count(reference) == 0
            and self._calendar_store.active_credential_ref_count(reference) == 0
        ):
            self._secrets.delete(reference)
        return True

    def _require_google_config(self) -> None:
        if not self._client_id or not self._client_secret:
            raise IntegrationConfigurationError(
                "尚未配置 Google OAuth。请在项目根目录 .env 添加 Client ID 和 Client Secret，然后重启后端。"
            )

    def _post_form(self, url: str, values: dict[str, str]) -> dict[str, Any]:
        request = Request(
            url,
            data=urlencode(values).encode(),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with self._opener(request, timeout=30.0) as response:
            return json.loads(response.read())

    def _get_json(self, url: str, *, access_token: str) -> dict[str, Any]:
        request = Request(
            url,
            headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
        )
        with self._opener(request, timeout=30.0) as response:
            return json.loads(response.read())
