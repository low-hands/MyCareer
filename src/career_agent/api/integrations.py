from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from career_agent.security.authentication import require_scope
from career_agent.services.integrations import (
    IntegrationConfigurationError,
    IntegrationConnectionService,
)
from career_agent.storage.api_keys import ApiKeyPrincipal, WORKSPACE_WRITE


logger = logging.getLogger(__name__)


class GoogleAuthorizationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    authorization_url: str


class QQConnectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email_address: str = Field(min_length=3, max_length=320)
    authorization_code: str = Field(min_length=1, max_length=500)


class ConnectionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    connected: Literal[True] = True
    account_id: str


class DisconnectionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    disconnected: Literal[True] = True


def build_integration_router(
    service_factory: Callable[[], IntegrationConnectionService],
) -> APIRouter:
    router = APIRouter(prefix="/v1/connections")
    cached: dict[str, IntegrationConnectionService] = {}

    def service() -> IntegrationConnectionService:
        if "service" not in cached:
            cached["service"] = service_factory()
        return cached["service"]

    @router.post("/google/{kind}/start", response_model=GoogleAuthorizationResponse)
    async def start_google(
        kind: Literal["gmail", "calendar"],
        principal: ApiKeyPrincipal = Depends(require_scope(WORKSPACE_WRITE)),
    ) -> GoogleAuthorizationResponse:
        try:
            url = service().google_authorization_url(
                user_id=principal.user_id, kind=kind
            )
        except IntegrationConfigurationError as error:
            raise HTTPException(
                status_code=503,
                detail={"code": "GOOGLE_OAUTH_NOT_CONFIGURED", "message": str(error)},
            ) from error
        return GoogleAuthorizationResponse(authorization_url=url)

    @router.get("/google/callback", include_in_schema=False)
    async def complete_google(
        state: str = Query(min_length=20, max_length=500),
        code: str | None = Query(default=None, min_length=1, max_length=4000),
        error: str | None = Query(default=None, max_length=500),
    ) -> RedirectResponse:
        integration_service = service()
        target = integration_service.frontend_url
        kind = integration_service.google_connection_kind(
            state=state,
            consume=bool(error or not code),
        ) or "google"
        if error or not code:
            logger.warning("Google OAuth was denied or returned no code: %s", error)
            return RedirectResponse(
                f"{target}/?{urlencode({'status': 'denied', 'kind': kind})}"
            )
        try:
            connected = integration_service.complete_google(state=state, code=code)
        except IntegrationConfigurationError:
            logger.exception("Google OAuth callback failed because configuration is missing")
            return RedirectResponse(
                f"{target}/?{urlencode({'status': 'configuration_error', 'kind': kind})}"
            )
        except ValueError as callback_error:
            logger.warning("Google OAuth callback was rejected: %s", callback_error)
            return RedirectResponse(
                f"{target}/?{urlencode({'status': 'invalid', 'kind': kind})}"
            )
        except (HTTPError, URLError, TimeoutError, OSError):
            logger.exception("Google OAuth provider request failed")
            return RedirectResponse(
                f"{target}/?{urlencode({'status': 'network_error', 'kind': kind})}"
            )
        except Exception:
            logger.exception("Unexpected Google OAuth callback failure")
            return RedirectResponse(
                f"{target}/?{urlencode({'status': 'internal_error', 'kind': kind})}"
            )
        return RedirectResponse(
            f"{target}/?{urlencode({'status': 'connected', 'kind': connected.kind})}"
        )

    @router.post("/qq", response_model=ConnectionResponse)
    async def connect_qq(
        request: QQConnectionRequest,
        principal: ApiKeyPrincipal = Depends(require_scope(WORKSPACE_WRITE)),
    ) -> ConnectionResponse:
        try:
            account_id = service().connect_qq(
                user_id=principal.user_id,
                email_address=request.email_address,
                authorization_code=request.authorization_code,
            )
        except Exception as error:
            raise HTTPException(
                status_code=400,
                detail={"code": "QQ_CONNECTION_FAILED", "message": "QQ 邮箱连接失败。"},
            ) from error
        return ConnectionResponse(account_id=account_id)

    @router.delete("/{kind}/{account_id}", response_model=DisconnectionResponse)
    async def disconnect(
        kind: Literal["email", "calendar"],
        account_id: str,
        principal: ApiKeyPrincipal = Depends(require_scope(WORKSPACE_WRITE)),
    ) -> DisconnectionResponse:
        if not service().disconnect(
            user_id=principal.user_id, kind=kind, account_id=account_id
        ):
            raise HTTPException(status_code=404, detail="Integration account not found")
        return DisconnectionResponse()

    return router
