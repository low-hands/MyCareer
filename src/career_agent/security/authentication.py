"""Turning a presented credential into the identity the API acts as.

The point of this module is what it does *not* offer: there is no way to ask it
"is this user_id allowed", because no endpoint receives a user_id any more. The
caller presents a credential; the identity comes back from the store. A request
cannot name a user it does not hold a key for, which is a stronger property than
checking that a stated user matches a token — there is nothing to state.

Scope checks live here too, for the same reason the projection boundary refuses
internal identifiers rather than trusting callers not to send them: the browser
extension's key can only capture, so a stolen extension key cannot read a
resume, whatever request it is put into.
"""

from __future__ import annotations

from fastapi import Depends, Header, HTTPException, Request

from career_agent.storage.api_keys import ApiKeyPrincipal, ApiKeyStore


def _presented_secret(authorization: str | None) -> str | None:
    """The bearer token, or nothing.

    Only the ``Bearer`` scheme is accepted. Reading the header loosely — say,
    falling back to the raw value — would let a malformed client authenticate by
    accident, which makes the failure mode of a broken client a security event
    rather than an error message.
    """
    if not authorization:
        return None
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return None
    return value.strip()


async def authenticate(
    request: Request,
    authorization: str | None = Header(default=None),
) -> ApiKeyPrincipal:
    store: ApiKeyStore | None = getattr(request.app.state, "api_key_store", None)
    if store is None:
        # A server that cannot check credentials must refuse, not wave requests
        # through. Failing closed turns a misconfiguration into an outage
        # instead of into an open database.
        raise HTTPException(
            status_code=503, detail="API credentials are not configured"
        )
    secret = _presented_secret(authorization)
    principal = store.verify(secret) if secret else None
    if principal is None:
        raise HTTPException(
            status_code=401,
            detail="A valid API key is required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return principal


def require_scope(scope: str):
    """A dependency that admits only keys carrying ``scope``.

    Returns 403 rather than 401 for a valid key without the scope: the caller is
    authenticated and simply not permitted, and telling the two apart is what
    lets an operator debug a mis-scoped key without guessing.
    """

    async def dependency(
        principal: ApiKeyPrincipal = Depends(authenticate),
    ) -> ApiKeyPrincipal:
        if not principal.allows(scope):
            raise HTTPException(
                status_code=403,
                detail=f"This API key does not carry the '{scope}' scope",
            )
        return principal

    return dependency
