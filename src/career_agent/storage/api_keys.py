"""Credentials the API authenticates callers with.

Every endpoint used to take ``user_id`` from the caller — a query parameter on
the ten read routes, a body field on chat and browser capture. Anyone who could
reach the port could name any user and read or write that person's resumes,
applications, interviews and calendar. This store is the other half of removing
that field: identity is derived from a credential, never stated by the client.

Keys are stored as SHA-256 digests. A password KDF would be the right choice for
a human-chosen secret; these are 32 bytes from ``secrets.token_urlsafe``, so
there is no dictionary to run and the cost of a slow hash would buy nothing.
What the digest does buy is that a leaked database is not a set of live
credentials.

Keys carry a ``cak_`` prefix and an expiry. The prefix is what makes a leaked
key *findable*: secret scanners match on recognisable shapes, and a bare
base64 blob in a commit or a support ticket looks like any other string. The
expiry is the other half of rotation — a key with no end date is a key nobody
ever rotates — so a caller has to ask for a non-expiring one on purpose.

Scopes exist because of where one of these keys has to live. A browser extension
cannot hold a secret safely — any script running in that context can read it —
so the extension's key grants ``capture:write`` and nothing else. The same
mechanism issues it as every other key; only its scope differs, which is how a
credential in the weakest storage stops being a credential to everything.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import sqlite3
from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from career_agent.storage.schema import apply_schema


KEY_PREFIX = "cak_"
"""Marks these strings as this project's credentials, for secret scanners.

Industry keys are written ``sk_live_...``, ``ghp_...`` and so on for exactly
this reason: the prefix is not a secret and buys nothing cryptographically, but
it is what lets a scanner flag the key in a commit, a log or a pasted snippet.
"""

DEFAULT_EXPIRY_DAYS = 90

WORKSPACE_READ = "workspace:read"
CHAT_WRITE = "chat:write"
CAPTURE_WRITE = "capture:write"
SETTINGS_WRITE = "settings:write"
"""Changing the owner's rules, which no other key may do.

Separate from ``chat:write`` on purpose. The rules exist to constrain what the
agent may do in a conversation, so the credential a conversation runs under must
not be the credential that can relax them — otherwise anything that can talk to
the agent is one persuasive message away from the settings that bound it.
"""

KNOWN_SCOPES = frozenset({WORKSPACE_READ, CHAT_WRITE, CAPTURE_WRITE, SETTINGS_WRITE})
"""Every scope the API knows how to require.

Closed rather than free-form: a key issued with a typo'd scope would otherwise
be a key that silently grants nothing, and the failure would appear as an
authorization error against the wrong endpoint.
"""


class ApiKeyPrincipal(BaseModel):
    """Who a verified credential says the caller is."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key_id: str
    user_id: str
    scopes: frozenset[str]

    def allows(self, scope: str) -> bool:
        return scope in self.scopes


class IssuedApiKey(BaseModel):
    """A newly minted key, the one time its secret is knowable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key_id: str
    user_id: str
    name: str
    scopes: frozenset[str]
    secret: str
    created_at: datetime
    expires_at: datetime | None


class ApiKeyRecord(BaseModel):
    """What a listing shows: everything except the secret, which is not kept."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key_id: str
    user_id: str
    name: str
    scopes: frozenset[str]
    created_at: datetime
    expires_at: datetime | None = None
    last_used_at: datetime | None = None
    revoked_at: datetime | None = None


class ApiKeyStore(Protocol):
    def verify(self, secret: str) -> ApiKeyPrincipal | None: ...

    def issue(
        self,
        *,
        user_id: str,
        name: str,
        scopes: frozenset[str],
        expires_in_days: int | None = DEFAULT_EXPIRY_DAYS,
    ) -> IssuedApiKey: ...

    def revoke(self, *, key_id: str) -> bool: ...

    def list_keys(self, *, user_id: str | None = None) -> tuple[ApiKeyRecord, ...]: ...


def hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


class SQLiteApiKeyStore:
    """API keys in the same SQLite file family as the rest of the workspace."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            apply_schema(connection, "api_keys", 1, self._migrate)
        os.chmod(self.path, 0o600)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30.0)

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS api_keys (
                key_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                secret_hash TEXT NOT NULL UNIQUE,
                scopes_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                last_used_at TEXT,
                revoked_at TEXT
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS api_keys_user_idx ON api_keys(user_id)"
        )

    def issue(
        self,
        *,
        user_id: str,
        name: str,
        scopes: frozenset[str],
        expires_in_days: int | None = DEFAULT_EXPIRY_DAYS,
    ) -> IssuedApiKey:
        if not user_id.strip():
            raise ValueError("an API key must belong to a user")
        if not name.strip():
            raise ValueError("an API key needs a name to be revocable by a human")
        unknown = sorted(scopes - KNOWN_SCOPES)
        if unknown:
            raise ValueError(f"unknown API key scopes: {unknown}")
        if not scopes:
            raise ValueError("an API key with no scopes can do nothing; grant one")
        if expires_in_days is not None and expires_in_days < 1:
            raise ValueError("an API key must be valid for at least one day")
        now = datetime.now(timezone.utc)
        secret = KEY_PREFIX + secrets.token_urlsafe(32)
        record = IssuedApiKey(
            key_id=uuid4().hex,
            user_id=user_id,
            name=name,
            scopes=scopes,
            secret=secret,
            created_at=now,
            expires_at=(
                now + timedelta(days=expires_in_days)
                if expires_in_days is not None
                else None
            ),
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO api_keys(
                    key_id, user_id, name, secret_hash, scopes_json,
                    created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.key_id,
                    record.user_id,
                    record.name,
                    hash_secret(secret),
                    json.dumps(sorted(scopes)),
                    record.created_at.isoformat(),
                    record.expires_at.isoformat() if record.expires_at else None,
                ),
            )
        return record

    def verify(self, secret: str) -> ApiKeyPrincipal | None:
        """Look a presented secret up by digest, or say nothing about why not.

        A revoked, expired or unknown key are all indistinguishable here on
        purpose: the caller learns only that the credential does not work, which
        is all a caller is entitled to learn.

        No constant-time comparison, and deliberately so. A timing side channel
        on this lookup would have to be exploitable byte by byte, which requires
        the attacker to control what is compared — and what is compared is the
        SHA-256 of their guess. Hashing destroys the incremental relationship
        that makes such an attack work, so ``compare_digest`` here would be
        ceremony. It would be required if the raw secret were ever compared.
        """
        if not secret:
            return None
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT key_id, user_id, scopes_json FROM api_keys
                WHERE secret_hash = ?
                  AND revoked_at IS NULL
                  AND (expires_at IS NULL OR expires_at > ?)
                """,
                (hash_secret(secret), datetime.now(timezone.utc).isoformat()),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "UPDATE api_keys SET last_used_at = ? WHERE key_id = ?",
                (datetime.now(timezone.utc).isoformat(), row[0]),
            )
        return ApiKeyPrincipal(
            key_id=row[0],
            user_id=row[1],
            scopes=frozenset(json.loads(row[2])),
        )

    def revoke(self, *, key_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE api_keys SET revoked_at = ? "
                "WHERE key_id = ? AND revoked_at IS NULL",
                (datetime.now(timezone.utc).isoformat(), key_id),
            )
            return cursor.rowcount > 0

    def list_keys(self, *, user_id: str | None = None) -> tuple[ApiKeyRecord, ...]:
        query = (
            "SELECT key_id, user_id, name, scopes_json, created_at, "
            "expires_at, last_used_at, revoked_at FROM api_keys"
        )
        parameters: tuple[str, ...] = ()
        if user_id is not None:
            query += " WHERE user_id = ?"
            parameters = (user_id,)
        query += " ORDER BY created_at"
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(
            ApiKeyRecord(
                key_id=row[0],
                user_id=row[1],
                name=row[2],
                scopes=frozenset(json.loads(row[3])),
                created_at=datetime.fromisoformat(row[4]),
                expires_at=(datetime.fromisoformat(row[5]) if row[5] else None),
                last_used_at=(
                    datetime.fromisoformat(row[6]) if row[6] else None
                ),
                revoked_at=(datetime.fromisoformat(row[7]) if row[7] else None),
            )
            for row in rows
        )
