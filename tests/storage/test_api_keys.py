"""The credential store, and the properties the API's safety rests on."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3

import pytest

from career_agent.storage.api_keys import (
    CAPTURE_WRITE,
    DEFAULT_EXPIRY_DAYS,
    KEY_PREFIX,
    CHAT_WRITE,
    SQLiteApiKeyStore,
    WORKSPACE_READ,
    hash_secret,
)


def test_a_key_verifies_to_its_owner_and_scopes(tmp_path: Path) -> None:
    store = SQLiteApiKeyStore(tmp_path / "api_keys.sqlite3")

    issued = store.issue(
        user_id="u1", name="desktop", scopes=frozenset({WORKSPACE_READ, CHAT_WRITE})
    )
    principal = store.verify(issued.secret)

    assert principal is not None
    assert principal.user_id == "u1"
    assert principal.allows(WORKSPACE_READ)
    assert principal.allows(CHAT_WRITE)
    # The scope the extension would hold is absent, which is the whole point of
    # scoping: one key is not implicitly every key.
    assert not principal.allows(CAPTURE_WRITE)


def test_the_secret_is_never_stored(tmp_path: Path) -> None:
    """A leaked database must not be a set of live credentials.

    SHA-256 rather than a password KDF because these secrets are 32 random
    bytes: there is no dictionary to run, so a slow hash would cost startup time
    and buy nothing. What the digest buys is that the file is not the key.
    """
    path = tmp_path / "api_keys.sqlite3"
    store = SQLiteApiKeyStore(path)

    issued = store.issue(user_id="u1", name="desktop", scopes=frozenset({CHAT_WRITE}))

    contents = path.read_bytes()
    assert issued.secret.encode() not in contents
    assert hash_secret(issued.secret).encode() in contents


def test_a_revoked_key_stops_working_and_says_nothing_about_why(tmp_path: Path) -> None:
    store = SQLiteApiKeyStore(tmp_path / "api_keys.sqlite3")
    issued = store.issue(user_id="u1", name="desktop", scopes=frozenset({CHAT_WRITE}))

    assert store.revoke(key_id=issued.key_id) is True

    # Indistinguishable from a secret that never existed: a caller learns only
    # that the credential does not work, which is all a caller is entitled to.
    assert store.verify(issued.secret) is None
    assert store.verify("never-issued") is None
    # Revoking twice is not an error the second time, it is simply no longer
    # true that anything changed.
    assert store.revoke(key_id=issued.key_id) is False


def test_two_users_keys_never_resolve_to_each_other(tmp_path: Path) -> None:
    store = SQLiteApiKeyStore(tmp_path / "api_keys.sqlite3")

    mine = store.issue(user_id="u1", name="mine", scopes=frozenset({WORKSPACE_READ}))
    theirs = store.issue(user_id="u2", name="theirs", scopes=frozenset({WORKSPACE_READ}))

    assert store.verify(mine.secret).user_id == "u1"
    assert store.verify(theirs.secret).user_id == "u2"
    assert mine.secret != theirs.secret


@pytest.mark.parametrize(
    ("user_id", "name", "scopes"),
    (
        ("", "desktop", frozenset({CHAT_WRITE})),
        ("u1", "", frozenset({CHAT_WRITE})),
        ("u1", "desktop", frozenset()),
        ("u1", "desktop", frozenset({"workspace:write"})),
    ),
)
def test_an_unusable_key_is_refused_at_issue_time(
    tmp_path: Path, user_id, name, scopes
) -> None:
    """Refused where a human can read the error, not where a request 403s.

    A key with an unknown scope grants nothing, and a key with no name cannot be
    revoked by whoever finds it in a config file. Both failures would otherwise
    surface much later, as an authorization error against the wrong endpoint.
    """
    store = SQLiteApiKeyStore(tmp_path / "api_keys.sqlite3")

    with pytest.raises(ValueError):
        store.issue(user_id=user_id, name=name, scopes=scopes)


def test_use_is_recorded_so_a_stale_key_can_be_found(tmp_path: Path) -> None:
    store = SQLiteApiKeyStore(tmp_path / "api_keys.sqlite3")
    issued = store.issue(user_id="u1", name="desktop", scopes=frozenset({CHAT_WRITE}))

    assert store.list_keys()[0].last_used_at is None
    store.verify(issued.secret)

    assert store.list_keys()[0].last_used_at is not None


def test_a_key_is_recognisable_so_a_leak_can_be_found(tmp_path: Path) -> None:
    """The prefix buys nothing cryptographically and everything operationally.

    A secret scanner matches shapes. A bare base64 blob in a commit, a log line
    or a pasted snippet looks like any other string, so the leak is found by
    whoever exploits it. ``sk_live_``, ``ghp_`` and this ``cak_`` exist for that
    reason and no other.
    """
    store = SQLiteApiKeyStore(tmp_path / "api_keys.sqlite3")

    issued = store.issue(user_id="u1", name="desktop", scopes=frozenset({CHAT_WRITE}))

    assert issued.secret.startswith(KEY_PREFIX)
    # Still the full random secret, not a prefix standing in for entropy.
    assert len(issued.secret) - len(KEY_PREFIX) >= 40


def test_a_key_expires_by_default_and_permanence_is_deliberate(tmp_path: Path) -> None:
    """A key with no end date is a key nobody rotates.

    So the default carries one, and a permanent key has to be asked for. The
    expiry is enforced at verification rather than by a sweep: a key that has
    outlived its date must stop working even if nothing has run to clean it up.
    """
    store = SQLiteApiKeyStore(tmp_path / "api_keys.sqlite3")

    default = store.issue(user_id="u1", name="desktop", scopes=frozenset({CHAT_WRITE}))
    permanent = store.issue(
        user_id="u1",
        name="server",
        scopes=frozenset({CHAT_WRITE}),
        expires_in_days=None,
    )

    assert default.expires_at is not None
    assert (default.expires_at - default.created_at).days == DEFAULT_EXPIRY_DAYS
    assert permanent.expires_at is None


def test_an_expired_key_stops_working_without_being_revoked(tmp_path: Path) -> None:
    path = tmp_path / "api_keys.sqlite3"
    store = SQLiteApiKeyStore(path)
    issued = store.issue(
        user_id="u1", name="desktop", scopes=frozenset({CHAT_WRITE}), expires_in_days=1
    )
    assert store.verify(issued.secret) is not None

    # Move the clock rather than the code: what matters is that verification
    # consults the date, not that some background job noticed.
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE api_keys SET expires_at = ? WHERE key_id = ?",
            (
                (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
                issued.key_id,
            ),
        )

    assert store.verify(issued.secret) is None
    # Still listed, and still distinguishable from a revoked key by an operator.
    record = store.list_keys()[0]
    assert record.revoked_at is None and record.expires_at is not None
