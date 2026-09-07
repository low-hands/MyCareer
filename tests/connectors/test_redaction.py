from urllib.parse import quote

import pytest

from career_agent.security.redaction import REDACTED, redact, redact_text


def test_redacts_sensitive_keys_and_values() -> None:
    value = {
        "cookie": "secret-cookie",
        "nested": {"authorization": "Bearer abc", "message": "wt2=secret"},
        "title": "Backend",
    }

    redacted = redact(value)

    assert redacted["cookie"] == "<redacted>"
    assert redacted["nested"]["authorization"] == "<redacted>"
    assert redacted["nested"]["message"] == "<redacted>"
    assert redacted["title"] == "Backend"


def test_cache_token_counts_are_not_mistaken_for_credentials() -> None:
    redacted = redact(
        {
            "cache_read_input_tokens": 700,
            "cache_creation_input_tokens": 100,
            "uncached_input_tokens": 200,
            "access_token": "secret",
        }
    )

    assert redacted["cache_read_input_tokens"] == 700
    assert redacted["cache_creation_input_tokens"] == 100
    assert redacted["uncached_input_tokens"] == 200
    assert redacted["access_token"] == REDACTED


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # A meeting link is the payload of an interview invitation, so the
        # address has to survive for the classifier to extract it.
        (
            "面试链接 https://meet.google.com/abc-defg-hij 周三 14:00",
            "面试链接 https://meet.google.com/abc-defg-hij 周三 14:00",
        ),
        # Query and fragment credentials go; the addressing parameters stay.
        (
            "https://mail.example.com/verify?token=eyJhbGciOi.xyz&uid=42",
            "https://mail.example.com/verify?token=<redacted>&uid=42",
        ),
        (
            "https://portal.example.com/#access_token=abc123",
            "https://portal.example.com/#access_token=<redacted>",
        ),
        # An opaque fragment carries no key to judge, so it cannot be kept.
        ("https://x.example.com/r#opaqueblob", "https://x.example.com/r#<redacted>"),
        ("https://cdn.example.com/f.pdf?sig=deadbeef", "https://cdn.example.com/f.pdf?sig=<redacted>"),
        # Numbers the classifier needs must not be swept up as codes.
        ("薪资 25000-35000，团队 30 人", "薪资 25000-35000，团队 30 人"),
        ("您的验证码是 483920，5分钟内有效", "您的验证码是 <redacted>，5分钟内有效"),
        ("Your verification code is A9F2K1.", "Your verification code is <redacted>."),
        ("Authorization: Bearer sk-live-abc", "Authorization: <redacted>"),
        # Credentials in the authority, where no parameter rule would see them.
        (
            "https://user:secret@example.com/interview",
            "https://example.com/interview",
        ),
        # Spelling variants an exact allowlist has to enumerate one by one.
        ("https://a.example.com/x?invite_token=S", "https://a.example.com/x?invite_token=<redacted>"),
        ("https://a.example.com/x?X-Amz-Signature=S", "https://a.example.com/x?X-Amz-Signature=<redacted>"),
        ("https://a.example.com/x?securityId=S", "https://a.example.com/x?securityId=<redacted>"),
        ("https://a.example.com/x?jwt=S", "https://a.example.com/x?jwt=<redacted>"),
        ("https://a.example.com/x?accessToken=S", "https://a.example.com/x?accessToken=<redacted>"),
        # Short or common names must NOT be swept up by fragment matching: these
        # are addressing parameters a JD search depends on.
        (
            "https://www.zhipin.com/j?city=101010100&keyword=AI&page=2&signup=1",
            "https://www.zhipin.com/j?city=101010100&keyword=AI&page=2&signup=1",
        ),
    ],
)
def test_redact_text_strips_credentials_and_keeps_signal(raw: str, expected: str) -> None:
    assert redact_text(raw) == expected


def test_trailing_punctuation_stays_outside_the_url() -> None:
    assert redact_text("见 https://a.example.com/p?token=x。") == (
        "见 https://a.example.com/p?token=<redacted>。"
    )


def test_a_token_nested_inside_a_redirect_parameter_is_scrubbed() -> None:
    """A `?next=` value is address-shaped but can carry its own query string."""
    scrubbed = redact_text(
        "https://a.example.com/r?next=https%3A%2F%2Fb.example.com%2Fp%3Ftoken%3DSECRET"
    )
    assert "SECRET" not in scrubbed
    assert "b.example.com" in scrubbed


def test_a_redirect_to_a_plain_address_is_left_intact() -> None:
    raw = "https://a.example.com/r?next=https%3A%2F%2Fb.example.com%2Fjobs"
    assert redact_text(raw) == raw


@pytest.mark.parametrize(
    "raw",
    [
        "https://a.example/r?next=https%253A%252F%252Fb.example%252Fp%253Finvite_token%253DSECRET",
        "https://a.example/r?next=%20https%3A%2F%2Fb.example%2Fp%3Ftoken%3DSECRET",
    ],
)
def test_encoded_or_indented_nested_urls_cannot_hide_a_token(raw: str) -> None:
    scrubbed = redact_text(raw)
    assert "SECRET" not in scrubbed
    assert "b.example" in scrubbed


def test_a_nested_url_beyond_the_inspection_budget_fails_closed() -> None:
    """A chain deeper than the budget is blanked, not waved through.

    Each nesting level costs another percent-encoding layer, so an attacker can
    always out-nest a fixed budget. The only safe stop is to drop the remainder.
    """
    inner = "https://z.example/?token=SECRET"
    for index in range(8):
        inner = f"https://h{index}.example/?next={quote(inner, safe='')}"

    scrubbed = redact_text(inner)
    assert "SECRET" not in scrubbed
    # The outer address the reader needs survives; the unreadable tail does not.
    assert scrubbed.startswith("https://h7.example/")
    # The marker sits under however many encoding layers the chain wrapped it in;
    # what matters is that the tail was replaced rather than passed through.
    assert "redacted" in scrubbed.replace("%25", "%")
    assert "h0.example" not in scrubbed


@pytest.mark.parametrize(
    "raw",
    [
        # OAuth `state` carries a CSRF value; a US state abbreviation shares the name.
        "https://a.example/j?state=CA",
        # Interview "ticket" numbers and MAC/salt-shaped names collide with credentials.
        "https://a.example/j?ticket=INTERVIEW-123",
        "https://a.example/j?salt=coarse",
    ],
)
def test_ambiguous_names_are_blanked_even_when_the_value_is_harmless(raw: str) -> None:
    """Deliberate over-redaction: these names are credentials more often than not.

    Nothing in a query string says which meaning was intended, and the cost is
    asymmetric — a blanked city filter degrades a search, a leaked CSRF token or
    password salt does not degrade, it discloses. Recorded as a test so the
    behaviour reads as chosen rather than as an oversight; loosen only with a
    concrete parameter that matters and a narrower rule than the whole name.
    """
    assert REDACTED in redact_text(raw)
