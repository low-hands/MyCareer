from career_agent.security.redaction import redact


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
