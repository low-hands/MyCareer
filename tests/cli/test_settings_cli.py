from io import StringIO
import json

from career_agent.cli import main


def _run(args):
    output = StringIO()
    code = main(args, stdout=output, stderr=StringIO())
    return code, json.loads(output.getvalue())


def test_cli_settings_distinguish_soft_preferences_from_behavior_policy(tmp_path):
    path = tmp_path / "context.sqlite3"
    code, changed = _run(
        [
            "settings", "set", "--user-id", "u1",
            "--boss-search", "allowed",
            "--application-confirmation", "always_ask",
            "--context-store", str(path),
        ]
    )
    _, history = _run(
        ["settings", "history", "--user-id", "u1", "--context-store", str(path)]
    )

    assert code == 0
    settings = changed["owner_settings"]
    assert settings["preferences"]["boss_search"] == "allowed"
    assert settings["behavior_policy"]["application_confirmation"] == "always_ask"
    assert history["events"][0]["changed_fields"] == [
        "preferences", "behavior_policy"
    ]



def test_cli_confirm_before_replaces_the_list_and_refuses_non_write_names(tmp_path):
    path = tmp_path / "context.sqlite3"
    code, changed = _run(
        [
            "settings", "set", "--user-id", "u1",
            "--confirm-before", "update_application_status, create_application,create_application",
            "--context-store", str(path),
        ]
    )
    assert code == 0
    assert changed["owner_settings"]["behavior_policy"]["confirm_before"] == [
        "create_application", "update_application_status",
    ]

    errors = StringIO()
    rejected = main(
        [
            "settings", "set", "--user-id", "u1",
            "--confirm-before", "search_career_history",
            "--context-store", str(path),
        ],
        stdout=StringIO(),
        stderr=errors,
    )
    assert rejected != 0
    assert "search_career_history" in errors.getvalue()

    code, kept = _run(
        [
            "settings", "set", "--user-id", "u1",
            "--boss-search", "allowed",
            "--context-store", str(path),
        ]
    )
    assert kept["owner_settings"]["behavior_policy"]["confirm_before"] == [
        "create_application", "update_application_status",
    ]

    code, cleared = _run(
        [
            "settings", "set", "--user-id", "u1",
            "--confirm-before", "",
            "--context-store", str(path),
        ]
    )
    assert code == 0
    assert cleared["owner_settings"]["behavior_policy"]["confirm_before"] == []


def test_cli_still_loads_and_clears_a_confirm_before_rule_for_a_renamed_capability(
    tmp_path,
):
    """A stored rule is judged where it was written, not on every read.

    The name below was a valid write when the owner saved it; it has since been
    renamed. Reading the document must not raise — every turn reads it — and
    ``settings show`` / ``settings set`` must still work so the owner can drop
    the stale rule without editing the database by hand.
    """
    import sqlite3

    path = tmp_path / "context.sqlite3"
    code, _ = _run(
        [
            "settings", "set", "--user-id", "u1",
            "--confirm-before", "create_interview",
            "--context-store", str(path),
        ]
    )
    assert code == 0
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE owner_settings_context SET payload = "
            "replace(payload, 'create_interview', 'record_interview_legacy') "
            "WHERE user_id = 'u1'"
        )

    code, shown = _run(
        ["settings", "show", "--user-id", "u1", "--context-store", str(path)]
    )
    assert code == 0
    assert shown["owner_settings"]["behavior_policy"]["confirm_before"] == [
        "record_interview_legacy"
    ]

    # Unrelated edits keep the stale entry rather than failing on it.
    code, kept = _run(
        [
            "settings", "set", "--user-id", "u1",
            "--boss-search", "allowed",
            "--context-store", str(path),
        ]
    )
    assert code == 0
    assert kept["owner_settings"]["behavior_policy"]["confirm_before"] == [
        "record_interview_legacy"
    ]

    # New lists are still checked against today's registry.
    errors = StringIO()
    rejected = main(
        [
            "settings", "set", "--user-id", "u1",
            "--confirm-before", "record_interview_legacy",
            "--context-store", str(path),
        ],
        stdout=StringIO(),
        stderr=errors,
    )
    assert rejected != 0
    assert "record_interview_legacy" in errors.getvalue()

    code, replaced = _run(
        [
            "settings", "set", "--user-id", "u1",
            "--confirm-before", "create_interview",
            "--context-store", str(path),
        ]
    )
    assert code == 0
    assert replaced["owner_settings"]["behavior_policy"]["confirm_before"] == [
        "create_interview"
    ]
