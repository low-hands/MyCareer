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
