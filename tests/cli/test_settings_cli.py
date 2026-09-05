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

