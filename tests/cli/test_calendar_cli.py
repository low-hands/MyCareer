import json
from io import StringIO

from career_agent.cli import main


def test_calendar_account_cli_stores_env_reference_without_secret_output(tmp_path) -> None:
    store = tmp_path / "calendar.sqlite3"
    output = StringIO()

    code = main(
        [
            "calendar", "add-account", "--user-id", "u1",
            "--address", "user@example.com", "--calendar-id", "primary",
            "--credential-env", "GOOGLE_CALENDAR_CREDENTIAL",
            "--calendar-store", str(store),
        ],
        stdout=output,
        stderr=StringIO(),
    )

    payload = json.loads(output.getvalue())
    assert code == 0
    assert payload["account"]["provider"] == "google"
    assert payload["account"]["calendar_id"] == "primary"
    assert "credential" not in output.getvalue().casefold()

    listed = StringIO()
    code = main(
        [
            "calendar", "list-accounts", "--user-id", "u1",
            "--calendar-store", str(store),
        ],
        stdout=listed,
        stderr=StringIO(),
    )
    assert code == 0
    assert json.loads(listed.getvalue())["accounts"][0]["email_address"] == "user@example.com"
