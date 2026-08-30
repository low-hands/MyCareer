"""The CLI surface for reclaiming summarised conversation history.

Deleting the originals a summary covers is irreversible, so it is a command the
operator runs rather than something a turn does on its own.
"""

from io import StringIO
import json

from career_agent.agent.context_manager import ContextManager
from career_agent.agent.conversation_memory_contracts import (
    ConversationSummaryContent,
)
from career_agent.agent.main_agent_contracts import (
    AgentDecision,
    ConversationTaskState,
)
from career_agent.agent.main_agent_runtime import MainAgentTurnResult
from career_agent.cli import EXIT_ARGUMENT_ERROR, EXIT_OK, main
from career_agent.storage.context import CareerContextStore


class _Summariser:
    def summarize(self, *, previous, messages):
        return ConversationSummaryContent(
            user_goals=("保持对话连续性",),
            confirmed_decisions=(f"covered-through-{messages[-1].sequence}",),
            unresolved_questions=(),
            active_constraints=(),
        )


def _seed(path, *, turns: int = 20) -> None:
    context_manager = ContextManager(
        CareerContextStore(path),
        summary_worker=_Summariser(),
        recent_message_limit=4,
        summary_batch_size=2,
    )
    for index in range(turns):
        context = context_manager.load_for_turn(
            user_id="u1", conversation_id="c1", user_message=f"user-{index}"
        )
        context_manager.commit_turn(
            context=context,
            task=ConversationTaskState(),
            assistant_message=f"assistant-{index}",
        )


def _run(arguments):
    output = StringIO()
    code = main(arguments, stdout=output, stderr=StringIO())
    return code, json.loads(output.getvalue())


def test_stat_reports_what_is_reclaimable_without_deleting_it(tmp_path) -> None:
    store_path = tmp_path / "context.sqlite3"
    _seed(store_path)

    code, payload = _run(
        [
            "context",
            "stat",
            "--user-id",
            "u1",
            "--context-store",
            str(store_path),
        ]
    )

    assert code == EXIT_OK
    assert payload["compacted_messages"] > 0
    assert payload["compacted_bytes"] > 0
    assert payload["reclaimable"] is True
    # Reading is not acting: a second stat sees the same rows.
    _, again = _run(
        ["context", "stat", "--user-id", "u1", "--context-store", str(store_path)]
    )
    assert again["compacted_messages"] == payload["compacted_messages"]


def test_prune_refuses_without_explicit_confirmation(tmp_path) -> None:
    """No prompt, a refusal: the command behaves the same in a script.

    An interactive confirmation would either block a scripted run forever or be
    silently skipped, and both are worse than making the caller say --yes.
    """
    store_path = tmp_path / "context.sqlite3"
    _seed(store_path)
    before = CareerContextStore(store_path).count_compacted_messages(user_id="u1")

    code, payload = _run(
        ["context", "prune", "--user-id", "u1", "--context-store", str(store_path)]
    )

    assert code == EXIT_ARGUMENT_ERROR
    assert payload["state"] == "failed"
    assert "--yes" in payload["error_detail"]
    assert CareerContextStore(store_path).count_compacted_messages(user_id="u1") == before


def test_prune_with_confirmation_deletes_and_reports_what_it_freed(tmp_path) -> None:
    store_path = tmp_path / "context.sqlite3"
    _seed(store_path)
    expected, expected_bytes = CareerContextStore(store_path).count_compacted_messages(
        user_id="u1"
    )

    code, payload = _run(
        [
            "context",
            "prune",
            "--user-id",
            "u1",
            "--yes",
            "--context-store",
            str(store_path),
        ]
    )

    assert code == EXIT_OK
    assert payload["deleted_messages"] == expected
    assert payload["reclaimed_bytes"] == expected_bytes
    assert CareerContextStore(store_path).count_compacted_messages(user_id="u1") == (0, 0)


def test_prune_can_be_scoped_to_one_session(tmp_path) -> None:
    store_path = tmp_path / "context.sqlite3"
    _seed(store_path)

    code, payload = _run(
        [
            "context",
            "prune",
            "--user-id",
            "u1",
            "--session-id",
            "c-other",
            "--yes",
            "--context-store",
            str(store_path),
        ]
    )

    assert code == EXIT_OK
    assert payload["deleted_messages"] == 0
    assert CareerContextStore(store_path).count_compacted_messages(user_id="u1")[0] > 0


class _Runtime:
    """A runtime whose context manager is the seeded one, as the real one is."""

    def __init__(self, context_manager) -> None:
        self.context_manager = context_manager

    def run_turn(self, *, user_id, conversation_id, user_message):
        return MainAgentTurnResult(
            decision=AgentDecision(action="final", message="好的。"),
            context=type("Context", (), {"task": None})(),
            assistant_message="好的。",
        )

    def close(self):
        pass


def test_chat_warns_the_operator_once_the_threshold_is_crossed(tmp_path) -> None:
    """The notice rides on the CLI payload, not on the agent's context.

    If the model could see it, the model could decide to prune, which is the one
    thing this whole arrangement exists to prevent.
    """
    store_path = tmp_path / "context.sqlite3"
    _seed(store_path)
    context_manager = ContextManager(
        CareerContextStore(store_path),
        summary_worker=_Summariser(),
        recent_message_limit=4,
        summary_batch_size=2,
        compacted_message_warning_threshold=4,
    )
    output = StringIO()

    code = main(
        [
            "chat",
            "--user-id",
            "u1",
            "--session-id",
            "c1",
            "--message",
            "继续",
        ],
        runtime_factory=lambda args: _Runtime(context_manager),
        stdout=output,
        stderr=StringIO(),
    )

    payload = json.loads(output.getvalue())
    assert code == EXIT_OK
    assert "context prune" in payload["maintenance_notice"]
    loaded = context_manager.load_for_turn(
        user_id="u1", conversation_id="c1", user_message="继续"
    )
    rendered = " ".join(message.content for message in loaded.recent_messages)
    assert "context prune" not in rendered
