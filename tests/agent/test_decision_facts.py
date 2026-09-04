"""The single place that says what structured facts the model can ever see.

Facts used to live in three hardcoded copies: a central state→keys registry, a
per-state type check beside it, and an if-chain in the runtime that re-parsed
the payload a handler had just built from typed objects. Keeping three copies in
step needed a test, and one direction of the drift was silent — a state declared
in the registry with no matching branch produced empty facts and raised nothing.

They are now declared once, by the capability, next to the typed objects it
holds; the contract bounds their *shape* rather than their membership. That
removes the drift but also removes the one real benefit of a central table:
being able to read, in one place, everything structured the model is ever given.
This file is that place. A capability that starts declaring facts without
appearing here fails the closure check below.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from career_agent.agent.main_agent_contracts import (
    MAX_DECISION_FACTS,
    ToolObservation,
    validate_decision_facts,
)
from career_agent.agent.main_agent_runtime import MainAgentRuntime
from career_agent.agent.main_agent_tools import MainAgentToolRegistry


_TOOLS_SOURCE = Path(inspect.getsourcefile(MainAgentToolRegistry))


# state -> the facts that state declares, and why the receipt cannot carry them.
DECLARED_FACTS = {
    "daily_brief_ready": {
        "keys": {"overdue", "due_today", "waiting"},
        "reason": "收据只能给出总数；分桶写进一句话就成了报告本身",
    },
    "resume_analysis_ready": {
        "keys": {"record_count", "clarification_count", "has_warnings"},
        "reason": "是否先追问用户取决于澄清与告警数，列举它们等于复述结果",
    },
    "job_research_ready": {
        "keys": {"cached", "finding_count", "status"},
        "reason": "三值状态需要精确取值，散文收据说不准",
    },
}


def _states_declaring_facts() -> set[str]:
    """Every state whose emitter passes ``facts=`` at construction."""
    tree = ast.parse(_TOOLS_SOURCE.read_text())
    states: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        if "facts" not in kwargs:
            continue
        state = kwargs.get("state")
        if isinstance(state, ast.Constant) and isinstance(state.value, str):
            states.add(state.value)
    return states


def test_no_capability_declares_facts_without_appearing_here() -> None:
    assert _states_declaring_facts() == set(DECLARED_FACTS)


def test_every_declared_fact_reaches_the_model_with_its_declared_keys() -> None:
    """Declaring a fact is not enough; it has to survive to the observation."""
    payloads = {
        "daily_brief_ready": {},
        "resume_analysis_ready": {},
        "job_research_ready": {},
    }
    for state, spec in DECLARED_FACTS.items():
        facts = {
            key: (1 if key not in {"cached", "has_warnings"} else True)
            for key in spec["keys"]
        }
        if state == "job_research_ready":
            facts["status"] = "current"
        observation = MainAgentRuntime._tool_observation(
            "probe",
            ToolObservation(
                tool_name="probe",
                state=state,
                message="收据。",
                facts=facts,
                payload=payloads[state],
            ),
        )
        assert set(observation.facts) == spec["keys"], state
        assert spec["reason"], state


def test_shape_is_bounded_and_identifier_free() -> None:
    validate_decision_facts({"overdue": 3, "cached": True, "status": "current"})

    with pytest.raises(ValueError, match="internal identifier"):
        validate_decision_facts({"report_id": 1})
    with pytest.raises(ValueError, match="internal identifier"):
        validate_decision_facts({"anchor": "a" * 32})
    with pytest.raises(ValueError, match="exceed"):
        validate_decision_facts({f"k{i}": i for i in range(MAX_DECISION_FACTS + 1)})


def test_a_failed_result_may_only_tell_the_model_about_retryability() -> None:
    with pytest.raises(ValueError, match="only declare retryability"):
        ToolObservation(
            tool_name="probe",
            state="failed",
            message="失败。",
            facts={"finding_count": 2},
        )


def test_no_capability_hands_the_model_an_enum_where_prose_was_meant() -> None:
    """``next_action`` is prose or nothing — never a snake_case token again.

    The field comes from an industry pattern whose whole effect lives in the
    sentence: "Do not retry without user input" is followed, ``review_job_research``
    is guessed at. This project had kept the shape and dropped the sentence.

    The scan exists because the first pass at fixing that missed four sites. It
    rewrote every ``next_action="literal"`` and left every conditional
    expression — ``"confirm_email_events" if pending else ...`` — untouched, and
    nothing failed, because a stale hint breaks no assertion. A reviewer caught
    them. An AST walk over the argument catches the shape wherever it hides.

    ASCII is the test: every surviving hint is a Chinese sentence, and every
    deleted one was an ASCII identifier. A future English hint would trip this
    and should — read it, and if it is really a sentence, widen the check to
    "contains a space" rather than deleting the guard.
    """
    offenders: list[str] = []
    for source in Path("src/career_agent").rglob("*.py"):
        # The mock interview worker's own ``next_action`` is a different field:
        # a closed three-value enum for the graph, paired with a prose
        # ``next_action_reason``. The CLI's is operator-facing English prose.
        if "mock_interviews" in str(source) or source.name == "cli.py":
            continue
        for node in ast.walk(ast.parse(source.read_text())):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if keyword.arg != "next_action":
                    continue
                for inner in _hint_values(keyword.value):
                    if isinstance(inner, str) and inner and inner.isascii():
                        offenders.append(f"{source.name}:{node.lineno} {inner!r}")

    assert offenders == []


def _hint_values(node: ast.expr) -> list[object]:
    """Every value ``next_action`` could take, ignoring how it is chosen.

    A conditional hint is written ``"..." if draft.status == "pending" else None``.
    Walking the whole expression would also read ``"pending"`` — the condition,
    not the hint — so only the branches count.
    """
    if isinstance(node, ast.Constant):
        return [node.value]
    if isinstance(node, ast.IfExp):
        return _hint_values(node.body) + _hint_values(node.orelse)
    if isinstance(node, ast.JoinedStr):
        return ["".join(
            part.value for part in node.values if isinstance(part, ast.Constant)
        )]
    return []
