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


# state -> every fact the state may declare, and why the receipt cannot carry it.
# Individual emissions may omit a fact that does not apply, such as
# ``status_changed_at`` for a current claim.
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
    "claim_source_found": {
        "keys": {
            "origin",
            "recorded_at",
            "source_locator",
            "resume_version",
            "claim_status",
            "status_changed_at",
            "body_clipped",
        },
        "reason": "来源元数据保持结构化；原始引文只进入低授权正文",
    },
    "career_memory_detail_found": {
        "keys": {
            "revision",
            "support_count",
            "lineage_count",
            "body_clipped",
        },
        "reason": "当前值与关系计数结构化；完整谱系只进入低授权正文",
    },
    "career_memory_search_found": {
        "keys": {"returned", "total", "body_clipped", "next_cursor"},
        "reason": "归档层查询必须显式报告有界结果、真实总数与可见分页",
    },
    "career_episode_search_found": {
        "keys": {"returned", "body_clipped"},
        "reason": "L1 检索正文有界，收据只报告返回数量和截断状态",
    },
    "career_history_found": {
        "keys": {"returned", "total", "body_clipped", "next_cursor"},
        "reason": "历史查询必须显式报告有界结果、真实总数与可见分页",
    },
}


def _fact_keys_by_state() -> dict[str, set[str]]:
    """Every literal fact key, including keys hidden in conditional spreads."""

    tree = ast.parse(_TOOLS_SOURCE.read_text())
    builders: dict[str, ast.expr] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        returns = [
            item.value
            for item in ast.walk(node)
            if isinstance(item, ast.Return) and item.value is not None
        ]
        if len(returns) == 1:
            builders[node.name] = returns[0]
    keys_by_state: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg}
        if "facts" not in kwargs:
            continue
        state = kwargs.get("state")
        if isinstance(state, ast.Constant) and isinstance(state.value, str):
            keys_by_state.setdefault(state.value, set()).update(
                _literal_fact_keys(kwargs["facts"], builders=builders)
            )
    return keys_by_state


def _literal_fact_keys(
    node: ast.expr,
    *,
    builders: dict[str, ast.expr],
) -> set[str]:
    if isinstance(node, ast.Dict):
        keys: set[str] = set()
        for key, value in zip(node.keys, node.values, strict=True):
            if key is None:
                keys.update(_literal_fact_keys(value, builders=builders))
            elif isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.add(key.value)
            else:
                raise AssertionError(
                    f"facts at line {node.lineno} contains a dynamic key"
                )
        return keys
    if isinstance(node, ast.IfExp):
        return _literal_fact_keys(
            node.body, builders=builders
        ) | _literal_fact_keys(node.orelse, builders=builders)
    if isinstance(node, ast.Call):
        builder_name = (
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else node.func.id
            if isinstance(node.func, ast.Name)
            else None
        )
        if builder_name in builders:
            return _literal_fact_keys(builders[builder_name], builders=builders)
    raise AssertionError(
        f"facts at line {node.lineno} is not a statically inspectable dict"
    )


def test_no_capability_declares_facts_without_appearing_here() -> None:
    assert set(_fact_keys_by_state()) == set(DECLARED_FACTS)


def test_capabilities_only_emit_declared_fact_keys() -> None:
    emitted = _fact_keys_by_state()
    unexpected = {
        state: keys - DECLARED_FACTS[state]["keys"]
        for state, keys in emitted.items()
        if keys - DECLARED_FACTS[state]["keys"]
    }
    missing = {
        state: spec["keys"] - emitted[state]
        for state, spec in DECLARED_FACTS.items()
        if spec["keys"] - emitted[state]
    }

    assert unexpected == {}
    assert missing == {}


def test_every_declared_fact_reaches_the_model_with_its_declared_keys() -> None:
    """Declaring a fact is not enough; it has to survive to the observation."""
    payloads = {
        "daily_brief_ready": {},
        "resume_analysis_ready": {},
        "job_research_ready": {},
        "claim_source_found": {},
        "career_memory_detail_found": {},
        "career_memory_search_found": {},
        "career_episode_search_found": {},
        "career_history_found": {},
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
        # The mock interview's own ``next_action`` is a different field: a
        # closed three-value enum the graph routes on (the worker and graph
        # build it too), paired with a prose ``next_action_reason``. The CLI's
        # is operator-facing English prose.
        if "mock_interview" in str(source) or source.name == "cli.py":
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
