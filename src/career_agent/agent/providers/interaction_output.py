"""Native output channels consumed by the provider, never the tool executor."""
import re
from typing import Any
from career_agent.agent.contracts.decisions import AgentDecision
from career_agent.agent.contracts.questionnaire import UserQuestion

INTERACTION_NAMES = frozenset({"ask_user", "questionnaire", "final_response", "respond_to_user"})
_OPTION_VALUE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")


def _normalize_question_identifiers(arguments: dict[str, Any]) -> dict[str, Any]:
    """Normalize presentation identifiers before storing a pending questionnaire."""
    questions = arguments.get("questions")
    if not isinstance(questions, list):
        return arguments
    normalized = []
    for index, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            normalized.append(question)
            continue
        fixed = {**question, "question_id": f"q{index}"}
        options = question.get("options")
        if isinstance(options, list):
            fixed_options = []
            for position, option in enumerate(options, start=1):
                if not isinstance(option, dict):
                    fixed_options.append(option)
                    continue
                value = option.get("value")
                valid = isinstance(value, str) and _OPTION_VALUE.fullmatch(value.strip())
                fixed_options.append(
                    dict(option) if valid else {**option, "value": f"option_{position}"}
                )
            fixed["options"] = fixed_options
        normalized.append(fixed)
    return {**arguments, "questions": normalized}


def interaction_schemas(*, continuation: bool = False) -> tuple[dict[str, Any], ...]:
    question = UserQuestion.model_json_schema()
    definitions = question.pop("$defs", {})
    message = {"type": "string", "minLength": 1}
    continuation_property = (
        {
            "continuation_capability": {
                "type": ["string", "null"],
                "description": (
                    "Exact capability name to resume after these answers when the "
                    "questionnaire pauses a requested task; otherwise null."
                ),
            }
        }
        if continuation else {}
    )
    specs = (
        ("ask_user", "Ask ONE missing fact, choice or confirmation needed to continue the current goal. Use this for a required follow-up question, never final_response. Ask before comparing or recommending using a note-only preference. Do not update notes or call an unrelated business tool instead of asking.",
         {"message": message, "selection_source": {"type": "string", "enum": ["latest_tool_result"]}}),
        ("questionnaire", "Ask 2-8 independent missing facts in a structured questionnaire, with ordered ids q1..qN. Do not pre-collect facts that a requested workflow collects through its own selection interface.",
         {"message": message, "questions": {"type": "array", "minItems": 2, "maxItems": 8, "items": question},
          **continuation_property}),
        ("final_response", "Finish with a grounded answer or explain a stopped operation. Do NOT ask required follow-up questions, obtain confirmation, invent missing evidence, or compare/recommend using unconfirmed notes. Do not reproduce report tables or bodies delivered through a card. Explain precisely when needed evidence is unavailable.",
         {"message": message}),
        ("respond_to_user", "Return a typed user-facing interaction. First determine whether the current task requires a user answer, choice or confirmation. If so, requires_user_input MUST be true, even when the message also explains a limitation. Set false only for a completed answer or a blocked outcome needing no answer. Questions are never completed answers.",
         {"requires_user_input": {"type": "boolean", "description": "Does the task need the user to answer this message before it can proceed?"},
          "message": message, "questions": {"type": "array", "minItems": 2, "maxItems": 8, "items": question},
          **continuation_property}),
    )
    return tuple({"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": (["requires_user_input", "message"] if name == "respond_to_user" else
                                    ["message", "questions", "continuation_capability"]
                                    if name == "questionnaire" and continuation else
                                    ["message", "questions"] if name == "questionnaire" else ["message"]),
                       "additionalProperties": False,
                       **({"$defs": definitions} if name in {"questionnaire", "respond_to_user"} else {})},
    }} for name, description, properties in specs)


def parse_interaction(name: str, arguments: dict[str, Any]) -> AgentDecision:
    allowed = {"message"}
    if name == "respond_to_user":
        if set(arguments) - {"message", "requires_user_input", "questions", "continuation_capability"}:
            raise ValueError("unexpected interaction fields")
        needs_input = arguments.get("requires_user_input")
        if not isinstance(needs_input, bool):
            raise ValueError("requires_user_input must be a boolean")
        questions = arguments.get("questions", [])
        if not isinstance(questions, list):
            raise ValueError("questions must be an array")
        if not needs_input and questions:
            raise ValueError("final output cannot carry questions")
        continuation_capability = arguments.get("continuation_capability")
        if continuation_capability is not None and not questions:
            raise ValueError("continuation_capability requires questionnaire questions")
        name = "questionnaire" if needs_input and questions else "ask_user" if needs_input else "final_response"
        arguments = {
            "message": arguments.get("message"),
            **({"questions": questions} if questions else {}),
            **(
                {"continuation_capability": continuation_capability}
                if continuation_capability is not None else {}
            ),
        }
    if name == "ask_user":
        allowed.add("selection_source")
    elif name == "questionnaire":
        allowed.update({"questions", "continuation_capability"})
    elif name != "final_response":
        raise ValueError("unknown interaction output")
    if set(arguments) - allowed:
        raise ValueError("unexpected interaction fields")
    if not isinstance(arguments.get("message"), str) or not arguments["message"].strip():
        raise ValueError("interaction message is required")
    if name == "questionnaire":
        arguments = _normalize_question_identifiers(arguments)
    return AgentDecision.model_validate({
        **arguments, "action": "final" if name == "final_response" else name,
    })
