from __future__ import annotations

from typing import Protocol

from career_agent.agent.main_agent_tools import MainAgentToolOutput
from career_agent.agent.main_state import MainAgentState
from career_agent.agent.summary_text import (
    DELIVERY_SUMMARY_LIMIT,
    MODEL_REPLY_LIMIT,
    clamp,
)


class PresentationRenderer(Protocol):
    """Rendering operations used by delivery selection and composition."""

    def _assistant_message(self, result: MainAgentToolOutput) -> str: ...

    def _last_result(self, state: MainAgentState) -> MainAgentToolOutput | None: ...

    def _screen_message(self, result: MainAgentToolOutput) -> str: ...

    def _turn_is_card_backed(
        self, results: tuple[MainAgentToolOutput, ...]
    ) -> bool: ...

    def _undelivered_bodies(
        self, results: tuple[MainAgentToolOutput, ...]
    ) -> str: ...


class PresentationEngine:
    """Select and compose a turn's final delivery without rendering payloads."""

    @staticmethod
    def present(
        state: MainAgentState, *, renderer: PresentationRenderer
    ) -> MainAgentState:
        decision = state["decision"]
        pending = state.get("pending", {})
        pending_result = pending.get("result")
        if (
            decision.action == "tool_call"
            and (
                pending.get("synthetic_kind") == "projection"
                or (
                    pending.get("synthetic_kind") == "authorization"
                    and pending.get("runtime_owned")
                )
            )
            and pending_result is not None
        ):
            return {
                "assistant_message": renderer._assistant_message(pending_result)
            }

        result = renderer._last_result(state)
        if decision.action in {"ask_user", "questionnaire"}:
            failures = tuple(
                item
                for item in state.get("tool_results", ())
                if item.disposition == "failed"
            )
            if failures:
                return {
                    "assistant_message": "\n\n".join(
                        dict.fromkeys(
                            renderer._screen_message(item) for item in failures
                        )
                    ),
                    "model_message": "",
                }

        if (
            decision.action == "final"
            and result is not None
            and result.disposition == "failed"
        ):
            return PresentationEngine._present_failed_final(
                state,
                result=result,
                renderer=renderer,
            )

        if decision.action == "final" and (decision.message or "").strip():
            results = state.get("tool_results", ())
            body = renderer._undelivered_bodies(results)
            only_cards = renderer._turn_is_card_backed(results)
            reply = clamp(
                decision.message,
                limit=(
                    DELIVERY_SUMMARY_LIMIT if only_cards else MODEL_REPLY_LIMIT
                ),
            )
            return {
                "assistant_message": f"{reply}\n\n{body}" if body else reply,
                "model_message": reply,
            }

        if result is not None:
            return {"assistant_message": renderer._screen_message(result)}
        return {"assistant_message": "本轮可执行步骤已达到上限，请确认后继续。"}

    @staticmethod
    def _present_failed_final(
        state: MainAgentState,
        *,
        result: MainAgentToolOutput,
        renderer: PresentationRenderer,
    ) -> MainAgentState:
        decision = state["decision"]
        results = state.get("tool_results", ())
        authoritative = renderer._screen_message(result)
        if all(item.disposition == "failed" for item in results):
            return {
                "assistant_message": authoritative,
                "model_message": "",
            }
        message = (decision.message or "").strip()
        if not message:
            return {"assistant_message": authoritative}
        only_cards = renderer._turn_is_card_backed(results)
        separator = "\n\n"
        limit = DELIVERY_SUMMARY_LIMIT if only_cards else MODEL_REPLY_LIMIT
        prose_limit = limit - len(authoritative) - len(separator)
        if prose_limit < 1:
            return {"assistant_message": authoritative}
        reply = clamp(message, limit=prose_limit)
        durable_reply = f"{authoritative}{separator}{reply}"
        body = renderer._undelivered_bodies(results)
        return {
            "assistant_message": (
                f"{durable_reply}\n\n{body}" if body else durable_reply
            ),
            "model_message": durable_reply,
        }
