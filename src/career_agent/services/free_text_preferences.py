from __future__ import annotations

from typing import Literal
import re

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FreeTextPreferenceMutation(BaseModel):
    """A deliberately narrow deterministic extraction result.

    This first slice recognizes only explicit employer-scale statements.  It
    abstains on implications (for example, merely viewing a big-company job),
    keeping the extraction boundary safer than a broad sentiment heuristic.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["quarantine", "delete"]
    topic_key: Literal["employer_scale"]
    statement: str | None = Field(default=None, min_length=1, max_length=2000)
    stance: Literal["negative", "positive"] | None = None

    @model_validator(mode="after")
    def payload_matches_action(self) -> "FreeTextPreferenceMutation":
        if self.action == "delete":
            if self.statement is not None or self.stance is not None:
                raise ValueError("delete mutations cannot carry a preference value")
        elif self.statement is None or self.stance is None:
            raise ValueError("quarantined mutations require a statement and stance")
        return self


_DELETE = re.compile(
    r"(?:删除|删掉|清除|忘掉|别记|不要再记).{0,24}(?:大厂|大公司)"
    r"|(?:大厂|大公司).{0,24}(?:偏好|记忆|记录).{0,12}(?:删除|删掉|清除|忘掉)"
)
_NEGATIVE = re.compile(
    r"^\s*(?:我(?:又|现在|还是|已经)?(?:想清楚|决定|确定|明确)?(?:了)?[，,、\s]*)?"
    r"(?:不去|不考虑|不想去|不会去|拒绝去|排除)(?:任何)?(?:大厂|大公司)"
)
_POSITIVE = re.compile(
    r"^\s*(?:我(?:又|现在|还是|已经)?(?:改主意|决定|确定|明确)?(?:了)?[，,、\s]*)?"
    r"(?:想去|考虑|接受|愿意去|可以去|会去)(?:大厂|大公司)"
)
_RELATED = re.compile(
    r"(?:岗位|职位|工作|求职|推荐|筛选|匹配|投递|公司|雇主|offer|大厂|大公司)",
    re.IGNORECASE,
)
_CONFIRMATION = re.compile(r"^\s*(?:确认|是的|对|同意|可以|就这样)[。.!！]?\s*$")


def extract_free_text_preference(message: str) -> FreeTextPreferenceMutation | None:
    """Extract the first supported explicit preference mutation, or abstain."""

    statement = message.strip()
    if not statement or len(statement) > 2000:
        return None
    if _DELETE.search(statement):
        return FreeTextPreferenceMutation(
            action="delete",
            topic_key="employer_scale",
        )
    if _NEGATIVE.search(statement) or _POSITIVE.search(statement):
        return FreeTextPreferenceMutation(
            action="quarantine",
            topic_key="employer_scale",
            statement=statement,
            stance="negative" if _NEGATIVE.search(statement) else "positive",
        )
    return None


def preference_topic_is_relevant(
    topic_key: str,
    message: str,
    statement: str | None = None,
) -> bool:
    """Whether a bounded preference projection is relevant on this turn."""

    if is_explicit_confirmation(message) or _RELATED.search(message):
        return True
    if statement is None:
        return False
    normalized_message = re.sub(r"\s+", "", message).casefold()
    normalized_statement = re.sub(r"\s+", "", statement).casefold()
    topic_terms = {
        term
        for term in topic_key.casefold().split("_")
        if len(term) >= 2
    }
    ascii_terms = set(
        re.findall(r"[a-z0-9][a-z0-9.+#-]{1,}", normalized_statement)
    )
    # This bounded lexical fallback is intentionally small-scale. Replace it
    # with the evidence projector's FTS/RRF path before topic volume makes
    # two-character overlap too permissive.
    cjk_terms = {
        normalized_statement[index : index + 2]
        for index in range(max(0, len(normalized_statement) - 1))
        if all(
            "\u4e00" <= character <= "\u9fff"
            for character in normalized_statement[index : index + 2]
        )
    }
    return any(
        term in normalized_message
        for term in (*topic_terms, *ascii_terms, *cjk_terms)
    )


def is_explicit_confirmation(message: str) -> bool:
    return _CONFIRMATION.fullmatch(message.strip()) is not None
