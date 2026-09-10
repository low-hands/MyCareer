from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal
import re
import unicodedata

from pydantic import BaseModel, ConfigDict, Field, model_validator

PreferenceStance = Literal["negative", "positive"]


def normalize_preference_stance(value: object) -> PreferenceStance | None:
    """Collapse model wording to the only polarity values storage compares."""

    if not isinstance(value, str):
        return None
    normalized = re.sub(
        r"[^a-z0-9]+",
        "_",
        unicodedata.normalize("NFKC", value).strip().casefold(),
    ).strip("_")
    if not normalized:
        return None
    head = normalized.split("_", 1)[0]
    if head in {
        "negative",
        "avoid",
        "reject",
        "exclude",
        "disallow",
        "dislike",
        "refuse",
        "oppose",
        "unwilling",
        "never",
    }:
        return "negative"
    if head in {
        "positive",
        "prefer",
        "accept",
        "allow",
        "favor",
        "like",
        "seek",
        "want",
        "willing",
        "open",
    }:
        return "positive"
    return None


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
    stance: PreferenceStance | None = None
    ownership: Literal[
        "person_stable",
        "person_default",
        "person_situational",
        "role",
        "situational",
    ] | None = None

    @model_validator(mode="after")
    def payload_matches_action(self) -> "FreeTextPreferenceMutation":
        if self.action == "delete":
            if (
                self.statement is not None
                or self.stance is not None
                or self.ownership is not None
            ):
                raise ValueError("delete mutations cannot carry a preference value")
        elif (
            self.statement is None
            or self.stance is None
            or self.ownership is None
        ):
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
_SCOPED_NEGATIVE = re.compile(
    r"^\s*(?:我\s*)?(?:对\s*)?"
    r"(?:这类岗位|这种岗位|这个方向|同类岗位)[，,\s]*"
    r"(?:不去|不考虑|不想去|不会去|排除)(?:任何)?(?:大厂|大公司)"
)
_STABLE_NEGATIVE = re.compile(
    r"^\s*(?:我\s*)?(?:以后都|永远|一直|始终|绝不)"
    r"(?:再)?(?:不去|不考虑|不想去|不会去|排除)"
    r"(?:任何)?(?:大厂|大公司)"
)
_TIMED_NEGATIVE = re.compile(
    r"^\s*(?:我\s*)?"
    r"(?:年底前|年内|未来\d{1,3}[天周月]|接下来\d{1,3}[天周月]|"
    r"这段时间|暂时|目前|最近)"
    r".{0,8}(?:不去|不考虑|不想去|不会去|排除)"
    r"(?:任何)?(?:大厂|大公司)"
)
_SITUATIONAL_EXCEPTION = re.compile(
    r"(?:这家|这个岗位|这份工作).{0,12}(?:例外|可以|愿意考虑|能接受)"
)
_STABLE_SCOPE = re.compile(r"(?:一直|始终|永远|以后都|无论什么|绝不)")
_ROLE_SCOPE = re.compile(r"(?:这类|这类岗位|这种岗位|这个方向|同类岗位)")
_SITUATIONAL_SCOPE = re.compile(
    r"(?:这家|这个岗位|这份工作|这次|当前这个|本次)"
)
_PERSON_SITUATIONAL_SCOPE = re.compile(
    r"(?:年底前|年内|未来\d{1,3}[天周月]|接下来\d{1,3}[天周月]|"
    r"这段时间|暂时|目前|最近)"
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
    if (
        _NEGATIVE.search(statement)
        or _SCOPED_NEGATIVE.search(statement)
        or _STABLE_NEGATIVE.search(statement)
        or _TIMED_NEGATIVE.search(statement)
        or _POSITIVE.search(statement)
        or _SITUATIONAL_EXCEPTION.search(statement)
    ):
        positive = bool(
            _POSITIVE.search(statement)
            or _SITUATIONAL_EXCEPTION.search(statement)
        )
        return FreeTextPreferenceMutation(
            action="quarantine",
            topic_key="employer_scale",
            statement=statement,
            stance="positive" if positive else "negative",
            ownership=classify_preference_ownership(statement),
        )
    return None


def preference_topic_is_relevant(
    topic_key: str,
    message: str,
    statement: str | None = None,
) -> bool:
    """Apply only deterministic shortcuts before persistent FTS/RRF retrieval."""

    if is_explicit_confirmation(message):
        return True
    if topic_key == "employer_scale" and _RELATED.search(message):
        return True
    return False


def preference_fts_query(message: str) -> str | None:
    """Build a bounded trigram MATCH query; two-character CJK is excluded."""

    normalized = unicodedata.normalize("NFKC", message).casefold()
    terms = {
        token
        for token in re.findall(
            r"[a-z0-9][a-z0-9.+#-]{2,}",
            normalized,
        )
    }
    for run in re.findall(r"[\u4e00-\u9fff]{3,}", normalized):
        terms.update(
            run[index : index + 3]
            for index in range(len(run) - 2)
        )
    if not terms:
        return None
    return " OR ".join(
        f'"{term.replace(chr(34), chr(34) * 2)}"'
        for term in sorted(terms)
    )


def is_explicit_confirmation(message: str) -> bool:
    return _CONFIRMATION.fullmatch(message.strip()) is not None


class PreferenceStorageAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pref_scope: str
    timescale: Literal["permanent", "situational"]
    layer: Literal["stable", "contextual", "transient"]
    valid_until: datetime | None = None
    scope_ambiguous: bool = False


def classify_preference_ownership(
    statement: str,
) -> Literal[
    "person_stable",
    "person_default",
    "person_situational",
    "role",
    "situational",
]:
    if _STABLE_SCOPE.search(statement):
        return "person_stable"
    if _SITUATIONAL_SCOPE.search(statement):
        return "situational"
    if _PERSON_SITUATIONAL_SCOPE.search(statement):
        return "person_situational"
    if _ROLE_SCOPE.search(statement):
        return "role"
    return "person_default"


def preference_ownership_from_storage(
    *,
    pref_scope: str,
    layer: str,
    timescale: str,
) -> Literal[
    "person_stable",
    "person_default",
    "person_situational",
    "role",
    "situational",
]:
    """Recover the four-layer display ownership from persisted metadata."""

    scope_name = (
        pref_scope.removeprefix("freeform.")
        if pref_scope.startswith("freeform.")
        else "person_default"
        if pref_scope == "freeform"
        else pref_scope
    )
    if layer == "stable":
        return "person_stable"
    if layer == "transient" or timescale == "situational":
        return (
            "person_situational"
            if scope_name == "person_situational"
            else "situational"
        )
    if scope_name.startswith("role."):
        return "role"
    return "person_default"


def preference_storage_assignment(
    ownership: str,
    *,
    observed_at: datetime,
    conversation_id: str,
    job_posting_id: str | None = None,
    role_domain: str | None = None,
    valid_for_days: int = 30,
    statement: str | None = None,
    valid_until: datetime | None = None,
) -> PreferenceStorageAssignment:
    if ownership == "person_stable":
        return PreferenceStorageAssignment(
            pref_scope="freeform.person_stable",
            timescale="permanent",
            layer="stable",
        )
    if ownership == "person_default":
        return PreferenceStorageAssignment(
            pref_scope="freeform.person_default",
            timescale="permanent",
            layer="contextual",
        )
    if ownership == "person_situational":
        return PreferenceStorageAssignment(
            pref_scope="freeform.person_situational",
            timescale="situational",
            layer="transient",
            valid_until=(
                valid_until
                or explicit_preference_valid_until(
                    statement, observed_at=observed_at
                )
                or observed_at + timedelta(days=valid_for_days)
            ),
        )
    if ownership == "role":
        domain = preference_scope_domain(role_domain)
        return PreferenceStorageAssignment(
            pref_scope=(
                f"freeform.role.{domain}"
                if domain is not None
                else "freeform.person_default"
            ),
            timescale="permanent",
            layer="contextual",
            scope_ambiguous=domain is None,
        )
    if ownership == "situational":
        situation = (
            f"job.{job_posting_id.casefold()}"
            if job_posting_id
            else (
                "conversation."
                f"{preference_scope_domain(conversation_id) or 'current'}"
            )
        )
        return PreferenceStorageAssignment(
            pref_scope=f"freeform.{situation}",
            timescale="situational",
            layer="transient",
            valid_until=(
                valid_until
                or explicit_preference_valid_until(
                    statement, observed_at=observed_at
                )
                or observed_at + timedelta(days=valid_for_days)
            ),
        )
    raise ValueError("unknown preference ownership")


def structured_pref_scope(free_text_pref_scope: str) -> str:
    scope = free_text_pref_scope.removeprefix("freeform.")
    if scope in {"person_stable", "global"}:
        return "global"
    if scope in {"person_default", "freeform"}:
        return "person_default"
    return scope


def explicit_preference_valid_until(
    statement: str | None,
    *,
    observed_at: datetime,
) -> datetime | None:
    """Resolve a small set of explicit user time bounds without an LLM."""

    if not statement:
        return None
    if re.search(r"(?:年底前|年内)", statement):
        return observed_at.replace(
            year=observed_at.year + 1,
            month=1,
            day=1,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
    relative = re.search(
        r"(?:未来|接下来)(\d{1,3})(天|周|月)",
        statement,
    )
    if relative is None:
        return None
    amount = int(relative.group(1))
    unit = relative.group(2)
    days = amount if unit == "天" else amount * 7 if unit == "周" else amount * 30
    return observed_at + timedelta(days=days)


def preference_scope_key(topic_key: str) -> str:
    relation = (
        "company_scale"
        if topic_key == "employer_scale"
        else f"{topic_key}_preference"
    )
    return f"person_intent/self/{relation}"


def preference_topic_key(scope_key: str) -> str:
    relation = scope_key.rsplit("/", 1)[-1]
    if relation == "company_scale":
        return "employer_scale"
    return relation.removesuffix("_preference")


def preference_scope_domain(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    domain = re.sub(r"[^a-z0-9_.:-]+", "_", normalized).strip("_.:-")
    return domain[:80] or None
