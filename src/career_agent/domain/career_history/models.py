from __future__ import annotations

from datetime import datetime
import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CareerHistoryContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


def career_evidence_source_ref(
    *,
    user_id: str,
    evidence_id: str,
    source_resume_version_id: str | None,
    source_locator: str | None,
) -> str | None:
    """Construct the stable public pointer persisted with resume evidence."""

    if source_resume_version_id is None:
        return None
    digest = hashlib.sha256(
        "\0".join(
            (
                user_id,
                evidence_id,
                source_resume_version_id,
                source_locator or "",
            )
        ).encode("utf-8")
    ).hexdigest()
    return f"evidence_{digest[:24]}"


def career_evidence_detail_ref(*, user_id: str, evidence_id: str) -> str:
    """Construct an opaque lookup handle for one claim revision."""

    digest = hashlib.sha256(
        f"{user_id}\0{evidence_id}\0career-evidence-detail".encode("utf-8")
    ).hexdigest()
    return f"detail_{digest[:24]}"


def career_evidence_lineage_ref(*, user_id: str, scope_key: str) -> str:
    """Name a correction lineage without reusing a direct-support reference."""

    digest = hashlib.sha256(
        f"{user_id}\0{scope_key}\0career-evidence-lineage".encode("utf-8")
    ).hexdigest()
    return f"lineage_{digest[:24]}"


def career_evidence_scope_key(evidence_id: str) -> str:
    """Anchor one correction lineage without hashing free-text claim semantics."""

    if not evidence_id.strip():
        raise ValueError("evidence_id is required")
    return f"career_evidence/{evidence_id}/claim"


class CareerRecord(CareerHistoryContract):
    id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    record_type: Literal[
        "education",
        "work",
        "internship",
        "project",
        "certification",
    ]
    organization: str | None = Field(default=None, min_length=1)
    title: str = Field(min_length=1)
    start_year: int | None = Field(default=None, ge=1900, le=2200)
    end_year: int | None = Field(default=None, ge=1900, le=2200)
    start_month: int | None = Field(default=None, ge=1, le=12)
    end_month: int | None = Field(default=None, ge=1, le=12)
    is_current: bool = False
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def validate_period(self) -> "CareerRecord":
        if self.start_month is not None and self.start_year is None:
            raise ValueError("start_month requires start_year")

        if self.end_month is not None and self.end_year is None:
            raise ValueError("end_month requires end_year")

        if self.is_current and (
            self.end_year is not None or self.end_month is not None
        ):
            raise ValueError("current record cannot have an end date")

        if self.start_year is not None and self.end_year is not None:
            if self.end_year < self.start_year:
                raise ValueError("end date cannot be earlier than start date")

            if (
                self.end_year == self.start_year
                and self.start_month is not None
                and self.end_month is not None
                and self.end_month < self.start_month
            ):
                raise ValueError("end date cannot be earlier than start date")

        return self


class CareerEvidence(CareerHistoryContract):
    id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    career_record_id: str = Field(min_length=1)

    claim: str = Field(min_length=1)

    origin: Literal[
        "resume_extraction",
        "user_input",
        "agent_inference",
    ]

    verification_status: Literal[
        "pending",
        "confirmed",
        "rejected",
    ] = "pending"

    source_resume_version_id: str | None = Field(default=None, min_length=1)
    source_locator: str | None = Field(default=None, min_length=1)
    source_quote: str | None = Field(default=None, min_length=1)
    source_ref: str | None = Field(
        default=None,
        pattern=r"^evidence_[a-f0-9]{24}$",
    )
    detail_ref: str = Field(pattern=r"^detail_[a-f0-9]{24}$")
    scope_key: str | None = Field(
        default=None,
        pattern=r"^career_evidence/[A-Za-z0-9_.:-]+/claim$",
    )
    update_id: str | None = Field(
        default=None,
        pattern=r"^career_evidence_update_[a-f0-9]{32}$",
    )
    content_digest: str | None = Field(
        default=None,
        pattern=r"^sha256:[a-f0-9]{64}$",
    )
    revision: int | None = Field(default=None, ge=1)
    valid_from: datetime | None = None
    supersedes_id: str | None = Field(default=None, min_length=1)
    superseded_at: datetime | None = None
    superseded_by: str | None = Field(default=None, min_length=1)
    mutation_id: str | None = Field(
        default=None,
        pattern=r"^career_evidence_mutation_[a-f0-9]{32}$",
    )
    rolled_back_at: datetime | None = None

    created_at: datetime
    updated_at: datetime

    @model_validator(mode="before")
    @classmethod
    def populate_detail_ref(cls, value: object) -> object:
        if not isinstance(value, dict) or value.get("detail_ref") is not None:
            return value
        user_id = value.get("user_id")
        evidence_id = value.get("id")
        if isinstance(user_id, str) and isinstance(evidence_id, str):
            return {
                **value,
                "detail_ref": career_evidence_detail_ref(
                    user_id=user_id,
                    evidence_id=evidence_id,
                ),
            }
        return value

    @model_validator(mode="after")
    def validate_source(self) -> CareerEvidence:
        if self.origin == "resume_extraction" and (
            self.source_resume_version_id is None
            or self.source_locator is None
            or self.source_quote is None
        ):
            raise ValueError(
                "resume extraction requires source_resume_version_id, source_locator, and source_quote"
            )

        if self.source_locator is not None and self.source_resume_version_id is None:
            raise ValueError("source_locator requires source_resume_version_id")

        if self.source_quote is not None and self.source_resume_version_id is None:
            raise ValueError("source_quote requires source_resume_version_id")

        if self.source_ref is not None and self.source_resume_version_id is None:
            raise ValueError("source_ref requires source_resume_version_id")

        version_fields = (
            self.scope_key,
            self.update_id,
            self.content_digest,
            self.revision,
            self.valid_from,
        )
        if any(item is None for item in version_fields) != all(
            item is None for item in version_fields
        ):
            raise ValueError(
                "scope_key, update_id, content_digest, revision, and valid_from "
                "must either all be set or all be null"
            )
        if self.verification_status == "confirmed" and self.revision is None:
            raise ValueError("confirmed evidence requires a version binding")
        if self.verification_status != "confirmed" and self.revision is not None:
            raise ValueError("only confirmed evidence may have a version binding")
        if self.revision == 1 and self.supersedes_id is not None:
            raise ValueError("revision 1 cannot supersede another evidence row")
        if self.revision is not None and self.revision > 1 and self.supersedes_id is None:
            raise ValueError("later revisions require a predecessor")
        if (self.superseded_at is None) != (self.superseded_by is None):
            raise ValueError(
                "superseded_at and superseded_by must either both be set or both be null"
            )
        if self.mutation_id is None and (
            self.supersedes_id is not None or self.rolled_back_at is not None
        ):
            raise ValueError("corrected or rolled-back evidence requires a mutation_id")
        return self

    @property
    def is_current(self) -> bool:
        return (
            self.verification_status == "confirmed"
            and self.superseded_by is None
            and self.rolled_back_at is None
        )


class CareerEvidenceEvent(CareerHistoryContract):
    id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    career_evidence_id: str = Field(min_length=1)

    event_type: Literal[
        "created",
        "confirmed",
        "rejected",
        "superseded",
        "corrected",
        "rolled_back",
        "restored",
    ]
    previous_status: Literal[
        "pending",
        "confirmed",
        "rejected",
    ] | None = None
    new_status: Literal[
        "pending",
        "confirmed",
        "rejected",
    ]
    actor_type: Literal[
        "user",
        "agent",
        "system",
    ]
    reason: str | None = Field(default=None, min_length=1)
    mutation_id: str | None = Field(
        default=None,
        pattern=r"^career_evidence_mutation_[a-f0-9]{32}$",
    )
    related_evidence_id: str | None = Field(default=None, min_length=1)
    occurred_at: datetime

    @model_validator(mode="after")
    def validate_transition(self) -> CareerEvidenceEvent:
        expected_transitions = {
            "created": (None, "pending"),
            "confirmed": ("pending", "confirmed"),
            "rejected": ("pending", "rejected"),
            "superseded": ("confirmed", "confirmed"),
            "corrected": ("confirmed", "confirmed"),
            "rolled_back": ("confirmed", "confirmed"),
            "restored": ("confirmed", "confirmed"),
        }
        if (self.previous_status, self.new_status) != expected_transitions[
            self.event_type
        ]:
            raise ValueError(f"invalid status transition for {self.event_type}")

        if self.event_type in {"confirmed", "rejected"} and self.actor_type != "user":
            raise ValueError(f"{self.event_type} event requires user actor")
        lineage_events = {"superseded", "corrected", "rolled_back", "restored"}
        if self.event_type in lineage_events and (
            self.mutation_id is None or self.related_evidence_id is None
        ):
            raise ValueError(
                f"{self.event_type} event requires mutation and related evidence ids"
            )

        return self


class CareerEvidencePreimage(CareerHistoryContract):
    """Application-visible active mapping captured before a correction."""

    scope_key: str = Field(
        pattern=r"^career_evidence/[A-Za-z0-9_.:-]+/claim$"
    )
    active_evidence_id: str = Field(min_length=1)
    active_revision: int = Field(ge=1)


class CareerEvidenceMutationSnapshot(CareerHistoryContract):
    id: str = Field(pattern=r"^career_evidence_mutation_[a-f0-9]{32}$")
    user_id: str = Field(min_length=1)
    scope_key: str = Field(
        pattern=r"^career_evidence/[A-Za-z0-9_.:-]+/claim$"
    )
    mutation_type: Literal["correction"]
    status: Literal["applied", "rolled_back"]
    preimage: CareerEvidencePreimage
    replacement_evidence_id: str = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=2000)
    created_at: datetime
    rolled_back_at: datetime | None = None

    @model_validator(mode="after")
    def rollback_time_matches_status(self) -> "CareerEvidenceMutationSnapshot":
        if (self.status == "rolled_back") != (self.rolled_back_at is not None):
            raise ValueError("rolled_back snapshots require rolled_back_at")
        return self


class CareerEvidenceCorrection(CareerHistoryContract):
    previous: CareerEvidence
    current: CareerEvidence
    snapshot: CareerEvidenceMutationSnapshot


class CareerEvidenceInvariantViolation(CareerHistoryContract):
    code: Literal[
        "version_binding",
        "pointer_target_missing",
        "pointer_not_reciprocal",
        "scope_mismatch",
        "revision_order",
        "active_count",
        "event_replay",
        "snapshot_binding",
    ]
    message: str = Field(min_length=1)
    scope_key: str | None = None
    evidence_ids: tuple[str, ...] = ()
    mutation_id: str | None = None


class CareerEvidenceInvariantReport(CareerHistoryContract):
    user_id: str | None = None
    checked_at: datetime
    violations: tuple[CareerEvidenceInvariantViolation, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.violations
