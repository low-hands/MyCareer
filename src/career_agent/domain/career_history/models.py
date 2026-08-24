from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CareerHistoryContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


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

    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def validate_source(self) -> CareerEvidence:
        if self.origin == "resume_extraction" and (
            self.source_resume_version_id is None or self.source_locator is None
        ):
            raise ValueError(
                "resume extraction requires source_resume_version_id and source_locator"
            )

        if self.source_locator is not None and self.source_resume_version_id is None:
            raise ValueError("source_locator requires source_resume_version_id")

        return self
