from __future__ import annotations

import os
from typing import Mapping

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


class JobDiscoveryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_search_attempts: int = Field(default=3, ge=1)
    max_search_results: int = Field(default=20, ge=1)
    max_detail_fetches: int = Field(default=10, ge=1)
    max_parallel_details: int = Field(default=1, ge=1, le=3)
    wall_clock_seconds: int = Field(default=300, ge=1)
    retry_limit: int = Field(default=1, ge=0)
    max_detail_attempts: int = Field(default=3, ge=1, le=3)

    @classmethod
    def from_env(cls, *, environ: Mapping[str, str] | None = None, prefix: str = "JOB_DISCOVERY") -> "JobDiscoveryPolicy":
        if environ is None:
            load_dotenv()
            environ = os.environ
        values = {}
        mapping = {
            "max_search_attempts": "MAX_SEARCH_ATTEMPTS",
            "max_search_results": "MAX_SEARCH_RESULTS",
            "max_detail_fetches": "MAX_DETAIL_FETCHES",
            "max_parallel_details": "MAX_PARALLEL_DETAILS",
            "wall_clock_seconds": "WALL_CLOCK_SECONDS",
            "retry_limit": "RETRY_LIMIT",
            "max_detail_attempts": "MAX_DETAIL_ATTEMPTS",
        }
        for field, suffix in mapping.items():
            value = environ.get(f"{prefix}_{suffix}", "").strip()
            if value:
                values[field] = int(value)
        return cls(**values)
