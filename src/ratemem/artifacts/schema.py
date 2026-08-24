from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

_SECRET = re.compile(r"(?:ak|as)-[A-Za-z0-9_-]{20,}")


class AttemptManifest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
    )

    run_id: str
    git_revision: str
    config_hash: str
    status: Literal["passed", "failed", "interrupted"]
    notes: str = ""

    @field_validator("run_id", "git_revision", "config_hash", "notes")
    @classmethod
    def reject_credentials(cls, value: str) -> str:
        if _SECRET.search(value):
            raise ValueError("credential-shaped value is forbidden")
        return value
