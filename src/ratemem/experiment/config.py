from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_PATH_TYPE = type(Path())


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["memx-experiment-v1"]
    profile: Literal["smoke", "sana-ratemem"]
    seed: int = Field(ge=0)
    max_steps: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    gradient_accumulation: int = Field(gt=0)
    learning_rate: float = Field(gt=0, allow_inf_nan=False)
    checkpoint_every: int = Field(gt=0)
    precision: Literal["fp32", "bf16"]
    dataset_manifest: Path

    @field_validator("dataset_manifest", mode="before")
    @classmethod
    def path_from_yaml(cls, value: object) -> Path:
        if type(value) is str:
            pure = PurePosixPath(value)
            if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
                raise ValueError("dataset_manifest must be a confined repository path")
            return Path(*pure.parts)
        if type(value) is _PATH_TYPE:
            return value
        raise TypeError("dataset_manifest must be a path string or exact Path")

    @model_validator(mode="after")
    def validate_profile_precision(self) -> ExperimentConfig:
        if self.profile == "smoke" and self.precision != "fp32":
            raise ValueError("smoke profile requires fp32")
        if self.profile == "sana-ratemem" and self.precision != "bf16":
            raise ValueError("sana-ratemem profile requires bf16")
        return self

    @classmethod
    def load(cls, path: Path) -> ExperimentConfig:
        if type(path) is not _PATH_TYPE:
            raise TypeError("experiment config path must be an exact Path")
        try:
            payload: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ValueError(f"experiment config is unreadable: {path}") from error
        if type(payload) is not dict:
            raise TypeError("experiment config root must be an exact mapping")
        return cls.model_validate(payload)

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()
