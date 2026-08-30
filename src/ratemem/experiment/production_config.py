"""Strict configuration for real SANA/RateMem engineering training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_PATH_TYPE = type(Path())


def _repository_path(value: object) -> Path:
    if type(value) is str:
        pure = PurePosixPath(value)
        if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
            raise ValueError("configuration paths must be confined repository paths")
        return Path(*pure.parts)
    if type(value) is _PATH_TYPE:
        return value
    raise TypeError("configuration path must be an exact Path or POSIX string")


class ProductionExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["memx-ratemem-training-v1"]
    profile: Literal["sana-ratemem"]
    publication_eligible: Literal[False]
    seed: Literal[17, 29, 43]
    max_steps: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    gradient_accumulation: Literal[1]
    learning_rate: float = Field(gt=0, allow_inf_nan=False)
    checkpoint_every: int = Field(gt=0)
    validation_batches: int = Field(gt=0)
    precision: Literal["bf16"]
    activation_checkpointing: Literal[True]
    shuffle_buffer: int = Field(gt=0)
    active_handle_slots: int = Field(gt=0)
    memory_budget_bytes: int = Field(gt=0)
    dataset_manifest: Path
    sana_config: Path
    method_policy: Path

    @field_validator("dataset_manifest", "sana_config", "method_policy", mode="before")
    @classmethod
    def path_from_yaml(cls, value: object) -> Path:
        return _repository_path(value)

    @model_validator(mode="after")
    def validate_training_bounds(self) -> ProductionExperimentConfig:
        if self.checkpoint_every > self.max_steps:
            raise ValueError("checkpoint_every cannot exceed max_steps")
        if self.active_handle_slots > 1024:
            raise ValueError("active_handle_slots exceeds the bounded training state")
        return self

    @classmethod
    def load(cls, path: Path) -> ProductionExperimentConfig:
        if type(path) is not _PATH_TYPE:
            raise TypeError("production experiment config path must be an exact Path")
        try:
            payload: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ValueError(f"production experiment config is unreadable: {path}") from error
        if type(payload) is not dict:
            raise TypeError("production experiment config root must be an exact mapping")
        return cls.model_validate(payload)

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


__all__ = ["ProductionExperimentConfig"]
