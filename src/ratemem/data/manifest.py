from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

_REVISION = re.compile(r"[0-9a-f]{40}")
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._/-]{0,254}")
_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
_SPDX = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]{0,63}")
_PATH_TYPE = type(Path())


def _tuple_from_yaml(value: object, name: str) -> tuple[object, ...]:
    if type(value) is list:
        return tuple(value)
    if type(value) is tuple:
        return value
    raise TypeError(f"{name} must be a YAML sequence")


class DatasetSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["generated", "huggingface", "https"]
    identifier: str

    @field_validator("identifier")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        if _IDENTIFIER.fullmatch(value) is None or ".." in value.split("/"):
            raise ValueError("source identifier must be canonical path-safe text")
        return value


class DatasetSplit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: Literal["train", "validation", "test"]
    concepts: tuple[str, ...]

    @field_validator("concepts", mode="before")
    @classmethod
    def tuple_concepts(cls, value: object) -> tuple[object, ...]:
        return _tuple_from_yaml(value, "split concepts")

    @field_validator("concepts")
    @classmethod
    def validate_concepts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("every dataset split must contain at least one concept")
        if len(value) != len(set(value)):
            raise ValueError("concept identifiers must be unique within a split")
        if any(_NAME.fullmatch(concept) is None for concept in value):
            raise ValueError("concept identifiers must be canonical path-safe names")
        return value


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["memx-dataset-v1"]
    name: str
    revision: str
    license_spdx: str
    profile: Literal["smoke", "training", "evaluation"]
    source: DatasetSource
    splits: tuple[DatasetSplit, ...]

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if _NAME.fullmatch(value) is None:
            raise ValueError("dataset name must be a canonical path-safe name")
        return value

    @field_validator("revision")
    @classmethod
    def validate_revision(cls, value: str) -> str:
        if _REVISION.fullmatch(value) is None:
            raise ValueError("dataset revision must be an immutable 40-character lowercase hash")
        return value

    @field_validator("license_spdx")
    @classmethod
    def validate_license(cls, value: str) -> str:
        if _SPDX.fullmatch(value) is None:
            raise ValueError("license_spdx must be one canonical SPDX identifier")
        return value

    @field_validator("splits", mode="before")
    @classmethod
    def tuple_splits(cls, value: object) -> tuple[object, ...]:
        return _tuple_from_yaml(value, "dataset splits")

    @model_validator(mode="after")
    def validate_split_partition(self) -> DatasetManifest:
        names = tuple(split.name for split in self.splits)
        if names != ("train", "validation", "test"):
            raise ValueError("dataset splits must be ordered train, validation, test")
        concepts = tuple(concept for split in self.splits for concept in split.concepts)
        if len(concepts) != len(set(concepts)):
            raise ValueError("dataset concept identities must be globally disjoint")
        return self

    @classmethod
    def load(cls, path: Path) -> DatasetManifest:
        if type(path) is not _PATH_TYPE:
            raise TypeError("dataset manifest path must be an exact Path")
        try:
            payload: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ValueError(f"dataset manifest is unreadable: {path}") from error
        if type(payload) is not dict:
            raise TypeError("dataset manifest root must be an exact mapping")
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

    def concepts_for(self, name: str) -> tuple[str, ...]:
        if type(name) is not str:
            raise TypeError("split name must be an exact str")
        for split in self.splits:
            if split.name == name:
                return split.concepts
        raise KeyError(name)
