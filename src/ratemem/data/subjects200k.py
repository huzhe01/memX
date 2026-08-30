"""Pinned Subjects200K snapshot download and verification."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

import yaml  # type: ignore[import-untyped]
from filelock import FileLock
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, ConfigDict, PositiveInt, field_validator, model_validator

from ratemem.evaluation.canonical import canonical_json_bytes, file_sha256, write_json_atomic
from ratemem.evaluation.types import GitCommit, Sha256

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True)
_PATH_TYPE = type(Path())
_REPOSITORY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_SNAPSHOT_SCHEMA = "memx-subjects200k-prepared-v1"


def _safe_relative(value: str, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty canonical text")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"{name} must be a confined relative POSIX path")
    return value


class SubjectsShard(BaseModel):
    model_config = _MODEL_CONFIG

    path: str
    sha256: Sha256
    size_bytes: PositiveInt

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        normalized = _safe_relative(value, "Subjects200K shard path")
        if not re.fullmatch(r"data/train-[0-9]{5}-of-[0-9]{5}\.parquet", normalized):
            raise ValueError("Subjects200K shard filename changed")
        return normalized


class CompositePairPolicy(BaseModel):
    model_config = _MODEL_CONFIG

    mode: Literal["RGB"]
    width: Literal[1056]
    height: Literal[528]
    image_size: Literal[512]
    support_crop: tuple[Literal[8], Literal[8], Literal[520], Literal[520]]
    query_crop: tuple[Literal[528], Literal[8], Literal[1040], Literal[520]]
    concept_field: Literal["item"]
    support_prompt_field: Literal["description_0"]
    query_prompt_field: Literal["description_1"]
    validity_field: Literal["description_valid"]

    @field_validator("support_crop", "query_crop", mode="before")
    @classmethod
    def tuple_crop(cls, value: object) -> object:
        return tuple(value) if type(value) is list else value


class ConceptPartitionPolicy(BaseModel):
    model_config = _MODEL_CONFIG

    algorithm: Literal["sha256_concept_identity_mod_10000"]
    seed: int
    train_upper_bound: Literal[9000]
    validation_upper_bound: Literal[10000]

    @field_validator("seed")
    @classmethod
    def validate_seed(cls, value: int) -> int:
        if type(value) is not int or not 0 <= value < 2**63:
            raise ValueError("partition seed must be a nonnegative signed 64-bit integer")
        return value


class Subjects200KManifest(BaseModel):
    """Repository-pinned raw snapshot and deterministic train/validation semantics."""

    model_config = _MODEL_CONFIG

    schema_version: Literal["memx-subjects200k-snapshot-v1"]
    name: Literal["subjects200k"]
    profile: Literal["training"]
    repository_id: str
    revision: GitCommit
    config_name: Literal["default"]
    split: Literal["train"]
    license_spdx: Literal["Apache-2.0"]
    expected_total_bytes: PositiveInt
    shards: tuple[SubjectsShard, ...]
    composite_pair: CompositePairPolicy
    partition: ConceptPartitionPolicy

    @field_validator("shards", mode="before")
    @classmethod
    def tuple_shards(cls, value: object) -> object:
        return tuple(value) if type(value) is list else value

    @field_validator("repository_id")
    @classmethod
    def validate_repository_id(cls, value: str) -> str:
        if _REPOSITORY_ID.fullmatch(value) is None:
            raise ValueError("repository_id must be one canonical Hugging Face repository id")
        return value

    @model_validator(mode="after")
    def validate_snapshot(self) -> Subjects200KManifest:
        if len(self.shards) != 32:
            raise ValueError("Subjects200K snapshot must bind exactly 32 parquet shards")
        expected_paths = tuple(
            f"data/train-{index:05d}-of-00032.parquet" for index in range(32)
        )
        if tuple(shard.path for shard in self.shards) != expected_paths:
            raise ValueError("Subjects200K shards must be complete and canonically ordered")
        if sum(shard.size_bytes for shard in self.shards) != self.expected_total_bytes:
            raise ValueError("Subjects200K expected_total_bytes differs from the shard inventory")
        return self

    @classmethod
    def load(cls, path: Path) -> Subjects200KManifest:
        if type(path) is not _PATH_TYPE:
            raise TypeError("Subjects200K manifest path must be an exact Path")
        try:
            payload: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ValueError(f"Subjects200K manifest is unreadable: {path}") from error
        if type(payload) is not dict:
            raise TypeError("Subjects200K manifest root must be an exact mapping")
        return cls.model_validate(payload)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(self.model_dump(mode="json"))
        ).hexdigest()


class PreparedSubjects200KSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    root: Path
    manifest_sha256: Sha256
    repository_id: str
    revision: GitCommit
    total_bytes: PositiveInt
    shard_count: PositiveInt

    @classmethod
    def load(
        cls,
        root: Path,
        manifest: Subjects200KManifest,
    ) -> PreparedSubjects200KSnapshot:
        if type(root) is not _PATH_TYPE or root.is_symlink():
            raise ValueError("Subjects200K snapshot root must be a real directory")
        metadata = root.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("Subjects200K snapshot root must be a real directory")
        try:
            raw: Any = json.loads((root / "snapshot-manifest.json").read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Subjects200K snapshot manifest is unreadable") from error
        expected_fields = {
            "schema_version",
            "dataset_manifest_sha256",
            "repository_id",
            "revision",
            "total_bytes",
            "shards",
        }
        if type(raw) is not dict or set(raw) != expected_fields:
            raise ValueError("Subjects200K snapshot manifest fields changed")
        payload = cast(dict[str, object], raw)
        if (
            payload["schema_version"] != _SNAPSHOT_SCHEMA
            or payload["dataset_manifest_sha256"] != manifest.sha256
            or payload["repository_id"] != manifest.repository_id
            or payload["revision"] != manifest.revision
            or payload["total_bytes"] != manifest.expected_total_bytes
        ):
            raise ValueError("Subjects200K snapshot identity differs from its pinned manifest")
        observed_shards = payload["shards"]
        expected_shards = [shard.model_dump(mode="json") for shard in manifest.shards]
        if observed_shards != expected_shards:
            raise ValueError("Subjects200K snapshot shard inventory changed")
        for shard in manifest.shards:
            path = root / shard.path
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Subjects200K shard is missing or unsafe: {shard.path}")
            if path.stat().st_size != shard.size_bytes or file_sha256(path) != shard.sha256:
                raise ValueError(f"Subjects200K shard hash or size changed: {shard.path}")
        return cls(
            root=root,
            manifest_sha256=manifest.sha256,
            repository_id=manifest.repository_id,
            revision=manifest.revision,
            total_bytes=manifest.expected_total_bytes,
            shard_count=len(manifest.shards),
        )


def _mirror_endpoint() -> str | None:
    value = os.environ.get("MEMX_HF_ENDPOINT")
    if value is None:
        return None
    if value != value.strip() or not value.startswith(("https://", "http://")):
        raise ValueError("MEMX_HF_ENDPOINT must be one canonical HTTP(S) endpoint")
    return value.rstrip("/")


def prepare_subjects200k_snapshot(
    manifest: Subjects200KManifest,
    data_root: Path,
    *,
    offline: bool = False,
) -> PreparedSubjects200KSnapshot:
    """Download every pinned shard, verify bytes, and publish one atomic snapshot."""

    if type(manifest) is not Subjects200KManifest:
        raise TypeError("manifest must be an exact Subjects200KManifest")
    if type(data_root) is not _PATH_TYPE:
        raise TypeError("data root must be an exact Path")
    if type(offline) is not bool:
        raise TypeError("offline must be an exact bool")
    data_root.mkdir(parents=True, exist_ok=True)
    if data_root.is_symlink():
        raise ValueError("data root cannot be a symlink")
    destination = data_root / f"{manifest.name}-{manifest.sha256[:16]}"
    lock_root = data_root / ".locks"
    lock_root.mkdir(exist_ok=True)
    lock = FileLock(lock_root / f"{manifest.name}-{manifest.sha256}.lock")
    with lock:
        if destination.exists() or destination.is_symlink():
            return PreparedSubjects200KSnapshot.load(destination, manifest)
        staging = data_root / f".staging-{manifest.name}-{manifest.sha256[:16]}"
        if staging.is_symlink():
            raise ValueError("Subjects200K staging path cannot be a symlink")
        staging.mkdir(mode=0o700, exist_ok=True)
        endpoint = _mirror_endpoint()
        for shard in manifest.shards:
            downloaded = hf_hub_download(
                repo_id=manifest.repository_id,
                filename=shard.path,
                repo_type="dataset",
                revision=manifest.revision,
                local_dir=staging,
                local_files_only=offline,
                endpoint=endpoint,
            )
            path = Path(downloaded)
            if path.resolve() != (staging / shard.path).resolve():
                raise RuntimeError("Hugging Face downloader returned an unexpected shard path")
            if path.is_symlink() or path.stat().st_size != shard.size_bytes:
                raise ValueError(f"Subjects200K shard size changed: {shard.path}")
            if file_sha256(path) != shard.sha256:
                raise ValueError(f"Subjects200K shard hash changed: {shard.path}")
        snapshot_payload = {
            "schema_version": _SNAPSHOT_SCHEMA,
            "dataset_manifest_sha256": manifest.sha256,
            "repository_id": manifest.repository_id,
            "revision": manifest.revision,
            "total_bytes": manifest.expected_total_bytes,
            "shards": [shard.model_dump(mode="json") for shard in manifest.shards],
        }
        write_json_atomic(staging / "snapshot-manifest.json", snapshot_payload)
        os.replace(staging, destination)
    return PreparedSubjects200KSnapshot.load(destination, manifest)


__all__ = [
    "CompositePairPolicy",
    "ConceptPartitionPolicy",
    "PreparedSubjects200KSnapshot",
    "Subjects200KManifest",
    "SubjectsShard",
    "prepare_subjects200k_snapshot",
]
