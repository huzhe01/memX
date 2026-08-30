"""Immutable source inventories and auditable scientific dataset locks."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal, TypeAlias

import yaml  # type: ignore[import-untyped]
from pydantic import (
    AnyUrl,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from ratemem.evaluation.canonical import (
    require_immutable_value,
    semantic_sha256,
    write_text_atomic,
    write_yaml_atomic,
)
from ratemem.evaluation.types import Sha256

InventoryMode: TypeAlias = Literal["synthetic", "scientific"]
SourceRole: TypeAlias = Literal[
    "engineering_pilot",
    "meta_training",
    "multi_image_training",
    "primary_one_shot_evaluation",
    "multi_shot_evaluation",
    "controlled_post_checkpoint_evaluation",
    "historical_stress",
]
PoolKind: TypeAlias = Literal["support", "query", "caption", "mask", "prompt"]
EvaluationPoolSemantics: TypeAlias = Literal[
    "held_out_query",
    "reference_prompt_only",
    "training_pairs",
    "historical_stress",
]
DatasetSplit: TypeAlias = Literal["train", "validation", "final_test"]
SupportShot: TypeAlias = Literal[1, 3, 5]

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)
_SPDX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+-]{0,63}$")
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_RECORD_ID = re.compile(r"^[a-z0-9][a-z0-9._:/-]{0,254}$")


class DatasetLockError(ValueError):
    """Raised when a source inventory cannot be sealed without ambiguity."""


def _canonical_relative_path(value: str, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{field} must be non-empty canonical text")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value.endswith("/"):
        raise ValueError(f"{field} must be a canonical relative path")
    return value


class LicenseAttestation(BaseModel):
    model_config = _MODEL_CONFIG

    spdx_id: str
    license_url: AnyUrl
    verified_by: str
    verified_at_utc: AwareDatetime
    research_use_allowed: Literal[True]
    redistribution_allowed: bool

    @field_validator("spdx_id")
    @classmethod
    def validate_spdx(cls, value: str) -> str:
        if _SPDX.fullmatch(value) is None:
            raise ValueError("spdx_id must be a canonical SPDX identifier")
        return value

    @field_validator("verified_by")
    @classmethod
    def validate_verifier(cls, value: str) -> str:
        return require_immutable_value("verified_by", value)


class PoolLock(BaseModel):
    model_config = _MODEL_CONFIG

    kind: PoolKind
    manifest_path: str
    sha256: Sha256
    record_count: PositiveInt
    concept_count: PositiveInt
    record_ids: tuple[str, ...]

    @field_validator("manifest_path")
    @classmethod
    def validate_manifest_path(cls, value: str) -> str:
        return _canonical_relative_path(value, "pool manifest_path")

    @field_validator("record_ids")
    @classmethod
    def validate_record_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) != len(set(value)):
            raise ValueError("pool record_ids must be non-empty and unique")
        if any(_RECORD_ID.fullmatch(record_id) is None for record_id in value):
            raise ValueError("pool record_ids must be canonical identifiers")
        return value

    @model_validator(mode="after")
    def validate_counts(self) -> PoolLock:
        if self.record_count != len(self.record_ids):
            raise ValueError("pool record_count must equal record_ids length")
        if self.concept_count > self.record_count:
            raise ValueError("pool concept_count cannot exceed record_count")
        return self


class SourceInventoryEntry(BaseModel):
    model_config = _MODEL_CONFIG

    source_id: str
    role: SourceRole
    upstream_uri: AnyUrl
    immutable_revision: str
    provenance_uri: AnyUrl
    license: LicenseAttestation
    concept_unit: str
    concept_tokens: tuple[str, ...]
    concept_count: PositiveInt
    image_count: PositiveInt
    pair_count: NonNegativeInt
    image_width_min: PositiveInt
    image_width_median: PositiveFloat
    image_width_max: PositiveInt
    image_height_min: PositiveInt
    image_height_median: PositiveFloat
    image_height_max: PositiveInt
    caption_count: NonNegativeInt
    mask_count: NonNegativeInt
    minimum_distinct_images_per_concept: PositiveInt
    assigned_splits: tuple[DatasetSplit, ...]
    pools: tuple[PoolLock, ...]
    allowed_uses: tuple[str, ...]
    pretraining_contamination: dict[str, str]
    evaluation_pool_semantics: EvaluationPoolSemantics
    limitations: tuple[str, ...]

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        if _IDENTIFIER.fullmatch(value) is None:
            raise ValueError("source_id must be a canonical identifier")
        return value

    @field_validator("concept_unit")
    @classmethod
    def validate_concept_unit(cls, value: str) -> str:
        if _IDENTIFIER.fullmatch(value) is None:
            raise ValueError("concept_unit must be a canonical identifier")
        return value

    @field_validator("assigned_splits", "allowed_uses", "limitations")
    @classmethod
    def validate_nonempty_unique_text(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) != len(set(value)):
            raise ValueError("record sequence must be non-empty and unique")
        if any(type(item) is not str or not item or item != item.strip() for item in value):
            raise ValueError("record sequence values must be canonical text")
        return value

    @model_validator(mode="after")
    def validate_inventory_counts(self) -> SourceInventoryEntry:
        if len(self.concept_tokens) != self.concept_count:
            raise ValueError("concept_count must equal concept_tokens length")
        if len(self.concept_tokens) != len(set(self.concept_tokens)):
            raise ValueError("concept_tokens must be unique within a source")
        if not (
            self.image_width_min <= self.image_width_median <= self.image_width_max
            and self.image_height_min <= self.image_height_median <= self.image_height_max
        ):
            raise ValueError("image dimension statistics are inconsistent")
        if self.minimum_distinct_images_per_concept > self.image_count:
            raise ValueError("minimum distinct images cannot exceed image_count")
        kinds = tuple(pool.kind for pool in self.pools)
        if len(kinds) != len(set(kinds)):
            raise ValueError("pool kinds must be unique within a source")
        return self


class SourceInventory(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"]
    mode: InventoryMode
    inventory_id: str
    observed_at_utc: AwareDatetime
    global_image_manifest_path: str
    global_image_manifest_sha256: Sha256
    duplicate_report_path: str
    duplicate_report_sha256: Sha256
    split_assignment_path: str
    split_assignment_sha256: Sha256
    sources: tuple[SourceInventoryEntry, ...]

    @field_validator("inventory_id")
    @classmethod
    def validate_inventory_id(cls, value: str) -> str:
        if _IDENTIFIER.fullmatch(value) is None:
            raise ValueError("inventory_id must be a canonical identifier")
        return value

    @field_validator(
        "global_image_manifest_path",
        "duplicate_report_path",
        "split_assignment_path",
    )
    @classmethod
    def validate_artifact_path(cls, value: str) -> str:
        return _canonical_relative_path(value, "inventory artifact path")

    @model_validator(mode="after")
    def validate_global_uniqueness(self) -> SourceInventory:
        source_ids = tuple(source.source_id for source in self.sources)
        if not source_ids or len(source_ids) != len(set(source_ids)):
            raise ValueError("inventory source_id values must be non-empty and unique")
        concepts = tuple(token for source in self.sources for token in source.concept_tokens)
        if len(concepts) != len(set(concepts)):
            raise ValueError("concept tokens must be globally source-disjoint")
        return self


class SourceLock(SourceInventoryEntry):
    eligible_support_shots: tuple[SupportShot, ...]


class DatasetLock(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"]
    lock_id: Sha256
    sealed_at_utc: AwareDatetime
    inventory_id: str
    inventory_mode: InventoryMode
    inventory_observed_at_utc: AwareDatetime
    global_image_manifest_path: str
    global_image_manifest_sha256: Sha256
    duplicate_report_path: str
    duplicate_report_sha256: Sha256
    split_assignment_path: str
    split_assignment_sha256: Sha256
    sources: tuple[SourceLock, ...]


class DatasetPolicy(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"]
    required_roles: dict[SourceRole, tuple[str, ...]]
    optional_roles: dict[SourceRole, tuple[str, ...]]
    anonymous_token_pattern: str
    forbid_real_identity_names: Literal[True]
    require_cluster_disjoint_splits: Literal[True]
    require_immutable_support_query_pools: Literal[True]
    require_license_attestation: Literal[True]
    require_pretraining_contamination_disclosure_for: tuple[str, ...]
    role_to_allowed_splits: dict[SourceRole, tuple[DatasetSplit, ...]]
    within_training_source_concept_split: dict[Literal["train", "validation"], float]
    scientific_modes: tuple[Literal["calibration", "validation", "final_test"], ...]

    @model_validator(mode="after")
    def validate_policy(self) -> DatasetPolicy:
        if set(self.required_roles) & set(self.optional_roles):
            raise ValueError("required and optional role maps must be disjoint")
        if not self.require_pretraining_contamination_disclosure_for:
            raise ValueError("contamination disclosure list must not be empty")
        split = self.within_training_source_concept_split
        if set(split) != {"train", "validation"} or abs(sum(split.values()) - 1.0) > 1e-9:
            raise ValueError("training concept split fractions must sum to one")
        re.compile(self.anonymous_token_pattern)
        return self


def load_inventory(path: Path) -> SourceInventory:
    """Load one fully resolved JSON source inventory."""

    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetLockError(f"cannot read source inventory: {error}") from error
    try:
        return SourceInventory.model_validate(raw)
    except ValueError as error:
        raise DatasetLockError(f"invalid source inventory: {error}") from error


def _load_policy(path: Path) -> DatasetPolicy:
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        return DatasetPolicy.model_validate(raw)
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise DatasetLockError(f"invalid dataset policy: {error}") from error


def _eligible_shots(minimum_distinct_images: int) -> tuple[SupportShot, ...]:
    candidates: tuple[SupportShot, ...] = (1, 3, 5)
    return tuple(shot for shot in candidates if shot <= minimum_distinct_images)


def _validate_source(source: SourceInventoryEntry, policy: DatasetPolicy) -> None:
    try:
        require_immutable_value("immutable_revision", source.immutable_revision)
    except (TypeError, ValueError) as error:
        raise DatasetLockError(f"{source.source_id} immutable_revision is invalid") from error

    token_pattern = re.compile(policy.anonymous_token_pattern)
    if any(token_pattern.fullmatch(token) is None for token in source.concept_tokens):
        raise DatasetLockError(f"{source.source_id} requires an anonymous concept token")

    allowed_splits = set(policy.role_to_allowed_splits[source.role])
    if not set(source.assigned_splits) <= allowed_splits:
        raise DatasetLockError(f"{source.source_id} uses a split forbidden for {source.role}")

    required_disclosures = set(policy.require_pretraining_contamination_disclosure_for)
    if not required_disclosures <= set(source.pretraining_contamination):
        missing = sorted(required_disclosures - set(source.pretraining_contamination))
        raise DatasetLockError(
            f"{source.source_id} is missing contamination disclosures: {', '.join(missing)}"
        )
    for component in sorted(required_disclosures):
        try:
            require_immutable_value(
                f"{source.source_id}.{component}",
                source.pretraining_contamination[component],
            )
        except (TypeError, ValueError) as error:
            raise DatasetLockError(
                f"{source.source_id} has an unresolved contamination disclosure"
            ) from error

    pools = {pool.kind: pool for pool in source.pools}
    if not {"support", "query"} <= set(pools):
        raise DatasetLockError(f"{source.source_id} requires immutable support and query pools")
    support_ids = set(pools["support"].record_ids)
    query_ids = set(pools["query"].record_ids)
    if support_ids & query_ids:
        raise DatasetLockError(f"{source.source_id} support/query records must be disjoint")

    if (
        source.source_id == "dreambench_plus_plus"
        and source.evaluation_pool_semantics != "reference_prompt_only"
    ):
        raise DatasetLockError("dreambench_plus_plus must use reference_prompt_only semantics")
    if source.source_id == "subjects200k" and not any(
        "one-shot" in limitation.lower() for limitation in source.limitations
    ):
        raise DatasetLockError("subjects200k must disclose its mostly-one-shot limitation")


def seal_dataset_lock(
    inventory: SourceInventory,
    *,
    policy_path: Path,
    mode: InventoryMode,
) -> DatasetLock:
    """Validate and seal an immutable, semantically hashed dataset inventory."""

    if type(inventory) is not SourceInventory:
        raise TypeError("inventory must be an exact SourceInventory")
    if mode not in {"synthetic", "scientific"}:
        raise ValueError("mode must be synthetic or scientific")
    if inventory.mode != mode:
        if mode == "scientific" and inventory.mode == "synthetic":
            raise DatasetLockError("scientific mode refuses a synthetic inventory")
        raise DatasetLockError("requested mode does not match the inventory mode")

    policy = _load_policy(policy_path)
    by_role: dict[str, set[str]] = {}
    for source in inventory.sources:
        by_role.setdefault(source.role, set()).add(source.source_id)
        _validate_source(source, policy)
    for role, required_ids in policy.required_roles.items():
        missing = set(required_ids) - by_role.get(role, set())
        if missing:
            raise DatasetLockError(
                f"required role {role} is missing source(s): {', '.join(sorted(missing))}"
            )

    locked_sources = tuple(
        SourceLock.model_validate(
            source.model_dump(mode="json")
            | {
                "pools": [
                    pool.model_dump(mode="json")
                    for pool in sorted(
                        source.pools,
                        key=lambda item: (item.kind, item.manifest_path),
                    )
                ],
                "eligible_support_shots": list(
                    _eligible_shots(source.minimum_distinct_images_per_concept)
                ),
            }
        )
        for source in sorted(inventory.sources, key=lambda item: item.source_id)
    )
    provisional = DatasetLock(
        schema_version="1.0",
        lock_id="0" * 64,
        sealed_at_utc=datetime.now(UTC),
        inventory_id=inventory.inventory_id,
        inventory_mode=inventory.mode,
        inventory_observed_at_utc=inventory.observed_at_utc,
        global_image_manifest_path=inventory.global_image_manifest_path,
        global_image_manifest_sha256=inventory.global_image_manifest_sha256,
        duplicate_report_path=inventory.duplicate_report_path,
        duplicate_report_sha256=inventory.duplicate_report_sha256,
        split_assignment_path=inventory.split_assignment_path,
        split_assignment_sha256=inventory.split_assignment_sha256,
        sources=locked_sources,
    )
    return provisional.model_copy(
        update={"lock_id": semantic_sha256(provisional.model_dump(mode="json"))}
    )


def render_data_card(lock: DatasetLock) -> str:
    """Render a deterministic disclosure card from the sealed record only."""

    lines = [
        "# RateMem-DiT scientific data card",
        "",
        f"- Dataset lock: `{lock.lock_id}`",
        f"- Inventory: `{lock.inventory_id}` (`{lock.inventory_mode}`)",
        f"- Inventory observed: `{lock.inventory_observed_at_utc.isoformat()}`",
        "",
        "## Sources and pool semantics",
        "",
        "| Source | Role | Concepts | Images | Semantics | Eligible shots |",
        "|---|---|---:|---:|---|---|",
    ]
    for source in lock.sources:
        shots = ", ".join(str(shot) for shot in source.eligible_support_shots)
        lines.append(
            f"| `{source.source_id}` | `{source.role}` | {source.concept_count} | "
            f"{source.image_count} | `{source.evaluation_pool_semantics}` | {shots} |"
        )
    lines.extend(
        [
            "",
            "## Licenses and allowed uses",
            "",
        ]
    )
    for source in lock.sources:
        uses = ", ".join(source.allowed_uses)
        lines.append(
            f"- `{source.source_id}`: `{source.license.spdx_id}`; allowed uses: {uses}; "
            f"redistribution: {str(source.license.redistribution_allowed).lower()}."
        )
    lines.extend(
        [
            "",
            "## Duplicate, derivative, and contamination audit",
            "",
            f"- Global image manifest: `{lock.global_image_manifest_sha256}`",
            f"- Duplicate report: `{lock.duplicate_report_sha256}`",
            f"- Split assignments: `{lock.split_assignment_sha256}`",
            "",
        ]
    )
    for source in lock.sources:
        disclosures = "; ".join(
            f"{name}={value}"
            for name, value in sorted(source.pretraining_contamination.items())
        )
        lines.append(f"- `{source.source_id}`: {disclosures}.")
    lines.extend(["", "## Limitations", ""])
    for source in lock.sources:
        for limitation in source.limitations:
            lines.append(f"- `{source.source_id}`: {limitation}")
    return "\n".join(lines) + "\n"


def write_dataset_lock_and_card(
    lock: DatasetLock,
    lock_path: Path,
    card_path: Path,
) -> None:
    """Publish the schema-ready YAML lock before its deterministic data card."""

    write_yaml_atomic(lock_path, lock.model_dump(mode="json"))
    write_text_atomic(card_path, render_data_card(lock))


__all__ = [
    "DatasetLock",
    "DatasetLockError",
    "LicenseAttestation",
    "PoolLock",
    "SourceInventory",
    "SourceInventoryEntry",
    "SourceLock",
    "load_inventory",
    "render_data_card",
    "seal_dataset_lock",
    "write_dataset_lock_and_card",
]
