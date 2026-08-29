from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Never, cast

from ratemem.adapters.sana_layout import (
    ATTENTION_KINDS,
    SANA_LAYOUT_VERSION,
    TARGET_MODULES,
    SanaAdapterLayout,
)

PILOT_OPTIMIZER_CLASS = "AdamW"


def _canonical_modal_budget_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "profile": "ratemem-pilot",
        "environment": "main",
        "workspace_budget_usd": "28.00",
        "internal_limit_usd": "27.00",
        "first_pilot_allocation_usd": "21.00",
        "setup_probe_allocation_usd": "2.00",
        "timing_probe_allocation_usd": "3.00",
        "held_in_pilot_allocation_usd": "16.00",
        "unallocated_safety_buffer_usd": "6.00",
        "attestation_max_age_seconds": 900,
        "gpu": "L40S",
        "gpu_count": 1,
        "cpu_cores": 4,
        "memory_gib": 32,
        "timeout_seconds": 7200,
        "startup_timeout_seconds": 1800,
        "storage_gib_bound": 24,
        "non_gpu_setup_allowance_usd": "2.00",
        "retries": 0,
        "max_containers": 1,
        "detached": False,
        "cache_volume": "ratemem-sana-cache",
        "artifact_volume": "ratemem-pilot-artifacts",
    }


def pilot_adamw_kwargs() -> dict[str, object]:
    """Return a fresh exact optimizer contract for construction boundaries."""
    return {
        "lr": 0.001,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0,
        "amsgrad": False,
        "maximize": False,
        "foreach": False,
        "capturable": False,
        "differentiable": False,
        "fused": False,
        "decoupled_weight_decay": True,
    }


def _canonical_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "sana": {
            "model_id": "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers",
            "revision": "b77948f2b4eed5c728e9b828ccff07f7427b43cc",
            "resolution": 1024,
            "latent_channels": 32,
            "latent_size": 32,
            "text_feature_dim": 2304,
            "max_sequence_length": 300,
            "dtype": "bfloat16",
        },
        "support_encoder": {
            "model_id": "facebook/dinov2-small",
            "revision": "ed25f3a31f01632728cabb09d1542f84ab7b0056",
            "feature_dim": 384,
        },
        "adapter": {
            "layout_version": "sana-qkv-v1",
            "num_blocks": 20,
            "attention_kinds": ["attn1", "attn2"],
            "target_modules": ["to_q", "to_k", "to_v"],
            "width": 2240,
            "rank": 4,
            "atom_count": 4,
            "projection_count": 120,
            "code_dim": 480,
            "atom_tensor_count": 240,
            "atom_parameter_count": 8_601_600,
        },
        "training": {
            "scheduler_class": "FlowMatchEulerDiscreteScheduler",
            "optimizer": {
                "class": "AdamW",
                "lr": 0.001,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0,
                "amsgrad": False,
                "maximize": False,
                "foreach": False,
                "capturable": False,
                "differentiable": False,
                "fused": False,
                "decoupled_weight_decay": True,
            },
            "num_train_timesteps": 1000,
            "flow_shift": 1.0,
            "use_dynamic_shifting": False,
            "timestep_sampling": "uniform",
            "prediction_target": "noise_minus_clean_latent",
            "mixed_precision": "bf16",
            "gradient_checkpointing": True,
            "max_support_images": 2,
            "query_passes_per_step": 1,
        },
    }


def _canonical_subjects_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "scope": "engineering_pilot_only",
        "publication_eligible": False,
        "dataset": {
            "dataset_id": "Yuanshi/Subjects200K",
            "revision": "0d1cf6536239888f1a8e218790649344810067bc",
            "config_name": "default",
            "split": "train",
            "source_file": "data/train-00000-of-00032.parquet",
            "source_file_sha256": (
                "3d696ccbdfc736961e75e5b7ce33adae40cd70ffb69cdc27020a25d643971903"
            ),
            "streaming": True,
            "row_indices": list(range(8)),
            "public": True,
            "gated": False,
            "license_spdx": "apache-2.0",
        },
        "semantics": {
            "held_in": True,
            "held_in_meaning": (
                "public_train_rows_engineering_smoke_not_scientific_holdout"
            ),
            "support_side": "left",
            "query_side": "right",
            "concept_field": "item",
            "support_prompt_field": "description_0",
            "query_prompt_field": "description_1",
        },
        "composite": {
            "mode": "RGB",
            "size": [1056, 528],
            "image_size": 512,
            "padding_pixels": 8,
            "left_crop": [8, 8, 520, 520],
            "right_crop": [528, 8, 1040, 520],
        },
        "feature_order": [
            "image",
            "collection",
            "quality_assessment",
            "description",
        ],
        "quality_field_order": [
            "compositeStructure",
            "objectConsistency",
            "imageQuality",
        ],
        "description_field_order": [
            "item",
            "description_0",
            "description_1",
            "category",
            "description_valid",
        ],
    }


SUBJECTS_PILOT_CANONICAL_SHA256 = hashlib.sha256(
    json.dumps(
        _canonical_subjects_payload(),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> Never:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _require_canonical_value(actual: object, expected: object, path: str) -> None:
    if type(actual) is not type(expected):
        raise ValueError(
            f"{path} must have exact canonical type {type(expected).__name__}"
        )
    if type(expected) is dict:
        actual_dict = cast(dict[str, object], actual)
        expected_dict = cast(dict[str, object], expected)
        if tuple(actual_dict) != tuple(expected_dict):
            raise ValueError(f"{path} object key order must be exactly canonical")
        for key, expected_child in expected_dict.items():
            _require_canonical_value(
                actual_dict[key], expected_child, f"{path}.{key}"
            )
        return
    if type(expected) is list:
        actual_list = cast(list[object], actual)
        expected_list = cast(list[object], expected)
        if len(actual_list) != len(expected_list):
            raise ValueError(f"{path} list length changed from the canonical contract")
        for index, (actual_child, expected_child) in enumerate(
            zip(actual_list, expected_list, strict=True)
        ):
            _require_canonical_value(
                actual_child, expected_child, f"{path}[{index}]"
            )
        return
    if actual != expected:
        raise ValueError(f"{path} changed from the exact canonical value")


@dataclass(frozen=True, slots=True)
class SanaPilotConfig:
    schema_version: str
    model_id: str
    revision: str
    resolution: int
    latent_channels: int
    latent_size: int
    text_feature_dim: int
    max_sequence_length: int
    dtype: str
    support_model_id: str
    support_revision: str
    support_feature_dim: int
    layout_version: str
    num_blocks: int
    attention_kinds: tuple[str, ...]
    target_modules: tuple[str, ...]
    width: int
    rank: int
    atom_count: int
    projection_count: int
    code_dim: int
    atom_tensor_count: int
    atom_parameter_count: int
    scheduler_class: str
    optimizer_class: str
    optimizer_lr: float
    optimizer_betas: tuple[float, float]
    optimizer_eps: float
    optimizer_weight_decay: float
    optimizer_amsgrad: bool
    optimizer_maximize: bool
    optimizer_foreach: bool
    optimizer_capturable: bool
    optimizer_differentiable: bool
    optimizer_fused: bool
    optimizer_decoupled_weight_decay: bool
    num_train_timesteps: int
    flow_shift: float
    use_dynamic_shifting: bool
    timestep_sampling: str
    prediction_target: str
    mixed_precision: str
    gradient_checkpointing: bool
    max_support_images: int
    query_passes_per_step: int

    def __post_init__(self) -> None:
        self.validate()

    @property
    def code_shape(self) -> tuple[int, int, int, int]:
        return (
            self.num_blocks,
            len(self.attention_kinds),
            len(self.target_modules),
            self.atom_count,
        )

    @property
    def optimizer_kwargs(self) -> dict[str, object]:
        configured = {
            "lr": self.optimizer_lr,
            "betas": self.optimizer_betas,
            "eps": self.optimizer_eps,
            "weight_decay": self.optimizer_weight_decay,
            "amsgrad": self.optimizer_amsgrad,
            "maximize": self.optimizer_maximize,
            "foreach": self.optimizer_foreach,
            "capturable": self.optimizer_capturable,
            "differentiable": self.optimizer_differentiable,
            "fused": self.optimizer_fused,
            "decoupled_weight_decay": self.optimizer_decoupled_weight_decay,
        }
        if self.optimizer_class != PILOT_OPTIMIZER_CLASS or configured != pilot_adamw_kwargs():
            raise RuntimeError("config optimizer diverged from the runtime pilot contract")
        return configured

    def validate(self) -> None:
        if type(self) is not SanaPilotConfig:
            raise TypeError("config must be an exact SanaPilotConfig")
        if type(self.attention_kinds) is not tuple:
            raise ValueError("config.attention_kinds must have exact canonical type tuple")
        if type(self.target_modules) is not tuple:
            raise ValueError("config.target_modules must have exact canonical type tuple")
        _require_canonical_value(
            self._as_payload(), _canonical_payload(), "config"
        )
        self._assert_derived_contract()

    def _as_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sana": {
                "model_id": self.model_id,
                "revision": self.revision,
                "resolution": self.resolution,
                "latent_channels": self.latent_channels,
                "latent_size": self.latent_size,
                "text_feature_dim": self.text_feature_dim,
                "max_sequence_length": self.max_sequence_length,
                "dtype": self.dtype,
            },
            "support_encoder": {
                "model_id": self.support_model_id,
                "revision": self.support_revision,
                "feature_dim": self.support_feature_dim,
            },
            "adapter": {
                "layout_version": self.layout_version,
                "num_blocks": self.num_blocks,
                "attention_kinds": list(self.attention_kinds),
                "target_modules": list(self.target_modules),
                "width": self.width,
                "rank": self.rank,
                "atom_count": self.atom_count,
                "projection_count": self.projection_count,
                "code_dim": self.code_dim,
                "atom_tensor_count": self.atom_tensor_count,
                "atom_parameter_count": self.atom_parameter_count,
            },
            "training": {
                "scheduler_class": self.scheduler_class,
                "optimizer": {
                    "class": self.optimizer_class,
                    "lr": self.optimizer_lr,
                    "betas": list(self.optimizer_betas),
                    "eps": self.optimizer_eps,
                    "weight_decay": self.optimizer_weight_decay,
                    "amsgrad": self.optimizer_amsgrad,
                    "maximize": self.optimizer_maximize,
                    "foreach": self.optimizer_foreach,
                    "capturable": self.optimizer_capturable,
                    "differentiable": self.optimizer_differentiable,
                    "fused": self.optimizer_fused,
                    "decoupled_weight_decay": self.optimizer_decoupled_weight_decay,
                },
                "num_train_timesteps": self.num_train_timesteps,
                "flow_shift": self.flow_shift,
                "use_dynamic_shifting": self.use_dynamic_shifting,
                "timestep_sampling": self.timestep_sampling,
                "prediction_target": self.prediction_target,
                "mixed_precision": self.mixed_precision,
                "gradient_checkpointing": self.gradient_checkpointing,
                "max_support_images": self.max_support_images,
                "query_passes_per_step": self.query_passes_per_step,
            },
        }

    @classmethod
    def load(cls, path: Path) -> SanaPilotConfig:
        text = path.read_text(encoding="utf-8")
        decoded = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_nonfinite_constant,
        )
        expected = _canonical_payload()
        _require_canonical_value(decoded, expected, "root")
        payload = cast(dict[str, object], decoded)
        sana = cast(dict[str, object], payload["sana"])
        support = cast(dict[str, object], payload["support_encoder"])
        adapter = cast(dict[str, object], payload["adapter"])
        training = cast(dict[str, object], payload["training"])
        optimizer = cast(dict[str, object], training["optimizer"])

        return cls(
            schema_version=cast(str, payload["schema_version"]),
            model_id=cast(str, sana["model_id"]),
            revision=cast(str, sana["revision"]),
            resolution=cast(int, sana["resolution"]),
            latent_channels=cast(int, sana["latent_channels"]),
            latent_size=cast(int, sana["latent_size"]),
            text_feature_dim=cast(int, sana["text_feature_dim"]),
            max_sequence_length=cast(int, sana["max_sequence_length"]),
            dtype=cast(str, sana["dtype"]),
            support_model_id=cast(str, support["model_id"]),
            support_revision=cast(str, support["revision"]),
            support_feature_dim=cast(int, support["feature_dim"]),
            layout_version=cast(str, adapter["layout_version"]),
            num_blocks=cast(int, adapter["num_blocks"]),
            attention_kinds=tuple(cast(list[str], adapter["attention_kinds"])),
            target_modules=tuple(cast(list[str], adapter["target_modules"])),
            width=cast(int, adapter["width"]),
            rank=cast(int, adapter["rank"]),
            atom_count=cast(int, adapter["atom_count"]),
            projection_count=cast(int, adapter["projection_count"]),
            code_dim=cast(int, adapter["code_dim"]),
            atom_tensor_count=cast(int, adapter["atom_tensor_count"]),
            atom_parameter_count=cast(int, adapter["atom_parameter_count"]),
            scheduler_class=cast(str, training["scheduler_class"]),
            optimizer_class=cast(str, optimizer["class"]),
            optimizer_lr=cast(float, optimizer["lr"]),
            optimizer_betas=(
                cast(list[float], optimizer["betas"])[0],
                cast(list[float], optimizer["betas"])[1],
            ),
            optimizer_eps=cast(float, optimizer["eps"]),
            optimizer_weight_decay=cast(float, optimizer["weight_decay"]),
            optimizer_amsgrad=cast(bool, optimizer["amsgrad"]),
            optimizer_maximize=cast(bool, optimizer["maximize"]),
            optimizer_foreach=cast(bool, optimizer["foreach"]),
            optimizer_capturable=cast(bool, optimizer["capturable"]),
            optimizer_differentiable=cast(bool, optimizer["differentiable"]),
            optimizer_fused=cast(bool, optimizer["fused"]),
            optimizer_decoupled_weight_decay=cast(
                bool, optimizer["decoupled_weight_decay"]
            ),
            num_train_timesteps=cast(int, training["num_train_timesteps"]),
            flow_shift=cast(float, training["flow_shift"]),
            use_dynamic_shifting=cast(bool, training["use_dynamic_shifting"]),
            timestep_sampling=cast(str, training["timestep_sampling"]),
            prediction_target=cast(str, training["prediction_target"]),
            mixed_precision=cast(str, training["mixed_precision"]),
            gradient_checkpointing=cast(bool, training["gradient_checkpointing"]),
            max_support_images=cast(int, training["max_support_images"]),
            query_passes_per_step=cast(int, training["query_passes_per_step"]),
        )

    def _assert_derived_contract(self) -> None:
        if (
            self.layout_version != SANA_LAYOUT_VERSION
            or self.attention_kinds != ATTENTION_KINDS
            or self.target_modules != TARGET_MODULES
        ):
            raise ValueError("adapter layout order changed from the runtime contract")
        layout = SanaAdapterLayout(self.num_blocks, self.atom_count)
        expected = (
            (20, 2, 3, 4),
            120,
            480,
            240,
            8_601_600,
        )
        actual = (
            self.code_shape,
            layout.projection_count,
            layout.code_dim,
            layout.atom_tensor_count,
            layout.trainable_parameter_count(width=self.width, rank=self.rank),
        )
        committed = (
            self.code_shape,
            self.projection_count,
            self.code_dim,
            self.atom_tensor_count,
            self.atom_parameter_count,
        )
        if actual != expected or committed != expected:
            raise ValueError("derived adapter dimensions changed from the canonical contract")


@dataclass(frozen=True, slots=True)
class SubjectsPilotConfig:
    schema_version: str
    scope: str
    publication_eligible: bool
    dataset_id: str
    revision: str
    config_name: str
    split: str
    source_file: str
    source_file_sha256: str
    streaming: bool
    row_indices: tuple[int, ...]
    public: bool
    gated: bool
    license_spdx: str
    held_in: bool
    held_in_meaning: str
    support_side: str
    query_side: str
    concept_field: str
    support_prompt_field: str
    query_prompt_field: str
    mode: str
    size: tuple[int, ...]
    image_size: int
    padding_pixels: int
    left_crop: tuple[int, ...]
    right_crop: tuple[int, ...]
    feature_order: tuple[str, ...]
    quality_field_order: tuple[str, ...]
    description_field_order: tuple[str, ...]

    def __post_init__(self) -> None:
        self.validate()

    def _as_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope,
            "publication_eligible": self.publication_eligible,
            "dataset": {
                "dataset_id": self.dataset_id,
                "revision": self.revision,
                "config_name": self.config_name,
                "split": self.split,
                "source_file": self.source_file,
                "source_file_sha256": self.source_file_sha256,
                "streaming": self.streaming,
                "row_indices": list(self.row_indices),
                "public": self.public,
                "gated": self.gated,
                "license_spdx": self.license_spdx,
            },
            "semantics": {
                "held_in": self.held_in,
                "held_in_meaning": self.held_in_meaning,
                "support_side": self.support_side,
                "query_side": self.query_side,
                "concept_field": self.concept_field,
                "support_prompt_field": self.support_prompt_field,
                "query_prompt_field": self.query_prompt_field,
            },
            "composite": {
                "mode": self.mode,
                "size": list(self.size),
                "image_size": self.image_size,
                "padding_pixels": self.padding_pixels,
                "left_crop": list(self.left_crop),
                "right_crop": list(self.right_crop),
            },
            "feature_order": list(self.feature_order),
            "quality_field_order": list(self.quality_field_order),
            "description_field_order": list(self.description_field_order),
        }

    def validate(self) -> None:
        if type(self) is not SubjectsPilotConfig:
            raise TypeError("config must be an exact SubjectsPilotConfig")
        for name in (
            "row_indices",
            "size",
            "left_crop",
            "right_crop",
            "feature_order",
            "quality_field_order",
            "description_field_order",
        ):
            if type(getattr(self, name)) is not tuple:
                raise ValueError(f"config.{name} must have exact canonical type tuple")
        _require_canonical_value(
            self._as_payload(), _canonical_subjects_payload(), "subjects config"
        )

    @property
    def canonical_sha256(self) -> str:
        self.validate()
        return SUBJECTS_PILOT_CANONICAL_SHA256

    @classmethod
    def load(cls, path: Path) -> SubjectsPilotConfig:
        decoded = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_nonfinite_constant,
        )
        expected = _canonical_subjects_payload()
        _require_canonical_value(decoded, expected, "subjects root")
        payload = cast(dict[str, object], decoded)
        dataset = cast(dict[str, object], payload["dataset"])
        semantics = cast(dict[str, object], payload["semantics"])
        composite = cast(dict[str, object], payload["composite"])
        return cls(
            schema_version=cast(str, payload["schema_version"]),
            scope=cast(str, payload["scope"]),
            publication_eligible=cast(bool, payload["publication_eligible"]),
            dataset_id=cast(str, dataset["dataset_id"]),
            revision=cast(str, dataset["revision"]),
            config_name=cast(str, dataset["config_name"]),
            split=cast(str, dataset["split"]),
            source_file=cast(str, dataset["source_file"]),
            source_file_sha256=cast(str, dataset["source_file_sha256"]),
            streaming=cast(bool, dataset["streaming"]),
            row_indices=tuple(cast(list[int], dataset["row_indices"])),
            public=cast(bool, dataset["public"]),
            gated=cast(bool, dataset["gated"]),
            license_spdx=cast(str, dataset["license_spdx"]),
            held_in=cast(bool, semantics["held_in"]),
            held_in_meaning=cast(str, semantics["held_in_meaning"]),
            support_side=cast(str, semantics["support_side"]),
            query_side=cast(str, semantics["query_side"]),
            concept_field=cast(str, semantics["concept_field"]),
            support_prompt_field=cast(str, semantics["support_prompt_field"]),
            query_prompt_field=cast(str, semantics["query_prompt_field"]),
            mode=cast(str, composite["mode"]),
            size=tuple(cast(list[int], composite["size"])),
            image_size=cast(int, composite["image_size"]),
            padding_pixels=cast(int, composite["padding_pixels"]),
            left_crop=tuple(cast(list[int], composite["left_crop"])),
            right_crop=tuple(cast(list[int], composite["right_crop"])),
            feature_order=tuple(cast(list[str], payload["feature_order"])),
            quality_field_order=tuple(cast(list[str], payload["quality_field_order"])),
            description_field_order=tuple(
                cast(list[str], payload["description_field_order"])
            ),
        )


@dataclass(frozen=True, slots=True)
class ModalBudgetConfig:
    profile: str
    environment: str
    workspace_budget_usd: Decimal
    internal_limit_usd: Decimal
    first_pilot_allocation_usd: Decimal
    setup_probe_allocation_usd: Decimal
    timing_probe_allocation_usd: Decimal
    held_in_pilot_allocation_usd: Decimal
    unallocated_safety_buffer_usd: Decimal
    attestation_max_age_seconds: int
    gpu: str
    gpu_count: int
    cpu_cores: int
    memory_gib: int
    timeout_seconds: int
    startup_timeout_seconds: int
    storage_gib_bound: int
    non_gpu_setup_allowance_usd: Decimal
    retries: int
    max_containers: int
    detached: bool
    cache_volume: str
    artifact_volume: str

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if type(self) is not ModalBudgetConfig:
            raise TypeError("budget config must be an exact ModalBudgetConfig")
        actual: dict[str, object] = {
            "schema_version": "1.0.0",
            "profile": self.profile,
            "environment": self.environment,
            "workspace_budget_usd": str(self.workspace_budget_usd),
            "internal_limit_usd": str(self.internal_limit_usd),
            "first_pilot_allocation_usd": str(self.first_pilot_allocation_usd),
            "setup_probe_allocation_usd": str(self.setup_probe_allocation_usd),
            "timing_probe_allocation_usd": str(self.timing_probe_allocation_usd),
            "held_in_pilot_allocation_usd": str(self.held_in_pilot_allocation_usd),
            "unallocated_safety_buffer_usd": str(self.unallocated_safety_buffer_usd),
            "attestation_max_age_seconds": self.attestation_max_age_seconds,
            "gpu": self.gpu,
            "gpu_count": self.gpu_count,
            "cpu_cores": self.cpu_cores,
            "memory_gib": self.memory_gib,
            "timeout_seconds": self.timeout_seconds,
            "startup_timeout_seconds": self.startup_timeout_seconds,
            "storage_gib_bound": self.storage_gib_bound,
            "non_gpu_setup_allowance_usd": str(self.non_gpu_setup_allowance_usd),
            "retries": self.retries,
            "max_containers": self.max_containers,
            "detached": self.detached,
            "cache_volume": self.cache_volume,
            "artifact_volume": self.artifact_volume,
        }
        _require_canonical_value(
            actual,
            _canonical_modal_budget_payload(),
            "Modal budget config",
        )
        if (
            self.setup_probe_allocation_usd
            + self.timing_probe_allocation_usd
            + self.held_in_pilot_allocation_usd
            != self.first_pilot_allocation_usd
            or self.first_pilot_allocation_usd + self.unallocated_safety_buffer_usd
            != self.internal_limit_usd
        ):
            raise ValueError("Modal budget phase allocation arithmetic changed")

    @classmethod
    def load(cls, path: Path) -> ModalBudgetConfig:
        decoded = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_nonfinite_constant,
        )
        expected = _canonical_modal_budget_payload()
        _require_canonical_value(decoded, expected, "Modal budget root")
        payload = cast(dict[str, object], decoded)
        decimal_fields = {
            "workspace_budget_usd",
            "internal_limit_usd",
            "first_pilot_allocation_usd",
            "setup_probe_allocation_usd",
            "timing_probe_allocation_usd",
            "held_in_pilot_allocation_usd",
            "unallocated_safety_buffer_usd",
            "non_gpu_setup_allowance_usd",
        }
        values: dict[str, object] = {
            key: Decimal(cast(str, value)) if key in decimal_fields else value
            for key, value in payload.items()
            if key != "schema_version"
        }
        return cls(**cast(Any, values))
