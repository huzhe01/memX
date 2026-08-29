from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Never, cast

from ratemem.adapters.sana_layout import (
    ATTENTION_KINDS,
    SANA_LAYOUT_VERSION,
    TARGET_MODULES,
    SanaAdapterLayout,
)


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
