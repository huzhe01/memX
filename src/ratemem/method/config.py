"""Frozen learned-method policy and scientific-input binding."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator

from ratemem.evaluation.canonical import file_sha256, semantic_sha256
from ratemem.evaluation.types import Sha256

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)
_LOSS_KEYS = frozenset(
    {
        "flow",
        "reconstruction",
        "rate",
        "reuse_affinity",
        "dictionary_balance",
        "dictionary_commitment",
        "utility_calibration",
    }
)
_BORROWED_COMPONENTS = (
    "hyperlora_style_support_amortizer",
    "grouped_residual_vector_quantization",
    "symmetric_integer_quantization",
    "partial_enumeration_submodular_knapsack_allocator",
)


class LockMismatch(ValueError):
    """Raised when an approved scientific input hash or scope changed."""


class CodePolicy(BaseModel):
    model_config = _MODEL_CONFIG

    projection_count: PositiveInt
    atom_count: PositiveInt
    dimension: PositiveInt

    @model_validator(mode="after")
    def validate_dimension(self) -> CodePolicy:
        if self.dimension != self.projection_count * self.atom_count:
            raise ValueError("code dimension must equal projection_count times atom_count")
        return self


class CodecPolicy(BaseModel):
    model_config = _MODEL_CONFIG

    group_size: PositiveInt
    base_bits: Literal[2, 4, 8]
    rvq_stages: PositiveInt
    entries_per_stage: PositiveInt
    incidence_gain_step: PositiveFloat
    maximum_packets_per_concept: PositiveInt
    sharing_rule: Literal["exact_payload_only"]
    packet_format_version: Literal["RTPKT001"]


class SoftCodecPolicy(BaseModel):
    model_config = _MODEL_CONFIG

    initial_temperature: PositiveFloat
    final_temperature: PositiveFloat
    anneal_steps: PositiveInt
    maximum_mean_code_error: PositiveFloat
    maximum_assignment_disagreement: float = Field(ge=0.0, le=1.0)
    maximum_topk_disagreement: float = Field(ge=0.0, le=0.0)
    ste_forward_atol: PositiveFloat

    @model_validator(mode="after")
    def validate_temperature_schedule(self) -> SoftCodecPolicy:
        if self.final_temperature > self.initial_temperature:
            raise ValueError("soft-codec temperature must not increase")
        if self.ste_forward_atol != 0.000001:
            raise ValueError("STE forward tolerance differs from the frozen policy")
        return self


class UtilityPolicy(BaseModel):
    model_config = _MODEL_CONFIG

    hidden_dimension: PositiveInt
    request_decay: float = Field(gt=0.0, le=1.0)
    calibration_bins: PositiveInt
    maximum_expected_calibration_error: float = Field(ge=0.0, le=1.0)


class ControllerPolicy(BaseModel):
    model_config = _MODEL_CONFIG

    outer_policy: Literal["request_density_size_aware"]
    allow_rejection: Literal[True]
    whole_concept_eviction: Literal[True]
    switching_penalty: float = Field(ge=0.0, le=0.0)
    certified_prescreen_max_bundles: Literal[24]
    theorem_scope: Literal["fixed_admitted_cohort_prescreened_packets_only"]


class TrainingPolicy(BaseModel):
    model_config = _MODEL_CONFIG

    segment_length: Literal[2]
    maximum_query_events_per_segment: Literal[2]
    maximum_transformer_passes_per_segment: Literal[2]
    truncated_bptt_length: Literal[2]
    detach_at_segment_boundary: Literal[True]
    precision: Literal["bfloat16"]
    activation_checkpointing: Literal[True]
    training_seeds: tuple[int, int, int]
    loss_weights: dict[str, float]

    @model_validator(mode="after")
    def validate_training(self) -> TrainingPolicy:
        if len(set(self.training_seeds)) != len(self.training_seeds):
            raise ValueError("training seeds must be unique")
        if set(self.loss_weights) != _LOSS_KEYS:
            raise ValueError("training loss weight keys do not match the locked objective")
        if any(value < 0.0 for value in self.loss_weights.values()):
            raise ValueError("training loss weights must be nonnegative")
        return self


class MethodPolicy(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"]
    method_id: Literal["ratemem_v1"]
    novelty_claim: Literal[
        "learned_multi_concept_immutable_packet_bundles_with_causal_exact_byte_lifecycle"
    ]
    borrowed_components: tuple[str, ...]
    code: CodePolicy
    codec: CodecPolicy
    soft_codec: SoftCodecPolicy
    utility: UtilityPolicy
    controller: ControllerPolicy
    training: TrainingPolicy

    @classmethod
    def from_yaml(cls, path: Path) -> MethodPolicy:
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            return cls.model_validate(payload)
        except (OSError, ValueError, yaml.YAMLError) as error:
            raise LockMismatch(f"invalid method policy: {error}") from error

    @model_validator(mode="after")
    def validate_policy(self) -> MethodPolicy:
        if self.code.dimension % self.codec.group_size:
            raise ValueError("code dimension must be divisible by codec group size")
        if self.borrowed_components != _BORROWED_COMPONENTS:
            raise ValueError("borrowed-component boundary differs from the frozen policy")
        if self.codec.maximum_packets_per_concept > (
            self.code.dimension // self.codec.group_size * self.codec.rvq_stages
        ):
            raise ValueError("maximum packets exceeds the dictionary assignment count")
        return self


class MethodLockInputs(BaseModel):
    model_config = _MODEL_CONFIG

    policy_path: Path
    dataset_lock_path: Path
    evaluation_lock_path: Path
    baseline_lock_path: Path
    visible_trace_manifest_paths: tuple[Path, ...]
    expected_dataset_lock_sha256: Sha256
    expected_evaluation_lock_sha256: Sha256
    expected_baseline_lock_sha256: Sha256

    @model_validator(mode="after")
    def validate_visible_scope(self) -> MethodLockInputs:
        if not self.visible_trace_manifest_paths:
            raise ValueError("at least one visible trace manifest is required")
        normalized = tuple(str(path.resolve()) for path in self.visible_trace_manifest_paths)
        if len(normalized) != len(set(normalized)):
            raise ValueError("visible trace manifest paths must be unique")
        return self


class MethodTrainingLock(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"] = "1.0"
    lock_id: Sha256
    method_id: Literal["ratemem_v1"]
    policy_sha256: Sha256
    dataset_lock_sha256: Sha256
    evaluation_lock_sha256: Sha256
    baseline_lock_sha256: Sha256
    visible_trace_manifest_sha256: tuple[Sha256, ...]


def _is_final_trace_path(path: Path) -> bool:
    normalized = path.name.lower().replace("_", "-")
    return "final-test" in normalized or normalized.startswith("final-")


def freeze_method_lock(inputs: MethodLockInputs) -> MethodTrainingLock:
    """Bind the learned method only to approved train and validation artifacts."""

    if any(_is_final_trace_path(path) for path in inputs.visible_trace_manifest_paths):
        raise LockMismatch("final-test trace manifests are forbidden during method training")
    actual = {
        "dataset": file_sha256(inputs.dataset_lock_path),
        "evaluation": file_sha256(inputs.evaluation_lock_path),
        "baseline": file_sha256(inputs.baseline_lock_path),
    }
    expected = {
        "dataset": inputs.expected_dataset_lock_sha256,
        "evaluation": inputs.expected_evaluation_lock_sha256,
        "baseline": inputs.expected_baseline_lock_sha256,
    }
    for name, digest in actual.items():
        if digest != expected[name]:
            raise LockMismatch(f"{name} lock content hash does not match approval")
    policy = MethodPolicy.from_yaml(inputs.policy_path)
    policy_sha256 = file_sha256(inputs.policy_path)
    visible_trace_manifest_sha256 = tuple(
        file_sha256(path) for path in sorted(inputs.visible_trace_manifest_paths)
    )
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "method_id": policy.method_id,
        "policy_sha256": policy_sha256,
        "dataset_lock_sha256": actual["dataset"],
        "evaluation_lock_sha256": actual["evaluation"],
        "baseline_lock_sha256": actual["baseline"],
        "visible_trace_manifest_sha256": visible_trace_manifest_sha256,
    }
    return MethodTrainingLock(
        lock_id=semantic_sha256(payload),
        method_id="ratemem_v1",
        policy_sha256=policy_sha256,
        dataset_lock_sha256=actual["dataset"],
        evaluation_lock_sha256=actual["evaluation"],
        baseline_lock_sha256=actual["baseline"],
        visible_trace_manifest_sha256=visible_trace_manifest_sha256,
    )


__all__ = [
    "CodePolicy",
    "CodecPolicy",
    "ControllerPolicy",
    "LockMismatch",
    "MethodLockInputs",
    "MethodPolicy",
    "MethodTrainingLock",
    "SoftCodecPolicy",
    "TrainingPolicy",
    "UtilityPolicy",
    "freeze_method_lock",
]
