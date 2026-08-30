"""Frozen backbone identities and the sole primary SANA execution bridge."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal, Protocol

import torch
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

from ratemem.adapters.sana_layout import SanaDynamicAdapterBank

_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)


class PrimaryBackboneSpec(BaseModel):
    model_config = _MODEL_CONFIG

    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    architecture: Literal["sana_transformer"]
    resolution: tuple[Literal[1024], Literal[1024]]
    target_suffixes: tuple[Literal["to_q", "to_k", "to_v"], ...]
    layout_lock_path: Path
    comparison_role: Literal["primary"]
    primary_eligible: Literal[True]
    ratemem_extension_available: Literal[True]
    expected_projection_count: Literal[120]
    code_dim: Literal[480]


class ContextualBackboneSpec(BaseModel):
    model_config = _MODEL_CONFIG

    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    architecture: Literal["sdxl_unet"]
    resolution: tuple[Literal[1024], Literal[1024]]
    target_suffixes: tuple[Literal["to_q", "to_k", "to_v"], ...]
    comparison_role: Literal["contextual_only"]
    primary_eligible: Literal[False]
    ratemem_extension_available: Literal[False]


BackboneSpec = Annotated[
    PrimaryBackboneSpec | ContextualBackboneSpec,
    Field(discriminator="comparison_role"),
]


class BackbonePolicy(BaseModel):
    model_config = _MODEL_CONFIG

    schema_version: Literal["1.0"]
    primary_backbone: Literal["sana_1_5_1_6b"]
    primary_backbone_is_fixed: Literal[True]
    contextual_backbones: tuple[Literal["sdxl_1_0"], ...]
    contextual_backbones_may_satisfy_primary_requirements: Literal[False]
    future_contextual_promotion_requires_separately_reviewed_ratemem_extension: Literal[True]
    published_different_backbone_numbers_are_contextual: Literal[True]
    backbones: dict[str, BackboneSpec]
    shared_input_scopes: dict[str, tuple[str, ...]]

    @model_validator(mode="after")
    def validate_fixed_route(self) -> BackbonePolicy:
        if set(self.backbones) != {"sana_1_5_1_6b", "sdxl_1_0"}:
            raise ValueError("unexpected backbone set")
        if self.contextual_backbones != ("sdxl_1_0",):
            raise ValueError("SDXL must be the sole contextual backbone")
        if self.backbones[self.primary_backbone].comparison_role != "primary":
            raise ValueError("SANA must be the primary backbone")
        if any(self.backbones[item].primary_eligible for item in self.contextual_backbones):
            raise ValueError("contextual backbone cannot be primary eligible")
        expected_scopes = {
            "same_code_and_candidates",
            "same_amortizer_recompute",
            "feature_native",
            "optimization_native",
            "upstream_native",
            "upper_reference",
        }
        if set(self.shared_input_scopes) != expected_scopes:
            raise ValueError("shared-input scope partition is incomplete")
        methods = [method for values in self.shared_input_scopes.values() for method in values]
        if len(methods) != len(set(methods)):
            raise ValueError("a method occurs in multiple shared-input scopes")
        return self

    def scope_for_method(self, method_id: str) -> str:
        matches = [
            scope
            for scope, method_ids in self.shared_input_scopes.items()
            if method_id in method_ids
        ]
        if len(matches) != 1:
            raise ValueError(f"method has no unique shared-input scope: {method_id}")
        return matches[0]


def load_backbone_policy(path: str | Path) -> BackbonePolicy:
    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return BackbonePolicy.model_validate(payload)
    except (OSError, ValueError, yaml.YAMLError) as error:
        raise ValueError(f"invalid backbone policy: {error}") from error


class BackboneRunner(Protocol):
    backbone_id: Literal["sana_1_5_1_6b"]
    spec: PrimaryBackboneSpec

    def install_code(self, code: Tensor) -> None: ...

    def clear_code(self) -> None: ...

    def generate(
        self,
        prompt: str,
        seed: int,
        *,
        sampler_id: str,
        cfg_scale: float,
        steps: int,
    ) -> Tensor: ...

    def one_step_latent(self, prompt: str, seed: int, timestep: int) -> Tensor: ...


class SanaBackboneRunner:
    """SANA-only bridge that scopes every generation under one installed code."""

    backbone_id: Literal["sana_1_5_1_6b"] = "sana_1_5_1_6b"

    def __init__(
        self,
        *,
        spec: PrimaryBackboneSpec,
        adapter_bank: SanaDynamicAdapterBank,
        generate_fn: Callable[[str, int, str, float, int], Tensor],
        one_step_fn: Callable[[str, int, int], Tensor],
    ) -> None:
        if spec.expected_projection_count != adapter_bank.layout.projection_count:
            raise ValueError("SANA projection count differs from the frozen backbone")
        if spec.code_dim != adapter_bank.layout.code_dim:
            raise ValueError("SANA code dimension differs from the frozen backbone")
        self.spec = spec
        self._adapter_bank = adapter_bank
        self._generate_fn = generate_fn
        self._one_step_fn = one_step_fn
        self._code: Tensor | None = None

    def install_code(self, code: Tensor) -> None:
        if self._code is not None:
            raise RuntimeError("a SANA code is already installed")
        if not isinstance(code, Tensor) or code.shape != (self.spec.code_dim,):
            raise ValueError("SANA code must be one flat 480-vector")
        if not torch.isfinite(code).all():
            raise ValueError("SANA code must be finite")
        self._code = code.detach().clone()

    def clear_code(self) -> None:
        self._code = None

    def _require_code(self) -> Tensor:
        if self._code is None:
            raise RuntimeError("no SANA code is installed")
        return self._code

    def generate(
        self,
        prompt: str,
        seed: int,
        *,
        sampler_id: str,
        cfg_scale: float,
        steps: int,
    ) -> Tensor:
        code = self._require_code()
        with self._adapter_bank.activate(code):
            return self._generate_fn(prompt, seed, sampler_id, cfg_scale, steps)

    def one_step_latent(self, prompt: str, seed: int, timestep: int) -> Tensor:
        code = self._require_code()
        with self._adapter_bank.activate(code):
            return self._one_step_fn(prompt, seed, timestep)


__all__ = [
    "BackbonePolicy",
    "BackboneRunner",
    "ContextualBackboneSpec",
    "PrimaryBackboneSpec",
    "SanaBackboneRunner",
    "load_backbone_policy",
]
