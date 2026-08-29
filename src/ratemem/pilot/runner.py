"""Closed orchestration boundary for the one authorized SANA engineering pilot."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import stat
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Final, Protocol, cast

import torch
from diffusers import SanaTransformer2DModel
from torch import nn

from ratemem.adapters.checkpoint import (
    CheckpointFileIdentity,
    CheckpointProvenance,
    save_trainable_checkpoint,
)
from ratemem.adapters.sana_layout import (
    SanaDynamicAdapterBank,
    install_sana_dynamic_atoms,
    validate_production_sana_layout,
)
from ratemem.pilot.artifacts import ArtifactWriter
from ratemem.pilot.config import (
    ModalBudgetConfig,
    SanaPilotConfig,
    SubjectsPilotConfig,
)
from ratemem.pilot.costs import CostRates
from ratemem.pilot.data import (
    PrecomputedPilotData,
    build_precomputed_cache,
    hydrate_locked_examples,
)
from ratemem.pilot.private_io import (
    canonical_json_bytes,
    ensure_private_directory,
)
from ratemem.pilot.probes import (
    ALLOWED_PROBES,
    CudaPeak,
    cuda_peak,
    held_in_step_cap,
    percentile,
    timed,
)
from ratemem.sana.components import (
    PinnedComponents,
    hydrate_pinned_snapshots,
    load_pinned_components,
)
from ratemem.sana.flow import (
    FlowBatch,
    FlowDraw,
    FlowStepResult,
    OneTimestepFlowTrainer,
)
from ratemem.support.amortizer import SupportAmortizer

_SANA_CONFIG_PATH: Final = Path("configs/pilot/sana-1.5-1.6b.json")
_SUBJECTS_CONFIG_PATH: Final = Path("configs/pilot/subjects200k-held-in.json")
_BUDGET_CONFIG_PATH: Final = Path("configs/pilot/modal-budget.json")
_CONFIG_PATHS: Final = (
    ("modal_budget", _BUDGET_CONFIG_PATH),
    ("sana", _SANA_CONFIG_PATH),
    ("subjects200k", _SUBJECTS_CONFIG_PATH),
)
_RATE_KEYS: Final = (
    "gpu_l40s_per_second",
    "cpu_core_per_second",
    "memory_gib_per_second",
    "volume_gib_month",
)
_REQUEST_KEYS: Final = frozenset(
    {
        "attempt_id",
        "workspace",
        "source_sha256",
        "git_commit",
        "git_diff_sha256",
        "config_sha256",
        "slot_sha256",
        "permit_sha256",
        "submission_receipt_sha256",
        "known_usage_before_usd",
        "pending_worst_case_usd",
        "phase_bound_usd",
        "rates",
        "rates_sha256",
    }
)
_MODAL_ID_KEYS: Final = frozenset(
    {
        "profile",
        "workspace",
        "environment",
        "launch_attempt_id",
        "launch_source_sha256",
        "pilot_slot_sha256",
        "submission_receipt_sha256",
        "function_call_id",
        "input_id",
        "task_id",
        "container_image_id",
        "execution_receipt_path",
        "execution_receipt_directory",
        "execution_receipt_count",
        "execution_receipts_sha256",
        "execution_receipt_semantics",
    }
)
_RECEIPT_SEMANTICS: Final = "lower_bound_may_miss_precommit_reschedule"
_INFERENCE_PROMPT: Final = "A studio photograph of a red ceramic teapot on a plain gray table."
_INFERENCE_SEED: Final = 20260824
_BACKWARD_SEED: Final = 20260825
_TRAIN_SEED_BASE: Final = 20260826
_EVALUATION_NOISE_SEEDS: Final = tuple(range(20260900, 20260908))
_EVALUATION_TIMESTEP_INDICES: Final = (37, 163, 289, 415, 541, 667, 793, 919)
_SHUTDOWN_RESERVE_SECONDS: Final = 120


def _lower_hex(value: object, name: str, length: int = 64) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact string")
    checked = value
    if len(checked) != length or any(character not in "0123456789abcdef" for character in checked):
        raise ValueError(f"{name} must be {length} lowercase hexadecimal characters")
    return checked


def _decimal_text(
    value: object,
    name: str,
    *,
    positive: bool = False,
) -> Decimal:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact decimal string")
    try:
        checked = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"{name} must be a valid decimal amount") from error
    if not checked.is_finite():
        raise ValueError(f"{name} must be finite")
    if checked < 0:
        raise ValueError(f"{name} must be nonnegative")
    if positive and checked <= 0:
        raise ValueError(f"{name} must be positive")
    if re.fullmatch(r"(0|[1-9][0-9]*)\.[0-9]{2,6}", value) is None:
        raise ValueError(f"{name} must be a fixed-point decimal with 2 to 6 places")
    return checked


def _exact_decimal(
    value: object,
    name: str,
    *,
    positive: bool = False,
) -> Decimal:
    if type(value) is not Decimal:
        raise TypeError(f"{name} must be an exact Decimal")
    checked = value
    if not checked.is_finite():
        raise ValueError(f"{name} must be finite")
    if checked < 0:
        raise ValueError(f"{name} must be nonnegative")
    if positive and checked <= 0:
        raise ValueError(f"{name} must be positive")
    return checked


def _finite_nonnegative(value: object, name: str, *, positive: bool = False) -> float:
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact float")
    checked = value
    if not math.isfinite(checked) or checked < 0 or (positive and checked <= 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return checked


def _exact_int(
    value: object,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int")
    checked = value
    if positive and checked <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and checked < 0:
        raise ValueError(f"{name} must be nonnegative")
    return checked


def _config_payload() -> dict[str, object]:
    payload: dict[str, object] = {}
    for name, path in _CONFIG_PATHS:
        try:
            decoded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"locked pilot config is unavailable: {path}") from error
        if type(decoded) is not dict:
            raise TypeError(f"locked pilot config root must be an object: {path}")
        payload[name] = decoded
    return payload


def pilot_config_sha256() -> str:
    """Hash the complete canonical three-config pilot bundle."""

    # The exact config classes perform the authoritative strict validation.
    SanaPilotConfig.load(_SANA_CONFIG_PATH)
    SubjectsPilotConfig.load(_SUBJECTS_CONFIG_PATH)
    ModalBudgetConfig.load(_BUDGET_CONFIG_PATH)
    return hashlib.sha256(canonical_json_bytes(_config_payload())).hexdigest()


@dataclass(frozen=True, slots=True)
class PilotRequest:
    attempt_id: str
    workspace: str
    source_sha256: str
    git_commit: str
    git_diff_sha256: str
    config_sha256: str
    slot_sha256: str
    permit_sha256: str
    submission_receipt_sha256: str
    known_usage_before_usd: Decimal
    pending_worst_case_usd: Decimal
    phase_bound_usd: Decimal
    rates: tuple[tuple[str, str], ...]
    rates_sha256: str

    def __post_init__(self) -> None:
        if type(self.attempt_id) is not str:
            raise TypeError("attempt_id must be an exact string")
        if type(self.workspace) is not str:
            raise TypeError("workspace must be an exact string")
        parsed = uuid.UUID(self.attempt_id)
        if str(parsed) != self.attempt_id or parsed.version != 7:
            raise ValueError("attempt_id must be one canonical UUID version 7")
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?", self.workspace):
            raise ValueError(
                "workspace must be a 1-64 character lowercase alphanumeric/hyphen slug"
            )
        _lower_hex(self.git_commit, "git_commit", 40)
        for name in (
            "source_sha256",
            "git_diff_sha256",
            "config_sha256",
            "slot_sha256",
            "permit_sha256",
            "submission_receipt_sha256",
            "rates_sha256",
        ):
            _lower_hex(getattr(self, name), name)
        expected_source = hashlib.sha256(self.git_commit.encode("ascii")).hexdigest()
        if self.source_sha256 != expected_source:
            raise ValueError("source SHA-256 is not bound to the exact git commit text")
        _exact_decimal(
            self.known_usage_before_usd,
            "known_usage_before_usd",
        )
        _exact_decimal(
            self.pending_worst_case_usd,
            "pending_worst_case_usd",
        )
        _exact_decimal(self.phase_bound_usd, "phase_bound_usd", positive=True)
        if self.phase_bound_usd > self.pending_worst_case_usd:
            raise ValueError("phase bound must not exceed pending worst-case cost")
        if self.known_usage_before_usd + self.pending_worst_case_usd > Decimal("27.00"):
            raise ValueError("request exceeds the internal USD 27.00 bound")
        if type(self.rates) is not tuple or tuple(name for name, _ in self.rates) != _RATE_KEYS:
            raise ValueError("request rates have missing, reordered, or unexpected keys")
        normalized_rates = dict(self.rates)
        CostRates.normalize(normalized_rates)
        expected_rates = hashlib.sha256(canonical_json_bytes(normalized_rates)).hexdigest()
        if expected_rates != self.rates_sha256:
            raise ValueError("rates SHA-256 does not bind the exact normalized rates")

    @classmethod
    def from_mapping(cls, value: object) -> PilotRequest:
        if type(value) is not dict:
            raise TypeError("pilot request must be an exact dict")
        payload = cast(dict[object, object], value)
        if any(type(key) is not str for key in payload) or set(payload) != _REQUEST_KEYS:
            raise ValueError("pilot request has missing or unexpected fields")
        raw_rates = payload["rates"]
        if type(raw_rates) is not dict:
            raise TypeError("pilot request rates must be an exact dict")
        rates_dict = cast(dict[object, object], raw_rates)
        if any(type(key) is not str for key in rates_dict) or set(rates_dict) != set(_RATE_KEYS):
            raise ValueError("pilot request rates have missing or unexpected keys")
        rates: list[tuple[str, str]] = []
        for name in _RATE_KEYS:
            rate = rates_dict[name]
            if type(rate) is not str:
                raise TypeError("pilot request rate values must be exact strings")
            rates.append((name, rate))
        return cls(
            attempt_id=cast(str, payload["attempt_id"]),
            workspace=cast(str, payload["workspace"]),
            source_sha256=cast(str, payload["source_sha256"]),
            git_commit=cast(str, payload["git_commit"]),
            git_diff_sha256=cast(str, payload["git_diff_sha256"]),
            config_sha256=cast(str, payload["config_sha256"]),
            slot_sha256=cast(str, payload["slot_sha256"]),
            permit_sha256=cast(str, payload["permit_sha256"]),
            submission_receipt_sha256=cast(str, payload["submission_receipt_sha256"]),
            known_usage_before_usd=_decimal_text(
                payload["known_usage_before_usd"],
                "known_usage_before_usd",
            ),
            pending_worst_case_usd=_decimal_text(
                payload["pending_worst_case_usd"],
                "pending_worst_case_usd",
            ),
            phase_bound_usd=_decimal_text(
                payload["phase_bound_usd"],
                "phase_bound_usd",
                positive=True,
            ),
            rates=tuple(rates),
            rates_sha256=cast(str, payload["rates_sha256"]),
        )

    @property
    def cost_rates(self) -> CostRates:
        return CostRates.normalize(dict(self.rates))

    @property
    def resource_usd_per_second(self) -> Decimal:
        rates = self.cost_rates
        return (
            rates.gpu_l40s_per_second
            + Decimal(4) * rates.cpu_core_per_second
            + Decimal(32) * rates.memory_gib_per_second
        )


@dataclass(frozen=True, slots=True)
class PilotLimits:
    warmup_steps: int
    measured_steps: int
    held_in_allocation_usd: Decimal
    resource_usd_per_second: Decimal
    timeout_seconds: int
    shutdown_reserve_seconds: int

    def __post_init__(self) -> None:
        warmup = _exact_int(self.warmup_steps, "warmup_steps", positive=True)
        measured = _exact_int(self.measured_steps, "measured_steps", positive=True)
        if warmup != 10 or measured != 20:
            raise ValueError("pilot timing contract requires 10 warm-up and 20 measured steps")
        _exact_decimal(
            self.held_in_allocation_usd,
            "held_in_allocation_usd",
        )
        _exact_decimal(
            self.resource_usd_per_second,
            "resource_usd_per_second",
            positive=True,
        )
        timeout = _exact_int(self.timeout_seconds, "timeout_seconds", positive=True)
        reserve = _exact_int(
            self.shutdown_reserve_seconds,
            "shutdown_reserve_seconds",
            nonnegative=True,
        )
        if reserve >= timeout:
            raise ValueError("shutdown reserve must be smaller than the timeout")


class PilotBackend(Protocol):
    def compatibility(self) -> dict[str, object]: ...

    def inference(self) -> dict[str, object]: ...

    def backward(self) -> float: ...

    def training_step(self) -> float: ...

    def evaluate_loss(self) -> float: ...

    def save_checkpoint(self, path: Path) -> CheckpointFileIdentity: ...


class PilotEvidenceBackend(PilotBackend, Protocol):
    @property
    def peak(self) -> CudaPeak: ...

    @property
    def dataset_manifest(self) -> dict[str, object]: ...

    @property
    def dataset_manifest_sha256(self) -> str: ...

    def diagnostics(self) -> dict[str, object]: ...


def _passed_probe(value: object, name: str) -> dict[str, str]:
    if type(value) is not dict or value != {"status": "pass"}:
        raise RuntimeError(f"{name} must return exactly one passing probe status")
    return {"status": "pass"}


class PilotRunner:
    """Execute the exact declared probe sequence with no implicit model passes."""

    def __init__(self, *, backend: PilotBackend, limits: PilotLimits) -> None:
        if type(limits) is not PilotLimits:
            raise TypeError("limits must be an exact PilotLimits")
        limits.__post_init__()
        self.backend = backend
        self.limits = limits
        self._used = False
        self._probe_results = {name: {"status": "not_run"} for name in ALLOWED_PROBES}
        self._p50: float | None = None
        self._p95: float | None = None
        self._held_in_step_cap = 0
        self._backward_loss: float | None = None
        self._initial_loss: float | None = None
        self._final_loss: float | None = None
        self._checkpoint_identity: CheckpointFileIdentity | None = None

    def _required_probe(self, name: str, value: object) -> None:
        try:
            self._probe_results[name] = _passed_probe(value, name)
        except BaseException:
            self._probe_results[name] = {"status": "fail"}
            raise

    def _result(self, status: str) -> dict[str, object]:
        checkpoint = self._checkpoint_identity
        result: dict[str, object] = {
            "status": status,
            "allowed_probe_names": list(ALLOWED_PROBES),
            "results": {name: dict(value) for name, value in self._probe_results.items()},
            **{name: dict(value) for name, value in self._probe_results.items()},
            "warmup_steps": self.limits.warmup_steps,
            "measured_steps": self.limits.measured_steps,
            "p50_step_seconds": self._p50,
            "p95_step_seconds": self._p95,
            "held_in_step_cap": self._held_in_step_cap,
            "one_timestep_backward_loss": self._backward_loss,
            "initial_flow_loss": self._initial_loss,
            "final_flow_loss": self._final_loss,
            "transformer_passes_per_step": 1,
            "checkpoint_sha256": None if checkpoint is None else checkpoint.sha256,
            "checkpoint_bytes": None if checkpoint is None else checkpoint.byte_count,
        }
        json.dumps(result, allow_nan=False)
        return result

    def failure_result(self, status: str) -> dict[str, object]:
        """Return only evidence completed before an OOM or exception."""

        if status not in {"oom", "exception"}:
            raise ValueError("incomplete pilot status must be oom or exception")
        return self._result(status)

    def run(self, checkpoint_dir: Path) -> dict[str, object]:
        if self._used:
            raise RuntimeError("PilotRunner instances are single-use")
        self._used = True
        if type(checkpoint_dir) is not type(Path()):
            raise TypeError("checkpoint_dir must be an exact Path")
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError("checkpoint staging directory must already exist")
        checkpoint_path = checkpoint_dir / "trainable.safetensors"
        if checkpoint_path.exists() or checkpoint_path.is_symlink():
            raise FileExistsError("checkpoint staging path must be create-only")

        run_started = time.monotonic()
        self._required_probe(
            "checkpoint_compatibility",
            self.backend.compatibility(),
        )
        self._required_probe(
            "one_step_inference",
            self.backend.inference(),
        )
        self._backward_loss = _finite_nonnegative(
            self.backend.backward(),
            "one-timestep backward loss",
        )
        for name in (
            "dynamic_numerics",
            "gradient_flow",
            "frozen_backbone",
            "peak_memory",
            "one_timestep_backward",
        ):
            self._probe_results[name] = {"status": "pass"}

        for _ in range(self.limits.warmup_steps):
            _finite_nonnegative(
                self.backend.training_step(),
                "warm-up step time",
                positive=True,
            )
        measured: list[float] = []
        for _ in range(self.limits.measured_steps):
            measured.append(
                _finite_nonnegative(
                    self.backend.training_step(),
                    "measured step time",
                    positive=True,
                )
            )
        self._p50 = percentile(measured, 0.50)
        self._p95 = percentile(measured, 0.95)
        self._probe_results["step_timing"] = {"status": "pass"}

        elapsed_seconds = math.ceil(time.monotonic() - run_started)
        remaining_timeout = max(0, self.limits.timeout_seconds - elapsed_seconds)
        self._held_in_step_cap = held_in_step_cap(
            p95_step_seconds=Decimal(str(self._p95)),
            remaining_compute_usd=self.limits.held_in_allocation_usd,
            requested_resource_usd_per_second=self.limits.resource_usd_per_second,
            remaining_timeout_seconds=remaining_timeout,
            shutdown_reserve_seconds=self.limits.shutdown_reserve_seconds,
        )
        initial_loss = _finite_nonnegative(
            self.backend.evaluate_loss(),
            "initial held-in flow loss",
        )
        for _ in range(self._held_in_step_cap):
            _finite_nonnegative(
                self.backend.training_step(),
                "held-in training step time",
                positive=True,
            )
        final_loss = _finite_nonnegative(
            self.backend.evaluate_loss(),
            "final held-in flow loss",
        )
        self._initial_loss = initial_loss
        self._final_loss = final_loss
        loss_fell = final_loss < initial_loss
        self._probe_results["held_in_loss"] = {"status": "pass" if loss_fell else "fail"}

        checkpoint_identity = self.backend.save_checkpoint(checkpoint_path)
        if type(checkpoint_identity) is not CheckpointFileIdentity:
            raise TypeError("backend checkpoint must return an exact CheckpointFileIdentity")
        checkpoint_identity.validate()
        self._checkpoint_identity = checkpoint_identity
        return self._result("succeeded" if loss_fell else "probe_failed")


class RealSanaPilotBackend:
    """The paid-only backend wired exclusively to Tasks 4--8 production APIs."""

    def __init__(
        self,
        *,
        cache_root: Path,
        sana_config: SanaPilotConfig,
        subjects_config: SubjectsPilotConfig,
    ) -> None:
        if type(cache_root) is not type(Path()):
            raise TypeError("cache_root must be an exact Path")
        if type(sana_config) is not SanaPilotConfig:
            raise TypeError("sana_config must be an exact SanaPilotConfig")
        if type(subjects_config) is not SubjectsPilotConfig:
            raise TypeError("subjects_config must be an exact SubjectsPilotConfig")
        sana_config.validate()
        subjects_config.validate()
        self.cache_root = cache_root
        self.sana_config = sana_config
        self.subjects_config = subjects_config
        self._private_cache_root = cache_root / "ratemem-engineering-pilot"
        self._components: PinnedComponents | None = None
        self._pipeline: object | None = None
        self._transformer: SanaTransformer2DModel | None = None
        self._adapter_bank: SanaDynamicAdapterBank | None = None
        self._amortizer: SupportAmortizer | None = None
        self._trainer: OneTimestepFlowTrainer | None = None
        self._data: PrecomputedPilotData | None = None
        self._device: torch.device | None = None
        self._provenance: CheckpointProvenance | None = None
        self._frozen_versions: tuple[tuple[int, int], ...] = ()
        self._compatibility_complete = False
        self._inference_complete = False
        self._backward_complete = False
        self._training_step_index = 0
        self._evaluation_call_count = 0
        self._inference_seconds: float | None = None
        self._backward_result: FlowStepResult | None = None
        self._last_training_result: FlowStepResult | None = None
        self._peak = CudaPeak(allocated_bytes=0, reserved_bytes=0)
        self._software: dict[str, str] = {}

    def _require_ready(
        self,
    ) -> tuple[
        SanaTransformer2DModel,
        SanaDynamicAdapterBank,
        SupportAmortizer,
        OneTimestepFlowTrainer,
        PrecomputedPilotData,
        torch.device,
    ]:
        values = (
            self._transformer,
            self._adapter_bank,
            self._amortizer,
            self._trainer,
            self._data,
            self._device,
        )
        if not self._compatibility_complete or any(value is None for value in values):
            raise RuntimeError("SANA compatibility must pass before any other probe")
        return cast(
            tuple[
                SanaTransformer2DModel,
                SanaDynamicAdapterBank,
                SupportAmortizer,
                OneTimestepFlowTrainer,
                PrecomputedPilotData,
                torch.device,
            ],
            values,
        )

    def _assert_frozen_versions(self) -> None:
        transformer, _bank, _amortizer, _trainer, _data, _device = self._require_ready()
        current = tuple(
            (id(parameter), parameter._version)
            for parameter in cast(nn.Module, transformer).parameters()
            if not parameter.requires_grad
        )
        if current != self._frozen_versions:
            raise RuntimeError("frozen SANA parameter version counters changed")

    def _update_peak(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("real SANA pilot requires CUDA")
        current = CudaPeak(
            allocated_bytes=int(torch.cuda.max_memory_allocated()),
            reserved_bytes=int(torch.cuda.max_memory_reserved()),
        )
        self._peak = CudaPeak(
            allocated_bytes=max(self._peak.allocated_bytes, current.allocated_bytes),
            reserved_bytes=max(self._peak.reserved_bytes, current.reserved_bytes),
        )

    def compatibility(self) -> dict[str, object]:
        if self._compatibility_complete:
            raise RuntimeError("checkpoint compatibility probe may run exactly once")
        if torch.cuda.device_count() != 1 or "L40S" not in torch.cuda.get_device_name(0):
            raise RuntimeError("real SANA pilot requires exactly one observed NVIDIA L40S")
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        ensure_private_directory(self._private_cache_root)
        hub_cache = self._private_cache_root / "hub"
        dataset_cache = self._private_cache_root / "datasets"
        ensure_private_directory(hub_cache)
        ensure_private_directory(dataset_cache)

        snapshots = hydrate_pinned_snapshots(self.sana_config, cache_dir=hub_cache)
        if (
            snapshots.sana.name != self.sana_config.revision
            or snapshots.dino.name != self.sana_config.support_revision
        ):
            raise RuntimeError("resolved SANA or DINO snapshot revision changed")
        components = load_pinned_components(
            self.sana_config,
            snapshots=snapshots,
            device=device,
        )
        layout = validate_production_sana_layout(
            cast(nn.Module, components.transformer),
            rank=self.sana_config.rank,
            atom_count=self.sana_config.atom_count,
        )
        if layout.projection_count != 120 or layout.atom_tensor_count != 240:
            raise RuntimeError("production SANA layout is not the locked 120-projection layout")
        pipeline = components.inference_pipeline()
        examples = hydrate_locked_examples(
            self.subjects_config,
            cache_dir=dataset_cache,
        )
        data = build_precomputed_cache(
            examples,
            components,
            self._private_cache_root / "held-in-cache",
            self.sana_config,
            self.subjects_config,
        )
        transformer = components.transformer
        bank = install_sana_dynamic_atoms(
            cast(nn.Module, transformer),
            rank=self.sana_config.rank,
            atom_count=self.sana_config.atom_count,
            expected_blocks=self.sana_config.num_blocks,
        )
        wrappers = bank.wrappers
        atom_parameters = tuple(bank.parameters())
        if (
            len(wrappers) != 120
            or sum(parameter.numel() for parameter in atom_parameters) != 8_601_600
        ):
            raise RuntimeError("installed SANA atom inventory changed")
        cast(Any, transformer).enable_gradient_checkpointing()
        cast(nn.Module, transformer).eval()
        amortizer = SupportAmortizer(
            support_dim=self.sana_config.support_feature_dim,
            description_dim=self.sana_config.text_feature_dim,
            hidden_dim=256,
            projection_count=self.sana_config.projection_count,
            atom_count=self.sana_config.atom_count,
            layers=2,
            heads=8,
        ).to(device=device, dtype=torch.float32)
        amortizer.train()
        optimizer_parameters = [*atom_parameters, *tuple(amortizer.parameters())]
        optimizer = cast(Any, torch.optim.AdamW)(
            optimizer_parameters,
            lr=self.sana_config.optimizer_lr,
            betas=self.sana_config.optimizer_betas,
            eps=self.sana_config.optimizer_eps,
            weight_decay=self.sana_config.optimizer_weight_decay,
            amsgrad=self.sana_config.optimizer_amsgrad,
            maximize=self.sana_config.optimizer_maximize,
            foreach=self.sana_config.optimizer_foreach,
            capturable=self.sana_config.optimizer_capturable,
            differentiable=self.sana_config.optimizer_differentiable,
            fused=self.sana_config.optimizer_fused,
            decoupled_weight_decay=(self.sana_config.optimizer_decoupled_weight_decay),
        )
        trainer = OneTimestepFlowTrainer(
            transformer,
            bank,
            amortizer,
            components.training_timesteps,
            components.training_sigmas,
            optimizer,
            expected_amortizer_signature=amortizer.architecture_signature,
            autocast_dtype=torch.bfloat16,
        )
        provenance = CheckpointProvenance(
            model_id=self.sana_config.model_id,
            model_revision=self.sana_config.revision,
            support_model_id=self.sana_config.support_model_id,
            support_model_revision=self.sana_config.support_revision,
        )
        self._components = components
        self._pipeline = pipeline
        self._transformer = transformer
        self._adapter_bank = bank
        self._amortizer = amortizer
        self._trainer = trainer
        self._data = data
        self._device = device
        self._provenance = provenance
        self._frozen_versions = tuple(
            (id(parameter), parameter._version)
            for parameter in cast(nn.Module, transformer).parameters()
            if not parameter.requires_grad
        )
        self._software = {
            "python": platform.python_version(),
            "torch": importlib.metadata.version("torch"),
            "diffusers": importlib.metadata.version("diffusers"),
            "peft": importlib.metadata.version("peft"),
            "transformers": importlib.metadata.version("transformers"),
            "modal": importlib.metadata.version("modal"),
        }
        self._compatibility_complete = True
        self._update_peak()
        self._assert_frozen_versions()
        return {"status": "pass"}

    def inference(self) -> dict[str, object]:
        if self._inference_complete:
            raise RuntimeError("one-step inference probe may run exactly once")
        self._require_ready()
        pipeline = self._pipeline
        device = self._device
        if pipeline is None or device is None:
            raise RuntimeError("pinned SANA inference pipeline is unavailable")
        generator = torch.Generator(device=device).manual_seed(20260824)
        generated: object | None = None

        def one_inference_call() -> None:
            nonlocal generated
            generated = cast(Any, pipeline)(
                prompt=_INFERENCE_PROMPT,
                generator=generator,
                height=1024,
                width=1024,
                guidance_scale=4.5,
                num_inference_steps=1,
            )

        self._update_peak()
        with cuda_peak() as peak:
            with torch.inference_mode():
                inference_seconds = timed(one_inference_call)
        observed = peak.get("peak")
        if type(observed) is not CudaPeak:
            raise RuntimeError("one-step inference did not produce a CUDA peak record")
        self._peak = CudaPeak(
            allocated_bytes=max(self._peak.allocated_bytes, observed.allocated_bytes),
            reserved_bytes=max(self._peak.reserved_bytes, observed.reserved_bytes),
        )
        if generated is None:
            raise RuntimeError("one-step inference returned no output")
        self._inference_seconds = inference_seconds
        self._inference_complete = True
        self._assert_frozen_versions()

        # Cache tensors are now sufficient for every remaining probe. Release the
        # heavyweight frozen VAE, text encoder, and DINO objects before backward.
        generated = None
        pipeline = None
        self._pipeline = None
        self._components = None
        gc.collect()
        torch.cuda.empty_cache()
        return {"status": "pass"}

    @staticmethod
    def _generator(device: torch.device, seed: int) -> torch.Generator:
        if type(seed) is not int or seed < 0:
            raise ValueError("pilot seed must be a nonnegative exact int")
        return torch.Generator(device=device).manual_seed(seed)

    def backward(self) -> float:
        if not self._inference_complete:
            raise RuntimeError("one-step inference must precede the backward probe")
        if self._backward_complete:
            raise RuntimeError("one-timestep backward probe may run exactly once")
        _transformer, _bank, _amortizer, trainer, data, device = self._require_ready()
        batch = FlowBatch.from_cache(data, device=device, row_indices=(0,))
        result = trainer.train_step(
            batch,
            generator=self._generator(device, 20260825),
        )
        if result.transformer_pass_count != 1:
            raise RuntimeError("backward probe used more than one transformer pass")
        if (
            result.gradients.code_l2 <= 0
            or result.gradients.atom_l2 <= 0
            or result.gradients.amortizer_l2 <= 0
        ):
            raise RuntimeError("backward probe did not produce every required gradient")
        self._backward_result = result
        self._backward_complete = True
        self._update_peak()
        self._assert_frozen_versions()
        return _finite_nonnegative(result.loss, "real backward loss")

    def training_step(self) -> float:
        if not self._backward_complete:
            raise RuntimeError("standalone backward probe must precede timed training")
        _transformer, _bank, _amortizer, trainer, data, device = self._require_ready()
        step_index = self._training_step_index
        row_index = step_index % 8
        batch = FlowBatch.from_cache(data, device=device, row_indices=(row_index,))
        result: FlowStepResult | None = None

        def one_training_step() -> None:
            nonlocal result
            result = trainer.train_step(
                batch,
                generator=self._generator(device, _TRAIN_SEED_BASE + step_index),
            )

        elapsed = timed(one_training_step)
        if type(result) is not FlowStepResult or result.transformer_pass_count != 1:
            raise RuntimeError("timed step did not execute exactly one transformer pass")
        self._last_training_result = result
        self._training_step_index += 1
        self._update_peak()
        self._assert_frozen_versions()
        return elapsed

    def _fixed_draw(
        self,
        batch: FlowBatch,
        *,
        row_index: int,
        device: torch.device,
    ) -> FlowDraw:
        generator = self._generator(device, _EVALUATION_NOISE_SEEDS[row_index])
        noise = torch.randn(
            batch.clean_latents.shape,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        timestep_indices = torch.tensor(
            [_EVALUATION_TIMESTEP_INDICES[row_index]],
            device=device,
            dtype=torch.int64,
        )
        return FlowDraw(noise=noise, timestep_indices=timestep_indices)

    def evaluate_loss(self) -> float:
        if not self._backward_complete:
            raise RuntimeError("backward probe must precede paired held-in evaluation")
        _transformer, _bank, _amortizer, trainer, data, device = self._require_ready()
        losses: list[float] = []
        for row_index in range(8):
            batch = FlowBatch.from_cache(
                data,
                device=device,
                row_indices=(row_index,),
            )
            draw = self._fixed_draw(batch, row_index=row_index, device=device)
            losses.append(
                _finite_nonnegative(
                    trainer.evaluate_loss(batch, draw=draw),
                    "paired held-in row loss",
                )
            )
        mean = math.fsum(losses) / len(losses)
        self._evaluation_call_count += 1
        self._update_peak()
        self._assert_frozen_versions()
        return _finite_nonnegative(float(mean), "paired held-in mean loss")

    def save_checkpoint(self, path: Path) -> CheckpointFileIdentity:
        _transformer, bank, amortizer, _trainer, _data, _device = self._require_ready()
        provenance = self._provenance
        if type(provenance) is not CheckpointProvenance:
            raise RuntimeError("checkpoint provenance is unavailable")
        self._assert_frozen_versions()
        identity = save_trainable_checkpoint(
            path,
            adapter_bank=bank,
            amortizer=amortizer,
            provenance=provenance,
        )
        identity.validate()
        return identity

    @property
    def peak(self) -> CudaPeak:
        if not self._compatibility_complete:
            raise RuntimeError("peak memory is unavailable before compatibility")
        return self._peak

    @property
    def software(self) -> dict[str, str]:
        if not self._software:
            raise RuntimeError("software inventory is unavailable before compatibility")
        return dict(self._software)

    @property
    def dataset_manifest(self) -> dict[str, object]:
        data = self._data
        if type(data) is not PrecomputedPilotData:
            raise RuntimeError("dataset manifest is unavailable before compatibility")
        return cast(dict[str, object], dict(data.manifest))

    @property
    def dataset_manifest_sha256(self) -> str:
        data = self._data
        if type(data) is not PrecomputedPilotData:
            raise RuntimeError("dataset receipt is unavailable before compatibility")
        return data.receipt.manifest_sha256

    def diagnostics(self) -> dict[str, object]:
        backward = self._backward_result
        latest = self._last_training_result
        return {
            "scope": "engineering_pilot_only",
            "inference_seconds": self._inference_seconds,
            "peak_allocated_bytes": self._peak.allocated_bytes,
            "peak_reserved_bytes": self._peak.reserved_bytes,
            "standalone_backward_transformer_passes": (
                None if backward is None else backward.transformer_pass_count
            ),
            "latest_training_transformer_passes": (
                None if latest is None else latest.transformer_pass_count
            ),
            "training_steps": self._training_step_index,
            "evaluation_calls": self._evaluation_call_count,
            "evaluation_rows_per_call": 8,
            "evaluation_transformer_passes_per_row": 1,
        }


def _validate_modal_ids(value: object, request: PilotRequest) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError("modal_ids must be an exact dict")
    payload = cast(dict[object, object], value)
    if any(type(key) is not str for key in payload) or set(payload) != _MODAL_ID_KEYS:
        raise ValueError("modal_ids has missing or unexpected fields")
    exact_matches = {
        "profile": "ratemem-pilot",
        "workspace": request.workspace,
        "environment": "main",
        "launch_attempt_id": request.attempt_id,
        "launch_source_sha256": request.source_sha256,
        "pilot_slot_sha256": request.slot_sha256,
        "submission_receipt_sha256": request.submission_receipt_sha256,
        "execution_receipt_semantics": _RECEIPT_SEMANTICS,
    }
    for name, expected in exact_matches.items():
        if payload[name] != expected:
            raise ValueError(f"modal_ids {name} is not bound to the launch request")
    for name in ("function_call_id", "input_id", "container_image_id"):
        if type(payload[name]) is not str or not payload[name]:
            raise ValueError(f"modal_ids {name} must be a nonempty exact string")
    if type(payload["task_id"]) is not str or not payload["task_id"]:
        raise ValueError("modal_ids task_id must be a nonempty exact string")
    _exact_int(
        payload["execution_receipt_count"],
        "execution_receipt_count",
        positive=True,
    )
    _lower_hex(payload["execution_receipts_sha256"], "execution_receipts_sha256")
    receipt_path = payload["execution_receipt_path"]
    if type(receipt_path) is not str or not receipt_path:
        raise ValueError("execution receipt path must be a nonempty exact string")
    receipt_directory = payload["execution_receipt_directory"]
    if type(receipt_directory) is not str or not receipt_directory:
        raise ValueError("execution receipt directory must be a nonempty exact string")
    absolute_receipt = Path(os.path.abspath(receipt_path))
    absolute_directory = Path(os.path.abspath(receipt_directory))
    if absolute_receipt.parent != absolute_directory:
        raise ValueError("execution receipt path must be a direct child of its directory")
    return cast(dict[str, object], dict(payload))


def _receipt_request_payload(request: PilotRequest) -> dict[str, object]:
    return {
        "attempt_id": request.attempt_id,
        "workspace": request.workspace,
        "source_sha256": request.source_sha256,
        "git_commit": request.git_commit,
        "git_diff_sha256": request.git_diff_sha256,
        "config_sha256": request.config_sha256,
        "slot_sha256": request.slot_sha256,
        "permit_sha256": request.permit_sha256,
        "submission_receipt_sha256": request.submission_receipt_sha256,
        "known_usage_before_usd": str(request.known_usage_before_usd),
        "pending_worst_case_usd": str(request.pending_worst_case_usd),
        "phase_bound_usd": str(request.phase_bound_usd),
        "rates": dict(request.rates),
        "rates_sha256": request.rates_sha256,
    }


def _assert_real_path_ancestors(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.parts[0])
    for part in absolute.parts[1:]:
        current = current / part
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise PermissionError("execution receipt paths must not contain symlinks")


def _read_descriptor(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


class _ReceiptSemanticError(ValueError):
    def __init__(self, message: str, snapshot: bytes) -> None:
        super().__init__(message)
        self.snapshot = snapshot


def _invalid_receipt_marker(request: PilotRequest, snapshot: bytes) -> bytes:
    """Describe rejected receipt evidence without copying its untrusted bytes."""

    marker = {
        "attempt_id": request.attempt_id,
        "evidence": "external_forensic_directory",
        "forensic_path": f"execution-receipts/{request.attempt_id}",
        "raw_snapshot_bytes": len(snapshot),
        "raw_snapshot_sha256": hashlib.sha256(snapshot).hexdigest(),
        "scope": "engineering_pilot_only",
        "status": "semantic_invalid",
    }
    return canonical_json_bytes(marker) + b"\n"


def _read_execution_receipts(
    modal_ids: dict[str, object],
    request: PilotRequest,
    artifact_root: Path,
) -> bytes:
    directory_path = Path(cast(str, modal_ids["execution_receipt_directory"]))
    current_path = Path(cast(str, modal_ids["execution_receipt_path"]))
    directory_path = Path(os.path.abspath(directory_path))
    current_path = Path(os.path.abspath(current_path))
    expected_directory = Path(os.path.abspath(artifact_root)) / (
        f"execution-receipts/{request.attempt_id}"
    )
    if directory_path != expected_directory:
        raise ValueError(
            "execution receipt directory must be the exact attempt directory under artifact root"
        )
    _assert_real_path_ancestors(directory_path)
    if current_path.parent != directory_path:
        raise ValueError("current execution receipt must be in the receipt directory")
    before_path = directory_path.lstat()
    if (
        not stat.S_ISDIR(before_path.st_mode)
        or before_path.st_uid != os.getuid()
        or stat.S_IMODE(before_path.st_mode) != 0o700
    ):
        raise PermissionError("execution receipt directory must be owner-only mode 0700")
    directory_descriptor = os.open(
        directory_path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        opened = os.fstat(directory_descriptor)
        if (
            (opened.st_dev, opened.st_ino) != (before_path.st_dev, before_path.st_ino)
            or opened.st_uid != os.getuid()
            or not stat.S_ISDIR(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) != 0o700
        ):
            raise RuntimeError("execution receipt directory changed during secure open")
        names = sorted(os.listdir(directory_descriptor))
        if len(names) != cast(int, modal_ids["execution_receipt_count"]):
            raise ValueError("execution receipt count differs from the exact directory")
        if current_path.name not in names:
            raise ValueError("current execution receipt is absent from its directory")
        records: list[tuple[str, bytes]] = []
        for name in names:
            if (
                len(name) != 69
                or not name.endswith(".json")
                or any(character not in "0123456789abcdef" for character in name[:-5])
            ):
                raise ValueError("execution receipt directory has an unexpected member")
            descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
                dir_fd=directory_descriptor,
            )
            try:
                before = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(before.st_mode)
                    or before.st_uid != os.getuid()
                    or stat.S_IMODE(before.st_mode) != 0o600
                    or before.st_nlink != 1
                ):
                    raise PermissionError(
                        "execution receipt must be an owner-only 0600 single-link file"
                    )
                content = _read_descriptor(descriptor)
                after = os.fstat(descriptor)
                if (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ) != (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                ) or len(content) != before.st_size:
                    raise RuntimeError("execution receipt changed while being read")
            finally:
                os.close(descriptor)
            records.append((name, content))
        snapshot = b"".join(content + b"\n" for _, content in records)
        if hashlib.sha256(snapshot).hexdigest() != modal_ids["execution_receipts_sha256"]:
            raise ValueError(
                "execution receipt snapshot SHA-256 differs from the committed snapshot"
            )
        expected_request = _receipt_request_payload(request)
        expected_keys = set(expected_request) | {
            "function_call_id",
            "input_id",
            "task_id",
            "receipt_id",
            "observed_at",
            "semantics",
        }
        try:
            for name, content in records:
                decoded = json.loads(
                    content,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"non-finite receipt constant: {value}")
                    ),
                )
                if type(decoded) is not dict or set(decoded) != expected_keys:
                    raise ValueError("execution receipt has missing or unexpected fields")
                receipt = cast(dict[str, object], decoded)
                if any(receipt[key] != value for key, value in expected_request.items()):
                    raise ValueError("execution receipt is not bound to the launch request")
                if (
                    receipt["receipt_id"] != name[:-5]
                    or receipt["semantics"] != _RECEIPT_SEMANTICS
                    or canonical_json_bytes(receipt) != content
                ):
                    raise ValueError("execution receipt identity or canonical content is invalid")
                for key in ("function_call_id", "input_id", "task_id"):
                    if type(receipt[key]) is not str or not receipt[key]:
                        raise ValueError(f"execution receipt {key} must be a nonempty exact string")
                observed_at = receipt["observed_at"]
                if type(observed_at) is not str:
                    raise TypeError("execution receipt observed_at must be an exact string")
                observed = datetime.fromisoformat(observed_at)
                if (
                    observed.tzinfo != UTC
                    or observed.isoformat(timespec="microseconds") != observed_at
                ):
                    raise ValueError(
                        "execution receipt observed_at must be canonical UTC microseconds"
                    )
                if name == current_path.name:
                    for key in ("function_call_id", "input_id", "task_id"):
                        if receipt[key] != modal_ids[key]:
                            raise ValueError(
                                "current execution receipt differs from Modal execution identity"
                            )
        except (TypeError, ValueError) as error:
            raise _ReceiptSemanticError(str(error), snapshot) from error
        after_path = directory_path.lstat()
        after_open = os.fstat(directory_descriptor)
        stable_fields = ("st_dev", "st_ino", "st_mtime_ns", "st_ctime_ns")
        if any(
            getattr(after_path, field) != getattr(before_path, field)
            or getattr(after_open, field) != getattr(opened, field)
            for field in stable_fields
        ):
            raise RuntimeError("execution receipt directory changed during snapshot")
        return snapshot
    finally:
        os.close(directory_descriptor)


def _estimated_cost(
    *,
    elapsed_seconds: float,
    request: PilotRequest,
    budget: ModalBudgetConfig,
) -> Decimal:
    elapsed = Decimal(str(_finite_nonnegative(elapsed_seconds, "pilot elapsed time")))
    rates = request.cost_rates
    estimate = (
        elapsed * request.resource_usd_per_second
        + Decimal(budget.storage_gib_bound) * rates.volume_gib_month
        + budget.non_gpu_setup_allowance_usd
    ).quantize(Decimal("0.000001"), rounding=ROUND_CEILING)
    return estimate


def _software_inventory() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": importlib.metadata.version("torch"),
        "diffusers": importlib.metadata.version("diffusers"),
        "peft": importlib.metadata.version("peft"),
        "transformers": importlib.metadata.version("transformers"),
        "modal": importlib.metadata.version("modal"),
    }


def _locked_dataset_manifest(
    sana_config: SanaPilotConfig,
    subjects_config: SubjectsPilotConfig,
) -> dict[str, object]:
    """Describe exact locked inputs when no validated runtime cache receipt exists."""

    return {
        "schema_version": "1.0.0",
        "scope": "engineering_pilot_only",
        "publication_eligible": False,
        "status": "runtime_cache_manifest_unavailable",
        "sana_revision": sana_config.revision,
        "support_revision": sana_config.support_revision,
        "dataset_id": subjects_config.dataset_id,
        "dataset_revision": subjects_config.revision,
        "dataset_config_sha256": subjects_config.canonical_sha256,
        "source_file": subjects_config.source_file,
        "source_file_sha256": subjects_config.source_file_sha256,
        "row_indices": list(subjects_config.row_indices),
        "held_in": subjects_config.held_in,
    }


def _dataset_evidence(
    backend: PilotEvidenceBackend | None,
    *,
    sana_config: SanaPilotConfig,
    subjects_config: SubjectsPilotConfig,
) -> tuple[dict[str, object], str, bool]:
    if backend is None:
        manifest = _locked_dataset_manifest(sana_config, subjects_config)
        return (
            manifest,
            hashlib.sha256(canonical_json_bytes(manifest)).hexdigest(),
            False,
        )
    try:
        manifest_value = backend.dataset_manifest
        claimed_sha = backend.dataset_manifest_sha256
    except RuntimeError:
        manifest = _locked_dataset_manifest(sana_config, subjects_config)
        return (
            manifest,
            hashlib.sha256(canonical_json_bytes(manifest)).hexdigest(),
            False,
        )
    if type(manifest_value) is not dict:
        raise TypeError("backend dataset manifest must be an exact dict")
    manifest = cast(dict[str, object], dict(manifest_value))
    actual_sha = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    if type(claimed_sha) is not str or claimed_sha != actual_sha:
        raise RuntimeError("backend dataset manifest hash does not bind its exact bytes")
    return manifest, actual_sha, True


def _observed_peak(backend: PilotEvidenceBackend | None) -> CudaPeak:
    peak: CudaPeak | None = None
    if backend is not None:
        try:
            peak = backend.peak
        except RuntimeError:
            pass
    if peak is None:
        if not torch.cuda.is_available():
            return CudaPeak(allocated_bytes=0, reserved_bytes=0)
        peak = CudaPeak(
            allocated_bytes=int(torch.cuda.max_memory_allocated()),
            reserved_bytes=int(torch.cuda.max_memory_reserved()),
        )
    if type(peak) is not CudaPeak:
        raise TypeError("backend peak must be an exact CudaPeak")
    return peak


def _not_run_result(limits: PilotLimits, status: str) -> dict[str, object]:
    if status not in {"oom", "exception"}:
        raise ValueError("incomplete pilot status must be oom or exception")
    probes = {name: {"status": "not_run"} for name in ALLOWED_PROBES}
    return {
        "status": status,
        "allowed_probe_names": list(ALLOWED_PROBES),
        "results": probes,
        **{name: dict(value) for name, value in probes.items()},
        "warmup_steps": limits.warmup_steps,
        "measured_steps": limits.measured_steps,
        "p50_step_seconds": None,
        "p95_step_seconds": None,
        "held_in_step_cap": 0,
        "one_timestep_backward_loss": None,
        "initial_flow_loss": None,
        "final_flow_loss": None,
        "transformer_passes_per_step": 1,
        "checkpoint_sha256": None,
        "checkpoint_bytes": None,
    }


def _error_payload(status: str) -> dict[str, str] | None:
    if status == "succeeded":
        return None
    if status == "probe_failed":
        return {
            "type": "ProbeFailure",
            "message": "held-in flow loss did not decrease",
        }
    if status == "oom":
        return {
            "type": "CudaOutOfMemory",
            "message": "CUDA allocation failed during the engineering pilot",
        }
    if status == "exception":
        return {
            "type": "PilotException",
            "message": "engineering pilot execution failed",
        }
    raise ValueError("pilot status is outside the closed result set")


def _attempt_payload(
    *,
    request: PilotRequest,
    modal_ids: dict[str, object],
    budget: ModalBudgetConfig,
    sana_config: SanaPilotConfig,
    subjects_config: SubjectsPilotConfig,
    software: dict[str, str],
    dataset_manifest_sha256: str,
    observed_gpu: str,
    peak: CudaPeak,
    result: dict[str, object],
    started_at: datetime,
    ended_at: datetime,
    estimated_cost: Decimal,
) -> dict[str, Any]:
    status = cast(str, result["status"])
    probe_results = cast(dict[str, dict[str, str]], result["results"])
    checkpoint_sha = result["checkpoint_sha256"]
    checkpoint_bytes = result["checkpoint_bytes"]
    checkpoint: dict[str, object] | None = None
    if checkpoint_sha is not None or checkpoint_bytes is not None:
        if type(checkpoint_sha) is not str or type(checkpoint_bytes) is not int:
            raise TypeError("checkpoint result identity must be a complete exact pair")
        checkpoint = {
            "path": "trainable.safetensors",
            "sha256": checkpoint_sha,
            "bytes": checkpoint_bytes,
        }
    return {
        "schema_version": "1.0.0",
        "scope": "engineering_pilot_only",
        "publication_eligible": False,
        "attempt_id": request.attempt_id,
        "phase": "first_pilot",
        "status": status,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "source": {
            "git_commit": request.git_commit,
            "git_diff_sha256": request.git_diff_sha256,
            "config_sha256": request.config_sha256,
        },
        "software": software | {"container_image_id": modal_ids["container_image_id"]},
        "model": {
            "model_id": sana_config.model_id,
            "revision": sana_config.revision,
            "support_model_id": sana_config.support_model_id,
            "support_revision": sana_config.support_revision,
        },
        "dataset": {
            "dataset_id": subjects_config.dataset_id,
            "revision": subjects_config.revision,
            "manifest_sha256": dataset_manifest_sha256,
            "row_indices": list(subjects_config.row_indices),
            "held_in": subjects_config.held_in,
        },
        "runtime": {
            "seed": _INFERENCE_SEED,
            "requested_gpu": "L40S",
            "observed_gpu": observed_gpu,
            "gpu_count": 1,
            "cpu_cores": 4,
            "memory_gib": 32,
            "timeout_seconds": budget.timeout_seconds,
            "peak_allocated_bytes": peak.allocated_bytes,
            "peak_reserved_bytes": peak.reserved_bytes,
        },
        "modal": {
            "profile": modal_ids["profile"],
            "workspace": modal_ids["workspace"],
            "environment": modal_ids["environment"],
            "launch_attempt_id": modal_ids["launch_attempt_id"],
            "launch_source_sha256": modal_ids["launch_source_sha256"],
            "pilot_slot_sha256": modal_ids["pilot_slot_sha256"],
            "submission_receipt_sha256": modal_ids["submission_receipt_sha256"],
            "function_call_id": modal_ids["function_call_id"],
            "input_id": modal_ids["input_id"],
            "task_id": modal_ids["task_id"],
            "execution_receipt_count": modal_ids["execution_receipt_count"],
            "execution_receipt_semantics": modal_ids["execution_receipt_semantics"],
            "retries": 0,
            "detached": False,
        },
        "cost": {
            "workspace_budget_usd": "28.00",
            "internal_limit_usd": "27.00",
            "known_usage_before_usd": str(request.known_usage_before_usd),
            "pending_worst_case_usd": str(request.pending_worst_case_usd),
            "phase_bound_usd": str(request.phase_bound_usd),
            "estimated_cost_usd": str(estimated_cost),
            "reconciliation_status": "pending",
            "reconciled_cost_usd": None,
            "rates_sha256": request.rates_sha256,
        },
        "probes": {
            "allowed_probe_names": list(ALLOWED_PROBES),
            "results": probe_results,
            "warmup_steps": result["warmup_steps"],
            "measured_steps": result["measured_steps"],
            "p50_step_seconds": result["p50_step_seconds"],
            "p95_step_seconds": result["p95_step_seconds"],
            "held_in_step_cap": result["held_in_step_cap"],
            "initial_flow_loss": result["initial_flow_loss"],
            "final_flow_loss": result["final_flow_loss"],
            "transformer_passes_per_step": 1,
        },
        "checkpoint": checkpoint,
        "files": {"checksums_sha256": "0" * 64},
        "error": _error_payload(status),
    }


def run_real_pilot(
    *,
    request: dict[str, object],
    cache_root: Path,
    artifact_root: Path,
    modal_ids: dict[str, object],
) -> dict[str, object]:
    """Run once and publish a Task 9 pending artifact for every execution outcome."""

    checked_request = PilotRequest.from_mapping(request)
    checked_modal = _validate_modal_ids(modal_ids, checked_request)
    if type(cache_root) is not type(Path()) or type(artifact_root) is not type(Path()):
        raise TypeError("cache_root and artifact_root must be exact Paths")
    locked_config_payload = _config_payload()
    locked_config_sha256 = hashlib.sha256(canonical_json_bytes(locked_config_payload)).hexdigest()
    if locked_config_sha256 != checked_request.config_sha256:
        raise ValueError("launch request config SHA-256 differs from the locked bundle")
    sana_config = SanaPilotConfig.load(_SANA_CONFIG_PATH)
    subjects_config = SubjectsPilotConfig.load(_SUBJECTS_CONFIG_PATH)
    budget = ModalBudgetConfig.load(_BUDGET_CONFIG_PATH)
    if _config_payload() != locked_config_payload:
        raise RuntimeError("locked pilot config changed while it was being loaded")
    if (
        budget.timeout_seconds != 7200
        or budget.gpu != "L40S"
        or budget.gpu_count != 1
        or budget.cpu_cores != 4
        or budget.memory_gib != 32
    ):
        raise RuntimeError("runtime budget diverged from the one-L40S pilot")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("real SANA pilot requires exactly one observed NVIDIA L40S")
    observed_gpu = torch.cuda.get_device_name(0)
    if type(observed_gpu) is not str or "L40S" not in observed_gpu:
        raise RuntimeError("real SANA pilot requires exactly one observed NVIDIA L40S")

    artifact_absolute = Path(os.path.abspath(artifact_root))
    staging_parent = cache_root / "ratemem-pilot-checkpoint-staging"
    staging_absolute = Path(os.path.abspath(staging_parent))
    if os.path.commonpath((artifact_absolute, staging_absolute)) == str(artifact_absolute):
        raise ValueError("checkpoint staging must remain outside the artifact root")
    started_at = datetime.now(UTC)
    wall_started = time.monotonic()
    limits = PilotLimits(
        warmup_steps=10,
        measured_steps=20,
        held_in_allocation_usd=budget.held_in_pilot_allocation_usd,
        resource_usd_per_second=checked_request.resource_usd_per_second,
        timeout_seconds=budget.timeout_seconds,
        shutdown_reserve_seconds=_SHUTDOWN_RESERVE_SECONDS,
    )
    backend: PilotEvidenceBackend | None = None
    runner: PilotRunner | None = None
    staging: Path | None = None
    temporary: tempfile.TemporaryDirectory[str] | None = None
    receipt_bytes: bytes
    preflight_failed = False
    receipt_semantic_invalid = False
    try:
        receipt_bytes = _read_execution_receipts(checked_modal, checked_request, artifact_root)
    except _ReceiptSemanticError as error:
        receipt_bytes = _invalid_receipt_marker(checked_request, error.snapshot)
        preflight_failed = True
        receipt_semantic_invalid = True
    try:
        if preflight_failed:
            result = _not_run_result(limits, "exception")
        else:
            ensure_private_directory(staging_parent)
            temporary = tempfile.TemporaryDirectory(
                prefix=f".{checked_request.attempt_id}-",
                dir=staging_parent,
            )
            staging = Path(temporary.name)
            os.chmod(staging, 0o700)
            backend = RealSanaPilotBackend(
                cache_root=cache_root,
                sana_config=sana_config,
                subjects_config=subjects_config,
            )
            runner = PilotRunner(backend=backend, limits=limits)
            result = runner.run(staging)
    except torch.OutOfMemoryError:
        result = _not_run_result(limits, "oom") if runner is None else runner.failure_result("oom")
    except Exception:
        result = (
            _not_run_result(limits, "exception")
            if runner is None
            else runner.failure_result("exception")
        )

    ended_at = datetime.now(UTC)
    estimated = _estimated_cost(
        elapsed_seconds=time.monotonic() - wall_started,
        request=checked_request,
        budget=budget,
    )
    phase_bound_exceeded = estimated > checked_request.phase_bound_usd
    if phase_bound_exceeded and result["status"] != "oom":
        result = dict(result)
        result["status"] = "exception"
    dataset_evidence_unavailable = False
    try:
        (
            dataset_manifest,
            dataset_manifest_sha256,
            dataset_evidence_available,
        ) = _dataset_evidence(
            backend,
            sana_config=sana_config,
            subjects_config=subjects_config,
        )
        dataset_evidence_unavailable = not dataset_evidence_available
    except Exception:
        dataset_evidence_unavailable = True
        dataset_manifest = _locked_dataset_manifest(sana_config, subjects_config)
        dataset_manifest_sha256 = hashlib.sha256(canonical_json_bytes(dataset_manifest)).hexdigest()
    if dataset_evidence_unavailable and result["status"] not in {"oom", "exception"}:
        result = dict(result)
        result["status"] = "exception"
    peak_unavailable = False
    try:
        peak = _observed_peak(backend)
    except Exception:
        if result["status"] != "oom":
            result = dict(result)
            result["status"] = "exception"
        peak_unavailable = True
        peak = CudaPeak(allocated_bytes=0, reserved_bytes=0)
    checkpoint_identity: CheckpointFileIdentity | None = None
    if result["checkpoint_sha256"] is not None:
        try:
            checkpoint_identity = CheckpointFileIdentity(
                sha256=cast(str, result["checkpoint_sha256"]),
                byte_count=cast(int, result["checkpoint_bytes"]),
            )
            checkpoint_identity.validate()
        except Exception:
            result = dict(result)
            result["status"] = "exception"
            result["checkpoint_sha256"] = None
            result["checkpoint_bytes"] = None
            checkpoint_identity = None
    diagnostics: dict[str, object] = {
        "scope": "engineering_pilot_only",
        "backend_initialized": backend is not None,
        "execution_receipt_semantic_invalid": receipt_semantic_invalid,
        "execution_receipt_evidence": (
            "external_forensic_directory"
            if receipt_semantic_invalid
            else "validated_canonical_snapshot"
        ),
        "dataset_evidence_unavailable": dataset_evidence_unavailable,
        "peak_unavailable": peak_unavailable,
        "phase_bound_exceeded": phase_bound_exceeded,
    }
    if backend is not None:
        try:
            diagnostic_value = backend.diagnostics()
            if type(diagnostic_value) is not dict:
                raise TypeError("backend diagnostics must be an exact dict")
            diagnostics = (
                diagnostic_value
                | diagnostics
                | {
                    "peak_allocated_bytes": peak.allocated_bytes,
                    "peak_reserved_bytes": peak.reserved_bytes,
                }
            )
        except Exception:
            if result["status"] != "oom":
                result = dict(result)
                result["status"] = "exception"
            diagnostics["diagnostics_unavailable"] = True
    try:
        software = _software_inventory()
    except Exception:
        if result["status"] != "oom":
            result = dict(result)
            result["status"] = "exception"
        software = {
            "python": platform.python_version(),
            "torch": "2.13.0",
            "diffusers": "0.40.0",
            "peft": "0.20.0",
            "transformers": "5.16.1",
            "modal": "1.5.4",
        }
        diagnostics["software_inventory_unavailable"] = True
    attempt = _attempt_payload(
        request=checked_request,
        modal_ids=checked_modal,
        budget=budget,
        sana_config=sana_config,
        subjects_config=subjects_config,
        software=software,
        dataset_manifest_sha256=dataset_manifest_sha256,
        observed_gpu=observed_gpu,
        peak=peak,
        result=result,
        started_at=started_at,
        ended_at=ended_at,
        estimated_cost=estimated,
    )
    attempts_root = artifact_root / "attempts"
    ensure_private_directory(attempts_root)
    attempt_root = attempts_root / checked_request.attempt_id
    try:
        with ArtifactWriter(
            attempt_root,
            attempt,
            checkpoint_identity=checkpoint_identity,
        ) as writer:
            writer.write_json("config.json", locked_config_payload)
            writer.write_json("rates.json", dict(checked_request.rates))
            writer.write_json("dataset-manifest.json", dataset_manifest)
            writer.write_bytes("execution-receipts.jsonl", receipt_bytes)
            metric_rows = (
                {
                    "scope": "engineering_pilot_only",
                    "request_permit_sha256": checked_request.permit_sha256,
                    "result": result,
                },
                diagnostics,
            )
            metrics = b"".join(canonical_json_bytes(row) + b"\n" for row in metric_rows)
            writer.write_bytes("metrics.jsonl", metrics)
            if checkpoint_identity is not None:
                if staging is None:
                    raise RuntimeError("checkpoint staging disappeared before publication")
                writer.write_checkpoint(staging / "trainable.safetensors")
            pending = writer.write_pending()
    finally:
        if temporary is not None:
            try:
                temporary.cleanup()
            except OSError:
                pass
    return {
        "attempt_id": checked_request.attempt_id,
        "status": result["status"],
        "pending_path": str(pending),
        "checkpoint_sha256": (None if checkpoint_identity is None else checkpoint_identity.sha256),
        "checkpoint_bytes": (
            None if checkpoint_identity is None else checkpoint_identity.byte_count
        ),
        "publication_eligible": False,
    }
