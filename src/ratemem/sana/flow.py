from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import FunctionType
from typing import Any, Final, cast

import torch
from diffusers import SanaTransformer2DModel
from torch import Tensor, nn

from ratemem.adapters.sana_layout import SanaDynamicAdapterBank
from ratemem.pilot.config import pilot_adamw_kwargs
from ratemem.pilot.data import PrecomputedPilotData
from ratemem.support.amortizer import AdapterPrediction, SupportAmortizer

_CACHE_KEYS: Final = (
    "clean_latents",
    "prompt_embeddings",
    "prompt_attention_mask",
    "support_features",
    "support_mask",
    "description_features",
)
_SCHEDULE_LENGTH: Final = 1000
_PRODUCTION_AMORTIZER_SIGNATURE: Final = (
    "b48d5f323a80803196ebacec91aea6c381399396639ea26b20c2f0b044bd9c9c"
)


def _normal_tensor(value: object, name: str) -> Tensor:
    if type(value) is not Tensor:
        raise TypeError(f"{name} must be an exact Tensor")
    tensor = value
    if tensor.is_inference():
        raise ValueError(f"{name} must be a normal non-inference tensor")
    if tensor.requires_grad:
        raise ValueError(f"{name} must not require gradients")
    return tensor


def _all_finite(tensor: Tensor) -> bool:
    return bool(torch.isfinite(tensor).all())


def _float_values(tensor: Tensor) -> tuple[float, ...]:
    values = cast(list[float], tensor.detach().cpu().tolist())
    return tuple(float(value) for value in values)


def _right_padded(mask: Tensor) -> bool:
    if mask.shape[1] <= 1:
        return True
    as_integer = mask.to(dtype=torch.int8)
    return not bool((as_integer[:, 1:] > as_integer[:, :-1]).any())


def _exact_device(value: object, name: str) -> torch.device:
    if type(value) is not torch.device:
        raise TypeError(f"{name} must be an exact torch.device")
    device = value
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"{name} must be a CPU or CUDA device")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested while CUDA is unavailable")
    return device


def _module_device(module: nn.Module, context: str) -> torch.device:
    tensors = tuple(module.parameters()) + tuple(module.buffers())
    if not tensors:
        raise ValueError(f"{context} must expose placement tensors")
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        raise ValueError(f"{context} tensors must share one device")
    device = next(iter(devices))
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"{context} must be on CPU or CUDA")
    return device


def _configured_int(transformer: SanaTransformer2DModel, name: str) -> int:
    value = getattr(cast(Any, transformer).config, name, None)
    if type(value) is not int or value <= 0:
        raise RuntimeError(f"transformer config {name} must be a positive exact int")
    return value


def _is_production_transformer(transformer: SanaTransformer2DModel) -> bool:
    return (
        _configured_int(transformer, "in_channels") == 32
        and _configured_int(transformer, "out_channels") == 32
        and _configured_int(transformer, "sample_size") == 32
        and _configured_int(transformer, "caption_channels") == 2304
        and _configured_int(transformer, "num_layers") == 20
    )


def _is_production_shape(
    transformer: SanaTransformer2DModel, amortizer: SupportAmortizer
) -> bool:
    return (
        _is_production_transformer(transformer)
        and amortizer.support_dim == 384
        and amortizer.description_dim == 2304
        and amortizer.hidden_dim == 256
        and amortizer.projection_count == 120
        and amortizer.atom_count == 4
        and amortizer.layers == 2
        and amortizer.heads == 8
    )


@dataclass(frozen=True, slots=True)
class FlowBatch:
    clean_latents: Tensor
    prompt_embeddings: Tensor
    prompt_attention_mask: Tensor
    support_features: Tensor
    support_mask: Tensor
    description_features: Tensor

    def as_mapping(self) -> dict[str, Tensor]:
        return {
            "clean_latents": self.clean_latents,
            "prompt_embeddings": self.prompt_embeddings,
            "prompt_attention_mask": self.prompt_attention_mask,
            "support_features": self.support_features,
            "support_mask": self.support_mask,
            "description_features": self.description_features,
        }

    @classmethod
    def from_cache(
        cls,
        cache: PrecomputedPilotData,
        *,
        device: torch.device,
        row_indices: tuple[int, ...] | None = None,
    ) -> FlowBatch:
        if type(cache) is not PrecomputedPilotData:
            raise TypeError("cache must be an exact PrecomputedPilotData")
        target = _exact_device(device, "cache target device")
        if not isinstance(cache.tensors, Mapping):
            raise TypeError("cache tensors must be a mapping")
        if tuple(cache.tensors) != _CACHE_KEYS:
            raise ValueError("cache tensor keys and order must be exactly canonical")
        expected_specs = (
            ("clean_latents", (8, 32, 32, 32), torch.float32),
            ("prompt_embeddings", (8, 300, 2304), torch.float32),
            ("prompt_attention_mask", (8, 300), torch.int64),
            ("support_features", (8, 1, 384), torch.float32),
            ("support_mask", (8, 1), torch.bool),
            ("description_features", (8, 2304), torch.float32),
        )
        sources: dict[str, Tensor] = {}
        for name, shape, dtype in expected_specs:
            tensor = _normal_tensor(cache.tensors[name], f"cache tensor {name}")
            if tensor.shape != shape:
                raise ValueError(f"cache tensor {name} has the wrong exact shape")
            if tensor.dtype is not dtype:
                raise TypeError(f"cache tensor {name} has the wrong exact dtype")
            if tensor.device.type != "cpu":
                raise ValueError(f"cache tensor {name} must remain on CPU")
            if not tensor.is_contiguous():
                raise ValueError(f"cache tensor {name} must remain contiguous")
            sources[name] = tensor

        index_tensor: Tensor | None = None
        if row_indices is not None:
            if type(row_indices) is not tuple or not row_indices:
                raise TypeError("row_indices must be a non-empty exact tuple")
            if any(type(index) is not int for index in row_indices):
                raise TypeError("every cache row index must be an exact int")
            if any(index < 0 or index >= 8 for index in row_indices):
                raise ValueError("cache row indices must be in the range 0 through 7")
            with torch.inference_mode(False), torch.no_grad():
                index_tensor = torch.tensor(row_indices, dtype=torch.int64)

        copied: dict[str, Tensor] = {}
        with torch.inference_mode(False), torch.no_grad():
            for name in _CACHE_KEYS:
                source = sources[name]
                selected = (
                    source if index_tensor is None else source.index_select(0, index_tensor)
                )
                copied[name] = selected.to(device=target).clone()
        batch = cls(**copied)
        batch._validate_base()
        return batch

    def _validate_base(self) -> torch.device:
        tensors = self.as_mapping()
        for name, tensor in tensors.items():
            _normal_tensor(tensor, name)
        clean = self.clean_latents
        prompt = self.prompt_embeddings
        prompt_mask = self.prompt_attention_mask
        support = self.support_features
        support_mask = self.support_mask
        description = self.description_features
        if clean.ndim != 4:
            raise ValueError("clean latents must be rank-4")
        if clean.shape[0] <= 0:
            raise ValueError("flow batch must contain at least one example")
        batch_size = clean.shape[0]
        if prompt.ndim != 3 or prompt.shape[0] != batch_size:
            raise ValueError("prompt embeddings must have the same batch size and rank 3")
        if prompt_mask.ndim != 2 or prompt_mask.shape != prompt.shape[:2]:
            raise ValueError("prompt attention mask shape must match prompt embeddings")
        if support.ndim != 3 or support.shape[0] != batch_size:
            raise ValueError("support features must have the same batch size and rank 3")
        if support_mask.ndim != 2 or support_mask.shape != support.shape[:2]:
            raise ValueError("support mask shape must match support features")
        if description.ndim != 2 or description.shape[0] != batch_size:
            raise ValueError("description features must have the same batch size and rank 2")
        for name, tensor in (
            ("clean latents", clean),
            ("prompt embeddings", prompt),
            ("support features", support),
            ("description features", description),
        ):
            if tensor.dtype is not torch.float32:
                raise TypeError(f"{name} must have dtype torch.float32")
        if prompt_mask.dtype is not torch.int64:
            raise TypeError("prompt attention mask must have dtype torch.int64")
        if support_mask.dtype is not torch.bool:
            raise TypeError("support mask must have dtype torch.bool")
        devices = {tensor.device for tensor in tensors.values()}
        if len(devices) != 1:
            raise ValueError("every flow batch tensor must share one device")
        device = next(iter(devices))
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("flow batch tensors must be on CPU or CUDA")
        if not bool(((prompt_mask == 0) | (prompt_mask == 1)).all()):
            raise ValueError("prompt attention mask must be binary")
        if bool((prompt_mask.sum(dim=1) == 0).any()):
            raise ValueError("every prompt must contain at least one valid token")
        if not _right_padded(prompt_mask):
            raise ValueError("prompt attention mask must be right-padded")
        if bool((support_mask.sum(dim=1) == 0).any()):
            raise ValueError("every example must contain at least one valid support")
        if not _right_padded(support_mask):
            raise ValueError("support mask must be right-padded")
        if not _all_finite(clean):
            raise ValueError("clean latents must be finite")
        if not _all_finite(prompt):
            raise ValueError("prompt embeddings must be finite")
        if not _all_finite(description):
            raise ValueError("description features must be finite")
        if not _all_finite(support[support_mask]):
            raise ValueError("valid support features must be finite")
        return device

    def validate(
        self,
        transformer: SanaTransformer2DModel,
        amortizer: SupportAmortizer,
    ) -> torch.device:
        if type(transformer) is not SanaTransformer2DModel:
            raise TypeError("transformer must be an exact SanaTransformer2DModel")
        if type(amortizer) is not SupportAmortizer:
            raise TypeError("amortizer must be an exact SupportAmortizer")
        device = self._validate_base()
        expected_clean = (
            self.clean_latents.shape[0],
            _configured_int(transformer, "in_channels"),
            _configured_int(transformer, "sample_size"),
            _configured_int(transformer, "sample_size"),
        )
        if self.clean_latents.shape != expected_clean:
            raise ValueError(f"clean latent shape must be exactly {expected_clean}")
        if self.prompt_embeddings.shape[2] != _configured_int(
            transformer, "caption_channels"
        ):
            raise ValueError("prompt embedding width must match transformer caption channels")
        if self.support_features.shape[2] != amortizer.support_dim:
            raise ValueError("support feature width must match the amortizer")
        if self.description_features.shape[1] != amortizer.description_dim:
            raise ValueError("description feature width must match the amortizer")
        if _is_production_shape(transformer, amortizer):
            if self.prompt_embeddings.shape[1] != 300:
                raise ValueError("production prompt length must be exactly 300")
            if self.support_features.shape[1] != 1:
                raise ValueError("production support count must be exactly one")
        if _module_device(cast(nn.Module, transformer), "transformer") != device:
            raise ValueError("flow batch and transformer must share one device")
        if _module_device(amortizer, "amortizer") != device:
            raise ValueError("flow batch and amortizer must share one device")
        return device


@dataclass(frozen=True, slots=True)
class FlowDraw:
    noise: Tensor
    timestep_indices: Tensor

    def validate(self, batch: FlowBatch, *, schedule_length: int) -> None:
        if type(batch) is not FlowBatch:
            raise TypeError("batch must be an exact FlowBatch")
        if type(schedule_length) is not int or schedule_length <= 0:
            raise TypeError("schedule_length must be a positive exact int")
        noise = _normal_tensor(self.noise, "flow noise")
        indices = _normal_tensor(self.timestep_indices, "timestep indices")
        if noise.dtype is not torch.float32:
            raise TypeError("flow noise must have dtype torch.float32")
        if noise.shape != batch.clean_latents.shape:
            raise ValueError("flow noise shape must exactly match clean latents")
        if noise.device != batch.clean_latents.device:
            raise ValueError("flow noise and batch must share one device")
        if not _all_finite(noise):
            raise ValueError("flow noise must be finite")
        if indices.dtype is not torch.int64:
            raise TypeError("timestep indices must have dtype torch.int64")
        if indices.shape != (batch.clean_latents.shape[0],):
            raise ValueError("timestep indices must have exact batch shape")
        if indices.device != batch.clean_latents.device:
            raise ValueError("timestep indices and batch must share one device")
        if bool(((indices < 0) | (indices >= schedule_length)).any()):
            raise ValueError("timestep indices are outside the schedule range")


@dataclass(frozen=True, slots=True)
class GradientMetrics:
    code_l2: float
    atom_l2: float
    amortizer_l2: float
    atom_tensor_count: int
    amortizer_tensor_count: int


@dataclass(frozen=True, slots=True)
class FlowStepResult:
    loss: float
    timestep_indices: tuple[int, ...]
    timesteps: tuple[float, ...]
    sigmas: tuple[float, ...]
    timestep_count: int
    transformer_pass_count: int
    gradients: GradientMetrics


def _require_flow_pair(clean: Tensor, noise: Tensor) -> None:
    if type(clean) is not Tensor or type(noise) is not Tensor:
        raise TypeError("clean and noise must be exact Tensors")
    if clean.shape != noise.shape or clean.ndim < 2 or clean.shape[0] <= 0:
        raise ValueError("clean and noise must have one exact non-empty shared shape")
    if clean.dtype is not torch.float32 or noise.dtype is not torch.float32:
        raise TypeError("clean and noise must have dtype torch.float32")
    if clean.device != noise.device:
        raise ValueError("clean and noise must share one device")
    if not _all_finite(clean) or not _all_finite(noise):
        raise ValueError("clean and noise must be finite")


def flow_interpolate(clean: Tensor, noise: Tensor, sigma: Tensor) -> Tensor:
    _require_flow_pair(clean, noise)
    if type(sigma) is not Tensor:
        raise TypeError("sigma must be an exact Tensor")
    expected_shape = (clean.shape[0], *([1] * (clean.ndim - 1)))
    if sigma.shape != expected_shape:
        raise ValueError(f"sigma shape must be exactly {expected_shape}")
    if sigma.dtype is not torch.float32 or sigma.device != clean.device:
        raise TypeError("sigma must be float32 on the clean tensor device")
    if not _all_finite(sigma) or bool(((sigma < 0) | (sigma > 1)).any()):
        raise ValueError("sigma must be finite and in the closed interval [0, 1]")
    result = (1.0 - sigma) * clean + sigma * noise
    if result.dtype is not torch.float32 or result.shape != clean.shape:
        raise RuntimeError("flow interpolation changed shape or dtype")
    if not _all_finite(result):
        raise RuntimeError("flow interpolation produced non-finite values")
    return result


def flow_target(clean: Tensor, noise: Tensor) -> Tensor:
    _require_flow_pair(clean, noise)
    result = noise - clean
    if result.dtype is not torch.float32 or not _all_finite(result):
        raise RuntimeError("flow target must remain finite float32")
    return result


def sigma_for_timesteps(
    timesteps: Tensor,
    schedule_timesteps: Tensor,
    schedule_sigmas: Tensor,
    *,
    n_dim: int,
) -> Tensor:
    for name, tensor in (
        ("timesteps", timesteps),
        ("schedule timesteps", schedule_timesteps),
        ("schedule sigmas", schedule_sigmas),
    ):
        if type(tensor) is not Tensor:
            raise TypeError(f"{name} must be an exact Tensor")
        if tensor.ndim != 1 or tensor.dtype is not torch.float32:
            raise TypeError(f"{name} must be a rank-1 float32 Tensor")
        if not _all_finite(tensor):
            raise ValueError(f"{name} must be finite")
    if type(n_dim) is not int or n_dim < 1:
        raise TypeError("n_dim must be a positive exact int")
    if schedule_timesteps.shape != schedule_sigmas.shape or not schedule_timesteps.numel():
        raise ValueError("schedule timesteps and sigmas must be non-empty and aligned")
    if not (
        timesteps.device == schedule_timesteps.device == schedule_sigmas.device
    ):
        raise ValueError("timestep lookup tensors must share one device")
    indices: list[int] = []
    for timestep in timesteps:
        matches = (schedule_timesteps == timestep).nonzero(as_tuple=False).flatten()
        if matches.numel() != 1:
            raise ValueError(
                "each sampled timestep must occur exactly once in the scheduler"
            )
        indices.append(int(matches.item()))
    selected = schedule_sigmas[
        torch.tensor(indices, dtype=torch.int64, device=schedule_sigmas.device)
    ]
    return selected.reshape(selected.shape[0], *([1] * (n_dim - 1)))


def _float32(value: float) -> float:
    return cast(float, struct.unpack("!f", struct.pack("!f", value))[0])


def _canonical_schedule_value(index: int) -> tuple[float, float]:
    sigma = _float32((_SCHEDULE_LENGTH - index) / float(_SCHEDULE_LENGTH))
    timestep = _float32(sigma * float(_SCHEDULE_LENGTH))
    return timestep, sigma


def _validate_schedule(
    timesteps: object, sigmas: object
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if type(timesteps) is not tuple or type(sigmas) is not tuple:
        raise TypeError("training schedules must be exact immutable tuples")
    typed_timesteps = cast(tuple[object, ...], timesteps)
    typed_sigmas = cast(tuple[object, ...], sigmas)
    if len(typed_timesteps) != _SCHEDULE_LENGTH or len(typed_sigmas) != _SCHEDULE_LENGTH:
        raise ValueError("training schedules must contain exactly 1000 values")
    for index, (timestep, sigma) in enumerate(
        zip(typed_timesteps, typed_sigmas, strict=True)
    ):
        if type(timestep) is not float or type(sigma) is not float:
            raise TypeError("every training schedule value must be an exact float")
        if not math.isfinite(timestep) or not math.isfinite(sigma):
            raise ValueError("training schedule values must be finite")
        expected_timestep, expected_sigma = _canonical_schedule_value(index)
        if timestep != expected_timestep or sigma != expected_sigma:
            raise ValueError(f"training schedule changed at canonical index {index}")
    return cast(tuple[float, ...], timesteps), cast(tuple[float, ...], sigmas)


def _storage_identity(tensor: Tensor) -> tuple[str, int, int]:
    return (
        str(tensor.device),
        tensor.untyped_storage().data_ptr(),
        tensor.untyped_storage().nbytes(),
    )


def _tensor_value_digest(named_tensors: Sequence[tuple[str, Tensor]]) -> str:
    """Hash exact tensor values with bounded host memory.

    The frozen production backbone is too large for a resident clone.  Reading it
    in byte chunks still provides an exact SHA-256 integrity check while keeping
    peak host memory bounded.
    """
    digest = hashlib.sha256()
    for name, tensor in named_tensors:
        if type(tensor) not in {Tensor, nn.Parameter}:
            raise RuntimeError(f"{name} must remain an exact tensor")
        if not tensor.is_contiguous() or tensor.storage_offset() != 0:
            raise RuntimeError(f"{name} must remain contiguous with zero storage offset")
        if tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size():
            raise RuntimeError(f"{name} storage must exactly cover its tensor")
        if tensor.is_floating_point() and not _all_finite(tensor):
            raise RuntimeError(f"{name} must remain finite")
        metadata = (
            name,
            tuple(tensor.shape),
            tuple(tensor.stride()),
            str(tensor.dtype),
            str(tensor.device),
        )
        encoded = repr(metadata).encode("utf-8")
        digest.update(struct.pack("!Q", len(encoded)))
        digest.update(encoded)
        raw = tensor.detach().reshape(-1).view(torch.uint8)
        chunk_size = 8 * 1024 * 1024
        for start in range(0, raw.numel(), chunk_size):
            chunk = raw[start : start + chunk_size].to(device="cpu")
            digest.update(chunk.numpy().tobytes())
    return digest.hexdigest()


def _parameter_topology(module: nn.Module) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            name,
            id(parameter),
            type(parameter),
            tuple(parameter.shape),
            tuple(parameter.stride()),
            str(parameter.device),
            parameter.dtype,
            _storage_identity(parameter),
            parameter.requires_grad,
        )
        for name, parameter in module.named_parameters(remove_duplicate=False)
    )


def _buffer_topology(module: nn.Module) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            name,
            id(buffer),
            type(buffer),
            tuple(buffer.shape),
            tuple(buffer.stride()),
            str(buffer.device),
            buffer.dtype,
            _storage_identity(buffer),
        )
        for name, buffer in module.named_buffers(remove_duplicate=False)
    )


def _module_topology(module: nn.Module) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (name, id(child), type(child))
        for name, child in module.named_modules(remove_duplicate=False)
    )


def _freeze_optimizer_value(value: object) -> object:
    if value is None or type(value) in {bool, int, float, str}:
        return value
    if type(value) is tuple:
        return tuple(_freeze_optimizer_value(item) for item in cast(tuple[object, ...], value))
    raise TypeError(f"unsupported mutable AdamW hyperparameter value: {type(value).__name__}")


def _optimizer_hyperparameters(optimizer: torch.optim.AdamW) -> tuple[object, ...]:
    if type(optimizer.defaults) is not dict:
        raise TypeError("AdamW defaults must be an exact dict")
    if type(optimizer.param_groups) is not list or len(optimizer.param_groups) != 1:
        raise ValueError("AdamW must contain exactly one parameter group")
    group = optimizer.param_groups[0]
    if type(group) is not dict:
        raise TypeError("AdamW parameter group must be an exact dict")
    defaults = tuple(
        (key, _freeze_optimizer_value(value))
        for key, value in optimizer.defaults.items()
    )
    configured = tuple(
        (key, _freeze_optimizer_value(value))
        for key, value in group.items()
        if key != "params"
    )
    return defaults, configured


def _validate_adamw_numbers(optimizer: torch.optim.AdamW) -> None:
    expected = pilot_adamw_kwargs()
    for context, configured in (
        ("defaults", optimizer.defaults),
        ("parameter group", optimizer.param_groups[0]),
    ):
        actual_keys = set(configured)
        if context == "parameter group":
            actual_keys.discard("params")
        if actual_keys != set(expected):
            raise ValueError(f"AdamW {context} keys changed from the exact pilot contract")
        for name, expected_value in expected.items():
            actual = configured[name]
            if type(actual) is not type(expected_value) or actual != expected_value:
                raise ValueError(
                    f"AdamW {name} must remain exactly {expected_value!r}"
                )


class OneTimestepFlowTrainer:
    def __init__(
        self,
        transformer: SanaTransformer2DModel,
        adapter_bank: SanaDynamicAdapterBank,
        amortizer: SupportAmortizer,
        training_timesteps: tuple[float, ...],
        training_sigmas: tuple[float, ...],
        optimizer: torch.optim.AdamW,
        *,
        expected_amortizer_signature: str,
        autocast_dtype: torch.dtype | None = None,
    ) -> None:
        if type(transformer) is not SanaTransformer2DModel:
            raise TypeError("transformer must be an exact SanaTransformer2DModel")
        if type(adapter_bank) is not SanaDynamicAdapterBank:
            raise TypeError("adapter_bank must be an exact SanaDynamicAdapterBank")
        if type(amortizer) is not SupportAmortizer:
            raise TypeError("amortizer must be an exact SupportAmortizer")
        if type(optimizer) is not torch.optim.AdamW:
            raise TypeError("optimizer must be an exact torch.optim.AdamW")
        timesteps, sigmas = _validate_schedule(training_timesteps, training_sigmas)
        if type(expected_amortizer_signature) is not str:
            raise TypeError("expected amortizer signature must be an exact str")
        if (
            len(expected_amortizer_signature) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_amortizer_signature
            )
        ):
            raise ValueError("expected amortizer signature must be lowercase SHA-256")
        if autocast_dtype not in {None, torch.bfloat16}:
            raise ValueError("autocast dtype must be None or torch.bfloat16")

        self.transformer = transformer
        self.adapter_bank = adapter_bank
        self.amortizer = amortizer
        self.training_timesteps = timesteps
        self.training_sigmas = sigmas
        self.optimizer = optimizer
        self.expected_amortizer_signature = expected_amortizer_signature
        self.autocast_dtype = autocast_dtype
        self._poison: BaseException | None = None
        self._busy = False
        if optimizer.state:
            raise ValueError("optimizer state must be empty for a new pilot trainer")
        self._optimizer_state_required = False
        self._schedule_cache: dict[torch.device, tuple[Tensor, Tensor]] = {}
        self._schedule_versions: dict[torch.device, tuple[int, int, int, int]] = {}
        self._atom_parameters = tuple(adapter_bank.parameters())
        self._amortizer_parameters = tuple(amortizer.parameters())
        self._trainable_versions_snapshot = tuple(
            (id(parameter), parameter._version)
            for parameter in self._atom_parameters + self._amortizer_parameters
        )
        transformer_module = cast(nn.Module, transformer)
        self.frozen_parameters = tuple(
            parameter
            for parameter in transformer_module.parameters()
            if not parameter.requires_grad
        )

        self._validate_initial_contract()
        self._transformer_module_topology = _module_topology(transformer_module)
        self._transformer_parameter_topology = _parameter_topology(transformer_module)
        self._transformer_buffer_topology = _buffer_topology(transformer_module)
        self._amortizer_module_topology = _module_topology(amortizer)
        self._amortizer_parameter_topology = _parameter_topology(amortizer)
        self._amortizer_buffer_topology = _buffer_topology(amortizer)
        self._frozen_versions = tuple(
            (id(parameter), parameter._version) for parameter in self.frozen_parameters
        )
        self._buffer_versions = tuple(
            (id(buffer), buffer._version) for buffer in transformer_module.buffers()
        )
        self._optimizer_hyperparameter_snapshot = _optimizer_hyperparameters(optimizer)
        self._optimizer_state_snapshot = self._optimizer_state_fingerprint()
        self._frozen_value_digest = self._value_digest("frozen")
        self._atom_value_digest = self._value_digest("atom")
        self._amortizer_value_digest = self._value_digest("amortizer")
        self._optimizer_state_value_digest = self._value_digest("optimizer state")
        self._object_ids = (
            id(transformer),
            id(adapter_bank),
            id(amortizer),
            id(optimizer),
        )
        self._device = _module_device(transformer_module, "transformer")
        self.schedule_tensors(self._device)
        self._validate_static_contract()

    def _validate_initial_contract(
        self, *, require_clear_gradients: bool = True
    ) -> None:
        transformer_ref = self.adapter_bank._transformer_ref
        if transformer_ref is None or transformer_ref() is not cast(
            nn.Module, self.transformer
        ):
            raise ValueError("adapter bank must be bound to the same transformer")
        _wrappers = self.adapter_bank.wrappers
        if self.adapter_bank.layout.projection_count != self.amortizer.projection_count:
            raise ValueError("bank and amortizer projection counts must match")
        if self.adapter_bank.layout.atom_count != self.amortizer.atom_count:
            raise ValueError("bank and amortizer atom counts must match")
        self.amortizer.assert_architecture_signature(
            self.expected_amortizer_signature
        )
        production_transformer = _is_production_transformer(self.transformer)
        if production_transformer:
            if not _is_production_shape(self.transformer, self.amortizer):
                raise ValueError("production amortizer architecture changed")
            if self.expected_amortizer_signature != _PRODUCTION_AMORTIZER_SIGNATURE:
                raise ValueError("production amortizer architecture signature changed")
        transformer_module = cast(nn.Module, self.transformer)
        if any(module.training for module in transformer_module.modules()):
            raise ValueError("every transformer module must remain in eval mode")
        if any(not module.training for module in self.amortizer.modules()):
            raise ValueError("every amortizer module must remain in train mode")
        if not cast(Any, self.transformer).is_gradient_checkpointing:
            raise ValueError("transformer gradient checkpointing must be enabled")
        checkpoint = getattr(self.transformer, "_gradient_checkpointing_func", None)
        if (
            type(checkpoint) is not FunctionType
            or checkpoint.__module__ != "diffusers.models.modeling_utils"
            or checkpoint.__qualname__
            != "ModelMixin.enable_gradient_checkpointing.<locals>._gradient_checkpointing_func"
            or "use_reentrant" not in checkpoint.__code__.co_consts
            or False not in checkpoint.__code__.co_consts
        ):
            raise ValueError(
                "gradient checkpointing must use the official non-reentrant function"
            )
        expected_transformer_trainable = {id(parameter) for parameter in self._atom_parameters}
        actual_transformer_trainable = {
            id(parameter)
            for parameter in transformer_module.parameters()
            if parameter.requires_grad
        }
        if actual_transformer_trainable != expected_transformer_trainable:
            raise ValueError("only bank atoms may be trainable in the transformer")
        if any(not parameter.requires_grad for parameter in self._atom_parameters):
            raise ValueError("every bank atom parameter must require gradients")
        if any(not parameter.requires_grad for parameter in self._amortizer_parameters):
            raise ValueError("every amortizer parameter must require gradients")
        expected = self._atom_parameters + self._amortizer_parameters
        if len({id(parameter) for parameter in expected}) != len(expected):
            raise ValueError("trainable parameter object aliases are forbidden")
        storages = [_storage_identity(parameter) for parameter in expected]
        if len(set(storages)) != len(storages):
            raise ValueError("trainable parameter storage aliases are forbidden")
        transformer_device = _module_device(transformer_module, "transformer")
        amortizer_device = _module_device(self.amortizer, "amortizer")
        if transformer_device != amortizer_device:
            raise ValueError("transformer and amortizer must share one device")
        if transformer_device.type == "cpu" and self.autocast_dtype is not None:
            raise ValueError("CPU flow training must disable autocast")
        if transformer_device.type == "cuda" and self.autocast_dtype is not torch.bfloat16:
            raise ValueError("CUDA flow training must use bfloat16 autocast")
        if production_transformer:
            if transformer_device.type != "cuda":
                raise ValueError("production SANA flow training requires CUDA")
            for name, tensor in (
                *transformer_module.named_parameters(),
                *transformer_module.named_buffers(),
            ):
                if tensor.is_floating_point() and tensor.dtype is not torch.bfloat16:
                    raise ValueError(
                        "production transformer floating tensors must be bfloat16; "
                        f"{name} has {tensor.dtype}"
                    )
        if any(parameter.dtype is not torch.float32 for parameter in self._amortizer_parameters):
            raise ValueError("every amortizer parameter must remain float32")
        self._validate_optimizer_ownership()
        _validate_adamw_numbers(self.optimizer)
        self._validate_optimizer_state(require_complete=False)
        self._assert_bank_inactive()
        if require_clear_gradients:
            self._assert_no_gradients()

    def _validate_optimizer_ownership(self) -> None:
        if type(self.optimizer.param_groups) is not list or len(self.optimizer.param_groups) != 1:
            raise ValueError("optimizer parameter ownership requires exactly one group")
        group = self.optimizer.param_groups[0]
        if type(group) is not dict or type(group.get("params")) is not list:
            raise TypeError("optimizer parameter group and params must be exact containers")
        supplied = cast(list[object], group["params"])
        if any(type(parameter) is not nn.Parameter for parameter in supplied):
            raise TypeError("optimizer params must be exact Parameters")
        supplied_ids = [id(parameter) for parameter in supplied]
        expected_ids = {
            id(parameter)
            for parameter in self._atom_parameters + self._amortizer_parameters
        }
        if len(set(supplied_ids)) != len(supplied_ids) or set(supplied_ids) != expected_ids:
            raise ValueError("optimizer parameter ownership must be exactly atoms plus amortizer")

    def _validate_optimizer_state(self, *, require_complete: bool) -> None:
        expected = self._atom_parameters + self._amortizer_parameters
        expected_ids = {id(parameter) for parameter in expected}
        if any(id(parameter) not in expected_ids for parameter in self.optimizer.state):
            raise RuntimeError("optimizer state contains an unexpected parameter")
        populated = [parameter for parameter in expected if self.optimizer.state.get(parameter)]
        if populated and len(populated) != len(expected):
            raise RuntimeError("optimizer state is only partially populated")
        if require_complete and len(populated) != len(expected):
            raise RuntimeError("optimizer state is incomplete after AdamW step")
        state_storages = {_storage_identity(parameter) for parameter in expected}
        amsgrad = self.optimizer.param_groups[0].get("amsgrad") is True
        expected_keys = (
            {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
            if amsgrad
            else {"step", "exp_avg", "exp_avg_sq"}
        )
        for parameter in populated:
            state = self.optimizer.state[parameter]
            if type(state) is not dict or set(state) != expected_keys:
                raise RuntimeError("AdamW state keys changed from the exact contract")
            step = state["step"]
            if type(step) is not Tensor or step.numel() != 1 or not _all_finite(step):
                raise RuntimeError("AdamW step state must be one finite Tensor")
            if (
                step.dtype is not torch.float32
                or step.requires_grad
                or step.is_inference()
            ):
                raise RuntimeError("AdamW step state must be normal detached float32")
            step_identity = _storage_identity(step)
            if step_identity in state_storages:
                raise RuntimeError("AdamW state storage aliases are forbidden")
            state_storages.add(step_identity)
            if float(step) < 0:
                raise RuntimeError("AdamW step state must be nonnegative")
            for name in expected_keys - {"step"}:
                moment = state[name]
                if type(moment) is not Tensor:
                    raise RuntimeError(f"AdamW {name} state must be an exact Tensor")
                if moment.shape != parameter.shape or moment.device != parameter.device:
                    raise RuntimeError(f"AdamW {name} state placement or shape changed")
                if moment.dtype is not parameter.dtype:
                    raise RuntimeError(f"AdamW {name} state dtype changed")
                if moment.stride() != parameter.stride():
                    raise RuntimeError(f"AdamW {name} state stride changed")
                if moment.requires_grad or moment.is_inference():
                    raise RuntimeError(f"AdamW {name} state must be normal and detached")
                if not moment.is_floating_point() or not _all_finite(moment):
                    raise RuntimeError(f"AdamW {name} state must be finite floating point")
                identity = _storage_identity(moment)
                if identity in state_storages:
                    raise RuntimeError("AdamW moment storage aliases are forbidden")
                state_storages.add(identity)

        self._validate_global_storage_aliases()

    def _named_value_tensors(self, context: str) -> tuple[tuple[str, Tensor], ...]:
        transformer = cast(nn.Module, self.transformer)
        if context == "frozen":
            return tuple(
                (f"frozen parameter {name}", parameter)
                for name, parameter in transformer.named_parameters()
                if not parameter.requires_grad
            ) + tuple(
                (f"frozen buffer {name}", buffer)
                for name, buffer in transformer.named_buffers()
            )
        if context == "atom":
            return tuple(
                (f"atom parameter {index}", parameter)
                for index, parameter in enumerate(self._atom_parameters)
            )
        if context == "amortizer":
            return tuple(
                (f"amortizer parameter {name}", parameter)
                for name, parameter in self.amortizer.named_parameters()
            ) + tuple(
                (f"amortizer buffer {name}", buffer)
                for name, buffer in self.amortizer.named_buffers()
            )
        if context == "optimizer state":
            result: list[tuple[str, Tensor]] = []
            for index, parameter in enumerate(
                self._atom_parameters + self._amortizer_parameters
            ):
                state = self.optimizer.state.get(parameter, {})
                for name in sorted(state):
                    value = state[name]
                    if type(value) is Tensor:
                        result.append((f"optimizer state {index}.{name}", value))
            return tuple(result)
        raise AssertionError(f"unknown value digest context: {context}")

    def _value_digest(self, context: str) -> str:
        return _tensor_value_digest(self._named_value_tensors(context))

    def _validate_global_storage_aliases(self) -> None:
        transformer = cast(nn.Module, self.transformer)
        named: list[tuple[str, Tensor]] = [
            (f"transformer parameter {name}", tensor)
            for name, tensor in transformer.named_parameters(remove_duplicate=False)
        ]
        named.extend(
            (f"transformer buffer {name}", tensor)
            for name, tensor in transformer.named_buffers(remove_duplicate=False)
        )
        named.extend(
            (f"amortizer parameter {name}", tensor)
            for name, tensor in self.amortizer.named_parameters(remove_duplicate=False)
        )
        named.extend(
            (f"amortizer buffer {name}", tensor)
            for name, tensor in self.amortizer.named_buffers(remove_duplicate=False)
        )
        for index, parameter in enumerate(
            self._atom_parameters + self._amortizer_parameters
        ):
            for state_name, value in self.optimizer.state.get(parameter, {}).items():
                if type(value) is Tensor:
                    named.append((f"optimizer state {index}.{state_name}", value))
        owners: dict[tuple[str, int, int], str] = {}
        for name, tensor in named:
            identity = _storage_identity(tensor)
            previous = owners.get(identity)
            if previous is not None:
                raise RuntimeError(
                    f"global tensor storage alias is forbidden: {previous} and {name}"
                )
            owners[identity] = name

    def _optimizer_state_fingerprint(self) -> tuple[object, ...]:
        result: list[object] = []
        for parameter in self._atom_parameters + self._amortizer_parameters:
            state = self.optimizer.state.get(parameter)
            if not state:
                result.append((id(parameter), None))
                continue
            entries: list[object] = []
            for name, value in state.items():
                if type(value) is Tensor:
                    entries.append(
                        (
                            name,
                            id(value),
                            tuple(value.shape),
                            tuple(value.stride()),
                            str(value.device),
                            value.dtype,
                            _storage_identity(value),
                            value._version,
                            value.requires_grad,
                            value.is_inference(),
                        )
                    )
                else:
                    entries.append((name, type(value), value))
            result.append((id(parameter), tuple(entries)))
        return tuple(result)

    def _assert_bank_inactive(self) -> None:
        for wrapper in self.adapter_bank.wrappers:
            if (
                wrapper._coefficients is not None
                or wrapper._activation_token is not None
                or wrapper._coefficient_version is not None
            ):
                raise RuntimeError("adapter bank must be inactive outside one model call")

    def _assert_no_gradients(self) -> None:
        for parameter in tuple(cast(nn.Module, self.transformer).parameters()) + tuple(
            self.amortizer.parameters()
        ):
            if parameter.grad is not None:
                raise RuntimeError("all parameter gradients must be clear before a flow call")

    def _clear_gradients(self) -> None:
        for parameter in tuple(cast(nn.Module, self.transformer).parameters()) + tuple(
            self.amortizer.parameters()
        ):
            parameter.grad = None

    def _validate_static_contract(
        self, *, require_clear_gradients: bool = True
    ) -> None:
        if (
            id(self.transformer),
            id(self.adapter_bank),
            id(self.amortizer),
            id(self.optimizer),
        ) != self._object_ids:
            raise RuntimeError("trainer component identity changed")
        try:
            _validate_schedule(self.training_timesteps, self.training_sigmas)
            self._validate_initial_contract(
                require_clear_gradients=require_clear_gradients
            )
        except (TypeError, ValueError) as error:
            raise RuntimeError(str(error)) from error
        transformer_module = cast(nn.Module, self.transformer)
        if _module_topology(transformer_module) != self._transformer_module_topology:
            raise RuntimeError("transformer module topology changed")
        if _parameter_topology(transformer_module) != self._transformer_parameter_topology:
            raise RuntimeError("transformer parameter topology changed")
        if _buffer_topology(transformer_module) != self._transformer_buffer_topology:
            raise RuntimeError("transformer buffer topology changed")
        if _module_topology(self.amortizer) != self._amortizer_module_topology:
            raise RuntimeError("amortizer module topology changed")
        if _parameter_topology(self.amortizer) != self._amortizer_parameter_topology:
            raise RuntimeError("amortizer parameter topology changed")
        if _buffer_topology(self.amortizer) != self._amortizer_buffer_topology:
            raise RuntimeError("amortizer buffer topology changed")
        frozen_versions = tuple(
            (id(parameter), parameter._version) for parameter in self.frozen_parameters
        )
        if frozen_versions != self._frozen_versions:
            raise RuntimeError("frozen transformer parameter versions changed")
        buffer_versions = tuple(
            (id(buffer), buffer._version) for buffer in transformer_module.buffers()
        )
        if buffer_versions != self._buffer_versions:
            raise RuntimeError("frozen transformer buffer versions changed")
        trainable_versions = tuple(
            (id(parameter), parameter._version)
            for parameter in self._atom_parameters + self._amortizer_parameters
        )
        if trainable_versions != self._trainable_versions_snapshot:
            raise RuntimeError("trainable parameter versions changed outside the trainer")
        self._validate_optimizer_ownership()
        if _optimizer_hyperparameters(self.optimizer) != self._optimizer_hyperparameter_snapshot:
            raise RuntimeError("optimizer hyperparameters changed")
        _validate_adamw_numbers(self.optimizer)
        self._validate_optimizer_state(
            require_complete=(
                self._optimizer_state_required or bool(self.optimizer.state)
            )
        )
        if self._optimizer_state_fingerprint() != self._optimizer_state_snapshot:
            raise RuntimeError("optimizer state changed outside the trainer")
        for context, expected in (
            ("frozen", self._frozen_value_digest),
            ("atom", self._atom_value_digest),
            ("amortizer", self._amortizer_value_digest),
            ("optimizer state", self._optimizer_state_value_digest),
        ):
            if self._value_digest(context) != expected:
                raise RuntimeError(f"{context} value digest changed outside the trainer")
        self._assert_bank_inactive()
        if require_clear_gradients:
            self._assert_no_gradients()
        self.schedule_tensors(self._device)

    def schedule_tensors(self, device: torch.device) -> tuple[Tensor, Tensor]:
        target = _exact_device(device, "schedule device")
        cached = self._schedule_cache.get(target)
        if cached is None:
            with torch.inference_mode(False), torch.no_grad():
                cached = (
                    torch.tensor(self.training_timesteps, dtype=torch.float32, device=target),
                    torch.tensor(self.training_sigmas, dtype=torch.float32, device=target),
                )
            self._schedule_cache[target] = cached
            self._schedule_versions[target] = (
                cached[0]._version,
                cached[1]._version,
                cached[0].untyped_storage().data_ptr(),
                cached[1].untyped_storage().data_ptr(),
            )
        expected_versions = self._schedule_versions[target]
        if (
            cached[0]._version,
            cached[1]._version,
            cached[0].untyped_storage().data_ptr(),
            cached[1].untyped_storage().data_ptr(),
        ) != expected_versions:
            raise RuntimeError("cached training schedule storage or versions changed")
        for tensor, expected, context in (
            (cached[0], self.training_timesteps, "timesteps"),
            (cached[1], self.training_sigmas, "sigmas"),
        ):
            if (
                type(tensor) is not Tensor
                or tensor.shape != (_SCHEDULE_LENGTH,)
                or tensor.dtype is not torch.float32
                or tensor.device != target
                or tensor.requires_grad
                or tensor.is_inference()
                or _float_values(tensor) != expected
            ):
                raise RuntimeError(f"cached training {context} changed")
        return cached

    def _require_available(self) -> None:
        if self._poison is not None:
            raise RuntimeError("flow trainer is permanently poisoned") from self._poison
        if self._busy:
            raise RuntimeError("flow trainer is already executing")

    def _validate_generator(self, generator: object, device: torch.device) -> torch.Generator:
        if type(generator) is not torch.Generator:
            raise TypeError("generator must be an exact torch.Generator")
        typed = generator
        if typed.device != device:
            raise ValueError("generator device must exactly match the flow batch device")
        return typed

    def _prediction(self, batch: FlowBatch) -> AdapterPrediction:
        prediction = self.amortizer(
            batch.support_features,
            batch.support_mask,
            batch.description_features,
        )
        if type(prediction) is not AdapterPrediction:
            raise TypeError("amortizer must return an exact AdapterPrediction")
        expected_shape = (
            batch.clean_latents.shape[0],
            self.adapter_bank.layout.projection_count,
            self.adapter_bank.layout.atom_count,
        )
        if (
            prediction.logits.shape != expected_shape
            or prediction.coefficients.shape != expected_shape
            or prediction.scales.shape
            != (self.adapter_bank.layout.projection_count, 1)
        ):
            raise ValueError("amortizer prediction has an invalid exact shape")
        if any(
            tensor.dtype is not torch.float32
            for tensor in (prediction.logits, prediction.scales, prediction.coefficients)
        ):
            raise TypeError("amortizer prediction must remain float32")
        if any(
            tensor.device != batch.clean_latents.device
            for tensor in (prediction.logits, prediction.scales, prediction.coefficients)
        ):
            raise ValueError("amortizer prediction must remain on the batch device")
        if any(
            not _all_finite(tensor)
            for tensor in (prediction.logits, prediction.scales, prediction.coefficients)
        ):
            raise ValueError("amortizer prediction must remain finite")
        return prediction

    def _model_loss(
        self,
        batch: FlowBatch,
        draw: FlowDraw,
        *,
        gradients: bool,
    ) -> tuple[Tensor, Tensor, Tensor, int]:
        schedule_timesteps, schedule_sigmas = self.schedule_tensors(
            batch.clean_latents.device
        )
        timesteps = schedule_timesteps[draw.timestep_indices]
        sigmas = schedule_sigmas[draw.timestep_indices]
        shaped_sigmas = sigmas.reshape(
            sigmas.shape[0], *([1] * (batch.clean_latents.ndim - 1))
        )
        noisy = flow_interpolate(batch.clean_latents, draw.noise, shaped_sigmas)
        target = flow_target(batch.clean_latents, draw.noise)
        prediction = self._prediction(batch)
        flat_coefficients = prediction.coefficients.reshape(
            batch.clean_latents.shape[0], -1
        )
        if gradients:
            flat_coefficients.retain_grad()
        pass_count = 0
        with self.adapter_bank.activate(flat_coefficients):
            with torch.autocast(
                device_type=batch.clean_latents.device.type,
                dtype=self.autocast_dtype,
                enabled=self.autocast_dtype is not None,
            ):
                output: object = cast(Any, self.transformer)(
                    hidden_states=noisy,
                    encoder_hidden_states=batch.prompt_embeddings,
                    encoder_attention_mask=batch.prompt_attention_mask,
                    timestep=timesteps,
                    return_dict=False,
                )
                pass_count += 1
            if type(output) is not tuple or len(output) != 1 or type(output[0]) is not Tensor:
                raise TypeError("SANA output must be an exact one-tensor tuple")
            model_prediction = output[0]
            if model_prediction.shape != batch.clean_latents.shape:
                raise ValueError("SANA prediction shape must exactly match clean latents")
            if model_prediction.device != batch.clean_latents.device:
                raise ValueError("SANA prediction device must match clean latents")
            expected_dtype = (
                torch.float32
                if self.autocast_dtype is None
                else torch.bfloat16
            )
            if model_prediction.dtype is not expected_dtype:
                raise TypeError(
                    f"SANA prediction dtype must be exactly {expected_dtype}"
                )
            if not _all_finite(model_prediction):
                raise ValueError("SANA prediction must be finite floating point")
            errors = model_prediction.float() - target
            per_example = errors.square().flatten(start_dim=1).mean(dim=1)
            loss = per_example.mean()
            if loss.shape != () or loss.dtype is not torch.float32 or not _all_finite(loss):
                raise RuntimeError("flow loss must be one finite float32 scalar")
            if gradients:
                if not loss.requires_grad:
                    raise RuntimeError("training flow loss must require gradients")
                loss.backward()  # type: ignore[no-untyped-call]
        return loss, flat_coefficients, timesteps, pass_count

    @staticmethod
    def _gradient_l2(parameters: Sequence[nn.Parameter], context: str) -> float:
        total = 0.0
        for parameter in parameters:
            gradient = parameter.grad
            if type(gradient) is not Tensor:
                raise RuntimeError(f"every {context} tensor must receive a gradient")
            if gradient.shape != parameter.shape or gradient.device != parameter.device:
                raise RuntimeError(f"{context} gradient shape or placement changed")
            if not gradient.is_floating_point() or not _all_finite(gradient):
                raise RuntimeError(f"every {context} gradient must be finite floating point")
            total += float(gradient.detach().double().square().sum())
        l2 = math.sqrt(total)
        if not math.isfinite(l2) or l2 <= 0:
            raise RuntimeError(f"aggregate {context} gradient norm must be finite and nonzero")
        return l2

    def _gradient_metrics(self, coefficients: Tensor) -> GradientMetrics:
        code_gradient = coefficients.grad
        if type(code_gradient) is not Tensor:
            raise RuntimeError("dynamic code must receive a gradient")
        if code_gradient.shape != coefficients.shape or not _all_finite(code_gradient):
            raise RuntimeError("dynamic code gradient must be finite with exact shape")
        code_l2 = float(code_gradient.detach().double().norm())
        if not math.isfinite(code_l2) or code_l2 <= 0:
            raise RuntimeError("dynamic code gradient norm must be finite and nonzero")
        atom_l2 = self._gradient_l2(self._atom_parameters, "atom")
        amortizer_l2 = self._gradient_l2(
            self._amortizer_parameters, "amortizer"
        )
        if any(parameter.grad is not None for parameter in self.frozen_parameters):
            raise RuntimeError("frozen transformer parameters must never receive gradients")
        return GradientMetrics(
            code_l2=code_l2,
            atom_l2=atom_l2,
            amortizer_l2=amortizer_l2,
            atom_tensor_count=len(self._atom_parameters),
            amortizer_tensor_count=len(self._amortizer_parameters),
        )

    def train_step(
        self,
        batch: FlowBatch,
        *,
        generator: torch.Generator,
    ) -> FlowStepResult:
        self._require_available()
        if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
            raise RuntimeError("training requires gradients enabled outside inference mode")
        if type(batch) is not FlowBatch:
            raise TypeError("batch must be an exact FlowBatch")
        device = batch.validate(self.transformer, self.amortizer)
        checked_generator = self._validate_generator(generator, device)
        self._validate_static_contract()
        self._busy = True
        primary_error: BaseException | None = None
        try:
            self.optimizer.zero_grad(set_to_none=True)
            self._assert_no_gradients()
            noise = torch.randn(
                batch.clean_latents.shape,
                generator=checked_generator,
                device=device,
                dtype=torch.float32,
            )
            timestep_indices = torch.randint(
                0,
                _SCHEDULE_LENGTH,
                (batch.clean_latents.shape[0],),
                generator=checked_generator,
                device=device,
                dtype=torch.int64,
            )
            draw = FlowDraw(noise=noise, timestep_indices=timestep_indices)
            draw.validate(batch, schedule_length=_SCHEDULE_LENGTH)
            # The production trainable state is small enough to clone (roughly tens of
            # MiB). Full trainable snapshots give a value-change proof without copying
            # any frozen SANA backbone tensor.
            atom_values_before = tuple(
                parameter.detach().clone() for parameter in self._atom_parameters
            )
            amortizer_values_before = tuple(
                parameter.detach().clone()
                for parameter in self._amortizer_parameters
            )
            atom_versions = tuple(parameter._version for parameter in self._atom_parameters)
            amortizer_versions = tuple(
                parameter._version for parameter in self._amortizer_parameters
            )
            loss, coefficients, timesteps, pass_count = self._model_loss(
                batch, draw, gradients=True
            )
            gradients = self._gradient_metrics(coefficients)
            for context, expected in (
                ("frozen", self._frozen_value_digest),
                ("atom", self._atom_value_digest),
                ("amortizer", self._amortizer_value_digest),
                ("optimizer state", self._optimizer_state_value_digest),
            ):
                if self._value_digest(context) != expected:
                    raise RuntimeError(
                        f"{context} value digest changed during forward or backward"
                    )
            self.optimizer.step()
            if not any(
                parameter._version != version
                for parameter, version in zip(
                    self._atom_parameters, atom_versions, strict=True
                )
            ):
                raise RuntimeError("AdamW did not update any atom parameter")
            if not any(
                parameter._version != version
                for parameter, version in zip(
                    self._amortizer_parameters, amortizer_versions, strict=True
                )
            ):
                raise RuntimeError("AdamW did not update any amortizer parameter")
            if not any(
                not torch.equal(parameter, original)
                for parameter, original in zip(
                    self._atom_parameters, atom_values_before, strict=True
                )
            ):
                raise RuntimeError("AdamW did not change any atom parameter value")
            if not any(
                not torch.equal(parameter, original)
                for parameter, original in zip(
                    self._amortizer_parameters,
                    amortizer_values_before,
                    strict=True,
                )
            ):
                raise RuntimeError("AdamW did not change any amortizer parameter value")
            self._validate_optimizer_state(require_complete=True)
            self._optimizer_state_required = True
            self._optimizer_state_snapshot = self._optimizer_state_fingerprint()
            self._trainable_versions_snapshot = tuple(
                (id(parameter), parameter._version)
                for parameter in self._atom_parameters + self._amortizer_parameters
            )
            for parameter in self._atom_parameters + self._amortizer_parameters:
                if not _all_finite(parameter):
                    raise RuntimeError("AdamW produced non-finite trainable parameters")
            if self._value_digest("frozen") != self._frozen_value_digest:
                raise RuntimeError("frozen value digest changed during AdamW step")
            self._atom_value_digest = self._value_digest("atom")
            self._amortizer_value_digest = self._value_digest("amortizer")
            self._optimizer_state_value_digest = self._value_digest("optimizer state")
            self._assert_bank_inactive()
            frozen_versions = tuple(
                (id(parameter), parameter._version) for parameter in self.frozen_parameters
            )
            if frozen_versions != self._frozen_versions:
                raise RuntimeError("frozen transformer parameter versions changed")
            self._validate_static_contract(require_clear_gradients=False)
            schedule_timesteps, schedule_sigmas = self.schedule_tensors(device)
            selected_sigmas = schedule_sigmas[draw.timestep_indices]
            result = FlowStepResult(
                loss=float(loss.detach()),
                timestep_indices=tuple(
                    int(value) for value in draw.timestep_indices.detach().cpu()
                ),
                timesteps=tuple(float(value) for value in timesteps.detach().cpu()),
                sigmas=tuple(float(value) for value in selected_sigmas.detach().cpu()),
                timestep_count=draw.timestep_indices.numel(),
                transformer_pass_count=pass_count,
                gradients=gradients,
            )
            return result
        except BaseException as error:
            primary_error = error
            self._poison = error
            raise
        finally:
            try:
                self._clear_gradients()
                self._assert_bank_inactive()
            except BaseException as cleanup_error:
                if self._poison is None:
                    self._poison = cleanup_error
                if primary_error is None:
                    raise
            finally:
                self._busy = False

    def evaluate_loss(self, batch: FlowBatch, *, draw: FlowDraw) -> float:
        self._require_available()
        if type(batch) is not FlowBatch:
            raise TypeError("batch must be an exact FlowBatch")
        batch.validate(self.transformer, self.amortizer)
        if type(draw) is not FlowDraw:
            raise TypeError("draw must be an exact FlowDraw")
        draw.validate(batch, schedule_length=_SCHEDULE_LENGTH)
        self._validate_static_contract()
        trainable_versions = tuple(
            parameter._version
            for parameter in self._atom_parameters + self._amortizer_parameters
        )
        optimizer_state_keys = tuple(
            (id(parameter), tuple(state), tuple(
                (name, value._version if type(value) is Tensor else value)
                for name, value in state.items()
            ))
            for parameter, state in self.optimizer.state.items()
        )
        self._busy = True
        primary_error: BaseException | None = None
        try:
            with torch.no_grad():
                loss, _coefficients, _timesteps, pass_count = self._model_loss(
                    batch, draw, gradients=False
                )
            if pass_count != 1:
                raise RuntimeError("evaluation must use exactly one transformer pass")
            current_versions = tuple(
                parameter._version
                for parameter in self._atom_parameters + self._amortizer_parameters
            )
            if current_versions != trainable_versions:
                raise RuntimeError("evaluation modified trainable parameters")
            current_optimizer_state = tuple(
                (id(parameter), tuple(state), tuple(
                    (name, value._version if type(value) is Tensor else value)
                    for name, value in state.items()
                ))
                for parameter, state in self.optimizer.state.items()
            )
            if current_optimizer_state != optimizer_state_keys:
                raise RuntimeError("evaluation modified optimizer state")
            self._validate_static_contract()
            return float(loss)
        except BaseException as error:
            primary_error = error
            self._poison = error
            raise
        finally:
            try:
                self._clear_gradients()
                self._assert_bank_inactive()
            except BaseException as cleanup_error:
                if self._poison is None:
                    self._poison = cleanup_error
                if primary_error is None:
                    raise
            finally:
                self._busy = False
