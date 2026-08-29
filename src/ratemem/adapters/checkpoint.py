"""Deterministic, transactional interchange for one trainable checkpoint file.

Directory-level transactions across a checkpoint and sibling artifact files belong to the
artifact writer; this module owns secure create-only publication of exactly one file.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast
from weakref import ReferenceType, ref

import safetensors
import torch
from safetensors import safe_open
from safetensors.torch import save
from torch import Tensor, nn

from ratemem.adapters.dynamic_atom_linear import DynamicAtomLinear
from ratemem.adapters.sana_layout import (
    SANA_LAYOUT_VERSION,
    SanaAdapterLayout,
    SanaDynamicAdapterBank,
)
from ratemem.support.amortizer import SupportAmortizer

_FORMAT: Final = "safetensors"
_FRAMEWORK: Final = "pt"
_SCHEMA: Final = "ratemem-trainable-checkpoint"
_SCHEMA_VERSION: Final = "1.0.0"
_METADATA_KEY: Final = "ratemem"
_SERIALIZER_VERSION: Final = "0.8.0"
_SHA256_LENGTH: Final = 64
_REVISION_LENGTH: Final = 40
_MAX_HEADER_BYTES: Final = 100_000_000
_CHUNK_BYTES: Final = 1024 * 1024

_DTYPE_TO_SAFE: Final = {
    torch.float32: "F32",
    torch.bfloat16: "BF16",
}
_SAFE_DTYPE_PRIORITY: Final = {"F32": 2, "BF16": 1}
_SAFE_DTYPE_BYTES: Final = {"F32": 4, "BF16": 2}


def _require_serializer_runtime() -> None:
    if type(safetensors.__version__) is not str or safetensors.__version__ != _SERIALIZER_VERSION:
        raise RuntimeError(f"checkpoint interchange requires safetensors {_SERIALIZER_VERSION}")


def _require_exact_string(value: object, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be an exact str")
    checked = value
    if not checked:
        raise ValueError(f"{name} must be nonempty")
    return checked


def _require_lower_hex(value: object, name: str, length: int) -> str:
    checked = _require_exact_string(value, name)
    if len(checked) != length or any(character not in "0123456789abcdef" for character in checked):
        raise ValueError(f"{name} must be {length} lowercase hexadecimal characters")
    return checked


def _require_positive_exact_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int")
    checked = value
    if checked <= 0:
        raise ValueError(f"{name} must be positive")
    return checked


@dataclass(frozen=True, slots=True)
class CheckpointProvenance:
    model_id: str
    model_revision: str
    support_model_id: str
    support_model_revision: str

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _require_exact_string(self.model_id, "model_id")
        _require_lower_hex(self.model_revision, "model_revision", _REVISION_LENGTH)
        _require_exact_string(self.support_model_id, "support_model_id")
        _require_lower_hex(
            self.support_model_revision,
            "support_model_revision",
            _REVISION_LENGTH,
        )


@dataclass(frozen=True, slots=True)
class CheckpointFileIdentity:
    sha256: str
    byte_count: int

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _require_lower_hex(self.sha256, "sha256", _SHA256_LENGTH)
        _require_positive_exact_int(self.byte_count, "byte_count")


@dataclass(frozen=True, slots=True)
class TrainableCheckpointMetadata:
    provenance: CheckpointProvenance
    layout_version: str
    num_blocks: int
    rank: int
    atom_count: int
    projection_count: int
    code_dim: int
    atom_tensor_count: int
    amortizer_architecture_canonical: str
    amortizer_architecture_sha256: str
    bank_tensor_count: int
    amortizer_tensor_count: int
    total_tensor_count: int
    tensor_spec_sha256: str

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if type(self.provenance) is not CheckpointProvenance:
            raise TypeError("provenance must be an exact CheckpointProvenance")
        self.provenance.validate()
        if _require_exact_string(self.layout_version, "layout_version") != SANA_LAYOUT_VERSION:
            raise ValueError(f"layout_version must be {SANA_LAYOUT_VERSION}")
        for name in (
            "num_blocks",
            "rank",
            "atom_count",
            "projection_count",
            "code_dim",
            "atom_tensor_count",
            "bank_tensor_count",
            "amortizer_tensor_count",
            "total_tensor_count",
        ):
            _require_positive_exact_int(getattr(self, name), name)
        _require_exact_string(
            self.amortizer_architecture_canonical,
            "amortizer_architecture_canonical",
        )
        _require_lower_hex(
            self.amortizer_architecture_sha256,
            "amortizer_architecture_sha256",
            _SHA256_LENGTH,
        )
        architecture = _strict_json(
            self.amortizer_architecture_canonical,
            "amortizer_architecture_canonical",
        )
        architecture_integer_keys = (
            "atom_count",
            "description_dim",
            "heads",
            "hidden_dim",
            "layers",
            "projection_count",
            "support_dim",
        )
        expected_architecture_keys = {*architecture_integer_keys, "schema_version"}
        if set(architecture) != expected_architecture_keys:
            raise ValueError("amortizer architecture has missing or unexpected keys")
        if architecture["schema_version"] != "ratemem-support-amortizer-v1":
            raise ValueError("amortizer architecture schema_version does not match")
        for name in architecture_integer_keys:
            _require_positive_exact_int(architecture[name], f"amortizer architecture {name}")
        if architecture["atom_count"] != self.atom_count:
            raise ValueError("amortizer architecture atom_count does not match layout")
        if architecture["projection_count"] != self.projection_count:
            raise ValueError("amortizer architecture projection_count does not match layout")
        if _canonical_json(architecture) != self.amortizer_architecture_canonical:
            raise ValueError("amortizer_architecture_canonical must be canonical JSON")
        architecture_sha256 = _sha256_bytes(self.amortizer_architecture_canonical.encode("ascii"))
        if architecture_sha256 != self.amortizer_architecture_sha256:
            raise ValueError("amortizer architecture canonical JSON and SHA-256 disagree")
        _require_lower_hex(
            self.tensor_spec_sha256,
            "tensor_spec_sha256",
            _SHA256_LENGTH,
        )
        if self.atom_tensor_count != self.projection_count * 2:
            raise ValueError("atom_tensor_count does not match projection_count")
        if self.bank_tensor_count != self.atom_tensor_count:
            raise ValueError("bank_tensor_count does not match atom_tensor_count")
        if self.code_dim != self.projection_count * self.atom_count:
            raise ValueError("code_dim does not match projection_count and atom_count")
        if self.total_tensor_count != self.bank_tensor_count + self.amortizer_tensor_count:
            raise ValueError("total_tensor_count does not match component tensor counts")


@dataclass(frozen=True, slots=True)
class _TensorTarget:
    key: str
    parameter: nn.Parameter
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device

    @property
    def safe_dtype(self) -> str:
        try:
            return _DTYPE_TO_SAFE[self.dtype]
        except KeyError as error:
            raise ValueError(
                f"unsupported checkpoint dtype for {self.key}: {self.dtype}"
            ) from error


@dataclass(frozen=True, slots=True)
class _ComponentContract:
    targets: tuple[_TensorTarget, ...]
    topology: tuple[object, ...]
    versions: tuple[int, ...]
    metadata: TrainableCheckpointMetadata
    manifest: dict[str, object]
    logical_keys: tuple[str, ...]
    offset_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PoisonRecord:
    bank_key: tuple[str, int]
    amortizer_key: tuple[str, int]
    bank_ref: ReferenceType[SanaDynamicAdapterBank]
    amortizer_ref: ReferenceType[SupportAmortizer]
    reason: str


_POISON_LOCK = threading.RLock()
_POISONED_COMPONENTS: dict[tuple[str, int], _PoisonRecord] = {}


def _remove_dead_poison_keys(record: _PoisonRecord) -> None:
    if (
        record.bank_ref() is None
        and _POISONED_COMPONENTS.get(record.bank_key) is record
    ):
        _POISONED_COMPONENTS.pop(record.bank_key)
    if (
        record.amortizer_ref() is None
        and _POISONED_COMPONENTS.get(record.amortizer_key) is record
    ):
        _POISONED_COMPONENTS.pop(record.amortizer_key)


def _poison_ref_callback(
    key: tuple[str, int], *, component: str
) -> Callable[[ReferenceType[Any]], None]:
    def remove_dead_reference(dead_ref: ReferenceType[Any]) -> None:
        with _POISON_LOCK:
            record = _POISONED_COMPONENTS.get(key)
            if record is None:
                return
            expected_ref = (
                record.bank_ref if component == "bank" else record.amortizer_ref
            )
            if expected_ref is not dead_ref:
                return
            _remove_dead_poison_keys(record)

    return remove_dead_reference


def _poison_record(
    adapter_bank: SanaDynamicAdapterBank,
    amortizer: SupportAmortizer,
) -> _PoisonRecord | None:
    with _POISON_LOCK:
        candidates = (
            _POISONED_COMPONENTS.get(("bank", id(adapter_bank))),
            _POISONED_COMPONENTS.get(("amortizer", id(amortizer))),
        )
        for record in candidates:
            if record is None:
                continue
            bound_bank = record.bank_ref()
            bound_amortizer = record.amortizer_ref()
            _remove_dead_poison_keys(record)
            if bound_bank is adapter_bank or bound_amortizer is amortizer:
                return record
        return None


def _mark_checkpoint_poisoned(
    adapter_bank: SanaDynamicAdapterBank,
    amortizer: SupportAmortizer,
    reason: str,
) -> None:
    bank_key = ("bank", id(adapter_bank))
    amortizer_key = ("amortizer", id(amortizer))
    record = _PoisonRecord(
        bank_key=bank_key,
        amortizer_key=amortizer_key,
        bank_ref=ref(
            adapter_bank,
            _poison_ref_callback(bank_key, component="bank"),
        ),
        amortizer_ref=ref(
            amortizer,
            _poison_ref_callback(amortizer_key, component="amortizer"),
        ),
        reason=reason,
    )
    with _POISON_LOCK:
        for existing in (
            _POISONED_COMPONENTS.get(bank_key),
            _POISONED_COMPONENTS.get(amortizer_key),
        ):
            if existing is not None:
                _remove_dead_poison_keys(existing)
        _POISONED_COMPONENTS[bank_key] = record
        _POISONED_COMPONENTS[amortizer_key] = record


def trainable_checkpoint_poison_reason(
    *,
    adapter_bank: SanaDynamicAdapterBank,
    amortizer: SupportAmortizer,
) -> str | None:
    """Return process-local persistent poison state for either component identity."""

    if type(adapter_bank) is not SanaDynamicAdapterBank:
        raise TypeError("adapter_bank must be an exact SanaDynamicAdapterBank")
    if type(amortizer) is not SupportAmortizer:
        raise TypeError("amortizer must be an exact SupportAmortizer")
    record = _poison_record(adapter_bank, amortizer)
    return None if record is None else record.reason


def _require_not_poisoned(
    adapter_bank: SanaDynamicAdapterBank,
    amortizer: SupportAmortizer,
) -> None:
    reason = trainable_checkpoint_poison_reason(
        adapter_bank=adapter_bank,
        amortizer=amortizer,
    )
    if reason is not None:
        raise RuntimeError(f"checkpoint components are persistently poisoned: {reason}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _storage_identity(tensor: Tensor) -> tuple[torch.device, int, int] | None:
    if tensor.device.type == "meta" or tensor.numel() == 0:
        return None
    storage = tensor.untyped_storage()
    return tensor.device, storage.data_ptr(), storage.nbytes()


def _require_unaliased(tensors: tuple[tuple[str, Tensor], ...], context: str) -> None:
    object_ids = [id(tensor) for _name, tensor in tensors]
    if len(object_ids) != len(set(object_ids)):
        raise ValueError(f"{context} tensor object aliases are forbidden")
    identities = [
        identity for _name, tensor in tensors if (identity := _storage_identity(tensor)) is not None
    ]
    if len(identities) != len(set(identities)):
        raise ValueError(f"{context} tensor storage aliases are forbidden")


def _require_parameter_health(
    name: str,
    parameter: nn.Parameter,
    *,
    expected_dtype: torch.dtype | None,
) -> None:
    if type(parameter) is not nn.Parameter:
        raise TypeError(f"{name} must be an exact nn.Parameter")
    if not parameter.requires_grad:
        raise ValueError(f"{name} must require gradients")
    if parameter.is_inference():
        raise ValueError(f"{name} must not be an inference tensor")
    if parameter.layout is not torch.strided:
        raise ValueError(f"{name} must have strided layout")
    if not parameter.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if expected_dtype is not None and parameter.dtype is not expected_dtype:
        raise ValueError(f"{name} must have dtype {expected_dtype}")
    if parameter.dtype not in _DTYPE_TO_SAFE:
        raise ValueError(f"{name} has unsupported dtype {parameter.dtype}")
    if parameter.device.type not in {"cpu", "cuda"}:
        raise ValueError(f"{name} must be on cpu or cuda")
    if not bool(torch.isfinite(parameter).all()):
        raise ValueError(f"{name} must be finite")


def _require_frozen_parameter_health(name: str, parameter: nn.Parameter) -> None:
    if type(parameter) is not nn.Parameter:
        raise TypeError(f"{name} must be an exact nn.Parameter")
    if parameter.requires_grad:
        raise ValueError(f"{name} must be frozen")
    if parameter.is_inference():
        raise ValueError(f"{name} must not be an inference tensor")
    if parameter.layout is not torch.strided or not parameter.is_contiguous():
        raise ValueError(f"{name} must be contiguous and strided")
    if parameter.dtype not in _DTYPE_TO_SAFE:
        raise ValueError(f"{name} has unsupported dtype {parameter.dtype}")
    if parameter.device.type not in {"cpu", "cuda"}:
        raise ValueError(f"{name} must be on cpu or cuda")
    if not bool(torch.isfinite(parameter).all()):
        raise ValueError(f"{name} must be finite")


def _bound_transformer_inventory(
    bank: SanaDynamicAdapterBank,
    targets: tuple[_TensorTarget, ...],
) -> tuple[tuple[object, ...], tuple[tuple[str, Tensor], ...]]:
    transformer_ref = bank._transformer_ref
    if transformer_ref is None:
        raise RuntimeError("adapter bank must be canonically installed and bound")
    transformer = transformer_ref()
    if transformer is None or not isinstance(transformer, nn.Module):
        raise RuntimeError("canonical transformer was released")

    modules = tuple(transformer.named_modules(remove_duplicate=False))
    if not modules or modules[0] != ("", transformer):
        raise RuntimeError("transformer module inventory is invalid")
    module_ids = [id(module) for _name, module in modules]
    if len(module_ids) != len(set(module_ids)):
        raise ValueError("transformer module object aliases are forbidden")
    if any(module.training for _name, module in modules):
        raise ValueError("transformer modules must remain in eval mode")

    parameters = tuple(transformer.named_parameters(remove_duplicate=False))
    buffers = tuple(transformer.named_buffers(remove_duplicate=False))
    tensors: tuple[tuple[str, Tensor], ...] = (
        *((f"transformer.{name}", parameter) for name, parameter in parameters),
        *((f"transformer.{name}", buffer) for name, buffer in buffers),
    )
    _require_unaliased(tensors, "transformer")

    expected_trainable = {
        (target.key.removeprefix("adapter_bank."), id(target.parameter)) for target in targets
    }
    actual_trainable = {
        (name, id(parameter)) for name, parameter in parameters if parameter.requires_grad
    }
    if actual_trainable != expected_trainable:
        raise ValueError("transformer trainable parameter inventory does not match Bank atoms")

    target_ids = {id(target.parameter) for target in targets}
    frozen: list[tuple[str, Tensor]] = []
    parameter_topology: list[object] = []
    for name, parameter in parameters:
        qualified = f"transformer.{name}"
        if type(parameter) is not nn.Parameter:
            raise TypeError(f"{qualified} must be an exact nn.Parameter")
        if parameter.is_inference():
            raise ValueError(f"{qualified} must not be an inference tensor")
        if parameter.layout is not torch.strided or not parameter.is_contiguous():
            raise ValueError(f"{qualified} must be contiguous and strided")
        if parameter.device.type not in {"cpu", "cuda"}:
            raise ValueError(f"{qualified} must be on cpu or cuda")
        if parameter.dtype not in _DTYPE_TO_SAFE:
            raise ValueError(f"{qualified} has unsupported dtype {parameter.dtype}")
        if not bool(torch.isfinite(parameter).all()):
            raise ValueError(f"{qualified} must be finite")
        if id(parameter) not in target_ids:
            if parameter.requires_grad:
                raise ValueError(f"{qualified} must be frozen")
            frozen.append((qualified, parameter))
        parameter_topology.append(
            (
                name,
                id(parameter),
                tuple(parameter.shape),
                parameter.dtype,
                parameter.device,
                parameter.layout,
                tuple(parameter.stride()),
                parameter.requires_grad,
                _storage_identity(parameter),
            )
        )

    buffer_topology: list[object] = []
    for name, buffer in buffers:
        qualified = f"transformer.{name}"
        if type(buffer) is not Tensor:
            raise TypeError(f"{qualified} must be an exact Tensor")
        if buffer.is_inference():
            raise ValueError(f"{qualified} must not be an inference tensor")
        if buffer.layout is not torch.strided or not buffer.is_contiguous():
            raise ValueError(f"{qualified} must be contiguous and strided")
        if buffer.device.type not in {"cpu", "cuda"}:
            raise ValueError(f"{qualified} must be on cpu or cuda")
        if buffer.requires_grad:
            raise ValueError(f"transformer buffer {qualified} must not require gradients")
        if (buffer.is_floating_point() or buffer.is_complex()) and not bool(
            torch.isfinite(buffer).all()
        ):
            raise ValueError(f"{qualified} must be finite")
        frozen.append((qualified, buffer))
        buffer_topology.append(
            (
                name,
                id(buffer),
                tuple(buffer.shape),
                buffer.dtype,
                buffer.device,
                buffer.layout,
                tuple(buffer.stride()),
                buffer.requires_grad,
                _storage_identity(buffer),
            )
        )

    topology: tuple[object, ...] = (
        id(transformer),
        type(transformer),
        tuple((name, id(module), type(module), module.training) for name, module in modules),
        tuple(parameter_topology),
        tuple(buffer_topology),
    )
    return topology, tuple(frozen)


def _bank_targets(
    bank: SanaDynamicAdapterBank,
) -> tuple[
    tuple[_TensorTarget, ...],
    tuple[object, ...],
    tuple[tuple[str, Tensor], ...],
    int,
]:
    if type(bank) is not SanaDynamicAdapterBank:
        raise TypeError("adapter_bank must be an exact SanaDynamicAdapterBank")
    if bank._transformer_ref is None or type(bank._bindings) is not tuple:
        raise RuntimeError("adapter bank must be canonically installed and bound")
    layout = bank.layout
    if type(layout) is not SanaAdapterLayout:
        raise TypeError("adapter bank layout must be an exact SanaAdapterLayout")
    _require_positive_exact_int(layout.num_blocks, "layout.num_blocks")
    _require_positive_exact_int(layout.atom_count, "layout.atom_count")
    wrappers = bank.wrappers
    if len(wrappers) != layout.projection_count:
        raise ValueError("adapter wrapper count does not match layout")
    if not wrappers:
        raise ValueError("adapter bank must contain wrappers")

    ranks: set[int] = set()
    trainable: list[tuple[str, Tensor]] = []
    frozen: list[tuple[str, Tensor]] = []
    targets: list[_TensorTarget] = []
    wrapper_topology: list[object] = []
    for path, wrapper in zip(layout.projection_names, wrappers, strict=True):
        if type(wrapper) is not DynamicAtomLinear:
            raise TypeError(f"{path} must be an exact DynamicAtomLinear")
        rank = _require_positive_exact_int(wrapper.rank, f"{path}.rank")
        ranks.add(rank)
        if type(wrapper.atom_count) is not int or wrapper.atom_count != layout.atom_count:
            raise ValueError(f"{path}.atom_count does not match layout")
        if (
            wrapper._coefficients is not None
            or wrapper._activation_token is not None
            or wrapper._coefficient_version is not None
        ):
            raise RuntimeError("adapter bank is active")
        if type(wrapper.base) is not nn.Linear:
            raise TypeError(f"{path}.base must be an exact nn.Linear")
        base = wrapper.base
        if wrapper.training or base.training:
            raise ValueError("adapter bank and frozen bases must remain in eval mode")
        direct_parameters = tuple(wrapper.named_parameters(recurse=False, remove_duplicate=False))
        if direct_parameters != (
            ("atom_down", wrapper.atom_down),
            ("atom_up", wrapper.atom_up),
        ):
            raise RuntimeError(f"{path} direct parameter topology changed")
        if tuple(wrapper.named_buffers(recurse=False, remove_duplicate=False)):
            raise RuntimeError(f"{path} direct buffer topology changed")
        if tuple(wrapper.named_children()) != (("base", base),):
            raise RuntimeError(f"{path} child module topology changed")
        if tuple(base.named_modules(remove_duplicate=False)) != (("", base),):
            raise RuntimeError(f"{path}.base child module topology changed")
        if tuple(base.named_buffers(remove_duplicate=False)):
            raise RuntimeError(f"{path}.base buffer topology changed")
        base_parameters = tuple(base.named_parameters(remove_duplicate=False))
        expected_base_names = ("weight", "bias") if base.bias is not None else ("weight",)
        if tuple(name for name, _parameter in base_parameters) != expected_base_names:
            raise RuntimeError(f"{path}.base parameter topology changed")
        if not base_parameters or any(
            parameter.requires_grad for _name, parameter in base_parameters
        ):
            raise ValueError(f"{path}.base parameters must be frozen")
        for base_name, parameter in base_parameters:
            frozen_name = f"{path}.base.{base_name}"
            _require_frozen_parameter_health(frozen_name, parameter)
            frozen.append((frozen_name, parameter))
        if tuple(base.weight.shape) != (base.out_features, base.in_features):
            raise ValueError(f"{path}.base.weight shape does not match Linear topology")
        if base.bias is not None and tuple(base.bias.shape) != (base.out_features,):
            raise ValueError(f"{path}.base.bias shape does not match Linear topology")

        expected_shapes = {
            "atom_down": (layout.atom_count, rank, base.in_features),
            "atom_up": (layout.atom_count, base.out_features, rank),
        }
        for atom_name, parameter in (
            ("atom_down", wrapper.atom_down),
            ("atom_up", wrapper.atom_up),
        ):
            canonical_name = f"{path}.{atom_name}"
            key = f"adapter_bank.{canonical_name}"
            _require_parameter_health(key, parameter, expected_dtype=base.weight.dtype)
            if tuple(parameter.shape) != expected_shapes[atom_name]:
                raise ValueError(f"{key} shape does not match wrapper topology")
            if parameter.device != base.weight.device:
                raise ValueError(f"{key} device does not match frozen base")
            trainable.append((key, parameter))
            targets.append(
                _TensorTarget(
                    key=key,
                    parameter=parameter,
                    shape=tuple(parameter.shape),
                    dtype=parameter.dtype,
                    device=parameter.device,
                )
            )
        wrapper_topology.append(
            (
                path,
                id(wrapper),
                id(base),
                rank,
                wrapper.atom_count,
                base.in_features,
                base.out_features,
                base.bias is not None,
                wrapper.training,
                base.training,
            )
        )

    if len(ranks) != 1:
        raise ValueError("all adapter wrappers must use one rank")
    rank = next(iter(ranks))
    named = tuple(
        (f"adapter_bank.{name}", parameter) for name, parameter in bank.named_parameters()
    )
    if tuple(name for name, _parameter in named) != tuple(target.key for target in targets):
        raise RuntimeError("adapter bank canonical parameter order changed")
    if any(
        parameter is not target.parameter
        for (_name, parameter), target in zip(named, targets, strict=True)
    ):
        raise RuntimeError("adapter bank canonical parameter identity changed")
    _require_unaliased(tuple(trainable), "adapter bank")
    _require_unaliased(tuple(frozen), "frozen base")
    _require_unaliased(tuple(trainable + frozen), "adapter and frozen base")
    transformer_topology, transformer_frozen = _bound_transformer_inventory(
        bank,
        tuple(targets),
    )
    topology: tuple[object, ...] = (
        id(layout),
        layout.num_blocks,
        layout.atom_count,
        tuple(wrapper_topology),
        transformer_topology,
    )
    return tuple(targets), topology, transformer_frozen, rank


def _amortizer_targets(
    amortizer: SupportAmortizer,
) -> tuple[tuple[_TensorTarget, ...], tuple[object, ...]]:
    if type(amortizer) is not SupportAmortizer:
        raise TypeError("amortizer must be an exact SupportAmortizer")
    architecture_canonical = amortizer.architecture_canonical
    architecture_signature = amortizer.architecture_signature
    if tuple(amortizer.named_buffers(remove_duplicate=False)):
        raise ValueError("checkpoint schema v1 does not permit amortizer buffers")
    named = tuple(amortizer.named_parameters(remove_duplicate=False))
    if not named:
        raise ValueError("amortizer must contain trainable parameters")
    targets: list[_TensorTarget] = []
    for name, parameter in named:
        key = f"amortizer.{name}"
        _require_parameter_health(key, parameter, expected_dtype=torch.float32)
        targets.append(
            _TensorTarget(
                key=key,
                parameter=parameter,
                shape=tuple(parameter.shape),
                dtype=parameter.dtype,
                device=parameter.device,
            )
        )
    _require_unaliased(tuple((target.key, target.parameter) for target in targets), "amortizer")
    module_topology = tuple(
        (name, id(module), type(module), module.training)
        for name, module in amortizer.named_modules(remove_duplicate=False)
    )
    topology: tuple[object, ...] = (
        architecture_canonical,
        architecture_signature,
        module_topology,
    )
    return tuple(targets), topology


def _tensor_spec(targets: tuple[_TensorTarget, ...]) -> list[dict[str, object]]:
    return [
        {
            "dtype": target.safe_dtype,
            "key": target.key,
            "shape": list(target.shape),
        }
        for target in sorted(targets, key=lambda item: item.key)
    ]


def _manifest(
    provenance: CheckpointProvenance,
    bank: SanaDynamicAdapterBank,
    amortizer: SupportAmortizer,
    bank_targets: tuple[_TensorTarget, ...],
    amortizer_targets: tuple[_TensorTarget, ...],
    rank: int,
) -> tuple[TrainableCheckpointMetadata, dict[str, object]]:
    specification = _tensor_spec(bank_targets + amortizer_targets)
    spec_sha256 = _sha256_bytes(_canonical_json(specification).encode("ascii"))
    layout = bank.layout
    metadata = TrainableCheckpointMetadata(
        provenance=provenance,
        layout_version=SANA_LAYOUT_VERSION,
        num_blocks=layout.num_blocks,
        rank=rank,
        atom_count=layout.atom_count,
        projection_count=layout.projection_count,
        code_dim=layout.code_dim,
        atom_tensor_count=layout.atom_tensor_count,
        amortizer_architecture_canonical=amortizer.architecture_canonical,
        amortizer_architecture_sha256=amortizer.architecture_signature,
        bank_tensor_count=len(bank_targets),
        amortizer_tensor_count=len(amortizer_targets),
        total_tensor_count=len(bank_targets) + len(amortizer_targets),
        tensor_spec_sha256=spec_sha256,
    )
    manifest: dict[str, object] = {
        "amortizer": {
            "architecture_canonical": metadata.amortizer_architecture_canonical,
            "architecture_sha256": metadata.amortizer_architecture_sha256,
        },
        "format": _FORMAT,
        "framework": _FRAMEWORK,
        "layout": {
            "atom_count": metadata.atom_count,
            "atom_tensor_count": metadata.atom_tensor_count,
            "code_dim": metadata.code_dim,
            "num_blocks": metadata.num_blocks,
            "projection_count": metadata.projection_count,
            "rank": metadata.rank,
            "version": metadata.layout_version,
        },
        "model": {
            "id": provenance.model_id,
            "revision": provenance.model_revision,
        },
        "schema": _SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "serializer_version": _SERIALIZER_VERSION,
        "support_model": {
            "id": provenance.support_model_id,
            "revision": provenance.support_model_revision,
        },
        "tensors": {
            "amortizer_tensor_count": metadata.amortizer_tensor_count,
            "bank_tensor_count": metadata.bank_tensor_count,
            "spec_sha256": metadata.tensor_spec_sha256,
            "total_tensor_count": metadata.total_tensor_count,
        },
    }
    return metadata, manifest


def _component_contract(
    adapter_bank: SanaDynamicAdapterBank,
    amortizer: SupportAmortizer,
    provenance: CheckpointProvenance,
) -> _ComponentContract:
    if type(provenance) is not CheckpointProvenance:
        raise TypeError("provenance must be an exact CheckpointProvenance")
    provenance.validate()
    bank_targets, bank_topology, frozen, rank = _bank_targets(adapter_bank)
    amortizer_targets, amortizer_topology = _amortizer_targets(amortizer)
    if amortizer.projection_count != adapter_bank.layout.projection_count:
        raise ValueError("amortizer projection_count does not match adapter bank")
    if amortizer.atom_count != adapter_bank.layout.atom_count:
        raise ValueError("amortizer atom_count does not match adapter bank")
    targets = bank_targets + amortizer_targets
    _require_unaliased(tuple((target.key, target.parameter) for target in targets), "checkpoint")
    _require_unaliased(
        tuple((target.key, target.parameter) for target in targets) + tuple(frozen),
        "checkpoint and transformer",
    )
    metadata, manifest = _manifest(
        provenance,
        adapter_bank,
        amortizer,
        bank_targets,
        amortizer_targets,
        rank,
    )
    logical_keys = tuple(sorted(target.key for target in targets))
    offset_keys = tuple(
        target.key
        for target in sorted(
            targets,
            key=lambda item: (-_SAFE_DTYPE_PRIORITY[item.safe_dtype], item.key),
        )
    )
    target_topology = tuple(
        (
            target.key,
            id(target.parameter),
            target.shape,
            target.dtype,
            target.device,
            target.parameter.requires_grad,
            _storage_identity(target.parameter),
        )
        for target in targets
    )
    frozen_topology = tuple(
        (
            name,
            id(parameter),
            tuple(parameter.shape),
            parameter.dtype,
            parameter.device,
            parameter.requires_grad,
            _storage_identity(parameter),
        )
        for name, parameter in frozen
    )
    topology: tuple[object, ...] = (
        bank_topology,
        amortizer_topology,
        target_topology,
        frozen_topology,
    )
    versions = tuple(target.parameter._version for target in targets) + tuple(
        parameter._version for _name, parameter in frozen
    )
    return _ComponentContract(
        targets=targets,
        topology=topology,
        versions=versions,
        metadata=metadata,
        manifest=manifest,
        logical_keys=logical_keys,
        offset_keys=offset_keys,
    )


def _assert_same_precommit_contract(
    before: _ComponentContract,
    after: _ComponentContract,
) -> None:
    if after.topology != before.topology or after.manifest != before.manifest:
        raise RuntimeError("checkpoint component topology changed during operation")
    if after.versions != before.versions:
        raise RuntimeError("checkpoint component values changed during operation")


def _assert_same_topology(before: _ComponentContract, after: _ComponentContract) -> None:
    if after.topology != before.topology or after.manifest != before.manifest:
        raise RuntimeError("checkpoint component topology changed during load")


def _absolute_without_resolution(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def _assert_no_symlink_ancestors(path: Path) -> None:
    absolute = _absolute_without_resolution(path)
    parts = absolute.parts
    current = Path(parts[0])
    for part in parts[1:]:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise OSError(f"checkpoint path is unsafe because {current} is a symlink")


def _validate_private_directory(path: Path) -> os.stat_result:
    _assert_no_symlink_ancestors(path)
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"checkpoint parent does not exist: {path}") from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise OSError("checkpoint parent must be a real directory")
    if metadata.st_uid != os.getuid():
        raise PermissionError("checkpoint parent must be owned by the current uid")
    if stat.S_IMODE(metadata.st_mode) != 0o700:
        raise PermissionError("checkpoint parent must have exact mode 0700")
    return metadata


def _validate_private_file_metadata(metadata: os.stat_result) -> None:
    if not stat.S_ISREG(metadata.st_mode):
        raise OSError("checkpoint must be a regular non-symlink file")
    if metadata.st_uid != os.getuid():
        raise PermissionError("checkpoint must be owned by the current uid")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise PermissionError("checkpoint must have exact mode 0600")
    if metadata.st_nlink != 1:
        raise OSError("checkpoint must have exactly one hard link")


def _validate_path(path: Path) -> Path:
    if type(path) is not type(Path()):
        raise TypeError("path must be an exact Path")
    if not path.name or path.name in {".", ".."}:
        raise ValueError("checkpoint path must name a file")
    absolute = _absolute_without_resolution(path)
    _validate_private_directory(absolute.parent)
    return absolute


@dataclass(frozen=True, slots=True)
class _PrivateDirectoryHandle:
    path: Path
    filename: str
    descriptor: int
    metadata: os.stat_result


def _open_private_directory(path: Path) -> _PrivateDirectoryHandle:
    absolute = _validate_path(path)
    before = _validate_private_directory(absolute.parent)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(absolute.parent, flags)
    try:
        after = os.fstat(descriptor)
        if (
            (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
            or not stat.S_ISDIR(after.st_mode)
            or after.st_uid != os.getuid()
            or stat.S_IMODE(after.st_mode) != 0o700
            or after.st_nlink != before.st_nlink
        ):
            raise OSError("checkpoint parent changed during secure open")
        return _PrivateDirectoryHandle(
            path=absolute.parent,
            filename=absolute.name,
            descriptor=descriptor,
            metadata=after,
        )
    except BaseException:
        os.close(descriptor)
        raise


def _verify_private_directory(directory: _PrivateDirectoryHandle) -> None:
    descriptor_metadata = os.fstat(directory.descriptor)
    expected = directory.metadata
    if (
        (descriptor_metadata.st_dev, descriptor_metadata.st_ino)
        != (expected.st_dev, expected.st_ino)
        or descriptor_metadata.st_mode != expected.st_mode
        or descriptor_metadata.st_uid != expected.st_uid
        or descriptor_metadata.st_nlink != expected.st_nlink
    ):
        raise OSError("checkpoint parent changed through its directory descriptor")
    try:
        _assert_no_symlink_ancestors(directory.path)
        path_metadata = directory.path.lstat()
    except (FileNotFoundError, OSError) as error:
        raise OSError("checkpoint parent changed while the operation was active") from error
    if (
        (path_metadata.st_dev, path_metadata.st_ino) != (expected.st_dev, expected.st_ino)
        or path_metadata.st_mode != expected.st_mode
        or path_metadata.st_uid != expected.st_uid
        or path_metadata.st_nlink != expected.st_nlink
    ):
        raise OSError("checkpoint parent changed while the operation was active")


def _write_all(descriptor: int, content: bytes) -> None:
    remaining = memoryview(content)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("checkpoint write made no progress")
        remaining = remaining[written:]


def _fsync_descriptor(descriptor: int) -> None:
    os.fsync(descriptor)


def _fsync_directory(descriptor: int) -> None:
    os.fsync(descriptor)


def _publish_no_replace(directory_fd: int, temporary: str, final: str) -> None:
    os.link(
        temporary,
        final,
        src_dir_fd=directory_fd,
        dst_dir_fd=directory_fd,
        follow_symlinks=False,
    )


def _atomic_create(
    directory: _PrivateDirectoryHandle,
    content: bytes,
) -> CheckpointFileIdentity:
    try:
        os.stat(
            directory.filename,
            dir_fd=directory.descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(f"checkpoint target already exists: {directory.filename}")
    temporary = f".{directory.filename}.{uuid.uuid4().hex}.tmp"
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = -1
    published = False
    temporary_metadata: os.stat_result | None = None
    try:
        descriptor = os.open(
            temporary,
            flags,
            0o600,
            dir_fd=directory.descriptor,
        )
        os.fchmod(descriptor, 0o600)
        _write_all(descriptor, content)
        _fsync_descriptor(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_size != len(content)
        ):
            raise OSError("temporary checkpoint file failed validation")
        temporary_metadata = metadata
        _publish_no_replace(
            directory.descriptor,
            temporary,
            directory.filename,
        )
        published = True
        os.unlink(temporary, dir_fd=directory.descriptor)
        _fsync_directory(directory.descriptor)
        final = os.stat(
            directory.filename,
            dir_fd=directory.descriptor,
            follow_symlinks=False,
        )
        _validate_private_file_metadata(final)
        if final.st_size != len(content):
            raise OSError("published checkpoint size changed")
        if temporary_metadata is None or (
            final.st_dev,
            final.st_ino,
        ) != (
            temporary_metadata.st_dev,
            temporary_metadata.st_ino,
        ):
            raise OSError("published checkpoint inode changed")
        published_content = _read_descriptor(descriptor)
        if published_content != content:
            raise OSError("published checkpoint bytes changed")
        descriptor_after_read = os.fstat(descriptor)
        final_after_read = os.stat(
            directory.filename,
            dir_fd=directory.descriptor,
            follow_symlinks=False,
        )
        _validate_private_file_metadata(final_after_read)
        if (
            (descriptor_after_read.st_dev, descriptor_after_read.st_ino)
            != (temporary_metadata.st_dev, temporary_metadata.st_ino)
            or (final_after_read.st_dev, final_after_read.st_ino)
            != (temporary_metadata.st_dev, temporary_metadata.st_ino)
            or descriptor_after_read.st_size != len(published_content)
            or final_after_read.st_size != len(published_content)
        ):
            raise OSError("published checkpoint identity changed")
        _verify_private_directory(directory)
        return CheckpointFileIdentity(
            sha256=_sha256_bytes(published_content),
            byte_count=len(published_content),
        )
    except BaseException:
        try:
            os.unlink(temporary, dir_fd=directory.descriptor)
        except OSError:
            pass
        if published:
            try:
                os.unlink(directory.filename, dir_fd=directory.descriptor)
                _fsync_directory(directory.descriptor)
            except OSError:
                pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _snapshot_for_save(targets: tuple[_TensorTarget, ...]) -> dict[str, Tensor]:
    snapshots: dict[str, Tensor] = {}
    with torch.inference_mode(False), torch.no_grad():
        for target in targets:
            snapshots[target.key] = target.parameter.detach().to(device="cpu").contiguous().clone()
    staged = tuple(snapshots.items())
    _require_unaliased(staged, "serialized checkpoint")
    for name, tensor in staged:
        if (
            tensor.device.type != "cpu"
            or tensor.is_inference()
            or tensor.requires_grad
            or tensor.layout is not torch.strided
            or not tensor.is_contiguous()
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError(f"serialized tensor {name} failed health validation")
    return snapshots


def _snapshot_for_transaction(targets: tuple[_TensorTarget, ...]) -> tuple[Tensor, ...]:
    with torch.inference_mode(False), torch.no_grad():
        snapshots = tuple(target.parameter.detach().contiguous().clone() for target in targets)
    _require_unaliased(
        tuple((target.key, snapshot) for target, snapshot in zip(targets, snapshots, strict=True)),
        "transaction originals",
    )
    return snapshots


def _assert_target_values_match(
    targets: tuple[_TensorTarget, ...],
    expected: tuple[Tensor, ...],
    context: str,
) -> None:
    if len(targets) != len(expected):
        raise RuntimeError(f"{context} snapshot count changed")
    with torch.no_grad():
        for target, reference in zip(targets, expected, strict=True):
            current = target.parameter.detach()
            if current.device != reference.device:
                current = current.to(device=reference.device)
            if (
                current.dtype is not reference.dtype
                or tuple(current.shape) != tuple(reference.shape)
                or not torch.equal(current, reference)
            ):
                raise RuntimeError(f"{context} value changed at {target.key}")


def save_trainable_checkpoint(
    path: Path,
    *,
    adapter_bank: SanaDynamicAdapterBank,
    amortizer: SupportAmortizer,
    provenance: CheckpointProvenance,
) -> CheckpointFileIdentity:
    """Create one deterministic, trainable-only checkpoint without replacement."""

    _require_serializer_runtime()
    _require_not_poisoned(adapter_bank, amortizer)
    directory = _open_private_directory(path)
    try:
        contract = _component_contract(adapter_bank, amortizer, provenance)
        tensors = _snapshot_for_save(contract.targets)
        snapshot_values = tuple(tensors[target.key] for target in contract.targets)
        after_snapshot = _component_contract(adapter_bank, amortizer, provenance)
        _assert_same_precommit_contract(contract, after_snapshot)
        _assert_target_values_match(contract.targets, snapshot_values, "checkpoint source")
        content = save(tensors, metadata={_METADATA_KEY: _canonical_json(contract.manifest)})
        after_serialization = _component_contract(adapter_bank, amortizer, provenance)
        _assert_same_precommit_contract(contract, after_serialization)
        _assert_target_values_match(contract.targets, snapshot_values, "checkpoint source")
        return _atomic_create(directory, content)
    finally:
        os.close(directory.descriptor)


def _read_descriptor(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, _CHUNK_BYTES)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


@contextmanager
def _secure_checkpoint_descriptor(
    directory: _PrivateDirectoryHandle,
) -> Iterator[tuple[int, bytes]]:
    try:
        before = os.stat(
            directory.filename,
            dir_fd=directory.descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError as error:
        raise FileNotFoundError(f"checkpoint does not exist: {directory.filename}") from error
    _validate_private_file_metadata(before)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(
        directory.filename,
        flags,
        dir_fd=directory.descriptor,
    )
    try:
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise OSError("checkpoint changed during secure open")
        if (
            not stat.S_ISREG(after.st_mode)
            or after.st_uid != os.getuid()
            or stat.S_IMODE(after.st_mode) != 0o600
            or after.st_nlink != 1
            or after.st_size != before.st_size
        ):
            raise OSError("checkpoint descriptor failed security validation")
        content = _read_descriptor(descriptor)
        final = os.fstat(descriptor)
        if (
            (final.st_dev, final.st_ino) != (after.st_dev, after.st_ino)
            or final.st_size != after.st_size
            or final.st_mtime_ns != after.st_mtime_ns
            or final.st_ctime_ns != after.st_ctime_ns
        ):
            raise OSError("checkpoint changed while being read")
        try:
            yield descriptor, content
        finally:
            ending = os.fstat(descriptor)
            if (
                (ending.st_dev, ending.st_ino) != (final.st_dev, final.st_ino)
                or ending.st_size != final.st_size
                or ending.st_mtime_ns != final.st_mtime_ns
                or ending.st_ctime_ns != final.st_ctime_ns
                or ending.st_mode != final.st_mode
                or ending.st_uid != final.st_uid
                or ending.st_nlink != final.st_nlink
            ):
                raise OSError("checkpoint changed during validation or materialization")
    finally:
        os.close(descriptor)


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> object:
    raise ValueError(f"nonfinite JSON constant is forbidden: {value}")


def _strict_json(content: str, context: str) -> dict[str, object]:
    try:
        value = json.loads(
            content,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{context} is not valid JSON") from error
    if type(value) is not dict:
        raise TypeError(f"{context} must be an exact JSON object")
    return cast(dict[str, object], value)


def _require_exact_value(actual: object, expected: object, path: str) -> None:
    if type(actual) is not type(expected):
        raise TypeError(f"{path} has the wrong exact type")
    if type(expected) is dict:
        actual_dict = cast(dict[str, object], actual)
        expected_dict = cast(dict[str, object], expected)
        if set(actual_dict) != set(expected_dict):
            raise ValueError(f"{path} has missing or unexpected keys")
        for key, value in expected_dict.items():
            _require_exact_value(actual_dict[key], value, f"{path}.{key}")
        return
    if type(expected) is list:
        actual_list = cast(list[object], actual)
        expected_list = cast(list[object], expected)
        if len(actual_list) != len(expected_list):
            raise ValueError(f"{path} has the wrong length")
        for index, value in enumerate(expected_list):
            _require_exact_value(actual_list[index], value, f"{path}[{index}]")
        return
    if actual != expected:
        raise ValueError(f"{path} does not match the expected checkpoint contract")


def _raw_header(content: bytes) -> dict[str, object]:
    if len(content) < 8:
        raise ValueError("checkpoint header is truncated")
    length = int.from_bytes(content[:8], "little", signed=False)
    if length <= 0 or length % 8 != 0 or length > _MAX_HEADER_BYTES or 8 + length > len(content):
        raise ValueError("checkpoint header length is invalid")
    encoded = content[8 : 8 + length]
    if not encoded or encoded[:1] != b"{":
        raise ValueError("checkpoint header must start with a JSON object")
    stripped = encoded.rstrip(b" ")
    if encoded[len(stripped) :] != b" " * (len(encoded) - len(stripped)):
        raise ValueError("checkpoint header has invalid padding")
    try:
        text = stripped.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("checkpoint header is not UTF-8") from error
    header = _strict_json(text, "checkpoint header")
    canonical = json.dumps(
        header,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    if canonical != text:
        raise ValueError("checkpoint header is not canonical JSON")
    return header


def _validate_raw_header(
    content: bytes,
    contract: _ComponentContract,
) -> None:
    header = _raw_header(content)
    expected_order = ("__metadata__", *contract.offset_keys)
    if tuple(header) != expected_order:
        raise ValueError("checkpoint header tensor order does not match")
    metadata = header.get("__metadata__")
    if type(metadata) is not dict or set(cast(dict[str, object], metadata)) != {_METADATA_KEY}:
        raise ValueError("checkpoint metadata keys do not exactly match")
    expected_targets = {target.key: target for target in contract.targets}
    expected_offset = 0
    for key in contract.offset_keys:
        entry = header[key]
        if type(entry) is not dict:
            raise TypeError(f"checkpoint tensor header {key} must be an object")
        entry_dict = cast(dict[str, object], entry)
        if set(entry_dict) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"checkpoint tensor header {key} has invalid keys")
        target = expected_targets[key]
        if type(entry_dict["dtype"]) is not str or entry_dict["dtype"] != target.safe_dtype:
            raise ValueError(f"checkpoint tensor {key} dtype does not match")
        if type(entry_dict["shape"]) is not list or entry_dict["shape"] != list(target.shape):
            raise ValueError(f"checkpoint tensor {key} shape does not match")
        offsets = entry_dict["data_offsets"]
        if (
            type(offsets) is not list
            or len(offsets) != 2
            or any(type(offset) is not int for offset in offsets)
        ):
            raise TypeError(f"checkpoint tensor {key} data_offsets are invalid")
        tensor_bytes = target.parameter.numel() * _SAFE_DTYPE_BYTES[target.safe_dtype]
        expected_offsets = [expected_offset, expected_offset + tensor_bytes]
        if offsets != expected_offsets:
            raise ValueError(f"checkpoint tensor {key} data_offsets do not match")
        expected_offset += tensor_bytes
    header_length = int.from_bytes(content[:8], "little", signed=False)
    if len(content) - 8 - header_length != expected_offset:
        raise ValueError("checkpoint tensor data length does not exactly match")


def _validate_manifest(
    metadata: object,
    contract: _ComponentContract,
) -> None:
    if type(metadata) is not dict:
        raise TypeError("checkpoint metadata must be an exact dict")
    metadata_dict = cast(dict[str, str], metadata)
    if set(metadata_dict) != {_METADATA_KEY}:
        raise ValueError("checkpoint metadata keys do not exactly match")
    encoded = metadata_dict[_METADATA_KEY]
    if type(encoded) is not str:
        raise TypeError("checkpoint ratemem metadata must be an exact str")
    decoded = _strict_json(encoded, "checkpoint ratemem metadata")
    if _canonical_json(decoded) != encoded:
        raise ValueError("checkpoint ratemem metadata is not canonical JSON")
    actual_amortizer = decoded.get("amortizer")
    expected_amortizer = contract.manifest["amortizer"]
    if (
        type(actual_amortizer) is dict
        and set(cast(dict[str, object], actual_amortizer))
        == {"architecture_canonical", "architecture_sha256"}
        and actual_amortizer != expected_amortizer
    ):
        raise ValueError("checkpoint amortizer architecture does not match")
    _require_exact_value(decoded, contract.manifest, "checkpoint metadata")


def _materialize_tensor(handle: Any, key: str) -> Tensor:
    return cast(Tensor, handle.get_tensor(key))


def _validate_loaded_tensor(target: _TensorTarget, tensor: Tensor) -> None:
    if type(tensor) is not Tensor:
        raise TypeError(f"checkpoint tensor {target.key} must be an exact Tensor")
    if tensor.device.type != "cpu":
        raise ValueError(f"checkpoint tensor {target.key} must materialize on CPU")
    if tuple(tensor.shape) != target.shape:
        raise ValueError(f"checkpoint tensor {target.key} shape does not match")
    if tensor.dtype is not target.dtype:
        raise ValueError(f"checkpoint tensor {target.key} dtype does not match")
    if tensor.layout is not torch.strided or not tensor.is_contiguous():
        raise ValueError(f"checkpoint tensor {target.key} must be contiguous and strided")
    if tensor.is_inference():
        raise ValueError(f"checkpoint tensor {target.key} must not be an inference tensor")
    if tensor.requires_grad:
        raise ValueError(f"checkpoint tensor {target.key} must not require gradients")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"checkpoint tensor {target.key} must be finite")


def _stage_loaded_tensors(
    handle: Any,
    contract: _ComponentContract,
) -> tuple[Tensor, ...]:
    source: list[tuple[str, Tensor]] = []
    target_by_key = {target.key: target for target in contract.targets}
    for key in contract.logical_keys:
        target = target_by_key[key]
        tensor = _materialize_tensor(handle, key)
        _validate_loaded_tensor(target, tensor)
        source.append((key, tensor))
    _require_unaliased(tuple(source), "materialized checkpoint")

    staged_by_key: dict[str, Tensor] = {}
    with torch.inference_mode(False), torch.no_grad():
        for key, tensor in source:
            target = target_by_key[key]
            staged_by_key[key] = (
                tensor.detach().to(device=target.device, dtype=target.dtype).contiguous().clone()
            )
    staged_items = tuple((key, staged_by_key[key]) for key in contract.logical_keys)
    _require_unaliased(staged_items, "staged checkpoint")
    for key, tensor in staged_items:
        target = target_by_key[key]
        if (
            tensor.device != target.device
            or tensor.dtype is not target.dtype
            or tuple(tensor.shape) != target.shape
            or tensor.is_inference()
            or tensor.requires_grad
            or not tensor.is_contiguous()
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError(f"staged checkpoint tensor {key} failed validation")
    return tuple(staged_by_key[target.key] for target in contract.targets)


def _copy_checkpoint_value(parameter: nn.Parameter, value: Tensor) -> None:
    parameter.copy_(value)


def _rollback_checkpoint_value(parameter: nn.Parameter, value: Tensor) -> None:
    parameter.copy_(value)


def _commit_transaction(
    contract: _ComponentContract,
    staged: tuple[Tensor, ...],
    originals: tuple[Tensor, ...],
    adapter_bank: SanaDynamicAdapterBank,
    amortizer: SupportAmortizer,
    provenance: CheckpointProvenance,
    postcommit_validate: Callable[[], None] | None = None,
) -> None:
    before_commit = _component_contract(adapter_bank, amortizer, provenance)
    _assert_same_precommit_contract(contract, before_commit)
    _assert_target_values_match(contract.targets, originals, "checkpoint destination")
    attempted = 0
    try:
        with torch.no_grad():
            for target, value in zip(contract.targets, staged, strict=True):
                attempted += 1
                _copy_checkpoint_value(target.parameter, value)
        after = _component_contract(adapter_bank, amortizer, provenance)
        _assert_same_topology(contract, after)
        for target, value in zip(contract.targets, staged, strict=True):
            if not torch.equal(target.parameter, value):
                raise RuntimeError(f"checkpoint post-load value mismatch at {target.key}")
        if postcommit_validate is not None:
            postcommit_validate()
    except BaseException:
        try:
            with torch.no_grad():
                for target, original in zip(
                    contract.targets[:attempted], originals[:attempted], strict=True
                ):
                    _rollback_checkpoint_value(target.parameter, original)
            restored = _component_contract(adapter_bank, amortizer, provenance)
            _assert_same_topology(contract, restored)
            for target, original in zip(
                contract.targets[:attempted], originals[:attempted], strict=True
            ):
                if not torch.equal(target.parameter, original):
                    raise RuntimeError(f"checkpoint rollback value mismatch at {target.key}")
        except BaseException as rollback_error:
            _mark_checkpoint_poisoned(
                adapter_bank,
                amortizer,
                f"rollback failed: {rollback_error!r}",
            )
            raise RuntimeError(
                "checkpoint rollback failed; destination state is poisoned"
            ) from rollback_error
        raise


def load_trainable_checkpoint(
    path: Path,
    *,
    adapter_bank: SanaDynamicAdapterBank,
    amortizer: SupportAmortizer,
    expected_provenance: CheckpointProvenance,
    expected_file: CheckpointFileIdentity,
) -> TrainableCheckpointMetadata:
    """Strictly validate, stage, and jointly load Bank and amortizer state."""

    _require_serializer_runtime()
    _require_not_poisoned(adapter_bank, amortizer)
    if type(expected_file) is not CheckpointFileIdentity:
        raise TypeError("expected_file must be an exact CheckpointFileIdentity")
    expected_file.validate()
    directory = _open_private_directory(path)
    try:
        contract = _component_contract(adapter_bank, amortizer, expected_provenance)
        originals = _snapshot_for_transaction(contract.targets)
        after_original_snapshot = _component_contract(
            adapter_bank,
            amortizer,
            expected_provenance,
        )
        _assert_same_precommit_contract(contract, after_original_snapshot)
        _assert_target_values_match(contract.targets, originals, "checkpoint destination")
        with _secure_checkpoint_descriptor(directory) as (descriptor, content):
            if len(content) != expected_file.byte_count:
                raise ValueError("checkpoint byte count does not match expected identity")
            if _sha256_bytes(content) != expected_file.sha256:
                raise ValueError("checkpoint sha256 does not match expected identity")
            _validate_raw_header(content, contract)
            proc_path = Path(f"/proc/self/fd/{descriptor}")
            if not proc_path.exists():
                raise RuntimeError("stable file-descriptor loading requires Linux procfs")
            with safe_open(
                proc_path,
                framework="pt",
                device="cpu",
                backend="pread",
            ) as handle:
                _validate_manifest(handle.metadata(), contract)
                if tuple(handle.keys()) != contract.logical_keys:
                    raise ValueError("checkpoint tensor keys do not exactly match")
                if tuple(handle.offset_keys()) != contract.offset_keys:
                    raise ValueError("checkpoint tensor offset order does not exactly match")
                target_by_key = {target.key: target for target in contract.targets}
                for key in contract.logical_keys:
                    tensor_slice = handle.get_slice(key)
                    target = target_by_key[key]
                    if tuple(tensor_slice.get_shape()) != target.shape:
                        raise ValueError(f"checkpoint tensor {key} header shape does not match")
                    if tensor_slice.get_dtype() != target.safe_dtype:
                        raise ValueError(f"checkpoint tensor {key} header dtype does not match")
                staged = _stage_loaded_tensors(handle, contract)

            after_staging = _component_contract(
                adapter_bank,
                amortizer,
                expected_provenance,
            )
            _assert_same_precommit_contract(contract, after_staging)
            _assert_target_values_match(contract.targets, originals, "checkpoint destination")
            if _sha256_bytes(_read_descriptor(descriptor)) != expected_file.sha256:
                raise ValueError("checkpoint changed during materialization")
        _verify_private_directory(directory)
        _commit_transaction(
            contract,
            staged,
            originals,
            adapter_bank,
            amortizer,
            expected_provenance,
            postcommit_validate=lambda: _verify_private_directory(directory),
        )
        return contract.metadata
    finally:
        os.close(directory.descriptor)
