from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

import torch

RuntimeKind = Literal["cpu", "nvidia", "ppu"]

_REQUESTED_KINDS = frozenset({"auto", "cpu", "nvidia", "ppu"})
_RUNTIME_KINDS = frozenset({"cpu", "nvidia", "ppu"})


def _exact_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be an exact bool")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int")
    checked = value
    if checked < 0:
        raise ValueError(f"{name} must be nonnegative")
    return checked


def _string_tuple(value: object, name: str, *, allow_empty: bool) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be an exact tuple")
    checked = cast(tuple[object, ...], value)
    if not allow_empty and not checked:
        raise ValueError(f"{name} must not be empty")
    if any(type(item) is not str or not item.strip() for item in checked):
        raise TypeError(f"every {name} entry must be a non-empty exact str")
    return cast(tuple[str, ...], checked)


def _backend(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a non-empty exact str")
    checked = value
    if checked != checked.strip() or checked.lower() != checked:
        raise ValueError(f"{name} must be canonical lowercase text")
    return checked


@dataclass(frozen=True, slots=True)
class RuntimeProbe:
    """Observed accelerator facts, separated from runtime policy for testability."""

    accelerator_available: bool
    device_count: int
    device_names: tuple[str, ...]
    bf16_supported: bool
    available_backends: tuple[str, ...]

    def __post_init__(self) -> None:
        available = _exact_bool(self.accelerator_available, "accelerator_available")
        count = _nonnegative_int(self.device_count, "device_count")
        names = _string_tuple(self.device_names, "device_names", allow_empty=True)
        _exact_bool(self.bf16_supported, "bf16_supported")
        backends = _string_tuple(
            self.available_backends, "available_backends", allow_empty=False
        )
        if len(backends) != len(set(backends)):
            raise ValueError("available_backends must not contain duplicates")
        for backend in backends:
            _backend(backend, "available backend")
        if available:
            if count < 1:
                raise ValueError("an available accelerator requires a positive device_count")
            if len(names) != count:
                raise ValueError("device_names length must equal device_count")
        elif count != 0 or names:
            raise ValueError("an unavailable accelerator requires zero devices and no device_names")


@dataclass(frozen=True, slots=True)
class DeviceRuntime:
    """Resolved runtime contract consumed by training and launch preflight."""

    kind: RuntimeKind
    device: torch.device
    distributed_backend: str
    device_count: int
    device_names: tuple[str, ...]
    bf16_supported: bool

    def __post_init__(self) -> None:
        if type(self.kind) is not str or self.kind not in _RUNTIME_KINDS:
            raise ValueError("runtime kind must be cpu, nvidia, or ppu")
        if type(self.device) is not torch.device:
            raise TypeError("runtime device must be an exact torch.device")
        _backend(self.distributed_backend, "distributed_backend")
        count = _nonnegative_int(self.device_count, "device_count")
        names = _string_tuple(self.device_names, "device_names", allow_empty=True)
        _exact_bool(self.bf16_supported, "bf16_supported")
        expected_device = "cpu" if self.kind == "cpu" else "cuda"
        if self.device.type != expected_device or self.device.index is not None:
            raise ValueError(f"{self.kind} runtime requires unindexed {expected_device} device")
        if self.kind == "cpu":
            if count != 0 or names or self.bf16_supported:
                raise ValueError("CPU runtime cannot expose accelerator inventory")
        elif count < 1 or len(names) != count:
            raise ValueError("accelerator runtime inventory is inconsistent")

    def as_manifest(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "device": self.device.type,
            "distributed_backend": self.distributed_backend,
            "device_count": self.device_count,
            "device_names": list(self.device_names),
            "bf16_supported": self.bf16_supported,
        }


def resolve_runtime(
    requested: str,
    probe: RuntimeProbe,
    *,
    backend_override: str | None = None,
) -> DeviceRuntime:
    """Resolve explicit accelerator policy without hiding an unavailable request."""

    if type(requested) is not str or requested not in _REQUESTED_KINDS:
        raise ValueError("device must be one of auto, cpu, nvidia, or ppu")
    if type(probe) is not RuntimeProbe:
        raise TypeError("probe must be an exact RuntimeProbe")
    if backend_override is not None:
        override = _backend(backend_override, "backend_override")
    else:
        override = None

    if requested == "cpu" or (requested == "auto" and not probe.accelerator_available):
        backend = override or "gloo"
        if backend not in probe.available_backends:
            raise RuntimeError(f"distributed backend {backend} is unavailable")
        return DeviceRuntime(
            kind="cpu",
            device=torch.device("cpu"),
            distributed_backend=backend,
            device_count=0,
            device_names=(),
            bf16_supported=False,
        )

    if not probe.accelerator_available:
        raise RuntimeError(f"requested {requested.upper()} accelerator is unavailable")

    kind: RuntimeKind = "ppu" if requested == "ppu" else "nvidia"
    backend = override or ("pccl" if kind == "ppu" else "nccl")
    if backend not in probe.available_backends:
        raise RuntimeError(f"distributed backend {backend} is unavailable")
    return DeviceRuntime(
        kind=kind,
        device=torch.device("cuda"),
        distributed_backend=backend,
        device_count=probe.device_count,
        device_names=probe.device_names,
        bf16_supported=probe.bf16_supported,
    )


def observe_runtime_probe(
    *,
    additional_backends: Sequence[str] = (),
) -> RuntimeProbe:
    """Observe only accelerator and collective facts needed by launch policy."""

    if not isinstance(additional_backends, Sequence) or isinstance(
        additional_backends, str | bytes
    ):
        raise TypeError("additional_backends must be a non-string sequence")
    requested_backends = ["gloo", "nccl", "mpi", "ucc", "pccl"]
    for value in additional_backends:
        checked = _backend(value, "additional backend")
        if checked not in requested_backends:
            requested_backends.append(checked)
    available_backends: list[str] = []
    for backend in requested_backends:
        try:
            available = bool(torch.distributed.is_backend_available(backend))
        except (AttributeError, RuntimeError, ValueError):
            available = False
        if available:
            available_backends.append(backend)
    if not available_backends:
        raise RuntimeError("PyTorch exposes no supported distributed backend")

    accelerator_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if accelerator_available else 0
    device_names = (
        tuple(str(torch.cuda.get_device_name(index)) for index in range(device_count))
        if accelerator_available
        else ()
    )
    bf16_probe = getattr(torch.cuda, "is_bf16_supported", None)
    bf16_supported = bool(bf16_probe()) if accelerator_available and callable(bf16_probe) else False
    return RuntimeProbe(
        accelerator_available=accelerator_available,
        device_count=device_count,
        device_names=device_names,
        bf16_supported=bf16_supported,
        available_backends=tuple(available_backends),
    )
