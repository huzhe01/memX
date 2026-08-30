from __future__ import annotations

import pytest
import torch

from ratemem.runtime.device import DeviceRuntime, RuntimeProbe, resolve_runtime


def accelerator_probe(
    *,
    count: int = 8,
    names: tuple[str, ...] | None = None,
    bf16: bool = True,
    backends: tuple[str, ...] = ("gloo", "nccl", "pccl"),
) -> RuntimeProbe:
    resolved_names = names if names is not None else tuple(f"accelerator-{i}" for i in range(count))
    return RuntimeProbe(
        accelerator_available=True,
        device_count=count,
        device_names=resolved_names,
        bf16_supported=bf16,
        available_backends=backends,
    )


def cpu_probe() -> RuntimeProbe:
    return RuntimeProbe(
        accelerator_available=False,
        device_count=0,
        device_names=(),
        bf16_supported=False,
        available_backends=("gloo",),
    )


def test_explicit_ppu_uses_pccl_without_marketing_name_dependency() -> None:
    runtime = resolve_runtime("ppu", accelerator_probe())

    assert runtime == DeviceRuntime(
        kind="ppu",
        device=torch.device("cuda"),
        distributed_backend="pccl",
        device_count=8,
        device_names=tuple(f"accelerator-{i}" for i in range(8)),
        bf16_supported=True,
    )


def test_missing_ppu_never_falls_back_to_cpu() -> None:
    with pytest.raises(RuntimeError, match="requested PPU accelerator is unavailable"):
        resolve_runtime("ppu", cpu_probe())


def test_explicit_nvidia_uses_nccl() -> None:
    runtime = resolve_runtime(
        "nvidia",
        accelerator_probe(count=2, names=("NVIDIA H20", "NVIDIA H20")),
    )

    assert runtime.kind == "nvidia"
    assert runtime.distributed_backend == "nccl"
    assert runtime.device == torch.device("cuda")


def test_explicit_cpu_uses_gloo_even_when_accelerators_exist() -> None:
    runtime = resolve_runtime("cpu", accelerator_probe())

    assert runtime.kind == "cpu"
    assert runtime.device == torch.device("cpu")
    assert runtime.distributed_backend == "gloo"
    assert runtime.device_count == 0
    assert runtime.device_names == ()
    assert runtime.bf16_supported is False


def test_auto_falls_back_only_when_accelerator_was_not_explicit() -> None:
    runtime = resolve_runtime("auto", cpu_probe())

    assert runtime.kind == "cpu"
    assert runtime.device == torch.device("cpu")


def test_auto_uses_nvidia_for_an_available_generic_accelerator() -> None:
    runtime = resolve_runtime("auto", accelerator_probe(count=1))

    assert runtime.kind == "nvidia"
    assert runtime.distributed_backend == "nccl"


def test_backend_override_must_be_observed() -> None:
    with pytest.raises(RuntimeError, match="distributed backend custom is unavailable"):
        resolve_runtime("ppu", accelerator_probe(), backend_override="custom")


def test_backend_override_supports_vendor_compatibility_registration() -> None:
    runtime = resolve_runtime(
        "ppu",
        accelerator_probe(backends=("gloo", "vendor_pccl")),
        backend_override="vendor_pccl",
    )

    assert runtime.distributed_backend == "vendor_pccl"


@pytest.mark.parametrize("requested", ["", "cuda", "PPU", "gpu", " ppu"])
def test_unknown_device_request_is_rejected(requested: str) -> None:
    with pytest.raises(ValueError, match="auto, cpu, nvidia, or ppu"):
        resolve_runtime(requested, cpu_probe())


def test_probe_rejects_name_count_mismatch() -> None:
    with pytest.raises(ValueError, match="device_names"):
        RuntimeProbe(
            accelerator_available=True,
            device_count=2,
            device_names=("only-one",),
            bf16_supported=True,
            available_backends=("gloo", "pccl"),
        )


def test_runtime_manifest_contains_only_declared_inventory() -> None:
    runtime = resolve_runtime(
        "ppu",
        accelerator_probe(count=1, names=("PPU-ZW810E",)),
    )

    assert runtime.as_manifest() == {
        "kind": "ppu",
        "device": "cuda",
        "distributed_backend": "pccl",
        "device_count": 1,
        "device_names": ["PPU-ZW810E"],
        "bf16_supported": True,
    }
