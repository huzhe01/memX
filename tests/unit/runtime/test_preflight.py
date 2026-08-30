from __future__ import annotations

import pytest
import torch

from ratemem.runtime.device import DeviceRuntime
from ratemem.runtime.distributed import RankEnvironment
from ratemem.runtime.preflight import build_preflight_receipt, validate_preflight


def ppu_runtime(*, device_count: int = 8, bf16: bool = True) -> DeviceRuntime:
    return DeviceRuntime(
        kind="ppu",
        device=torch.device("cuda"),
        distributed_backend="pccl",
        device_count=device_count,
        device_names=tuple("PPU-ZW810E" for _ in range(device_count)),
        bf16_supported=bf16,
    )


def cpu_runtime() -> DeviceRuntime:
    return DeviceRuntime(
        kind="cpu",
        device=torch.device("cpu"),
        distributed_backend="gloo",
        device_count=0,
        device_names=(),
        bf16_supported=False,
    )


def ranks(*, world_size: int, local_world_size: int) -> RankEnvironment:
    return RankEnvironment(
        rank=0,
        local_rank=0,
        world_size=world_size,
        local_world_size=local_world_size,
    )


def test_preflight_rejects_local_world_larger_than_visible_devices() -> None:
    with pytest.raises(RuntimeError, match="visible device count"):
        validate_preflight(
            ppu_runtime(device_count=4),
            ranks(world_size=8, local_world_size=8),
        )


def test_production_accelerator_requires_bf16() -> None:
    with pytest.raises(RuntimeError, match="BF16"):
        validate_preflight(
            ppu_runtime(device_count=8, bf16=False),
            ranks(world_size=8, local_world_size=8),
        )


def test_ppu_runtime_requires_pccl_name_or_explicit_compatibility_override() -> None:
    runtime = DeviceRuntime(
        kind="ppu",
        device=torch.device("cuda"),
        distributed_backend="nccl",
        device_count=8,
        device_names=tuple("PPU-ZW810E" for _ in range(8)),
        bf16_supported=True,
    )
    with pytest.raises(RuntimeError, match="PCCL-compatible"):
        validate_preflight(runtime, ranks(world_size=8, local_world_size=8))


def test_explicit_vendor_pccl_compatibility_backend_is_accepted() -> None:
    runtime = DeviceRuntime(
        kind="ppu",
        device=torch.device("cuda"),
        distributed_backend="vendor_pccl",
        device_count=8,
        device_names=tuple("accelerator" for _ in range(8)),
        bf16_supported=True,
    )

    validate_preflight(
        runtime,
        ranks(world_size=8, local_world_size=8),
        ppu_compatible_backends=("pccl", "vendor_pccl"),
    )


def test_cpu_fixture_preflight_accepts_single_process_gloo() -> None:
    validate_preflight(cpu_runtime(), ranks(world_size=1, local_world_size=1))


def test_cpu_preflight_rejects_accelerator_collective() -> None:
    runtime = DeviceRuntime(
        kind="cpu",
        device=torch.device("cpu"),
        distributed_backend="nccl",
        device_count=0,
        device_names=(),
        bf16_supported=False,
    )
    with pytest.raises(RuntimeError, match="CPU runtime requires gloo"):
        validate_preflight(runtime, ranks(world_size=1, local_world_size=1))


def test_preflight_receipt_binds_runtime_and_rank_inventory() -> None:
    runtime = ppu_runtime(device_count=8)
    rank_environment = ranks(world_size=16, local_world_size=8)

    receipt = build_preflight_receipt(
        runtime,
        rank_environment,
        torch_version="2.8.0+ppu",
        python_version="3.12.4",
    )

    assert receipt == {
        "schema_version": "memx-runtime-preflight-v1",
        "status": "passed",
        "runtime": runtime.as_manifest(),
        "distributed": rank_environment.as_manifest(),
        "software": {"python": "3.12.4", "torch": "2.8.0+ppu"},
    }
