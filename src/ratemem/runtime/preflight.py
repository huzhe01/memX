from __future__ import annotations

from ratemem.runtime.device import DeviceRuntime
from ratemem.runtime.distributed import RankEnvironment


def _version(value: object, name: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be non-empty canonical text")
    return value


def validate_preflight(
    runtime: DeviceRuntime,
    ranks: RankEnvironment,
    *,
    ppu_compatible_backends: tuple[str, ...] = ("pccl",),
) -> None:
    if type(runtime) is not DeviceRuntime:
        raise TypeError("runtime must be an exact DeviceRuntime")
    if type(ranks) is not RankEnvironment:
        raise TypeError("ranks must be an exact RankEnvironment")
    if type(ppu_compatible_backends) is not tuple or not ppu_compatible_backends:
        raise TypeError("PPU-compatible backends must be a non-empty exact tuple")
    if any(type(value) is not str or not value for value in ppu_compatible_backends):
        raise TypeError("every PPU-compatible backend must be a non-empty exact str")

    if runtime.kind == "cpu":
        if runtime.distributed_backend != "gloo":
            raise RuntimeError("CPU runtime requires gloo")
        return

    if not runtime.bf16_supported:
        raise RuntimeError("production accelerator requires BF16 support")
    if ranks.local_world_size > runtime.device_count:
        raise RuntimeError("local world size exceeds visible device count")
    if runtime.kind == "nvidia" and runtime.distributed_backend != "nccl":
        raise RuntimeError("NVIDIA runtime requires NCCL")
    if runtime.kind == "ppu" and runtime.distributed_backend not in ppu_compatible_backends:
        raise RuntimeError("PPU runtime requires a declared PCCL-compatible backend")


def build_preflight_receipt(
    runtime: DeviceRuntime,
    ranks: RankEnvironment,
    *,
    torch_version: str,
    python_version: str,
    ppu_compatible_backends: tuple[str, ...] = ("pccl",),
) -> dict[str, object]:
    validate_preflight(
        runtime,
        ranks,
        ppu_compatible_backends=ppu_compatible_backends,
    )
    return {
        "schema_version": "memx-runtime-preflight-v1",
        "status": "passed",
        "runtime": runtime.as_manifest(),
        "distributed": ranks.as_manifest(),
        "software": {
            "python": _version(python_version, "python_version"),
            "torch": _version(torch_version, "torch_version"),
        },
    }
