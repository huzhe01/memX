"""Closed engineering-pilot probes and exact timing/cap primitives."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from typing import Final

import torch

ALLOWED_PROBES: Final = (
    "checkpoint_compatibility",
    "dynamic_numerics",
    "gradient_flow",
    "frozen_backbone",
    "peak_memory",
    "one_step_inference",
    "one_timestep_backward",
    "step_timing",
    "held_in_loss",
)


def percentile(values: list[float], quantile: float) -> float:
    """Return the nearest-rank percentile without interpolation."""

    if type(values) is not list:
        raise TypeError("percentile samples must be an exact list")
    if type(quantile) is not float:
        raise TypeError("percentile quantile must be an exact float")
    if not math.isfinite(quantile) or not 0 < quantile <= 1:
        raise ValueError("percentile quantile must be finite and in (0, 1]")
    if not values:
        raise ValueError("percentile requires at least one sample")
    if any(type(value) is not float for value in values):
        raise TypeError("percentile samples must be exact floats")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("percentile samples must be finite")
    ordered = sorted(values)
    rank = math.ceil(quantile * len(ordered))
    return ordered[rank - 1]


def _decimal(
    value: object,
    name: str,
    *,
    positive: bool = False,
) -> Decimal:
    if type(value) is not Decimal:
        raise TypeError(f"{name} must be an exact Decimal")
    checked = value
    if not checked.is_finite() or checked < 0 or (positive and checked <= 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return checked


def _nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int")
    checked = value
    if checked < 0:
        raise ValueError(f"{name} must be nonnegative")
    return checked


def held_in_step_cap(
    *,
    p95_step_seconds: Decimal,
    remaining_compute_usd: Decimal,
    requested_resource_usd_per_second: Decimal,
    remaining_timeout_seconds: int,
    shutdown_reserve_seconds: int,
) -> int:
    """Bound held-in optimizer steps by both exact cost and remaining wall time."""

    p95 = _decimal(p95_step_seconds, "p95_step_seconds", positive=True)
    remaining = _decimal(
        remaining_compute_usd,
        "remaining_compute_usd",
    )
    resource_rate = _decimal(
        requested_resource_usd_per_second,
        "requested_resource_usd_per_second",
        positive=True,
    )
    timeout = _nonnegative_int(
        remaining_timeout_seconds,
        "remaining_timeout_seconds",
    )
    shutdown_reserve = _nonnegative_int(
        shutdown_reserve_seconds,
        "shutdown_reserve_seconds",
    )
    usable_timeout = max(0, timeout - shutdown_reserve)
    seconds_from_cost = remaining / resource_rate
    usable_seconds = min(Decimal(usable_timeout), seconds_from_cost)
    cap = int(usable_seconds // p95)
    if cap < 0:
        raise RuntimeError("held-in step cap became negative")
    return cap


@dataclass(frozen=True, slots=True)
class CudaPeak:
    allocated_bytes: int
    reserved_bytes: int

    def __post_init__(self) -> None:
        for name in ("allocated_bytes", "reserved_bytes"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact int")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.reserved_bytes < self.allocated_bytes:
            raise ValueError("reserved CUDA bytes must cover allocated CUDA bytes")


@contextmanager
def cuda_peak() -> Iterator[dict[str, CudaPeak]]:
    """Record synchronized peak CUDA allocation for one declared probe region."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the paid pilot")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    result: dict[str, CudaPeak] = {}
    try:
        yield result
    finally:
        torch.cuda.synchronize()
        result["peak"] = CudaPeak(
            allocated_bytes=int(torch.cuda.max_memory_allocated()),
            reserved_bytes=int(torch.cuda.max_memory_reserved()),
        )


def timed(callable_: Callable[[], object]) -> float:
    """Synchronously time exactly one already-declared CUDA operation."""

    if not callable(callable_):
        raise TypeError("timed requires one callable")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for synchronized pilot timing")
    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        callable_()
    finally:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if type(elapsed) is not float or not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError("synchronized pilot timing must be finite and positive")
    return elapsed
