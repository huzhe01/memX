from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from ratemem.runtime.device import DeviceRuntime

_RANK_NAMES = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
_CANONICAL_NONNEGATIVE = re.compile(r"0|[1-9][0-9]*")


def _nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _canonical_environment_int(value: object, name: str) -> int:
    if type(value) is not str or _CANONICAL_NONNEGATIVE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical nonnegative decimal integer")
    return int(value)


@dataclass(frozen=True, slots=True)
class RankEnvironment:
    rank: int
    local_rank: int
    world_size: int
    local_world_size: int

    def __post_init__(self) -> None:
        rank = _nonnegative_int(self.rank, "rank")
        local_rank = _nonnegative_int(self.local_rank, "local_rank")
        world_size = _nonnegative_int(self.world_size, "world_size")
        local_world_size = _nonnegative_int(self.local_world_size, "local_world_size")
        if world_size < 1 or local_world_size < 1:
            raise ValueError("world sizes must be positive")
        if rank >= world_size:
            raise ValueError("RANK is outside WORLD_SIZE")
        if local_rank >= local_world_size:
            raise ValueError("LOCAL_RANK is outside LOCAL_WORLD_SIZE")
        if world_size % local_world_size:
            raise ValueError("LOCAL_WORLD_SIZE must divide WORLD_SIZE")
        if rank % local_world_size != local_rank:
            raise ValueError("RANK and LOCAL_RANK do not describe a canonical node layout")

    @property
    def node_count(self) -> int:
        return self.world_size // self.local_world_size

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, str],
        *,
        visible_devices: int,
    ) -> RankEnvironment:
        if not isinstance(values, Mapping):
            raise TypeError("rank environment must be a mapping")
        checked_visible = _nonnegative_int(visible_devices, "visible_devices")
        present = tuple(name for name in _RANK_NAMES if name in values)
        if not present:
            return cls(rank=0, local_rank=0, world_size=1, local_world_size=1)
        if len(present) != len(_RANK_NAMES):
            missing = ", ".join(name for name in _RANK_NAMES if name not in values)
            raise ValueError(f"distributed environment is incomplete; missing {missing}")
        parsed = {
            name: _canonical_environment_int(values[name], name) for name in _RANK_NAMES
        }
        result = cls(
            rank=parsed["RANK"],
            local_rank=parsed["LOCAL_RANK"],
            world_size=parsed["WORLD_SIZE"],
            local_world_size=parsed["LOCAL_WORLD_SIZE"],
        )
        if checked_visible and result.local_world_size > checked_visible:
            raise ValueError("LOCAL_WORLD_SIZE exceeds visible device count")
        return result

    def as_manifest(self) -> dict[str, int]:
        return {
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "local_world_size": self.local_world_size,
            "node_count": self.node_count,
        }


@dataclass(frozen=True, slots=True)
class DistributedContext:
    runtime: DeviceRuntime
    ranks: RankEnvironment

    def __post_init__(self) -> None:
        if type(self.runtime) is not DeviceRuntime:
            raise TypeError("runtime must be an exact DeviceRuntime")
        if type(self.ranks) is not RankEnvironment:
            raise TypeError("ranks must be an exact RankEnvironment")

    @property
    def is_primary(self) -> bool:
        return self.ranks.rank == 0

    @property
    def device(self) -> torch.device:
        if self.runtime.kind == "cpu":
            return torch.device("cpu")
        return torch.device("cuda", self.ranks.local_rank)


@contextmanager
def distributed_session(
    runtime: DeviceRuntime,
    ranks: RankEnvironment,
) -> Iterator[DistributedContext]:
    if type(runtime) is not DeviceRuntime:
        raise TypeError("runtime must be an exact DeviceRuntime")
    if type(ranks) is not RankEnvironment:
        raise TypeError("ranks must be an exact RankEnvironment")
    if runtime.kind != "cpu":
        torch.cuda.set_device(ranks.local_rank)

    owned_group = False
    if ranks.world_size > 1:
        if torch.distributed.is_initialized():
            if (
                torch.distributed.get_rank() != ranks.rank
                or torch.distributed.get_world_size() != ranks.world_size
            ):
                raise RuntimeError("existing process group differs from rank environment")
        else:
            torch.distributed.init_process_group(
                backend=runtime.distributed_backend,
                rank=ranks.rank,
                world_size=ranks.world_size,
            )
            owned_group = True

    context = DistributedContext(runtime=runtime, ranks=ranks)
    completed = False
    try:
        yield context
        completed = True
    finally:
        if completed and ranks.world_size > 1:
            torch.distributed.barrier()
        if owned_group and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
