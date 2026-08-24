from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypeVar

_K = TypeVar("_K")
_V = TypeVar("_V")


def _frozen_copy(values: Mapping[_K, _V]) -> Mapping[_K, _V]:
    return MappingProxyType(dict(values))


@dataclass(frozen=True, slots=True)
class BaseRecord:
    handle: str
    payload: bytes
    reads: int
    created_at: int

    def __post_init__(self) -> None:
        if not self.handle:
            raise ValueError("handle must be nonempty")
        if not 0 <= self.reads <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("reads must fit uint64")
        if not 0 <= self.created_at <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("created_at must fit uint64")


@dataclass(frozen=True, slots=True)
class Packet:
    packet_id: str
    payload: bytes


@dataclass(frozen=True, slots=True)
class Incidence:
    handle: str
    packet_id: str
    gain_q: int

    def __post_init__(self) -> None:
        if not -0x8000 <= self.gain_q <= 0x7FFF:
            raise ValueError("gain_q must fit int16")


@dataclass(frozen=True, slots=True)
class MemoryState:
    bases: Mapping[str, BaseRecord] = field(default_factory=dict)
    packets: Mapping[str, Packet] = field(default_factory=dict)
    incidences: Mapping[tuple[str, str], Incidence] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bases", _frozen_copy(self.bases))
        object.__setattr__(self, "packets", _frozen_copy(self.packets))
        object.__setattr__(self, "incidences", _frozen_copy(self.incidences))

    @property
    def serialized_bytes(self) -> int:
        from ratemem.state.serialization import encode_state

        return len(encode_state(self))
