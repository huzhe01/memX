from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


def _owned_payload(payload: bytes | bytearray | memoryview) -> bytes:
    if not isinstance(payload, bytes | bytearray | memoryview):
        raise TypeError("payload must be bytes-like")
    return bytes(payload)


@dataclass(frozen=True, slots=True)
class BaseRecord:
    handle: str
    payload: bytes
    reads: int
    created_at: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _owned_payload(self.payload))
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _owned_payload(self.payload))


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
        bases = dict(self.bases)
        packets = dict(self.packets)
        incidences = dict(self.incidences)

        if len({record.handle for record in bases.values()}) != len(bases):
            raise ValueError("duplicate embedded base identity")
        if len({packet.packet_id for packet in packets.values()}) != len(packets):
            raise ValueError("duplicate embedded packet identity")
        if len(
            {(edge.handle, edge.packet_id) for edge in incidences.values()}
        ) != len(incidences):
            raise ValueError("duplicate embedded incidence identity")

        if any(key != record.handle for key, record in bases.items()):
            raise ValueError("base mapping key mismatch")
        if any(key != packet.packet_id for key, packet in packets.items()):
            raise ValueError("packet mapping key mismatch")
        if any(
            key != (edge.handle, edge.packet_id)
            for key, edge in incidences.items()
        ):
            raise ValueError("incidence mapping key mismatch")

        object.__setattr__(self, "bases", MappingProxyType(bases))
        object.__setattr__(self, "packets", MappingProxyType(packets))
        object.__setattr__(self, "incidences", MappingProxyType(incidences))

    @property
    def serialized_bytes(self) -> int:
        from ratemem.state.serialization import encode_state

        return len(encode_state(self))
