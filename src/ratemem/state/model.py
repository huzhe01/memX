from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


def _owned_payload(payload: bytes | bytearray | memoryview) -> bytes:
    if not isinstance(payload, bytes | bytearray | memoryview):
        raise TypeError("payload must be bytes-like")
    return bytes(payload)


def _validate_identity(name: str, value: object) -> None:
    if type(value) is not str:
        raise TypeError(f"{name} must be a nonempty string")
    if not value:
        raise ValueError(f"{name} must be a nonempty string")


def _validate_integer(name: str, value: object, lower: int, upper: int) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if not lower <= value <= upper:
        width = "uint64" if lower == 0 else "int16"
        raise ValueError(f"{name} must fit {width}")


def _validate_incidence_mapping_key(key: object) -> None:
    if type(key) is not tuple or len(key) != 2:
        raise TypeError(
            "incidence mapping key must be an exact pair of nonempty strings"
        )
    handle, packet_id = key
    if type(handle) is not str or type(packet_id) is not str:
        raise TypeError(
            "incidence mapping key must be an exact pair of nonempty strings"
        )
    if not handle or not packet_id:
        raise ValueError(
            "incidence mapping key must be an exact pair of nonempty strings"
        )


@dataclass(frozen=True, slots=True)
class BaseRecord:
    handle: str
    payload: bytes
    reads: int
    created_at: int

    def __post_init__(self) -> None:
        _validate_identity("handle", self.handle)
        object.__setattr__(self, "payload", _owned_payload(self.payload))
        _validate_integer("reads", self.reads, 0, 0xFFFFFFFFFFFFFFFF)
        _validate_integer("created_at", self.created_at, 0, 0xFFFFFFFFFFFFFFFF)


@dataclass(frozen=True, slots=True)
class Packet:
    packet_id: str
    payload: bytes

    def __post_init__(self) -> None:
        _validate_identity("packet_id", self.packet_id)
        object.__setattr__(self, "payload", _owned_payload(self.payload))


@dataclass(frozen=True, slots=True)
class Incidence:
    handle: str
    packet_id: str
    gain_q: int

    def __post_init__(self) -> None:
        _validate_identity("handle", self.handle)
        _validate_identity("packet_id", self.packet_id)
        _validate_integer("gain_q", self.gain_q, -0x8000, 0x7FFF)


@dataclass(frozen=True, slots=True)
class MemoryState:
    bases: Mapping[str, BaseRecord] = field(default_factory=dict)
    packets: Mapping[str, Packet] = field(default_factory=dict)
    incidences: Mapping[tuple[str, str], Incidence] = field(default_factory=dict)

    def __post_init__(self) -> None:
        bases = dict(self.bases)
        packets = dict(self.packets)
        incidences = dict(self.incidences)

        for base_key in bases:
            _validate_identity("base mapping key", base_key)
        for packet_key in packets:
            _validate_identity("packet mapping key", packet_key)
        for incidence_key in incidences:
            _validate_incidence_mapping_key(incidence_key)

        if any(type(record) is not BaseRecord for record in bases.values()):
            raise TypeError("bases values must be exact BaseRecord instances")
        if any(type(packet) is not Packet for packet in packets.values()):
            raise TypeError("packets values must be exact Packet instances")
        if any(type(edge) is not Incidence for edge in incidences.values()):
            raise TypeError("incidences values must be exact Incidence instances")

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
