from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

_MAPPING_PROXY_TYPE: type[object] = type(MappingProxyType({}))


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


def _validate_owned_payload(name: str, value: object) -> None:
    if type(value) is not bytes:
        raise TypeError(f"{name} payload must be exact bytes")


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

        object.__setattr__(self, "bases", MappingProxyType(bases))
        object.__setattr__(self, "packets", MappingProxyType(packets))
        object.__setattr__(self, "incidences", MappingProxyType(incidences))
        _validate_state_runtime(self)

    @property
    def serialized_bytes(self) -> int:
        from ratemem.state.serialization import encode_state

        return len(encode_state(self))


def _validate_base_record(record: object) -> None:
    if type(record) is not BaseRecord:
        raise TypeError("bases values must be exact BaseRecord instances")
    _validate_identity("handle", record.handle)
    _validate_owned_payload("base", record.payload)
    _validate_integer("reads", record.reads, 0, 0xFFFFFFFFFFFFFFFF)
    _validate_integer("created_at", record.created_at, 0, 0xFFFFFFFFFFFFFFFF)


def _validate_packet_record(packet: object) -> None:
    if type(packet) is not Packet:
        raise TypeError("packets values must be exact Packet instances")
    _validate_identity("packet_id", packet.packet_id)
    _validate_owned_payload("packet", packet.payload)


def _validate_incidence_record(incidence: object) -> None:
    if type(incidence) is not Incidence:
        raise TypeError("incidences values must be exact Incidence instances")
    _validate_identity("handle", incidence.handle)
    _validate_identity("packet_id", incidence.packet_id)
    _validate_integer("gain_q", incidence.gain_q, -0x8000, 0x7FFF)


def _validate_state_runtime(
    state: object,
    *,
    require_references: bool = False,
    reject_orphans: bool = False,
    require_hashes: bool = False,
) -> None:
    """Revalidate a state without normalizing or mutating any caller-owned value."""
    if type(state) is not MemoryState:
        raise TypeError("state must be an exact MemoryState instance")
    if type(state.bases) is not _MAPPING_PROXY_TYPE:
        raise TypeError("bases must be an owned immutable mapping")
    if type(state.packets) is not _MAPPING_PROXY_TYPE:
        raise TypeError("packets must be an owned immutable mapping")
    if type(state.incidences) is not _MAPPING_PROXY_TYPE:
        raise TypeError("incidences must be an owned immutable mapping")

    for base_key in state.bases:
        _validate_identity("base mapping key", base_key)
    for packet_key in state.packets:
        _validate_identity("packet mapping key", packet_key)
    for incidence_key in state.incidences:
        _validate_incidence_mapping_key(incidence_key)

    for record in state.bases.values():
        _validate_base_record(record)
    for packet in state.packets.values():
        _validate_packet_record(packet)
    for edge in state.incidences.values():
        _validate_incidence_record(edge)

    if len({record.handle for record in state.bases.values()}) != len(state.bases):
        raise ValueError("duplicate embedded base identity")
    if len({packet.packet_id for packet in state.packets.values()}) != len(
        state.packets
    ):
        raise ValueError("duplicate embedded packet identity")
    if len(
        {(edge.handle, edge.packet_id) for edge in state.incidences.values()}
    ) != len(state.incidences):
        raise ValueError("duplicate embedded incidence identity")

    if any(key != record.handle for key, record in state.bases.items()):
        raise ValueError("base mapping key mismatch")
    if any(key != packet.packet_id for key, packet in state.packets.items()):
        raise ValueError("packet mapping key mismatch")
    if any(
        key != (edge.handle, edge.packet_id)
        for key, edge in state.incidences.items()
    ):
        raise ValueError("incidence mapping key mismatch")

    if require_hashes:
        for packet in state.packets.values():
            if hashlib.sha256(packet.payload).hexdigest() != packet.packet_id:
                raise ValueError("packet hash mismatch")

    if require_references or reject_orphans:
        referenced_packets: set[str] = set()
        for edge in state.incidences.values():
            if require_references and (
                edge.handle not in state.bases
                or edge.packet_id not in state.packets
            ):
                raise ValueError("dangling packet incidence")
            referenced_packets.add(edge.packet_id)
        if reject_orphans and state.packets.keys() - referenced_packets:
            raise ValueError("orphan packet")
