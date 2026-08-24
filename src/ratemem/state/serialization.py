from __future__ import annotations

import hashlib
import struct
from typing import Any, cast

import cbor2

from ratemem.state.model import (
    BaseRecord,
    Incidence,
    MemoryState,
    Packet,
    _owned_payload,
)

_MAGIC = b"RTMEM001"
_VERSION = 1
_HEADER = struct.Struct("<8sIQQQ")
_LENGTH = struct.Struct("<I")
_UINT64 = struct.Struct("<Q")
_INT16 = struct.Struct("<h")


def _frame(row: list[Any]) -> bytes:
    payload = cbor2.dumps(row, canonical=True)
    if len(payload) > 0xFFFFFFFF:
        raise ValueError("record exceeds the uint32 frame limit")
    return _LENGTH.pack(len(payload)) + payload


def _base_frame(record: BaseRecord) -> bytes:
    return _frame(
        [
            record.handle,
            record.payload,
            _UINT64.pack(record.reads),
            _UINT64.pack(record.created_at),
        ]
    )


def _packet_frame(packet: Packet) -> bytes:
    return _frame([packet.packet_id, packet.payload])


def _incidence_frame(incidence: Incidence) -> bytes:
    return _frame(
        [incidence.handle, incidence.packet_id, _INT16.pack(incidence.gain_q)]
    )


def packet_from_payload(payload: bytes | bytearray | memoryview) -> Packet:
    owned_payload = _owned_payload(payload)
    packet_id = hashlib.sha256(owned_payload).hexdigest()
    return Packet(packet_id=packet_id, payload=owned_payload)


def bundle_cost_bytes(packet: Packet, incidences: tuple[Incidence, ...]) -> int:
    if not incidences:
        raise ValueError("packet bundle must contain at least one incidence")
    if any(edge.packet_id != packet.packet_id for edge in incidences):
        raise ValueError("bundle incidence points at another packet")
    if len({edge.handle for edge in incidences}) != len(incidences):
        raise ValueError("packet bundle repeats a concept incidence")
    return len(_packet_frame(packet)) + sum(
        len(_incidence_frame(edge)) for edge in incidences
    )


def encode_state(state: MemoryState) -> bytes:
    bases = sorted(state.bases.values(), key=lambda item: item.handle)
    packets = sorted(state.packets.values(), key=lambda item: item.packet_id)
    incidences = sorted(
        state.incidences.values(), key=lambda item: (item.handle, item.packet_id)
    )
    output = bytearray(
        _HEADER.pack(_MAGIC, _VERSION, len(bases), len(packets), len(incidences))
    )
    for record in bases:
        output.extend(_base_frame(record))
    for packet in packets:
        output.extend(_packet_frame(packet))
    for incidence in incidences:
        output.extend(_incidence_frame(incidence))
    return bytes(output)


def _decode_canonical_row(payload: bytes) -> list[object]:
    try:
        decoded: object = cbor2.loads(payload)
        canonical = cbor2.dumps(decoded, canonical=True)
    except (cbor2.CBORDecodeError, cbor2.CBOREncodeError) as error:
        raise ValueError("invalid CBOR record") from error
    if canonical != payload:
        raise ValueError("record is not one canonical CBOR value")
    if not isinstance(decoded, list):
        raise ValueError("serialized record must be a list")
    return cast(list[object], decoded)


def _typed_row(row: list[object], arity: int, kind: str) -> list[object]:
    if len(row) != arity:
        raise ValueError(f"{kind} record must contain {arity} fields")
    return row


def _string_field(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _bytes_field(value: object, name: str, width: int | None = None) -> bytes:
    if not isinstance(value, bytes):
        raise ValueError(f"{name} must be bytes")
    if width is not None and len(value) != width:
        raise ValueError(f"{name} must contain exactly {width} bytes")
    return value


def _decode_base(row: list[object]) -> BaseRecord:
    handle, payload, reads, created_at = _typed_row(row, 4, "base")
    return BaseRecord(
        _string_field(handle, "base handle"),
        _bytes_field(payload, "base payload"),
        _UINT64.unpack(_bytes_field(reads, "base reads", _UINT64.size))[0],
        _UINT64.unpack(
            _bytes_field(created_at, "base created_at", _UINT64.size)
        )[0],
    )


def _decode_packet(row: list[object]) -> Packet:
    packet_id, payload = _typed_row(row, 2, "packet")
    return Packet(
        _string_field(packet_id, "packet id"),
        _bytes_field(payload, "packet payload"),
    )


def _decode_incidence(row: list[object]) -> Incidence:
    handle, packet_id, gain_q = _typed_row(row, 3, "incidence")
    return Incidence(
        _string_field(handle, "incidence handle"),
        _string_field(packet_id, "incidence packet id"),
        _INT16.unpack(_bytes_field(gain_q, "incidence gain_q", _INT16.size))[0],
    )


def decode_state(payload: bytes) -> MemoryState:
    if len(payload) < _HEADER.size:
        raise ValueError("truncated memory-state header")
    magic, version, base_count, packet_count, incidence_count = _HEADER.unpack_from(
        payload
    )
    if magic != _MAGIC or version != _VERSION:
        raise ValueError("unsupported memory-state version")
    offset = _HEADER.size

    def take_row() -> list[object]:
        nonlocal offset
        if offset + _LENGTH.size > len(payload):
            raise ValueError("truncated record length")
        (size,) = _LENGTH.unpack_from(payload, offset)
        offset += _LENGTH.size
        end = offset + size
        if end > len(payload):
            raise ValueError("truncated record payload")
        row = _decode_canonical_row(payload[offset:end])
        offset = end
        return row

    bases: dict[str, BaseRecord] = {}
    for _ in range(base_count):
        record = _decode_base(take_row())
        if record.handle in bases:
            raise ValueError("duplicate serialized state key")
        bases[record.handle] = record

    packets: dict[str, Packet] = {}
    for _ in range(packet_count):
        packet = _decode_packet(take_row())
        if packet.packet_id in packets:
            raise ValueError("duplicate serialized state key")
        if hashlib.sha256(packet.payload).hexdigest() != packet.packet_id:
            raise ValueError("packet hash mismatch")
        packets[packet.packet_id] = packet

    incidences: dict[tuple[str, str], Incidence] = {}
    for _ in range(incidence_count):
        edge = _decode_incidence(take_row())
        key = (edge.handle, edge.packet_id)
        if key in incidences:
            raise ValueError("duplicate serialized state key")
        incidences[key] = edge

    if offset != len(payload):
        raise ValueError("trailing bytes after memory state")
    for edge in incidences.values():
        if edge.handle not in bases or edge.packet_id not in packets:
            raise ValueError("dangling packet incidence")
    return MemoryState(bases=bases, packets=packets, incidences=incidences)
