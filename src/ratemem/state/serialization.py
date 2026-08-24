from __future__ import annotations

import hashlib
import struct
from typing import Any, cast

import cbor2

from ratemem.state.model import BaseRecord, Incidence, MemoryState, Packet

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


def packet_from_payload(payload: bytes) -> Packet:
    packet_id = hashlib.sha256(payload).hexdigest()
    return Packet(packet_id=packet_id, payload=payload)


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


def decode_state(payload: bytes) -> MemoryState:
    if len(payload) < _HEADER.size:
        raise ValueError("truncated memory-state header")
    magic, version, base_count, packet_count, incidence_count = _HEADER.unpack_from(
        payload
    )
    if magic != _MAGIC or version != _VERSION:
        raise ValueError("unsupported memory-state version")
    offset = _HEADER.size

    def take_row() -> list[Any]:
        nonlocal offset
        if offset + _LENGTH.size > len(payload):
            raise ValueError("truncated record length")
        (size,) = _LENGTH.unpack_from(payload, offset)
        offset += _LENGTH.size
        end = offset + size
        if end > len(payload):
            raise ValueError("truncated record payload")
        row = cast(list[Any], cbor2.loads(payload[offset:end]))
        offset = end
        return row

    base_rows = [take_row() for _ in range(base_count)]
    packet_rows = [take_row() for _ in range(packet_count)]
    incidence_rows = [take_row() for _ in range(incidence_count)]
    if offset != len(payload):
        raise ValueError("trailing bytes after memory state")
    bases = {
        row[0]: BaseRecord(
            row[0],
            row[1],
            _UINT64.unpack(row[2])[0],
            _UINT64.unpack(row[3])[0],
        )
        for row in base_rows
    }
    packets = {row[0]: Packet(row[0], row[1]) for row in packet_rows}
    incidences = {
        (row[0], row[1]): Incidence(row[0], row[1], _INT16.unpack(row[2])[0])
        for row in incidence_rows
    }
    if (
        len(bases) != base_count
        or len(packets) != packet_count
        or len(incidences) != incidence_count
    ):
        raise ValueError("duplicate serialized state key")
    for packet in packets.values():
        if hashlib.sha256(packet.payload).hexdigest() != packet.packet_id:
            raise ValueError("packet hash mismatch")
    for edge in incidences.values():
        if edge.handle not in bases or edge.packet_id not in packets:
            raise ValueError("dangling packet incidence")
    return MemoryState(bases=bases, packets=packets, incidences=incidences)
