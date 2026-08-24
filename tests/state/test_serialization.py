import hashlib
import struct

import cbor2
import pytest

from ratemem.state.model import BaseRecord, Incidence, MemoryState, Packet
from ratemem.state.serialization import (
    bundle_cost_bytes,
    decode_state,
    encode_state,
    packet_from_payload,
)

_HEADER = struct.Struct("<8sIQQQ")
_LENGTH = struct.Struct("<I")
_UINT64 = struct.Struct("<Q")
_INT16 = struct.Struct("<h")


def _row(value: object) -> bytes:
    return cbor2.dumps(value, canonical=True)


def _artifact(
    *,
    bases: tuple[bytes, ...] = (),
    packets: tuple[bytes, ...] = (),
    incidences: tuple[bytes, ...] = (),
) -> bytes:
    output = bytearray(
        _HEADER.pack(b"RTMEM001", 1, len(bases), len(packets), len(incidences))
    )
    for record in (*bases, *packets, *incidences):
        output.extend(_LENGTH.pack(len(record)))
        output.extend(record)
    return bytes(output)


def _valid_base_row() -> list[object]:
    return ["concept-a", b"base", _UINT64.pack(0), _UINT64.pack(1)]


def _valid_packet_row() -> list[object]:
    payload = b"packet"
    return [hashlib.sha256(payload).hexdigest(), payload]


def _valid_incidence_row() -> list[object]:
    return ["concept-a", _valid_packet_row()[0], _INT16.pack(1)]


def test_packet_hash_state_bytes_and_bundle_delta_are_exact() -> None:
    packet = packet_from_payload(b"enhancement")
    assert packet.packet_id == packet_from_payload(b"enhancement").packet_id
    base = BaseRecord("concept-a", b"base", reads=2, created_at=1)
    incidence = Incidence("concept-a", packet.packet_id, gain_q=7)
    empty_packets = MemoryState(bases={"concept-a": base})
    state = MemoryState(
        bases={"concept-a": base},
        packets={packet.packet_id: packet},
        incidences={("concept-a", packet.packet_id): incidence},
    )
    encoded = encode_state(state)
    assert encode_state(decode_state(encoded)) == encoded
    assert state.serialized_bytes == len(encoded)
    assert len(encoded) - len(encode_state(empty_packets)) == bundle_cost_bytes(
        packet, (incidence,)
    )


def test_state_owns_immutable_mapping_copies() -> None:
    source = {"concept-a": BaseRecord("concept-a", b"base", reads=0, created_at=1)}
    state = MemoryState(bases=source)
    source.clear()
    assert tuple(state.bases) == ("concept-a",)
    with pytest.raises(TypeError):
        state.bases["concept-b"] = BaseRecord(  # type: ignore[index]
            "concept-b", b"base", 0, 2
        )


def test_packet_payload_and_references_are_checked_on_decode() -> None:
    packet = Packet(packet_id="0" * 64, payload=b"wrong")
    state = MemoryState(bases={}, packets={packet.packet_id: packet}, incidences={})
    with pytest.raises(ValueError, match="packet hash mismatch"):
        decode_state(encode_state(state))

    valid = packet_from_payload(b"valid")
    dangling = MemoryState(
        packets={valid.packet_id: valid},
        incidences={
            ("missing", valid.packet_id): Incidence(
                "missing", valid.packet_id, gain_q=1
            )
        },
    )
    with pytest.raises(ValueError, match="dangling packet incidence"):
        decode_state(encode_state(dangling))


@pytest.mark.parametrize(
    "record",
    [
        _row(_valid_base_row()) + _row(0),
        b"\x9f" + b"".join(_row(item) for item in _valid_base_row()) + b"\xff",
    ],
)
def test_decode_rejects_nonexclusive_or_noncanonical_cbor_frame(record: bytes) -> None:
    with pytest.raises(ValueError):
        decode_state(_artifact(bases=(record,)))


@pytest.mark.parametrize(
    "payload",
    [
        _artifact(bases=(_row([*_valid_base_row(), "extra"]),)),
        _artifact(bases=(_row(_valid_base_row()[:-1]),)),
        _artifact(packets=(_row([*_valid_packet_row(), "extra"]),)),
        _artifact(packets=(_row(_valid_packet_row()[:-1]),)),
        _artifact(
            bases=(_row(_valid_base_row()),),
            packets=(_row(_valid_packet_row()),),
            incidences=(_row([*_valid_incidence_row(), "extra"]),),
        ),
        _artifact(
            bases=(_row(_valid_base_row()),),
            packets=(_row(_valid_packet_row()),),
            incidences=(_row(_valid_incidence_row()[:-1]),),
        ),
    ],
)
def test_decode_rejects_rows_with_wrong_arity(payload: bytes) -> None:
    with pytest.raises(ValueError):
        decode_state(payload)


@pytest.mark.parametrize(
    "payload",
    [
        _artifact(bases=(_row([b"concept-a", b"base", b"\0" * 8, b"\0" * 8]),)),
        _artifact(bases=(_row(["concept-a", "base", b"\0" * 8, b"\0" * 8]),)),
        _artifact(bases=(_row(["concept-a", b"base", 0, b"\0" * 8]),)),
        _artifact(packets=(_row([_valid_packet_row()[0], "packet"]),)),
        _artifact(
            bases=(_row(_valid_base_row()),),
            packets=(_row(_valid_packet_row()),),
            incidences=(_row(["concept-a", _valid_packet_row()[0], 1]),),
        ),
    ],
)
def test_decode_rejects_rows_with_wrong_field_types(payload: bytes) -> None:
    with pytest.raises(ValueError):
        decode_state(payload)


@pytest.mark.parametrize(
    "payload",
    [
        _artifact(bases=(_row(["concept-a", b"base", b"\0" * 7, b"\0" * 8]),)),
        _artifact(bases=(_row(["concept-a", b"base", b"\0" * 8, b"\0" * 9]),)),
        _artifact(
            bases=(_row(_valid_base_row()),),
            packets=(_row(_valid_packet_row()),),
            incidences=(_row(["concept-a", _valid_packet_row()[0], b"\0"]),),
        ),
        _artifact(
            bases=(_row(_valid_base_row()),),
            packets=(_row(_valid_packet_row()),),
            incidences=(_row(["concept-a", _valid_packet_row()[0], b"\0" * 3]),),
        ),
    ],
)
def test_decode_rejects_wrong_fixed_width_fields(payload: bytes) -> None:
    with pytest.raises(ValueError):
        decode_state(payload)


@pytest.mark.parametrize("record", [b"\x84", b"\xff", b"\x81\xff"])
def test_decode_normalizes_malformed_cbor_to_value_error(record: bytes) -> None:
    with pytest.raises(ValueError):
        decode_state(_artifact(bases=(record,)))


def test_state_rejects_base_mapping_key_mismatch() -> None:
    with pytest.raises(ValueError, match="base mapping key mismatch"):
        MemoryState(
            bases={"wrong": BaseRecord("concept-a", b"base", reads=0, created_at=1)}
        )


def test_state_rejects_packet_mapping_key_mismatch() -> None:
    packet = packet_from_payload(b"packet")
    with pytest.raises(ValueError, match="packet mapping key mismatch"):
        MemoryState(packets={"wrong": packet})


def test_state_rejects_incidence_mapping_key_mismatch() -> None:
    packet = packet_from_payload(b"packet")
    edge = Incidence("concept-a", packet.packet_id, gain_q=1)
    with pytest.raises(ValueError, match="incidence mapping key mismatch"):
        MemoryState(incidences={("wrong", packet.packet_id): edge})


def test_state_rejects_duplicate_embedded_base_identity() -> None:
    record = BaseRecord("concept-a", b"base", reads=0, created_at=1)
    with pytest.raises(ValueError, match="duplicate embedded base identity"):
        MemoryState(bases={"first": record, "second": record})


def test_state_rejects_duplicate_embedded_packet_identity() -> None:
    packet = packet_from_payload(b"packet")
    with pytest.raises(ValueError, match="duplicate embedded packet identity"):
        MemoryState(packets={"first": packet, "second": packet})


def test_state_rejects_duplicate_embedded_incidence_identity() -> None:
    packet = packet_from_payload(b"packet")
    edge = Incidence("concept-a", packet.packet_id, gain_q=1)
    with pytest.raises(ValueError, match="duplicate embedded incidence identity"):
        MemoryState(
            incidences={
                ("first", packet.packet_id): edge,
                ("second", packet.packet_id): edge,
            }
        )


def test_records_and_state_own_immutable_payload_bytes() -> None:
    base_source = bytearray(b"base")
    packet_source = bytearray(b"packet")
    packet_id = hashlib.sha256(packet_source).hexdigest()
    base = BaseRecord(  # type: ignore[arg-type]
        "concept-a", base_source, reads=0, created_at=1
    )
    packet = Packet(packet_id, memoryview(packet_source))  # type: ignore[arg-type]
    state = MemoryState(
        bases={base.handle: base},
        packets={packet.packet_id: packet},
    )

    base_source[:] = b"edit"
    packet_source[:] = b"mutate"

    assert type(base.payload) is bytes
    assert base.payload == b"base"
    assert type(packet.payload) is bytes
    assert packet.payload == b"packet"
    assert encode_state(decode_state(encode_state(state))) == encode_state(state)


def test_packet_from_payload_hashes_and_owns_the_same_bytes() -> None:
    source = bytearray(b"packet")
    packet = packet_from_payload(source)
    source[:] = b"mutate"

    assert type(packet.payload) is bytes
    assert packet.payload == b"packet"
    assert packet.packet_id == hashlib.sha256(packet.payload).hexdigest()
