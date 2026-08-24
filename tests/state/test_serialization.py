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


class _IntSubclass(int):
    pass


class _StrSubclass(str):
    pass


class _TupleSubclass(tuple[str, str]):
    pass


class _BaseRecordSubclass(BaseRecord):
    pass


class _PacketSubclass(Packet):
    pass


class _IncidenceSubclass(Incidence):
    pass


class _UnsafeBaseRecord(BaseRecord):
    def __post_init__(self) -> None:
        pass


class _UnsafePacket(Packet):
    def __post_init__(self) -> None:
        pass


class _UnsafeIncidence(Incidence):
    def __post_init__(self) -> None:
        pass


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


def _noncanonical_section_artifacts() -> tuple[bytes, bytes, bytes]:
    bases = (
        _row(["concept-b", b"base-b", _UINT64.pack(0), _UINT64.pack(2)]),
        _row(["concept-a", b"base-a", _UINT64.pack(0), _UINT64.pack(1)]),
    )

    packet_rows = [
        [hashlib.sha256(payload).hexdigest(), payload]
        for payload in (b"packet-a", b"packet-b")
    ]
    packet_rows.sort(key=lambda row: row[0])
    packets = tuple(_row(row) for row in reversed(packet_rows))

    canonical_packets = tuple(_row(row) for row in packet_rows)
    incidences = tuple(
        _row(["concept-a", row[0], _INT16.pack(index + 1)])
        for index, row in enumerate(reversed(packet_rows))
    )
    incidence_artifact = _artifact(
        bases=(_row(_valid_base_row()),),
        packets=canonical_packets,
        incidences=incidences,
    )
    return _artifact(bases=bases), _artifact(packets=packets), incidence_artifact


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
    _noncanonical_section_artifacts(),
    ids=("bases", "packets", "incidences"),
)
def test_decode_rejects_noncanonical_within_section_ordering(
    payload: bytes,
) -> None:
    with pytest.raises(ValueError, match="serialized state is not canonical"):
        decode_state(payload)


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


@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param(1, id="int"),
        pytest.param(True, id="bool"),
        pytest.param(_StrSubclass("concept-a"), id="str-subclass"),
    ],
)
def test_base_record_requires_an_exact_string_handle(invalid: object) -> None:
    with pytest.raises(TypeError, match="handle must be a nonempty string"):
        BaseRecord(invalid, b"base", reads=0, created_at=1)  # type: ignore[arg-type]


def test_base_record_rejects_an_empty_handle() -> None:
    with pytest.raises(ValueError, match="handle must be a nonempty string"):
        BaseRecord("", b"base", reads=0, created_at=1)


@pytest.mark.parametrize("field", ["reads", "created_at"])
@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param(True, id="bool"),
        pytest.param(_IntSubclass(1), id="int-subclass"),
        pytest.param(1.0, id="float"),
    ],
)
def test_base_record_counters_require_exact_integers(
    field: str, invalid: object
) -> None:
    arguments: dict[str, object] = {
        "handle": "concept-a",
        "payload": b"base",
        "reads": 0,
        "created_at": 1,
    }
    arguments[field] = invalid

    with pytest.raises(TypeError, match=f"{field} must be an integer"):
        BaseRecord(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param("", id="empty"),
        pytest.param(1, id="int"),
        pytest.param(True, id="bool"),
        pytest.param(_StrSubclass("packet"), id="str-subclass"),
    ],
)
def test_packet_requires_an_exact_nonempty_string_id(invalid: object) -> None:
    error_type = ValueError if invalid == "" else TypeError
    with pytest.raises(error_type, match="packet_id must be a nonempty string"):
        Packet(invalid, b"packet")  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["handle", "packet_id"])
@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param("", id="empty"),
        pytest.param(1, id="int"),
        pytest.param(True, id="bool"),
        pytest.param(_StrSubclass("identity"), id="str-subclass"),
    ],
)
def test_incidence_requires_exact_nonempty_string_identities(
    field: str, invalid: object
) -> None:
    arguments: dict[str, object] = {
        "handle": "concept-a",
        "packet_id": "packet",
        "gain_q": 1,
    }
    arguments[field] = invalid
    error_type = ValueError if invalid == "" else TypeError

    with pytest.raises(error_type, match=f"{field} must be a nonempty string"):
        Incidence(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param(True, id="bool"),
        pytest.param(_IntSubclass(1), id="int-subclass"),
        pytest.param(1.0, id="float"),
    ],
)
def test_incidence_gain_requires_an_exact_integer(invalid: object) -> None:
    with pytest.raises(TypeError, match="gain_q must be an integer"):
        Incidence("concept-a", "packet", invalid)  # type: ignore[arg-type]


def test_state_rejects_nonexact_base_mapping_key() -> None:
    base = BaseRecord("concept-a", b"base", reads=0, created_at=1)

    with pytest.raises(TypeError, match="base mapping key"):
        MemoryState(bases={_StrSubclass(base.handle): base})


def test_state_rejects_nonexact_packet_mapping_key() -> None:
    packet = packet_from_payload(b"packet")

    with pytest.raises(TypeError, match="packet mapping key"):
        MemoryState(packets={_StrSubclass(packet.packet_id): packet})


@pytest.mark.parametrize(
    "key",
    [
        pytest.param(
            _TupleSubclass(("concept-a", "packet")), id="tuple-subclass"
        ),
        pytest.param(
            (_StrSubclass("concept-a"), "packet"), id="str-subclass-member"
        ),
        pytest.param(("concept-a", 1), id="non-string-member"),
    ],
)
def test_state_rejects_nonexact_incidence_mapping_key(
    key: tuple[str, str],
) -> None:
    edge = Incidence("concept-a", "packet", gain_q=1)

    with pytest.raises(TypeError, match="incidence mapping key"):
        MemoryState(incidences={key: edge})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param(
            "bases",
            _BaseRecordSubclass("concept-a", b"base", reads=0, created_at=1),
            id="validated-base-subclass",
        ),
        pytest.param(
            "packets",
            _PacketSubclass("packet", b"payload"),
            id="validated-packet-subclass",
        ),
        pytest.param(
            "incidences",
            _IncidenceSubclass("concept-a", "packet", gain_q=1),
            id="validated-incidence-subclass",
        ),
        pytest.param(
            "bases",
            _UnsafeBaseRecord("concept-a", b"base", reads=True, created_at=1),
            id="unsafe-base-subclass",
        ),
        pytest.param(
            "packets",
            _UnsafePacket(_StrSubclass("packet"), b"payload"),
            id="unsafe-packet-subclass",
        ),
        pytest.param(
            "incidences",
            _UnsafeIncidence("concept-a", "packet", gain_q=True),
            id="unsafe-incidence-subclass",
        ),
    ],
)
def test_state_rejects_record_subclasses(field: str, value: object) -> None:
    if field == "bases":
        arguments = {"bases": {"concept-a": value}}
    elif field == "packets":
        arguments = {"packets": {"packet": value}}
    else:
        arguments = {"incidences": {("concept-a", "packet"): value}}

    with pytest.raises(TypeError, match=f"{field} values must be exact"):
        MemoryState(**arguments)  # type: ignore[arg-type]
