import pytest

from ratemem.state.model import BaseRecord, Incidence, MemoryState, Packet
from ratemem.state.serialization import (
    bundle_cost_bytes,
    decode_state,
    encode_state,
    packet_from_payload,
)


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
