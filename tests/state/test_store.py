import pytest

from ratemem.state.model import Incidence, Packet
from ratemem.state.serialization import packet_from_payload
from ratemem.state.store import BudgetExceeded, PacketStore


def test_delete_reclaims_only_unreferenced_packets() -> None:
    store = PacketStore.empty(budget_bytes=4096)
    packet = packet_from_payload(b"shared")
    store = store.create("a", b"base-a", created_at=1)
    store = store.create("b", b"base-b", created_at=2)
    store = store.attach_bundle(
        packet,
        (
            Incidence("a", packet.packet_id, 4),
            Incidence("b", packet.packet_id, 5),
        ),
    )

    after_a = store.delete("a")
    assert packet.packet_id in after_a.state.packets
    after_b = after_a.delete("b")
    assert packet.packet_id not in after_b.state.packets
    assert after_b.state.incidences == {}


def test_failed_transaction_does_not_mutate_old_state() -> None:
    store = PacketStore.empty(budget_bytes=512).create("a", b"small", created_at=1)
    packet = packet_from_payload(b"x" * 512)
    with pytest.raises(BudgetExceeded):
        store.attach(packet, Incidence("a", packet.packet_id, 1))
    assert packet.packet_id not in store.state.packets


def test_replace_redirects_one_concept_atomically_and_preserves_shared_packet() -> None:
    shared = packet_from_payload(b"shared")
    private = packet_from_payload(b"private-a")
    store = PacketStore.empty(budget_bytes=4096)
    store = store.create("a", b"old-a", created_at=1).create("b", b"base-b", created_at=2)
    store = store.attach(shared, Incidence("a", shared.packet_id, 2))
    store = store.attach(shared, Incidence("b", shared.packet_id, 3))

    updated = store.replace(
        "a", b"new-a", ((private, Incidence("a", private.packet_id, 4)),)
    )
    assert updated.state.bases["a"].payload == b"new-a"
    assert ("a", shared.packet_id) not in updated.state.incidences
    assert ("b", shared.packet_id) in updated.state.incidences
    assert shared.packet_id in updated.state.packets
    assert private.packet_id in updated.state.packets
    assert store.state.bases["a"].payload == b"old-a"


def test_attach_rejects_forged_content_address() -> None:
    store = PacketStore.empty(budget_bytes=2048).create("a", b"base", created_at=1)
    forged = Packet("0" * 64, b"payload")
    with pytest.raises(ValueError, match="packet hash mismatch"):
        store.attach(forged, Incidence("a", forged.packet_id, 1))
