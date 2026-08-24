import pytest

from ratemem.state.model import BaseRecord, Incidence, MemoryState, Packet
from ratemem.state.serialization import encode_state, packet_from_payload
from ratemem.state.store import BudgetExceeded, PacketStore


class _IntSubclass(int):
    pass


class _SpoofedIntType:
    @property
    def __class__(self) -> type[int]:
        return int

    def __lt__(self, other: object) -> bool:
        return False

    def __rlt__(self, other: object) -> bool:
        return False


@pytest.mark.parametrize(
    "budget_bytes",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
        pytest.param(4096.0, id="finite-float"),
        pytest.param(True, id="true"),
        pytest.param(False, id="false"),
        pytest.param("4096", id="string"),
        pytest.param(_IntSubclass(4096), id="int-subclass"),
        pytest.param(_SpoofedIntType(), id="spoofed-int-type"),
    ],
)
def test_budget_requires_exact_non_bool_int(budget_bytes: object) -> None:
    with pytest.raises(TypeError, match="budget_bytes must be an integer"):
        PacketStore.empty(budget_bytes)  # type: ignore[arg-type]


def test_constructor_rejects_forged_packet_hash() -> None:
    forged = Packet("0" * 64, b"payload")
    state = MemoryState(packets={forged.packet_id: forged})

    with pytest.raises(ValueError, match="packet hash mismatch"):
        PacketStore(state=state, budget_bytes=4096)


@pytest.mark.parametrize("missing_reference", ["base", "packet"])
def test_constructor_rejects_dangling_incidence(missing_reference: str) -> None:
    packet = packet_from_payload(b"packet")
    base = BaseRecord("a", b"base", reads=0, created_at=1)
    state = MemoryState(
        bases={} if missing_reference == "base" else {base.handle: base},
        packets={packet.packet_id: packet} if missing_reference == "base" else {},
        incidences={
            (base.handle, packet.packet_id): Incidence(
                base.handle, packet.packet_id, gain_q=1
            )
        },
    )

    with pytest.raises(ValueError, match="dangling packet incidence"):
        PacketStore(state=state, budget_bytes=4096)


def test_constructor_rejects_orphan_packet() -> None:
    packet = packet_from_payload(b"orphan")
    state = MemoryState(packets={packet.packet_id: packet})

    with pytest.raises(ValueError, match="orphan packet"):
        PacketStore(state=state, budget_bytes=4096)


@pytest.mark.parametrize("duplicate_gain", [4, 5], ids=["identical", "conflicting"])
def test_replace_rejects_duplicate_packet_attachments_atomically(
    duplicate_gain: int,
) -> None:
    packet = packet_from_payload(b"packet")
    store = PacketStore.empty(budget_bytes=4096).create(
        "a", b"original", created_at=1
    )
    original_state = store.state
    original_bytes = encode_state(store.state)

    with pytest.raises(ValueError, match="replacement repeats packet attachment"):
        store.replace(
            "a",
            b"replacement",
            (
                (packet, Incidence("a", packet.packet_id, gain_q=4)),
                (packet, Incidence("a", packet.packet_id, gain_q=duplicate_gain)),
            ),
        )

    assert store.state is original_state
    assert encode_state(store.state) == original_bytes


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


def test_attach_accepts_exact_budget_and_rejects_one_byte_under_atomically() -> None:
    packet = packet_from_payload(b"packet")
    incidence = Incidence("a", packet.packet_id, gain_q=1)
    base_store = PacketStore.empty(budget_bytes=4096).create(
        "a", b"base", created_at=1
    )
    candidate = base_store.attach(packet, incidence)

    exact = PacketStore(
        state=base_store.state, budget_bytes=candidate.state.serialized_bytes
    ).attach(packet, incidence)
    assert exact.state.serialized_bytes == exact.budget_bytes

    one_under = PacketStore(
        state=base_store.state, budget_bytes=candidate.state.serialized_bytes - 1
    )
    original_state = one_under.state
    original_bytes = encode_state(one_under.state)
    with pytest.raises(BudgetExceeded):
        one_under.attach(packet, incidence)
    assert one_under.state is original_state
    assert encode_state(one_under.state) == original_bytes


def test_create_and_read_update_usage_without_mutating_prior_store() -> None:
    store = PacketStore.empty(budget_bytes=4096).create(
        "a", b"base", created_at=7
    )

    read_only, record = store.read("a", update_usage=False)
    assert read_only is store
    assert record.reads == 0
    assert record.created_at == 7

    updated, returned = store.read("a", update_usage=True)
    assert returned is record
    assert updated.state.bases["a"].reads == 1
    assert store.state.bases["a"].reads == 0


def test_read_uint64_overflow_is_atomic() -> None:
    record = BaseRecord("a", b"base", reads=0xFFFFFFFFFFFFFFFF, created_at=1)
    state = MemoryState(bases={record.handle: record})
    store = PacketStore(state=state, budget_bytes=state.serialized_bytes)
    original_bytes = encode_state(store.state)

    with pytest.raises(ValueError, match="reads must fit uint64"):
        store.read("a", update_usage=True)

    assert store.state is state
    assert encode_state(store.state) == original_bytes


def test_duplicate_attach_is_byte_idempotent() -> None:
    packet = packet_from_payload(b"packet")
    incidence = Incidence("a", packet.packet_id, gain_q=1)
    attached = (
        PacketStore.empty(budget_bytes=4096)
        .create("a", b"base", created_at=1)
        .attach(packet, incidence)
    )

    repeated = attached.attach(packet, incidence)

    assert encode_state(repeated.state) == encode_state(attached.state)
    assert len(repeated.state.packets) == 1
    assert len(repeated.state.incidences) == 1


def test_replace_budget_failure_is_atomic() -> None:
    roomy = PacketStore.empty(budget_bytes=4096).create("a", b"base", created_at=1)
    store = PacketStore(
        state=roomy.state, budget_bytes=roomy.state.serialized_bytes
    )
    original_state = store.state
    original_bytes = encode_state(store.state)

    with pytest.raises(BudgetExceeded):
        store.replace("a", b"base!", attachments=())

    assert store.state is original_state
    assert encode_state(store.state) == original_bytes


def test_attachment_identity_mismatches_are_atomic() -> None:
    packet = packet_from_payload(b"packet")
    store = PacketStore.empty(budget_bytes=4096).create("a", b"base", created_at=1)
    original_state = store.state
    original_bytes = encode_state(store.state)

    with pytest.raises(ValueError, match="incidence handle does not match operation"):
        store.replace(
            "a",
            b"replacement",
            ((packet, Incidence("other", packet.packet_id, gain_q=1)),),
        )
    with pytest.raises(ValueError, match="incidence packet id does not match payload"):
        store.attach(packet, Incidence("a", "0" * 64, gain_q=1))

    assert store.state is original_state
    assert encode_state(store.state) == original_bytes
