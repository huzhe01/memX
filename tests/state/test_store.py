import hashlib
from types import MappingProxyType

import pytest

from ratemem.state.model import BaseRecord, Incidence, MemoryState, Packet
from ratemem.state.serialization import decode_state, encode_state, packet_from_payload
from ratemem.state.store import BudgetExceeded, PacketStore


class _IntSubclass(int):
    pass


class _StrSubclass(str):
    pass


class _SpoofedIntType:
    @property
    def __class__(self) -> type[int]:
        return int

    def __lt__(self, other: object) -> bool:
        return False

    def __rlt__(self, other: object) -> bool:
        return False


class _MemoryStateSubclass(MemoryState):
    pass


class _BaseRecordSubclass(BaseRecord):
    pass


class _PacketSubclass(Packet):
    pass


class _IncidenceSubclass(Incidence):
    pass


class _BytesSubclass(bytes):
    pass


class _TupleSubclass(tuple[str, str]):
    pass


class _MappingSubclass(dict[str, BaseRecord]):
    pass


def _raw_state(
    *,
    bases: object = MappingProxyType({}),
    packets: object = MappingProxyType({}),
    incidences: object = MappingProxyType({}),
) -> MemoryState:
    state = object.__new__(MemoryState)
    object.__setattr__(state, "bases", bases)
    object.__setattr__(state, "packets", packets)
    object.__setattr__(state, "incidences", incidences)
    return state


def _raw_base(
    *,
    handle: object = "a",
    payload: object = b"base",
    reads: object = 0,
    created_at: object = 1,
    record_type: type[BaseRecord] = BaseRecord,
) -> BaseRecord:
    record = object.__new__(record_type)
    object.__setattr__(record, "handle", handle)
    object.__setattr__(record, "payload", payload)
    object.__setattr__(record, "reads", reads)
    object.__setattr__(record, "created_at", created_at)
    return record


def _raw_packet(
    *,
    packet_id: object,
    payload: object,
    record_type: type[Packet] = Packet,
) -> Packet:
    packet = object.__new__(record_type)
    object.__setattr__(packet, "packet_id", packet_id)
    object.__setattr__(packet, "payload", payload)
    return packet


def _raw_incidence(
    *,
    handle: object = "a",
    packet_id: object,
    gain_q: object = 1,
    record_type: type[Incidence] = Incidence,
) -> Incidence:
    incidence = object.__new__(record_type)
    object.__setattr__(incidence, "handle", handle)
    object.__setattr__(incidence, "packet_id", packet_id)
    object.__setattr__(incidence, "gain_q", gain_q)
    return incidence


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


def test_create_rejects_non_string_handle_before_producing_noncanonical_state() -> None:
    store = PacketStore.empty(256)

    with pytest.raises(TypeError, match="handle must be a nonempty string"):
        store.create(1, b"x", created_at=0)  # type: ignore[arg-type]

    assert decode_state(encode_state(store.state)) == store.state


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


@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param(1, id="int"),
        pytest.param(True, id="bool"),
        pytest.param(_IntSubclass(1), id="int-subclass"),
        pytest.param(_StrSubclass("a"), id="str-subclass"),
    ],
)
@pytest.mark.parametrize("operation", ["create", "replace", "read", "delete"])
def test_store_identity_entrypoints_reject_nonexact_handles(
    operation: str, invalid: object
) -> None:
    store = PacketStore.empty(4096).create("a", b"base", created_at=0)

    with pytest.raises(TypeError, match="handle must be a nonempty string"):
        if operation == "create":
            store.create(invalid, b"other", created_at=1)  # type: ignore[arg-type]
        elif operation == "replace":
            store.replace(invalid, b"other", attachments=())  # type: ignore[arg-type]
        elif operation == "read":
            store.read(invalid, update_usage=False)  # type: ignore[arg-type]
        else:
            store.delete(invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param(True, id="bool"),
        pytest.param(_IntSubclass(1), id="int-subclass"),
    ],
)
def test_store_create_rejects_nonexact_created_at(invalid: object) -> None:
    with pytest.raises(TypeError, match="created_at must be an integer"):
        PacketStore.empty(4096).create(
            "a", b"base", created_at=invalid  # type: ignore[arg-type]
        )


def test_every_accepted_store_transition_roundtrips_canonically() -> None:
    def assert_roundtrip(store: PacketStore) -> None:
        payload = encode_state(store.state)
        decoded = decode_state(payload)
        assert decoded == store.state
        assert encode_state(decoded) == payload

    packet = packet_from_payload(b"shared")
    stores = [PacketStore.empty(4096)]
    stores.append(stores[-1].create("a", b"base-a", created_at=0))
    stores.append(stores[-1].create("b", b"base-b", created_at=1))
    stores.append(
        stores[-1].attach_bundle(
            packet,
            (
                Incidence("a", packet.packet_id, gain_q=1),
                Incidence("b", packet.packet_id, gain_q=2),
            ),
        )
    )
    stores.append(stores[-1].read("a", update_usage=True)[0])
    stores.append(stores[-1].replace("a", b"new-a", attachments=()))
    stores.append(stores[-1].delete("b"))

    for store in stores:
        assert_roundtrip(store)


def test_store_rejects_memory_state_subclasses() -> None:
    with pytest.raises(TypeError, match="state must be an exact MemoryState"):
        PacketStore(state=_MemoryStateSubclass(), budget_bytes=4096)


@pytest.mark.parametrize("subclassed_value", ["packet", "incidence"])
def test_attach_rejects_record_subclasses(subclassed_value: str) -> None:
    raw_packet = packet_from_payload(b"packet")
    packet = (
        _PacketSubclass(raw_packet.packet_id, raw_packet.payload)
        if subclassed_value == "packet"
        else raw_packet
    )
    incidence = (
        _IncidenceSubclass("a", packet.packet_id, gain_q=1)
        if subclassed_value == "incidence"
        else Incidence("a", packet.packet_id, gain_q=1)
    )
    store = PacketStore.empty(4096).create("a", b"base", created_at=0)

    with pytest.raises(TypeError, match="must be an exact"):
        store.attach(packet, incidence)


def test_store_rejects_low_level_forged_embedded_key_mismatch() -> None:
    base = BaseRecord("actual-handle", b"base", reads=0, created_at=1)
    state = _raw_state(
        bases=MappingProxyType({"wrong-key": base}),
    )

    with pytest.raises(ValueError, match="base mapping key mismatch"):
        PacketStore(state=state, budget_bytes=4096)


def test_store_rejects_low_level_forged_mapping_container() -> None:
    base = BaseRecord("a", b"base", reads=0, created_at=1)
    state = _raw_state(bases=_MappingSubclass({"a": base}))

    with pytest.raises(TypeError, match="bases must be an owned immutable mapping"):
        PacketStore(state=state, budget_bytes=4096)


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        pytest.param(
            "handle",
            _StrSubclass("a"),
            "handle must be a nonempty string",
            id="str-subclass-handle",
        ),
        pytest.param(
            "reads", True, "reads must be an integer", id="bool-reads"
        ),
        pytest.param(
            "reads",
            _IntSubclass(0),
            "reads must be an integer",
            id="int-subclass-reads",
        ),
        pytest.param(
            "created_at",
            True,
            "created_at must be an integer",
            id="bool-created-at",
        ),
        pytest.param(
            "created_at",
            _IntSubclass(1),
            "created_at must be an integer",
            id="int-subclass-created-at",
        ),
        pytest.param(
            "payload",
            bytearray(b"base"),
            "base payload must be exact bytes",
            id="mutable-payload",
        ),
        pytest.param(
            "payload",
            "base",
            "base payload must be exact bytes",
            id="nonbytes-payload",
        ),
        pytest.param(
            "payload",
            _BytesSubclass(b"base"),
            "base payload must be exact bytes",
            id="bytes-subclass-payload",
        ),
    ],
)
def test_store_revalidates_low_level_forged_base_fields(
    field: str, invalid: object, message: str
) -> None:
    arguments: dict[str, object] = {
        "handle": "a",
        "payload": b"base",
        "reads": 0,
        "created_at": 1,
    }
    arguments[field] = invalid
    base = _raw_base(**arguments)
    state = _raw_state(bases=MappingProxyType({"a": base}))

    with pytest.raises(TypeError, match=message):
        PacketStore(state=state, budget_bytes=4096)


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        pytest.param("reads", -1, "reads must fit uint64", id="negative-reads"),
        pytest.param(
            "created_at",
            0x10000000000000000,
            "created_at must fit uint64",
            id="created-at-overflow",
        ),
    ],
)
def test_store_revalidates_low_level_forged_base_counter_ranges(
    field: str, invalid: int, message: str
) -> None:
    arguments: dict[str, object] = {
        "handle": "a",
        "payload": b"base",
        "reads": 0,
        "created_at": 1,
    }
    arguments[field] = invalid
    state = _raw_state(
        bases=MappingProxyType({"a": _raw_base(**arguments)})
    )

    with pytest.raises(ValueError, match=message):
        PacketStore(state=state, budget_bytes=4096)


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        pytest.param(
            "packet_id",
            _StrSubclass("packet-id"),
            "packet_id must be a nonempty string",
            id="str-subclass-id",
        ),
        pytest.param(
            "packet_id",
            1,
            "packet_id must be a nonempty string",
            id="int-id",
        ),
        pytest.param(
            "payload",
            bytearray(b"packet"),
            "packet payload must be exact bytes",
            id="mutable-payload",
        ),
        pytest.param(
            "payload",
            "packet",
            "packet payload must be exact bytes",
            id="nonbytes-payload",
        ),
    ],
)
def test_store_revalidates_low_level_forged_packet_fields(
    field: str, invalid: object, message: str
) -> None:
    payload = b"packet"
    packet_id = hashlib.sha256(payload).hexdigest()
    arguments: dict[str, object] = {
        "packet_id": packet_id,
        "payload": payload,
    }
    arguments[field] = invalid
    packet = _raw_packet(**arguments)
    base = BaseRecord("a", b"base", reads=0, created_at=1)
    incidence = Incidence("a", packet_id, gain_q=1)
    state = _raw_state(
        bases=MappingProxyType({"a": base}),
        packets=MappingProxyType({packet_id: packet}),
        incidences=MappingProxyType({("a", packet_id): incidence}),
    )

    with pytest.raises(TypeError, match=message):
        PacketStore(state=state, budget_bytes=4096)


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        pytest.param(
            "handle",
            _StrSubclass("a"),
            "handle must be a nonempty string",
            id="str-subclass-handle",
        ),
        pytest.param(
            "packet_id",
            _StrSubclass("packet-id"),
            "packet_id must be a nonempty string",
            id="str-subclass-packet-id",
        ),
        pytest.param(
            "gain_q", True, "gain_q must be an integer", id="bool-gain"
        ),
        pytest.param(
            "gain_q",
            _IntSubclass(1),
            "gain_q must be an integer",
            id="int-subclass-gain",
        ),
    ],
)
def test_store_revalidates_low_level_forged_incidence_fields(
    field: str, invalid: object, message: str
) -> None:
    packet = packet_from_payload(b"packet")
    arguments: dict[str, object] = {
        "handle": "a",
        "packet_id": packet.packet_id,
        "gain_q": 1,
    }
    arguments[field] = invalid
    incidence = _raw_incidence(**arguments)
    base = BaseRecord("a", b"base", reads=0, created_at=1)
    state = _raw_state(
        bases=MappingProxyType({"a": base}),
        packets=MappingProxyType({packet.packet_id: packet}),
        incidences=MappingProxyType({("a", packet.packet_id): incidence}),
    )

    with pytest.raises(TypeError, match=message):
        PacketStore(state=state, budget_bytes=4096)


@pytest.mark.parametrize("gain_q", [-0x8001, 0x8000])
def test_store_revalidates_low_level_forged_incidence_gain_range(
    gain_q: int,
) -> None:
    packet = packet_from_payload(b"packet")
    incidence = _raw_incidence(packet_id=packet.packet_id, gain_q=gain_q)
    state = _raw_state(
        bases=MappingProxyType(
            {"a": BaseRecord("a", b"base", reads=0, created_at=1)}
        ),
        packets=MappingProxyType({packet.packet_id: packet}),
        incidences=MappingProxyType({("a", packet.packet_id): incidence}),
    )

    with pytest.raises(ValueError, match="gain_q must fit int16"):
        PacketStore(state=state, budget_bytes=4096)


@pytest.mark.parametrize(
    ("mapping_name", "state"),
    [
        pytest.param(
            "bases",
            _raw_state(
                bases=MappingProxyType(
                    {
                        "a": _raw_base(
                            record_type=_BaseRecordSubclass,
                        )
                    }
                )
            ),
            id="base-record-subclass",
        ),
        pytest.param(
            "packets",
            _raw_state(
                packets=MappingProxyType(
                    {
                        hashlib.sha256(b"packet").hexdigest(): _raw_packet(
                            packet_id=hashlib.sha256(b"packet").hexdigest(),
                            payload=b"packet",
                            record_type=_PacketSubclass,
                        )
                    }
                )
            ),
            id="packet-record-subclass",
        ),
        pytest.param(
            "incidences",
            _raw_state(
                incidences=MappingProxyType(
                    {
                        ("a", "packet"): _raw_incidence(
                            packet_id="packet",
                            record_type=_IncidenceSubclass,
                        )
                    }
                )
            ),
            id="incidence-record-subclass",
        ),
    ],
)
def test_store_rejects_low_level_forged_record_subclasses(
    mapping_name: str, state: MemoryState
) -> None:
    with pytest.raises(TypeError, match=f"{mapping_name} values must be exact"):
        PacketStore(state=state, budget_bytes=4096)


def test_store_rejects_low_level_forged_incidence_key_subclass() -> None:
    state = _raw_state(
        incidences=MappingProxyType(
            {
                _TupleSubclass(("a", "packet")): Incidence(
                    "a", "packet", gain_q=1
                )
            }
        )
    )

    with pytest.raises(TypeError, match="incidence mapping key"):
        PacketStore(state=state, budget_bytes=4096)


def test_store_rejects_low_level_duplicate_embedded_base_identity() -> None:
    state = _raw_state(
        bases=MappingProxyType(
            {
                "a": BaseRecord("a", b"a", reads=0, created_at=1),
                "b": _raw_base(handle="a", payload=b"b"),
            }
        )
    )

    with pytest.raises(ValueError, match="duplicate embedded base identity"):
        PacketStore(state=state, budget_bytes=4096)


def test_store_preserves_valid_exact_state_identity_after_revalidation() -> None:
    state = MemoryState(
        bases={"a": BaseRecord("a", b"base", reads=0, created_at=1)}
    )

    store = PacketStore(state=state, budget_bytes=4096)

    assert store.state is state
