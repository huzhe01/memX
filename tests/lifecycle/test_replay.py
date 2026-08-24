import pytest
from hypothesis import given
from hypothesis import strategies as st

from ratemem.lifecycle.events import (
    CreateEvent,
    DeleteEvent,
    ProbeEvent,
    ReadEvent,
    UpdateEvent,
)
from ratemem.lifecycle.replay import replay
from ratemem.state.model import MemoryState
from ratemem.state.store import BudgetExceeded, PacketStore


class _SpoofedStringType:
    @property
    def __class__(self) -> type[str]:
        return str

    def __bool__(self) -> bool:
        return True


class _HashChangingString(str):
    def __new__(cls, value: str):
        instance = super().__new__(cls, value)
        instance._runtime_hash = super(_HashChangingString, instance).__hash__()
        return instance

    def change_hash(self) -> None:
        self._runtime_hash += 1

    def __hash__(self) -> int:
        return self._runtime_hash


class _UnsafeCreateEvent(CreateEvent):
    def __post_init__(self) -> None:
        pass


class _UnsafeUpdateEvent(UpdateEvent):
    def __post_init__(self) -> None:
        pass


class _ReadEventSubclass(ReadEvent):
    pass


_EVENT_FACTORIES = (
    pytest.param(
        lambda event_id, handle: CreateEvent(event_id, handle, b"payload"),
        id="create",
    ),
    pytest.param(lambda event_id, handle: ReadEvent(event_id, handle), id="read"),
    pytest.param(
        lambda event_id, handle: UpdateEvent(event_id, handle, b"payload"),
        id="update",
    ),
    pytest.param(lambda event_id, handle: ProbeEvent(event_id, handle), id="probe"),
    pytest.param(
        lambda event_id, handle: DeleteEvent(event_id, handle), id="delete"
    ),
)


@pytest.mark.parametrize("event_type", [CreateEvent, UpdateEvent])
@pytest.mark.parametrize(
    "payload",
    [
        pytest.param("payload", id="str"),
        pytest.param(1, id="int"),
        pytest.param(None, id="none"),
    ],
)
def test_payload_events_reject_non_bytes_like_values(event_type, payload) -> None:
    with pytest.raises(TypeError, match="base_payload must be bytes-like"):
        event_type("1", "a", payload)


@pytest.mark.parametrize("event_type", [CreateEvent, UpdateEvent])
@pytest.mark.parametrize(
    "source_factory",
    [
        pytest.param(lambda: bytearray(b"old"), id="bytearray"),
        pytest.param(lambda: memoryview(bytearray(b"old")), id="memoryview"),
    ],
)
def test_payload_events_own_mutable_bytes_like_values(
    event_type, source_factory
) -> None:
    source = source_factory()
    event = event_type("1", "a", source)

    source[:] = b"new"

    assert type(event.base_payload) is bytes
    assert event.base_payload == b"old"


@pytest.mark.parametrize("factory", _EVENT_FACTORIES)
@pytest.mark.parametrize(
    ("field", "invalid", "error_type"),
    [
        pytest.param("event_id", "", ValueError, id="empty-event-id"),
        pytest.param("event_id", 1, TypeError, id="non-string-event-id"),
        pytest.param(
            "event_id",
            _SpoofedStringType(),
            TypeError,
            id="spoofed-string-event-id",
        ),
        pytest.param(
            "event_id",
            _HashChangingString("event"),
            TypeError,
            id="str-subclass-event-id",
        ),
        pytest.param("handle", "", ValueError, id="empty-handle"),
        pytest.param("handle", 1, TypeError, id="non-string-handle"),
        pytest.param(
            "handle",
            _SpoofedStringType(),
            TypeError,
            id="spoofed-string-handle",
        ),
        pytest.param(
            "handle",
            _HashChangingString("handle"),
            TypeError,
            id="str-subclass-handle",
        ),
    ],
)
def test_events_require_nonempty_string_identities(
    factory, field: str, invalid: object, error_type: type[Exception]
) -> None:
    arguments: dict[str, object] = {"event_id": "1", "handle": "a"}
    arguments[field] = invalid

    with pytest.raises(error_type, match=f"{field} must be a nonempty string"):
        factory(**arguments)


def test_hash_changing_string_reproduction_changes_after_construction() -> None:
    identity = _HashChangingString("identity")
    original_hash = hash(identity)

    identity.change_hash()

    assert hash(identity) != original_hash


@pytest.mark.parametrize(
    ("event_type", "type_name"),
    [
        pytest.param(_UnsafeCreateEvent, "_UnsafeCreateEvent", id="create"),
        pytest.param(_UnsafeUpdateEvent, "_UnsafeUpdateEvent", id="update"),
    ],
)
def test_replay_rejects_payload_event_subclasses_with_mutable_payload(
    event_type, type_name: str
) -> None:
    source = bytearray(b"old")
    event = event_type("1", "a", source)
    assert event.base_payload is source

    source[:] = b"new"

    with pytest.raises(TypeError, match=f"unsupported event: {type_name}"):
        replay((event,), budget_bytes=2048)


def test_replay_rejects_nonpayload_event_subclass() -> None:
    with pytest.raises(TypeError, match="unsupported event: _ReadEventSubclass"):
        replay((_ReadEventSubclass("1", "a"),), budget_bytes=2048)


def test_probe_does_not_refresh_usage_or_change_bytes() -> None:
    events = (
        CreateEvent(event_id="1", handle="a", base_payload=b"base"),
        ReadEvent(event_id="2", handle="a"),
        ProbeEvent(event_id="3", handle="a"),
    )
    result = replay(events, budget_bytes=2048)
    assert result.state.bases["a"].reads == 1
    assert result.probe_sizes == (result.state.serialized_bytes,)


def test_replay_is_deterministic_and_probe_rejects_stale_handle() -> None:
    events = (
        CreateEvent(event_id="1", handle="a", base_payload=b"base"),
        DeleteEvent(event_id="2", handle="a"),
        ProbeEvent(event_id="3", handle="a"),
    )
    first = replay(events, budget_bytes=2048)
    second = replay(events, budget_bytes=2048)
    assert first == second
    assert first.errors == ("3:stale-handle:a",)


def test_update_preserves_usage_and_stale_delete_is_recorded() -> None:
    events = (
        CreateEvent(event_id="1", handle="a", base_payload=b"old"),
        ReadEvent(event_id="2", handle="a"),
        UpdateEvent(event_id="3", handle="a", base_payload=b"new"),
        DeleteEvent(event_id="4", handle="missing"),
    )
    result = replay(events, budget_bytes=2048)
    assert result.state.bases["a"].payload == b"new"
    assert result.state.bases["a"].reads == 1
    assert result.errors == ("4:stale-handle:missing",)


def test_duplicate_create_retains_exact_error_classification() -> None:
    result = replay(
        (
            CreateEvent(event_id="1", handle="a", base_payload=b"old"),
            CreateEvent(event_id="2", handle="a", base_payload=b"new"),
        ),
        budget_bytes=2048,
    )

    assert result.state.bases["a"].payload == b"old"
    assert result.errors == ("2:duplicate-handle:a",)


def test_create_does_not_reclassify_unrelated_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_create(
        self: PacketStore, handle: str, payload: bytes, created_at: int
    ) -> PacketStore:
        raise ValueError("independent create validation")

    monkeypatch.setattr(PacketStore, "create", fail_create)

    with pytest.raises(ValueError, match="independent create validation"):
        replay(
            (CreateEvent(event_id="1", handle="a", base_payload=b"payload"),),
            budget_bytes=2048,
        )


def test_replay_rejects_nan_budget_before_large_create() -> None:
    with pytest.raises(TypeError, match="budget_bytes must be an integer"):
        replay(
            (CreateEvent(event_id="1", handle="a", base_payload=b"x" * 100_000),),
            budget_bytes=float("nan"),  # type: ignore[arg-type]
        )


def test_replay_rejects_budget_below_empty_state_before_processing_events() -> None:
    minimum_budget = MemoryState().serialized_bytes

    with pytest.raises(BudgetExceeded):
        replay(
            (CreateEvent(event_id="1", handle="a", base_payload=b"base"),),
            budget_bytes=minimum_budget - 1,
        )


def test_replay_records_create_failure_at_exact_empty_state_budget() -> None:
    minimum_budget = MemoryState().serialized_bytes

    result = replay(
        (CreateEvent(event_id="1", handle="a", base_payload=b"base"),),
        budget_bytes=minimum_budget,
    )

    assert result.state == MemoryState()
    assert result.probe_sizes == ()
    assert result.errors == ("1:budget-exceeded:a",)


@given(
    operations=st.lists(
        st.tuples(
            st.sampled_from(("create", "update", "read", "delete", "probe")),
            st.integers(min_value=0, max_value=4),
            st.binary(min_size=0, max_size=256),
        ),
        min_size=1,
        max_size=40,
    )
)
def test_randomized_replay_never_exceeds_budget(
    operations: list[tuple[str, int, bytes]],
) -> None:
    events = []
    for index, (kind, slot, payload) in enumerate(operations):
        event_id = str(index)
        handle = f"h{slot}"
        if kind == "create":
            events.append(CreateEvent(event_id, handle, payload))
        elif kind == "update":
            events.append(UpdateEvent(event_id, handle, payload))
        elif kind == "read":
            events.append(ReadEvent(event_id, handle))
        elif kind == "delete":
            events.append(DeleteEvent(event_id, handle))
        else:
            events.append(ProbeEvent(event_id, handle))
    result = replay(tuple(events), budget_bytes=512)
    assert result.state.serialized_bytes <= 512
    assert replay(tuple(events), budget_bytes=512) == result
