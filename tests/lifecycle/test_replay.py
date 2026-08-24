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
from ratemem.state.store import PacketStore

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
        pytest.param("handle", "", ValueError, id="empty-handle"),
        pytest.param("handle", 1, TypeError, id="non-string-handle"),
    ],
)
def test_events_require_nonempty_string_identities(
    factory, field: str, invalid: object, error_type: type[Exception]
) -> None:
    arguments: dict[str, object] = {"event_id": "1", "handle": "a"}
    arguments[field] = invalid

    with pytest.raises(error_type, match=f"{field} must be a nonempty string"):
        factory(**arguments)


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
