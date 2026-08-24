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
