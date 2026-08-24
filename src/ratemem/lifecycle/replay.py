from __future__ import annotations

from dataclasses import dataclass

from ratemem.lifecycle.events import (
    CreateEvent,
    DeleteEvent,
    LifecycleEvent,
    ProbeEvent,
    ReadEvent,
    UpdateEvent,
)
from ratemem.state.model import MemoryState
from ratemem.state.store import BudgetExceeded, PacketStore


@dataclass(frozen=True, slots=True)
class ReplayResult:
    state: MemoryState
    probe_sizes: tuple[int, ...]
    errors: tuple[str, ...]


def replay(events: tuple[LifecycleEvent, ...], budget_bytes: int) -> ReplayResult:
    store = PacketStore.empty(budget_bytes)
    probes: list[int] = []
    errors: list[str] = []
    for index, event in enumerate(events):
        if isinstance(event, CreateEvent):
            if event.handle in store.state.bases:
                errors.append(f"{event.event_id}:duplicate-handle:{event.handle}")
                continue
            try:
                store = store.create(event.handle, event.base_payload, created_at=index)
            except BudgetExceeded:
                errors.append(f"{event.event_id}:budget-exceeded:{event.handle}")
        elif isinstance(event, ReadEvent):
            try:
                store, _ = store.read(event.handle, update_usage=True)
            except KeyError:
                errors.append(f"{event.event_id}:stale-handle:{event.handle}")
        elif isinstance(event, UpdateEvent):
            try:
                store = store.replace(event.handle, event.base_payload, attachments=())
            except KeyError:
                errors.append(f"{event.event_id}:stale-handle:{event.handle}")
            except BudgetExceeded:
                errors.append(f"{event.event_id}:budget-exceeded:{event.handle}")
        elif isinstance(event, ProbeEvent):
            try:
                snapshot, _ = store.read(event.handle, update_usage=False)
                probes.append(snapshot.state.serialized_bytes)
            except KeyError:
                errors.append(f"{event.event_id}:stale-handle:{event.handle}")
        elif isinstance(event, DeleteEvent):
            try:
                store = store.delete(event.handle)
            except KeyError:
                errors.append(f"{event.event_id}:stale-handle:{event.handle}")
        else:
            raise TypeError(f"unsupported event: {type(event).__name__}")
    return ReplayResult(store.state, tuple(probes), tuple(errors))
