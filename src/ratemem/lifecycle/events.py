from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class CreateEvent:
    event_id: str
    handle: str
    base_payload: bytes


@dataclass(frozen=True, slots=True)
class ReadEvent:
    event_id: str
    handle: str


@dataclass(frozen=True, slots=True)
class UpdateEvent:
    event_id: str
    handle: str
    base_payload: bytes


@dataclass(frozen=True, slots=True)
class ProbeEvent:
    event_id: str
    handle: str


@dataclass(frozen=True, slots=True)
class DeleteEvent:
    event_id: str
    handle: str


LifecycleEvent: TypeAlias = (
    CreateEvent | ReadEvent | UpdateEvent | ProbeEvent | DeleteEvent
)
