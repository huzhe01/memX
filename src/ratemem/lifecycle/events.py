from dataclasses import dataclass
from typing import TypeAlias


def _validate_identity(event_id: object, handle: object) -> None:
    for name, value in (("event_id", event_id), ("handle", handle)):
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a nonempty string")
        if not value:
            raise ValueError(f"{name} must be a nonempty string")


def _owned_payload(payload: object) -> bytes:
    if not isinstance(payload, bytes | bytearray | memoryview):
        raise TypeError("base_payload must be bytes-like")
    return bytes(payload)


@dataclass(frozen=True, slots=True)
class CreateEvent:
    event_id: str
    handle: str
    base_payload: bytes

    def __post_init__(self) -> None:
        _validate_identity(self.event_id, self.handle)
        object.__setattr__(self, "base_payload", _owned_payload(self.base_payload))


@dataclass(frozen=True, slots=True)
class ReadEvent:
    event_id: str
    handle: str

    def __post_init__(self) -> None:
        _validate_identity(self.event_id, self.handle)


@dataclass(frozen=True, slots=True)
class UpdateEvent:
    event_id: str
    handle: str
    base_payload: bytes

    def __post_init__(self) -> None:
        _validate_identity(self.event_id, self.handle)
        object.__setattr__(self, "base_payload", _owned_payload(self.base_payload))


@dataclass(frozen=True, slots=True)
class ProbeEvent:
    event_id: str
    handle: str

    def __post_init__(self) -> None:
        _validate_identity(self.event_id, self.handle)


@dataclass(frozen=True, slots=True)
class DeleteEvent:
    event_id: str
    handle: str

    def __post_init__(self) -> None:
        _validate_identity(self.event_id, self.handle)


LifecycleEvent: TypeAlias = (
    CreateEvent | ReadEvent | UpdateEvent | ProbeEvent | DeleteEvent
)
