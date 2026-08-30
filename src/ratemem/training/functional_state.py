"""Small immutable differentiable state carried inside a training segment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from torch import Tensor


@dataclass(frozen=True, slots=True)
class FunctionalMemoryState:
    codes: Mapping[str, Tensor] = field(default_factory=dict)
    last_event: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        codes = dict(self.codes)
        events = dict(self.last_event)
        if set(codes) != set(events):
            raise ValueError("functional codes and event indices must name the same handles")
        if any(type(handle) is not str or not handle for handle in codes):
            raise ValueError("functional state handles must be nonempty exact strings")
        if any(type(code) is not Tensor for code in codes.values()):
            raise TypeError("functional state codes must be exact tensors")
        if any(type(index) is not int or index < 0 for index in events.values()):
            raise ValueError("functional event indices must be nonnegative exact integers")
        object.__setattr__(self, "codes", MappingProxyType(codes))
        object.__setattr__(self, "last_event", MappingProxyType(events))

    def upsert(
        self,
        handle: str,
        code: Tensor,
        event_index: int,
    ) -> FunctionalMemoryState:
        if type(handle) is not str or not handle:
            raise ValueError("functional state handle must be a nonempty exact string")
        if type(code) is not Tensor:
            raise TypeError("functional state code must be an exact tensor")
        if type(event_index) is not int or event_index < 0:
            raise ValueError("functional event index must be a nonnegative exact integer")
        if handle in self.last_event and event_index <= self.last_event[handle]:
            raise ValueError("functional updates must move forward in event order")
        codes = dict(self.codes)
        events = dict(self.last_event)
        codes[handle] = code
        events[handle] = event_index
        return FunctionalMemoryState(codes, events)

    def delete(self, handle: str) -> FunctionalMemoryState:
        codes = dict(self.codes)
        events = dict(self.last_event)
        codes.pop(handle, None)
        events.pop(handle, None)
        return FunctionalMemoryState(codes, events)

    def detach_boundary(self) -> FunctionalMemoryState:
        return FunctionalMemoryState(
            {handle: code.detach() for handle, code in self.codes.items()},
            self.last_event,
        )
