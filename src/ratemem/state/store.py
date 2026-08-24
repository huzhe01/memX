from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ratemem.state.model import (
    BaseRecord,
    Incidence,
    MemoryState,
    Packet,
    _validate_identity,
)


class BudgetExceeded(ValueError):
    pass


def _validate_packet(packet: Packet) -> None:
    if type(packet) is not Packet:
        raise TypeError("packet must be an exact Packet instance")
    if hashlib.sha256(packet.payload).hexdigest() != packet.packet_id:
        raise ValueError("packet hash mismatch")


def _validate_state(state: MemoryState) -> None:
    if type(state) is not MemoryState:
        raise TypeError("state must be an exact MemoryState instance")
    for packet in state.packets.values():
        _validate_packet(packet)

    referenced_packets: set[str] = set()
    for incidence in state.incidences.values():
        if (
            incidence.handle not in state.bases
            or incidence.packet_id not in state.packets
        ):
            raise ValueError("dangling packet incidence")
        referenced_packets.add(incidence.packet_id)

    if state.packets.keys() - referenced_packets:
        raise ValueError("orphan packet")


@dataclass(frozen=True, slots=True)
class PacketStore:
    state: MemoryState
    budget_bytes: int

    def __post_init__(self) -> None:
        if type(self.budget_bytes) is not int:
            raise TypeError("budget_bytes must be an integer")
        if self.budget_bytes < 0:
            raise ValueError("budget_bytes must be nonnegative")
        _validate_state(self.state)
        if self.state.serialized_bytes > self.budget_bytes:
            raise BudgetExceeded(
                f"state uses {self.state.serialized_bytes} bytes, "
                f"budget is {self.budget_bytes}"
            )

    @classmethod
    def empty(cls, budget_bytes: int) -> PacketStore:
        return cls(state=MemoryState(), budget_bytes=budget_bytes)

    def _checked(self, state: MemoryState) -> PacketStore:
        return PacketStore(state=state, budget_bytes=self.budget_bytes)

    @staticmethod
    def _validate_incidence(
        incidence: Incidence, handle: str, packet_id: str
    ) -> None:
        if type(incidence) is not Incidence:
            raise TypeError("incidence must be an exact Incidence instance")
        if incidence.handle != handle:
            raise ValueError("incidence handle does not match operation")
        if incidence.packet_id != packet_id:
            raise ValueError("incidence packet id does not match payload")

    @staticmethod
    def _collect_referenced(
        packets: dict[str, Packet], incidences: dict[tuple[str, str], Incidence]
    ) -> dict[str, Packet]:
        referenced = {edge.packet_id for edge in incidences.values()}
        return {key: value for key, value in packets.items() if key in referenced}

    def create(self, handle: str, payload: bytes, created_at: int) -> PacketStore:
        _validate_identity("handle", handle)
        if handle in self.state.bases:
            raise ValueError(f"handle already exists: {handle}")
        bases = dict(self.state.bases)
        bases[handle] = BaseRecord(handle, payload, reads=0, created_at=created_at)
        return self._checked(MemoryState(bases, self.state.packets, self.state.incidences))

    def attach(self, packet: Packet, incidence: Incidence) -> PacketStore:
        return self.attach_bundle(packet, (incidence,))

    def attach_bundle(
        self, packet: Packet, bundle: tuple[Incidence, ...]
    ) -> PacketStore:
        if not bundle:
            raise ValueError("packet bundle must contain at least one incidence")
        _validate_packet(packet)
        for incidence in bundle:
            if type(incidence) is not Incidence:
                raise TypeError("incidence must be an exact Incidence instance")
        handles = [incidence.handle for incidence in bundle]
        if len(set(handles)) != len(handles):
            raise ValueError("packet bundle repeats a concept incidence")
        for incidence in bundle:
            if incidence.handle not in self.state.bases:
                raise KeyError(incidence.handle)
        for incidence in bundle:
            self._validate_incidence(incidence, incidence.handle, packet.packet_id)
        packets = dict(self.state.packets)
        packets[packet.packet_id] = packet
        incidences = dict(self.state.incidences)
        for incidence in bundle:
            incidences[(incidence.handle, incidence.packet_id)] = incidence
        return self._checked(MemoryState(self.state.bases, packets, incidences))

    def replace(
        self,
        handle: str,
        payload: bytes,
        attachments: tuple[tuple[Packet, Incidence], ...],
    ) -> PacketStore:
        _validate_identity("handle", handle)
        if handle not in self.state.bases:
            raise KeyError(handle)
        for packet, incidence in attachments:
            _validate_packet(packet)
            self._validate_incidence(incidence, handle, packet.packet_id)
        packet_ids = [packet.packet_id for packet, _ in attachments]
        if len(set(packet_ids)) != len(packet_ids):
            raise ValueError("replacement repeats packet attachment")
        old = self.state.bases[handle]
        bases = dict(self.state.bases)
        bases[handle] = BaseRecord(handle, payload, old.reads, old.created_at)
        packets = dict(self.state.packets)
        incidences = {
            key: value
            for key, value in self.state.incidences.items()
            if value.handle != handle
        }
        for packet, incidence in attachments:
            packets[packet.packet_id] = packet
            incidences[(handle, packet.packet_id)] = incidence
        packets = self._collect_referenced(packets, incidences)
        return self._checked(MemoryState(bases, packets, incidences))

    def read(
        self, handle: str, update_usage: bool = True
    ) -> tuple[PacketStore, BaseRecord]:
        _validate_identity("handle", handle)
        record = self.state.bases[handle]
        if not update_usage:
            return self, record
        bases = dict(self.state.bases)
        bases[handle] = BaseRecord(handle, record.payload, record.reads + 1, record.created_at)
        return self._checked(MemoryState(bases, self.state.packets, self.state.incidences)), record

    def delete(self, handle: str) -> PacketStore:
        _validate_identity("handle", handle)
        if handle not in self.state.bases:
            raise KeyError(handle)
        bases = {key: value for key, value in self.state.bases.items() if key != handle}
        incidences = {
            key: value for key, value in self.state.incidences.items() if value.handle != handle
        }
        packets = self._collect_referenced(dict(self.state.packets), incidences)
        return self._checked(MemoryState(bases, packets, incidences))
