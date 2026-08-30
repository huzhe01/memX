"""Causal immutable packet proposals built from the deployed RateMem codec."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ratemem.method.codec import HardConceptEncoding, RateMemHardCodec
from ratemem.state.model import BaseRecord, Incidence, MemoryState, Packet
from ratemem.state.serialization import bundle_cost_bytes


@dataclass(frozen=True, slots=True)
class ImmutableBundleProposal:
    """One packet plus every concept incidence that must move with it."""

    packet: Packet
    incidences: tuple[Incidence, ...]
    cost_bytes: int

    def __post_init__(self) -> None:
        canonical = tuple(
            sorted(self.incidences, key=lambda edge: (edge.handle, edge.packet_id))
        )
        if tuple(self.incidences) != canonical:
            raise ValueError("bundle incidences must use canonical order")
        object.__setattr__(self, "incidences", canonical)
        if any(edge.packet_id != self.packet.packet_id for edge in canonical):
            raise ValueError("bundle incidence references another packet")
        if len({edge.handle for edge in canonical}) != len(canonical):
            raise ValueError("bundle repeats one concept")
        if self.cost_bytes != bundle_cost_bytes(self.packet, canonical):
            raise ValueError("bundle cost does not match canonical serialization")

    def measured_cost_bytes(self) -> int:
        return bundle_cost_bytes(self.packet, self.incidences)


@dataclass(frozen=True, slots=True)
class ConceptProposal:
    """A complete immutable candidate snapshot for one causal event."""

    handle: str
    event_index: int
    base_record: BaseRecord
    bundles: tuple[ImmutableBundleProposal, ...]

    def __post_init__(self) -> None:
        if type(self.handle) is not str or not self.handle:
            raise ValueError("proposal handle must be a nonempty exact string")
        if type(self.event_index) is not int or self.event_index < 0:
            raise ValueError("proposal event_index must be nonnegative")
        canonical = tuple(self.bundles)
        if tuple(bundle.packet.packet_id for bundle in canonical) != tuple(
            sorted(bundle.packet.packet_id for bundle in canonical)
        ):
            raise ValueError("proposal bundles must use canonical packet order")
        object.__setattr__(self, "bundles", canonical)


class CausalCandidateProposer:
    """Encode only the current target and merge it with resident packet state."""

    def __init__(self, codec: RateMemHardCodec) -> None:
        if type(codec) is not RateMemHardCodec:
            raise TypeError("codec must be an exact RateMemHardCodec")
        self.codec = codec

    def propose(
        self,
        state: MemoryState,
        handle: str,
        current_target_code: NDArray[np.float32],
        event_index: int,
    ) -> ConceptProposal:
        if type(state) is not MemoryState:
            raise TypeError("state must be an exact MemoryState")
        if type(event_index) is not int or event_index < 0:
            raise ValueError("event_index must be nonnegative")
        encoded: HardConceptEncoding = self.codec.encode(handle, current_target_code)
        previous = state.bases.get(handle)
        base = BaseRecord(
            handle=handle,
            payload=encoded.base_payload,
            reads=0 if previous is None else previous.reads,
            created_at=event_index if previous is None else previous.created_at,
        )
        by_packet: dict[str, tuple[Packet, dict[str, Incidence]]] = {}
        for packet_id, packet in state.packets.items():
            edges = {
                edge.handle: edge
                for edge in state.incidences.values()
                if edge.packet_id == packet_id and edge.handle != handle
            }
            by_packet[packet_id] = (packet, edges)
        for row in encoded.incidences:
            packet, edges = by_packet.get(row.packet.packet_id, (row.packet, {}))
            edges[row.incidence.handle] = row.incidence
            by_packet[row.packet.packet_id] = (packet, edges)
        bundles: list[ImmutableBundleProposal] = []
        for _packet_id, (packet, edges) in sorted(by_packet.items()):
            if not edges:
                continue
            incidences = tuple(
                sorted(edges.values(), key=lambda edge: (edge.handle, edge.packet_id))
            )
            bundles.append(
                ImmutableBundleProposal(
                    packet=packet,
                    incidences=incidences,
                    cost_bytes=bundle_cost_bytes(packet, incidences),
                )
            )
        return ConceptProposal(handle, event_index, base, tuple(bundles))
