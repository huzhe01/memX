from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from ratemem.allocation.objective import CoverageOracle, PacketBundle
from ratemem.method.base_quantizer import BlockwiseBaseQuantizer
from ratemem.method.codec import RateMemHardCodec
from ratemem.method.controller import RateMemController
from ratemem.method.dictionary import GroupRVQDictionary, freeze_dictionary
from ratemem.method.proposal import (
    CausalCandidateProposer,
    ConceptProposal,
    ImmutableBundleProposal,
)
from ratemem.state.model import BaseRecord, Incidence, MemoryState
from ratemem.state.serialization import bundle_cost_bytes, packet_from_payload


def _value_oracle(
    cohort: Sequence[str],
    bundles: Sequence[ImmutableBundleProposal],
) -> CoverageOracle:
    return CoverageOracle(
        bundles={
            row.packet.packet_id: PacketBundle(
                row.packet.packet_id,
                row.cost_bytes,
                {edge.handle: (1.0,) for edge in row.incidences},
            )
            for row in bundles
        },
        request_weights={handle: 1.0 for handle in cohort},
        group_weights={handle: (1.0,) for handle in cohort},
    )


def _proposer() -> CausalCandidateProposer:
    dictionary = GroupRVQDictionary(1, 4, 1, 2)
    with torch.no_grad():
        dictionary.codebooks.copy_(
            torch.tensor([[[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]])
        )
        dictionary.normalize_codebooks_()
    codec = RateMemHardCodec(
        BlockwiseBaseQuantizer(4, 4),
        freeze_dictionary(dictionary),
        gain_step=1 / 256,
        maximum_packets=1,
    )
    return CausalCandidateProposer(codec)


def _code(offset: float = 0.0) -> np.ndarray:
    return np.array([0.91 + offset, -0.52, 0.33, -1.0], dtype=np.float32)


def test_controller_never_exceeds_serialized_budget() -> None:
    proposer = _proposer()
    controller = RateMemController(budget_bytes=700, oracle_factory=_value_oracle)
    first = proposer.propose(MemoryState(), "a", _code(), event_index=1)
    after_first = controller.apply_create(MemoryState(), first)
    second = proposer.propose(after_first.state, "b", _code(), event_index=2)
    after_second = controller.apply_create(after_first.state, second)
    assert after_second.state.serialized_bytes <= 700
    assert after_second.theorem_scope == (
        "fixed_admitted_cohort_prescreened_packets_only"
    )


def test_oversized_base_is_rejected_without_mutating_old_state() -> None:
    proposal = _proposer().propose(MemoryState(), "a", _code(), event_index=1)
    controller = RateMemController(budget_bytes=64, oracle_factory=_value_oracle)
    original = MemoryState()
    result = controller.apply_create(original, proposal)
    assert result.outcome == "rejected"
    assert result.state is original


def test_delete_collects_only_packets_without_remaining_dependents() -> None:
    packet = packet_from_payload(b"shared")
    incidences = {
        (handle, packet.packet_id): Incidence(handle, packet.packet_id, 1)
        for handle in ("a", "b")
    }
    state = MemoryState(
        bases={
            "a": BaseRecord("a", b"a", 0, 1),
            "b": BaseRecord("b", b"b", 0, 2),
        },
        packets={packet.packet_id: packet},
        incidences=incidences,
    )
    controller = RateMemController(4096, _value_oracle)
    after_a = controller.delete(state, "a")
    assert after_a.outcome == "deleted"
    assert after_a.state.packets
    after_b = controller.delete(after_a.state, "b")
    assert after_b.state.packets == {}


def test_controller_prescreens_more_than_twenty_four_packets(monkeypatch) -> None:
    import ratemem.method.controller as controller_module

    bases = {
        handle: BaseRecord(handle, handle.encode(), 0, index)
        for index, handle in enumerate(("a", "b", "c"), start=1)
    }
    bundles: list[ImmutableBundleProposal] = []
    packets = {}
    incidences = {}
    for handle in ("a", "b", "c", "d"):
        for packet_index in range(8):
            packet = packet_from_payload(f"{handle}-{packet_index}".encode())
            edge = Incidence(handle, packet.packet_id, 1)
            bundle = ImmutableBundleProposal(
                packet,
                (edge,),
                bundle_cost_bytes(packet, (edge,)),
            )
            bundles.append(bundle)
            if handle != "d":
                packets[packet.packet_id] = packet
                incidences[(handle, packet.packet_id)] = edge
    state = MemoryState(bases=bases, packets=packets, incidences=incidences)
    proposal = ConceptProposal(
        "d",
        4,
        BaseRecord("d", b"d", 0, 4),
        tuple(sorted(bundles, key=lambda row: row.packet.packet_id)),
    )
    observed: list[tuple[int, int]] = []
    real_prescreen = controller_module.prescreen_certified_oracle

    def recording_prescreen(
        oracle: CoverageOracle,
        budget_bytes: int,
        *,
        max_bundles: int = 24,
    ) -> CoverageOracle:
        observed.append((len(oracle.bundles), max_bundles))
        return real_prescreen(oracle, budget_bytes, max_bundles=max_bundles)

    monkeypatch.setattr(
        controller_module,
        "prescreen_certified_oracle",
        recording_prescreen,
    )
    decision = RateMemController(1_000_000, _value_oracle).apply_create(
        state,
        proposal,
    )
    assert decision.outcome == "created"
    assert observed == [(32, 24)]
    assert decision.state.serialized_bytes <= 1_000_000
