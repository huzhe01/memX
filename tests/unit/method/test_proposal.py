from __future__ import annotations

import numpy as np
import pytest
import torch

from ratemem.method.base_quantizer import BlockwiseBaseQuantizer
from ratemem.method.codec import RateMemHardCodec
from ratemem.method.dictionary import GroupRVQDictionary, freeze_dictionary
from ratemem.method.proposal import CausalCandidateProposer
from ratemem.state.model import Incidence, MemoryState


def _hard_codec() -> RateMemHardCodec:
    dictionary = GroupRVQDictionary(1, 4, 1, 2)
    with torch.no_grad():
        dictionary.codebooks.copy_(
            torch.tensor([[[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]])
        )
        dictionary.normalize_codebooks_()
    return RateMemHardCodec(
        BlockwiseBaseQuantizer(4, 4),
        freeze_dictionary(dictionary),
        gain_step=1 / 256,
        maximum_packets=1,
    )


def _code(offset: float = 0.0) -> np.ndarray:
    return np.array([0.91 + offset, -0.52, 0.33, -1.0], dtype=np.float32)


def _state_from_proposal(proposal: object) -> MemoryState:
    from ratemem.method.proposal import ConceptProposal

    if type(proposal) is not ConceptProposal:
        raise AssertionError("expected exact ConceptProposal")
    return MemoryState(
        bases={proposal.handle: proposal.base_record},
        packets={bundle.packet.packet_id: bundle.packet for bundle in proposal.bundles},
        incidences={
            (edge.handle, edge.packet_id): edge
            for bundle in proposal.bundles
            for edge in bundle.incidences
        },
    )


def test_existing_packet_is_stored_once_and_bundle_closes_all_incidences() -> None:
    proposer = CausalCandidateProposer(_hard_codec())
    first = proposer.propose(MemoryState(), "a", _code(), event_index=1)
    state = _state_from_proposal(first)
    second = proposer.propose(state, "b", _code(), event_index=2)
    reused = [
        bundle
        for bundle in second.bundles
        if {edge.handle for edge in bundle.incidences} == {"a", "b"}
    ]
    assert len(reused) == 1
    assert reused[0].packet.packet_id in state.packets
    assert reused[0].cost_bytes == reused[0].measured_cost_bytes()


def test_bundle_is_frozen_and_incidence_order_is_canonical() -> None:
    bundle = CausalCandidateProposer(_hard_codec()).propose(
        MemoryState(), "b", _code(), event_index=2
    ).bundles[0]
    assert tuple(edge.handle for edge in bundle.incidences) == tuple(
        sorted(edge.handle for edge in bundle.incidences)
    )
    with pytest.raises((AttributeError, TypeError)):
        bundle.incidences += (Incidence("x", bundle.packet.packet_id, 1),)


def test_update_replaces_only_its_incidence_without_mutating_prior_proposal() -> None:
    proposer = CausalCandidateProposer(_hard_codec())
    first = proposer.propose(MemoryState(), "a", _code(), event_index=1)
    second = proposer.propose(_state_from_proposal(first), "b", _code(), event_index=2)
    before = second.bundles[0]
    updated = proposer.propose(_state_from_proposal(second), "b", _code(0.01), event_index=3)
    after = updated.bundles[0]
    assert before is not after
    assert next(edge for edge in before.incidences if edge.handle == "a") == next(
        edge for edge in after.incidences if edge.handle == "a"
    )
    assert next(edge for edge in before.incidences if edge.handle == "b") != next(
        edge for edge in after.incidences if edge.handle == "b"
    )
    assert tuple(edge.gain_q for edge in before.incidences) == (14, 14)
