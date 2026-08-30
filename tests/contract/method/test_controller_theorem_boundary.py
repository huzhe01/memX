from __future__ import annotations

import inspect
from collections.abc import Sequence

from ratemem.allocation.objective import CoverageOracle, PacketBundle
from ratemem.method.controller import ControllerDecision, RateMemController
from ratemem.method.proposal import ImmutableBundleProposal
from ratemem.state.model import BaseRecord, MemoryState


def _empty_oracle(
    cohort: Sequence[str],
    bundles: Sequence[ImmutableBundleProposal],
) -> CoverageOracle:
    return CoverageOracle(
        bundles={
            bundle.packet.packet_id: PacketBundle(
                bundle.packet.packet_id,
                bundle.cost_bytes,
                {edge.handle: (1.0,) for edge in bundle.incidences},
            )
            for bundle in bundles
        },
        request_weights={handle: 1.0 for handle in cohort},
        group_weights={handle: (1.0,) for handle in cohort},
    )


def test_outer_policy_cannot_be_reported_as_theorem_covered() -> None:
    assert ControllerDecision.__dataclass_fields__["theorem_scope"].default == (
        "fixed_admitted_cohort_prescreened_packets_only"
    )
    source = inspect.getsource(RateMemController._admit_bases)
    assert "allocate_snapshot" not in source
    assert "switching" not in source


def test_read_only_probe_returns_identical_state() -> None:
    state = MemoryState(bases={"a": BaseRecord("a", b"base", 0, 1)})
    decision = RateMemController(4096, _empty_oracle).read(
        state,
        "a",
        update_usage=False,
    )
    assert decision.state is state
    assert decision.state.serialized_bytes == state.serialized_bytes


def test_operational_read_updates_only_fixed_width_usage() -> None:
    state = MemoryState(bases={"a": BaseRecord("a", b"base", 0, 1)})
    decision = RateMemController(4096, _empty_oracle).read(
        state,
        "a",
        update_usage=True,
    )
    assert decision.state is not state
    assert decision.state.bases["a"].reads == 1
    assert decision.state.serialized_bytes == state.serialized_bytes
