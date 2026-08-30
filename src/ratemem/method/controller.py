"""Empirical lifecycle policy around the certified fixed-snapshot allocator.

Whole-base admission, eviction, cohort projection, causal pre-screening loss, and
rejection are empirical outer-policy operations.  The certified claim starts only
after the admitted cohort, reduced packet ground set, costs, and residual budget
have been fixed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from ratemem.allocation.objective import CoverageOracle
from ratemem.allocation.snapshot import allocate_snapshot, prescreen_certified_oracle
from ratemem.method.proposal import ConceptProposal, ImmutableBundleProposal
from ratemem.state.model import BaseRecord, Incidence, MemoryState, Packet
from ratemem.state.serialization import bundle_cost_bytes

OracleFactory = Callable[
    [Sequence[str], Sequence[ImmutableBundleProposal]], CoverageOracle
]
_THEOREM_SCOPE: Literal[
    "fixed_admitted_cohort_prescreened_packets_only"
] = "fixed_admitted_cohort_prescreened_packets_only"


@dataclass(frozen=True, slots=True)
class ControllerDecision:
    state: MemoryState
    outcome: Literal[
        "created",
        "updated",
        "deleted",
        "rejected",
        "read",
        "stale_handle",
    ]
    evicted_handles: tuple[str, ...] = ()
    selected_packet_ids: tuple[str, ...] = ()
    theorem_scope: Literal[
        "fixed_admitted_cohort_prescreened_packets_only"
    ] = _THEOREM_SCOPE


def _without_handle(state: MemoryState, handle: str) -> MemoryState:
    bases = {key: row for key, row in state.bases.items() if key != handle}
    incidences = {
        key: row for key, row in state.incidences.items() if row.handle != handle
    }
    referenced = {row.packet_id for row in incidences.values()}
    packets = {
        key: row for key, row in state.packets.items() if key in referenced
    }
    return MemoryState(bases=bases, packets=packets, incidences=incidences)


def _base_only(bases: dict[str, BaseRecord]) -> MemoryState:
    return MemoryState(bases=bases, packets={}, incidences={})


def _base_increment_bytes(bases: dict[str, BaseRecord], handle: str) -> int:
    with_row = _base_only(bases).serialized_bytes
    without_row = _base_only(
        {key: row for key, row in bases.items() if key != handle}
    ).serialized_bytes
    return with_row - without_row


class RateMemController:
    def __init__(
        self,
        budget_bytes: int,
        oracle_factory: OracleFactory,
        certified_prescreen_max_bundles: Literal[24] = 24,
    ) -> None:
        if type(budget_bytes) is not int or budget_bytes < 0:
            raise ValueError("budget_bytes must be a nonnegative exact integer")
        if not callable(oracle_factory):
            raise TypeError("oracle_factory must be callable")
        if (
            type(certified_prescreen_max_bundles) is not int
            or certified_prescreen_max_bundles != 24
        ):
            raise ValueError("certified prescreen cap must equal the locked value 24")
        self.budget_bytes = budget_bytes
        self.oracle_factory = oracle_factory
        self.certified_prescreen_max_bundles = certified_prescreen_max_bundles

    def _admit_bases(
        self,
        state: MemoryState,
        proposal: ConceptProposal,
    ) -> tuple[dict[str, BaseRecord] | None, tuple[str, ...]]:
        bases = dict(state.bases)
        bases[proposal.handle] = proposal.base_record
        evicted: list[str] = []
        while _base_only(bases).serialized_bytes > self.budget_bytes:
            candidates = [handle for handle in bases if handle != proposal.handle]
            if not candidates:
                return None, tuple(evicted)
            victim = min(
                candidates,
                key=lambda handle: (
                    (bases[handle].reads + 1)
                    / _base_increment_bytes(bases, handle),
                    bases[handle].created_at,
                    handle,
                ),
            )
            del bases[victim]
            evicted.append(victim)
        return bases, tuple(evicted)

    @staticmethod
    def _project_bundles(
        bundles: Sequence[ImmutableBundleProposal],
        cohort: set[str],
    ) -> tuple[ImmutableBundleProposal, ...]:
        projected: list[ImmutableBundleProposal] = []
        for bundle in bundles:
            incidences = tuple(
                edge for edge in bundle.incidences if edge.handle in cohort
            )
            if incidences:
                projected.append(
                    ImmutableBundleProposal(
                        bundle.packet,
                        incidences,
                        bundle_cost_bytes(bundle.packet, incidences),
                    )
                )
        return tuple(projected)

    def _apply(
        self,
        state: MemoryState,
        proposal: ConceptProposal,
        outcome: Literal["created", "updated"],
    ) -> ControllerDecision:
        if type(state) is not MemoryState or type(proposal) is not ConceptProposal:
            raise TypeError("controller requires exact state and proposal values")
        admitted, evicted = self._admit_bases(state, proposal)
        if admitted is None:
            return ControllerDecision(state=state, outcome="rejected")
        base_state = _base_only(admitted)
        cohort = tuple(sorted(admitted))
        bundles = self._project_bundles(proposal.bundles, set(cohort))
        oracle = self.oracle_factory(cohort, bundles)
        if type(oracle) is not CoverageOracle:
            raise TypeError("oracle_factory must return an exact CoverageOracle")
        if set(oracle.bundles) != {
            row.packet.packet_id for row in bundles
        }:
            raise ValueError("oracle and immutable proposal ground sets differ")
        residual_budget = self.budget_bytes - base_state.serialized_bytes
        certified_oracle = prescreen_certified_oracle(
            oracle,
            residual_budget,
            max_bundles=self.certified_prescreen_max_bundles,
        )
        selected = allocate_snapshot(
            certified_oracle,
            residual_budget,
            max_bundles=self.certified_prescreen_max_bundles,
        )
        packets: dict[str, Packet] = {}
        incidences: dict[tuple[str, str], Incidence] = {}
        by_id = {row.packet.packet_id: row for row in bundles}
        for packet_id in sorted(selected):
            bundle = by_id[packet_id]
            packets[packet_id] = bundle.packet
            for edge in bundle.incidences:
                incidences[(edge.handle, edge.packet_id)] = edge
        result = MemoryState(
            bases=admitted,
            packets=packets,
            incidences=incidences,
        )
        if result.serialized_bytes > self.budget_bytes:
            raise RuntimeError(
                "allocator produced a state above the exact byte budget"
            )
        return ControllerDecision(
            state=result,
            outcome=outcome,
            evicted_handles=evicted,
            selected_packet_ids=tuple(sorted(selected)),
        )

    def apply_create(
        self,
        state: MemoryState,
        proposal: ConceptProposal,
    ) -> ControllerDecision:
        if proposal.handle in state.bases:
            raise ValueError("create received an active handle")
        return self._apply(state, proposal, "created")

    def apply_update(
        self,
        state: MemoryState,
        proposal: ConceptProposal,
    ) -> ControllerDecision:
        if proposal.handle not in state.bases:
            return ControllerDecision(state=state, outcome="stale_handle")
        return self._apply(state, proposal, "updated")

    def delete(self, state: MemoryState, handle: str) -> ControllerDecision:
        if handle not in state.bases:
            return ControllerDecision(state=state, outcome="stale_handle")
        return ControllerDecision(
            state=_without_handle(state, handle),
            outcome="deleted",
        )

    def read(
        self,
        state: MemoryState,
        handle: str,
        update_usage: bool,
    ) -> ControllerDecision:
        if type(update_usage) is not bool:
            raise TypeError("update_usage must be an exact bool")
        if handle not in state.bases:
            return ControllerDecision(state=state, outcome="stale_handle")
        if not update_usage:
            return ControllerDecision(state=state, outcome="read")
        old = state.bases[handle]
        bases = dict(state.bases)
        bases[handle] = BaseRecord(
            old.handle,
            old.payload,
            old.reads + 1,
            old.created_at,
        )
        updated = MemoryState(
            bases=bases,
            packets=state.packets,
            incidences=state.incidences,
        )
        if updated.serialized_bytes != state.serialized_bytes:
            raise RuntimeError(
                "fixed-width usage update changed serialized state length"
            )
        return ControllerDecision(state=updated, outcome="read")
