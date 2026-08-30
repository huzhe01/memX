"""Plain marginal-density greedy control over RateMem's shared packet stream."""

from __future__ import annotations

from dataclasses import dataclass

from ratemem.allocation.objective import CoverageOracle


@dataclass(frozen=True, slots=True)
class GreedyResult:
    selected_packet_ids: tuple[str, ...]
    total_cost: int
    objective_value: float


def plain_density_greedy(oracle: CoverageOracle, budget_bytes: int) -> GreedyResult:
    """Run one deterministic pass with no seed enumeration or switching term."""

    if type(oracle) is not CoverageOracle:
        raise TypeError("oracle must be an exact CoverageOracle")
    if type(budget_bytes) is not int or budget_bytes < 0:
        raise ValueError("budget bytes must be a nonnegative integer")
    selected: tuple[str, ...] = ()
    remaining = set(oracle.bundles)
    used = 0
    while remaining:
        feasible = [
            packet_id
            for packet_id in remaining
            if used + oracle.bundles[packet_id].cost_bytes <= budget_bytes
        ]
        if not feasible:
            break
        selected_set = frozenset(selected)
        packet_id = min(
            feasible,
            key=lambda item: (
                -(
                    oracle.exact_marginal(selected_set, item)
                    / oracle.bundles[item].cost_bytes
                ),
                item,
            ),
        )
        if oracle.exact_marginal(selected_set, packet_id) <= 0:
            break
        selected += (packet_id,)
        remaining.remove(packet_id)
        used += oracle.bundles[packet_id].cost_bytes
    return GreedyResult(selected, used, oracle.value(frozenset(selected)))


__all__ = ["GreedyResult", "plain_density_greedy"]
