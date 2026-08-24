from __future__ import annotations

from itertools import combinations

from ratemem.allocation.objective import CoverageOracle


def _validate_budget(budget_bytes: int) -> None:
    if type(budget_bytes) is not int:
        raise TypeError("budget_bytes must be an integer byte count")
    if budget_bytes < 0:
        raise ValueError("budget_bytes must be nonnegative")


def _density_fill(
    oracle: CoverageOracle, seed: frozenset[str], budget_bytes: int
) -> frozenset[str]:
    selected = set(seed)
    remaining = set(oracle.bundles) - selected
    while remaining:
        ranked = sorted(
            remaining,
            key=lambda item: (
                oracle.marginal(frozenset(selected), item)
                / oracle.bundles[item].cost_bytes,
                item,
            ),
            reverse=True,
        )
        item = ranked[0]
        remaining.remove(item)
        candidate = frozenset(selected | {item})
        if oracle.cost(candidate) <= budget_bytes:
            selected.add(item)
    return frozenset(selected)


def allocate_snapshot(oracle: CoverageOracle, budget_bytes: int) -> frozenset[str]:
    _validate_budget(budget_bytes)
    names = tuple(sorted(oracle.bundles))
    best = frozenset[str]()
    for size in range(min(3, len(names)) + 1):
        for rows in combinations(names, size):
            seed = frozenset(rows)
            if oracle.cost(seed) > budget_bytes:
                continue
            candidate = _density_fill(oracle, seed, budget_bytes)
            if (oracle.value(candidate), tuple(sorted(candidate))) > (
                oracle.value(best),
                tuple(sorted(best)),
            ):
                best = candidate
    return best
