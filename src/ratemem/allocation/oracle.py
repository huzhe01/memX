from __future__ import annotations

from itertools import combinations

from ratemem.allocation.objective import CoverageOracle


def _validate_budget(budget_bytes: int) -> None:
    if type(budget_bytes) is not int:
        raise TypeError("budget_bytes must be an integer byte count")
    if budget_bytes < 0:
        raise ValueError("budget_bytes must be nonnegative")


def exhaustive_optimum(oracle: CoverageOracle, budget_bytes: int) -> frozenset[str]:
    _validate_budget(budget_bytes)
    names = tuple(sorted(oracle.bundles))
    if len(names) > 24:
        raise ValueError("exhaustive oracle supports at most 24 packet bundles")
    best = frozenset[str]()
    for size in range(len(names) + 1):
        for rows in combinations(names, size):
            candidate = frozenset(rows)
            if oracle.cost(candidate) <= budget_bytes:
                if (oracle.value(candidate), tuple(sorted(candidate))) > (
                    oracle.value(best),
                    tuple(sorted(best)),
                ):
                    best = candidate
    return best
