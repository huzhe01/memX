from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations

from ratemem.allocation.objective import CoverageOracle

DEFAULT_MAX_BUNDLES = 24


@dataclass(frozen=True, slots=True)
class _FillResult:
    selected: frozenset[str]
    cost_bytes: int
    exact_value: Fraction


def _validate_budget(budget_bytes: int) -> None:
    if type(budget_bytes) is not int:
        raise TypeError("budget_bytes must be an integer byte count")
    if budget_bytes < 0:
        raise ValueError("budget_bytes must be nonnegative")


def _validate_max_bundles(max_bundles: int) -> None:
    if type(max_bundles) is not int:
        raise TypeError("max_bundles must be an integer")
    if max_bundles <= 0:
        raise ValueError("max_bundles must be positive")


def _density_fill(
    oracle: CoverageOracle,
    seed: frozenset[str],
    seed_cost: int,
    budget_bytes: int,
) -> _FillResult:
    selected = set(seed)
    selected_cost = seed_cost
    coverage = oracle._empty_exact_coverage()
    for item in sorted(seed):
        oracle._add_exact_gains(coverage, item)
    remaining = set(oracle.bundles) - selected
    while remaining:
        remaining = {
            item
            for item in remaining
            if selected_cost + oracle.bundles[item].cost_bytes <= budget_bytes
        }
        if not remaining:
            break
        item = max(
            remaining,
            key=lambda item: (
                oracle._exact_marginal_from_coverage(coverage, item)
                / oracle.bundles[item].cost_bytes,
                item,
            ),
        )
        remaining.remove(item)
        item_cost = oracle.bundles[item].cost_bytes
        selected.add(item)
        selected_cost += item_cost
        oracle._add_exact_gains(coverage, item)
    return _FillResult(
        selected=frozenset(selected),
        cost_bytes=selected_cost,
        exact_value=oracle._exact_value_from_coverage(coverage),
    )


def allocate_snapshot(
    oracle: CoverageOracle,
    budget_bytes: int,
    *,
    max_bundles: int = DEFAULT_MAX_BUNDLES,
) -> frozenset[str]:
    _validate_budget(budget_bytes)
    _validate_max_bundles(max_bundles)
    names = tuple(sorted(oracle.bundles))
    if len(names) > max_bundles:
        raise ValueError(
            f"certified allocator ground set exceeds max_bundles={max_bundles}; "
            "raise the limit explicitly or call allocate_density_greedy_heuristic"
        )

    costs = {item: oracle.bundles[item].cost_bytes for item in names}
    best = frozenset[str]()
    best_value = oracle.exact_value(best)
    best_ids: tuple[str, ...] = ()
    for size in range(min(3, len(names)) + 1):
        for rows in combinations(names, size):
            seed = frozenset(rows)
            seed_cost = sum(costs[item] for item in rows)
            if seed_cost > budget_bytes:
                continue
            result = _density_fill(oracle, seed, seed_cost, budget_bytes)
            candidate_ids = tuple(sorted(result.selected))
            if (result.exact_value, candidate_ids) > (best_value, best_ids):
                best = result.selected
                best_value = result.exact_value
                best_ids = candidate_ids
    return best


def allocate_density_greedy_heuristic(
    oracle: CoverageOracle, budget_bytes: int
) -> frozenset[str]:
    """Return deterministic exact-density greedy output without an approximation claim."""
    _validate_budget(budget_bytes)
    return _density_fill(oracle, frozenset(), 0, budget_bytes).selected
