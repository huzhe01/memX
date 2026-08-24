from __future__ import annotations

from itertools import combinations

from ratemem.allocation.objective import CoverageOracle

DEFAULT_MAX_STATES = 2**18


def _validate_budget(budget_bytes: int) -> None:
    if type(budget_bytes) is not int:
        raise TypeError("budget_bytes must be an integer byte count")
    if budget_bytes < 0:
        raise ValueError("budget_bytes must be nonnegative")


def _validate_max_states(max_states: int) -> None:
    if type(max_states) is not int:
        raise TypeError("max_states must be an integer")
    if max_states <= 0:
        raise ValueError("max_states must be positive")


def exhaustive_optimum(
    oracle: CoverageOracle,
    budget_bytes: int,
    *,
    max_states: int = DEFAULT_MAX_STATES,
) -> frozenset[str]:
    _validate_budget(budget_bytes)
    _validate_max_states(max_states)
    all_names = tuple(sorted(oracle.bundles))
    if len(all_names) > 24:
        raise ValueError("exhaustive oracle supports at most 24 packet bundles")
    names = tuple(
        item for item in all_names if oracle.bundles[item].cost_bytes <= budget_bytes
    )
    required_states = 1 << len(names)
    if required_states > max_states:
        raise ValueError(
            f"exhaustive oracle requires {required_states} states; increase max_states explicitly"
        )

    costs = {item: oracle.bundles[item].cost_bytes for item in names}
    ascending_costs = tuple(sorted(costs.values()))
    best = frozenset[str]()
    best_value = oracle.exact_value(best)
    best_ids: tuple[str, ...] = ()
    for size in range(len(names) + 1):
        if sum(ascending_costs[:size]) > budget_bytes:
            break
        for rows in combinations(names, size):
            candidate_cost = sum(costs[item] for item in rows)
            if candidate_cost > budget_bytes:
                continue
            candidate = frozenset(rows)
            candidate_value = oracle.exact_value(candidate)
            candidate_ids = tuple(sorted(candidate))
            if (candidate_value, candidate_ids) > (best_value, best_ids):
                best = candidate
                best_value = candidate_value
                best_ids = candidate_ids
    return best
